"""The durable-jobs engine, exercised at its public boundary: submit work, observe
its lifecycle through the store, and check dedupe, cancellation, and notification."""

from __future__ import annotations

import threading
import time

from elbi_core import InMemoryJobStore, Job, JobRunner


def _runner(max_workers: int = 4, on_complete=None) -> JobRunner:
    return JobRunner(
        store=InMemoryJobStore(), max_workers=max_workers, on_complete=on_complete
    )


def test_submit_runs_to_success_with_result() -> None:
    runner = _runner()
    try:
        job = runner.submit("k1", "double", lambda progress, cancelled: 21 * 2)
        assert job.state == "queued"
        done = runner.wait(job.id, timeout=5)
        assert done is not None
        assert done.state == "succeeded"
        assert done.result == 42
        assert done.done is True
    finally:
        runner.close()


def test_identical_key_dedupes_and_runs_once() -> None:
    runner = _runner()
    runs = 0
    lock = threading.Lock()

    def work(progress, cancelled):
        nonlocal runs
        with lock:
            runs += 1
        return runs

    try:
        first = runner.submit("same", "x", work)
        runner.wait(first.id, timeout=5)
        second = runner.submit("same", "x", work)  # succeeded key -> same job
        assert second.id == first.id
        assert runs == 1  # the work did not run a second time
    finally:
        runner.close()


def test_distinct_keys_are_distinct_jobs() -> None:
    runner = _runner()
    try:
        a = runner.submit("a", "x", lambda p, c: "a")
        b = runner.submit("b", "x", lambda p, c: "b")
        assert a.id != b.id
    finally:
        runner.close()


def test_failure_is_captured_as_error() -> None:
    runner = _runner()

    def boom(progress, cancelled):
        raise ValueError("training diverged")

    try:
        job = runner.submit("bad", "x", boom)
        done = runner.wait(job.id, timeout=5)
        assert done is not None
        assert done.state == "failed"
        assert "training diverged" in (done.error or "")
    finally:
        runner.close()


def test_failed_key_can_be_retried() -> None:
    runner = _runner()
    calls = {"n": 0}

    def flaky(progress, cancelled):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt fails")
        return "ok"

    try:
        first = runner.submit("retry", "x", flaky)
        assert runner.wait(first.id, timeout=5).state == "failed"  # type: ignore[union-attr]
        second = runner.submit("retry", "x", flaky)  # failed key -> fresh job
        assert second.id != first.id
        assert runner.wait(second.id, timeout=5).result == "ok"  # type: ignore[union-attr]
    finally:
        runner.close()


def test_progress_is_recorded() -> None:
    runner = _runner()
    seen = threading.Event()

    def work(progress, cancelled):
        progress("epoch 1/3")
        seen.set()
        return "done"

    try:
        job = runner.submit("prog", "x", work)
        done = runner.wait(job.id, timeout=5)
        assert done is not None and done.progress == "epoch 1/3"
    finally:
        runner.close()


def test_on_complete_notifies_once_terminal() -> None:
    completed: list[Job] = []
    runner = _runner(on_complete=completed.append)
    try:
        job = runner.submit("notify", "x", lambda p, c: "r")
        runner.wait(job.id, timeout=5)
        assert len(completed) == 1
        assert completed[0].state == "succeeded" and completed[0].result == "r"
    finally:
        runner.close()


def test_cancel_a_queued_job_never_runs_it() -> None:
    # One worker, occupied by a blocking job, so the second stays queued till cancelled.
    runner = _runner(max_workers=1)
    release = threading.Event()
    ran = threading.Event()
    try:
        blocker = runner.submit("block", "x", lambda p, c: release.wait(5))
        queued = runner.submit("q", "x", lambda p, c: ran.set())
        assert queued.state == "queued"
        cancelled = runner.cancel(queued.id)
        assert cancelled is not None and cancelled.state == "cancelled"
        release.set()
        runner.wait(blocker.id, timeout=5)
        assert not ran.is_set()  # the cancelled job's work never ran
    finally:
        release.set()
        runner.close()


def test_cancel_signals_a_running_job() -> None:
    runner = _runner()
    started = threading.Event()

    def cooperative(progress, cancelled):
        started.set()
        while not cancelled():
            pass
        return "stopped"

    try:
        job = runner.submit("run", "x", cooperative)
        assert started.wait(5)
        runner.cancel(job.id)
        done = runner.wait(job.id, timeout=5)
        assert done is not None and done.state == "cancelled"
    finally:
        runner.close()


def test_cancel_racing_the_worker_fires_on_complete_once() -> None:
    # A queued job cancelled while the one worker is busy is finished by cancel; when
    # the worker frees and its _run also sees the cancel, the terminal guard makes
    # on_complete fire once. Without the guard the freed worker fires a second time.
    completed: list[Job] = []
    runner = _runner(max_workers=1, on_complete=completed.append)
    release = threading.Event()
    try:
        runner.submit("block", "x", lambda p, c: release.wait(5))  # occupies the worker
        queued = runner.submit("q", "x", lambda p, c: "r")
        cancelled = runner.cancel(queued.id)
        assert cancelled is not None and cancelled.state == "cancelled"
        release.set()
        runner.close()  # waits for the freed worker to run (and no-op on) the cancel
        fired = [j for j in completed if j.id == queued.id]
        assert len(fired) == 1 and fired[0].state == "cancelled"
    finally:
        release.set()
        runner.close()


def test_terminal_jobs_release_their_handles() -> None:
    # _cancels / _futures must not grow unbounded; a finished job's handles get pruned.
    runner = _runner()
    try:
        job = runner.submit("k", "x", lambda p, c: "r")
        runner.wait(job.id, timeout=5)
        assert job.id not in runner._cancels
        assert job.id not in runner._futures
    finally:
        runner.close()


def test_orphaned_running_jobs_fail_on_startup() -> None:
    # A job left running when its process stopped is failed by a fresh runner, so it is
    # observable and re-submittable rather than stuck 'running' forever.
    store = InMemoryJobStore()
    store.create(Job(id="job_x", key="k", label="orphan", state="running"))
    runner = JobRunner(store=store)
    try:
        recovered = store.get("job_x")
        assert recovered is not None and recovered.state == "failed"
        assert "restarted" in (recovered.error or "")
    finally:
        runner.close()


def test_systemexit_in_work_still_fails_the_job() -> None:
    # A BaseException (SystemExit) from the work must finish the job, not leave it stuck
    # 'running' with a dead worker.
    runner = _runner()

    def bail(progress: object, cancelled: object) -> object:
        raise SystemExit("stop")

    try:
        job = runner.submit("k", "x", bail)
        done = None
        for _ in range(200):
            done = runner.get(job.id)
            if done is not None and done.done:
                break
            time.sleep(0.01)
        assert done is not None and done.state == "failed"
        assert "SystemExit" in (done.error or "")
    finally:
        runner.close()
