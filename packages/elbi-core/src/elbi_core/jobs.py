"""Durable background jobs: run a long computation asynchronously and observe it.

A job is submitted, runs detached on a worker, and moves through
``queued -> running -> succeeded | failed | cancelled``. The caller gets a job id
back at once and never holds the call open; completion is delivered by an
``on_complete`` notification (and by polling :meth:`JobRunner.get`). This is the
launch-and-continue pattern data agents use for training that outlives a single
tool call, mirroring the async-job lifecycle the platforms converged on (submit ->
handle -> poll/notify -> cancel).

Jobs are **content-addressed** by a ``key`` the caller derives from what fully
determines the result (for a derivation: its source plus its input binding). A
submit whose key matches a live or finished job returns that job instead of running
the work twice, so identical training dedupes for free. That is the same property
the derivation cache already provides, lifted to the job layer.

:class:`JobStore` is the pluggable persistence seam, like
:class:`~elbi.cache.CacheStore`: :class:`InMemoryJobStore` here for a single
process, and a database-backed store where jobs must survive a restart. The runner
itself is stdlib threads only; the heavy compute runs behind the
:class:`~elbi.Executor` seam, so a local worker today swaps for a hosted or
GPU backend with no change to this layer.
"""

from __future__ import annotations

import contextlib
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

#: The lifecycle of a job. ``queued`` and ``running`` are live; the rest are terminal.
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]

_LIVE_STATES: frozenset[JobState] = frozenset({"queued", "running", "succeeded"})


def new_job_id() -> str:
    """A short, unguessable job id.

    Public because work that runs inline -- with no runner to queue it -- still answers
    with a job, and must mint an id in the same shape a queued one carries.
    """
    return "job_" + secrets.token_hex(8)


@dataclass(frozen=True)
class Job:
    """One background computation and its observable state.

    ``key`` is the content address the caller submitted under, so an identical
    submission maps back to this job. ``progress`` is the latest line the work
    reported (a training epoch, say); ``result`` and ``error`` are set once terminal.
    """

    id: str
    key: str
    label: str
    state: JobState
    progress: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.state in ("succeeded", "failed", "cancelled")


class JobStore(Protocol):
    """Where jobs live: the pluggable seam between the runner and its persistence.

    An in-memory implementation suffices for one process; a database-backed one lets
    jobs (and their results) outlive a restart, which is what makes the execution
    durable. Implementations must be safe to call from several worker threads at once.
    """

    def create(self, job: Job) -> None:
        """Persist a newly submitted job."""
        ...

    def get(self, job_id: str) -> Job | None:
        """The job of this id, or ``None`` if it was never submitted."""
        ...

    def find_by_key(self, key: str) -> Job | None:
        """The most recent job submitted under ``key``, for content-addressed dedupe."""
        ...

    def update(self, job_id: str, **changes: Any) -> Job:
        """Apply ``changes`` to the stored job and return the new value."""
        ...

    def list(self) -> list[Job]:
        """Every job, newest first."""
        ...


@dataclass
class InMemoryJobStore:
    """A process-local :class:`JobStore` backed by a dict under a lock.

    Loses its jobs when the process exits; use a database-backed store where they must
    survive a restart. The lock makes concurrent worker updates safe, and every read
    returns the immutable :class:`Job`, so callers never observe a partial update.
    """

    _jobs: dict[str, Job] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, job: Job) -> None:
        """Persist a newly submitted job."""
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)

    def get(self, job_id: str) -> Job | None:
        """The job of this id, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    def find_by_key(self, key: str) -> Job | None:
        """The most recent job submitted under ``key``."""
        with self._lock:
            for job_id in reversed(self._order):
                if self._jobs[job_id].key == key:
                    return self._jobs[job_id]
            return None

    def update(self, job_id: str, **changes: Any) -> Job:
        """Apply ``changes`` and return the new value."""
        with self._lock:
            updated = replace(self._jobs[job_id], **changes)
            self._jobs[job_id] = updated
            return updated

    def list(self) -> list[Job]:
        """Every job, newest first."""
        with self._lock:
            return [self._jobs[job_id] for job_id in reversed(self._order)]


#: The unit of work a job runs. It is handed a ``progress`` callback to report a line
#: and a ``cancelled`` predicate to check for cooperative cancellation, and returns the
#: job's result (which a persistent store must be able to serialize).
JobWork = Callable[[Callable[[str], None], Callable[[], bool]], Any]


@dataclass
class JobRunner:
    """Runs submitted work on a bounded thread pool and records its lifecycle.

    :meth:`submit` returns immediately with a :class:`Job`; the work runs on a worker
    and the store is updated as it progresses and finishes. ``on_complete`` fires once
    per job as it reaches a terminal state, which is how a caller is notified without
    polling (the app forwards it to the chat as a completion event).

    Cancellation is cooperative: :meth:`cancel` signals the job, a not-yet-started one
    is marked cancelled at once, and a running one stops when its work next checks the
    ``cancelled`` predicate (or when it finishes, whichever comes first). Hard-killing a
    running compute is the backend's job (the executor's own timeout bounds it) and is
    layered in with the hosted backends, not here.
    """

    store: JobStore
    max_workers: int = 4
    on_complete: Callable[[Job], None] | None = None
    _pool: ThreadPoolExecutor = field(init=False, repr=False)
    _cancels: dict[str, threading.Event] = field(
        init=False, default_factory=dict, repr=False
    )
    _futures: dict[str, Future[None]] = field(
        init=False, default_factory=dict, repr=False
    )
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="elbi-job"
        )
        self._reconcile_orphans()

    def _reconcile_orphans(self) -> None:
        """Fail jobs a prior process left mid-flight, so none stays stuck 'running'.

        A persistent store can hold a job that was ``queued`` or ``running`` when its
        process stopped; this fresh runner has no worker for it. Marking such a job
        failed makes it observable (and re-submittable) instead of pending forever,
        and ``on_complete`` fires for it like any other terminal transition, so a
        completion consumer (a notification, say) is not silently skipped just
        because the failure was a restart. An in-memory store starts empty, so this
        is a no-op there.
        """
        for job in self.store.list():
            if job.state in ("queued", "running"):
                failed = self.store.update(
                    job.id,
                    state="failed",
                    finished_at=time.time(),
                    error="interrupted: the process running this job restarted",
                )
                if self.on_complete is not None:
                    self.on_complete(failed)

    def submit(self, key: str, label: str, work: JobWork) -> Job:
        """Submit ``work`` under content address ``key``; dedupe onto a matching job.

        If a job already exists for ``key`` that is queued, running, or succeeded, it is
        returned as-is and ``work`` does not run again (the first submitter's
        A previously failed or cancelled key starts a
        fresh job, so a retry is possible.
        """
        with self._lock:
            existing = self.store.find_by_key(key)
            if existing is not None and existing.state in _LIVE_STATES:
                return existing
            job = Job(
                id=new_job_id(),
                key=key,
                label=label,
                state="queued",
                created_at=time.time(),
            )
            self.store.create(job)
            event = threading.Event()
            self._cancels[job.id] = event
            self._futures[job.id] = self._pool.submit(self._run, job.id, work, event)
            return job

    def get(self, job_id: str) -> Job | None:
        """The current state of a job (poll it, or read after an ``on_complete``)."""
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Signal a job to stop; a queued one is cancelled at once. Return its state."""
        event = self._cancels.get(job_id)
        if event is not None:
            event.set()
        job = self.store.get(job_id)
        if job is not None and job.state == "queued":
            return self._finish(job_id, state="cancelled")
        return job

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        """Block until the job finishes (or ``timeout``); return its state.

        For synchronous callers and tests; the async path uses ``on_complete`` instead.
        """
        future = self._futures.get(job_id)
        if future is not None:
            with contextlib.suppress(Exception):
                future.result(timeout=timeout)
        return self.store.get(job_id)

    def close(self) -> None:
        """Stop accepting work and wait for in-flight jobs to return."""
        self._pool.shutdown(wait=True)

    def _run(self, job_id: str, work: JobWork, cancel: threading.Event) -> None:
        if cancel.is_set():
            self._finish(job_id, state="cancelled")
            return
        self.store.update(job_id, state="running", started_at=time.time())

        def progress(line: str) -> None:
            self.store.update(job_id, progress=line)

        try:
            result = work(progress, cancel.is_set)
        except Exception as exc:  # any work failure becomes the job's error
            self._finish(job_id, state="failed", error=str(exc))
            return
        except BaseException as exc:
            # A BaseException (SystemExit, KeyboardInterrupt) from the work must still
            # finish the job before it propagates, not leave it stuck 'running'.
            self._finish(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        # A cancel that arrived while the work ran (and that the work did not honor)
        # still wins, so a cancelled job never reports a result.
        if cancel.is_set():
            self._finish(job_id, state="cancelled")
            return
        self._finish(job_id, state="succeeded", result=result)

    def _finish(self, job_id: str, **changes: Any) -> Job:
        # Transition to a terminal state at most once, so a cancel racing the worker
        # (both reaching for one job) cannot fire ``on_complete`` twice or clobber the
        # first terminal state. The check and write are one critical section, and the
        # per-job cancel/future handles are pruned here so they do not grow unbounded.
        with self._lock:
            current = self.store.get(job_id)
            if current is not None and current.done:
                return current
            job = self.store.update(job_id, finished_at=time.time(), **changes)
            self._cancels.pop(job_id, None)
            self._futures.pop(job_id, None)
        if self.on_complete is not None:
            self.on_complete(job)
        return job
