"""Integration tests for the orchestration API over a real store and project runner.

Drive the HTTP surface: read asset staleness, materialize a selection (recording a run),
see fresh assets skipped, watch a source change make an asset stale and re-materialize,
and fire a due cron schedule through the scheduler tick.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import open_store
from elbi.orchestration import OrchestrationService
from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    Runner,
    derivation,
    param,
    serve,
)
from elbi_core.config import DataBindings
from elbi_core.registry import use_registry

_TERMINAL = {"succeeded", "failed", "cancelled"}


def _materialize(http: TestClient, **body: object) -> dict[str, object]:
    """Start a materialization, wait for it to finish, and summarize the run.

    The endpoint runs the run asynchronously (returning ``{runId, status: running}``);
    for tests we poll the run detail until it reaches a terminal state and return the
    familiar ``ok/materialized/skipped/failed`` shape derived from its steps.
    """
    run_id = http.post("/api/orchestration/materialize", json=body).json()["runId"]
    for _ in range(200):
        detail = http.get(f"/api/orchestration/runs/{run_id}").json()
        if detail["status"] in _TERMINAL:
            break
        time.sleep(0.02)
    else:  # pragma: no cover - only trips if a run wedges 'running'
        raise AssertionError(f"run {run_id} did not finish: {detail['status']}")
    steps = detail["steps"]
    return {
        "runId": run_id,
        "status": detail["status"],
        "ok": detail["status"] == "succeeded",
        "materialized": [s["asset"] for s in steps if s["state"] == "succeeded"],
        "skipped": [s["asset"] for s in steps if s["state"] == "skipped"],
        "failed": [s["asset"] for s in steps if s["state"] == "failed"],
    }


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def clean_sales(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("sales").rows)

        @derivation(inputs={"rows": clean_sales}, serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("rows").value)

    return registry


def _build_client(
    tmp_path: Path, registry: Registry
) -> Iterator[tuple[TestClient, Path]]:
    csv = tmp_path / "sales.csv"
    csv.write_text("amount\n10\n20\n", encoding="utf-8")
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    bindings = DataBindings(bindings={"sales": "sales.csv"})

    def make_runner() -> Runner:
        return Runner(registry, bindings=bindings, base_dir=tmp_path)

    service = OrchestrationService(
        store=store,
        registry_provider=lambda: registry,
        make_runner=make_runner,
        dataset_names=lambda: ["sales"],
    )
    app = create_app(
        load_datasets=lambda: {},
        client=MagicMock(),
        store=store,
        orchestration_service=service,
    )
    with TestClient(app) as http:
        yield http, csv


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    yield from _build_client(tmp_path, _registry())


def test_status_starts_all_never(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    status = http.get("/api/orchestration/status").json()
    assert {s["asset"]: s["status"] for s in status} == {
        "clean_sales": "never",
        "revenue": "never",
    }


def test_materialize_all_then_fresh(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    result = _materialize(http, selection="all")
    assert result["ok"]
    assert set(result["materialized"]) == {"clean_sales", "revenue"}

    status = {
        s["asset"]: s["status"] for s in http.get("/api/orchestration/status").json()
    }
    assert status == {"clean_sales": "materialized", "revenue": "materialized"}

    # A second run finds everything fresh and skips it.
    again = _materialize(http, selection="all")
    assert again["skipped"] == ["clean_sales", "revenue"]
    assert again["materialized"] == []


def test_source_change_makes_stale_and_rematerializes(
    client: tuple[TestClient, Path],
) -> None:
    http, csv = client
    _materialize(http, selection="all")
    csv.write_text("amount\n10\n20\n30\n", encoding="utf-8")  # source changed

    status = {
        s["asset"]: s["status"] for s in http.get("/api/orchestration/status").json()
    }
    assert status["clean_sales"] == "stale"

    result = _materialize(http, selection="stale")
    assert "clean_sales" in result["materialized"]


def test_runs_history_and_detail(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    run_id = _materialize(http, selection="all")["runId"]
    runs = http.get("/api/orchestration/runs").json()
    assert runs[0]["id"] == run_id
    assert runs[0]["status"] == "succeeded"
    detail = http.get(f"/api/orchestration/runs/{run_id}").json()
    assert {s["asset"] for s in detail["steps"]} == {"clean_sales", "revenue"}


def test_cron_schedule_fires_on_tick(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    http.post(
        "/api/orchestration/schedules",
        json={
            "name": "nightly",
            "selection": "all",
            "mode": "cron",
            "cron": "* * * * *",
        },
    )
    assert http.get("/api/orchestration/schedules").json()[0]["name"] == "nightly"
    # The scheduler tick fires a due schedule (never-run cron is due immediately).
    http.app.state.orchestration_tick()
    runs = http.get("/api/orchestration/runs").json()
    assert any(r["cause"] == "schedule" for r in runs)


def test_delete_schedule(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    sid = http.post(
        "/api/orchestration/schedules",
        json={"name": "s", "selection": "stale", "mode": "cron", "cron": "0 9 * * *"},
    ).json()["id"]
    assert http.delete(f"/api/orchestration/schedules/{sid}").status_code == 200
    assert http.get("/api/orchestration/schedules").json() == []


def test_status_reports_last_materialized(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    before = http.get("/api/orchestration/status").json()
    assert all(s["lastMaterializedAt"] is None for s in before)  # never run yet

    _materialize(http, selection="all")
    after = {s["asset"]: s for s in http.get("/api/orchestration/status").json()}
    assert after["clean_sales"]["lastMaterializedAt"] is not None
    assert after["clean_sales"]["lastDurationMs"] >= 0


def test_history_matrix_has_runs_and_cells(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    _materialize(http, selection="all")
    _materialize(http, selection="all")  # second run: everything skipped
    history = http.get("/api/orchestration/history").json()
    assert set(history["assets"]) == {"clean_sales", "revenue"}
    assert len(history["runs"]) == 2
    # Newest first; the latest run found everything fresh and skipped it.
    assert history["runs"][0]["cells"] == {
        "clean_sales": "skipped",
        "revenue": "skipped",
    }
    assert history["runs"][1]["cells"] == {
        "clean_sales": "succeeded",
        "revenue": "succeeded",
    }


def test_graph_has_nodes_and_dependency_edge(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    graph = http.get("/api/orchestration/graph").json()
    assert {n["asset"] for n in graph["nodes"]} == {"clean_sales", "revenue"}
    # revenue depends on clean_sales, so the DAG has that one directed edge.
    assert {"from": "clean_sales", "to": "revenue"} in graph["edges"]


def _logging_registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def noisy(ctx: Context) -> Artifact:
            print("hello from the derivation")  # captured into the run log
            return Artifact.table(ctx.input("sales").rows)

    return registry


def test_step_captures_stdout_as_logs(tmp_path: Path) -> None:
    for http, _ in _build_client(tmp_path, _logging_registry()):
        run_id = _materialize(http, selection="all")["runId"]
        detail = http.get(f"/api/orchestration/runs/{run_id}").json()
        step = next(s for s in detail["steps"] if s["asset"] == "noisy")
        assert "hello from the derivation" in step["logs"]


def _flaky_registry(fail_file: Path) -> Registry:
    """A derivation that fails until ``fail_file`` is removed: lets a retry succeed."""
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def maybe(ctx: Context) -> Artifact:
            if fail_file.exists():
                raise RuntimeError("boom")
            return Artifact.table(ctx.input("sales").rows)

    return registry


def test_retry_rematerializes_failed_assets(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail"
    fail_file.write_text("x", encoding="utf-8")
    for http, _ in _build_client(tmp_path, _flaky_registry(fail_file)):
        first = _materialize(http, selection="all")
        assert first["failed"] == ["maybe"]
        run_id = first["runId"]

        fail_file.unlink()  # clear the failure so the retry can succeed
        retry = http.post(f"/api/orchestration/runs/{run_id}/retry").json()
        retry_id = retry["runId"]
        for _ in range(200):
            detail = http.get(f"/api/orchestration/runs/{retry_id}").json()
            if detail["status"] in _TERMINAL:
                break
            time.sleep(0.02)
        assert detail["status"] == "succeeded"
        assert detail["parentRunId"] == run_id
        assert {s["asset"] for s in detail["steps"]} == {"maybe"}


def _service_with_notify(
    tmp_path: Path, registry: Registry, events: list[dict[str, object]]
) -> OrchestrationService:
    (tmp_path / "sales.csv").write_text("amount\n1\n", encoding="utf-8")
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    bindings = DataBindings(bindings={"sales": "sales.csv"})
    return OrchestrationService(
        store=store,
        registry_provider=lambda: registry,
        make_runner=lambda: Runner(registry, bindings=bindings, base_dir=tmp_path),
        dataset_names=lambda: ["sales"],
        on_notify=events.append,
    )


def _flappy_registry(state: dict[str, int]) -> Registry:
    """A derivation that raises on the first ``state['fails']`` tries, then succeeds."""
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def flappy(ctx: Context) -> Artifact:
            if state["fails"] > 0:
                state["fails"] -= 1
                raise RuntimeError("transient")
            return Artifact.table(ctx.input("sales").rows)

    return registry


def _partitioned_registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(params={"region": param.string()}, serve=serve.table())
        def by_region(ctx: Context) -> Artifact:
            return Artifact.table([{"region": ctx.param("region"), "n": 1}])

    return registry


def test_backfill_materializes_each_partition(tmp_path: Path) -> None:
    service = _service_with_notify(tmp_path, _partitioned_registry(), [])
    result = service.backfill(
        asset="by_region", param="region", values=["west", "east", "north"]
    )
    assert result["ok"] is True
    assert result["partitions"] == 3
    detail = service.run(result["run_id"])
    assert detail is not None
    labels = {s["asset"] for s in detail["steps"]}
    assert labels == {
        "by_region [region=west]",
        "by_region [region=east]",
        "by_region [region=north]",
    }
    assert all(s["state"] == "succeeded" for s in detail["steps"])


def test_backfill_bad_partition_fails_only_its_step(tmp_path: Path) -> None:
    service = _service_with_notify(tmp_path, _partitioned_registry(), [])
    # An unknown param name makes every partition's run raise → all steps fail.
    result = service.backfill(asset="by_region", param="nope", values=["x"])
    assert result["ok"] is False
    assert result["failed"] == ["x"]
    detail = service.run(result["run_id"])
    assert detail is not None
    assert detail["steps"][0]["state"] == "failed"


def test_settings_reports_retries_and_compute(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    settings = http.get("/api/orchestration/settings").json()
    assert settings["maxRetries"] == 1
    assert "compute" in settings  # the configured sandbox/compute backend
    http.put("/api/orchestration/settings", json={"maxRetries": 3})
    assert http.get("/api/orchestration/settings").json()["maxRetries"] == 3


def test_run_failed_notification_fires(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail"
    fail_file.write_text("x", encoding="utf-8")
    events: list[dict[str, object]] = []
    service = _service_with_notify(tmp_path, _flaky_registry(fail_file), events)
    service.materialize(selection="all")
    assert [e["event"] for e in events] == ["run.failed"]
    assert events[0]["failed"] == ["maybe"]


def test_successful_run_does_not_notify(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    service = _service_with_notify(tmp_path, _registry(), events)
    service.materialize(selection="all")
    assert events == []  # a clean, first run is neither failed nor slow


def test_slow_run_notification_uses_median(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, object]] = []
    service = _service_with_notify(tmp_path, _registry(), events)

    def _run(rid: str, micros: int) -> dict[str, object]:
        return {
            "id": rid,
            "cause": "manual",
            "started_at": "2026-07-10T00:00:00+00:00",
            "finished_at": f"2026-07-10T00:00:00.{micros:06d}+00:00",
        }

    # cur ran 300ms; the three priors ran 100ms → median 100ms, and 300 > 2x100.
    history = [_run("cur", 300000)] + [_run(x, 100000) for x in ("a", "b", "c")]
    monkeypatch.setattr(
        service._store, "list_orchestration_runs", lambda limit=30: history
    )
    service._maybe_notify("cur", "succeeded", [])
    assert [e["event"] for e in events] == ["run.slow"]
    assert events[0]["median_ms"] == 100

    # Fewer than three prior runs → no median → no false "slow" alert.
    events.clear()
    monkeypatch.setattr(
        service._store, "list_orchestration_runs", lambda limit=30: history[:2]
    )
    service._maybe_notify("cur", "succeeded", [])
    assert events == []


def test_run_if_met_matches_databricks_rules() -> None:
    from elbi.orchestration import _run_if_met

    assert _run_if_met("all_success", []) is True  # no upstreams → always runs
    assert _run_if_met("all_success", ["succeeded", "succeeded"]) is True
    assert _run_if_met("all_success", ["succeeded", "failed"]) is False
    assert _run_if_met("all_success", ["succeeded", "excluded"]) is True  # excluded==ok
    assert _run_if_met("at_least_one_failed", ["succeeded", "failed"]) is True
    assert _run_if_met("at_least_one_failed", ["succeeded"]) is False
    assert _run_if_met("all_done", ["failed"]) is True
    assert _run_if_met("all_failed", ["failed", "succeeded"]) is False
    assert _run_if_met("at_least_one_success", ["failed", "succeeded"]) is True


def test_topo_steps_orders_by_dependency() -> None:
    from elbi.orchestration import _topo_steps

    steps = [
        {"id": "c", "depends_on": ["b"]},
        {"id": "a", "depends_on": []},
        {"id": "b", "depends_on": ["a"]},
    ]
    assert [s["id"] for s in _topo_steps(steps)] == ["a", "b", "c"]


def _mixed_registry(fail_file: Path) -> Registry:
    """Two assets: ``flaky`` raises while ``fail_file`` exists; ``steady`` always ok."""
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def flaky(ctx: Context) -> Artifact:
            if fail_file.exists():
                raise RuntimeError("boom")
            return Artifact.table(ctx.input("sales").rows)

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def steady(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("sales").rows)

    return registry


def test_a_workflow_can_be_created_and_deleted_over_http(
    client: tuple[TestClient, Path],
) -> None:
    """Deleting one had no test, so neither did its 404.

    The service-level workflow tests below call `upsert_workflow` directly, which skips
    the routes entirely.
    """
    http, _ = client
    created = http.post(
        "/api/orchestration/workflows",
        json={"name": "nightly", "steps": [{"id": "a", "selection": "all"}]},
    )
    assert created.status_code == 200, created.text
    workflow_id = created.json()["id"]

    listed = http.get("/api/orchestration/workflows").json()
    assert workflow_id in {w["id"] for w in listed}

    assert http.delete(f"/api/orchestration/workflows/{workflow_id}").status_code == 200
    assert http.get("/api/orchestration/workflows").json() == []
    assert http.delete(f"/api/orchestration/workflows/{workflow_id}").status_code == 404


def test_workflow_success_branch(tmp_path: Path) -> None:
    service = _service_with_notify(tmp_path, _registry(), [])
    wid = service.upsert_workflow(
        name="wf",
        steps=[
            {"id": "a", "selection": "all"},
            {
                "id": "b",
                "depends_on": ["a"],
                "selection": "all",
                "run_if": "all_success",
            },
            {
                "id": "c",
                "depends_on": ["a"],
                "selection": "all",
                "run_if": "at_least_one_failed",
            },
        ],
    )
    result = service.run_workflow(wid)
    states = {s["id"]: s["state"] for s in result["steps"]}
    assert states == {"a": "succeeded", "b": "succeeded", "c": "excluded"}
    assert result["ok"] is True


def test_workflow_failure_branch_runs_cleanup(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail"
    fail_file.write_text("x", encoding="utf-8")
    service = _service_with_notify(tmp_path, _mixed_registry(fail_file), [])
    wid = service.upsert_workflow(
        name="wf",
        steps=[
            {"id": "build", "assets": ["flaky"]},
            {
                "id": "publish",
                "depends_on": ["build"],
                "assets": ["steady"],
                "run_if": "all_success",
            },
            {
                "id": "cleanup",
                "depends_on": ["build"],
                "assets": ["steady"],
                "run_if": "at_least_one_failed",
            },
        ],
    )
    result = service.run_workflow(wid)
    states = {s["id"]: s["state"] for s in result["steps"]}
    assert states["build"] == "failed"
    assert states["publish"] == "excluded"  # all_success not met → skipped
    assert states["cleanup"] == "succeeded"  # at_least_one_failed met → ran
    assert result["ok"] is False


def test_evaluate_checks_counts_failures_and_flags_error() -> None:
    from elbi.orchestration import _evaluate_checks

    rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    results, ok = _evaluate_checks(
        rows,
        [
            {"name": "small", "expr": "x < 3", "severity": "error"},
            {"name": "positive", "expr": "x > 0", "severity": "warn"},
        ],
    )
    by_name = {r["name"]: r for r in results}
    assert by_name["small"]["failed"] == 1 and by_name["small"]["total"] == 3
    assert by_name["positive"]["failed"] == 0
    assert ok is False  # an error-severity check failed


def test_evaluate_checks_bad_expr_surfaces_error() -> None:
    """A broken check reports no count: it is not evidence that any row is bad."""
    from elbi.orchestration import _evaluate_checks

    results, ok = _evaluate_checks(
        [{"x": 1}], [{"name": "bad", "expr": "no_such_col >", "severity": "warn"}]
    )
    assert results[0]["state"] == "error" and "error" in results[0]
    assert results[0]["failed"] is None, "no rows were judged, so none can be blamed"
    assert ok is True  # warn severity, so the run still passes


def test_a_broken_error_check_blocks_without_blaming_every_row() -> None:
    """The aggregate mistake, which is the one an author actually makes.

    Writing a check as an aggregate is natural and wrong -- checks are row conditions --
    and it used to be reported as every row failing, sending the reader to look for bad
    data that does not exist. It must block the asset and say the check is the problem.
    """
    from elbi.orchestration import _evaluate_checks

    rows = [{"arm": "a"}, {"arm": "b"}]
    results, ok = _evaluate_checks(
        rows,
        [{"name": "balanced", "expr": "count(*) = 2", "severity": "error"}],
    )
    assert ok is False, "an error-severity check that cannot run must still block"
    assert results[0]["state"] == "error"
    assert results[0]["failed"] is None
    assert "row" in results[0]["error"], results[0]["error"]


def test_the_step_names_a_broken_check_rather_than_a_data_problem() -> None:
    from elbi_core.orchestration.run import _check_failure

    assert _check_failure(({"name": "x", "state": "failed"},)) == (
        "data-quality check failed"
    )
    broken = _check_failure(({"name": "balanced", "state": "error"},))
    assert "could not run" in broken and "balanced" in broken


def test_error_check_fails_the_run(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    http.post(
        "/api/orchestration/checks",
        json={
            "asset": "clean_sales",
            "name": "big_amounts",
            "expr": "CAST(amount AS INTEGER) > 100",
            "severity": "error",
        },
    )
    result = _materialize(http, selection="all")
    assert result["status"] == "failed"
    assert "clean_sales" in result["failed"]
    detail = http.get(f"/api/orchestration/runs/{result['runId']}").json()
    step = next(s for s in detail["steps"] if s["asset"] == "clean_sales")
    assert step["state"] == "failed"
    check = step["checks"][0]
    assert check["name"] == "big_amounts"
    assert check["failed"] == 2 and check["total"] == 2


def test_warn_check_records_but_run_succeeds(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    http.post(
        "/api/orchestration/checks",
        json={
            "asset": "clean_sales",
            "name": "big_amounts",
            "expr": "CAST(amount AS INTEGER) > 100",
            "severity": "warn",
        },
    )
    result = _materialize(http, selection="all")
    assert result["status"] == "succeeded"
    detail = http.get(f"/api/orchestration/runs/{result['runId']}").json()
    step = next(s for s in detail["steps"] if s["asset"] == "clean_sales")
    assert step["state"] == "succeeded"
    assert step["checks"][0]["failed"] == 2


def test_check_runs_on_skipped_fresh_asset(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    _materialize(http, selection="all")  # both materialize and are now fresh
    # A check added *after* materialization still takes effect on the next run, even
    # though the asset is skipped (fresh): checks re-validate the current data.
    http.post(
        "/api/orchestration/checks",
        json={
            "asset": "clean_sales",
            "name": "big_amounts",
            "expr": "CAST(amount AS INTEGER) > 100",
            "severity": "error",
        },
    )
    result = _materialize(http, selection="all")
    assert result["status"] == "failed"
    detail = http.get(f"/api/orchestration/runs/{result['runId']}").json()
    step = next(s for s in detail["steps"] if s["asset"] == "clean_sales")
    assert step["state"] == "failed"
    assert step["checks"][0]["failed"] == 2


def test_check_crud(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    cid = http.post(
        "/api/orchestration/checks",
        json={"asset": "clean_sales", "name": "not_null", "expr": "amount IS NOT NULL"},
    ).json()["id"]
    checks = http.get("/api/orchestration/checks?asset=clean_sales").json()
    assert [c["name"] for c in checks] == ["not_null"]
    assert checks[0]["severity"] == "warn"  # the default
    assert http.delete(f"/api/orchestration/checks/{cid}").status_code == 200
    assert http.get("/api/orchestration/checks").json() == []


def test_retry_with_no_failures_is_rejected(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    run_id = _materialize(http, selection="all")["runId"]
    resp = http.post(f"/api/orchestration/runs/{run_id}/retry")
    assert resp.status_code == 400


def test_cancel_unknown_run_is_404(client: tuple[TestClient, Path]) -> None:
    http, _ = client
    assert http.post("/api/orchestration/runs/nope/cancel").status_code == 404


def test_cancel_stops_run_at_asset_boundary(tmp_path: Path) -> None:
    # Drive the service directly: cancelling before the loop starts leaves every asset
    # unstarted, so the run is recorded 'cancelled' with no successful steps.
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    csv = tmp_path / "sales.csv"
    csv.write_text("amount\n1\n", encoding="utf-8")
    registry = _registry()
    bindings = DataBindings(bindings={"sales": "sales.csv"})
    service = OrchestrationService(
        store=store,
        registry_provider=lambda: registry,
        make_runner=lambda: Runner(registry, bindings=bindings, base_dir=tmp_path),
        dataset_names=lambda: ["sales"],
    )
    run_id = service.open_run(cause="manual")
    service.request_cancel(run_id)
    result = service.materialize(selection="all", run_id=run_id)
    assert result["ok"] is False
    detail = store.get_orchestration_run(run_id)
    assert detail is not None
    assert detail["status"] == "cancelled"
    assert detail["steps"] == []


def test_a_missing_required_field_is_a_400_naming_it(
    client: tuple[TestClient, Path],
) -> None:
    """A body without a field the route needs gets an answer it can act on.

    These routes parse their own bodies rather than going through a schema, so the
    check is theirs to make. Indexing the body directly turns the omission into a
    KeyError and a 500, which tells the caller only that something broke.
    """
    http, _ = client
    for path, field in (
        ("/api/orchestration/checks", "asset"),
        ("/api/orchestration/backfill", "asset"),
        ("/api/orchestration/workflows", "name"),
        ("/api/orchestration/schedules", "name"),
    ):
        response = http.post(path, json={})
        assert response.status_code == 400, f"{path} -> {response.status_code}"
        assert field in response.json()["detail"], path
    retries = http.put("/api/orchestration/settings", json={})
    assert retries.status_code == 400
    assert "max_retries" in retries.json()["detail"]
