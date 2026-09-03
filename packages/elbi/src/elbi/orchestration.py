"""The orchestration service: materialize assets, track runs, run schedules.

Treats each derivation as a software-defined asset. It materializes a selection (all,
stale-only, or named assets and optionally their downstream) in dependency order via the
project runner, recording a run with per-asset status; reports staleness; and fires cron
schedules and data-change sensors. Materializing is idempotent (a fresh asset is a cache
hit), so re-runs and backfills are safe.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from elbi_core import Registry, Runner, asset_status, materialize_assets
from elbi_core.lineage import from_registry, node_id
from elbi_core.orchestration.run import MaterializationStep, RunResult

from .db import Store


def cron_due(
    cron: str, last_run_at: datetime | None, now: datetime, tz: str = ""
) -> bool:
    """Whether ``cron`` has come due since ``last_run_at``, read in ``tz``.

    ``tz`` is an IANA name, defaulting to UTC. It matters because "the 1st at 06:00" is
    a wall-clock statement: read in UTC it drifts an hour across a daylight-saving
    boundary, which is why Databricks pairs a schedule with a zone. An unknown zone
    falls back to UTC rather than silently never firing.

    croniter takes the zone from the base time it is given, so the whole of it is
    passing a zone-aware base; the offset arithmetic is zoneinfo's, including the
    boundary where a local hour repeats or does not exist.
    """
    if not cron:
        return False
    from croniter import croniter

    if not croniter.is_valid(cron):
        return False
    zone: Any = timezone.utc
    if tz:
        with contextlib.suppress(ZoneInfoNotFoundError, ValueError):
            zone = ZoneInfo(tz)
    if last_run_at is None:
        return True
    base = last_run_at
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    next_fire = croniter(cron, base.astimezone(zone)).get_next(datetime)
    return bool(next_fire <= now)


def _duration_ms(run: dict[str, Any]) -> float | None:
    """A run's wall-clock in ms from its start/finish, or None if it hasn't finished."""
    started, finished = run.get("started_at"), run.get("finished_at")
    if not started or not finished:
        return None
    delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    return delta.total_seconds() * 1000


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    """A run summary without internal fields."""
    return dict(run)


def _run_if_met(condition: str, upstream: list[str]) -> bool:
    """Whether a step's trigger rule is satisfied by its upstreams' states.

    An ``excluded`` (skipped) upstream is treated as a success, matching Databricks;
    a step with no upstreams always runs.
    """
    states = ["succeeded" if s == "excluded" else s for s in upstream]
    if not states:
        return True
    total = len(states)
    succeeded = states.count("succeeded")
    failed = states.count("failed")
    if condition == "at_least_one_success":
        return succeeded >= 1
    if condition == "all_done":
        return True
    if condition == "at_least_one_failed":
        return failed >= 1
    if condition == "all_failed":
        return failed == total
    # all_success / none_failed (an excluded upstream already counts as succeeded).
    return failed == 0


def _topo_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order steps so each follows its ``depends_on`` (a cycle keeps input order)."""
    by_id = {str(s["id"]): s for s in steps}
    ordered: list[dict[str, Any]] = []
    done: set[str] = set()
    remaining = list(steps)
    while remaining:
        progressed = False
        for step in list(remaining):
            deps = [str(d) for d in step.get("depends_on", []) if str(d) in by_id]
            if all(d in done for d in deps):
                ordered.append(step)
                done.add(str(step["id"]))
                remaining.remove(step)
                progressed = True
        if not progressed:  # a cycle (or dangling deps): append the rest as-is
            ordered.extend(remaining)
            break
    return ordered


def _check_error(exc: Exception) -> str:
    """A check's failure to run, said in terms of what the author has to change.

    The aggregate complaint is worth naming: a check is a condition each row must
    satisfy, the shape every data-quality tool uses, and an author reaching for
    ``sum(...) = 0`` otherwise gets a binder error about a WHERE clause never written.
    """
    first = str(exc).splitlines()[0]
    if "aggregate" in first.lower():
        return (
            f"{first}: a check is a condition each row must satisfy, so write it over "
            "columns (converted in (0, 1)), not as an aggregate over the table"
        )
    return first


def _evaluate_checks(
    rows: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    """Evaluate each check's boolean SQL ``expr`` against ``rows`` with DuckDB.

    Each check yields ``{name, severity, expr, state, failed, total}``, where ``failed``
    counts rows the constraint is not true for, DuckDB-evaluated exactly like a Delta
    Live Tables expectation. ``state`` is ``passed``, ``failed``, or ``error``: a check
    whose expression will not run is broken, not evidence about the data, so it reports
    no count at all rather than blaming every row, the distinction dbt draws between a
    test that fails and one that errors. Either still blocks at ``error`` severity, so a
    broken check can never pass silently.
    """
    import duckdb
    import pyarrow as pa

    results: list[dict[str, Any]] = []
    ok = True
    con = duckdb.connect(":memory:")
    try:
        # Evaluate each constraint through the relation API rather than a formatted
        # SELECT: the operator-authored expr is a boolean condition, and .filter takes
        # a condition (not a full statement) against an ephemeral in-memory table.
        base = con.from_arrow(pa.Table.from_pylist(rows)) if rows else None
        for check in checks:
            name, expr, severity = check["name"], check["expr"], check["severity"]
            result: dict[str, Any] = {
                "name": name,
                "severity": severity,
                "expr": expr,
                "total": len(rows),
                "failed": 0,
                "state": "passed",
            }
            if base is not None:
                try:
                    row = base.filter(f"NOT ({expr})").aggregate("count()").fetchone()
                    result["failed"] = int(row[0]) if row else 0
                    result["state"] = "failed" if result["failed"] else "passed"
                except duckdb.Error as exc:
                    result["state"] = "error"
                    result["failed"] = None
                    result["error"] = _check_error(exc)
            if severity == "error" and result["state"] != "passed":
                ok = False
            results.append(result)
        return results, ok
    finally:
        con.close()


class OrchestrationService:
    """Materialization, staleness, run history, and schedules over a project."""

    #: A run is flagged slow when it takes more than this multiple of the recent median.
    SLOW_RUN_FACTOR = 2.0

    def __init__(
        self,
        *,
        store: Store,
        registry_provider: Callable[[], Registry],
        make_runner: Callable[[], Runner],
        dataset_names: Callable[[], Sequence[str]] = tuple,
        dataset_hash: Callable[[str], str | None] | None = None,
        on_notify: Callable[[dict[str, Any]], None] | None = None,
        compute_backend: str = "subprocess",
    ) -> None:
        self._store = store
        self._registry_provider = registry_provider
        self._make_runner = make_runner
        # A callable, like ``registry_provider`` beside it: each graph is built per
        # call, so a table synced during this session becomes an asset without a
        # restart.
        self._dataset_names = dataset_names
        self._dataset_hash = dataset_hash
        # The compute backend derivations run on, from the project's sandbox config. The
        # runtime routes by provenance (human-authored → in-process; agent-authored →
        # this sandbox), all through the pluggable Executor protocol.
        self._compute_backend = compute_backend
        # Fires on a terminal run that warrants attention (failed, or slow vs. its
        # recent median): delivered as a webhook + audit record by the caller.
        self._on_notify = on_notify
        # Run ids requested to cancel; the materialize loop checks this between assets.
        self._cancelling: set[str] = set()

    # -- queries -----------------------------------------------------------------
    def status(self) -> list[dict[str, Any]]:
        """Every asset with its freshness, verdict, and last run, in dep order."""
        runner = self._make_runner()
        names = self._ordered_assets()
        freshness = asset_status(runner, names, self._store.last_materialized_version)
        verdicts = self._verdicts()
        latest = self._store.latest_materializations()
        return [
            {
                "asset": name,
                "status": freshness.get(name),
                "verdict": verdicts.get(name),
                "last_materialized_at": latest.get(name, {}).get("at"),
                "last_duration_ms": latest.get(name, {}).get("duration_ms"),
            }
            for name in names
            if name in freshness
        ]

    def history(self, *, limit: int = 20) -> dict[str, Any]:
        """The run-history matrix: ordered assets and recent runs with per-asset cells.

        Each run carries its per-asset step state (``cells``) so the UI can draw the
        runs-by-assets grid that surfaces flaky assets and failures at a glance.
        """
        assets = self._ordered_assets()
        runs = [_public_run(r) for r in self._store.list_orchestration_runs(limit)]
        cells = self._store.asset_states_for_runs([r["id"] for r in runs])
        return {
            "assets": assets,
            "runs": [{**run, "cells": cells.get(run["id"], {})} for run in runs],
        }

    def runs(self) -> list[dict[str, Any]]:
        """Run history, newest first."""
        return [_public_run(r) for r in self._store.list_orchestration_runs()]

    def run(self, run_id: str) -> dict[str, Any] | None:
        """A run with its per-asset steps."""
        return self._store.get_orchestration_run(run_id)

    def graph(self) -> dict[str, Any]:
        """The asset dependency DAG: nodes (with freshness + verdict) and edges.

        Nodes are the derivation assets; edges are their derivation→derivation deps (the
        same graph the runner materializes in order), so the UI can draw the pipeline
        instead of a flat list.
        """
        registry = self._registry_provider()
        graph = from_registry(registry, self._dataset_names())
        runner = self._make_runner()
        names = self._ordered_assets()
        freshness = asset_status(runner, names, self._store.last_materialized_version)
        verdicts = self._verdicts()
        present = {n for n in names if n in freshness}
        nodes = [
            {"asset": n, "status": freshness.get(n), "verdict": verdicts.get(n)}
            for n in names
            if n in present
        ]
        edges = []
        for edge in graph.edges():
            if not (
                edge.source.startswith("derivation:")
                and edge.target.startswith("derivation:")
            ):
                continue
            src, tgt = edge.source.split(":", 1)[1], edge.target.split(":", 1)[1]
            if src in present and tgt in present:
                edges.append({"from": src, "to": tgt})
        return {"nodes": nodes, "edges": edges}

    def open_run(
        self,
        *,
        cause: str = "manual",
        parent_run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> str:
        """Open a run in the 'running' state and return its id (for an async loop)."""
        return self._store.create_orchestration_run(
            cause=cause, parent_run_id=parent_run_id, workflow_id=workflow_id
        )

    def failed_assets(self, run_id: str) -> list[str]:
        """A run's failed assets: a retry re-runs these plus their downstream."""
        run = self._store.get_orchestration_run(run_id)
        if run is None:
            return []
        return [s["asset"] for s in run["steps"] if s["state"] == "failed"]

    def checks(self, asset: str | None = None) -> list[dict[str, Any]]:
        """Data-quality checks, optionally scoped to one asset."""
        return self._store.list_asset_checks(asset)

    def upsert_check(
        self,
        *,
        check_id: str | None = None,
        asset: str,
        name: str,
        expr: str,
        severity: str = "warn",
        enabled: bool = True,
    ) -> str:
        """Create or update a data-quality check; returns its id."""
        return self._store.upsert_asset_check(
            check_id=check_id,
            asset=asset,
            name=name,
            expr=expr,
            severity=severity,
            enabled=enabled,
        )

    def delete_check(self, check_id: str) -> bool:
        """Remove a data-quality check; return whether it existed."""
        return self._store.delete_asset_check(check_id)

    #: Config key for the per-asset retry count applied to a materialization.
    _RETRIES_KEY = "orchestration_max_retries"

    def retry_policy(self) -> int:
        """The configured per-asset retry count (default 1: one retry on failure)."""
        raw = self._store.get_config(self._RETRIES_KEY)
        try:
            return max(0, int(raw)) if raw is not None else 1
        except ValueError:
            return 1

    def set_retry_policy(self, max_retries: int) -> None:
        """Set how many times a transient asset failure is retried with backoff."""
        self._store.set_config(self._RETRIES_KEY, str(max(0, max_retries)))

    def compute(self) -> str:
        """The compute backend derivations materialize on (the project's sandbox)."""
        return self._compute_backend

    def request_cancel(self, run_id: str) -> None:
        """Ask a running materialization to stop at the next asset boundary."""
        self._cancelling.add(run_id)

    def finish_failed(self, run_id: str) -> None:
        """Mark a run failed: a backstop so an async run can't get stuck 'running'."""
        self._store.finish_orchestration_run(run_id, "failed")
        self._cancelling.discard(run_id)
        self._maybe_notify(run_id, "failed", [])

    def _maybe_notify(self, run_id: str, status: str, failed: list[str]) -> None:
        """Emit a notification for a terminal run that warrants attention.

        Fires on ``failed`` (per-asset retries are already exhausted by then, so a
        transient blip is muted) and on a run that ran slow versus its recent median:
        the two signals Databricks surfaces as Failure and Duration Warning.
        """
        if self._on_notify is None:
            return
        runs = self._store.list_orchestration_runs(limit=30)
        current = next((r for r in runs if r["id"] == run_id), None)
        if current is None:
            return
        if status == "failed":
            self._on_notify(
                {
                    "event": "run.failed",
                    "run_id": run_id,
                    # Popped by the notification handler before the webhook sees it.
                    "cause": current["cause"],
                    "failed": failed,
                }
            )
            return
        if status != "succeeded":
            return
        duration = _duration_ms(current)
        others = [
            d for r in runs if r["id"] != run_id and (d := _duration_ms(r)) is not None
        ]
        # Need a few prior runs before a median means anything, else every early run
        # looks anomalous. Mirrors the "2x median duration" pipeline-alerting heuristic.
        if duration is None or len(others) < 3:
            return
        median = _median(others)
        if median > 0 and duration > self.SLOW_RUN_FACTOR * median:
            self._on_notify(
                {
                    "event": "run.slow",
                    "run_id": run_id,
                    "cause": current["cause"],
                    "duration_ms": round(duration),
                    "median_ms": round(median),
                }
            )

    # -- materialization ---------------------------------------------------------
    def materialize(
        self,
        *,
        selection: str = "stale",
        assets: Sequence[str] | None = None,
        include_downstream: bool = False,
        cause: str = "manual",
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Materialize a selection of assets in dependency order, recording a run.

        ``assets`` names an explicit selection (with ``include_downstream`` to add their
        dependents); otherwise ``selection`` is 'all' or 'stale'. ``run_id`` reuses a
        run opened by :meth:`open_run` (async path); otherwise one is created here. The
        run stops at the next asset boundary if cancelled via :meth:`request_cancel`.
        """
        runner = self._make_runner()
        if run_id is None:
            run_id = self._store.create_orchestration_run(
                cause=cause, parent_run_id=parent_run_id
            )
        try:
            result = self._run_selection(
                run_id, runner, selection, assets, include_downstream
            )
            cancelled = run_id in self._cancelling
            status = (
                "cancelled" if cancelled else ("succeeded" if result.ok else "failed")
            )
            self._store.finish_orchestration_run(run_id, status)
            self._maybe_notify(run_id, status, result.failed)
        finally:
            self._cancelling.discard(run_id)
        return {
            "run_id": run_id,
            "ok": result.ok and not cancelled,
            "materialized": [s.asset for s in result.steps if s.state == "succeeded"],
            "skipped": [s.asset for s in result.steps if s.state == "skipped"],
            "failed": result.failed,
        }

    def _run_selection(
        self,
        run_id: str,
        runner: Runner,
        selection: str,
        assets: Sequence[str] | None,
        include_downstream: bool,
    ) -> RunResult:
        """Materialize a selection into an already-open run (no open/finish/notify).

        Shared by :meth:`materialize` (one selection) and :meth:`run_workflow` (a step
        per selection), so both record asset runs, checks, retries, and cancellation the
        same way.
        """
        ordered = self._select(runner, selection, assets, include_downstream)
        verdicts = self._verdicts()

        def record(step: MaterializationStep) -> None:
            self._store.record_asset_run(
                run_id,
                asset=step.asset,
                state=step.state,
                data_version=step.data_version,
                verdict=step.verdict,
                error=step.error,
                attempts=step.attempts,
                duration_ms=step.duration_ms,
                logs=step.logs,
                checks=list(step.checks),
            )

        return materialize_assets(
            runner,
            ordered,
            last_version=self._store.last_materialized_version,
            verdict_of=verdicts.get,
            check_asset=self._check_asset,
            max_retries=self.retry_policy(),
            on_step=record,
            should_cancel=lambda: run_id in self._cancelling,
        )

    def run_workflow(
        self, workflow_id: str, *, run_id: str | None = None
    ) -> dict[str, Any]:
        """Execute a workflow: run its steps in dependency order, gated by run-if rules.

        Each step materializes its selection; a step whose trigger rule isn't met by its
        upstreams is *excluded* (skipped). All steps record into one run so the run
        detail shows every asset that ran, and the per-step outcomes drive the control
        flow: the Databricks/Airflow model of conditional, branching orchestration.
        """
        workflow = self._store.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"workflow {workflow_id!r} not found")
        steps = _topo_steps(workflow["steps"])
        runner = self._make_runner()
        if run_id is None:
            run_id = self._store.create_orchestration_run(
                cause="workflow", workflow_id=workflow_id
            )
        states: dict[str, str] = {}
        outcomes: list[dict[str, Any]] = []
        for step in steps:
            sid = str(step["id"])
            run_if = str(step.get("run_if", "all_success"))
            deps = step.get("depends_on", [])
            upstream = [states.get(str(d), "excluded") for d in deps]
            if not _run_if_met(run_if, upstream):
                states[sid] = "excluded"
            else:
                result = self._run_selection(
                    run_id,
                    runner,
                    str(step.get("selection", "stale")),
                    list(step["assets"]) if step.get("assets") else None,
                    bool(step.get("include_downstream", False)),
                )
                states[sid] = "succeeded" if result.ok else "failed"
            outcomes.append({"id": sid, "state": states[sid], "run_if": run_if})
        self._store.set_workflow_steps(run_id, outcomes)
        failed = [sid for sid, state in states.items() if state == "failed"]
        status = "failed" if failed else "succeeded"
        self._store.finish_orchestration_run(run_id, status)
        self._maybe_notify(run_id, status, failed)
        return {"run_id": run_id, "ok": not failed, "steps": outcomes}

    def workflows(self) -> list[dict[str, Any]]:
        """All workflow definitions."""
        return self._store.list_workflows()

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        """One workflow definition, or None."""
        return self._store.get_workflow(workflow_id)

    def upsert_workflow(
        self, *, workflow_id: str | None = None, name: str, steps: list[dict[str, Any]]
    ) -> str:
        """Create or replace a workflow definition; returns its id."""
        return self._store.upsert_workflow(
            workflow_id=workflow_id, name=name, steps=steps
        )

    def delete_workflow(self, workflow_id: str) -> bool:
        """Remove a workflow definition; return whether it existed."""
        return self._store.delete_workflow(workflow_id)

    def backfill(
        self,
        *,
        asset: str,
        param: str,
        values: Sequence[Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Materialize a parameterized ``asset`` once per partition value.

        Each value is one partition; the derivation runs with ``{param: value}`` and is
        recorded as its own step (labelled with the partition), the way Dagster
        backfills an asset across partition keys. Unchanged partitions are cache hits at
        the runtime layer, so a re-run is cheap. ``run_id`` reuses a run opened by
        :meth:`open_run` (the async path); otherwise one is created here.
        """
        runner = self._make_runner()
        verdict = self._verdicts().get(asset)
        if run_id is None:
            run_id = self._store.create_orchestration_run(cause="backfill")
        failed: list[Any] = []
        for value in values:
            params = {param: value}
            label = f"{asset} [{param}={value}]"
            started = time.monotonic()
            try:
                runner.run(asset, params)
                version: str | None = runner.data_version(asset, params)
                state, error = "succeeded", None
            except Exception as exc:  # a bad partition fails only its own step
                version, state, error = None, "failed", str(exc)
                failed.append(value)
            self._store.record_asset_run(
                run_id,
                asset=label,
                state=state,
                data_version=version,
                verdict=verdict,
                error=error,
                attempts=1,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        status = "failed" if failed else "succeeded"
        self._store.finish_orchestration_run(run_id, status)
        self._maybe_notify(run_id, status, [f"{asset} [{param}={v}]" for v in failed])
        return {
            "run_id": run_id,
            "ok": not failed,
            "partitions": len(values),
            "failed": failed,
        }

    # -- schedules ---------------------------------------------------------------
    def upsert_schedule(
        self,
        *,
        name: str,
        selection: str = "stale",
        mode: str = "cron",
        cron: str = "",
        dataset: str | None = None,
    ) -> str:
        """Create or replace a materialization schedule (cron or data-change sensor)."""
        return self._store.upsert_materialization_schedule(
            name=name, selection=selection, mode=mode, cron=cron, dataset=dataset
        )

    def schedules(self) -> list[dict[str, Any]]:
        """The materialization schedules."""
        return [
            {
                "id": row.id,
                "name": row.name,
                "selection": row.selection,
                "mode": row.mode,
                "cron": row.cron,
                "dataset": row.dataset,
                "enabled": row.enabled,
                "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
            }
            for row in self._store.list_materialization_schedules()
        ]

    def delete_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule; return whether it existed."""
        return self._store.delete_materialization_schedule(schedule_id)

    def run_due_schedules(self, now: datetime | None = None) -> list[str]:
        """Fire every schedule that is due; return the names that fired.

        A cron schedule is due when its next fire time after the last run has passed; a
        sensor is due when its dataset's content hash has moved since it last fired.
        """
        moment = now or datetime.now(timezone.utc)
        fired: list[str] = []
        for schedule in self._store.active_materialization_schedules():
            data_hash: str | None = None
            if schedule.mode == "on_data_change":
                if schedule.dataset is None or self._dataset_hash is None:
                    continue
                data_hash = self._dataset_hash(schedule.dataset)
                if data_hash is None or data_hash == schedule.last_data_hash:
                    continue
            elif not self._cron_due(schedule.cron, schedule.last_run_at, moment):
                continue
            selection = schedule.selection
            explicit = None if selection in ("all", "stale") else selection.split(",")
            cause = "sensor" if schedule.mode == "on_data_change" else "schedule"
            self.materialize(
                selection=selection if explicit is None else "stale",
                assets=[a.strip() for a in explicit] if explicit else None,
                cause=cause,
            )
            self._store.touch_materialization_schedule(
                schedule.id, last_data_hash=data_hash
            )
            fired.append(schedule.name)
        return fired

    # -- internals ---------------------------------------------------------------
    def _ordered_assets(self) -> list[str]:
        graph = from_registry(self._registry_provider(), self._dataset_names())
        return [
            nid.split(":", 1)[1]
            for nid in graph.topo_order()
            if nid.startswith("derivation:")
        ]

    def _select(
        self,
        runner: Runner,
        selection: str,
        assets: Sequence[str] | None,
        include_downstream: bool,
    ) -> list[str]:
        registry = self._registry_provider()
        graph = from_registry(registry, self._dataset_names())
        ordered = [
            nid.split(":", 1)[1]
            for nid in graph.topo_order()
            if nid.startswith("derivation:")
        ]
        if assets:
            chosen = set(assets)
            if include_downstream:
                for asset in list(assets):
                    for dep in graph.descendants(node_id("derivation", asset)):
                        if dep.startswith("derivation:"):
                            chosen.add(dep.split(":", 1)[1])
        elif selection == "all":
            chosen = set(ordered)
        else:
            freshness = asset_status(
                runner, ordered, self._store.last_materialized_version
            )
            chosen = {n for n, s in freshness.items() if s in ("stale", "never")}
        return [name for name in ordered if name in chosen]

    def _verdicts(self) -> dict[str, str | None]:
        return {row.name: row.verdict for row in self._store.list_derivations()}

    def _check_asset(self, name: str) -> tuple[list[dict[str, Any]], bool]:
        """Run an asset's enabled data-quality checks on its output (a cache hit)."""
        checks = self._store.list_asset_checks(name, enabled_only=True)
        if not checks:
            return [], True
        artifact = self._make_runner().run(name)
        rows = artifact.value if artifact.kind == "table" else []
        return _evaluate_checks(list(rows), checks)

    def _cron_due(self, cron: str, last_run_at: datetime | None, now: datetime) -> bool:
        return cron_due(cron, last_run_at, now)
