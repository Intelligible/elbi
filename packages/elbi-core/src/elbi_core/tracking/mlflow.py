"""Emit a certified run to a team's existing MLflow, so the certification is visible.

This does not reimplement MLflow; it is a thin *client* that logs a sealed certified run
to whatever MLflow server a team already runs, using only the stable public surface
(``MlflowClient`` plus entity classes). The attestation rides in run tags namespaced
``elbi.*`` (queryable in MLflow's UI and ``search_runs``), the certified
estimate becomes the run's metric, and the content-addressed ``data_hash`` becomes a
tag so the run's provenance is a visible fingerprint.

The client ships with elbi (the skinny, client-only MLflow distribution), so the
export works out of the box; a team only sets a tracking URI. It is *fail-soft by
design*: certification must never depend on the server being reachable, so an
unreachable or misconfigured server (or, defensively, an absent client) warns without
raising, and the run is recorded in the local history regardless.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

from .run import CertifiedRun


def emit_run(
    run: CertifiedRun, *, tracking_uri: str, experiment: str | None = None
) -> bool:
    """Log ``run`` to the MLflow server at ``tracking_uri``; return whether it was sent.

    Each derivation maps to one MLflow experiment (its name, so all its certified
    versions group there); ``experiment`` overrides that name. Returns ``False`` without
    raising when the server cannot be reached (or, defensively, the client is
    unavailable), so the caller's certification is unaffected either way.
    """
    entities = _mlflow_entities()
    if entities is None:
        return False
    client_cls, metric_cls, param_cls, tag_cls = entities
    try:
        client = client_cls(tracking_uri=tracking_uri)
        experiment_id = _experiment_id(client, experiment or run.name)
        timestamp = _epoch_ms(run.created_at)
        mlflow_run = client.create_run(
            experiment_id,
            start_time=timestamp,
            run_name=run.short_version,
            tags=_tags(run),
        )
        run_id = mlflow_run.info.run_id
        client.log_batch(
            run_id,
            metrics=[
                metric_cls(_key(key), value, timestamp, 0)
                for key, value in run.metrics().items()
            ],
            params=[
                param_cls(_key(key), str(value)) for key, value in _config(run).items()
            ],
            tags=[tag_cls(_key(key), str(value)) for key, value in _tags(run).items()],
        )
        client.set_terminated(run_id, status="FINISHED")
    # Fail-soft: a tracking export must never break the certification it records.
    except Exception as exc:
        warnings.warn(f"MLflow emit for {run.name!r} failed: {exc}", stacklevel=2)
        return False
    return True


def _mlflow_entities() -> tuple[Any, Any, Any, Any] | None:
    """Import MLflow's client and entity classes, or ``None`` if it is unavailable.

    The skinny client ships with elbi, so the import normally succeeds; the
    guard is defensive, keeping a broken install from turning an export into a crash.
    """
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Metric, Param, RunTag
    except ImportError:
        return None
    return MlflowClient, Metric, Param, RunTag


def _experiment_id(client: Any, name: str) -> str:
    """The id of the MLflow experiment ``name``, creating it if it does not exist."""
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    return str(client.create_experiment(name))


def _tags(run: CertifiedRun) -> dict[str, str]:
    """The attestation as MLflow run tags, namespaced to avoid the reserved space.

    These are what make the run legibly *certified* in MLflow's UI and searchable
    (``tags."elbi.verdict" = 'sound'``): the verdict, the reproducibility
    fingerprints, and the gates that ran.
    """
    tags = {
        "elbi.verdict": run.verdict,
        "elbi.derivation_version": run.derivation_version,
        "mlflow.source.name": "elbi",
    }
    if run.data_hash:
        tags["elbi.data_hash"] = run.data_hash
    if run.checks:
        tags["elbi.checks"] = ",".join(name for name, _, _ in run.checks)
    if run.estimate_label:
        tags["elbi.estimate_label"] = run.estimate_label
    return tags


def _config(run: CertifiedRun) -> dict[str, Any]:
    """The run's inputs as MLflow params: its claim roles, params, controls, and deps.

    Namespaced (``claim:x``, ``param:k``) so a reader can tell the input dimensions
    apart; the ``:`` is sanitized to ``.`` by :func:`_key`, which MLflow requires.
    """
    config: dict[str, Any] = {f"claim:{role}": col for role, col in run.claim.items()}
    config.update({f"param:{key}": value for key, value in run.params.items()})
    if run.adjusted_for:
        config["adjusted_for"] = ",".join(run.adjusted_for)
    if run.deps:
        config["deps"] = ",".join(run.deps)
    return config


def _key(key: str) -> str:
    """Sanitize a config/tag key for MLflow, which forbids ``:`` in keys."""
    return key.replace(":", ".")


def _epoch_ms(created_at: str) -> int:
    """Parse an ISO 8601 timestamp to epoch milliseconds for MLflow's time fields."""
    return int(datetime.fromisoformat(created_at).timestamp() * 1000)
