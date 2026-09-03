"""Batch scoring: run a registered model over a whole dataset, durably.

The batch analog of the ``/invocations`` endpoint, shaped like the platform a
data scientist is used to: score every row of a dataset with a chosen version,
keep the full predictions as a run artifact (``predictions.csv``, the input
columns plus a ``prediction`` column), and return a summary (distribution stats
and a sample) small enough to read in a chat or a UI. The scoring run lands in
the model's own MLflow experiment, tagged as a batch run, so the record of what
was scored, with which version, over which data is in the same place as the
training history.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ModelError
from ..versioning import hash_json
from .registry import ModelRegistry
from .training import scoped_tracking

#: How many scored rows ride back in the summary (the full set is the artifact).
_SAMPLE_ROWS = 10


@dataclass(frozen=True)
class BatchScoreReport:
    """The durable record of one batch scoring run."""

    name: str
    version: int
    dataset: str
    n_rows: int
    run_id: str
    artifact: str
    stats: dict[str, Any]
    sample: list[dict[str, Any]]

    def render(self) -> str:
        """The report as compact markdown, for the chat and MCP surfaces."""
        stats = ", ".join(f"{k}={v}" for k, v in self.stats.items())
        return (
            f"Scored {self.n_rows} rows of `{self.dataset}` with "
            f"**{self.name}** v{self.version}.\n"
            f"- predictions: {stats}\n"
            f"- full results: `{self.artifact}` on run `{self.run_id}` "
            "(browse or download it from the run's artifacts in MLflow)"
        )


def batch_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    dataset: str,
    tracking_uri: str,
    ref: str | None = None,
    limit: int | None = None,
) -> BatchScoreReport:
    """Score ``rows`` with the model ``ref`` names and record the run.

    ``rows`` may carry more columns than the model's signature (the target, ids);
    the signature's columns are selected and cast exactly as serving would, and
    every input column is preserved next to the ``prediction`` column in the
    artifact so the output joins back to the source data trivially.
    """
    import mlflow

    from .scoring import parse_invocations

    if not rows:
        raise ModelError(f"dataset {dataset!r} has no rows to score")
    if limit is not None:
        rows = list(rows)[: max(0, limit)]
    registry = ModelRegistry(tracking_uri)
    version = registry.resolve(name, ref)
    model = registry.load(name, str(version))
    schema = model.metadata.get_input_schema()
    needed = [c["name"] for c in schema.to_dict()] if schema else []
    missing = sorted(set(needed) - set(rows[0]))
    if missing:
        raise ModelError(
            f"dataset {dataset!r} lacks the model's input columns: "
            + ", ".join(missing)
        )
    records = [{k: r.get(k) for k in needed} for r in rows]
    frame, _ = parse_invocations({"dataframe_records": records}, schema=schema)
    try:
        predictions = model.predict(frame)
    except Exception as exc:
        raise ModelError(f"scoring failed: {exc}") from exc
    values = list(
        predictions.tolist() if hasattr(predictions, "tolist") else predictions
    )

    stats = _prediction_stats(values)
    scored = [dict(r) | {"prediction": v} for r, v in zip(rows, values, strict=True)]
    with scoped_tracking(tracking_uri):
        experiment = mlflow.get_experiment_by_name(name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(name)
        )
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"batch-score-{dataset}",
            tags={
                "elbi.batch_score": "true",
                "elbi.dataset": dataset,
                "elbi.model_version": str(version),
                "elbi.data_hash": hash_json([dict(r) for r in rows]),
            },
        ) as active:
            mlflow.log_params(
                {"dataset": dataset, "model_version": version, "n_rows": len(rows)}
            )
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"prediction_{key}", float(value))
            with tempfile.TemporaryDirectory() as scratch:
                out = Path(scratch) / "predictions.csv"
                _write_csv(out, scored)
                mlflow.log_artifact(str(out))
            run_id = active.info.run_id
    return BatchScoreReport(
        name=name,
        version=version,
        dataset=dataset,
        n_rows=len(rows),
        run_id=run_id,
        artifact="predictions.csv",
        stats=stats,
        sample=scored[:_SAMPLE_ROWS],
    )


def _prediction_stats(values: list[Any]) -> dict[str, Any]:
    """A summary of the predictions: numeric moments, or class counts."""
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            numeric = []
            break
    if numeric and len(set(numeric)) > 10:
        ordered = sorted(numeric)
        return {
            "mean": round(sum(numeric) / len(numeric), 4),
            "min": round(ordered[0], 4),
            "median": round(ordered[len(ordered) // 2], 4),
            "max": round(ordered[-1], 4),
        }
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
    return {f"class {label}": count for label, count in top}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write scored rows as CSV (stdlib; the rows are already plain values)."""
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
