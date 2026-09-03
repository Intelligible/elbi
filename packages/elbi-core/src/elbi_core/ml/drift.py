"""Data-drift analysis between a model's training data and its live traffic.

Evidently is the de facto open-source standard for drift reports, so the
statistics are its (Kolmogorov-Smirnov and friends via the data-drift preset);
this module turns one comparison into a typed summary the platform's surfaces
share, plus Evidently's own HTML report for humans. The caller supplies the two
sides: the reference sample logged at training time, and the recent feature rows
from the inference table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ModelError

#: Below this many current rows a drift test is noise, not signal.
MIN_DRIFT_ROWS = 30


@dataclass(frozen=True)
class ColumnDrift:
    """One column's drift test: the method, its score, and the call."""

    column: str
    method: str
    score: float
    threshold: float
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    """The outcome of one reference-vs-current comparison."""

    n_columns: int
    n_drifted: int
    share_drifted: float
    dataset_drift: bool
    columns: tuple[ColumnDrift, ...]
    html: str

    def summary(self) -> dict[str, Any]:
        """The JSON-safe summary (everything but the HTML report)."""
        return {
            "n_columns": self.n_columns,
            "n_drifted": self.n_drifted,
            "share_drifted": round(self.share_drifted, 4),
            "dataset_drift": self.dataset_drift,
            "columns": [
                {
                    "column": c.column,
                    "method": c.method,
                    "score": round(c.score, 6),
                    "threshold": c.threshold,
                    "drifted": c.drifted,
                }
                for c in self.columns
            ],
        }


def data_drift(
    reference: Any, current: Any, columns: list[str] | None = None
) -> DriftReport:
    """Compare ``current`` rows against ``reference`` rows column by column.

    Both sides are records (or DataFrames); ``columns`` restricts the comparison
    (the model's input signature, in practice). Dataset-level drift follows the
    preset's rule: at least half the columns drifted.
    """
    try:
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise ModelError(
            "drift monitoring requires the 'ml' extra: pip install 'elbi[ml]'"
        ) from exc

    reference_df = (
        reference
        if isinstance(reference, pd.DataFrame)
        else pd.DataFrame.from_records(list(reference))
    )
    current_df = (
        current
        if isinstance(current, pd.DataFrame)
        else pd.DataFrame.from_records(list(current))
    )
    if columns:
        shared = [
            c for c in columns if c in reference_df.columns and c in current_df.columns
        ]
        if not shared:
            raise ModelError("no shared columns between reference and current data")
        reference_df = reference_df[shared]
        current_df = current_df[shared]
    if len(current_df) < MIN_DRIFT_ROWS:
        raise ModelError(
            f"drift needs at least {MIN_DRIFT_ROWS} current rows "
            f"(have {len(current_df)}); serve more traffic first"
        )

    snapshot = Report([DataDriftPreset()]).run(
        reference_data=reference_df, current_data=current_df
    )
    return _parse_snapshot(snapshot)


def _parse_snapshot(snapshot: Any) -> DriftReport:
    """Evidently's snapshot as the platform's typed report."""
    payload = snapshot.dict()
    per_column: list[ColumnDrift] = []
    share = 0.0
    count = 0
    for metric in payload.get("metrics", []):
        config = metric.get("config") or {}
        kind = str(config.get("type", ""))
        if kind.endswith("DriftedColumnsCount"):
            value = metric.get("value") or {}
            share = float(value.get("share") or 0.0)
            count = int(value.get("count") or 0)
        elif kind.endswith("ValueDrift"):
            method = str(config.get("method") or "")
            threshold = float(config.get("threshold") or 0.05)
            score = float(metric.get("value"))
            # p-value tests drift when small; distance metrics when large.
            drifted = score < threshold if "p_value" in method else score > threshold
            per_column.append(
                ColumnDrift(
                    column=str(config.get("column")),
                    method=method,
                    score=score,
                    threshold=threshold,
                    drifted=drifted,
                )
            )
    html = ""
    get_html = getattr(snapshot, "get_html_str", None)
    if callable(get_html):
        html = get_html(as_iframe=False)
    return DriftReport(
        n_columns=len(per_column),
        n_drifted=count,
        share_drifted=share,
        # The preset's own rule: dataset drift once half the columns moved.
        dataset_drift=share >= 0.5,
        columns=tuple(sorted(per_column, key=lambda c: c.column)),
        html=html,
    )
