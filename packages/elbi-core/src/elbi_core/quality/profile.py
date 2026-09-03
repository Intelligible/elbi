"""Column profiling, computed through the shared compute seam.

The per-column summary a contract suggestion is built from. The statistics are
produced by a :class:`~elbi.compute.backend.ComputeBackend`, so profiling a
``list[dict]`` runs in pure Python while profiling a DuckDB or Polars handle pushes the
single-pass aggregate down to the engine, over a larger-than-memory file if need be. The
result is exact rather than sketch-based: determinism matters more than the memory an
approximate cardinality would save.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compute import ColumnStats, ComputeBackend, backend_for


@dataclass(frozen=True)
class ColumnProfile:
    """A one-column summary used to propose (never to certify) a contract."""

    name: str
    count: int  # total rows
    present: int  # non-missing values
    distinct: int  # distinct present values
    inferred_type: str
    minimum: float | None
    maximum: float | None
    #: (value, frequency) for the most common present values, most frequent first.
    top_values: tuple[tuple[Any, int], ...]
    #: distinct present values appearing exactly once, over distinct: high for an
    #: identifier, low for a categorical column.
    fraction_unique_once: float

    @property
    def completeness(self) -> float:
        """Fraction of rows whose value is present."""
        return self.present / self.count if self.count else 0.0

    @property
    def is_unique(self) -> bool:
        """Whether every present value is distinct."""
        return self.present > 0 and self.distinct == self.present

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable summary for an agent to read."""
        return {
            "name": self.name,
            "count": self.count,
            "present": self.present,
            "completeness": round(self.completeness, 4),
            "distinct": self.distinct,
            "inferredType": self.inferred_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "topValues": [
                {"value": value, "count": count} for value, count in self.top_values
            ],
        }

    @classmethod
    def from_stats(cls, stats: ColumnStats) -> ColumnProfile:
        """Build a profile from a backend's column statistics."""
        return cls(
            name=stats.name,
            count=stats.count,
            present=stats.present,
            distinct=stats.distinct,
            inferred_type=stats.inferred_type,
            minimum=stats.minimum,
            maximum=stats.maximum,
            top_values=stats.top_values,
            fraction_unique_once=stats.fraction_unique_once,
        )


def profile_columns(data: Any) -> tuple[ColumnProfile, ...]:
    """Profile every column of ``data`` in order.

    ``data`` is anything a compute backend handles (``list[dict]`` rows, or an engine
    handle), or a backend itself.
    """
    backend = _as_backend(data)
    columns = backend.columns()
    stats = backend.profile(columns)
    return tuple(ColumnProfile.from_stats(stats[column]) for column in columns)


def profile_column(data: Any, column: str) -> ColumnProfile:
    """Profile a single column of ``data``."""
    backend = _as_backend(data)
    return ColumnProfile.from_stats(backend.profile([column])[column])


def _as_backend(data: Any) -> ComputeBackend:
    return data if isinstance(data, ComputeBackend) else backend_for(data)
