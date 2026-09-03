"""The engine-agnostic compute seam.

One fixed catalog of operations (structure, per-column profile, located validity
primitives, bounded materialization) expressed as the
:class:`~elbi.compute.backend.ComputeBackend` protocol, and selected for a given
data representation by :func:`~elbi.compute.backend.backend_for`. The
``list[dict]`` reference backend ships dependency-free; DuckDB and Polars backends push
the same operations down to their engine (out-of-core, larger than memory) and are
optional extras. This is the seam the profiler, the contract checker, and the
verification oracle run over, so each scales from in-memory rows to
a warehouse-sized file without changing.
"""

from .backend import (
    Cell,
    ColumnStats,
    ComputeBackend,
    Row,
    backend_for,
    register_backend,
)
from .rows import RowsFrame, key_set

__all__ = [
    "Cell",
    "ColumnStats",
    "ComputeBackend",
    "Row",
    "RowsFrame",
    "backend_for",
    "key_set",
    "register_backend",
]
