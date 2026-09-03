"""Contract checking runs on the shared compute seam.

The validity primitives a data contract is checked in terms of now live in
:mod:`elbi.compute`, alongside profiling and the oracle's data reduction, so one
implementation of each runs over ``list[dict]`` rows or pushes down to DuckDB/Polars.
This module keeps the historical names the contract checker imports; ``CleaningBackend``
is the same protocol as :class:`~elbi.compute.backend.ComputeBackend`.
"""

from __future__ import annotations

from ..compute import (
    Cell,
    ComputeBackend,
    Row,
    RowsFrame,
    backend_for,
    key_set,
    register_backend,
)

#: Historical alias: the contract checker's backend is the shared compute backend.
CleaningBackend = ComputeBackend
#: Historical alias for the ``list[dict]`` reference backend.
RowsBackend = RowsFrame

__all__ = [
    "Cell",
    "CleaningBackend",
    "ComputeBackend",
    "Row",
    "RowsBackend",
    "RowsFrame",
    "backend_for",
    "key_set",
    "register_backend",
]
