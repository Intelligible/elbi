"""Arrow interchange helpers, kept strictly behind the engine seam.

Arrow is the zero-copy lingua franca the engine backends share: a ``list[dict]`` becomes
an Arrow table once at ingress, engines hand data along as Arrow without copying,
and results cross back to ``list[dict]`` only when a consumer needs Python scalars. The
core's public contract stays ``list[dict]``; none appears in it. ``pyarrow`` is an
optional dependency (the ``data`` extra), imported lazily with a graceful hint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .backend import Row


def require_pyarrow() -> Any:
    """Import and return ``pyarrow``, or raise a clear install hint."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            'the compute engine tier needs pyarrow: pip install "elbi[data]"'
        ) from exc
    return pa


def rows_to_arrow(rows: Sequence[Row]) -> Any:
    """Convert ``list[dict]`` rows to an Arrow table (the bridge into an engine).

    A genuine copy from Python objects into columnar buffers, done once when entering a
    pushdown engine. The union of keys across rows becomes the schema.
    """
    pa = require_pyarrow()
    return pa.Table.from_pylist([dict(row) for row in rows])


def is_arrow_source(obj: Any) -> bool:
    """Whether ``obj`` advertises the Arrow PyCapsule stream protocol.

    Any object implementing ``__arrow_c_stream__`` (a Polars/pandas/DuckDB frame, a
    PyArrow table) can be consumed zero-copy without importing its library.
    """
    return hasattr(obj, "__arrow_c_stream__")
