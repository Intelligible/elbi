"""Lightweight tabular data loading for the local runner.

The SDK stays dependency-light: CSV, JSON, and JSONL are read with the standard
library. Parquet support is optional and requires the ``data`` extra
(``pip install "elbi[data]"``), which pulls in ``pyarrow``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataBindingError

Row = dict[str, Any]


@dataclass(frozen=True)
class Table:
    """An in-memory table: an ordered list of row mappings.

    This is intentionally minimal. Compute functions operate on ``rows`` (a list
    of dicts), which keeps the SDK free of a heavyweight dataframe dependency.
    """

    rows: list[Row]

    @property
    def columns(self) -> list[str]:
        """Column names in first-seen order across all rows."""
        seen: dict[str, None] = {}
        for row in self.rows:
            for key in row:
                seen.setdefault(key, None)
        return list(seen)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Any:
        return iter(self.rows)


def load_table(path: Path) -> Table:
    """Load a :class:`Table` from a file, dispatching on its suffix.

    Supported: ``.csv``, ``.json``, ``.jsonl``/``.ndjson``, and ``.parquet``
    (with the ``data`` extra).

    Raises:
        DataBindingError: if the file is missing or the format is unsupported.
    """
    if not path.exists():
        raise DataBindingError(f"data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(path)
    if suffix == ".parquet":
        return _load_parquet(path)
    raise DataBindingError(
        f"unsupported data format {suffix!r} for {path}; "
        "supported: .csv, .json, .jsonl, .parquet"
    )


def _load_csv(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[Row] = [dict(record) for record in reader]
    return Table(rows=rows)


def _load_json(path: Path) -> Table:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return Table(rows=[_as_row(item) for item in payload])
    if isinstance(payload, dict):
        return Table(rows=[payload])
    raise DataBindingError(
        f"{path}: JSON data must be an object or an array of objects"
    )


def _load_jsonl(path: Path) -> Table:
    rows: list[Row] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(_as_row(json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise DataBindingError(f"{path}:{line_number}: invalid JSON line") from exc
    return Table(rows=rows)


def _load_parquet(path: Path) -> Table:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataBindingError(
            "reading .parquet requires the 'data' extra: pip install \"elbi[data]\""
        ) from exc
    table = pq.read_table(path)
    records: list[Row] = table.to_pylist()
    return Table(rows=records)


def _as_row(item: Any) -> Row:
    if not isinstance(item, dict):
        raise DataBindingError(f"expected an object, got {type(item).__name__}")
    return dict(item)
