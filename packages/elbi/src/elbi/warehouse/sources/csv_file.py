"""CSV / Parquet file connector: the simplest source, zero external services.

Reads a local path or an object-store URL (``s3://``, ``gs://``) via pyarrow and lands
it as one Delta table. Full-refresh only (a file has no incremental cursor).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry


@SourceRegistry.register
class CsvFileSource(SimpleSource):
    """Load a CSV or Parquet file (local or object-store) as a warehouse table."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "csv"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="csv",
            label="CSV / Parquet file",
            category="File storage",
            icon="📄",
            caption="Upload a CSV or Parquet file, or point at a local path or "
            "object-store URL.",
            fields=[
                SourceField(
                    name="path",
                    label="File path or URL",
                    placeholder="/data/sales.csv  or  s3://bucket/events.parquet",
                    caption="Upload a file, or enter a local path / object-store URL "
                    "(s3://, gs://).",
                    upload=True,
                ),
                SourceField(
                    name="table_name",
                    label="Table name",
                    required=False,
                    placeholder="(derived from the filename)",
                ),
            ],
        )

    def _table_name(self, config: dict[str, Any]) -> str:
        return config.get("table_name") or Path(str(config["path"])).stem

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """The single table this file produces."""
        return [SourceSchema(name=self._table_name(config))]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Read the file into one Arrow table (Parquet or CSV, by extension).

        A CSV is read in the encoding and separator it was actually written in, detected
        from a sample -- so a spreadsheet export (Windows-1252, or semicolon-separated)
        lands as columns rather than one column of garbage. See :mod:`..csv_dialect`.
        """
        path = str(inputs.config["path"])
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            yield pq.read_table(path)
        else:
            from .. import csv_dialect
            from ..storage import open_readable

            with open_readable(path) as source:
                yield csv_dialect.read_table(source)
