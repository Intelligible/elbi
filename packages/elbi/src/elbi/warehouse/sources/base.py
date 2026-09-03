"""The Source contract every connector implements.

A connector declares its `config` (identity + connection-form schema), the `schemas`
(tables/endpoints) it can sync given a connection, and an `extract` generator that
yields Arrow tables. The sync layer (``warehouse.sync``) drives extraction and writes
the result to the Delta Lake warehouse. Keeping extraction as an Arrow generator makes a
connector engine-agnostic: it can pull with a vendor SDK, SQLAlchemy, or pyarrow, the
warehouse only sees Arrow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceSchema


@dataclass
class SourceInputs:
    """Everything a single-table sync run needs."""

    config: dict[str, Any]  # user connection values, keyed by SourceField.name
    schema: str  # which table/endpoint to sync
    incremental_field: str | None = None
    # The last cursor value already synced (``None`` ⇒ full extract).
    incremental_since: Any = None
    logger: Any = None


class Source(ABC):
    """Base class for all warehouse connectors."""

    # Whether the wizard lets the user pick which columns to sync.
    supports_column_selection: bool = False

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Stable id, e.g. ``"postgres"``: matches ``config.name``."""

    @property
    @abstractmethod
    def config(self) -> SourceConfig:
        """Catalog entry + connection-form schema for the wizard."""

    @abstractmethod
    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """The tables/endpoints available to sync for a given connection."""

    @abstractmethod
    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Yield Arrow tables for ``inputs.schema`` (batched for large sources)."""

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check a connection before saving; returns ``(ok, errors)``."""
        missing = [
            f.name for f in self.config.fields if f.required and not config.get(f.name)
        ]
        if missing:
            return False, [f"Missing required field: {name}" for name in missing]
        return True, []


class SimpleSource(Source, ABC):
    """A source with no resumable or webhook machinery: one extract yields the rows."""
