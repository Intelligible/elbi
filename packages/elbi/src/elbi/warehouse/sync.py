"""Run a connector's extraction and land it in the Delta Lake warehouse.

Full-refresh overwrites the table; incremental appends rows past the stored cursor and
returns the new high-water mark to persist for next time. Either way the rows land in a
Delta Lake table written via delta-rs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from . import storage
from .sources.base import Source, SourceInputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the table a sync wrote, as the Delta log describes it."""

    name: str
    data_type: str
    nullable: bool
    ordinal: int
    #: The column's comment, when the written schema carried one. Delta keeps it in the
    #: field's metadata, which is where a connector that describes its columns puts it.
    description: str | None = None


@dataclass
class SyncResult:
    """The result of syncing one table: where it landed and how many rows moved."""

    table: str
    rows: int
    sync_type: str
    metadata_location: str | None
    # New incremental high-water mark (max cursor value seen), for the next run.
    new_cursor: Any = None
    #: The table's columns after the write, or None when there is no table to describe.
    #: None means "leave what is recorded alone", which is also what happened to the
    #: Delta table.
    columns: list[ColumnSpec] | None = None


def warehouse_table_name(prefix: str, schema: str) -> str:
    """A safe Delta table name for a synced schema.

    With a ``prefix`` the schema is namespaced under it (``sql__orders``); with an empty
    prefix the schema stands alone (``orders``), so a project-declared source's table
    takes its own name.
    """
    raw = f"{prefix}__{schema}".lower() if prefix else schema.lower()
    return re.sub(r"[^a-z0-9_]", "_", raw)


def run_sync(
    source: Source,
    config: dict[str, Any],
    schema: str,
    *,
    table: str,
    sync_type: str = "full_refresh",
    incremental_field: str | None = None,
    since: Any = None,
    logger: Any = None,
) -> SyncResult:
    """Extract one table from ``source`` and write it to warehouse table ``table``.

    ``table`` is the destination (the schema's persisted table name), so the write
    target matches the catalog row and honors the prefix chosen at source creation.
    """
    incremental = sync_type == "incremental" and incremental_field is not None
    inputs = SourceInputs(
        config=config,
        schema=schema,
        incremental_field=incremental_field if incremental else None,
        incremental_since=since if incremental else None,
        logger=logger,
    )
    mode = "append" if incremental else "overwrite"

    rows = 0
    cursor = since
    location: str | None = None
    first = True
    for batch in source.extract(inputs):
        if batch.num_rows == 0:
            continue
        location = storage.write_arrow(table, batch, mode=(mode if first else "append"))
        rows += batch.num_rows
        first = False
        if (
            incremental
            and incremental_field is not None
            and incremental_field in batch.column_names
        ):
            col_max = _column_max(batch, incremental_field)
            if col_max is not None and (cursor is None or col_max > cursor):
                cursor = col_max

    if first:
        # Extraction produced nothing; leave any existing table untouched.
        location = storage.table_location(table)
    return SyncResult(
        table=table,
        rows=rows,
        sync_type=sync_type,
        metadata_location=location,
        new_cursor=cursor,
        columns=columns_of(location),
    )


def columns_of(location: str | None) -> list[ColumnSpec] | None:
    """The table's columns as the Delta log records them, or None if there is no table.

    Read from the log rather than from the batches just written, because an append
    merges its schema instead of replacing it (:func:`storage.write_arrow`) -- so a
    batch that omits a nullable column would drop it from the catalog while it stays
    perfectly selectable. One rule, right for both write modes.

    The log, not the data files, and once per sync rather than once per query.
    """
    if location is None:
        return None
    from deltalake import DeltaTable

    try:
        schema = DeltaTable(
            location, storage_options=storage.storage_options() or None
        ).schema()
    except Exception:
        # Not worth failing a good write over, but it does leave the catalog stale.
        logger.warning("could not read the schema of %s", location, exc_info=True)
        return None
    return [
        ColumnSpec(
            name=field.name,
            data_type=_type_name(field.type),
            nullable=bool(field.nullable),
            ordinal=index,
            description=_comment(field),
        )
        for index, field in enumerate(schema.fields)
    ]


def _comment(field: Any) -> str | None:
    """A column's comment, which Delta keeps in its field metadata.

    None for every connector shipped today; the path exists so one that describes its
    columns keeps them.
    """
    metadata = getattr(field, "metadata", None) or {}
    comment = metadata.get("comment")
    return str(comment) if comment else None


def _type_name(delta_type: Any) -> str:
    """A Delta type as its written name: ``long``, ``string``, ``array``, ``struct``.

    ``str()`` on one is a Python repr (``PrimitiveType("long")``), which is not what
    belongs in a catalog people search.
    """
    name = getattr(delta_type, "type", None)
    return str(name) if name is not None else str(delta_type)


def _column_max(batch: Any, field: str) -> Any:
    import pyarrow.compute as pc

    values = batch.column(field)
    if values.null_count == len(values):
        return None
    return pc.max(values).as_py()
