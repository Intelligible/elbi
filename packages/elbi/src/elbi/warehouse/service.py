"""The warehouse service: configure sources, sync them into Delta Lake, expose them.

Ties the connector :class:`SourceRegistry` to the persistence models and the sync
runner: the catalog and connection wizard read from the registry, a created source and
its selected tables are persisted (:class:`ExternalDataSource` /
:class:`ExternalDataSchema`), and a sync runs each selected table into the Delta Lake
lakehouse and records where it landed. Synced tables are then queryable through DuckDB
(see :meth:`register_duckdb`).

The control plane runs source → schemas → sync, writing Delta Lake via delta-rs and
keeping state in the app's own database.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from elbi_core.errors import ElbiError

from .. import crypto
from ..db import (
    SYNC_INTERVALS,
    ExternalDataSchema,
    ExternalDataSource,
    Store,
    WarehouseColumn,
    _now,
)
from . import storage, sync
from .config import CATEGORIES, SourceSchema
from .sources.base import Source
from .sources.registry import SourceRegistry
from .sync import ColumnSpec

logger = logging.getLogger(__name__)


class WarehouseError(ElbiError):
    """A user-facing warehouse problem (bad connection, missing secret, bad type).

    An ``ElbiError`` like every other SDK error, so generic handlers treat a
    missing or just-deleted warehouse table as a domain error -- skipped in a catalog
    listing, a clean refusal in a draft -- never an unhandled 500.
    """


@dataclass
class SyncOutcome:
    """What a sync run did to one table."""

    schema_id: str
    table: str
    rows: int
    ok: bool
    error: str | None = None


class WarehouseService:
    """Create/sync warehouse sources and surface the tables they produce."""

    def __init__(self, store: Store, runner: Any | None = None) -> None:
        self._store = store
        # Where SQL executes. None means this process, which is what a laptop wants and
        # what a query worker itself uses -- a worker delegating to another worker would
        # recurse forever.
        self._runner = runner

    # --- Catalog (served to the connection wizard) -----------------------------

    def catalog(self) -> dict[str, Any]:
        """The connection wizard's catalog: connector form schemas + the categories.

        The frontend renders this into a category rail, a searchable tile grid, and
        forms generated from each connector's declared fields.
        """
        sources = [src.config.to_dict() for src in SourceRegistry.all().values()]
        sources.sort(key=lambda c: c["label"].lower())
        return {"sources": sources, "categories": list(CATEGORIES)}

    # --- Sources ---------------------------------------------------------------

    def create_source(
        self,
        source_type: str,
        name: str,
        config: dict[str, Any],
        sync_frequency: str = "day",
        description: str = "",
        prefix: str | None = None,
    ) -> ExternalDataSource:
        """Validate a connection, discover its tables, and persist the source.

        Raises :class:`WarehouseError` if the source type is unknown, the name is taken,
        the connection fails validation, or the config carries a secret but no
        ``APP_SECRET_KEY`` is set to encrypt it.
        """
        name = name.strip()
        if not name:
            raise WarehouseError("A source name is required.")
        if self._store.get_external_source_by_name(name) is not None:
            raise WarehouseError(f"A source named {name!r} already exists.")

        connector = self._connector(source_type)
        if connector.config.coming_soon:
            raise WarehouseError(f"{connector.config.label} is not available yet.")
        # Prefix sentinel: None → default to the source type (the connector-UI default);
        # "" → no prefix, so the table lands under the schema's own name (a project-
        # declared source); any other string → that explicit prefix.
        prefix = source_type if prefix is None else _valid_prefix(prefix)
        ok, errors = connector.validate(config)
        if not ok:
            raise WarehouseError("; ".join(errors) or "Could not connect to the source")

        schemas = connector.schemas(config)
        source = ExternalDataSource(
            name=name,
            source_type=source_type,
            config_encrypted=self._encode_config(connector, config),
            sync_frequency=_valid_frequency(sync_frequency),
            description=description.strip(),
            prefix=prefix,
        )
        source_id = source.id
        self._store.save_external_source(source)
        for schema in schemas:
            self._store.save_external_schema(
                ExternalDataSchema(
                    source_id=source_id,
                    name=schema.name,
                    table=sync.warehouse_table_name(prefix, schema.name),
                    should_sync=schema.default_selected,
                    sync_type="full_refresh",
                    incremental_fields=json.dumps(schema.incremental_fields),
                )
            )
        return self.get_source(source_id)

    def source_config_view(self, source_id: str) -> dict[str, Any]:
        """A source's config with every secret removed, for pre-filling an edit form.

        Password fields come back as empty strings rather than their values or a mask.
        Empty is what the edit path already reads as "unchanged", so a form can
        round-trip the config it was handed without the operator re-entering a key,
        and without the key ever leaving the server.
        """
        source = self.get_source(source_id)
        connector = self._connector(source.source_type)
        secret_fields = {f.name for f in connector.config.fields if f.is_secret}
        config = self._decode_config(source)
        return {
            "source_type": source.source_type,
            "config": {
                key: ("" if key in secret_fields else value)
                for key, value in config.items()
            },
            "secret_fields": sorted(secret_fields & set(config)),
        }

    def update_source(
        self,
        source_id: str,
        *,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        sync_frequency: str | None = None,
    ) -> ExternalDataSource:
        """Edit an existing source in place, re-validating before anything is saved.

        Without this a source can only be deleted and rebuilt, which means re-entering
        every secret to correct a typo in a manifest — so a small fix is paid for in
        credentials, and the tables the source produced are destroyed on the way.

        Secrets are kept unless replaced. A password field left blank means "leave it
        alone", which is how every connection form behaves and the only way editing a
        manifest is workable: the operator is changing a query, not a key.
        """
        source = self.get_source(source_id)
        connector = self._connector(source.source_type)
        fields: dict[str, Any] = {}

        if name is not None:
            name = name.strip()
            if not name:
                raise WarehouseError("A source name is required.")
            existing = self._store.get_external_source_by_name(name)
            if existing is not None and existing.id != source_id:
                raise WarehouseError(f"A source named {name!r} already exists.")
            fields["name"] = name

        if config is not None:
            merged = self._merge_config(connector, self._decode_config(source), config)
            ok, errors = connector.validate(merged)
            if not ok:
                raise WarehouseError(
                    "; ".join(errors) or "Could not connect to the source"
                )
            fields["config_encrypted"] = self._encode_config(connector, merged)

        if description is not None:
            fields["description"] = description.strip()
        if sync_frequency is not None:
            fields["sync_frequency"] = _valid_frequency(sync_frequency)

        # Persist before touching schemas: a rejected config leaves both untouched, and
        # reconciliation is only right once the new config is the stored one.
        self._store.update_external_source(source_id, **fields)
        if config is not None:
            self._reconcile_schemas(source, connector.schemas(merged))
        return self.get_source(source_id)

    @staticmethod
    def _merge_config(
        connector: Source, stored: dict[str, Any], incoming: dict[str, Any]
    ) -> dict[str, Any]:
        """Overlay an edit on the stored config, keeping secrets the edit left blank."""
        fields = connector.config.fields
        secret_fields = {f.name for f in fields if f.is_secret}
        merged = dict(stored)
        for key, value in incoming.items():
            if key in secret_fields and not value:
                continue  # blank means "unchanged", not "erase"
            merged[key] = value
        return merged

    def _reconcile_schemas(
        self, source: ExternalDataSource, schemas: list[SourceSchema]
    ) -> None:
        """Bring a source's tables in line with its new config.

        An edited manifest can rename, add or drop resources. Tables that survive keep
        whether they were enabled: re-ticking everything after a one-word fix to a query
        would be its own annoyance.

        A resource that disappears is **disabled, not deleted**. Its warehouse table
        holds rows an edit did not ask to destroy, and a resource dropped by a typo
        comes back when the typo is fixed. Deleting the source remains the way to
        remove data, and it still says so.
        """
        existing = {s.name: s for s in self._store.list_external_schemas(source.id)}
        incoming = {schema.name: schema for schema in schemas}

        for name, schema in incoming.items():
            if (current := existing.get(name)) is not None:
                self._store.update_external_schema(
                    current.id,
                    incremental_fields=json.dumps(schema.incremental_fields),
                )
                continue
            self._store.save_external_schema(
                ExternalDataSchema(
                    source_id=source.id,
                    name=name,
                    table=sync.warehouse_table_name(source.prefix, name),
                    should_sync=schema.default_selected,
                    sync_type="full_refresh",
                    incremental_fields=json.dumps(schema.incremental_fields),
                )
            )

        for name, stored_schema in existing.items():
            if name not in incoming and stored_schema.should_sync:
                self._store.update_external_schema(stored_schema.id, should_sync=False)

    def register_interest(self, source_type: str) -> None:
        """Record a notify-me for a coming-soon connector (audit trail of demand)."""
        self._store.record_audit(
            "warehouse.source_interest",
            target_type="warehouse_source",
            target_id=source_type,
        )

    def list_sources(self) -> list[ExternalDataSource]:
        """Every configured source, newest first."""
        return self._store.list_external_sources()

    def get_source(self, source_id: str) -> ExternalDataSource:
        """The source of this id, raising :class:`WarehouseError` if it's gone."""
        source = self._store.get_external_source(source_id)
        if source is None:
            raise WarehouseError("Source not found.")
        return source

    def delete_source(self, source_id: str) -> bool:
        """Delete a source, its schemas, and the warehouse tables they produced."""
        schemas = self._store.list_external_schemas(source_id)
        deleted = self._store.delete_external_source(source_id)
        if deleted:
            for schema in schemas:
                storage.drop_table(schema.table)
        return deleted

    def set_sync_frequency(self, source_id: str, frequency: str) -> dict[str, Any]:
        """Change how often a source auto-syncs; return its refreshed detail view."""
        self.get_source(source_id)  # 404 if missing
        self._store.update_external_source(
            source_id, sync_frequency=_valid_frequency(frequency)
        )
        return self.source_detail_view(source_id)

    def list_schemas(self, source_id: str) -> list[ExternalDataSchema]:
        """Every table/endpoint belonging to a source."""
        return self._store.list_external_schemas(source_id)

    def update_schema(
        self,
        schema_id: str,
        should_sync: bool | None = None,
        sync_type: str | None = None,
        incremental_field: str | None = None,
    ) -> dict[str, Any]:
        """Toggle a table on/off or change how it syncs; return the updated view."""
        schema = self._store.get_external_schema(schema_id)
        if schema is None:
            raise WarehouseError("Schema not found.")
        fields: dict[str, Any] = {}
        if should_sync is not None:
            fields["should_sync"] = should_sync
        if sync_type is not None:
            if sync_type not in ("full_refresh", "incremental"):
                raise WarehouseError(f"Unknown sync type {sync_type!r}.")
            fields["sync_type"] = sync_type
        if incremental_field is not None:
            fields["incremental_field"] = incremental_field or None
        self._store.update_external_schema(schema_id, **fields)
        updated = self._store.get_external_schema(schema_id)
        if updated is None:
            raise WarehouseError("Schema not found.")
        return _schema_dict(updated)

    # --- Sync ------------------------------------------------------------------

    def sync_source(self, source_id: str) -> list[SyncOutcome]:
        """Sync every enabled table of a source into the warehouse."""
        source = self.get_source(source_id)
        connector = self._connector(source.source_type)
        config = self._decode_config(source)
        schema_ids = [
            s.id for s in self._store.list_external_schemas(source_id) if s.should_sync
        ]

        self._mark_source(source_id, status="syncing", last_error=None)
        outcomes = [self._sync_schema(connector, config, sid) for sid in schema_ids]
        had_error = any(not o.ok for o in outcomes)
        self._mark_source(
            source_id,
            status="error" if had_error else "idle",
            last_error=next((o.error for o in outcomes if o.error), None),
            synced=not had_error,
        )
        return outcomes

    def sync_due(self, now: datetime) -> list[str]:
        """Sync every source whose cadence has elapsed; return the source ids synced.

        Called by the maintenance scheduler. Each source is isolated: one that fails
        logs and records its error without stopping the others.
        """
        synced: list[str] = []
        for source in self._store.due_external_sources(now):
            try:
                self.sync_source(source.id)
                synced.append(source.id)
            except Exception:
                logger.exception("scheduled warehouse sync failed for %s", source.id)
        return synced

    def _sync_schema(
        self, connector: Source, config: dict[str, Any], schema_id: str
    ) -> SyncOutcome:
        # Re-fetch fresh so this is the single mutate-and-save of the instance.
        schema = self._store.get_external_schema(schema_id)
        if schema is None:
            return SyncOutcome(schema_id, "", 0, ok=False, error="Schema disappeared.")
        try:
            result = sync.run_sync(
                connector,
                config,
                schema.name,
                table=schema.table,
                sync_type=schema.sync_type,
                incremental_field=schema.incremental_field,
                since=_parse_cursor(schema.cursor),
                logger=logger,
            )
        except Exception as exc:  # surface on the schema, keep the other tables going
            logger.exception("Warehouse sync failed for %s", schema.name)
            self._store.update_external_schema(
                schema_id, status="error", last_error=str(exc)
            )
            return SyncOutcome(schema_id, schema.table, 0, ok=False, error=str(exc))

        fields: dict[str, Any] = {
            "status": "synced",
            "last_error": None,
            "last_synced_at": _now(),
        }
        if schema.sync_type == "incremental":
            fields["row_count"] = (schema.row_count or 0) + result.rows
            if result.new_cursor is not None:
                fields["cursor"] = _dump_cursor(result.new_cursor)
        else:
            fields["row_count"] = result.rows
        self._store.update_external_schema(schema_id, **fields)
        self._record_columns(schema, result.columns)
        return SyncOutcome(schema_id, schema.table, result.rows, ok=True)

    def _record_columns(
        self, schema: ExternalDataSchema, columns: list[ColumnSpec] | None
    ) -> bool:
        """Persist the table's columns, so a column search never opens a Delta table.

        Every warehouse table is written through this path -- connector sync,
        project-declared source, uploaded CSV -- so this covers them all. ``None`` means
        there was no table to describe, and what is recorded stays as it is.

        Returns whether this call is the one that wrote them; another process racing to
        record the same table wins instead, and its set describes the same log.
        """
        if columns is None:
            return False
        return self._store.replace_warehouse_columns(
            schema.table,
            schema.source_id,
            [
                WarehouseColumn(
                    table=schema.table,
                    source_id=schema.source_id,
                    name=column.name,
                    data_type=column.data_type,
                    nullable=column.nullable,
                    ordinal=column.ordinal,
                    description=column.description,
                )
                for column in columns
            ],
        )

    def backfill_columns(self) -> list[str]:
        """Record columns for tables synced before there was anywhere to record them.

        Sync-time capture only helps a table that syncs again, and one synced earlier
        already reads as ``synced``: an unchanged declaration is skipped at boot, and a
        ``manual`` source may never run again. Without this an existing project's
        catalog stays empty, and a search that finds nothing reads as a bad feature
        rather than an unpopulated one.

        Reads one transaction log per table still missing columns, so it costs nothing
        once they all have them and needs no flag to run once.
        """
        known = {column.table for column in self._store.list_warehouse_columns()}
        filled: list[str] = []
        for schema in self._store.list_all_external_schemas():
            if schema.status != "synced" or schema.table in known:
                continue
            columns = sync.columns_of(storage.table_location(schema.table))
            if not columns:
                continue
            if self._record_columns(schema, columns):
                filled.append(schema.table)
        return filled

    def _mark_source(
        self,
        source_id: str,
        status: str,
        last_error: str | None = None,
        synced: bool = False,
    ) -> None:
        """Set a source's run state in place."""
        fields: dict[str, Any] = {"status": status, "last_error": last_error}
        if synced:
            fields["last_synced_at"] = _now()
        self._store.update_external_source(source_id, **fields)

    # --- Query surface ---------------------------------------------------------

    def tables(self) -> list[dict[str, Any]]:
        """Every synced warehouse table."""
        return [
            {
                "table": schema.table,
                "source": source.name if source else schema.source_id,
                "source_type": source.source_type if source else None,
                "rows": schema.row_count,
                "location": storage.table_location(schema.table),
            }
            for schema, source in self._reachable()
        ]

    def _reachable(
        self,
    ) -> list[tuple[ExternalDataSchema, ExternalDataSource | None]]:
        """Synced schemas by table name.

        The database-only half of :meth:`tables`, shared with :meth:`columns`. Touches
        no storage, because resolving a location is a filesystem probe per table that a
        column read must not pay.
        """
        by_source = {s.id: s for s in self._store.list_external_sources()}
        out = [
            (schema, by_source.get(schema.source_id))
            for schema in self._store.list_all_external_schemas()
            if schema.status == "synced"
        ]
        out.sort(key=lambda item: item[0].table)
        return out

    def columns(self) -> list[dict[str, Any]]:
        """Every column of every synced table, without opening one.

        Answers "which datasets have a ``customer_id`` column" from the database alone,
        cheap enough to sit behind a search box.

        Reachability comes from :meth:`_reachable`, shared with :meth:`tables` so the
        two cannot drift -- but not from ``tables`` itself, which resolves each table's
        location and would put a filesystem probe per table back on this path. Defaults
        to ``BROWSE`` for the reason ``tables`` does: knowing a column exists is how
        someone knows to ask for the data.
        """
        reachable = {
            schema.table: (source.name if source else schema.source_id)
            for schema, source in self._reachable()
        }
        return [
            {
                "table": column.table,
                "name": column.name,
                "type": column.data_type,
                "nullable": column.nullable,
                "ordinal": column.ordinal,
                "description": column.description,
                "source": reachable[column.table],
            }
            for column in self._store.list_warehouse_columns(list(reachable))
        ]

    # --- Serialized views for the API (no secrets) -----------------------------

    def list_sources_view(self) -> list[dict[str, Any]]:
        """Every source as a JSON-safe dict with its table counts (never the config)."""
        out = []
        for source in self._store.list_external_sources():
            schemas = self._store.list_external_schemas(source.id)
            out.append(_source_dict(source, schemas))
        return out

    def source_detail_view(self, source_id: str) -> dict[str, Any]:
        """A source plus its schemas, JSON-safe (for the source detail page)."""
        source = self.get_source(source_id)
        schemas = self._store.list_external_schemas(source_id)
        view = _source_dict(source, schemas)
        view["schemas"] = [_schema_dict(s) for s in schemas]
        return view

    def set_runner(self, runner: Any | None) -> None:
        """Choose where SQL executes: a runner, or this process when None.

        Set after construction because a runner needs :meth:`_query_here` to call, and
        that only exists once the service does.
        """
        self._runner = runner

    def query(
        self, sql: str, max_rows: int = 1000
    ) -> tuple[list[str], list[dict[str, Any]], bool]:
        """Run read-only SQL over the warehouse tables.

        Only their readable tables are bound as views, so the SQL is authorized by what
        it can resolve rather than by inspecting it. Fetches one past ``max_rows`` to
        flag a clipped result. Raises :class:`WarehouseError` on a query error.

        With a runner configured the work happens on compute of its own rather than in
        the web server: DuckDB is in-process, so a query that exhausts memory here would
        take the whole service with it. The caller travels with the request and the
        runner binds its own views, so that happens where the views are made.
        """
        if self._runner is not None:
            from ..query_runner import QueryError

            try:
                return self._runner.run(  # type: ignore[no-any-return]
                    sql, max_rows=max_rows
                )
            except QueryError as exc:
                raise WarehouseError(str(exc)) from exc
        return self._query_here(sql, max_rows=max_rows)

    def _query_here(
        self, sql: str, max_rows: int = 1000
    ) -> tuple[list[str], list[dict[str, Any]], bool]:
        """Execute in this process. The runner's implementation, and the laptop path."""
        import duckdb

        con = duckdb.connect()
        bound = set(self.register_duckdb(con))
        try:
            cursor = con.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchmany(max_rows + 1)
        except duckdb.Error as exc:
            raise WarehouseError(self._explain_query_error(exc, bound)) from exc
        truncated = len(fetched) > max_rows
        rows = [dict(zip(columns, row, strict=True)) for row in fetched[:max_rows]]
        return columns, rows, truncated

    def _explain_query_error(self, exc: Exception, bound: set[str]) -> str:
        """A query error, saying so when the real cause is an unbound table.

        A table in the catalog whose data has not landed yet is not registered, so a
        reference to it arrives as "table does not exist" -- which sends someone hunting
        for a typo in a name that is spelled correctly. Name it instead.
        """
        message = str(exc)
        # DuckDB quotes the unresolved name in its binder error; find one that is a
        # catalogued table nothing bound.
        withheld = sorted(
            {
                table["table"]
                for table in self.tables()
                if table["table"] not in bound and table["table"] in message
            }
        )
        if not withheld:
            return message
        named = ", ".join(repr(name) for name in withheld)
        return (
            f"{message}\n\nYou do not have read access to {named}. "
            "It is listed in the catalog because you can see that it exists; "
            "request access to read it."
        )

    def read_any(
        self, name: str, max_rows: int
    ) -> tuple[list[str], list[dict[str, Any]], bool]:
        """Read a Delta table by name (the single runtime read path).

        Resolves the table by its own location and registers just that one Delta
        directory into a fresh DuckDB connection, so any warehouse table (connector or
        project-declared) is read by name without registering the whole catalog. Fetches
        one past ``max_rows`` to flag a clipped result. Raises :class:`WarehouseError`
        if the table does not exist or cannot be read.
        """
        import duckdb
        from deltalake import DeltaTable

        location = storage.table_location(name)
        if location is None:
            raise WarehouseError(f"warehouse table {name!r} not found")
        con = duckdb.connect()
        options = storage.storage_options() or None
        dataset = DeltaTable(location, storage_options=options).to_pyarrow_dataset()
        con.register(name, dataset)
        try:
            relation = con.table(name).limit(max_rows + 1)
            columns = list(relation.columns)
            fetched = relation.fetchall()
        except duckdb.Error as exc:
            raise WarehouseError(str(exc)) from exc
        truncated = len(fetched) > max_rows
        rows = [dict(zip(columns, row, strict=True)) for row in fetched[:max_rows]]
        return columns, rows, truncated

    def read_table(
        self, name: str, max_rows: int
    ) -> tuple[list[str], list[dict[str, Any]], bool]:
        """Read one *catalogued* synced table's rows by name; columns, rows, truncated.

        The connector-facing API entry: it guards on catalog membership (so the
        warehouse tables UI reads only registered sources), then delegates the read to
        :meth:`read_any`. Raises :class:`WarehouseError` if the table is not a synced
        catalog table.
        """
        if name not in {t["table"] for t in self.tables()}:
            raise WarehouseError(f"warehouse table {name!r} not found")
        return self.read_any(name, max_rows=max_rows)

    def table_shape(self, name: str) -> tuple[list[str], int]:
        """A table's column names and row count, for the Explore schema browser."""
        import duckdb
        from deltalake import DeltaTable

        location = storage.table_location(name)
        if location is None:
            raise WarehouseError(f"warehouse table {name!r} not found")
        con = duckdb.connect()
        options = storage.storage_options() or None
        dataset = DeltaTable(location, storage_options=options).to_pyarrow_dataset()
        con.register(name, dataset)
        relation = con.table(name)
        columns = list(relation.columns)
        # Relational API (not an interpolated SQL string) so the identifier is never
        # spliced into a query.
        count = relation.aggregate("count() AS n").fetchone()
        return columns, int(count[0]) if count else 0

    def duckdb_catalog(self) -> dict[str, Any]:
        """The readable warehouse tables and their columns, for the schema browser."""
        import duckdb

        con = duckdb.connect()
        names = self.register_duckdb(con)
        row_counts = {t["table"]: t["rows"] for t in self.tables()}
        tables = []
        for name in names:
            described = con.execute(f'DESCRIBE "{name}"').fetchall()
            columns = [{"name": d[0], "type": d[1]} for d in described]
            tables.append(
                {"name": name, "columns": columns, "rows": row_counts.get(name)}
            )
        return {"tables": tables}

    def register_duckdb(self, con: Any) -> list[str]:
        """Register the warehouse tables as DuckDB views; return their names.

        Each Delta table is registered directly as an Arrow dataset (read through its
        transaction log), so a query engine wired with these views joins warehouse data
        like a local dataset: no SQL-string building and no DuckDB extension needed.

        Registering by name rather than parsing the SQL for table names: matching
        DuckDB's own resolution -- through CTEs, aliases and subqueries -- is a job
        nothing here should be doing.
        """
        from deltalake import DeltaTable

        options = storage.storage_options() or None
        names: list[str] = []
        for table in self.tables():
            location = table["location"]
            if not location:
                continue
            name = table["table"]
            dataset = DeltaTable(location, storage_options=options).to_pyarrow_dataset()
            con.register(name, dataset)
            names.append(name)
        return names

    def _register_one(self, con: Any, table: str) -> None:
        """Bind one Delta table, unguarded, for a policy check or a direct read."""
        from deltalake import DeltaTable

        location = storage.table_location(table)
        if location is None:
            raise WarehouseError(f"warehouse table {table!r} not found")
        options = storage.storage_options() or None
        con.register(
            table, DeltaTable(location, storage_options=options).to_pyarrow_dataset()
        )

    # --- Internals -------------------------------------------------------------

    def _connector(self, source_type: str) -> Source:
        if not SourceRegistry.is_registered(source_type):
            raise WarehouseError(f"Unknown source type {source_type!r}.")
        return SourceRegistry.get(source_type)

    def _encode_config(self, connector: Source, config: dict[str, Any]) -> str:
        """Serialize a config, encrypting when an app secret is set.

        Refuses to store a secret-bearing config in plaintext: if any password field
        has a value and ``APP_SECRET_KEY`` is unset, the operator must set it first.
        """
        has_secret = any(
            f.is_secret and config.get(f.name) for f in connector.config.fields
        )
        payload = json.dumps(config)
        if crypto.is_configured():
            return "enc:" + crypto.encrypt(payload)
        if has_secret:
            raise WarehouseError(
                "This source has a credential; set APP_SECRET_KEY to encrypt it."
            )
        return "plain:" + payload

    def _decode_config(self, source: ExternalDataSource) -> dict[str, Any]:
        token = source.config_encrypted
        if token.startswith("enc:"):
            raw = crypto.decrypt(token[len("enc:") :])
        elif token.startswith("plain:"):
            raw = token[len("plain:") :]
        else:
            raw = token  # legacy/unprefixed: treat as plaintext JSON
        config: dict[str, Any] = json.loads(raw)
        return config


_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _valid_prefix(prefix: str) -> str:
    """Normalize a table prefix; empty is allowed (caller defaults to source type)."""
    prefix = prefix.strip()
    if prefix and not _PREFIX_RE.match(prefix):
        raise WarehouseError(
            "Table prefix must use only letters, numbers, and underscores, and start "
            "with a letter or underscore."
        )
    return prefix


def _valid_frequency(frequency: str) -> str:
    """Return ``frequency`` if it is a known cadence (or ``"manual"``), else raise."""
    if frequency == "manual" or frequency in SYNC_INTERVALS:
        return frequency
    allowed = ", ".join(["manual", *SYNC_INTERVALS])
    raise WarehouseError(f"Unknown sync frequency {frequency!r}; use: {allowed}.")


def warehouse_available() -> bool:
    """Whether the ``warehouse`` extra (delta-rs) is installed."""
    import importlib.util

    return importlib.util.find_spec("deltalake") is not None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _source_dict(
    source: ExternalDataSource, schemas: list[ExternalDataSchema]
) -> dict[str, Any]:
    return {
        "id": source.id,
        "name": source.name,
        "source_type": source.source_type,
        "description": source.description,
        "prefix": source.prefix,
        "sync_frequency": source.sync_frequency,
        "status": source.status,
        "last_error": source.last_error,
        "last_synced_at": _iso(source.last_synced_at),
        "created_at": _iso(source.created_at),
        "schema_count": len(schemas),
        "synced_count": sum(1 for s in schemas if s.status == "synced"),
        "rows": sum(s.row_count or 0 for s in schemas),
    }


def _schema_dict(schema: ExternalDataSchema) -> dict[str, Any]:
    return {
        "id": schema.id,
        "name": schema.name,
        "table": schema.table,
        "should_sync": schema.should_sync,
        "sync_type": schema.sync_type,
        "incremental_field": schema.incremental_field,
        "incremental_fields": json.loads(schema.incremental_fields or "[]"),
        "status": schema.status,
        "row_count": schema.row_count,
        "last_error": schema.last_error,
        "last_synced_at": _iso(schema.last_synced_at),
    }


def _dump_cursor(value: Any) -> str:
    """Stringify an incremental high-water mark for storage."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_cursor(raw: str | None) -> Any:
    """Reconstruct a stored cursor to its usable type (int, float, datetime, or str)."""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return raw
