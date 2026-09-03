"""The exploration surface: ad-hoc SQL and one-click profiling over bound data.

A person queries the project's bound datasets (through DuckDB) or a registered external
data source (through the runtime's read-only ``load_sql``), browses their schema, and
profiles a dataset or a query result. This is deliberately ungoverned: a human exploring
is trusted and never routed through the verification oracle. The one bridge to the
governed world is :meth:`ExploreService.promote`, which authors a query as a certified
derivation through the same loop the chat and notebooks use -- an explicit, opt-in step,
not a gate on exploration.

Saved queries persist as plain exploration artifacts (a name and its SQL), scoped to the
caller when authentication is enabled.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from elbi_core import query_datasets
from elbi_core.errors import DataBindingError, ElbiError
from elbi_core.quality import profile_columns

from . import datasources
from .authoring import DeriveFactory, DraftFn
from .db import SavedQuery, Store
from .duplicate import copy_label
from .warehouse.service import WarehouseError, WarehouseService

#: The synthetic source id for the Delta Lake warehouse in the source picker (alongside
#: ``None`` for bound datasets and a UUID for a registered external database).
WAREHOUSE_SOURCE = "warehouse"

Rows = list[dict[str, Any]]
DraftFactory = Callable[[str, str], DraftFn]
#: A queryable dataset source: a file path (scanned lazily by DuckDB) or in-memory rows.
Source = Any
#: A dataset's columns and row count, read cheaply without materializing every row.
Schema = tuple[list[str], int]

#: Default cap on rows returned by a workbench query.
DEFAULT_MAX_ROWS = 1000
#: Rows sampled when profiling a query result (a dataset is profiled in full).
DEFAULT_PROFILE_ROWS = 50_000
#: A derivation name is a spec identifier; a promoted query's name must match.
_DERIVATION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: Tokens in a query, used to register only the datasets a query actually references.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ExploreService:
    """Run ad-hoc SQL, browse schemas, profile data, and save or promote queries."""

    def __init__(
        self,
        store: Store,
        dataset_names: Callable[[], list[str]],
        resolve_source: Callable[[str], Source],
        read_schema: Callable[[str], Schema],
        derive_factory: DeriveFactory | None = None,
        draft_factory: DraftFactory | None = None,
        promote_external: (Callable[[str, str, str], dict[str, Any]] | None) = None,
        warehouse: WarehouseService | None = None,
    ) -> None:
        self._store = store
        # A callable, not a snapshot: a table synced during this session has to be
        # queryable and promotable without a restart, and a list captured at
        # construction would leave it invisible until one.
        self._dataset_names = dataset_names
        # The Delta Lake warehouse, when the 'warehouse' extra is installed: its synced
        # tables become a queryable source in the workbench, via DuckDB delta_scan.
        self._warehouse = warehouse
        # resolve_source returns a file path (scanned lazily) or rows; read_schema reads
        # columns + row count cheaply. Both keep the workbench off the full-load path.
        self._resolve_source = resolve_source
        self._read_schema = read_schema
        self._derive_factory = derive_factory
        # Verify-only: reports the oracle's verdict for review, certifying nothing.
        self._draft_factory = draft_factory
        # Authors an external-source query as a trusted in-process derivation (the
        # sandbox has no database access), when the app wires the capability.
        self._promote_external = promote_external

    # -- schema browsing ------------------------------------------------------
    def sources(self) -> list[dict[str, Any]]:
        """The registered external data sources a query can target, plus bound data."""
        external = [
            {"id": source.id, "name": source.name, "kind": source.kind}
            for source in self._store.list_data_sources()
        ]
        warehouse = (
            [{"id": WAREHOUSE_SOURCE, "name": "Data warehouse", "kind": "warehouse"}]
            if self._warehouse is not None
            else []
        )
        return [
            {"id": None, "name": "Bound datasets", "kind": "project"},
            *warehouse,
            *external,
        ]

    def catalog(self, source_id: str | None = None) -> dict[str, Any]:
        """The tables and columns a query can reference for the chosen source.

        For bound datasets, each project dataset with its columns; for an external
        source, its tables and views reflected through SQLAlchemy.
        """
        if source_id == WAREHOUSE_SOURCE:
            return self._require_warehouse().duckdb_catalog()
        if source_id is not None:
            source = self._store.get_data_source(source_id)
            if source is None:
                raise DataBindingError(f"data source {source_id!r} not found")
            try:
                return datasources.catalog(source)
            except SQLAlchemyError as exc:
                raise DataBindingError(f"could not read source schema: {exc}") from exc
        tables: list[dict[str, Any]] = []
        for name in self._dataset_names():
            try:
                names, row_count = self._read_schema(name)
            except (ElbiError, WarehouseError):
                # Unbound or gone: leave the table out rather than fail the listing.
                continue
            columns = [{"name": column, "type": ""} for column in names]
            tables.append({"name": name, "columns": columns, "rows": row_count})
        return {"tables": tables}

    # -- querying -------------------------------------------------------------
    def run_sql(
        self, sql: str, source_id: str | None = None, max_rows: int = DEFAULT_MAX_ROWS
    ) -> dict[str, Any]:
        """Run read-only ``sql``; columns, bounded rows, truncation flag.

        Raises:
            DataBindingError: on a SQL error or an unresolvable source (a 400, since
                the query is the thing to fix).
        """
        if source_id == WAREHOUSE_SOURCE:
            try:
                columns, rows, truncated = self._require_warehouse().query(
                    sql, max_rows=max_rows
                )
            except WarehouseError as exc:
                raise DataBindingError(str(exc)) from exc
        elif source_id is not None:
            columns, rows, truncated = self._run_external(sql, source_id, max_rows)
        else:
            result = query_datasets(
                self._referenced_sources(sql), sql, max_rows=max_rows
            )
            columns, rows, truncated = result.columns, result.rows, result.truncated
        return {
            "columns": columns,
            "rows": rows,
            "rowCount": len(rows),
            "truncated": truncated,
        }

    def profile(
        self,
        dataset: str | None = None,
        sql: str | None = None,
        source_id: str | None = None,
        max_rows: int = DEFAULT_PROFILE_ROWS,
    ) -> dict[str, Any]:
        """Profile a bound dataset in full, or a query result sampled to ``max_rows``.

        Returns per-column statistics -- completeness, distinct count, inferred type,
        range, and top values -- the same profiler the contract engine uses.
        """
        if dataset is not None:
            data: Any = self._resolve_source(dataset)  # a path is profiled lazily
        elif sql is not None:
            data = self.run_sql(sql, source_id=source_id, max_rows=max_rows)["rows"]
        else:
            raise DataBindingError("profile needs a dataset or a query")
        profiles = profile_columns(data)
        columns = [
            {
                **profile.to_dict(),
                "isUnique": profile.is_unique,
                "fractionUniqueOnce": round(profile.fraction_unique_once, 4),
            }
            for profile in profiles
        ]
        return {"rowCount": profiles[0].count if profiles else 0, "columns": columns}

    # -- saved queries --------------------------------------------------------
    def save_query(
        self,
        name: str,
        sql: str,
        source_id: str | None = None,
        query_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a saved query and return its stored view."""
        existing = (
            self._store.get_saved_query(query_id) if query_id is not None else None
        )
        label = name.strip() or "Untitled query"
        if existing is not None:
            row = SavedQuery(
                id=existing.id,
                name=label,
                sql=sql,
                source_id=source_id,
                created_at=existing.created_at,
            )
        else:
            row = SavedQuery(name=label, sql=sql, source_id=source_id)
        query_id = row.id  # captured before the write commits and expires the instance
        self._store.save_saved_query(row)
        # Re-read through the read-only session: a committed instance has its attributes
        # expired, so build the view from a fresh load rather than the stale object.
        saved = self._store.get_saved_query(query_id)
        if saved is None:  # pragma: no cover - the row was just written
            raise DataBindingError("saved query could not be read back")
        return _query_view(saved)

    def list_queries(self) -> list[dict[str, Any]]:
        """The caller's saved queries, most-recently-updated first."""
        return [_query_view(row) for row in self._store.list_saved_queries()]

    def get_query(self, query_id: str) -> dict[str, Any] | None:
        """A saved query's view, or ``None`` when there is no such query."""
        row = self._store.get_saved_query(query_id)
        return _query_view(row) if row is not None else None

    def duplicate_query(self, query_id: str) -> dict[str, Any] | None:
        """Copy a saved query into a new one; ``None`` if the original is not found.

        A fresh :class:`~elbi.db.SavedQuery` rather than a re-save: writing over an
        existing id is an update, which would rewrite the query being copied.
        """
        row = self._store.get_saved_query(query_id)
        if row is None:
            return None
        taken = {q.name for q in self._store.list_saved_queries()}
        copy = SavedQuery(
            name=copy_label(row.name, taken),
            sql=row.sql,
            source_id=row.source_id,
            copied_from=query_id,
        )
        new_id = copy.id
        self._store.save_saved_query(copy)
        saved = self._store.get_saved_query(new_id)
        if saved is None:  # pragma: no cover - the row was just written
            raise DataBindingError("the duplicated query could not be read back")
        return _query_view(saved)

    def delete_query(self, query_id: str, permanent: bool = False) -> bool:
        """Move a saved query to trash, or erase it immediately with ``permanent``.

        Returns whether it existed.
        """
        if permanent:
            return self._store.erase_saved_query(query_id)
        return self._store.trash_saved_query(query_id)

    # -- promotion (the opt-in bridge to a certified derivation) --------------
    def promote(
        self,
        name: str,
        sql: str,
        source_id: str | None = None,
        claim: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Author a bound-dataset query as a certified derivation, the opt-in bridge.

        A bound-dataset query becomes an ordinary agent-authored derivation: generated
        source runs the query over its referenced datasets, routed through the project's
        authoring loop (sandboxed run, reproducibility and oracle/contract checks,
        certification) -- the same path the chat and notebooks use. An external-source
        query cannot run in the sandbox (no database access there), so it becomes a
        trusted, human-origin derivation that reads the source in-process. Either way a
        promoted query is a governed artifact, not a way around one.
        """
        if not _DERIVATION_NAME.match(name):
            return {
                "ok": False,
                "error": f"invalid derivation name {name!r}; use lowercase letters, "
                "digits, and underscores, starting with a letter",
            }
        if source_id is not None:
            if claim is not None:
                # The trusted external path runs no oracle; refuse rather than
                # certify around an unverified claim.
                return {
                    "ok": False,
                    "name": name,
                    "certified": False,
                    "error": "a claim requires the sandboxed bound-dataset path; "
                    "external-source promotion is trusted and runs no oracle",
                }
            if self._promote_external is None:
                return {
                    "ok": False,
                    "error": "external-source promotion is not configured",
                }
            return self._promote_external(name, sql, source_id)
        if self._derive_factory is None:
            return {"ok": False, "error": "derivation authoring is not configured"}
        try:
            # A referenced table can vanish mid-session; refuse, never 500.
            referenced = sorted(self._referenced_sources(sql))
        except ElbiError as exc:
            return {"ok": False, "name": name, "certified": False, "error": str(exc)}
        source = _promote_source(name, sql, referenced)
        derive = self._derive_factory("explore", f"Promoted from Explore: {name}")
        outcome = derive(name, source, claim, None, "table", (), ["duckdb", "pyarrow"])
        return {
            "ok": outcome.certified,
            "name": name,
            "certified": outcome.certified,
            "verdict": outcome.verdict,
            "rendered": outcome.rendered,
            "error": outcome.error,
            "detail": getattr(outcome, "detail", None),
        }

    def draft(
        self,
        name: str,
        sql: str,
        source_id: str | None = None,
        claim: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify a bound-dataset query as a derivation and report the verdict.

        The verify-only half of :meth:`promote`: the same generated source through the
        same authoring loop, but nothing is certified or persisted, so a draft can
        never serve. Certifying is the explicit :meth:`promote` follow-up. An external
        source has no sandbox, so it is promoted directly rather than drafted.
        """
        if not _DERIVATION_NAME.match(name):
            return {
                "ok": False,
                "error": f"invalid derivation name {name!r}; use lowercase letters, "
                "digits, and underscores, starting with a letter",
            }
        if source_id is not None:
            return {
                "ok": False,
                "error": "verify-only drafting covers bound-dataset queries; choose "
                "'Bound datasets' as the source, or promote an external source",
            }
        if self._draft_factory is None:
            return {"ok": False, "error": "derivation drafting is not configured"}
        try:
            referenced = sorted(self._referenced_sources(sql))
        except ElbiError as exc:
            return {"ok": False, "name": name, "certified": False, "error": str(exc)}
        source = _promote_source(name, sql, referenced)
        draft = self._draft_factory("explore", f"Drafted from Explore: {name}")
        result = draft(name, source, claim, None, "table", ["duckdb", "pyarrow"])
        result.setdefault("sql", sql)
        return result

    # -- internals ------------------------------------------------------------
    def _require_warehouse(self) -> WarehouseService:
        if self._warehouse is None:
            raise DataBindingError("the data warehouse is not available")
        return self._warehouse

    def _run_external(
        self, sql: str, source_id: str, max_rows: int
    ) -> tuple[list[str], Rows, bool]:
        source = self._store.get_data_source(source_id)
        if source is None:
            raise DataBindingError(f"data source {source_id!r} not found")
        # Fetch one past the cap so a full result is distinguishable from a clipped one.
        rows = datasources.load_rows(source, query=sql, max_rows=max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        return _columns_of(rows), rows, truncated

    def _referenced_sources(self, sql: str) -> dict[str, Source]:
        """Resolve only the bound datasets whose name appears as a token in ``sql``.

        A dataset a query never names cannot be in its ``FROM``/``JOIN``, so skipping it
        avoids touching unrelated datasets. A file source resolves to its path, which
        DuckDB scans in place rather than loading into memory.
        """
        tokens = set(_TOKEN.findall(sql))
        sources: dict[str, Source] = {}
        for name in self._dataset_names():
            if name in tokens:
                sources[name] = self._resolve_source(name)
        return sources


def _promote_source(name: str, sql: str, datasets: list[str]) -> str:
    """Generate the derivation source that runs ``sql`` over its referenced datasets."""
    bindings = "".join(f'        "{d}": ctx.input("{d}").rows,\n' for d in datasets)
    return (
        f"def {name}(ctx):\n"
        f'    """Promoted from the Explore SQL workbench."""\n'
        f"    from elbi_core import Artifact, query_datasets\n\n"
        f"    sources = {{\n{bindings}    }}\n"
        f"    result = query_datasets(sources, {json.dumps(sql)})\n"
        f"    return Artifact.table(result.rows)\n"
    )


def _columns_of(rows: Rows) -> list[str]:
    """Column names across ``rows``, in first-seen order (rows can differ in keys)."""
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _query_view(row: SavedQuery) -> dict[str, Any]:
    """The wire form of a saved query."""
    return {
        "id": row.id,
        "name": row.name,
        "sql": row.sql,
        "sourceId": row.source_id,
        "copiedFrom": row.copied_from,
        "createdAt": row.created_at.isoformat(),
        "updatedAt": row.updated_at.isoformat(),
    }
