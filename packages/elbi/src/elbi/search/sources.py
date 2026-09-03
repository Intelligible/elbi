"""Documents that are not one table's rows.

Everything in :mod:`~elbi.search.specs` is a straight projection of one model,
and the extractor there is deliberately incapable of anything else -- it selects the
declared columns and nothing more, which is what makes a sensitive column unreachable by
construction rather than by vigilance. Some documents are not shaped like that: a
warehouse table is a synced schema, its source and its recorded columns, assembled; a
registered model lives in MLflow and not in the store at all.

That assembly lives here so the strict path stays strict. A source is a plain function
from a :class:`Sources` to documents, and the guard tests treat its output as a spec's.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..db import Store
from .schema import SearchDoc

if TYPE_CHECKING:
    from ..ml import ModelService

logger = logging.getLogger(__name__)

#: The entity type a warehouse table is indexed under. Its columns are a *field* of this
#: document, not documents of their own: someone searching a column name wants the table
#: they can open, not a row they cannot.
WAREHOUSE_TABLE = "warehouse_table"

#: The entity type a registered model is indexed under.
MODEL = "model"


@dataclass(frozen=True)
class Sources:
    """What the non-table sources read from.

    The registry is a separate service, so a source that needs it says so here rather
    than reaching for a global. ``models`` is optional: an installation with no tracking
    server still has a searchable corpus.
    """

    store: Store
    models: ModelService | None = None
    #: Entity types this caller cannot speak for, however they extract. ``models=None``
    #: is ambiguous on its own -- for a server without the ml extra it means "there are
    #: no models", and for a command that simply did not build the service it means "I
    #: did not look" -- and a builder deletes what a pass did not produce. Naming the
    #: difference is what stops a rebuild emptying a corpus it could not see.
    unavailable: frozenset[str] = frozenset()


def warehouse_tables(context: Sources) -> Iterator[SearchDoc]:
    """Every synced warehouse table, with its column names as searchable text.

    Columns are their own field so their weight can be tuned apart from prose, and are
    matchable at ``browse`` because ``WarehouseService.columns`` already serves them
    there -- gating higher would let the catalog list a table search cannot find by the
    same column name.
    """
    store = context.store
    columns: dict[str, list[str]] = {}
    for column in store.list_warehouse_columns():
        columns.setdefault(column.table, []).append(column.name)
    for schema in store.list_all_external_schemas():
        if schema.status != "synced":
            # An unsynced table has no shape to describe and may never land at all.
            continue
        yield SearchDoc(
            entity_type=WAREHOUSE_TABLE,
            entity_id=schema.table,
            title=schema.table,
            columns=" ".join(columns.get(schema.table, ())),
            created_at=schema.created_at,
            route=f"/warehouse?table={schema.table}",
        )


#: Where a model version's verdict is recorded, promoted into a facet so a reader
#: filtering by verdict need not know the tag it lives under.
VERDICT_TAG = "elbi.oracle_verdict"

#: The platform's own tag namespace, excluded from the body: a caller searches for the
#: words a person chose, not machine bookkeeping.
INTERNAL_TAG_PREFIX = "elbi."


class SourceUnavailable(RuntimeError):
    """A source could not be read, so what it *would* have produced is unknown.

    Distinct from producing nothing. The builder deletes what a pass did not produce, so
    a source that swallowed its failure and yielded nothing would have every one of its
    documents read as deleted. Raising says "unknown"; the builder then leaves them be.
    """


def registry_models(context: Sources) -> Iterator[SearchDoc]:
    """Every registered model, titled by name and described by its own text.

    An unreachable registry raises :class:`SourceUnavailable` rather than yielding
    nothing, so the rest of the corpus indexes and the models already there survive.

    The verdict comes from the champion version, the one the platform serves: a model
    whose champion is rejected should not be findable as approved on the strength of
    some other version.
    """
    service = context.models
    if service is None:
        # Not a failure: an installation with no tracking server has no models, and an
        # empty corpus for this kind is the truth rather than a gap.
        return
    try:
        models = list(service.registry().models())
    except Exception as exc:
        raise SourceUnavailable("the model registry is unreachable") from exc

    for model in models:
        tags: dict[str, Any] = dict(getattr(model, "tags", {}) or {})
        described = [
            str(value)
            for key, value in tags.items()
            if not key.startswith(INTERNAL_TAG_PREFIX)
        ]
        yield SearchDoc(
            entity_type=MODEL,
            entity_id=model.name,
            title=model.name,
            body=" ".join(x for x in (model.description or "", *described) if x),
            verdict=_champion_verdict(service, model),
            created_at=_from_millis(getattr(model, "created_at_ms", None)),
            route=f"/models?model={model.name}",
        )


def _champion_verdict(service: ModelService, model: Any) -> str | None:
    """The oracle's verdict on the champion version, or ``None`` without a champion.

    A read failure raises rather than returning ``None``, because ``None`` is a real
    value here and the digest covers the verdict: returning it on error would rewrite
    the indexed model with an empty verdict, clearing the facet a reviewer filters on.
    """
    if not getattr(model, "champion_version", None):
        return None
    try:
        versions = service.registry().versions(model.name)
    except Exception as exc:
        raise SourceUnavailable(f"could not read versions of {model.name}") from exc
    champion = next(
        (v for v in versions if str(v.version) == str(model.champion_version)), None
    )
    return (champion.tags or {}).get(VERDICT_TAG) if champion else None


def _from_millis(value: Any) -> datetime | None:
    """MLflow's epoch milliseconds as a datetime, like every other ``created_at``."""
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


#: Every non-table document source, by entity type.
SOURCES: dict[str, Callable[[Sources], Iterator[SearchDoc]]] = {
    WAREHOUSE_TABLE: warehouse_tables,
    MODEL: registry_models,
}
