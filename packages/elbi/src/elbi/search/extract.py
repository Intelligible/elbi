"""Turn store rows into search documents, reading only what a spec declares.

The one place a spec becomes a query. It builds an explicit ``SELECT`` of the declared
columns and nothing else, which is what makes exclusion structural rather than a matter
of remembering: a column no spec names is never in the statement, so it cannot reach the
index however sensitive it later turns out to be.

A row that hangs off another, like a message from its conversation, is keyed to that
parent rather than to itself, so dropping the parent drops what it produced.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, col, select

from .schema import FIELD_BODY, FIELD_COLUMNS, FIELD_TITLE, SearchDoc
from .sources import SOURCES, Sources, SourceUnavailable
from .specs import SPECS, SqlSpec, windows

logger = logging.getLogger(__name__)


def _value(row: Any, column: str) -> Any:
    """One selected column's value, whatever shape the row came back as."""
    return getattr(row, column, None)


def _text_for(spec: SqlSpec, row: Any, weight: str) -> str:
    """Join every declared column contributing to one field, in order."""
    return " ".join(text for text, _, _ in _parts(spec, row, weight))


def _parts(spec: SqlSpec, row: Any, weight: str) -> list[tuple[str, int, int]]:
    """Each contributing column's rendered text, with its span in the joined field.

    Chunking works off these spans rather than re-deriving offsets, so a late chunk's
    span means the same thing under either mode.
    """
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for text in spec.text:
        if text.weight != weight:
            continue
        raw = _value(row, text.column)
        rendered = text.transform(raw) if text.transform else str(raw or "")
        if not rendered:
            continue
        if out:
            cursor += 1  # the space the join inserts
        out.append((rendered, cursor, cursor + len(rendered)))
        cursor += len(rendered)
    return out


def _chunks(
    spec: SqlSpec, body: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """The body text this row becomes, one entry per document, with its span.

    Always at least one: an artifact with only a name must still be findable by it.
    """
    joined = " ".join(text for text, _, _ in body)
    if spec.chunk.mode == "field":
        return body or [("", 0, 0)]
    if spec.chunk.mode == "split" and joined:
        return [
            (joined[start:end], start, end)
            for start, end in windows(len(joined), spec.chunk)
        ]
    return [(joined, 0, len(joined))]


def _context(
    spec: SqlSpec, title: str, body: list[tuple[str, int, int]]
) -> tuple[str, int]:
    """The text a late chunk is pooled over, and how far it shifts the chunk's span.

    The title leads it, so a fragment is encoded knowing what it belongs to -- a cell's
    title is its notebook's, the only human label it has. The span still covers the
    chunk alone, so the title reaches it through attention rather than concatenation.

    Empty unless the entity is late-chunked; the document then embeds ``title body``
    like any other.
    """
    if not spec.chunk.late:
        return "", 0
    joined = " ".join(text for text, _, _ in body)
    prefix = f"{title} " if title else ""
    return prefix + joined, len(prefix)


@dataclass(frozen=True)
class _Parent:
    """What a child inherits from the row it hangs off: its title.

    A message has no name of its own, so the conversation's is what labels the result.
    """

    title: str = ""


def _parents(spec: SqlSpec, session: Session) -> dict[str, _Parent]:
    """Every parent row's inheritable facts, keyed by the foreign key pointing at it.

    One query for the parent table rather than a lookup per child; parents are the
    small side.
    """
    if not spec.parent:
        return {}
    parent_type, _ = spec.parent
    parent = next(s for s in SPECS if s.entity_type == parent_type)
    columns = [parent.id_column]
    columns.extend(t.column for t in parent.text if t.weight == FIELD_TITLE)
    statement = select(
        *[col(getattr(parent.model, name)) for name in dict.fromkeys(columns)]
    )
    if parent.where:
        # The parent's own exclusion (e.g. trashed) applies here too: a row that
        # would not extract on its own should not resolve as a child's parent
        # either, which is what makes a trashed notebook's cells drop out of the
        # index along with it instead of surviving as unreachable orphans.
        where_column = col(getattr(parent.model, parent.where[0]))
        where_value = parent.where[1]
        statement = statement.where(
            where_column.is_(None)
            if where_value is None
            else where_column == where_value
        )
    rows = session.execute(statement)
    return {
        str(_value(row, parent.id_column)): _Parent(
            title=_text_for(parent, row, FIELD_TITLE),
        )
        for row in rows
    }


def extract(
    spec: SqlSpec,
    session: Session,
    *,
    ids: Collection[str] | None = None,
    parent_ids: Collection[str] | None = None,
) -> Iterator[SearchDoc]:
    """Every document this entity contributes, reading only its declared columns.

    ``ids`` narrows to particular rows and ``parent_ids`` to the children of particular
    parents, which is what makes an incremental update cost the rows that changed. Both
    are matched against the *row's* own key, not the document id: a child's document id
    carries its parent in front, and reassembling it here rather than parsing it back is
    what keeps the two definitions from drifting.
    """
    columns = [col(getattr(spec.model, name)) for name in spec.declared_columns()]
    statement = select(*columns)
    if spec.where:
        where_column = col(getattr(spec.model, spec.where[0]))
        where_value = spec.where[1]
        statement = statement.where(
            where_column.is_(None)
            if where_value is None
            else where_column == where_value
        )
    if ids is not None:
        statement = statement.where(col(getattr(spec.model, spec.id_column)).in_(ids))
    if parent_ids is not None and spec.parent:
        statement = statement.where(
            col(getattr(spec.model, spec.parent[1])).in_(parent_ids)
        )

    parents = _parents(spec, session)
    governing = spec.key_type()

    # ``execute``, not ``exec``: a one-column ``select`` is a *scalar* select, which
    # ``exec`` unwraps to bare values -- and three specs declare exactly one column.
    for row in session.execute(statement):
        entity_id = str(_value(row, spec.id_column) or "")
        if not entity_id:
            continue

        inherited = _Parent()
        route_parent = ""
        if spec.parent:
            parent_type, fk = spec.parent
            fk_value = _value(row, fk)
            resolved = parents.get(str(fk_value)) if fk_value is not None else None
            if resolved is None:
                # Nothing to key it under, and a document keyed under nothing is one
                # dropping its parent would leave behind.
                continue
            route_parent = f"{parent_type}:{fk_value}"
            inherited = resolved
            entity_id = f"{fk_value}/{entity_id}"
            parent_id = str(fk_value)
        else:
            parent_id = entity_id

        title = (
            inherited.title
            if spec.title_from_parent
            else _text_for(spec, row, FIELD_TITLE)
        )

        facets = {
            facet.facet: facet.derive(_value(row, facet.column))
            if facet.derive
            else _value(row, facet.column)
            for facet in spec.facets
        }
        body = _parts(spec, row, FIELD_BODY)
        chunks = _chunks(spec, body)
        route = _route_for(spec, entity_id, route_parent)
        context, offset = _context(spec, title, body)
        for index, (text, start, end) in enumerate(chunks):
            yield SearchDoc(
                entity_type=spec.entity_type,
                entity_id=entity_id,
                title=title,
                body=text,
                columns=_text_for(spec, row, FIELD_COLUMNS),
                parent_type=governing,
                parent_id=parent_id,
                chunk_index=index,
                context=context,
                span=(start + offset, end + offset) if context else None,
                verdict=facets.get("verdict"),
                status=facets.get("status"),
                incident_state=facets.get("incident_state"),
                created_at=facets.get("created_at"),
                data_hash=facets.get("data_hash"),
                route=route,
            )


def _route_for(spec: SqlSpec, entity_id: str, route_parent: str) -> str:
    """Fill the spec's route template, so a result needs no lookup to be clickable."""
    if not spec.route:
        return ""
    parent = route_parent.split(":", 1)[-1]
    identifier = entity_id
    if spec.parent and "/" in entity_id:
        # A child's id carries its parent, e.g. a message as "conversation/message".
        parent, identifier = entity_id.split("/", 1)
    return spec.route.replace("{id}", identifier).replace("{parent}", parent)


@dataclass(frozen=True)
class Extraction:
    """Everything a pass produced, and what it could not read.

    ``incomplete`` names the entity types whose source failed. "Produced nothing of kind
    X" and "could not look at kind X" are indistinguishable to a caller diffing against
    the index, and only the first means the documents were deleted.
    """

    docs: list[SearchDoc]
    incomplete: frozenset[str]


def extract_all(context: Sources) -> Extraction:
    """Every document there is, for a full build.

    A source raising :class:`~.sources.SourceUnavailable` marks its entity type
    incomplete rather than aborting the pass; the rest of the corpus still indexes.
    """
    docs: list[SearchDoc] = []
    incomplete: set[str] = set(context.unavailable)
    with Session(context.store._engine) as session:
        for spec in SPECS:
            docs.extend(extract(spec, session))
    for entity_type, source in SOURCES.items():
        if entity_type in context.unavailable:
            continue
        try:
            docs.extend(source(context))
        except SourceUnavailable as exc:
            logger.warning(
                "%s is unavailable; leaving what is indexed alone: %s", entity_type, exc
            )
            incomplete.add(entity_type)
            docs = [doc for doc in docs if doc.entity_type != entity_type]
    return Extraction(docs=docs, incomplete=frozenset(incomplete))


def extract_type(context: Sources, entity_type: str) -> Iterator[SearchDoc]:
    """Every document of one entity type, for a bulk reindex of that kind."""
    source = SOURCES.get(entity_type)
    if source is not None:
        yield from source(context)
        return
    spec = next((s for s in SPECS if s.entity_type == entity_type), None)
    if spec is None:
        return
    with Session(context.store._engine) as session:
        yield from extract(spec, session)


def _row_key(spec: SqlSpec, entity_id: str) -> str:
    """The row's own key behind a document id, dropping a child's parent prefix."""
    return entity_id.split("/", 1)[1] if spec.parent and "/" in entity_id else entity_id


def extract_selected(
    context: Sources,
    wanted: Mapping[str, Sequence[str]],
    children_of: Mapping[str, Sequence[str]] | None = None,
) -> Iterator[SearchDoc]:
    """Just the named documents, for an incremental update.

    ``wanted`` maps an entity type to the document ids to rebuild; ``children_of`` maps
    a parent entity type to parent ids whose children must all be rebuilt, which a
    rename up the tree requires -- a child carries its parent's title, so the row that
    changed is not the row that has to be rewritten.
    """
    by_type = {spec.entity_type: spec for spec in SPECS}
    with Session(context.store._engine) as session:
        for entity_type, entity_ids in wanted.items():
            source = SOURCES.get(entity_type)
            if source is not None:
                # No single row to select, so recompute the kind and keep what was
                # asked for. Cheap: these sources are small by construction.
                requested = set(entity_ids)
                yield from (d for d in source(context) if d.entity_id in requested)
                continue
            spec = by_type.get(entity_type)
            if spec is None or not entity_ids:
                continue
            keys = {_row_key(spec, entity_id) for entity_id in entity_ids}
            yield from extract(spec, session, ids=keys)
        for parent_type, parent_ids in (children_of or {}).items():
            if not parent_ids:
                continue
            for spec in SPECS:
                if spec.parent and spec.parent[0] == parent_type:
                    yield from extract(spec, session, parent_ids=set(parent_ids))
