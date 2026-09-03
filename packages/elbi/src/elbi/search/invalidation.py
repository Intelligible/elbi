"""Keep the index current by watching the store's writes.

Freshness is an event, not a schedule: a notebook someone just saved should be
findable now, not within whatever interval the maintenance thread runs on. This turns
every committed write to the application database into a queue entry, which a drain
turns into index writes.

The listeners attach to the :class:`~sqlmodel.Session` *class*, filtered by bind, rather
than to :meth:`~elbi.db.Store._write`. That is the one decision here worth
explaining, and it is not tidiness: :class:`~elbi.db.DbJobStore` shares the
store's engine and opens its own sessions, and it is not the only thing that can. A hook
on the store's own write path sees none of them. Listening on the class catches both,
and catches whatever opens a session against that engine next year. Filtering by bind
leaves a second store in the same process alone, which every test suite has.

Capture and publication are split across two events. ``after_flush`` runs inside the
transaction, the only place ``session.new``/``dirty``/``deleted`` are still populated;
``after_commit`` runs only if the transaction actually committed. One event could not do
both without either missing the identities or enqueueing writes that were rolled back.

Changes go on a bounded queue a drain empties later, so a write never waits on an
index pass, and dropping one costs freshness the periodic sweep repairs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import event, inspect
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from .. import db
from .schema import FIELD_TITLE
from .sources import WAREHOUSE_TABLE
from .specs import SPECS, SqlSpec

logger = logging.getLogger(__name__)

#: Where a session's pending changes live between flush and commit.
_SESSION_KEY = "search_pending"

#: How many changes may queue before new ones are dropped. Generous enough that ordinary
#: traffic never reaches it, small enough that a runaway import cannot exhaust memory.
DEFAULT_LIMIT = 20_000

#: Which spec, if any, indexes a given model.
SPEC_BY_MODEL: dict[type, SqlSpec] = {spec.model: spec for spec in SPECS}

#: Parent models whose *title* their children's documents carry, mapped to the column it
#: comes from. A message, a notebook cell and a monitor incident each have no name of
#: their own and are labelled by what contains them -- so renaming the parent leaves
#: every one of them advertising the old name until a full rebuild.
TITLES_ITS_CHILDREN: dict[type, tuple[str, str]] = {}
for _spec in SPECS:
    if _spec.title_from_parent and _spec.parent:
        _parent = next(s for s in SPECS if s.entity_type == _spec.parent[0])
        _title = next((t.column for t in _parent.text if t.weight == FIELD_TITLE), None)
        if _title:
            TITLES_ITS_CHILDREN[_parent.model] = (_parent.entity_type, _title)

#: Rows that are not documents themselves but contribute to one assembled elsewhere,
#: mapped to the entity type they belong to and the attribute naming the document -- or
#: ``None`` where the row does not name one, and the whole kind is recomputed.
#:
#: A warehouse table's document is a synced schema plus its recorded columns plus its
#: connection, so writing any of the three reindexes the table rather than the row that
#: changed. A connection names no single table, and recomputing that kind costs
#: the same as asking for one of them, so it asks for all.
ASSEMBLED_INTO: dict[type, tuple[str, str | None]] = {
    db.ExternalDataSchema: (WAREHOUSE_TABLE, "table"),
    db.WarehouseColumn: (WAREHOUSE_TABLE, "table"),
    db.ExternalDataSource: (WAREHOUSE_TABLE, None),
}


@dataclass(frozen=True)
class Change:
    """One document to reindex, or one whose row is gone.

    ``entity_id`` is the document's, not the row's: a child carries its parent in front,
    matching the extractor, so the two cannot disagree about a document's id.
    """

    entity_type: str
    entity_id: str
    removed: bool = False


@dataclass(frozen=True)
class Retitled:
    """A parent renamed, so the children it labels are advertising the old name."""

    entity_type: str
    entity_id: str


@dataclass(frozen=True)
class Kind:
    """A whole entity type to recompute, for a row that names no single document."""

    entity_type: str


#: What a drain can be handed.
Queued = Change | Retitled | Kind


class PendingWrites:
    """A bounded, deduplicated queue of what the index still owes the store.

    Deduplicated because a request that touches one row twice owes one reindex, and
    because the unit here is a document rather than an edit: reindexing is idempotent,
    so collapsing repeats loses nothing.
    """

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._changes: set[Change] = set()
        self._retitled: set[Retitled] = set()
        self._kinds: set[Kind] = set()
        self._dropped = 0

    def _target(self, item: Queued) -> set[Any]:
        if isinstance(item, Kind):
            return self._kinds
        if isinstance(item, Retitled):
            return self._retitled
        return self._changes

    def add(self, items: Sequence[Queued]) -> None:
        """Queue changes, dropping them once the queue is full."""
        with self._lock:
            for item in items:
                target = self._target(item)
                if item in target:
                    # Already owed; counting it would report a backlog that is not.
                    continue
                if len(self) >= self._limit:
                    self._dropped += 1
                    continue
                target.add(item)

    def drain(self) -> tuple[list[Change], list[Retitled], list[Kind], int]:
        """Take everything queued, leaving the queue empty."""
        with self._lock:
            changes = sorted(self._changes, key=lambda c: (c.entity_type, c.entity_id))
            retitled = sorted(
                self._retitled, key=lambda r: (r.entity_type, r.entity_id)
            )
            kinds = sorted(self._kinds, key=lambda k: k.entity_type)
            dropped = self._dropped
            self._changes = set()
            self._retitled = set()
            self._kinds = set()
            self._dropped = 0
        return changes, retitled, kinds, dropped

    def __len__(self) -> int:
        # ``add`` and ``drain`` already hold the lock; re-entering would deadlock.
        return len(self._changes) + len(self._retitled) + len(self._kinds)


def _document_id(spec: SqlSpec, obj: Any) -> str | None:
    """The document id this row contributes, mirroring the extractor exactly.

    A child's id carries its parent's, which is what the extractor writes; a child whose
    parent is missing has no access to inherit and so is not a document at all.
    """
    raw = getattr(obj, spec.id_column, None)
    if raw is None or str(raw) == "":
        return None
    if spec.parent:
        fk = getattr(obj, spec.parent[1], None)
        if fk is None:
            return None
        return f"{fk}/{raw}"
    return str(raw)


def _changed(obj: Any, column: str | None) -> bool:
    """Whether this write altered ``column``, assuming it did if that cannot be told."""
    if not column:
        return False
    try:
        return bool(inspect(obj).attrs[column].history.has_changes())
    except Exception:
        # Unmapped or detached. Assuming a change costs one reindex; assuming none
        # leaves the index describing a row that has moved on.
        return True


def changes_for(obj: Any, *, removed: bool, created: bool = False) -> Iterator[Queued]:
    """Every queue entry one written row implies."""
    assembled = ASSEMBLED_INTO.get(type(obj))
    if assembled is not None:
        entity_type, attribute = assembled
        if attribute is None:
            yield Kind(entity_type)
        else:
            target = getattr(obj, attribute, None)
            if target:
                # Never ``removed``: the row that vanished is a part, not the
                # document, and the drain drops whatever no longer extracts.
                yield Change(entity_type, str(target))
        # No early return: a row can be both a part of an assembled document and a
        # document in its own right, as a warehouse connection is.

    spec = SPEC_BY_MODEL.get(type(obj))
    if spec is None:
        return
    doc_id = _document_id(spec, obj)
    if doc_id is not None:
        yield Change(spec.entity_type, doc_id, removed=removed)
    titles = TITLES_ITS_CHILDREN.get(type(obj))
    if titles and not created and _changed(obj, titles[1]):
        raw = getattr(obj, spec.id_column, None)
        if raw is not None:
            yield Retitled(titles[0], str(raw))


def install(engine: Engine, pending: PendingWrites) -> Callable[[], None]:
    """Record every committed change to ``engine`` on ``pending``; return a remover.

    The returned callable detaches the listeners. Without it, a process that opens many
    stores -- every test suite -- accumulates them on the global ``Session`` class and
    retains every past store through the closures.
    """

    def _ours(session: Session) -> bool:
        try:
            return session.get_bind() is engine
        except Exception:
            return False

    def _capture(session: Session, _context: Any) -> None:
        if not _ours(session):
            return
        collected: list[Queued] = []
        for obj, removed in (
            *((o, False) for o in (*session.new, *session.dirty)),
            *((o, True) for o in session.deleted),
        ):
            try:
                created = not removed and obj in session.new
                collected.extend(changes_for(obj, removed=removed, created=created))
                if (
                    not created
                    and not removed
                    and isinstance(obj, db.Notebook)
                    and _changed(obj, "deleted_at")
                ):
                    # A trash/restore is an UPDATE on the notebook, not its cells, so
                    # nothing above already asked to re-check them -- without this
                    # they would sit in the index exactly as they were, findable
                    # regardless of the notebook's own trash state.
                    for cell_id in session.exec(
                        select(db.NotebookCell.id).where(
                            db.NotebookCell.notebook_id == obj.id
                        )
                    ):
                        collected.append(Change("notebook_cell", f"{obj.id}/{cell_id}"))
            except Exception:
                # Per object: one row that cannot be read must not cost the rest of the
                # transaction its reindex.
                logger.exception("search: capturing a change failed")
        if collected:
            session.info.setdefault(_SESSION_KEY, []).extend(collected)

    def _publish(session: Session) -> None:
        if not _ours(session):
            return
        collected = session.info.pop(_SESSION_KEY, None)
        if collected:
            pending.add(collected)

    def _discard(session: Session) -> None:
        if _ours(session):
            session.info.pop(_SESSION_KEY, None)

    event.listen(Session, "after_flush", _capture)
    event.listen(Session, "after_commit", _publish)
    event.listen(Session, "after_rollback", _discard)

    def remove() -> None:
        event.remove(Session, "after_flush", _capture)
        event.remove(Session, "after_commit", _publish)
        event.remove(Session, "after_rollback", _discard)

    return remove
