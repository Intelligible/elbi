"""Filling the index: extract, embed what changed, write, and drop what is gone.

One entry point, :meth:`SearchBuilder.build`, and it is idempotent. A maintenance pass
runs it repeatedly, so a build that re-embedded its corpus each time would cost more
than the search it serves.

Three things a naive version gets wrong, each of which was got wrong here:

* **Delete.** Insert-only leaves a deleted artifact findable for the life of the file.
* **Re-embed only what changed**, which the document digest decides.
* **Skip the embedder where it buys nothing** -- see :data:`LEXICAL_ONLY`.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from elbi_core.retrieval import Embedder, SpanEmbedder

from .extract import extract_all, extract_selected, extract_type
from .index import SearchIndex
from .invalidation import PendingWrites
from .schema import SearchDoc, split_doc_id
from .sources import Sources

logger = logging.getLogger(__name__)

#: Entity types indexed lexically, with no vector.
#:
#: ``audit_event`` is 50,000 of ~70,000 estimated chunks and the only entity with no
#: ceiling. Its text is identifiers, where BM25 is strongest and a vector weakest, so
#: embedding it is the one part that buys nothing: 108 MB of vectors becomes 32 MB and
#: "who deleted the revenue dashboard" still answers.
LEXICAL_ONLY: frozenset[str] = frozenset({"audit_event"})

#: How many documents are embedded per call. Large enough to amortise a model
#: invocation, small enough that a build's peak memory is bounded by the batch rather
#: than by the corpus.
EMBED_BATCH = 64


@dataclass
class BuildReport:
    """What one build did, per entity type.

    ``indexed`` counts documents *written*, so a second run over an unchanged store
    reports zero: idempotence you can see rather than assert. ``total`` is the corpus
    after the run, and what :func:`~.document_map.estimate` is compared against.
    """

    indexed: Counter[str] = field(default_factory=Counter)
    removed: int = 0
    total: Counter[str] = field(default_factory=Counter)
    embedded: int = 0

    def summary(self) -> str:
        """One line for the log: what moved, not the whole census."""
        changed = sum(self.indexed.values())
        return (
            f"{changed} written, {self.removed} removed, {self.embedded} embedded; "
            f"{sum(self.total.values())} documents over {len(self.total)} entity "
            f"types"
        )


class SearchBuilder:
    """Builds and maintains the index from the store and the registry.

    Args:
        index: The open index to write into.
        sources: What the non-table document sources read from.
        embedder: ``None`` is ``search: lexical``: indexed and ranked with no vectors
            and no model, so nothing is downloaded.
    """

    def __init__(
        self, index: SearchIndex, sources: Sources, embedder: Embedder | None = None
    ) -> None:
        self._index = index
        self._sources = sources
        self._embedder = embedder
        # The index guards each statement, not the read-decide-write sequence, so a
        # drain writing between them lands in the other pass's ``gone`` set.
        self._pass = threading.Lock()

    @property
    def index(self) -> SearchIndex:
        """The index this builder writes, for the readers that serve it."""
        return self._index

    def embed_query(self, text: str) -> list[float] | None:
        """One query as a vector, or ``None`` where this build carries no vectors.

        The reader needs it to ask the dense half anything at all, and the builder is
        what holds the embedder -- ``search: lexical`` has none, and neither has any
        deployment whose first embed failed, so a caller must treat ``None`` as "rank
        lexically" rather than as an error.
        """
        if self._embedder is None or not text.strip():
            return None
        try:
            return list(self._embedder.embed([text])[0])
        except Exception:
            # A query that cannot be embedded still has a lexical answer, and a search
            # box that fails closed on a model hiccup is worse than one that narrows.
            logger.warning("search: could not embed the query", exc_info=True)
            return None

    def close(self) -> None:
        """Release the index file, so a clean shutdown does not leave it held."""
        self._index.close()

    def build(self) -> BuildReport:
        """Bring the index up to date with the store. Safe to run repeatedly."""
        with self._pass:
            return self._build()

    def _build(self) -> BuildReport:
        extraction = extract_all(self._sources)
        docs = _deduplicated(extraction.docs)
        changed, gone = self._index.diff(docs)
        changed = self._with_repairs(changed, docs)
        gone = _keep_deletable(gone, extraction.incomplete)
        report = BuildReport(total=Counter(doc.entity_type for doc in docs))

        for batch in _batched(changed, EMBED_BATCH):
            vectors = self._embed(batch)
            self._index.upsert(batch, vectors)
            report.indexed.update(doc.entity_type for doc in batch)
            report.embedded += sum(1 for v in vectors if v is not None)

        if gone:
            report.removed = self._index.forget(gone)
        if not self._index.trusted():
            # Whatever they disagreed about is settled. Without this a single
            # transient failure disabled search until somebody ran the CLI.
            self._index.set_trusted(True)
            logger.info("search index rebuilt; serving it again")

        logger.info("search index build: %s", report.summary())
        return report

    def apply_pending(self, pending: PendingWrites) -> int:
        """Drain queued writes into the index. Returns documents written and removed.

        The incremental counterpart to :meth:`build`: it reindexes what changed instead
        of scanning the corpus, which is what makes a saved artifact findable now rather
        than on the next housekeeping pass.

        Anything asked for that did not extract is forgotten. That rule is what stops a
        missed delete stranding a document: the queue can be wrong about *why* a row
        needs looking at, but "it is no longer there" is answered by looking.
        """
        with self._pass:
            return self._apply_pending(pending)

    def _apply_pending(self, pending: PendingWrites) -> int:
        changes, retitled, kinds, dropped = pending.drain()
        if dropped:
            logger.warning(
                "search: %d queued changes dropped; the repair sweep will catch them",
                dropped,
            )
        try:
            written = 0
            if not (changes or retitled or kinds):
                return written

            # The event is not proof: several kinds are edited as delete-then-insert
            # across two commits, so "deleted" arrives for a row that exists again.
            wanted: dict[str, list[str]] = {}
            for change in changes:
                wanted.setdefault(change.entity_type, []).append(change.entity_id)
            children_of: dict[str, list[str]] = {}
            for parent in retitled:
                children_of.setdefault(parent.entity_type, []).append(parent.entity_id)

            docs = _deduplicated(
                [
                    *extract_selected(self._sources, wanted, children_of),
                    *(
                        doc
                        for kind in kinds
                        for doc in extract_type(self._sources, kind.entity_type)
                    ),
                ]
            )
            produced = {(doc.entity_type, doc.entity_id) for doc in docs}
            asked = {(kind, i) for kind, ids in wanted.items() for i in ids}
            for kind in kinds:
                asked |= {
                    (kind.entity_type, entity_id)
                    for entity_id in self._index.entity_ids(kind.entity_type)
                }
            stale = asked - produced
            chunks = Counter((doc.entity_type, doc.entity_id) for doc in docs)

            for batch in _batched(docs, EMBED_BATCH):
                self._index.upsert(batch, self._embed(batch))
                written += len(batch)
            if stale:
                written += self._index.forget_entities(stale)
            # An entity whose text shrank keeps every chunk past its new end otherwise.
            written += self._index.trim_chunks(chunks)
        except Exception:
            # ``drain`` emptied the queue; without this the batch is simply lost, and
            # everything above can raise -- extraction reads the store and the write
            # touches the index.
            pending.add([*changes, *retitled, *kinds])
            raise
        return written

    def _with_repairs(
        self, changed: list[SearchDoc], docs: Sequence[SearchDoc]
    ) -> list[SearchDoc]:
        """``changed``, plus anything stored in a state the digest cannot see.

        The digest covers what a document *says*, so a row written while the embedder
        was failing has no vector and a correct hash, and every later pass skips it.
        Asking which rows have no vector is what makes the retry real.
        """
        if self._embedder is None:
            return changed
        wanted = self._index.unembedded()
        if not wanted:
            return changed
        already = {doc.doc_id for doc in changed}
        return changed + [
            doc
            for doc in docs
            if doc.doc_id in wanted and doc.doc_id not in already and _wants_vector(doc)
        ]

    def _embed(self, docs: Sequence[SearchDoc]) -> list[Any]:
        """One vector per document, or ``None`` where a document carries no vector."""
        if self._embedder is None:
            return [None] * len(docs)

        wanted = [index for index, doc in enumerate(docs) if _wants_vector(doc)]
        vectors: list[Any] = [None] * len(docs)
        if not wanted:
            return vectors

        try:
            for position, vector in zip(
                wanted, self._encode([docs[i] for i in wanted]), strict=True
            ):
                vectors[position] = list(vector)
        except Exception:
            # Still findable lexically, and the next build retries the vector.
            logger.warning(
                "could not embed a batch; indexing it lexically", exc_info=True
            )
            return [None] * len(docs)
        return vectors

    def _encode(self, docs: Sequence[SearchDoc]) -> list[Sequence[float]]:
        """Encode a batch, pooling over the parent row where a document asks for it.

        Late-chunked documents are grouped by the text they were cut from, so a parent
        is encoded once and every chunk pooled out of that pass rather than one pass per
        chunk.

        A plain :class:`~elbi_core.retrieval.Embedder` has no context to pool --
        a static model gives each token one fixed vector however it is surrounded -- so
        those documents fall back to per-chunk embedding rather than failing.
        """
        embedder = self._embedder
        assert embedder is not None  # noqa: S101 - guarded by the only caller
        vectors: list[Any] = [None] * len(docs)

        late: dict[str, list[int]] = {}
        plain: list[int] = []
        contextual = isinstance(embedder, SpanEmbedder)
        for position, doc in enumerate(docs):
            if contextual and doc.context and doc.span:
                late.setdefault(doc.context, []).append(position)
            else:
                plain.append(position)

        if late:
            assert isinstance(embedder, SpanEmbedder)  # noqa: S101 - `contextual`
            for context, positions in late.items():
                spans = [docs[i].span for i in positions]
                pooled = embedder.embed_spans(
                    context, [s for s in spans if s is not None]
                )
                for position, vector in zip(positions, pooled, strict=True):
                    vectors[position] = vector

        if plain:
            texts = [f"{docs[i].title} {docs[i].body}".strip() for i in plain]
            for position, vector in zip(plain, embedder.embed(texts), strict=True):
                vectors[position] = vector
        return vectors


#: How long a queued write waits before the index is asked to catch up. Short enough
#: that saving something and searching for it feels immediate, long enough that a burst
#: of writes is one pass rather than one per row.
DRAIN_SECONDS = 2.0


class Drain:
    """Applies queued writes on a short cadence, off the thread that made them.

    The maintenance scheduler is the wrong clock for this: it runs hourly, and an
    artifact that takes an hour to become findable is the staleness the ticket exists to
    remove. So this is its own thread on its own interval, and the hourly full build
    stays as the sweep that repairs whatever the queue dropped.

    A daemon thread that logs and continues, following ``webhooks.py``: keeping the
    index current must never take down the process it runs beside.
    """

    def __init__(
        self,
        builder: SearchBuilder,
        pending: PendingWrites,
        interval: float = DRAIN_SECONDS,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._builder = builder
        self._pending = pending
        self._interval = interval
        self._on_stop = on_stop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin draining. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="search-invalidation", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Detach the listeners, stop draining, and apply whatever is left. Idempotent.

        The listeners go first so nothing new arrives, and the final pass runs on this
        thread rather than the drain's: the caller closes the index next, and joining
        with a timeout can return while the drain is still inside a write.
        """
        if self._on_stop is not None:
            self._on_stop()
            self._on_stop = None
        self._stop.set()
        running = self._thread
        if running is not None:
            running.join(timeout=30)
            self._thread = None
        if running is not None and running.is_alive():
            # It outlived the join, so it may be mid-write. A final pass here would be a
            # second writer against an index the caller closes next, and the close takes
            # the same lock only for the statement it is running.
            logger.critical(
                "search: the drain did not stop; leaving %d changes for the next build",
                len(self._pending),
            )
            return
        if len(self._pending):
            try:
                self._builder.apply_pending(self._pending)
            except Exception:
                logger.exception("search: the final drain failed")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if not len(self._pending):
                continue
            try:
                written = self._builder.apply_pending(self._pending)
            except Exception:
                logger.exception("search: applying queued writes failed")
                continue
            if written:
                logger.debug("search index updated: %d documents", written)


def _wants_vector(doc: SearchDoc) -> bool:
    """Whether this document should carry an embedding at all."""
    return doc.entity_type not in LEXICAL_ONLY and bool(doc.title or doc.body)


def _deduplicated(docs: Sequence[SearchDoc]) -> list[SearchDoc]:
    """``docs`` with one document per id, keeping the first and logging the rest.

    ``doc_id`` is the primary key, so a duplicate aborts the whole batch -- and every
    later build, since the collision recurs each pass. One entity type naming two rows
    alike would stop the *entire* corpus from indexing. Two warehouse sources of the
    same type syncing a same-named table is a real case of it.
    """
    seen: dict[str, SearchDoc] = {}
    collisions: Counter[str] = Counter()
    for doc in docs:
        if doc.doc_id in seen:
            collisions[doc.entity_type] += 1
            continue
        seen[doc.doc_id] = doc
    for entity_type, count in collisions.items():
        logger.warning(
            "%s produced %d document(s) sharing an id with another; indexed once",
            entity_type,
            count,
        )
    return list(seen.values())


def _keep_deletable(gone: set[str], incomplete: frozenset[str]) -> set[str]:
    """``gone``, minus anything of an entity type this pass could not read.

    A document is deleted because the pass did not produce it, and that inference holds
    only where the pass could see the kind. Otherwise a tracking server restarting
    during housekeeping empties the model corpus.
    """
    if not incomplete:
        return gone
    return {doc_id for doc_id in gone if split_doc_id(doc_id)[0] not in incomplete}


def _batched(items: Sequence[SearchDoc], size: int) -> Iterator[list[SearchDoc]]:
    """``items`` in chunks of ``size``, the last one short."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def counts_against_estimate(report: BuildReport) -> list[tuple[str, int, int]]:
    """``(entity_type, indexed, estimated)`` for every entity either side knows about.

    The parent's "no vector store" decision rests on an estimated corpus size, so
    reporting actuals is what makes it checkable. Both numbers rather than a verdict: a
    large divergence is the signal to revisit the decision, not to fail a build.
    """
    from .document_map import estimate

    estimated = {row.entity_type: row.chunks for row in estimate()}
    names = sorted(set(report.total) | set(estimated))
    return [(name, report.total.get(name, 0), estimated.get(name, 0)) for name in names]


def log_counts(report: BuildReport) -> None:
    """Log the per-entity counts beside what :mod:`.document_map` predicted."""
    for name, actual, expected in counts_against_estimate(report):
        logger.info(
            "search corpus: %s %d indexed, %d estimated", name, actual, expected
        )


def documents(sources: Sources) -> list[SearchDoc]:
    """Every document the store would produce, without touching an index."""
    return _deduplicated(extract_all(sources).docs)
