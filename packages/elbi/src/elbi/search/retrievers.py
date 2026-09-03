"""The persisted index, presented as the retrieval seam the registry already uses.

The in-memory retrievers recompute corpus statistics per call over the candidates they
are handed -- right for a project's derivations, wrong for a platform's corpus. The
*interface* is the same, so this is a swap at a seam rather than a second ranking stack.

Pointing a ``Registry`` here makes the agent's ``search_derivations`` read the same
index the palette reads. Fusion is untouched: the same
:class:`~elbi_core.retrieval.HybridRetriever`, the same RRF constant.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from elbi_core.derivation import Derivation
from elbi_core.retrieval import (
    DEFAULT_MIN_SIMILARITY,
    Bm25Retriever,
    Embedder,
    EmbeddingRetriever,
    HybridRetriever,
    Retriever,
)

from . import sql as q
from .index import DEFAULT_CANDIDATES, SearchIndex
from .schema import split_doc_id

logger = logging.getLogger(__name__)

#: The entity type a derivation is indexed under, and the only one a ``Registry``'s
#: corpus can contain.
DERIVATION = "derivation"


def _resolve(
    ranked: Iterable[tuple[str, float]],
    by_name: dict[str, Derivation],
    limit: int,
) -> list[Derivation]:
    """Ranked document ids back to the derivations the caller passed in.

    A hit with no derivation behind it is one the index has not caught up on, and is
    skipped rather than treated as an error.
    """
    out: list[Derivation] = []
    seen: set[str] = set()
    for doc_id, _ in ranked:
        _, entity_id = split_doc_id(doc_id)
        derivation = by_name.get(entity_id)
        if derivation is None or entity_id in seen:
            continue
        seen.add(entity_id)
        out.append(derivation)
        if len(out) >= limit:
            break
    return out


def _scope(names: Iterable[str]) -> dict[str, Any]:
    """The predicate restricting a ranking to exactly these derivations.

    In the query, not after it. Ranking the whole store and intersecting afterwards
    spends the candidate window on rows the caller excluded -- for
    ``search_derivations`` those are drafts and uncertified derivations, whose whole
    stated invariant is that they can never crowd out a real match.
    """
    where_sql, where_params = q.filter_sql({"entity_type": [DERIVATION]})
    return {
        "where_sql": f"({where_sql}) AND d.entity_id IN (SELECT UNNEST(?::VARCHAR[]))",
        "where_params": [*where_params, list(names)],
    }


class IndexedBm25Retriever:
    """BM25F over the persisted index, ranking a registry's derivations.

    The same scoring as :class:`~elbi_core.retrieval.Bm25Retriever`, over
    statistics maintained as documents are written rather than recomputed per call.

    ``candidates`` exceeds any sensible ``limit`` because a chunked entity contributes
    several documents, all competing for the same window.
    """

    def __init__(
        self, index: SearchIndex, *, candidates: int = DEFAULT_CANDIDATES
    ) -> None:
        self._index = index
        self._candidates = candidates

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Rank ``corpus`` by BM25F over the index; abstain on no hit."""
        by_name = {d.name: d for d in corpus}
        if not by_name or not query.strip():
            return []
        ranked = self._index.lexical(query, limit=self._candidates, **_scope(by_name))
        return _resolve(ranked, by_name, limit)


class IndexedEmbeddingRetriever:
    """Cosine similarity over the persisted vectors, ranking a registry's derivations.

    The query is embedded per call; the documents were embedded once at index time. That
    asymmetry is the reason the index is worth having -- the in-memory retriever
    re-embeds the corpus on every search.

    An embedder that fails abstains rather than raising, leaving the lexical half to
    answer -- the degradation ``search: lexical`` selects deliberately.
    """

    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        *,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._candidates = candidates

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Rank ``corpus`` by similarity to the query vector; abstain on no hit."""
        by_name = {d.name: d for d in corpus}
        if not by_name or not query.strip():
            return []
        try:
            vector = self._embedder.embed([query])[0]
        except Exception:
            logger.warning(
                "could not embed the query; ranking lexically", exc_info=True
            )
            return []
        ranked = self._index.dense(vector, limit=self._candidates, **_scope(by_name))
        return _resolve(ranked, by_name, limit)


class WhenCaughtUp:
    """Ranks through the index where it covers the corpus, in memory where it does not.

    Two failures this stands between, both of which shipped or nearly did:

    *The index nobody filled.* It was wired to every consumer and nothing called a
    build, so search returned nothing -- and every test passed, because each had put its
    own documents in.

    *The index that has not caught up.* The registry ranked whatever was registered at
    the moment of the call, so a derivation was findable the instant it was certified;
    the index refreshes hourly. Coverage is therefore checked per call against the
    corpus passed in, not once against a corpus-wide count -- ``audit_event`` alone
    would make that count non-zero and latch it forever.

    The in-memory ranker also answers when the index *errors*: a broken index should
    cost ranking quality, not the request.
    """

    def __init__(
        self, index: SearchIndex, indexed: Retriever, memory: Retriever
    ) -> None:
        self._index = index
        self._indexed = indexed
        self._memory = memory

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Rank with the index where it covers ``corpus``, else in memory."""
        docs = list(corpus)
        try:
            if self._covers(docs):
                return self._indexed.search(query, docs, limit=limit)
            logger.debug("search index has not caught up with the registry; in memory")
        except Exception:
            logger.warning("search index unreadable; ranking in memory", exc_info=True)
        return self._memory.search(query, docs, limit=limit)

    def _covers(self, docs: Sequence[Derivation]) -> bool:
        """Whether every derivation in ``docs`` is in the index.

        All or nothing rather than per-derivation: mixing the two rankers would put two
        score scales in one list.
        """
        return bool(docs) and not (
            {d.name for d in docs} - self._index.entity_ids(DERIVATION)
        )


def indexed_retriever(
    index: SearchIndex, embedder: Embedder | None = None
) -> Retriever:
    """The hybrid a ``Registry`` should rank with when an index is available.

    Fused by the same :class:`~elbi_core.retrieval.HybridRetriever` and the same
    rank-only constant as the in-memory pair. With no embedder it is the lexical half
    alone (``search: lexical``) and still fuses, so the two configurations differ in
    what they retrieve rather than in how results combine.

    Wrapped in :class:`WhenCaughtUp`, so a registry pointed here before or ahead of a
    build ranks the way it would have anyway.
    """
    retrievers: list[Retriever] = [IndexedBm25Retriever(index)]
    memory: list[Retriever] = [Bm25Retriever()]
    if embedder is not None:
        retrievers.append(IndexedEmbeddingRetriever(index, embedder))
        memory.append(
            EmbeddingRetriever(embedder, min_similarity=DEFAULT_MIN_SIMILARITY)
        )
    return WhenCaughtUp(index, HybridRetriever(*retrievers), HybridRetriever(*memory))
