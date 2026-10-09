"""Validating and stamping OpenReasoningComponents (ORC) components.

ORC is a separate standard, vendored here the way OSI is (see
:mod:`elbi_core.metrics.osi`): ``spec/component.schema.json`` is a verbatim copy of
the schema at commit da37293 of
https://github.com/OpenReasoningComponents/OpenReasoningComponents (no release yet),
validated against but not published under ``spec/`` (see
``tests/test_spec_vendoring.py``). A ``components``-format derivation's artifact is
a plain list of ORC-shaped dicts the compute function returns by hand; this module
is what checks that shape and attaches the one thing ORC has no way to compute for
itself -- which of *this* system's versioned computations produced it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

from .errors import ComponentError
from .retrieval import CachedEmbedder, Embedder, cosine
from .text import BM25_B, BM25_K1, rrf_fuse
from .text import tokens as _tokens


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the bundled ORC component JSON Schema as a dict."""
    resource = files("elbi_core") / "spec" / "component.schema.json"
    data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return data


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_component(component: dict[str, Any]) -> None:
    """Validate one component against ORC's schema.

    Raises:
        ComponentError: if the component does not conform. The error carries a
            list of human-readable messages, one per violation.
    """
    errors = sorted(_validator().iter_errors(component), key=lambda e: list(e.path))
    if errors:
        messages = [_format_error(error) for error in errors]
        joined = "\n  - ".join(messages)
        raise ComponentError(f"component failed ORC validation:\n  - {joined}")


def is_valid_component(component: dict[str, Any]) -> bool:
    """Return whether a component conforms to ORC's schema, without raising."""
    return bool(_validator().is_valid(component))


def validate_components(value: object) -> list[dict[str, Any]]:
    """Validate a whole ``components`` artifact value and return it as a list.

    Raises:
        ComponentError: if the value is not a list, an item fails ORC validation
            (reported with its position), or two items share an ``id``.
    """
    if not isinstance(value, list):
        raise ComponentError(
            f"components artifact must be a list, got {type(value).__name__}"
        )
    seen: set[str] = set()
    for index, item in enumerate(value):
        try:
            validate_component(item)
        except ComponentError as exc:
            raise ComponentError(f"item {index}: {exc}") from None
        if item["id"] in seen:
            raise ComponentError(f"item {index}: duplicate id {item['id']!r}")
        seen.add(item["id"])
    return value


def stamp_provenance(
    component: dict[str, Any], *, derivation: str, derivation_version: str
) -> dict[str, Any]:
    """Set ``provenance.derivation``/``derivation_version`` on a component.

    Always this derivation and this version: whatever code emitted the item, the
    computation being served is its producer, and a value the author (or an
    upstream system) wrote there would let a component claim a producer, or a
    freshness, it does not have. Every other provenance field (``source``,
    ``author``, ``method``, ...) is the author's and is kept as is.

    Returns a new dict; the input is not mutated.
    """
    provenance = dict(component.get("provenance") or {})
    provenance["derivation"] = derivation
    provenance["derivation_version"] = derivation_version
    return {**component, "provenance": provenance}


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def search_components(
    query: str,
    components: Sequence[dict[str, Any]],
    *,
    limit: int = 5,
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Rank components by relevance to ``query`` over their ``statement`` text.

    BM25 alone when no ``embedder`` is given; fused with dense cosine similarity by
    Reciprocal Rank Fusion when one is (the same fusion
    :mod:`elbi_core.retrieval` uses for derivation search), so a query that shares
    no words with any statement can still surface the right component.

    Results are keyed by position, not ``id``, so two derivations that serve the
    same ``id`` both surface. The ranking is rebuilt every call and does no
    chunking; pass a :class:`~elbi_core.retrieval.CachedEmbedder` to keep statement
    vectors between calls, otherwise every statement is embedded each call. The
    persisted, chunked index under ``elbi.search`` (already used for a derivation's
    own fields) is the right home once components need to be indexed at real
    scale, incrementally, across restarts.
    """
    docs = list(components)
    if not query.strip() or not docs:
        return []

    bm25_ranking = _bm25_rank(query, docs)
    if embedder is None:
        return [docs[int(key)] for key in bm25_ranking[:limit]]

    cached = (
        embedder if isinstance(embedder, CachedEmbedder) else CachedEmbedder(embedder)
    )
    embedding_ranking = _embedding_rank(query, docs, cached)
    fused = rrf_fuse(bm25_ranking, embedding_ranking)
    return [docs[int(key)] for key in fused[:limit]]


def _bm25_rank(query: str, docs: Sequence[dict[str, Any]]) -> list[str]:
    """Okapi BM25 over each component's ``statement`` alone.

    There is only the one text field here, unlike a derivation's separately
    weighted name and description.
    """
    terms = set(_tokens(query))
    if not terms:
        return []

    ids = [str(index) for index in range(len(docs))]
    statement_tokens = [_tokens(doc.get("statement", "")) for doc in docs]
    n_docs = len(docs)
    avg_length = sum(len(toks) for toks in statement_tokens) / n_docs

    document_frequency: Counter[str] = Counter()
    for toks in statement_tokens:
        for token in set(toks):
            document_frequency[token] += 1

    scored: list[tuple[float, str]] = []
    for cid, toks in zip(ids, statement_tokens, strict=True):
        counts = Counter(toks)
        length = len(toks)
        score = 0.0
        for term in terms:
            df = document_frequency.get(term, 0)
            if df == 0 or not counts[term]:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            norm = 1 - BM25_B + BM25_B * length / avg_length if avg_length else 1.0
            weighted_tf = counts[term] / norm
            score += idf * weighted_tf / (BM25_K1 + weighted_tf)
        if score > 0:
            scored.append((score, cid))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [cid for _, cid in scored]


def _embedding_rank(
    query: str, docs: Sequence[dict[str, Any]], embedder: CachedEmbedder
) -> list[str]:
    ids = [str(index) for index in range(len(docs))]
    texts = [doc.get("statement", "") for doc in docs]
    vectors = embedder.embed(texts)
    query_vector = embedder.embed_query(query)

    scored = [
        (cosine(query_vector, vector), cid)
        for cid, vector in zip(ids, vectors, strict=True)
    ]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [cid for _, cid in scored]
