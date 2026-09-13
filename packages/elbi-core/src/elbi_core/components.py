"""Validating and stamping OpenReasoningComponents (ORC) components.

ORC is somebody else's standard (like OSI, see :mod:`elbi_core.metrics.osi`): its
schema is vendored here for validation but not published under ``spec/`` (see
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
from typing import Any, Protocol, runtime_checkable

import jsonschema

from .errors import ComponentError
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


def stamp_provenance(
    component: dict[str, Any], *, derivation: str, derivation_version: str
) -> dict[str, Any]:
    """Attach ``derivation``/``derivation_version`` provenance to a component.

    Only fills in fields the author hasn't already set: a hand-authored
    ``domain_knowledge`` component with real human provenance (``source: "human"``,
    an ``author``) is left alone. A component that already names a *different*
    ``derivation`` (e.g. authored against another system's computation) is also
    left alone -- this only fills gaps, it never overwrites.

    Returns a new dict; the input is not mutated.
    """
    provenance = dict(component.get("provenance") or {})
    provenance.setdefault("derivation", derivation)
    provenance.setdefault("derivation_version", derivation_version)
    return {**component, "provenance": provenance}


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to dense vectors.

    The same protocol as :class:`elbi_core.retrieval.Embedder`; any implementation
    of one satisfies the other structurally, so an already-constructed embedder
    (e.g. the ``OnnxEmbedder`` a server already loads for derivation search) can be
    reused here as-is.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text in ``texts``, in order."""
        ...


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

    This is a thin, in-memory, rebuild-every-call index: it holds no state of its
    own between calls and does no chunking. It exists to make a ``components``
    artifact searchable at all; the persisted, chunked index under ``elbi.search``
    (already used for a derivation's own fields) is the right home once components
    need to be indexed at real scale, incrementally, across restarts.
    """
    docs = list(components)
    if not query.strip() or not docs:
        return []

    by_id = {
        component.get("id", str(index)): component
        for index, component in enumerate(docs)
    }
    bm25_ranking = _bm25_rank(query, docs)
    if embedder is None:
        return [by_id[cid] for cid in bm25_ranking[:limit]]

    embedding_ranking = _embedding_rank(query, docs, embedder)
    fused = rrf_fuse(bm25_ranking, embedding_ranking)
    return [by_id[cid] for cid in fused[:limit]]


def _component_id(component: dict[str, Any], index: int) -> str:
    return str(component.get("id", index))


def _bm25_rank(query: str, docs: Sequence[dict[str, Any]]) -> list[str]:
    """Okapi BM25 over each component's ``statement`` alone.

    There is only the one text field here, unlike a derivation's separately
    weighted name and description.
    """
    terms = set(_tokens(query))
    if not terms:
        return []

    ids = [_component_id(doc, index) for index, doc in enumerate(docs)]
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
    query: str, docs: Sequence[dict[str, Any]], embedder: Embedder
) -> list[str]:
    ids = [_component_id(doc, index) for index, doc in enumerate(docs)]
    texts = [doc.get("statement", "") for doc in docs]
    vectors = embedder.embed(texts)
    query_vector = embedder.embed([query])[0]

    scored = [
        (_cosine(query_vector, vector), cid)
        for cid, vector in zip(ids, vectors, strict=True)
    ]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [cid for _, cid in scored]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
