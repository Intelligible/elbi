"""The text primitives ranking shares: tokenization, BM25 constants, and RRF.

These live apart from :mod:`elbi_core.retrieval` because two rankers consume
them. The in-memory :class:`~elbi_core.retrieval.Bm25Retriever` recomputes its
corpus statistics per call, which is right for a project's derivations; a persisted
index over every artifact instead stores them and scores in SQL. Both have to tokenize
identically and weight fields identically, or the same query ranks differently depending
on which surface asked, so the tokenizer and the constants have exactly one definition
and both rankers read it.

Nothing here touches a corpus or a document type, so it stays free of the derivation
model and of any storage engine.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

#: Tokens are runs of lowercase letters and digits, so ``price_by_zipcode`` splits into
#: words rather than reading as one unknown token.
WORD_RE: Final = re.compile(r"[a-z0-9]+")

#: Bumped when :func:`tokens` or :func:`fold` changes what a document yields. A stored
#: index records it and rebuilds on a mismatch, since postings written by one tokenizer
#: are not comparable with queries tokenized by another.
TOKENIZER_VERSION: Final = 1

#: Term-frequency saturation. The BM25 standard 1.5; higher lets repeated terms
#: contribute for longer before saturating.
BM25_K1: Final = 1.5

#: Length-normalization strength in [0, 1]. The standard 0.75.
BM25_B: Final = 0.75

#: Weight on a name or title field. A name hit outranks a description-only hit, which
#: is the property selection relies on.
NAME_BOOST: Final = 2.0

#: Weight on a description or body field, the baseline the name is boosted against.
DESCRIPTION_BOOST: Final = 1.0

#: The RRF smoothing constant. The literature standard is 60; larger flattens the
#: contribution of rank position.
RRF_K: Final = 60


def fold(token: str) -> str:
    """Fold the common English plural of ``token`` to its singular.

    Light, deterministic normalization so "homes" matches "home" and "prices" matches
    "price". It is plural folding only, not a full stemmer: morphological pairs like
    "renovated"/"renovation" are left to the semantic retriever, which matches them on
    meaning. Short tokens are left untouched to avoid mangling.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    """Plural-folded word tokens of ``text`` (also splits snake_case names)."""
    return [fold(word) for word in WORD_RE.findall(text.lower())]


def rrf_fuse(*rankings: Sequence[str], k: int = RRF_K) -> list[str]:
    """Fuse ranked id lists with Reciprocal Rank Fusion, best first.

    Scores an id by the sum of ``1 / (k + rank)`` over each ranking that contains it.
    Ranks rather than scores, so a BM25 score and a cosine similarity (unbounded and
    [-1, 1] respectively, on unrelated scales) combine without normalization or
    hand-tuned weights. An id both rankings place highly rises to the top; one found by
    only a single ranking still surfaces.

    Ties break on the id itself, so the result is deterministic and independent of the
    order the rankings were passed in.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: (-scores[item], item))
