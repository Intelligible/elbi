"""Retrieval over the derivation library.

An agent finds derivations by searching, not by scanning every served tool, so
the quality of that search sets how well selection holds up as a project grows
from a handful of derivations to hundreds. Retrieval is a pluggable seam, like
the cache store and the executor: :class:`Registry` ranks with whatever
:class:`Retriever` it is given.

Three retrievers compose into the standard hybrid design, which is the default:

* :class:`Bm25Retriever`. Okapi BM25 over a derivation's name and description, with
  the name weighted above the description. Lexical, exact, and deterministic, with no
  index to maintain. Strong whenever the query shares words with a derivation, which
  is the common case. Set ``search: lexical`` to run this alone.
* :class:`EmbeddingRetriever`, dense semantic search by cosine similarity over an
  :class:`Embedder`'s vectors. It matches on meaning, so "income by area" can find
  ``revenue_by_region`` where lexical overlap is zero.
* :class:`HybridRetriever`, which fuses any set of retrievers with Reciprocal Rank
  Fusion. RRF combines rankings rather than scores, so a sparse BM25 score and a
  dense cosine score (on entirely different scales) merge without normalization. This
  is what :func:`default_retriever` builds: the robust default, since a query that
  shares words and one that only shares meaning both resolve.

The semantic half needs an :class:`Embedder`; :class:`Model2VecEmbedder` provides one
from static embeddings and is a core dependency, so hybrid works with no extra to
install. Any other embedding model fits the same protocol.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from .text import (
    BM25_B,
    BM25_K1,
    DESCRIPTION_BOOST,
    NAME_BOOST,
    RRF_K,
    rrf_fuse,
)
from .text import (
    tokens as _tokens,
)

if TYPE_CHECKING:
    from .derivation import Derivation

#: The floor below which a semantic hit is dropped, so fusion adds meaning matches
#: without noise. Tuned to :class:`OnnxEmbedder`, whose similarities sit in a high,
#: narrow band; it will not transfer to another model. Shared so the in-memory and SQL
#: dense paths agree.
DEFAULT_MIN_SIMILARITY: Final = 0.5


@runtime_checkable
class Retriever(Protocol):
    """Ranks derivations by relevance to a query.

    ``search`` returns the best matches from ``corpus``, most relevant first, at
    most ``limit`` of them. A retriever abstains on a derivation it finds no signal
    for (it is omitted) rather than returning everything in arbitrary order, so an
    empty result is meaningful. Ordering is deterministic: ties break by name.
    """

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Return up to ``limit`` matches from ``corpus``, most relevant first."""
        ...


class Embedder(Protocol):
    """Maps texts to dense vectors for semantic retrieval.

    ``embed`` returns one vector per input text, in order. Vectors need not be
    normalized; :class:`EmbeddingRetriever` compares them by cosine similarity.
    """

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """Return one vector per text in ``texts``, in order."""
        ...


@runtime_checkable
class SpanEmbedder(Embedder, Protocol):
    """An :class:`Embedder` that can also embed a span *in the context of its document*.

    Inherits :class:`Embedder` so the ``isinstance`` check below enforces what this
    docstring advertises: a caller that selects on it goes on to call ``embed`` for the
    spans no token covered, and would otherwise get an object that cannot.

    Its own protocol rather than a method on :class:`Embedder`, because only a
    contextual encoder can do it. A static model assigns each token one fixed vector
    regardless of what surrounds it, so pooling a span of one gives the same answer as
    embedding the span alone -- there is no context to carry. Widening ``Embedder``
    would make :class:`Model2VecEmbedder` claim a capability it cannot have; a separate
    protocol lets a caller check with ``isinstance`` and fall back honestly.

    This is "late chunking": encode the whole document once, then pool the token vectors
    belonging to each span. A chunk that reads "then we filtered to Q3" keeps the fact
    that its document is about churn, which embedding the chunk alone throws away.
    """

    def embed_spans(
        self, text: str, spans: Sequence[tuple[int, int]]
    ) -> list[Sequence[float]]:
        """One vector per ``(start, end)`` span, pooled over the whole text."""
        ...


def default_retriever() -> Retriever:
    """The retriever a :class:`Registry` uses unless given another: hybrid.

    BM25 fused with semantic embeddings by RRF, so both a query that shares words and
    one that only shares meaning resolve. The embedder loads lazily on the first search,
    so constructing this (which every :class:`Registry` does) stays cheap and downloads
    no model until a search actually runs. Pass a bare :class:`Bm25Retriever` (``search:
    lexical``) for a purely lexical, offline ranker.
    """
    semantic = EmbeddingRetriever(OnnxEmbedder(), min_similarity=DEFAULT_MIN_SIMILARITY)
    return HybridRetriever(Bm25Retriever(), semantic)


def _materialize(corpus: Iterable[Derivation]) -> list[Derivation]:
    return list(corpus)


class Bm25Retriever:
    """Okapi BM25 over a derivation's name and description.

    BM25 weighs terms three ways that matter as a library grows: rare query terms
    weigh more than common ones (inverse document frequency), repeated terms
    saturate rather than dominate, and a long description does not outscore a
    focused one (length normalization). The name and description are scored as
    separate fields (BM25F) with the name boosted, so a name hit outranks a
    description-only hit, the property selection relies on.

    Corpus statistics are computed per call over the candidates passed in, so the
    ranker needs no index and stays correct as derivations are added, certified, or
    removed at runtime. The candidate set is small (a project's derivations), so
    this is cheap.

    Args:
        k1: Term-frequency saturation. The BM25 standard 1.5; higher lets repeated
            terms contribute longer before saturating.
        b: Length-normalization strength in [0, 1]. The standard 0.75.
        name_boost: Weight on the name field relative to the description.
    """

    def __init__(
        self, *, k1: float = BM25_K1, b: float = BM25_B, name_boost: float = NAME_BOOST
    ) -> None:
        self.k1 = k1
        self.b = b
        self.name_boost = name_boost
        self.description_boost = DESCRIPTION_BOOST

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Rank ``corpus`` by BM25F over name and description; abstain on no hit."""
        terms = set(_tokens(query))
        docs = _materialize(corpus)
        if not terms or not docs:
            return []

        names = [_tokens(d.name) for d in docs]
        descriptions = [_tokens(d.description or "") for d in docs]
        n_docs = len(docs)
        avg_name = sum(len(t) for t in names) / n_docs
        avg_description = sum(len(t) for t in descriptions) / n_docs

        document_frequency: Counter[str] = Counter()
        for name_tokens, description_tokens in zip(names, descriptions, strict=True):
            for token in set(name_tokens) | set(description_tokens):
                document_frequency[token] += 1

        scored: list[tuple[float, str, Derivation]] = []
        for index, derivation in enumerate(docs):
            name_counts = Counter(names[index])
            description_counts = Counter(descriptions[index])
            score = 0.0
            for term in terms:
                df = document_frequency.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                weighted_tf = self._weighted_tf(
                    term,
                    name_counts,
                    description_counts,
                    len(names[index]),
                    len(descriptions[index]),
                    avg_name,
                    avg_description,
                )
                if weighted_tf > 0:
                    score += idf * weighted_tf / (self.k1 + weighted_tf)
            if score > 0:
                scored.append((score, derivation.name, derivation))

        scored.sort(key=lambda row: (-row[0], row[1]))
        return [derivation for _, _, derivation in scored[:limit]]

    def _weighted_tf(
        self,
        term: str,
        name_counts: Counter[str],
        description_counts: Counter[str],
        name_length: int,
        description_length: int,
        avg_name: float,
        avg_description: float,
    ) -> float:
        """BM25F field-weighted term frequency, summed over name and description."""
        weighted = 0.0
        if avg_name > 0 and name_counts[term]:
            norm = 1 - self.b + self.b * name_length / avg_name
            weighted += self.name_boost * name_counts[term] / norm
        if avg_description > 0 and description_counts[term]:
            norm = 1 - self.b + self.b * description_length / avg_description
            weighted += self.description_boost * description_counts[term] / norm
        return weighted


def _searchable_text(derivation: Derivation) -> str:
    """The text an embedder sees for a derivation: readable name plus description.

    Underscores in the name become spaces so ``price_by_zipcode`` reads as words
    the model can embed, rather than one unknown token.
    """
    readable_name = derivation.name.replace("_", " ")
    description = derivation.description or ""
    return f"{readable_name}. {description}".strip()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingRetriever:
    """Dense semantic retrieval by cosine similarity over an :class:`Embedder`.

    Matches on meaning rather than shared words, so it finds a derivation whose
    wording differs from the query. Document vectors are cached by their text, so a
    derivation is embedded once and re-embedded only if its name or description
    changes; each query embeds just the query string and compares it to the cached
    vectors. A brute-force comparison is ample for a project's library (no vector
    index needed at this scale).

    Args:
        embedder: The model that turns text into vectors.
        min_similarity: Drop candidates at or below this cosine similarity, so
            clearly unrelated derivations are not returned. The default 0.0 keeps
            only positively-similar ones.
    """

    def __init__(self, embedder: Embedder, *, min_similarity: float = 0.0) -> None:
        self._embedder = embedder
        self._min_similarity = min_similarity
        # Vectors memoized by text; text alone is a sufficient key because the
        # cache is in-memory and bound to this retriever's single embedder.
        self._cache: dict[str, list[float]] = {}

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Rank ``corpus`` by cosine similarity of embeddings to the query."""
        docs = _materialize(corpus)
        if not query.strip() or not docs:
            return []

        texts = [_searchable_text(derivation) for derivation in docs]
        self._ensure_cached(texts)
        query_vector = list(self._embedder.embed([query])[0])

        scored: list[tuple[float, str, Derivation]] = []
        for derivation, text in zip(docs, texts, strict=True):
            similarity = _cosine(query_vector, self._cache[text])
            if similarity > self._min_similarity:
                scored.append((similarity, derivation.name, derivation))

        scored.sort(key=lambda row: (-row[0], row[1]))
        return [derivation for _, _, derivation in scored[:limit]]

    def _ensure_cached(self, texts: Sequence[str]) -> None:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if not missing:
            return
        for text, vector in zip(missing, self._embedder.embed(missing), strict=True):
            self._cache[text] = list(vector)


class HybridRetriever:
    """Fuse several retrievers with Reciprocal Rank Fusion.

    RRF scores a derivation by the sum of ``1 / (k + rank)`` over each retriever
    that ranked it. It uses ranks, not raw scores, so a BM25 score and a cosine
    similarity (unbounded and [-1, 1] respectively, on unrelated scales) combine
    without any normalization or hand-tuned weights. A derivation that both a
    lexical and a semantic retriever rank highly rises to the top; one found by
    only one of them still surfaces. This is the standard, robust hybrid of exact
    and semantic search.

    Args:
        retrievers: The retrievers to fuse, e.g. a :class:`Bm25Retriever` and an
            :class:`EmbeddingRetriever`. Fusion helps most when they disagree, so
            pass genuinely different rankers.
        k: The RRF smoothing constant. The literature standard is 60; larger
            flattens the contribution of rank position.
    """

    def __init__(self, *retrievers: Retriever, k: int = RRF_K) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = retrievers
        self._k = k

    def search(
        self, query: str, corpus: Iterable[Derivation], *, limit: int
    ) -> list[Derivation]:
        """Fuse each retriever's ranking with RRF; return the top ``limit``."""
        docs = _materialize(corpus)
        if not docs:
            return []

        by_name: dict[str, Derivation] = {}
        rankings: list[list[str]] = []
        for retriever in self._retrievers:
            ranked = retriever.search(query, docs, limit=len(docs))
            by_name.update({derivation.name: derivation for derivation in ranked})
            rankings.append([derivation.name for derivation in ranked])

        # The same fusion the persisted index uses, from one definition: a query must
        # not rank differently depending on which surface asked.
        fused = rrf_fuse(*rankings, k=self._k)
        return [by_name[name] for name in fused[:limit]]


class OnnxEmbedder:
    """An :class:`Embedder` backed by a transformer graph run through onnxruntime.

    A contextual encoder reads word order, so it matches a query against wording it has
    never seen: "how many users cancelled" against "churn rate by cohort". A static
    model, which sums one fixed vector per token, cannot: that is the gap this closes,
    at a few milliseconds per query on CPU and without PyTorch.

    Vectors are pooled from the leading classification token and L2-normalized, the
    convention the default model was trained under. Normalizing also makes squared
    Euclidean distance rank identically to cosine, so these vectors drop into a
    vector index unchanged.

    Text longer than the model's window is truncated rather than split, leaving
    chunking to the caller that knows the document's structure.

    The tokenizer and graph load lazily on the first :meth:`embed`, so building the
    default retriever (which every :class:`Registry` does) downloads nothing until a
    search actually runs.

    Args:
        model: A Hugging Face repo id, or a local directory holding the ONNX graph
            and ``tokenizer.json``. A directory reads no network, which is the path
            an air-gapped install takes.
        onnx_file: Location of the graph within the repo or directory.
        max_tokens: Length above which input is truncated.
        batch_size: Texts encoded per forward pass. Each batch pads to its own
            longest text, so batching bounds the padding a single long document
            imposes on everything queued behind it.
    """

    #: Retrieval-tuned, MIT-licensed, 33 M parameters over 384 dimensions.
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_ONNX_FILE = "onnx/model.onnx"
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model: str | None = None,
        *,
        onnx_file: str | None = None,
        max_tokens: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self._onnx_file = onnx_file or self.DEFAULT_ONNX_FILE
        self._max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self._session: Any = None  # loaded on first embed (see _load)
        self._tokenizer: Any = None
        self._input_names: frozenset[str] = frozenset()

    def _resolve(self) -> tuple[str, str]:
        """Return local paths to the ONNX graph and tokenizer, downloading if needed."""
        from pathlib import Path

        directory = Path(self._model_name)
        if directory.is_dir():
            return str(directory / self._onnx_file), str(directory / "tokenizer.json")

        from huggingface_hub import hf_hub_download

        return (
            hf_hub_download(self._model_name, self._onnx_file),
            hf_hub_download(self._model_name, "tokenizer.json"),
        )

    def _load(self) -> tuple[Any, Any]:
        """Load and memoize the tokenizer and inference session on first use."""
        if self._session is None:
            try:
                import onnxruntime
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - both are core deps
                raise ImportError(
                    "semantic search needs onnxruntime and tokenizers, core "
                    "dependencies of elbi; reinstall the package to "
                    "restore them."
                ) from exc

            onnx_path, tokenizer_path = self._resolve()
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=self._max_tokens)
            tokenizer.enable_padding()
            session = onnxruntime.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._input_names = frozenset(i.name for i in session.get_inputs())
            self._tokenizer = tokenizer
            self._session = session
        return self._session, self._tokenizer

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """Embed ``texts`` into normalized contextual vectors."""
        if not texts:
            return []

        import numpy as np

        session, tokenizer = self._load()
        vectors: list[Sequence[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            # Some exports of the same architecture omit this input entirely.
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)

            pooled = session.run(None, feed)[0][:, 0]
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            # A degenerate all-zero row would divide by zero; leave it at zero, where
            # cosine similarity reads it as related to nothing.
            normalized = np.divide(
                pooled, norms, out=np.zeros_like(pooled), where=norms > 0
            )
            vectors.extend([float(value) for value in row] for row in normalized)
        return vectors

    def embed_spans(
        self, text: str, spans: Sequence[tuple[int, int]]
    ) -> list[Sequence[float]]:
        """One vector per character span, pooled over the whole ``text``'s tokens.

        Encodes ``text`` once and mean-pools the token vectors whose characters fall in
        each span, so a chunk carries what surrounded it rather than only what it says.

        Two things a caller should know:

        A span past the model's window gets no tokens, because the encoder truncates at
        ``max_tokens``. Those fall back to embedding the span's own text, which is what
        chunking without context would have produced anyway -- correct, just no better.

        The pooling differs from :meth:`embed`, which takes the leading classification
        token as the default model was trained to. A span has no such token, so this
        means the two are not the same projection, and a mean-pooled document vector
        compared against a CLS-pooled query is a mismatch the eval harness should
        measure rather than this docstring assume away.
        """
        if not spans:
            return []

        import numpy as np

        session, tokenizer = self._load()
        encoded = tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        # The full sequence, not [:, 0]: these are the token vectors embed() discards.
        tokens = session.run(None, feed)[0][0]
        offsets = encoded.offsets
        attention = encoded.attention_mask

        out: list[Sequence[float]] = []
        uncovered: list[int] = []
        for index, (start, end) in enumerate(spans):
            rows = [
                position
                for position, (token_start, token_end) in enumerate(offsets)
                if attention[position] and token_end > start and token_start < end
            ]
            if not rows:
                uncovered.append(index)
                out.append([])
                continue
            pooled = tokens[rows].mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            out.append(
                [float(v) / norm for v in pooled] if norm > 0 else [0.0] * len(pooled)
            )

        if uncovered:
            fallback = self.embed([text[spans[i][0] : spans[i][1]] for i in uncovered])
            for index, vector in zip(uncovered, fallback, strict=True):
                out[index] = vector
        return out


class Model2VecEmbedder:
    """An :class:`Embedder` backed by model2vec static embeddings.

    A static model assigns each token one fixed vector and sums them, so it reads no
    word order and matches paraphrases less reliably than :class:`OnnxEmbedder`. What
    it buys is a quarter of the weights and inference fast enough to be free: worth
    trading accuracy for where a corpus is large and queries are heavy, or where
    onnxruntime cannot be installed.

    The model name is recorded at construction but the model itself loads lazily on
    the first :meth:`embed`, so nothing is downloaded until an embed actually runs.
    The model is downloaded once on first use and cached by model2vec thereafter.

    Args:
        model: A model2vec model name or local path. Defaults to a small
            retrieval-tuned static embedding model.
    """

    #: model2vec's retrieval-tuned static model (~32 MB), a better fit for search
    #: than the general-purpose models.
    DEFAULT_MODEL = "minishlab/potion-retrieval-32M"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self._model: Any = None  # loaded on first embed (see _load)

    def _load(self) -> Any:
        """Load and memoize the static model on first use."""
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as exc:  # pragma: no cover - model2vec is a core dep
                raise ImportError(
                    "semantic search needs model2vec, a core dependency of "
                    "elbi; reinstall the package to restore it."
                ) from exc
            self._model = StaticModel.from_pretrained(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """Embed ``texts`` into static-embedding vectors."""
        vectors = self._load().encode(list(texts))
        return [[float(value) for value in vector] for vector in vectors]
