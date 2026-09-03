"""Retrieval tests, at the behavior boundary (rank derivations for a query).

These exercise BM25, the semantic retriever (via a deterministic stub embedder so
the test needs no model and no network), and the RRF hybrid. The properties are
the ones selection depends on: a name hit beats a description-only hit, rare terms
discriminate, a synonym is found by meaning, and fusion surfaces both lexical and
semantic matches. The accompanying ``test_retrieval_catches_bugs`` documents how
each was checked to actually fail on the obvious regression.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elbi_core import serve
from elbi_core.derivation import Derivation
from elbi_core.retrieval import (
    Bm25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
    OnnxEmbedder,
    default_retriever,
)
from elbi_core.text import fold


def _derivation(name: str, description: str | None = None) -> Derivation:
    return Derivation(
        name=name,
        compute=lambda ctx: None,
        serve=serve.text(),
        description=description,
    )


CORPUS = [
    _derivation("revenue_by_region", "Total revenue grouped by sales region."),
    _derivation("churn_risk", "Per-customer churn risk scores."),
    _derivation("price_by_zipcode", "Median home price for each zip code."),
    _derivation("grade_price_effect", "Effect of construction grade on price."),
    _derivation("inventory_levels", "Units in stock per warehouse."),
]


# --- BM25 (the lexical default) -------------------------------------------


def test_bm25_ranks_name_match_above_description_match() -> None:
    # "region" is in revenue_by_region's name AND description; only in churn? no.
    bm25 = Bm25Retriever()
    results = bm25.search("region", CORPUS, limit=5)
    assert results[0].name == "revenue_by_region"


def test_bm25_name_field_outranks_description_only_field() -> None:
    # Hold both fields to equal length (two tokens each) so the only thing that can
    # separate a name hit from a description hit is the name-field boost. The
    # name-match doc ("price_z") is named to LOSE an unboosted tie to "a_plain", so
    # this passes only because the name field is genuinely weighted higher.
    bm25 = Bm25Retriever()
    corpus = [
        _derivation("a_plain", "price here"),  # term in description
        _derivation("price_z", "filler stuff"),  # term in name
    ]
    results = bm25.search("price", corpus, limit=5)
    assert results[0].name == "price_z"


def test_bm25_abstains_when_no_overlap() -> None:
    bm25 = Bm25Retriever()
    assert bm25.search("photosynthesis", CORPUS, limit=5) == []


def test_bm25_empty_query_returns_nothing() -> None:
    assert Bm25Retriever().search("   ", CORPUS, limit=5) == []


def test_bm25_respects_limit() -> None:
    bm25 = Bm25Retriever()
    assert len(bm25.search("price revenue churn", CORPUS, limit=1)) == 1


def test_bm25_rare_term_outweighs_common_term() -> None:
    # A term in every document carries almost no IDF; a term in one document
    # carries a lot. A doc matching only the rare term must beat docs matching only
    # the ubiquitous term. Field lengths are held equal (one description token
    # each) so the ranking turns purely on IDF, not length normalization, and the
    # rare doc is named to sort LAST so a broken (constant) IDF would lose the tie.
    bm25 = Bm25Retriever()
    corpus = [
        _derivation("aaaone", "alpha"),
        _derivation("aaatwo", "alpha"),
        _derivation("aaathree", "alpha"),
        _derivation("zzztarget", "beta"),
    ]
    results = bm25.search("alpha beta", corpus, limit=4)
    assert results[0].name == "zzztarget"


def test_bm25_empty_corpus_returns_nothing() -> None:
    # An empty candidate set must short-circuit, not divide by zero on avg length.
    assert Bm25Retriever().search("anything", [], limit=5) == []


def test_bm25_length_normalization_favors_the_focused_match() -> None:
    # The query term appears once in each description, but one is short and one is
    # padded. Length normalization ranks the focused (short) one higher. The short
    # doc is named to lose an unnormalized tie, so this passes only because b > 0.
    bm25 = Bm25Retriever()
    corpus = [
        _derivation("zfocused", "term"),
        _derivation("apadded", "term filler filler filler filler"),
    ]
    results = bm25.search("term", corpus, limit=2)
    assert results[0].name == "zfocused"


def test_bm25_is_deterministic_on_ties() -> None:
    # Two docs with identical searchable content must always order by name.
    bm25 = Bm25Retriever()
    corpus = [_derivation("b_dup", "same words"), _derivation("a_dup", "same words")]
    names = [d.name for d in bm25.search("same words", corpus, limit=2)]
    assert names == ["a_dup", "b_dup"]


@pytest.mark.parametrize(
    ("word", "folded"),
    [
        ("homes", "home"),
        ("prices", "price"),
        ("neighborhoods", "neighborhood"),
        ("companies", "company"),
        ("boxes", "box"),
        ("class", "class"),  # -ss is left alone
        ("ssh", "ssh"),  # too short to fold
    ],
)
def test_plural_folding(word: str, folded: str) -> None:
    assert fold(word) == folded


def test_bm25_matches_across_plural() -> None:
    bm25 = Bm25Retriever()
    corpus = [_derivation("home_value", "Average value of a single home.")]
    assert bm25.search("homes", corpus, limit=5)  # "homes" folds to "home"


@given(
    st.lists(
        st.tuples(
            st.text(alphabet="abcdefghij ", min_size=1, max_size=12),
            st.text(alphabet="abcdefghij ", min_size=0, max_size=20),
        ),
        min_size=1,
        max_size=8,
    ),
    st.text(alphabet="abcdefghij ", min_size=0, max_size=10),
    st.integers(min_value=1, max_value=10),
)
def test_bm25_never_exceeds_limit_or_invents_docs(
    docs: list[tuple[str, str]], query: str, limit: int
) -> None:
    corpus = [
        _derivation(f"d{i}_{name.replace(' ', '_') or 'x'}", desc)
        for i, (name, desc) in enumerate(docs)
    ]
    results = Bm25Retriever().search(query, corpus, limit=limit)
    assert len(results) <= limit
    assert all(r in corpus for r in results)
    # No duplicates in the ranking.
    assert len({r.name for r in results}) == len(results)


# --- Semantic retrieval (stub embedder, no model/network) ------------------


class _StubEmbedder:
    """Deterministic embedder over a tiny synonym lexicon.

    Each text is embedded as a bag of *concept* dimensions, where related words
    map to the same concept. This gives genuine cosine similarity between
    synonyms (income/revenue, area/region) without any model, so the semantic
    behavior can be tested deterministically and offline.
    """

    _CONCEPTS: ClassVar[dict[str, int]] = {
        "revenue": 0,
        "income": 0,
        "sales": 0,
        "region": 1,
        "area": 1,
        "zone": 1,
        "churn": 2,
        "attrition": 2,
        "price": 3,
        "cost": 3,
    }

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * (max(self._CONCEPTS.values()) + 1)
            for word in text.lower().replace("_", " ").split():
                concept = self._CONCEPTS.get(word.rstrip("s"))
                if concept is None:
                    concept = self._CONCEPTS.get(word)
                if concept is not None:
                    vector[concept] += 1.0
            vectors.append(vector)
        return vectors


def test_embedding_finds_synonym_that_lexical_misses() -> None:
    corpus = [_derivation("revenue_by_region", "Total revenue grouped by region.")]
    # "income per zone" shares NO tokens with the derivation: lexical abstains.
    assert Bm25Retriever().search("income per zone", corpus, limit=5) == []
    # Semantic matches on meaning (income~revenue, zone~region).
    semantic = EmbeddingRetriever(_StubEmbedder())
    results = semantic.search("income per zone", corpus, limit=5)
    assert [d.name for d in results] == ["revenue_by_region"]


def test_embedding_ranks_more_similar_first() -> None:
    # Two docs at different similarity to the query must come back most-similar
    # first; pins the sort direction (a single-doc test cannot).
    corpus = [
        _derivation("revenue_strong", "revenue revenue"),  # cosine 1.0 to "revenue"
        _derivation("revenue_mixed", "revenue region"),  # cosine ~0.71
    ]
    results = EmbeddingRetriever(_StubEmbedder()).search("revenue", corpus, limit=2)
    assert [d.name for d in results] == ["revenue_strong", "revenue_mixed"]


def test_embedding_caches_and_reembeds_only_changes() -> None:
    class _CountingEmbedder(_StubEmbedder):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
            self.calls.extend(texts)
            return super().embed(texts)

    embedder = _CountingEmbedder()
    retriever = EmbeddingRetriever(embedder)
    corpus = [_derivation("revenue_by_region", "revenue per region")]
    retriever.search("income", corpus, limit=5)
    retriever.search("income", corpus, limit=5)
    # The query is re-embedded each call; the document is embedded only once.
    doc_text = "revenue by region. revenue per region"
    assert embedder.calls.count(doc_text) == 1


def test_embedding_excludes_zero_similarity_docs() -> None:
    # A doc the embedder finds unrelated (zero vector, cosine 0) is dropped at the
    # default floor rather than returned as a weak match.
    corpus = [
        _derivation("revenue_match", "revenue"),
        _derivation("unrelated_zero", "zzz"),  # no known concept, zero vector
    ]
    results = EmbeddingRetriever(_StubEmbedder()).search("revenue", corpus, limit=5)
    assert [d.name for d in results] == ["revenue_match"]


def test_embedding_empty_query_returns_nothing() -> None:
    semantic = EmbeddingRetriever(_StubEmbedder())
    assert semantic.search("  ", CORPUS, limit=5) == []


# --- Hybrid (RRF fusion) ---------------------------------------------------


def test_hybrid_surfaces_both_lexical_and_semantic_hits() -> None:
    corpus = [
        _derivation("revenue_by_region", "Total revenue grouped by region."),
        _derivation("price_table", "A table of price points."),
    ]
    hybrid = HybridRetriever(Bm25Retriever(), EmbeddingRetriever(_StubEmbedder()))
    # "income price": lexical hits price_table on "price"; semantic hits
    # revenue_by_region on "income"~"revenue". Both must appear.
    names = {d.name for d in hybrid.search("income price", corpus, limit=5)}
    assert names == {"revenue_by_region", "price_table"}


def test_hybrid_ranks_mutual_top_first() -> None:
    # A derivation ranked #1 by both retrievers must be the fused #1.
    corpus = [
        _derivation("revenue_by_region", "Total revenue grouped by region."),
        _derivation("inventory_levels", "Units in stock per warehouse."),
    ]
    hybrid = HybridRetriever(Bm25Retriever(), EmbeddingRetriever(_StubEmbedder()))
    results = hybrid.search("revenue region", corpus, limit=5)
    assert results[0].name == "revenue_by_region"


def test_hybrid_orders_fused_results_by_score() -> None:
    # Both retrievers return both docs, with a clear fused winner. Pins the fusion
    # sort direction, which a single-result fused set cannot.
    corpus = [
        _derivation("revenue_by_region", "revenue grouped by region"),
        _derivation("aaa_region_only", "region zone area"),
    ]
    hybrid = HybridRetriever(Bm25Retriever(), EmbeddingRetriever(_StubEmbedder()))
    results = hybrid.search("revenue region", corpus, limit=2)
    assert results[0].name == "revenue_by_region"


def test_hybrid_fusion_is_order_independent() -> None:
    corpus = CORPUS
    bm25 = Bm25Retriever()
    semantic = EmbeddingRetriever(_StubEmbedder())
    one = HybridRetriever(bm25, semantic).search("revenue region", corpus, limit=5)
    two = HybridRetriever(semantic, bm25).search("revenue region", corpus, limit=5)
    assert [d.name for d in one] == [d.name for d in two]


def test_hybrid_requires_a_retriever() -> None:
    with pytest.raises(ValueError, match="at least one retriever"):
        HybridRetriever()


def test_hybrid_empty_corpus() -> None:
    hybrid = HybridRetriever(Bm25Retriever())
    assert hybrid.search("anything", [], limit=5) == []


# --- Model2VecEmbedder (with a fake model2vec, so no model/network) --------


def test_model2vec_embedder_encodes_via_static_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from elbi_core.retrieval import Model2VecEmbedder

    captured: dict[str, str] = {}

    class FakeStaticModel:
        @classmethod
        def from_pretrained(cls, name: str) -> FakeStaticModel:
            captured["model"] = name
            return cls()

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0] for text in texts]

    fake = types.ModuleType("model2vec")
    fake.StaticModel = FakeStaticModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "model2vec", fake)

    embedder = Model2VecEmbedder()
    assert captured == {}  # lazy: constructing loads no model
    assert embedder.embed(["abc", "de"]) == [[3.0, 1.0], [2.0, 1.0]]
    assert captured["model"] == Model2VecEmbedder.DEFAULT_MODEL  # loaded on first embed


# --- OnnxEmbedder (with a fake runtime, so no model/network) ---------------


def _fake_onnx_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    declares_token_type_ids: bool = True,
    hidden: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Install fake onnxruntime/tokenizers/huggingface_hub modules.

    The fake session returns a ``(batch, tokens, features)`` block whose leading
    token differs from the rest, so a test can tell classification-token pooling
    from mean pooling.
    """
    import sys
    import types

    import numpy as np

    seen: dict[str, Any] = {"downloads": [], "batches": [], "feeds": []}

    class FakeEncoding:
        #: The fake truncates here, as the real tokenizer does at ``max_tokens``. Its
        #: two tokens describe only this much of the text, so a span past it covers no
        #: token -- which is the only way to reach the uncovered-span fallback.
        WINDOW = 8

        def __init__(self, text: str) -> None:
            self.ids = [1, 2]
            self.attention_mask = [1, 1]
            # Two tokens covering the first and second halves of the *kept* text, so a
            # span test can say which token a character range belongs to.
            kept = min(len(text), self.WINDOW)
            half = max(kept // 2, 1)
            self.offsets = [(0, half), (half, max(kept, half))]

    class FakeTokenizer:
        @classmethod
        def from_file(cls, path: str) -> FakeTokenizer:
            seen["tokenizer_path"] = path
            return cls()

        def enable_truncation(self, max_length: int) -> None:
            seen["max_length"] = max_length

        def enable_padding(self) -> None:
            seen["padded"] = True

        def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
            seen["batches"].append(list(texts))
            return [FakeEncoding(text) for text in texts]

        def encode(self, text: str) -> FakeEncoding:
            seen["encoded"] = [*seen.get("encoded", []), text]
            return FakeEncoding(text)

    class FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            seen["onnx_path"] = path
            seen["providers"] = providers

        def get_inputs(self) -> list[FakeInput]:
            names = ["input_ids", "attention_mask"]
            if declares_token_type_ids:
                names.append("token_type_ids")
            return [FakeInput(name) for name in names]

        def run(self, _outputs: None, feed: dict[str, Any]) -> list[Any]:
            seen["feeds"].append(sorted(feed))
            rows = len(feed["input_ids"])
            lead = hidden or [[3.0, 4.0]]
            block = [[lead[index % len(lead)], [99.0, 99.0]] for index in range(rows)]
            return [np.array(block, dtype=np.float32)]

    runtime = types.ModuleType("onnxruntime")
    runtime.InferenceSession = FakeSession  # type: ignore[attr-defined]
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = FakeTokenizer  # type: ignore[attr-defined]
    hub = types.ModuleType("huggingface_hub")

    def fake_download(repo: str, filename: str) -> str:
        seen["downloads"].append((repo, filename))
        return f"/cache/{repo}/{filename}"

    hub.hf_hub_download = fake_download  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return seen


def test_onnx_embedder_pools_leading_token_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _fake_onnx_stack(monkeypatch)

    embedder = OnnxEmbedder()
    assert seen["downloads"] == []  # lazy: constructing downloads nothing

    vectors = embedder.embed(["anything"])

    # [3, 4] is the leading token; mean pooling would drag in the [99, 99] tail.
    assert vectors == [[pytest.approx(0.6), pytest.approx(0.8)]]
    assert seen["downloads"] == [
        (OnnxEmbedder.DEFAULT_MODEL, OnnxEmbedder.DEFAULT_ONNX_FILE),
        (OnnxEmbedder.DEFAULT_MODEL, "tokenizer.json"),
    ]
    assert seen["max_length"] == OnnxEmbedder.DEFAULT_MAX_TOKENS


def test_onnx_embedder_reads_a_local_directory_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The air-gapped path: weights on disk, so nothing may be downloaded.
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"")
    (tmp_path / "tokenizer.json").write_text("{}")
    seen = _fake_onnx_stack(monkeypatch)

    OnnxEmbedder(str(tmp_path)).embed(["anything"])

    assert seen["downloads"] == []
    assert seen["onnx_path"] == str(tmp_path / "onnx" / "model.onnx")
    assert seen["tokenizer_path"] == str(tmp_path / "tokenizer.json")


def test_onnx_embedder_batches_and_feeds_token_type_ids_only_when_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _fake_onnx_stack(monkeypatch, declares_token_type_ids=False)

    OnnxEmbedder(batch_size=2).embed(["a", "b", "c"])

    assert seen["batches"] == [["a", "b"], ["c"]]
    # An export without this input rejects it, so it must not be sent.
    assert seen["feeds"] == [
        ["attention_mask", "input_ids"],
        ["attention_mask", "input_ids"],
    ]


def test_onnx_embedder_survives_a_zero_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Normalizing a degenerate all-zero row must not divide by zero.
    _fake_onnx_stack(monkeypatch, hidden=[[0.0, 0.0]])

    assert OnnxEmbedder().embed(["anything"]) == [[0.0, 0.0]]


def test_onnx_embedder_embeds_nothing_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _fake_onnx_stack(monkeypatch)

    assert OnnxEmbedder().embed([]) == []
    assert seen["downloads"] == []


# --- the default retriever (hybrid, lazily loaded) -------------------------


def test_default_retriever_is_hybrid_bm25_plus_semantic() -> None:
    retriever = default_retriever()
    assert isinstance(retriever, HybridRetriever)
    kinds = {type(r) for r in retriever._retrievers}
    assert kinds == {Bm25Retriever, EmbeddingRetriever}


def test_default_retriever_loads_no_model_until_searched() -> None:
    # Every Registry builds this at construction (including at import), so it must not
    # download or load the embedding model until a search actually runs.
    retriever = default_retriever()
    semantic = next(
        r for r in retriever._retrievers if isinstance(r, EmbeddingRetriever)
    )
    embedder = semantic._embedder
    assert isinstance(embedder, OnnxEmbedder)
    assert embedder._session is None  # lazy: unloaded until the first embed


def test_model2vec_embedder_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from elbi_core.retrieval import Model2VecEmbedder

    # A None entry makes `import model2vec` raise ImportError. Construction stays lazy,
    # so the failure surfaces on the first embed, not on building the retriever.
    monkeypatch.setitem(sys.modules, "model2vec", None)
    embedder = Model2VecEmbedder()
    with pytest.raises(ImportError, match="model2vec"):
        embedder.embed(["x"])


# --- late chunking: spans pooled over their document -----------------------


def test_span_embedding_pools_the_tokens_a_span_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A span's vector comes from the whole document's encoding, not the span alone.

    The fake returns a distinct vector per token position, so which tokens a span
    pooled is visible in the answer. This is the property late chunking exists for:
    the document is encoded once and each span reads its own slice of that context.
    """
    seen = _fake_onnx_stack(monkeypatch)
    embedder = OnnxEmbedder()

    # "abcd" -> token 0 covers chars 0-2 ([3,4]), token 1 covers 2-4 ([99,99]).
    vectors = embedder.embed_spans("abcd", [(0, 2), (2, 4)])

    assert len(vectors) == 2
    assert vectors[0] == [pytest.approx(0.6), pytest.approx(0.8)]  # [3,4] normalized
    assert vectors[1] == [pytest.approx(0.7071068), pytest.approx(0.7071068)]
    # Encoded once for the document, not once per span.
    assert seen["encoded"] == ["abcd"]


def test_a_span_past_the_window_falls_back_to_embedding_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncation means a late span has no tokens; it must still get a vector.

    The encoder truncates at ``max_tokens``, so a span beyond that covers no token and
    cannot be pooled. Falling back to embedding the span's own text is what chunking
    without context would have produced -- correct, just no better.
    """
    _fake_onnx_stack(monkeypatch)
    embedder = OnnxEmbedder()

    # The fake truncates at 8 characters, so (90, 99) is past every offset.
    vectors = embedder.embed_spans("abcd" + "x" * 95, [(90, 99)])

    assert len(vectors) == 1
    assert vectors[0], "a span past the window must still get a vector"


def test_embedding_no_spans_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_onnx_stack(monkeypatch)
    assert OnnxEmbedder().embed_spans("anything", []) == []
    assert seen["downloads"] == []


def test_only_the_contextual_embedder_claims_span_pooling() -> None:
    """A static model has no context to pool, so it must not satisfy the protocol.

    ``Model2VecEmbedder`` assigns each token one fixed vector regardless of what
    surrounds it. Were ``embed_spans`` a method on ``Embedder``, it would have to claim
    a capability it cannot have, and a caller would get context-free vectors believing
    otherwise.
    """
    from elbi_core.retrieval import Model2VecEmbedder, SpanEmbedder

    assert isinstance(OnnxEmbedder(), SpanEmbedder)
    assert not isinstance(Model2VecEmbedder(), SpanEmbedder)

    # And it is an Embedder as well as a span pooler: a caller selects on this check and
    # then calls embed() for the spans no token covered, so half the contract is a
    # failure waiting for a truncated document.
    class SpansOnly:
        def embed_spans(
            self, text: str, spans: list[tuple[int, int]]
        ) -> list[list[float]]:
            return []

    assert not isinstance(SpansOnly(), SpanEmbedder)
    # Still a perfectly good Embedder -- it just cannot pool a span.
    assert hasattr(Model2VecEmbedder(), "embed")
