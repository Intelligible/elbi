"""Runs the components-vs-RAG demo end to end -- the three-beat verification script,
automated, plus the structural claims each beat depends on. Part of the repo's CI
suite, like every other example.

No live LLM call is required for anything here: `demo_agent.ask`/`rag.pipeline.ask`
are always called with `use_llm=False`, which exercises the exact same retrieval and
gating code a live run would, just with the extractive fallback instead of an LLM's
phrasing (see `llm.py`'s module docstring for why that split exists -- this repo has
no live model credits in every environment it runs in). What a live model would only
add is prose; it can't change whether Elbi declines or which chunks RAG retrieves,
which is what these tests actually check.

Building the embedder downloads a small ONNX model from the Hugging Face Hub on
first use (see `elbi_core.retrieval.OnnxEmbedder`); tests that need it skip, with a
clear reason, if that download fails -- the same pattern this repo already uses for
tests needing a live DynamoDB/Postgres/etc (see packages/elbi/tests/test_warehouse_*).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
for _path in (EXAMPLE_ROOT, EXAMPLE_ROOT / "agent", EXAMPLE_ROOT / "rag"):
    sys.path.insert(0, str(_path))

import demo_agent  # noqa: E402
import pipeline  # noqa: E402
import publish_update  # noqa: E402

from elbi_core.components import is_valid_component  # noqa: E402

DETENTION_QUESTION = "what's the detention fee after two hours"
HAZMAT_QUESTION = "what is the fee for a hazmat certification delay"


@pytest.fixture(scope="module")
def embedder():
    try:
        from elbi_core.retrieval import OnnxEmbedder

        model = OnnxEmbedder()
        model.embed(["warmup"])  # forces the download/load now, not mid-test
        return model
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"no network to fetch the embedding model: {exc}")


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    """A fresh runtime working copy per test -- never the committed fixtures/, and
    never the shared `runtime/` a live `serve_demo.py` would use."""
    return publish_update.ensure_runtime(tmp_path / "runtime")


# ---------------------------------------------------------------------------
# The derivation itself
# ---------------------------------------------------------------------------


def test_freight_components_are_all_valid_orc_components(runtime: Path) -> None:
    components = demo_agent.load_all_components(runtime / "rate_schedule.csv")
    assert len(components) == 10  # the 9 line items + the diesel_price_index stub
    for component in components:
        assert is_valid_component(component)


def test_fuel_surcharge_tiers_depend_on_diesel_price_index(runtime: Path) -> None:
    components = demo_agent.load_all_components(runtime / "rate_schedule.csv")
    by_key = {c["structure"]["key"]: c for c in components}
    for key in ("fuel_surcharge_low", "fuel_surcharge_high"):
        targets = {
            r["target_id"]
            for r in by_key[key]["relations"]
            if r["type"] == "depends_on"
        }
        assert any("diesel_price_index" in t for t in targets)


def test_hazmat_is_genuinely_absent_from_the_seed_data(runtime: Path) -> None:
    """Both beat three's premise and a guard against a future edit accidentally
    adding hazmat coverage and silently invalidating that beat."""
    components = demo_agent.load_all_components(runtime / "rate_schedule.csv")
    assert not any("hazmat" in c["statement"].lower() for c in components)
    docs_text = (runtime / "rag_docs" / "base_tariff.md").read_text().lower()
    assert "hazmat" not in docs_text


# ---------------------------------------------------------------------------
# Beat one: both sides agree, boring on purpose
# ---------------------------------------------------------------------------


def test_beat_one_both_sides_answer_fifty_dollars(runtime: Path, embedder) -> None:
    elbi_answer = demo_agent.ask(DETENTION_QUESTION, use_llm=False)
    assert not elbi_answer.declined
    assert "$50/hour" in elbi_answer.answer

    store = pipeline.build_store(runtime / "rag_docs", embedder=embedder)
    rag_answer = pipeline.ask(
        DETENTION_QUESTION, store, embedder=embedder, use_llm=False
    )
    assert "$50/hour" in rag_answer.answer


# ---------------------------------------------------------------------------
# Beat two: trigger the update, ask again
# ---------------------------------------------------------------------------


def test_beat_two_elbi_updates_instantly_and_keeps_an_audit_trail(
    runtime: Path,
) -> None:
    result = publish_update.run(runtime)
    assert result.applied

    components = demo_agent.load_all_components(runtime / "rate_schedule.csv")
    current = demo_agent.current_components(components)
    # Still exactly one current detention component -- the update didn't duplicate
    # the live view, only the history grew.
    current_detention = [c for c in current if c["structure"]["key"] == "detention_fee"]
    assert len(current_detention) == 1
    assert current_detention[0]["structure"]["rate"] == 75.0

    elbi_answer = demo_agent.ask(DETENTION_QUESTION, components, use_llm=False)
    assert not elbi_answer.declined
    assert "$75/hour" in elbi_answer.answer
    assert "$50/hour" not in elbi_answer.answer

    # The audit trail: the new component cites the old one by id, and the old one
    # is still present (just no longer "current"), not deleted.
    new_component = elbi_answer.components[0]
    supersedes = [r for r in new_component["relations"] if r["type"] == "supersedes"]
    assert len(supersedes) == 1
    old_id = supersedes[0]["target_id"]
    assert any(c["id"] == old_id and not c["structure"]["current"] for c in components)


def test_beat_two_rag_keeps_the_stale_chunk_retrievable(
    runtime: Path, embedder
) -> None:
    """The structural claim beat two dramatizes: nothing in the vector store marks
    the old chunk as superseded, so both the $50 and $75 chunks stay retrievable
    after the update -- top-k can return either, or both, which is exactly why the
    live answer is unreliable (see rag/pipeline.py's module docstring). This
    asserts the structural fact directly rather than a specific top-1 answer, which
    is the part that's actually guaranteed; which one an LLM would pick from a
    context containing both is not, and isn't what this test claims.
    """
    store = pipeline.build_store(runtime / "rag_docs", embedder=embedder)
    result = publish_update.run(runtime)
    assert result.applied
    amendment_chunks = pipeline.chunk_document(runtime / "rag_docs" / "amendment_1.md")
    store.add(amendment_chunks, embedder)

    hits = store.top_k(DETENTION_QUESTION, embedder, k=5)
    hit_texts = [h.text for h in hits]
    assert any("$50/hour" in t for t in hit_texts), "stale chunk should still rank"
    assert any("$75/hour" in t for t in hit_texts), "new chunk should also rank"


def test_publish_update_is_idempotent(runtime: Path) -> None:
    first = publish_update.run(runtime)
    assert first.applied
    second = publish_update.run(runtime)
    assert not second.applied

    components = demo_agent.load_all_components(runtime / "rate_schedule.csv")
    detention_components = [
        c for c in components if c["structure"]["key"] == "detention_fee"
    ]
    # Exactly two total (original + the one update), not three -- a repeat trigger
    # changed nothing.
    assert len(detention_components) == 2


# ---------------------------------------------------------------------------
# Beat three: out of scope for both, decline vs. confidently wrong
# ---------------------------------------------------------------------------


def test_beat_three_elbi_declines_rather_than_hallucinate(runtime: Path) -> None:
    answer = demo_agent.ask(HAZMAT_QUESTION, use_llm=False)
    assert answer.declined
    assert answer.components == []


def test_beat_three_rag_answers_anyway_with_no_relevance_gate(
    runtime: Path, embedder
) -> None:
    """RAG has no coverage gate at all (see pipeline.py's module docstring): it
    always returns k chunks and always hands them to the answer step, however
    irrelevant. This asserts that absence directly -- it always retrieves
    something for an out-of-scope question -- rather than asserting a specific
    wrong answer, since which wrong chunk wins is not the guaranteed part.
    """
    store = pipeline.build_store(runtime / "rag_docs", embedder=embedder)
    rag_answer = pipeline.ask(HAZMAT_QUESTION, store, embedder=embedder, use_llm=False)
    assert len(rag_answer.chunks) == 3  # the default k, unconditionally
    assert rag_answer.answer  # some confident-looking text, not an empty response
    assert "hazmat" not in rag_answer.answer.lower()  # it can't be -- nothing says it


# ---------------------------------------------------------------------------
# The actual HTTP surface (extension.py + serve_demo.py), not just the library code
# ---------------------------------------------------------------------------


def test_split_screen_api_end_to_end(tmp_path: Path, embedder) -> None:
    """Builds the real app via the real `elbi.extensions` protocol (`extension.py`'s
    `install_extension`, the exact call the shipped app makes for any installed
    extension) and drives the three beats through the actual HTTP routes the
    browser UI calls, not just the library functions tested above.
    """
    import extension
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from elbi.extensions import ExtensionContext

    app = FastAPI()
    # An isolated runtime dir, not the shared one `serve_demo.py` uses for a live
    # demo -- this test must never touch or depend on that.
    extension.install_extension(
        ExtensionContext(app=app, store=None), runtime_dir=tmp_path / "runtime"
    )
    client = TestClient(app)

    page = client.get("/demo/components-vs-rag")
    assert page.status_code == 200
    assert "Elbi vs. RAG" in page.text

    beat1 = client.post(
        "/demo/components-vs-rag/api/ask", json={"question": DETENTION_QUESTION}
    ).json()
    assert "$50/hour" in beat1["rag"]["answer"]
    assert "$50/hour" in beat1["elbi"]["answer"]

    published = client.post("/demo/components-vs-rag/api/publish-update").json()
    assert published["applied"]

    beat2 = client.post(
        "/demo/components-vs-rag/api/ask", json={"question": DETENTION_QUESTION}
    ).json()
    assert "$75/hour" in beat2["elbi"]["answer"]
    assert not beat2["elbi"]["declined"]

    beat3 = client.post(
        "/demo/components-vs-rag/api/ask", json={"question": HAZMAT_QUESTION}
    ).json()
    assert beat3["elbi"]["declined"]
    assert beat3["rag"]["answer"]  # RAG still answers something, confidently
