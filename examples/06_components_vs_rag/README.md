# 06 · Components vs. RAG

Nobody puts a whole dataset in a prompt, but plenty of teams put a whole *document*
in a vector store and call it done. This example runs the same three questions
against a real RAG pipeline (chunk, embed, top-k, LLM) and against Elbi's
`components` serve format, over the same nine-line-item freight accessorial rate
schedule, to show what a component database gets you that RAG doesn't:

- **An answer that updates the moment a source is superseded.** RAG has no concept
  of "this chunk replaced that one" -- it just accumulates vectors. Elbi versions
  every component, so the moment a rate changes, the old value stops being served
  and the new one starts, with no re-indexing step.
- **A decline instead of a hallucination when nothing covers the question.** Elbi's
  agent (`agent/demo_agent.py`) checks for real keyword coverage before it ever
  calls a model; if nothing does, it says so. A plain top-k pipeline always
  retrieves *something* and always hands it to the model, confidently or not.
- **An audit trail a prospect can click into**, not one described in a pitch. Every
  component carries a citation, an effective date, and (for a superseded one) a
  `supersedes` relation pointing at exactly what it replaced.

The scenario is a stand-in for a real pricing catch that already landed with a
prospect, scaled down to something runnable and testable.

## Run it

```bash
cd examples/06_components_vs_rag
uv run --package elbi python serve_demo.py
# open http://localhost:8000/demo/components-vs-rag
```

A single query box asks both pipelines at once and shows their answers side by
side. Each Elbi answer is clickable: it expands to the component's id, effective
date, citation, and dependency chain. A "Publish update" button triggers the
three-beat script below live, against both sides at once.

`serve_demo.py` builds a bare FastAPI app and mounts this example's `extension.py`
on it via the real `elbi.extensions.ExtensionContext`/`install_extension`
protocol (`packages/elbi/src/elbi/extensions.py`) -- the same call `elbi serve`
itself would make for an installed extension. See `extension.py`'s docstring for
why this example calls it directly instead of registering a real entry point.

All of it operates on `runtime/`, a gitignored working copy seeded from
`fixtures/` on first run (`publish_update.ensure_runtime`). The committed fixtures
never change; delete `runtime/` to reset the demo to its starting state.

## The three-beat verification script

Automated in `tests/test_example.py` (`uv run pytest examples/06_components_vs_rag/tests`,
runs as part of the repo suite), and reproducible by hand in the UI:

1. **Ask "what's the detention fee after two hours."** Both panes answer
   $50/hour. Boring on purpose -- this beat just establishes the baseline.
2. **Click "Publish update," then ask again.** Elbi answers $75/hour immediately,
   citing the new component, which names the old one it supersedes. RAG's vector
   store now holds *both* the $50/hour and $75/hour chunks -- nothing removed the
   old one -- so its top-k retrieval can surface either, or both at once. What an
   LLM does when handed conflicting context varies: it may repeat the stale
   figure, or blend the two. The one thing it can't reliably do is know to prefer
   the new one, because nothing in the index says which is current.
3. **Ask "what's the fee for a hazmat certification delay."** Nothing in the rate
   schedule covers this, on either side, on purpose. Elbi declines --
   `demo_agent.py`'s coverage gate finds no distinctive keyword overlap with any
   current component and never calls a model. RAG has no such gate: top-k always
   returns its k nearest chunks, so it always answers, confidently, from whatever
   scored closest -- in practice, some unrelated fee.

If beat two doesn't visibly break for RAG in a live run, that's an LLM
occasionally guessing right from ambiguous context, not the underlying claim
being wrong: `tests/test_example.py::test_beat_two_rag_keeps_the_stale_chunk_retrievable`
asserts the structural fact directly (both chunks are retrievable after the
update) rather than betting on a specific wrong answer, which is what the demo
actually guarantees. What a live model does with two conflicting chunks in its
context is not guaranteed, and this example doesn't claim it is.

## Why no live model call is required to trust this

Not every environment this repo runs in has a working, funded model API key.
`llm.py`'s `phrase_answer` always tries a real call
(LiteLLM, default `openai/gpt-4o-mini`) and falls back to an **extractive**
answer -- the retrieved component's `statement`, or the top-scoring chunk's text,
verbatim -- whenever it fails. Every structural claim above (declines, updates,
stale-chunk survival) holds under the extractive fallback alone, which is what
`tests/test_example.py` always uses (`use_llm=False`). A live model changes the
prose, not whether Elbi declined or which chunks RAG retrieved.

## What's real here, and what's a deliberately simplified stand-in

- **The RAG pipeline** (`rag/pipeline.py`) chunks by document section, embeds with
  `elbi_core.retrieval.OnnxEmbedder` (a real contextual model, `BAAI/bge-small-en-v1.5`,
  already vendored for Elbi's own derivation search -- not a toy), and ranks by
  cosine similarity. No recency tracking and no relevance floor are not omissions;
  they're the two things a minimal real deployment commonly lacks, and precisely
  what this example is about.
- **The rate schedule** (`fixtures/rate_schedule.csv`) is human-authored, not
  computed from raw data -- unlike `05_components`'s `churn_components`, which
  derives its facts by statistics. `derivations/freight_components.py`'s
  `provenance.source` says `"human"` accordingly. Two components
  (`fuel_surcharge_low`/`_high`) declare a real `depends_on` relation to a stubbed
  `diesel_price_index` component, so the dependency graph has something to walk,
  as asked -- the index itself is a fixed placeholder, not a live DOE feed.
- **"Elbi agent"** (`agent/demo_agent.py`) is a small, purpose-built module for
  this demo, not `elbi_agent.Runtime` (the package at `packages/elbi-agent`).
  That runtime drives a heavier verifying-analysis loop (propose code, run it
  under isolation, gate on the oracle) built for authoring new derivations, not
  for answering a question by retrieval over already-served ones -- a poor fit
  here. This module reuses the real, public `elbi_core.components.search_components`
  and the real `freight_components` derivation; only the thin "ask, gate, answer"
  wrapper around them is demo-specific.
