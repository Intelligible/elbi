"""The Elbi-side agent for this demo: coverage-gated retrieval over current components.

Two properties this demo is built to show, both structural -- neither relies on an
LLM's good judgment, so both hold even with no live model available at all:

1. **Instant update.** `freight_components` (see ../derivations) returns every
   version of every fact, but this agent only ever searches the *current* ones
   (`structure["current"]`), so the moment `../publish_update.py` marks the old
   detention component superseded and activates a new one, this agent's answers
   change -- no re-embedding, no stale index to invalidate, nothing left over from
   the old version except the audit trail (`relations: [{"type": "supersedes", ...}]`)
   a click can still follow.
2. **Decline over hallucinate.** `elbi_core.components.search_components` run with
   `embedder=None` is plain BM25 (see its docstring): it returns nothing for a query
   that shares no vocabulary with any component's statement. This agent uses that as
   a hard coverage gate *before* calling an LLM at all -- no keyword overlap with any
   current component means no LLM call, so there is no context for a model to
   confabulate from in the first place. Semantic (embedding) search only ranks
   *among* components that already cleared this keyword gate, which is why a
   paraphrase ("how much for detention after the free hours") still answers while an
   out-of-scope question ("hazmat certification delay") does not.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elbi_core import Registry, Runner
from elbi_core.components import Embedder, search_components
from elbi_core.config import DataBindings
from elbi_core.discovery import discover
from elbi_core.text import tokens as _tokens

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_ROOT))
from llm import phrase_answer  # noqa: E402 -- needs the sys.path insert above

#: Pure function words, not a general-purpose stopword list -- just enough to keep
#: grammatical filler ("what is the fee for") from counting as coverage below.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "what", "how", "much",
        "does", "do", "did", "for", "of", "in", "on", "at", "to", "with",
        "and", "or", "this", "that", "there", "when",
    }
)  # fmt: skip


def _content_terms(text: str) -> set[str]:
    return {term for term in _tokens(text) if term not in _STOPWORDS}


#: A term counts as coverage only if it names at most this many current components.
#: Plain BM25-score-above-zero is too permissive here: the rate schedule's own
#: vocabulary ("fee", "charged", "costs") is shared by half the statements or more,
#: so a raw keyword-overlap score is nonzero for almost any English question and
#: never declines. A word that could refer to several different fees carries no
#: signal about which one (if any) the question is actually about -- coverage needs
#: a term that picks out a small, specific handful of components, not the domain's
#: own vocabulary.
_MAX_DISTINCTIVE_DOCUMENT_FREQUENCY = 3


def _is_covered(question: str, components: list[dict[str, Any]]) -> bool:
    """Whether a *distinctive* word in `question` names a current component."""
    query_terms = _content_terms(question)
    if not query_terms:
        return False

    document_frequency: Counter[str] = Counter()
    for component in components:
        for term in _content_terms(component["statement"]):
            document_frequency[term] += 1

    return any(
        0 < document_frequency.get(term, 0) <= _MAX_DISTINCTIVE_DOCUMENT_FREQUENCY
        for term in query_terms
    )


@dataclass
class AgentAnswer:
    question: str
    declined: bool
    answer: str
    components: list[dict[str, Any]] = field(default_factory=list)


DEFAULT_RATE_SCHEDULE = EXAMPLE_ROOT / "fixtures" / "rate_schedule.csv"


def load_all_components(
    rate_schedule_csv: Path = DEFAULT_RATE_SCHEDULE,
) -> list[dict[str, Any]]:
    """Every version of every component -- current and superseded -- straight from
    the derivation, computed over `rate_schedule_csv`.

    The derivation's code (`../derivations/freight_components.py`) never changes at
    runtime; only which CSV it's bound to does, which is how `serve_demo.py` points
    this at a live, mutable copy (`../runtime/rate_schedule.csv`) while
    `tests/test_example.py` points it at the committed seed fixture, never mutating
    it. :func:`current_components` is the live view an answer is actually drawn
    from; this is the full audit trail.
    """
    registry = Registry()
    discover(EXAMPLE_ROOT / "derivations", registry=registry)
    bindings = DataBindings(bindings={"rate_schedule": str(rate_schedule_csv)})
    runner = Runner(registry, bindings=bindings, base_dir=EXAMPLE_ROOT)
    artifact = runner.run("freight_components")
    return list(artifact.value)


def current_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The live view: every component whose `valid_until` is unset."""
    return [c for c in components if c["structure"]["current"]]


def ask(
    question: str,
    components: list[dict[str, Any]] | None = None,
    *,
    embedder: Embedder | None = None,
    use_llm: bool = True,
    limit: int = 3,
) -> AgentAnswer:
    """Answer `question` from the current components only, or decline.

    `components` defaults to a fresh load of every version (current and
    superseded); pass an already-loaded list to avoid re-running the derivation on
    every call (the split-screen UI does this, loading once per request and again
    right after a publish so it sees the fresh version).
    """
    if components is None:
        components = load_all_components()
    current = current_components(components)

    # The coverage gate (see _is_covered): no distinctive term overlap with any
    # current component's statement means decline before an LLM ever sees it.
    if not _is_covered(question, current):
        return AgentAnswer(
            question=question,
            declined=True,
            answer=(
                "I don't have a component covering that -- nothing in the current "
                "rate schedule shares any wording with this question."
            ),
        )

    hits = search_components(question, current, limit=limit, embedder=embedder)
    context = "\n".join(
        f"- {hit['statement']} (effective {hit['structure']['effective_date']}, "
        f"{hit['evidence'].get('citation', 'no citation')})"
        for hit in hits
    )
    answer = phrase_answer(question, context) if use_llm else None
    if answer is None:
        # Extractive fallback: an ORC component's `statement` is written to be woven
        # into reasoning as-is, so it is already a correct answer on its own -- no
        # live model is needed for this demo's core claim to hold.
        answer = hits[0]["statement"]

    return AgentAnswer(
        question=question, declined=False, answer=answer, components=hits
    )
