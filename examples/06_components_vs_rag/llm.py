"""Shared, model-agnostic LLM call for both sides of this demo.

Backed by LiteLLM (already an `elbi` dependency; see
`packages/elbi-agent/src/elbi_agent/litellm_client.py`), so any provider works by
changing `model`. Both the RAG pipeline and the Elbi agent call this exact function
with the exact same prompt template -- the only difference between them is what
context each one decides to pass in (or whether it calls this at all). That is
deliberate: the demo's claim is about what gets retrieved and gated, not about one
side getting a cleverer prompt than the other.

`phrase_answer` never raises. This repo has no live model credits available in every
environment it runs in (see the example README), so a failed call falls back to
`None` and callers fall back to an extractive answer built directly from whatever
was retrieved -- which keeps both pipelines' core, structural claims (gated vs.
ungated retrieval) verifiable without a live model, and the LLM call an optional
polish layer on top.
"""

from __future__ import annotations

DEFAULT_MODEL = "openai/gpt-4o-mini"

PROMPT_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer the question using only the context above, in one sentence. If the context \
doesn't cover the question, say plainly that you don't have that information rather \
than guessing."""


def phrase_answer(
    question: str, context: str, *, model: str = DEFAULT_MODEL, timeout: float = 20.0
) -> str | None:
    """Return a one-sentence answer grounded in `context`, or None if it failed."""
    try:
        import litellm
    except ImportError:
        return None

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            timeout=timeout,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception:
        # Any failure (no key, no credits, network, rate limit, ...) falls back to
        # the caller's extractive answer rather than breaking the demo.
        return None
