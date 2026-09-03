"""Condense the older turns of a long conversation into a compact state note.

A sliding window replays only the recent turns into the model's context; older turns
would otherwise be dropped entirely. This summarizes them into a short, structured note
that is carried alongside the window, so a long conversation keeps its earlier goals and
decisions without replaying every message. It mirrors a rolling summarizing condenser:
keep the recent tail verbatim, fold everything before it into one running summary.

The certified derivations already serve as durable memory for *established results*;
this captures the conversational state around them (what the user asked for, what was
tried, what is still open) that the derivations do not record.
"""

from __future__ import annotations

from collections.abc import Sequence

from .llm import LLMClient, Transcript

_CONDENSE_SYSTEM = (
    "You maintain a concise running summary of a data-analysis conversation so it "
    "survives context truncation. Given the earlier turns (which may include a prior "
    "summary), return a compact state note with these sections when relevant: "
    "USER_CONTEXT (the user's goals, data, and clarifications), FINDINGS (what was "
    "established, with specifics like columns and effects), and PENDING (open items). "
    "Keep it short; preserve concrete details; return only the note, no preamble."
)


def condense_turns(client: LLMClient, turns: Sequence[tuple[str, str]]) -> str:
    """Summarize earlier conversation turns into a compact state note.

    ``turns`` are ``(role, text)`` pairs, oldest first. Returns ``""`` when there is
    nothing to summarize or the model produces no text (the caller then simply carries
    no summary rather than failing the turn).
    """
    kept = [(role, text) for role, text in turns if text.strip()]
    if not kept:
        return ""
    transcript = Transcript(system=_CONDENSE_SYSTEM)
    rendered = "\n\n".join(f"{role.upper()}: {text}" for role, text in kept)
    transcript.add_user_text(
        "Summarize these earlier turns into the running state note:\n\n" + rendered
    )
    step = client.step(transcript, [])
    return (step.text or "").strip()
