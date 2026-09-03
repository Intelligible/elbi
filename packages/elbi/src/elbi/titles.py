"""Generate a conversation title from its opening message.

Mirrors the two-tier approach used by mature chat agents: ask the configured LLM for a
short, descriptive title, and fall back to truncating the first message when the model
is unavailable or errs. It is model-agnostic: it drives whatever ``LLMClient`` the app
was built with (the same one the chat uses), with no tools, so it works for any provider
that backs that seam, not a specific vendor or key.
"""

from __future__ import annotations

from elbi_agent import LLMClient, Transcript

#: The default title length. Titles beyond this are truncated with an ellipsis.
MAX_TITLE_LENGTH = 50

_TITLE_SYSTEM = (
    "You generate a concise, descriptive title for a conversation with a data-analysis "
    "assistant. Given the user's opening message (which may be truncated), return only "
    "the title: a short phrase, with no quotes, no trailing punctuation, no emoji, and "
    "no explanation."
)


def llm_title(
    message: str, client: LLMClient, *, max_length: int = MAX_TITLE_LENGTH
) -> str | None:
    """Ask the LLM for a title, or ``None`` on an empty answer or any error.

    Drives ``client`` with a single, tool-free completion over the opening message
    (truncated to bound tokens). Returning ``None``, never raising, lets the caller keep
    the plain truncated-message title, so titling can never break a conversation.
    """
    text = message.strip()
    if not text:
        return None
    transcript = Transcript(system=_TITLE_SYSTEM)
    transcript.add_user_text(
        f"Generate a title (at most {max_length} characters) for a conversation that "
        f"starts with this message:\n\n{text[:1000]}"
    )
    try:
        step = client.step(transcript, ())
    except Exception:
        return None
    return _clean(step.text, max_length)


def fallback_title(message: str, *, max_length: int = MAX_TITLE_LENGTH) -> str:
    """The first line of the opening message, truncated: the no-LLM default."""
    title = message.strip().splitlines()[0].strip() if message.strip() else "analysis"
    return _truncate(title, max_length)


def make_title(
    message: str, client: LLMClient | None, *, max_length: int = MAX_TITLE_LENGTH
) -> str:
    """A title for the opening ``message``: the LLM's if available, else truncation.

    With no ``client`` (or on any LLM failure) this is the plain truncated first line,
    so a title is always produced.
    """
    if client is not None:
        generated = llm_title(message, client, max_length=max_length)
        if generated:
            return generated
    return fallback_title(message, max_length=max_length)


def _clean(text: str | None, max_length: int) -> str | None:
    """Reduce a model reply to a bare title: first line, unquoted, no trailing dot."""
    if not text or not text.strip():
        return None
    title = text.strip().splitlines()[0].strip().strip("\"'“”").strip()
    title = title.rstrip(".").strip()
    return _truncate(title, max_length) if title else None


def _truncate(title: str, max_length: int) -> str:
    if len(title) <= max_length:
        return title
    return title[: max_length - 1].rstrip() + "…"
