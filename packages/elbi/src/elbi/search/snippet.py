"""A window of matching text, so a result says why it matched.

Not a truncated description: two artifacts matching one query should show two different
fragments. Matching folds the way the ranker folds, through
:func:`~elbi_core.text.tokens`, so a snippet highlights the text that scored.

**The security constraint is the shape of the API.** A snippet drawn from a body the
caller may not read leaks it one guessed term at a time, so the text to search is passed
in by the caller who knows what may be read. This module never reaches for a field.
"""

from __future__ import annotations

from elbi_core.text import WORD_RE, fold, tokens

#: How much text a snippet carries. Long enough to be a sentence, short enough that a
#: page of them stays scannable.
WINDOW = 180

#: What replaces the text a window cuts off.
ELLIPSIS = "…"


def snippet(text: str, query: str, *, window: int = WINDOW) -> str:
    """A window of ``text`` around its best match for ``query``.

    Falls back to the opening when nothing matches, which happens when the hit came
    from a field this snippet is not drawn from. An empty snippet would read as "no
    content" rather than "matched elsewhere".
    """
    text = " ".join(text.split())
    if not text:
        return ""
    if len(text) <= window:
        return text

    wanted = set(tokens(query))
    if not wanted:
        return text[:window].rstrip() + ELLIPSIS

    spans = [span for span in _word_spans(text) if span[2] in wanted]
    if not spans:
        return text[:window].rstrip() + ELLIPSIS

    start = _best_start(text, spans, window)
    end = min(len(text), start + window)
    # Do not cut mid-word at either edge: a fragment of a word reads as a typo.
    if start > 0:
        start = text.find(" ", start) + 1 or start
    if end < len(text):
        cut = text.rfind(" ", start, end)
        end = cut if cut > start else end
    return (
        (ELLIPSIS if start > 0 else "")
        + text[start:end].strip()
        + (ELLIPSIS if end < len(text) else "")
    )


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    """Every word as ``(start, end, folded)``, folded the way the ranker folds.

    Over ``WORD_RE`` rather than whitespace, because the ranker's tokenizer splits
    inside a word: ``churn_risk`` is two terms to it, and taking only the first left a
    query for ``risk`` matching nothing here while matching in the ranking. A snippet
    then opened at the top of the text instead of at the hit.

    ``tokens`` returns folded forms without offsets, so the folding is applied per match
    to keep the offsets pointing back at the original characters.
    """
    return [
        (match.start(), match.end(), fold(match.group()))
        for match in WORD_RE.finditer(text.lower())
    ]


def _best_start(text: str, spans: list[tuple[int, int, str]], window: int) -> int:
    """Where to open a window so it covers the densest cluster of matches.

    Density rather than the first hit: several terms are best explained by the passage
    where they occur together, rarely where the first one appears.
    """
    best_start, best_score = 0, -1
    for start, _end, _folded in spans:
        opening = max(0, start - window // 4)
        covered = {
            folded
            for hit_start, _hit_end, folded in spans
            if opening <= hit_start < opening + window
        }
        density = sum(
            1 for hit_start, _e, _f in spans if opening <= hit_start < opening + window
        )
        score = len(covered) * 1000 + density
        if score > best_score:
            best_start, best_score = opening, score
    return min(best_start, max(0, len(text) - window))
