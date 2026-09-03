"""Name generation for artifact duplication: a distinct name, never an error."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from uuid import uuid4

_MAX_NUMBERED_ATTEMPTS = 100

_NON_IDENTIFIER_CHARS = re.compile(r"[^a-z0-9_]+")
_LEADING_NON_ALPHA = re.compile(r"^[^a-z]+")


def _slugify(base: str, *, limit: int) -> str:
    """``base`` as a valid, non-empty identifier body, truncated to ``limit``."""
    slug = _NON_IDENTIFIER_CHARS.sub("_", base.strip().lower()).strip("_")
    slug = _LEADING_NON_ALPHA.sub("", slug)
    return (slug or "copy")[:limit]


def _suffixes(first: str, numbered: str, random: str) -> Iterable[str]:
    """``first``, then ``numbered`` per attempt, then a random tail.

    The bound is what keeps the generators total: past it they stop looking for a
    readable name and take a random one rather than looping or raising.
    """
    yield first
    for n in range(2, _MAX_NUMBERED_ATTEMPTS + 1):
        yield numbered.format(n=n)
    yield random.format(tail=uuid4().hex[:6])


def copy_identifier(base: str, taken: Collection[str], *, limit: int = 128) -> str:
    """A schema-valid identifier outside ``taken``: ``"x_copy"``, ``"x_copy_2"``.

    Matches ``^[a-z][a-z0-9_]*$`` and stays within ``limit``, which is what a
    dashboard or metric manifest's own ``name`` requires.
    """
    candidate = ""
    for suffix in _suffixes("_copy", "_copy_{n}", "_{tail}"):
        room = max(limit - len(suffix), 1)
        candidate = f"{_slugify(base, limit=room)}{suffix}"[:limit]
        if candidate not in taken:
            break
    return candidate


def copy_label(base: str, taken: Collection[str], *, limit: int = 256) -> str:
    """A display name outside ``taken``: ``"X (copy)"``, ``"X (copy 2)"``, ...

    For free text: a notebook or saved-query name, or a dashboard's title.
    """
    stripped = base.strip() or "Untitled"
    candidate = ""
    for suffix in _suffixes("(copy)", "(copy {n})", "(copy {tail})"):
        # Reserve the suffix and its joining space before truncating, the way
        # :func:`copy_identifier` does. Trimming the composed string instead cuts the
        # suffix off a base already at the limit, and every attempt then collapses to
        # the name the caller asked to avoid.
        room = max(limit - len(suffix) - 1, 1)
        candidate = f"{stripped[:room].rstrip()} {suffix}"[:limit]
        if candidate not in taken:
            break
    return candidate
