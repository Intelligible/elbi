"""Check the numbers an answer states against the served outputs it cites.

A written number is supported when a served value, rounded to the precision it is
written with, equals it. A number with a percent unit may also be a served fraction
times 100 (``28.6%`` for ``0.2857``). Nothing else is accepted.

Extraction is lexical: words are not read, so a sign carried by a verb ("fell
21.9%") is compared by magnitude. Numbers that label rather than claim (dates, list
markers, footnote marks, a confidence level, a ``name@hash`` citation) are masked,
and one glued to letters (``Q3``, ``2nd``, ``5k``) is part of a word.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

_CITATION = re.compile(
    r"(?<![\w.@-])(?P<name>[A-Za-z_]\w*)@(?P<version>[0-9a-f]{8,64})(?!\.?\w)"
)
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_MASK = re.compile(
    _CITATION.pattern
    + r"|\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    + rf"|\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"
    + rf"|\b\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?\b|\b{_MONTH}\s+\d{{4}}\b"
    + r"|^[ \t]*\d+[.)](?=\s)|\[\d{1,3}\]"  # a list marker, a footnote mark
    + r"|(?<![-+\u2212.,$])\b(?:19|20)\d{2}\b(?![.,]\d|\s?%)"  # a bare year
    + r"|\b\d{2}(?:\.\d+)?\s?%\s+(?:CI|confidence|credible)\b",
    re.MULTILINE,
)
# A sign only where it cannot be a hyphen; never after `x-` (`COVID-19`, `x-5`).
_NUMBER = re.compile(
    r"(?:(?<![\w.])(?P<sign>[-+\u2212]))?(?<![\w.])(?<![A-Za-z_][-\u2212])"
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<exponent>[eE][-+]?\d+)?(?!\w|\.\d)"
    r"(?P<unit>\s?%|\s*(?:percentage points?|percent|pct|pts?|points?|pp)\b)?"
)


@dataclass(frozen=True)
class Mention:
    """A number as written: its signed value and written precision.

    ``precision`` is the decimal places written (negative for an exponent above the
    digits); ``percent`` marks a ``%`` or points unit; ``signed`` an explicit sign,
    without which only the magnitude is claimed.
    """

    text: str
    value: float
    precision: int
    percent: bool
    signed: bool


@dataclass(frozen=True)
class Citation:
    """A ``name@hash`` token: a served output and the version it was read at."""

    name: str
    version: str


@dataclass(frozen=True)
class Leaf:
    """One number inside a served value, at its JSON path."""

    path: str
    value: float


@dataclass(frozen=True)
class Support:
    """A mention and the served value that supports it, or none."""

    mention: Mention
    derivation: str | None = None
    path: str | None = None
    value: float | None = None
    scale: float = 1.0

    @property
    def supported(self) -> bool:
        """Whether a served value rounds to the mention."""
        return self.derivation is not None


@dataclass(frozen=True)
class CitationStatus:
    """A citation and whether it names the version served now."""

    citation: Citation
    status: str  # "current" | "stale" | "unknown"
    current: str | None


@dataclass(frozen=True)
class CitationReport:
    """The outcome of checking an answer's numbers and citations."""

    verdict: str  # "faithful" | "unfaithful" | "inconclusive"
    numbers: tuple[Support, ...]
    citations: tuple[CitationStatus, ...]
    sources: tuple[str, ...]

    def render(self) -> str:
        """Format the report as markdown for an agent to read."""
        supported = sum(s.supported for s in self.numbers)
        lines = [
            f"# Citations: **{self.verdict.upper()}**",
            "",
            f"checked against: {', '.join(self.sources) or '(no served output)'}",
            f"{supported} of {len(self.numbers)} number(s) supported",
        ]
        if self.numbers:
            lines.append("\nnumbers:")
            lines += [_render_support(s) for s in self.numbers]
        if self.citations:
            lines.append("\ncitations:")
            lines += [_render_status(c) for c in self.citations]
        return "\n".join(lines)


def extract_numbers(text: str) -> list[Mention]:
    """Every number in ``text`` that states a quantity, in order of appearance."""
    masked = _MASK.sub(lambda m: " " * len(m.group()), text)
    mentions = []
    for m in _NUMBER.finditer(masked):
        number = Decimal(m["digits"].replace(",", "") + (m["exponent"] or ""))
        negative = m["sign"] in ("-", "\u2212")
        mentions.append(
            Mention(
                text=text[m.start() : m.end()],
                value=float(-number if negative else number),
                precision=-int(number.as_tuple().exponent),
                percent=m["unit"] is not None,
                signed=m["sign"] is not None,
            )
        )
    return mentions


def extract_citations(text: str) -> list[Citation]:
    """Every ``name@hash`` token in ``text``, in order of appearance."""
    return [Citation(m["name"], m["version"]) for m in _CITATION.finditer(text)]


def numeric_leaves(value: object, path: str = "$") -> list[Leaf]:
    """Flatten a served value into its numbers, each with a JSON path.

    A string contributes the numbers written in it, so a component ``statement`` or
    a text-typed cell counts.
    """
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [Leaf(path, float(value))] if math.isfinite(value) else []
    if isinstance(value, str):
        return [Leaf(path, mention.value) for mention in extract_numbers(value)]
    if isinstance(value, Mapping):
        return [
            leaf
            for key, item in value.items()
            for leaf in numeric_leaves(item, f"{path}.{key}")
        ]
    if isinstance(value, list | tuple):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in numeric_leaves(item, f"{path}[{index}]")
        ]
    return []


def check(
    text: str,
    sources: Mapping[str, object],
    current_versions: Mapping[str, str],
) -> CitationReport:
    """Check every number and citation in ``text`` against served ``sources``.

    ``sources`` maps a derivation name to its served value; ``current_versions`` maps
    a name to the version served now. A name it lacks is resolved from the
    ``name@hash`` strings the sources carry (a ``source`` provenance field), and a
    citation resolvable from neither is ``unknown``.
    """
    leaves = [
        (name, leaf) for name in sources for leaf in numeric_leaves(sources[name])
    ]
    versions: dict[str, str] = {}
    for value in sources.values():
        for cited in extract_citations(json.dumps(value, default=str)):
            versions.setdefault(cited.name, cited.version)
    versions.update(current_versions)

    numbers = tuple(_support(mention, leaves) for mention in extract_numbers(text))
    unique = {(c.name, c.version): c for c in extract_citations(text)}
    citations = tuple(_status(c, versions.get(c.name)) for c in unique.values())
    if any(not s.supported for s in numbers) or any(
        c.status != "current" for c in citations
    ):
        verdict = "unfaithful"
    else:
        verdict = "faithful" if numbers else "inconclusive"
    return CitationReport(verdict, numbers, citations, tuple(sources))


def _support(mention: Mention, leaves: list[tuple[str, Leaf]]) -> Support:
    # The conversion the unit implies is tried first, so `5%` reports `rate = 0.05`
    # before a count of 5 that happens to round the same.
    for scale in (100.0, 1.0) if mention.percent else (1.0,):
        for name, leaf in leaves:
            if _rounds_to(leaf.value * scale, mention):
                return Support(mention, name, leaf.path, leaf.value, scale)
    return Support(mention)


def _rounds_to(value: float, mention: Mention) -> bool:
    if not math.isfinite(mention.value):
        return False
    candidate = value if mention.signed else abs(value)
    try:
        half = 0.5 * 10.0**-mention.precision
    except OverflowError:  # a finite number written as 0.0e400
        return False
    # Inclusive of half a unit so either rounding of a tie passes; the 1e-9 factor
    # only absorbs float error.
    return abs(candidate - mention.value) <= half * (1 + 1e-9)


def _status(citation: Citation, current: str | None) -> CitationStatus:
    if current is None:
        return CitationStatus(citation, "unknown", None)
    fresh = current.startswith(citation.version) or citation.version.startswith(current)
    return CitationStatus(citation, "current" if fresh else "stale", current)


def _render_support(support: Support) -> str:
    written = support.mention.text.strip()
    if not support.supported:
        return f"- [FAIL] {written}: no served value rounds to it"
    scale = " (x100)" if support.scale == 100.0 else ""
    source = f"{support.derivation} {support.path} = {support.value:.10g}"
    return f"- [ok] {written}: {source}{scale}"


def _render_status(status: CitationStatus) -> str:
    cited = f"{status.citation.name}@{status.citation.version}"
    if status.status == "current":
        return f"- [ok] {cited}: current"
    if status.status == "stale":
        now = f"{status.citation.name}@{(status.current or '')[:8]}"
        return f"- [FAIL] {cited}: stale, now {now}"
    return f"- [FAIL] {cited}: unknown, no served output names {status.citation.name}"
