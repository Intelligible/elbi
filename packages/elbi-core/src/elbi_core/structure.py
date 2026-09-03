"""Structure discovery: the relational skeleton of a dataset, from values alone.

An agent reasoning over unfamiliar data needs to know how its columns relate
before it can analyze them well: which columns are redundant, which determine
others, which are identifiers, and which carry shared information (the candidates
for a confounder). A schema and one example row do not show any of this.

This module recovers that skeleton with classical, deterministic methods, so the
result is reproducible and carries no model or training:

* **functional dependencies** (``A`` determines ``B`` when ``B`` is constant within
  every group of equal ``A``) and **derived columns** (``C = A + B`` exactly), which
  expose redundancy and perfect collinearity;
* **candidate keys** (a near-unique column is an identifier, not a variable to
  model);
* a **dependency graph** by normalized mutual information, computed on
  quantile-binned values so a high-cardinality column does not spuriously appear to
  determine everything (the bias of naive entropy on continuous data).

It describes *structure*, never meaning: it says ``severity`` shares information
with ``days``, not what either means. Meaning is the semantic model's job; this is
the grounding an agent reads before it forms a hypothesis.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Number of quantile bins continuous columns are discretized into for mutual
#: information. Enough to capture dependence, coarse enough to avoid the
#: high-cardinality bias.
_BINS = 10

#: Report a dependency only above this normalized mutual information, and at most
#: this many, so the map highlights real structure instead of every faint link.
_MI_FLOOR = 0.1
_MAX_LINKS = 15

#: A column whose values are at least this unique is treated as an identifier.
_KEY_UNIQUENESS = 0.99


@dataclass(frozen=True)
class StructureMap:
    """The discovered relational skeleton of a dataset.

    ``kinds`` maps each column to ``categorical``, ``discrete`` (a small integer
    set, i.e. a code), or ``continuous``. ``dependencies`` are ``(a, b, strength)``
    by normalized mutual information, strongest first.
    """

    n_rows: int
    kinds: dict[str, str]
    candidate_keys: tuple[str, ...]
    functional_dependencies: tuple[tuple[str, str], ...]
    derived_columns: tuple[str, ...]
    dependencies: tuple[tuple[str, str, float], ...] = field(default=())

    def render(self) -> str:
        """A compact markdown view for an agent to read before analyzing."""
        lines = ["# Data structure", ""]
        kinds = ", ".join(f"{c} ({k})" for c, k in self.kinds.items())
        lines.append(f"Columns: {kinds}")
        if self.candidate_keys:
            lines.append(
                f"\nIdentifiers (near-unique; not variables to model): "
                f"{', '.join(self.candidate_keys)}"
            )
        if self.derived_columns:
            lines.append(
                "\nDerived / collinear (do not put all together in a regression):"
            )
            lines += [f"- {d}" for d in self.derived_columns]
        if self.functional_dependencies:
            lines.append("\nFunctional dependencies (one value fixes another):")
            lines += [f"- {a} determines {b}" for a, b in self.functional_dependencies]
        if self.dependencies:
            lines.append(
                "\nRelated columns (shared information; a driver of one may confound "
                "another):"
            )
            lines += [f"- {a} ~ {b} ({s:.2f})" for a, b, s in self.dependencies]
        return "\n".join(lines)


def analyze(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> StructureMap:
    """Discover the structure of ``rows`` (string-valued) over ``columns``."""
    cols = list(columns)
    values = {c: [_cell(r.get(c)) for r in rows] for c in cols}
    n = len(rows)
    kinds = {c: _kind(values[c]) for c in cols}
    binned = {c: _binned(values[c], kinds[c]) for c in cols}

    return StructureMap(
        n_rows=n,
        kinds=kinds,
        candidate_keys=_candidate_keys(values, kinds, n),
        functional_dependencies=_functional_dependencies(values, n),
        derived_columns=_derived_columns(values),
        dependencies=_dependencies(binned),
    )


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def _floats(values: Sequence[str]) -> list[float] | None:
    """Parse a fully-numeric column to floats, else None."""
    out: list[float] = []
    for v in values:
        if v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            return None
    return out


def _kind(values: Sequence[str]) -> str:
    floats = _floats(values)
    if floats is None or not floats:
        return "categorical"
    if all(x.is_integer() for x in floats) and len({*floats}) <= 12:
        return "discrete"
    return "continuous"


def _binned(values: Sequence[str], kind: str) -> list[str]:
    """Map a column to discrete labels: quantile bins for continuous, else itself."""
    if kind != "continuous":
        return list(values)
    floats = [float(v) for v in values if v != ""]
    cuts = _quantile_cuts(floats, _BINS)
    return ["" if v == "" else f"q{_bin_index(float(v), cuts)}" for v in values]


def _quantile_cuts(floats: Sequence[float], bins: int) -> list[float]:
    ordered = sorted(floats)
    m = len(ordered)
    return [ordered[min(m - 1, i * m // bins)] for i in range(1, bins)]


def _bin_index(x: float, cuts: Sequence[float]) -> int:
    index = 0
    for cut in cuts:
        if x <= cut:
            return index
        index += 1
    return index


def _entropy(values: Sequence[str]) -> float:
    present = [v for v in values if v != ""]
    n = len(present)
    if not n:
        return 0.0
    counts = Counter(present)
    return -sum((k / n) * math.log2(k / n) for k in counts.values())


def _conditional_entropy(given: Sequence[str], target: Sequence[str]) -> float:
    """H(target | given) over rows where both are present."""
    groups: dict[str, list[str]] = {}
    pairs = [(g, t) for g, t in zip(given, target, strict=True) if g != "" and t != ""]
    for g, t in pairs:
        groups.setdefault(g, []).append(t)
    n = len(pairs)
    if not n:
        return 0.0
    return sum((len(members) / n) * _entropy(members) for members in groups.values())


def _candidate_keys(
    values: dict[str, list[str]], kinds: dict[str, str], n: int
) -> tuple[str, ...]:
    """Near-unique columns that are an identity, not a quantity to model.

    A continuous column is near-unique merely by being continuous, so it only
    counts as a key when its name marks it as one; categorical near-unique columns
    are keys on their own.
    """
    keys = []
    for column, cells in values.items():
        present = [v for v in cells if v != ""]
        if not present or len(set(present)) < _KEY_UNIQUENESS * len(present):
            continue
        named = any(h in column.lower() for h in ("id", "key", "uuid", "guid", "code"))
        if kinds[column] != "continuous" or named:
            keys.append(column)
    return tuple(keys)


def _functional_dependencies(
    values: dict[str, list[str]], n: int
) -> tuple[tuple[str, str], ...]:
    """``a`` determines ``b`` when ``b`` is constant within every group of equal ``a``.

    Trivial determiners (a near-unique ``a`` determines everything) are excluded.
    """
    cols = list(values)
    groups = {c: _group_indices(values[c]) for c in cols}
    out = []
    for a in cols:
        if len(groups[a]) >= _KEY_UNIQUENESS * n:
            continue  # a is ~unique; it trivially determines everything
        for b in cols:
            if a == b:
                continue
            if all(len({values[b][i] for i in idx}) == 1 for idx in groups[a].values()):
                out.append((a, b))
    return tuple(out)


def _group_indices(cells: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, value in enumerate(cells):
        if value != "":
            groups.setdefault(value, []).append(i)
    return groups


def _derived_columns(values: dict[str, list[str]]) -> tuple[str, ...]:
    """Columns that are an exact sum of two others (perfect collinearity)."""
    numeric = {c: f for c in values if (f := _floats_aligned(values[c])) is not None}
    cols = list(numeric)
    out = []
    for c in cols:
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                if c in (a, b):
                    continue
                if _is_sum(numeric[c], numeric[a], numeric[b]):
                    out.append(f"{c} = {a} + {b}")
    return tuple(out)


def _floats_aligned(cells: Sequence[str]) -> list[float] | None:
    """Per-row floats with NaN for blanks (kept aligned for arithmetic checks)."""
    out: list[float] = []
    for v in cells:
        if v == "":
            out.append(math.nan)
            continue
        try:
            out.append(float(v))
        except ValueError:
            return None
    return out


def _is_sum(c: list[float], a: list[float], b: list[float]) -> bool:
    return all(
        math.isnan(ci) or math.isnan(ai) or math.isnan(bi) or abs(ci - (ai + bi)) < 1e-6
        for ci, ai, bi in zip(c, a, b, strict=True)
    )


def _dependencies(
    binned: dict[str, list[str]],
) -> tuple[tuple[str, str, float], ...]:
    """Normalized mutual information between column pairs, strongest first.

    ``nMI(a, b) = I(a; b) / min(H(a), H(b))`` in [0, 1]: 1 when one perfectly
    predicts the other. Computed on binned values, so a continuous column does not
    spuriously dominate.
    """
    cols = [c for c in binned if _entropy(binned[c]) > 0]
    entropies = {c: _entropy(binned[c]) for c in cols}
    links = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            info = entropies[a] - _conditional_entropy(binned[b], binned[a])
            floor = min(entropies[a], entropies[b])
            nmi = info / floor if floor > 0 else 0.0
            if nmi >= _MI_FLOOR:
                links.append((a, b, round(nmi, 3)))
    links.sort(key=lambda link: -link[2])
    return tuple(links[:_MAX_LINKS])
