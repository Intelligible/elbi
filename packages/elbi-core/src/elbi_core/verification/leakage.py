"""Data-leakage gate: a feature that is implausibly predictive of the target.

Leakage (a feature that is a proxy or consequence of the label, unavailable at
prediction time) is the most frequent error in applied ML (Kapoor & Narayanan 2023) and
is invisible at runtime: the model simply scores suspiciously well. Whether a feature is
*illegitimate* needs provenance the data cannot supply, so this gate does not assert
leakage; it flags a feature whose single-handed predictive power of the target is
implausibly high (a near-perfect AUC for a binary target, or correlation for a
continuous one) for a human to confirm: the taught "if it looks too good to be true,
it's leakage" heuristic made mechanical.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _corr
from ._report import Check, VerificationReport

#: A single feature predicting the target this well is implausible without leakage.
_SUSPECT = 0.95


def verify_leakage(
    rows: Sequence[dict[str, Any]],
    target: str,
    features: Sequence[str] | None = None,
) -> VerificationReport:
    """Flag any feature that predicts ``target`` implausibly well on its own.

    For a binary target each numeric feature's single-feature AUC is computed; for a
    continuous target, its absolute correlation. A feature at or above 0.95 is flagged
    as likely leakage: a proxy for the label rather than a legitimate predictor. The
    verdict is ``unsound`` (a leak-suspect feature, named, to confirm against its
    provenance), ``sound`` (no feature is implausibly predictive), or ``inconclusive``
    (no usable target or features).
    """
    cols = list(features) if features else [c for c in rows[0] if c != target]
    pairs: list[tuple[float, float]] = []  # (feature, target) for present rows
    target_vals = [_as_float(r.get(target)) for r in rows]
    binary = _is_binary([v for v in target_vals if v is not None])
    flagged: list[tuple[str, float]] = []
    for col in cols:
        if col == target:
            continue
        fv: list[float] = []
        tv: list[float] = []
        for r in rows:
            f = _as_float(r.get(col))
            t = _as_float(r.get(target))
            if f is not None and t is not None:
                fv.append(f)
                tv.append(t)
        if len(fv) < 20:
            continue
        power = _auc(fv, tv) if binary else abs(_corr(fv, tv))
        pairs.append((power, 0.0))
        if power >= _SUSPECT:
            flagged.append((col, power))
    if not pairs:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("no numeric features to screen against the target",),
        )
    flagged.sort(key=lambda t: t[1], reverse=True)
    worst = max(p for p, _ in pairs)
    checks = (
        Check(
            "no-leakage",
            not flagged,
            "no feature predicts the target implausibly well"
            if not flagged
            else f"{flagged[0][0]} predicts the target at {flagged[0][1]:.3f}",
        ),
    )
    if flagged:
        names = ", ".join(f"'{c}' ({p:.3f})" for c, p in flagged[:3])
        pivotal = (
            f"feature(s) {names} predict the target near-perfectly: likely leakage "
            "(a proxy or consequence of the label); confirm each is available at "
            "prediction time before trusting the model"
        )
        return VerificationReport("unsound", worst, True, checks, pivotal, ())
    return VerificationReport(
        "sound",
        worst,
        False,
        checks,
        None,
        (
            "leakage from provenance (a label built from future data) is not "
            "detectable here: this only screens single-feature predictive power",
        ),
    )


def _is_binary(values: Sequence[float]) -> bool:
    return bool(values) and {round(v) for v in values} <= {0, 1} and len({*values}) == 2


def _auc(feature: Sequence[float], labels: Sequence[float]) -> float:
    """Single-feature AUC for a binary target (rank form), folded to ``max(a, 1-a)``."""
    order = sorted(range(len(feature)), key=lambda i: feature[i])
    ranks = [0.0] * len(feature)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and feature[order[j + 1]] == feature[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [ranks[i] for i in range(len(labels)) if round(labels[i]) == 1]
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = (math.fsum(pos) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(auc, 1.0 - auc)
