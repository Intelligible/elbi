"""Fairness gate: subgroup disparity in a binary classifier's errors.

A model can be accurate overall yet treat groups unequally. This gate compares, across
the levels of a protected ``group``, the selection rate (demographic parity), the
true-positive rate (equal opportunity), the false-positive rate, and the positive
predictive value (predictive parity), flagging any gap beyond 0.1 or a selection-rate
ratio outside the four-fifths rule. It also surfaces the impossibility result
(Kleinberg et al. 2016; Chouldechova 2017): when base rates differ across groups, equal
PPV and equal error rates cannot both hold, so the gate reports that tension rather
than demanding both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float
from ._report import Check, VerificationReport

#: A rate gap beyond this between groups is flagged as a disparity.
_GAP = 0.1
#: A selection-rate ratio below this violates the four-fifths (80%) rule.
_DI_LOW = 0.8


def verify_fairness(
    rows: Sequence[dict[str, Any]], group: str, y_true: str, y_pred: str
) -> VerificationReport:
    """Verify a binary classifier does not disparately harm a subgroup.

    Splits ``rows`` by the levels of ``group`` and compares the selection rate, true-
    and false-positive rates, and positive predictive value across them. The verdict
    is ``unsound`` (a gap beyond 0.1 or a selection ratio outside the 80% rule, named),
    ``sound`` (parity within tolerance on every metric), or ``inconclusive`` (too few
    groups or rows). When base rates differ, the report notes that predictive parity
    and equal error rates are mathematically incompatible.
    """
    by_group: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        g = str(row.get(group, "")).strip()
        yt, yp = _as_float(row.get(y_true)), _as_float(row.get(y_pred))
        if g and yt is not None and yp is not None:
            by_group.setdefault(g, []).append((round(yt), round(yp)))
    groups = {g: v for g, v in by_group.items() if len(v) >= 20}
    if len(groups) < 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a fairness check needs two or more groups of at least 20",),
        )

    rates = {g: _rates(v) for g, v in groups.items()}
    checks: list[Check] = []
    pivotal: str | None = None
    for metric, label in (
        ("selection", "selection rate (demographic parity)"),
        ("tpr", "true-positive rate (equal opportunity)"),
        ("fpr", "false-positive rate"),
        ("ppv", "positive predictive value (predictive parity)"),
    ):
        vals: dict[str, float] = {}
        for g, r in rates.items():
            mv = r[metric]
            if mv is not None:
                vals[g] = mv
        if len(vals) < 2:
            continue
        hi, lo = max(vals.values()), min(vals.values())
        gap = hi - lo
        held = gap <= _GAP
        checks.append(
            Check(
                metric,
                held,
                f"{label} gap = {gap:.2f}" + ("" if held else " (disparity)"),
            )
        )
        if not held and pivotal is None:
            g_hi = max(vals, key=lambda g: vals[g])
            g_lo = min(vals, key=lambda g: vals[g])
            pivotal = (
                f"{label} differs by {gap:.2f} between '{g_hi}' and '{g_lo}': the "
                "model's benefit or harm is not distributed equally across groups"
            )

    # four-fifths rule on the selection rate
    sel: dict[str, float] = {}
    for g, r in rates.items():
        sv = r["selection"]
        if sv:
            sel[g] = sv
    if len(sel) >= 2 and min(sel.values()) > 0:
        ratio = min(sel.values()) / max(sel.values())
        held = ratio >= _DI_LOW
        checks.append(Check("four-fifths", held, f"selection-rate ratio = {ratio:.2f}"))
        if not held and pivotal is None:
            pivotal = (
                f"selection-rate ratio {ratio:.2f} is below the four-fifths (0.8) "
                "rule: disparate impact"
            )

    base: list[float] = [r["base"] for r in rates.values() if r["base"] is not None]
    incompatible = max(base) - min(base) > _GAP if base else False
    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    note = (
        "base rates differ across groups, so predictive parity and equal error rates "
        "cannot both hold (an impossibility result): choose which to prioritise"
        if incompatible
        else "compares group error rates as observed"
    )
    return VerificationReport(verdict, 0.0, True, tuple(checks), pivotal, (note,))


def _rates(pairs: Sequence[tuple[int, int]]) -> dict[str, float | None]:
    """Selection rate, TPR, FPR, PPV, and base rate from (true, pred) pairs."""
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    n = len(pairs)
    return {
        "selection": (tp + fp) / n if n else None,
        "tpr": tp / (tp + fn) if (tp + fn) else None,
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "ppv": tp / (tp + fp) if (tp + fp) else None,
        "base": (tp + fn) / n if n else None,
    }
