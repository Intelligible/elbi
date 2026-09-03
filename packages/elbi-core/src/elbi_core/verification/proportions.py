"""Proportion / contingency-table association gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _chi2_sf
from ._report import Check, VerificationReport


def verify_proportions(
    rows: Sequence[dict[str, Any]], group: str, outcome: str
) -> VerificationReport:
    """Verify a categorical association before reporting it.

    Builds the ``group`` x ``outcome`` contingency table and tests independence.
    The chi-square approximation is only valid when expected cell counts are large
    (Cochran's rule), so on a sparse table it is replaced by an exact test (Fisher
    for 2x2) or a deterministic label permutation test. The headline failure caught:
    a chi-square that reads significant only because its large-count approximation is
    invalid. The verdict is ``sound`` (a valid test finds the association), ``unsound``
    (chi-square claims significance but the exact test does not), or ``inconclusive``
    (no significant association).
    """
    glevels = sorted(
        {str(r.get(group, "")).strip() for r in rows if str(r.get(group, "")).strip()}
    )
    olevels = sorted(
        {
            str(r.get(outcome, "")).strip()
            for r in rows
            if str(r.get(outcome, "")).strip()
        }
    )
    if len(glevels) < 2 or len(olevels) < 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("need at least two levels in each column",),
        )
    table = [
        [
            sum(
                1
                for r in rows
                if str(r.get(group, "")).strip() == g
                and str(r.get(outcome, "")).strip() == o
            )
            for o in olevels
        ]
        for g in glevels
    ]
    n = sum(sum(row) for row in table)
    if n < 12:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few observations",)
        )
    chi2, dof, expected = _contingency(table)
    chi_p = _chi2_sf(chi2, dof)
    cramers_v = math.sqrt(chi2 / (n * min(len(glevels) - 1, len(olevels) - 1)))

    # Cochran's rule: the chi-square approximation needs most expected counts >= 5
    flat = [e for row in expected for e in row]
    sparse = sum(1 for e in flat if e < 5) > 0.2 * len(flat) or min(flat) < 1
    checks: list[Check] = []
    if not sparse:
        held = chi_p <= 0.05
        checks.append(
            Check("validity", True, "expected counts adequate for the chi-square test")
        )
        verdict = "sound" if held else "inconclusive"
        pivotal = None if held else "no significant association"
    else:
        exact_p = (
            _fisher_2x2(table) if len(glevels) == 2 == len(olevels) else _perm_p(table)
        )
        chi_sig = chi_p <= 0.05
        exact_sig = exact_p <= 0.05
        held = not (chi_sig and not exact_sig)
        checks.append(
            Check(
                "validity",
                held,
                f"sparse table (min expected {min(flat):.1f}); "
                f"exact test p = {exact_p:.3g}"
                + ("" if held else f" does not confirm chi-square p = {chi_p:.3g}"),
            )
        )
        if not held:
            verdict, pivotal = (
                "unsound",
                (
                    "chi-square is invalid on this sparse table and the exact test "
                    "is not significant, no association is established"
                ),
            )
        elif exact_sig:
            verdict, pivotal = "sound", None
        else:
            verdict, pivotal = "inconclusive", "no significant association"

    caveats = (f"Cramer's V = {cramers_v:.2f}; an association is not by itself causal",)
    return VerificationReport(
        verdict, cramers_v, verdict == "sound", tuple(checks), pivotal, caveats
    )


def _contingency(table: list[list[int]]) -> tuple[float, int, list[list[float]]]:
    """Pearson chi-square statistic, degrees of freedom, and expected counts."""
    n = sum(sum(row) for row in table)
    row_tot = [sum(row) for row in table]
    col_tot = [
        sum(table[r][c] for r in range(len(table))) for c in range(len(table[0]))
    ]
    expected = [
        [row_tot[r] * col_tot[c] / n for c in range(len(col_tot))]
        for r in range(len(row_tot))
    ]
    chi2 = math.fsum(
        (table[r][c] - expected[r][c]) ** 2 / expected[r][c]
        for r in range(len(table))
        for c in range(len(table[0]))
        if expected[r][c] > 0
    )
    dof = (len(row_tot) - 1) * (len(col_tot) - 1)
    return chi2, dof, expected


def _fisher_2x2(table: list[list[int]]) -> float:
    """Two-sided Fisher exact p-value for a 2x2 table (hypergeometric)."""
    a, b = table[0]
    c, d = table[1]
    row1, row2, col1 = a + b, c + d, a + c
    n = a + b + c + d

    def hyper(k: int) -> float:
        return math.comb(row1, k) * math.comb(row2, col1 - k) / math.comb(n, col1)

    p_obs = hyper(a)
    lo = max(0, col1 - row2)
    hi = min(col1, row1)
    return math.fsum(
        p for k in range(lo, hi + 1) if (p := hyper(k)) <= p_obs * (1 + 1e-9)
    )


#: Constants for the permutation test's LCG shuffle (a reproducible RNG).
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_MASK64 = (1 << 64) - 1


def _perm_p(table: list[list[int]], reps: int = 1000) -> float:
    """Deterministic permutation p-value for an RxC table (label reshuffling).

    The column labels are repeatedly Fisher-Yates shuffled against the fixed row
    labels (breaking any association) with a seeded LCG, so the result is exact-test-
    like yet reproducible without a global RNG.
    """
    rows: list[int] = []
    cols: list[int] = []
    for r in range(len(table)):
        for c in range(len(table[0])):
            rows.extend([r] * table[r][c])
            cols.extend([c] * table[r][c])
    n = len(rows)
    obs = _contingency(table)[0]
    nr, nc = len(table), len(table[0])
    ge = 0
    for rep in range(reps):
        shuffled = cols[:]
        state = (rep * 2654435761 + _LCG_C) & _MASK64
        for i in range(n - 1, 0, -1):
            state = (_LCG_A * state + _LCG_C) & _MASK64
            j = state % (i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        perm = [[0] * nc for _ in range(nr)]
        for i in range(n):
            perm[rows[i]][shuffled[i]] += 1
        if _contingency(perm)[0] >= obs - 1e-9:
            ge += 1
    return (1 + ge) / (reps + 1)
