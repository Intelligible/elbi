"""Missing-data gate: is a column's missingness completely at random (MCAR)?

Dropping rows with missing values, or mean-imputing them, is only unbiased when the data
is missing completely at random: when whether a value is missing does not depend on the
other observed columns. If missingness *is* predicted by the observed data (it is at
most missing-at-random), those shortcuts bias every downstream estimate. Little's test
needs EM over the covariance matrix; the stdlib-feasible equivalent (and the one that
localises the culprit) is to test whether each observed column predicts the missingness
indicator (a two-sample comparison for numeric columns, a chi-square for categorical)
and reject MCAR if any does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _chi2_sf, _mann_whitney, _p_two_sided
from ._report import Check, VerificationReport


def verify_missingness(
    rows: Sequence[dict[str, Any]], column: str
) -> VerificationReport:
    """Verify that ``column`` is missing completely at random (safe to drop or impute).

    Builds the missingness indicator for ``column`` and tests whether any other column
    predicts it. The verdict is ``unsound`` (missingness depends on an observed column,
    named: it is not MCAR, so dropping or mean-imputing biases the result; use multiple
    imputation), ``sound`` (no observed column predicts missingness, MCAR is not
    contradicted), or ``inconclusive`` (too few missing or present values).
    """
    present = [str(r.get(column, "")).strip() != "" for r in rows]
    n_missing = present.count(False)
    if n_missing < 10 or present.count(True) < 10:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("need at least 10 missing and 10 present values to test MCAR",),
        )
    others = [c for c in rows[0] if c != column]
    tested: list[tuple[str, float]] = []
    for col in others:
        p = _predicts_missingness(rows, col, present)
        if p is not None:
            tested.append((col, p))
    if not tested:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("no other columns to test missingness against",),
        )
    # Bonferroni across the columns tested, so one chance hit does not reject MCAR
    threshold = 0.05 / len(tested)
    culprits = sorted((c, p) for c, p in tested if p < threshold)
    rate = n_missing / len(rows)
    if culprits:
        col, p = culprits[0]
        checks = (
            Check("mcar", False, f"missingness depends on '{col}' (p = {p:.2g})"),
        )
        pivotal = (
            f"whether '{column}' is missing depends on '{col}'; the data is not "
            "missing completely at random, so dropping or mean-imputing these rows "
            "biases the result; use multiple imputation"
        )
        return VerificationReport("unsound", rate, True, checks, pivotal, ())
    checks = (Check("mcar", True, "no observed column predicts missingness"),)
    return VerificationReport(
        "sound",
        rate,
        True,
        checks,
        None,
        (
            f"{rate:.0%} missing; MCAR is not contradicted, but missingness on an "
            "unobserved cause (MNAR) cannot be ruled out from data",
        ),
    )


def _predicts_missingness(
    rows: Sequence[dict[str, Any]], col: str, present: Sequence[bool]
) -> float | None:
    """p-value that ``col`` differs between present-target and missing-target rows."""
    numeric: list[tuple[float, bool]] = []
    for r, here in zip(rows, present, strict=True):
        v = _as_float(r.get(col))
        if v is not None:
            numeric.append((v, here))
    if len(numeric) >= len(rows) * 0.8:  # treat as numeric when mostly parseable
        a = [v for v, here in numeric if here]
        b = [v for v, here in numeric if not here]
        if len(a) >= 10 and len(b) >= 10:
            _, z = _mann_whitney(a, b)
            return _p_two_sided(z)
        return None
    # categorical: chi-square of (level x present/missing)
    levels: dict[str, list[int]] = {}
    for r, here in zip(rows, present, strict=True):
        lvl = str(r.get(col, "")).strip()
        if lvl:
            levels.setdefault(lvl, [0, 0])[0 if here else 1] += 1
    if len(levels) < 2:
        return None
    return _chi2_independence(levels)


def _chi2_independence(levels: dict[str, list[int]]) -> float:
    """Chi-square p-value for a (level x present/missing) contingency table."""
    rows_tot = {lvl: counts[0] + counts[1] for lvl, counts in levels.items()}
    col_tot = [
        math.fsum(c[0] for c in levels.values()),
        math.fsum(c[1] for c in levels.values()),
    ]
    n = sum(rows_tot.values())
    if n == 0 or min(col_tot) == 0:
        return 1.0
    chi2 = 0.0
    for lvl, counts in levels.items():
        for j in (0, 1):
            exp = rows_tot[lvl] * col_tot[j] / n
            if exp > 0:
                chi2 += (counts[j] - exp) ** 2 / exp
    return _chi2_sf(chi2, len(levels) - 1)
