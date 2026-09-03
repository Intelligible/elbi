"""Measurement-reliability gate: internal consistency of a multi-item scale.

A claim that several columns measure one underlying construct (a survey scale, a set of
repeated readings) is only sound if they actually agree. Cronbach's alpha is the
standard internal-consistency coefficient: the proportion of the total-score variance
that is not item-specific noise. An unreliable scale (low alpha) attenuates every
downstream correlation and effect, so a conclusion built on it is built on sand. The
gate flags alpha below the conventional 0.7, and notes when an implausibly high alpha
(>0.95) signals redundant items rather than a good scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _mean_var
from ._report import Check, VerificationReport

#: The conventional acceptable internal-consistency threshold (Nunnally 1978).
_ACCEPTABLE = 0.7
#: Above this, alpha signals redundant items rather than a better scale.
_REDUNDANT = 0.95


def verify_reliability(
    rows: Sequence[dict[str, Any]], items: Sequence[str]
) -> VerificationReport:
    """Verify that ``items`` form an internally consistent scale (Cronbach's alpha).

    Uses the rows where every item is present, sums the item variances against the
    total-score variance, and reports alpha. The verdict is ``sound`` (alpha at or above
    0.7, a reliable scale), ``unsound`` (below 0.7, the items do not cohere: any
    construct built on them is unreliable), or ``inconclusive`` (fewer than three items
    or too few complete rows). A redundancy note fires when alpha exceeds 0.95.
    """
    items = list(dict.fromkeys(items))
    if len(items) < 3:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("internal consistency needs three or more scale items",),
        )
    matrix: list[list[float]] = []
    for row in rows:
        vals = [_as_float(row.get(it)) for it in items]
        if all(v is not None for v in vals):
            matrix.append([v for v in vals if v is not None])
    if len(matrix) < 10:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few rows with every item present",),
        )
    k = len(items)
    item_var = math.fsum(_mean_var([row[j] for row in matrix])[1] for j in range(k))
    totals = [math.fsum(row) for row in matrix]
    total_var = _mean_var(totals)[1]
    if total_var <= 0:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("no variance in the total score",)
        )
    alpha = (k / (k - 1)) * (1.0 - item_var / total_var)
    reliable = alpha >= _ACCEPTABLE
    checks = (
        Check("internal-consistency", reliable, f"Cronbach's alpha = {alpha:.2f}"),
    )
    pivotal = (
        None
        if reliable
        else (
            f"Cronbach's alpha is {alpha:.2f} (below 0.7): the items do not measure "
            "one construct reliably, so any score built from them is noisy"
        )
    )
    note = (
        f"alpha = {alpha:.2f} over {k} items; an alpha above 0.95 can mean the items "
        "are redundant rather than the scale being better"
        if alpha > _REDUNDANT
        else f"alpha = {alpha:.2f} over {k} items"
    )
    verdict = "sound" if reliable else "unsound"
    return VerificationReport(verdict, alpha, True, checks, pivotal, (note,))
