"""Probability-calibration gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float
from ._report import Check, VerificationReport
from .classification import _roc_auc

#: Expected calibration error above which the probabilities cannot be read as risks
#: (well-calibrated models sit below ~0.05; the temperature-scaling literature uses
#: 0.1 as the line for "miscalibrated").
_ECE_MAX = 0.1


def verify_calibration(
    rows: Sequence[dict[str, Any]], probability: str, outcome: str
) -> VerificationReport:
    """Verify that predicted probabilities can be trusted as risks, not just ranks.

    A model can rank cases well (high ROC-AUC) yet be badly miscalibrated (its "0.9"
    cohort is right only 60% of the time), so any decision that treats the score as a
    literal probability is wrong. This bins the scores and compares mean predicted
    probability to the observed rate (expected calibration error). The verdict is
    ``sound`` (calibrated, ECE below the threshold), ``unsound`` (miscalibrated; the
    score ranks but cannot be used as a probability), or ``inconclusive`` (too little
    data, or the score has no ranking signal to calibrate).
    """
    pairs = [
        (p, round(y))
        for r in rows
        if (p := _as_float(r.get(probability))) is not None
        and (y := _as_float(r.get(outcome))) is not None
    ]
    pairs = [(p, y) for p, y in pairs if 0.0 <= p <= 1.0 and y in (0, 1)]
    if len(pairs) < 100:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few scored rows in [0, 1]",)
        )
    n = len(pairs)
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, y) for p, y in pairs if (lo <= p < hi or (b == bins - 1 and p == hi))
        ]
        if members:
            conf = math.fsum(p for p, _ in members) / len(members)
            acc = math.fsum(y for _, y in members) / len(members)
            ece += len(members) / n * abs(acc - conf)

    auc = _roc_auc([(p, y) for p, y in pairs])
    checks = [
        Check(
            "calibration",
            ece <= _ECE_MAX,
            f"expected calibration error {ece:.3f}"
            + (
                ""
                if ece <= _ECE_MAX
                else ": the score is not a trustworthy probability"
            ),
        )
    ]
    if ece <= _ECE_MAX:
        verdict, pivotal = "sound", None
    else:
        verdict, pivotal = (
            "unsound",
            (
                f"the probabilities are miscalibrated (ECE {ece:.2f}); the model ranks "
                f"(ROC-AUC {auc:.2f}) but its scores cannot be used as risks without "
                "recalibration"
            ),
        )
    caveats = (f"the score ranks cases with ROC-AUC {auc:.2f}",)
    return VerificationReport(
        verdict, ece, verdict == "sound", tuple(checks), pivotal, caveats
    )
