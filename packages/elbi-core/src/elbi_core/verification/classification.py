"""Classifier-evaluation gate (imbalance, baseline skill, ranking)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float
from ._report import Check, VerificationReport


def verify_classification(
    rows: Sequence[dict[str, Any]],
    y_true: str,
    *,
    y_pred: str | None = None,
    y_score: str | None = None,
) -> VerificationReport:
    """Verify that a classifier has real skill, not just the accuracy paradox.

    On imbalanced data a model that always predicts the majority class scores high
    accuracy while learning nothing. Given predictions (``y_pred``), skill is balanced
    accuracy (the majority-class rule scores 0.5 on it at any prevalence) with raw
    accuracy and the majority baseline reported alongside. Given scores (``y_score``),
    ranking must beat the prevalence baseline (PR) and chance (ROC). The verdict is
    ``sound`` (skill beyond the baseline) or ``inconclusive``, naming which of the two
    fell short.
    """
    yt = [round(v) for r in rows if (v := _as_float(r.get(y_true))) is not None]
    if len(yt) < 40 or set(yt) - {0, 1}:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("need a binary 0/1 target with enough rows",),
        )
    n = len(yt)
    prevalence = sum(yt) / n
    baseline = max(prevalence, 1 - prevalence)
    checks: list[Check] = []
    skill = 0.0

    if y_pred is not None:
        yp = [round(v) for r in rows if (v := _as_float(r.get(y_pred))) is not None]
        if len(yp) == n:
            acc = sum(1 for a, b in zip(yt, yp, strict=True) if a == b) / n
            tpr = _rate(yt, yp, 1)
            tnr = _rate(yt, yp, 0)
            bal = (tpr + tnr) / 2
            # Balanced accuracy alone, because the majority-class rule scores exactly
            # 0.5 on it at *any* prevalence, which is what makes it the imbalance-proof
            # test. Also requiring raw accuracy to beat the majority baseline would
            # demand the very behaviour this gate exists to catch: at 5% positives the
            # baseline is 0.95, so only a model that almost never predicts the positive
            # class can clear it, and a threshold chosen for recall is failed for being
            # correct. Accuracy is still reported: worth seeing, not worth deciding on.
            held = bal >= 0.55
            skill = max(skill, bal - 0.5)
            checks.append(
                Check(
                    "skill",
                    held,
                    f"accuracy {acc:.2f} vs majority baseline {baseline:.2f}; "
                    f"balanced accuracy {bal:.2f}"
                    + (
                        ""
                        if held
                        else " (no skill beyond predicting the majority class)"
                    ),
                )
            )

    if y_score is not None:
        scored = [
            (s, yt[i])
            for i, r in enumerate(rows)
            if i < n and (s := _as_float(r.get(y_score))) is not None
        ]
        if len(scored) == n:
            auc = _roc_auc(scored)
            pr = _pr_auc(scored)
            held = pr - prevalence >= 0.05 and auc >= 0.55
            skill = max(skill, auc - 0.5)
            checks.append(
                Check(
                    "ranking",
                    held,
                    f"ROC-AUC {auc:.2f}; PR-AUC {pr:.2f} vs prevalence {prevalence:.2f}"
                    + ("" if held else " (scores do not rank the positive class)"),
                )
            )

    if not checks:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("supply y_pred and/or y_score to evaluate",),
        )
    failed = {c.name for c in checks if not c.survived}
    if not failed:
        verdict, pivotal = "sound", None
    elif "skill" in failed:
        verdict, pivotal = (
            "inconclusive",
            "the model's decisions have no skill beyond the majority class: it "
            "separates the two classes no better than answering with the common one",
        )
    else:
        verdict, pivotal = (
            "inconclusive",
            "the scores do not rank the positive class above the negative one by "
            "enough to beat prevalence, so the ranking carries no usable signal",
        )
    caveats = (f"the positive class is {prevalence:.0%} of the data",)
    return VerificationReport(
        verdict, skill, verdict == "sound", tuple(checks), pivotal, caveats
    )


def _rate(yt: Sequence[int], yp: Sequence[int], cls: int) -> float:
    """Per-class accuracy (recall for ``cls``)."""
    idx = [i for i, a in enumerate(yt) if a == cls]
    if not idx:
        return 0.0
    return sum(1 for i in idx if yp[i] == cls) / len(idx)


def _roc_auc(scored: Sequence[tuple[float, int]]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) identity."""
    order = sorted(range(len(scored)), key=lambda i: scored[i][0])
    ranks = [0.0] * len(scored)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scored[order[j + 1]][0] == scored[order[i]][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [idx for idx in range(len(scored)) if scored[idx][1] == 1]
    npos, nneg = len(pos), len(scored) - len(pos)
    if npos == 0 or nneg == 0:
        return 0.5
    return (math.fsum(ranks[i] for i in pos) - npos * (npos + 1) / 2) / (npos * nneg)


def _pr_auc(scored: Sequence[tuple[float, int]]) -> float:
    """Average precision (area under the precision-recall curve)."""
    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    total_pos = sum(1 for _, y in scored if y == 1)
    if total_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if scored[i][1] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap
