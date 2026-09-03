"""Multiple-comparisons gate: false-discovery-rate control over a family of metrics.

Screening one treatment against many outcomes and reporting the ones that came out
significant is the classic p-hacking trap: testing 20 independent null metrics at p<0.05
yields a false positive about two-thirds of the time. Verifying each metric in isolation
cannot catch this: the fix has to see the whole family at once and correct for its size.
This gate takes the family, tests each outcome, and applies the Benjamini-Hochberg
procedure (Benjamini & Hochberg 1995) so only outcomes whose signal survives
false-discovery-rate control are reported as real.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _benjamini_hochberg, _p_two_sided, _welch
from ._report import Check, VerificationReport


def verify_screen(
    rows: Sequence[dict[str, Any]],
    group: str,
    outcomes: Sequence[str],
    *,
    alpha: float = 0.05,
) -> VerificationReport:
    """Verify which of many ``outcomes`` a two-level ``group`` really affects.

    Each outcome is compared across the two group levels (Welch), and the family of
    p-values is corrected with Benjamini-Hochberg at ``alpha``. The verdict is ``sound``
    (at least one outcome survives the correction: those are the real effects, named),
    ``inconclusive`` (nothing survives: any raw-significant hit is what testing this
    many metrics is expected to throw up by chance), or ``inconclusive`` when the family
    or the group is too small to test.
    """
    labels = sorted(
        {str(r.get(group, "")).strip() for r in rows if str(r.get(group, "")).strip()}
    )
    tested = [c for c in outcomes if c != group]
    if len(labels) != 2 or len(tested) < 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a screen needs a two-level group and at least two outcome columns",),
        )
    a, b = labels
    names: list[str] = []
    pvalues: list[float] = []
    for col in tested:
        av = [
            v
            for r in rows
            if str(r.get(group, "")).strip() == a
            and (v := _as_float(r.get(col))) is not None
        ]
        bv = [
            v
            for r in rows
            if str(r.get(group, "")).strip() == b
            and (v := _as_float(r.get(col))) is not None
        ]
        if len(av) < 10 or len(bv) < 10:
            continue
        _, t = _welch(av, bv)
        names.append(col)
        pvalues.append(_p_two_sided(t))
    if len(pvalues) < 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few rows per group to test the outcome family",),
        )

    survive = _benjamini_hochberg(pvalues, alpha)
    discoveries = [names[i] for i in range(len(names)) if survive[i]]
    raw_hits = [names[i] for i in range(len(names)) if pvalues[i] < alpha]
    smallest = min(pvalues)
    m = len(pvalues)

    if discoveries:
        check = Check(
            "false-discovery-rate",
            True,
            f"{len(discoveries)} of {m} outcomes survive FDR control: "
            f"{', '.join(discoveries)}",
        )
        return VerificationReport(
            "sound",
            float(len(discoveries)),
            True,
            (check,),
            None,
            (
                f"treatment affects {len(discoveries)} of {m} tested outcomes after "
                f"Benjamini-Hochberg correction ({', '.join(discoveries)}); "
                f"{len(raw_hits)} were significant before correction",
            ),
        )
    check = Check(
        "false-discovery-rate",
        False,
        f"0 of {m} outcomes survive FDR control (smallest p = {smallest:.3g})",
    )
    return VerificationReport(
        "inconclusive",
        0.0,
        False,
        (check,),
        f"no outcome survives false-discovery-rate control across {m} tests: the "
        f"{len(raw_hits)} raw-significant result(s) are what testing {m} metrics is "
        "expected to produce by chance, pre-register a primary metric or replicate",
        (f"smallest p = {smallest:.3g}; Benjamini-Hochberg threshold not met",),
    )
