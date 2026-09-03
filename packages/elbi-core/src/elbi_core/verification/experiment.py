"""Online experiment (A/B test) gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _as_float,
    _categorical_columns,
    _chi2_sf,
    _cohens_d,
    _ols,
    _p_two_sided,
    _simpson_reversal,
    _welch,
)
from ._report import _T, Check, VerificationReport

#: A sample-ratio mismatch is declared at this p-value. It is deliberately strict
#: (about 1 in 2000) because a true mismatch invalidates the whole experiment, so a
#: false alarm is costly; Microsoft's ExP uses the same 0.0005 convention.
_SRM = 0.0005


def verify_experiment(
    rows: Sequence[dict[str, Any]],
    variant: str,
    metric: str,
    *,
    expected_ratio: Sequence[float] | None = None,
    period: str | None = None,
) -> VerificationReport:
    """Verify an A/B test result before trusting it.

    First checks for a sample-ratio mismatch: the observed split across the two variants
    must match the intended allocation (``expected_ratio``, defaulting to even). A
    mismatch means randomization or logging is broken and the whole comparison is void,
    so it is reported ``unsound`` regardless of the metric. If the split is sound, the
    difference in ``metric`` between the variants is tested (Welch), with its effect
    size, and checked against Simpson's paradox: a pooled lift that reverses inside a
    segment is confounded by that segment's mix. The verdict is ``sound`` (clean split
    and a real difference that holds within every segment), ``unsound`` (sample-ratio
    mismatch, or a within-segment reversal), or ``inconclusive`` (clean split but no
    significant difference).
    """
    pairs: list[tuple[str, float]] = []
    for row in rows:
        label = str(row.get(variant, "")).strip()
        value = _as_float(row.get(metric))
        if label and value is not None:
            pairs.append((label, value))
    labels = sorted({v for v, _ in pairs})
    if len(labels) != 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"an A/B test needs two variants in '{variant}', found {len(labels)}",),
        )
    a, b = labels
    counts = [sum(1 for v, _ in pairs if v == a), sum(1 for v, _ in pairs if v == b)]
    total = counts[0] + counts[1]
    ratio = list(expected_ratio) if expected_ratio else [0.5, 0.5]
    share = [r / sum(ratio) for r in ratio]
    expected = [total * share[0], total * share[1]]
    srm_stat = sum((o - e) ** 2 / e for o, e in zip(counts, expected, strict=True) if e)
    srm_p = _chi2_sf(srm_stat, 1)
    checks: list[Check] = []
    if srm_p < _SRM:
        pct = counts[0] / total
        checks.append(
            Check(
                "sample-ratio",
                False,
                f"observed split {pct:.0%}/{1 - pct:.0%} differs from the intended "
                f"{share[0]:.0%}/{share[1]:.0%} (p = {srm_p:.1g})",
            )
        )
        return VerificationReport(
            "unsound",
            0.0,
            False,
            tuple(checks),
            "sample-ratio mismatch: randomization or logging is broken, so the "
            "experiment result cannot be trusted at all, fix the split and re-run",
            ("a sample-ratio mismatch voids the comparison regardless of the metric",),
        )
    checks.append(
        Check(
            "sample-ratio",
            True,
            f"split matches the intended allocation (p = {srm_p:.2g})",
        )
    )

    av = [v for lab, v in pairs if lab == a]
    bv = [v for lab, v in pairs if lab == b]
    if len(av) < 10 or len(bv) < 10:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            tuple(checks),
            None,
            ("too few rows in a variant",),
        )
    diff, t = _welch(av, bv)
    if abs(t) <= _T:
        return VerificationReport(
            "inconclusive",
            diff,
            False,
            tuple(checks),
            None,
            (f"no significant difference in '{metric}' between '{a}' and '{b}'",),
        )
    # report the second variant relative to the first (lift = mean(b) - mean(a));
    # _welch returns mean(a) - mean(b), so negate it for the "b vs a" framing
    lift = -diff
    d = -_cohens_d(av, bv)
    p = _p_two_sided(t)
    caveats = (
        f"lift in '{metric}' for '{b}' vs '{a}' is {lift:+.3g} (Cohen's d = {d:+.2f}, "
        f"p = {p:.2g}); judge practical significance against your minimum detectable "
        "effect, and guard against peeking and multiple metrics over time",
    )

    # Simpson's paradox: a pooled lift that significantly reverses inside a segment is
    # an artifact of that segment's mix, not a real win, the flagship A/B trap. Scan
    # every categorical column (not only one the caller named) so the guard does not
    # depend on being handed the right segment.
    reversal = _simpson_reversal(
        rows,
        variant,
        metric,
        a,
        b,
        diff > 0,
        # the period is the time axis for the novelty check, not a segment: excluding it
        # keeps a decaying effect from reading as a within-period Simpson reversal
        _categorical_columns(rows, exclude=(variant, metric, period)),
    )
    if reversal is not None:
        col, level = reversal
        checks.append(
            Check("subgroup-consistency", False, f"reverses within '{col}' = {level}")
        )
        return VerificationReport(
            "unsound",
            lift,
            True,
            tuple(checks),
            f"Simpson's paradox: the lift for '{b}' reverses within '{col}' = {level}, "
            f"so the pooled result is confounded by '{col}', compare within segments",
            caveats,
        )
    checks.append(Check("subgroup-consistency", True, "the lift holds within segments"))

    # Novelty / primacy: when the test spans time, a lift that fades is a novelty
    # effect, not a durable one (Sadeghi & Gupta 2022). Fit a time-by-treatment
    # interaction over all rows (a threshold-free decay slope that also catches a
    # gradual decline) and require the late-window lift to still hold. A lift that is
    # significantly declining and no longer significant by the end is not durable.
    if period is not None:
        stamped = [
            (lab, v, pv)
            for row in rows
            if (lab := str(row.get(variant, "")).strip()) in (a, b)
            and (v := _as_float(row.get(metric))) is not None
            and (pv := _as_float(row.get(period))) is not None
        ]
        periods = sorted({pv for _, _, pv in stamped})
        if len(periods) >= 4 and len(stamped) >= 40:
            g = [1.0 if lab == b else 0.0 for lab, _, _ in stamped]
            data = {
                "m": [v for _, v, _ in stamped],
                "g": g,
                "t": [pv for _, _, pv in stamped],
                "gt": [gi * pv for gi, (_, _, pv) in zip(g, stamped, strict=True)],
            }
            # the treatment-by-time interaction: how the lift changes each period
            slope, slope_t = _ols(data, "m", "gt", ["g", "t"])
            per_period = slope  # decay rate in metric units per period
            mid = periods[len(periods) // 2]
            la = [v for lab, v, pv in stamped if lab == a and pv >= mid]
            lb = [v for lab, v, pv in stamped if lab == b and pv >= mid]
            late_lift, late_t = (
                _welch(lb, la)
                if len(la) >= 10 and len(lb) >= 10
                else (
                    0.0,
                    0.0,
                )
            )
            # the lift shrinks toward zero over time (a negative slope when the pooled
            # lift is positive, a positive slope when it is negative), so the verdict
            # does not depend on which arm happens to sort first
            declining = abs(slope_t) > _T and (slope > 0) != (lift > 0)
            durable_late = abs(late_t) > _T and (late_lift > 0) == (lift > 0)
            if declining and not durable_late:
                checks.append(
                    Check(
                        "durability",
                        False,
                        f"lift declines {per_period:+.3g}/period "
                        f"(t = {slope_t:+.1f}); late-window lift {late_lift:+.3g} "
                        f"(t = {late_t:+.1f})",
                    )
                )
                return VerificationReport(
                    "inconclusive",
                    lift,
                    True,
                    tuple(checks),
                    f"novelty effect: the lift declines over '{period}' (interaction "
                    f"t = {slope_t:+.1f}) and is no longer significant by the end "
                    f"(late-window lift {late_lift:+.3g}); the durable effect is not "
                    "established, so extend the test or model the decay",
                    caveats,
                )
            checks.append(
                Check(
                    "durability", True, "the lift does not significantly decline away"
                )
            )

    # A clean split plus a significant difference that survives every segment and
    # persists over time is a sound experiment result. Whether the lift is *practically*
    # large needs a business threshold (a minimum detectable effect) the data cannot
    # supply, and a standardized effect size badly understates a binary conversion
    # metric (a 30% relative lift has a small Cohen's d), so the size is reported, never
    # used to fail an otherwise-valid result.
    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    return VerificationReport(verdict, lift, True, tuple(checks), None, caveats)
