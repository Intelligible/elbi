"""Survival / time-to-event gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _chi2_sf, _welch
from ._report import _T, Check, VerificationReport


def verify_survival(
    rows: Sequence[dict[str, Any]], time: str, event: str, group: str
) -> VerificationReport:
    """Verify a survival difference between two groups, honouring censoring.

    Comparing mean observed times ignores censoring and can reverse the conclusion
    when one group is followed for less time. This uses the log-rank test, which
    accounts for who is still at risk at each event time, and flags when a naive
    mean-time comparison would have pointed the other way. The verdict is ``sound``
    (a real survival difference by log-rank), ``inconclusive`` (no significant
    difference), or ``inconclusive`` with a note when censoring is absent or data is
    thin.
    """
    rec = []
    for r in rows:
        g = str(r.get(group, "")).strip()
        t = _as_float(r.get(time))
        e = _as_float(r.get(event))
        if g and t is not None and e is not None and t >= 0 and round(e) in (0, 1):
            rec.append((g, t, round(e)))
    groups = sorted({g for g, _, _ in rec})
    if len(groups) != 2:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, (f"need two groups in '{group}'",)
        )
    a, b = groups
    if (
        sum(1 for g, _, _ in rec if g == a) < 15
        or sum(1 for g, _, _ in rec if g == b) < 15
    ):
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few subjects per group",)
        )
    censored = sum(1 for _, _, e in rec if e == 0)
    chi2, o1, e1 = _logrank(rec, a)
    p = _chi2_sf(chi2, 1)
    checks: list[Check] = []
    held = p <= 0.05
    checks.append(
        Check(
            "log-rank",
            True,
            f"log-rank chi2 = {chi2:.2f}, p = {p:.3g} "
            + ("(groups differ)" if held else "(no significant difference)"),
        )
    )

    # does a naive mean-time comparison disagree with the censoring-aware test?
    ta = [t for g, t, _ in rec if g == a]
    tb = [t for g, t, _ in rec if g == b]
    _naive_diff, naive_t = _welch(ta, tb)
    naive_sig = abs(naive_t) > _T
    if censored > 0 and naive_sig != held:
        checks.append(
            Check(
                "censoring",
                False,
                "a naive mean-time comparison disagrees with the log-rank test "
                "(differential censoring)",
            )
        )

    if held and all(c.survived for c in checks):
        better = a if o1 < e1 else b  # fewer deaths than expected => better survival
        verdict, pivotal = "sound", None
        effect = chi2
        caveats = (f"'{better}' has the better survival by the log-rank test",)
    elif not held:
        verdict, pivotal, effect = (
            "inconclusive",
            "no significant survival difference",
            0.0,
        )
        caveats = (
            (
                f"{censored} of {len(rec)} observations are censored; a mean-time "
                "comparison here would be misleading",
            )
            if censored
            else ("no censoring present",)
        )
    else:
        verdict, pivotal, effect = (
            "unsound",
            (
                "the apparent difference is an artifact of differential censoring; the "
                "censoring-aware (log-rank) test and a naive comparison disagree"
            ),
            chi2,
        )
        caveats = (f"{censored} of {len(rec)} observations are censored",)
    return VerificationReport(verdict, effect, held, tuple(checks), pivotal, caveats)


def _logrank(
    rec: Sequence[tuple[str, float, int]], a: str
) -> tuple[float, float, float]:
    """Log-rank statistic for group ``a`` vs the rest; returns (chi2, O_a, E_a)."""
    death_times = sorted({t for _, t, e in rec if e == 1})
    o_a = 0.0
    e_a = 0.0
    var = 0.0
    for t in death_times:
        at_risk = [(g, e) for g, ti, e in rec if ti >= t]
        n = len(at_risk)
        n_a = sum(1 for g, _ in at_risk if g == a)
        d = sum(1 for g, ti, e in rec if e == 1 and ti == t)
        d_a = sum(1 for g, ti, e in rec if e == 1 and ti == t and g == a)
        if n <= 1:
            continue
        o_a += d_a
        e_a += d * n_a / n
        var += d * (n_a / n) * (1 - n_a / n) * (n - d) / (n - 1)
    chi2 = (o_a - e_a) ** 2 / var if var > 0 else 0.0
    return chi2, o_a, e_a
