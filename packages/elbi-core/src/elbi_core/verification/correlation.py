"""Linear-association gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _autocorr1,
    _corr,
    _corr_t,
    _frame,
    _iqr_mask,
    _numeric_columns,
    _residual_on,
)
from ._report import (
    _MIN_KEEP,
    _RANDOM_WALK,
    _T,
    Check,
    VerificationReport,
)


def verify_correlation(
    rows: Sequence[dict[str, Any]], x: str, y: str
) -> VerificationReport:
    """Verify a robust linear association between ``x`` and ``y`` (not causation).

    Reports the correlation if it is significant, survives outlier removal, and is
    not explained away by an observed covariate; always caveats that association is
    not causation.
    """
    pool = [c for c in _numeric_columns(rows) if c not in (x, y)]
    data, n = _frame(rows, [x, y, *pool])
    if n < 30 or x not in data or y not in data:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few numeric rows",)
        )
    pool = [c for c in pool if c in data]
    r = _corr(data[x], data[y])
    if abs(_corr_t(r, n)) <= _T:
        return VerificationReport(
            "inconclusive",
            r,
            False,
            (),
            None,
            (f"no significant linear association between '{x}' and '{y}'",),
        )
    positive = r > 0
    checks: list[Check] = []
    pivotal: str | None = None

    keep = _iqr_mask(data, (x, y))
    if sum(keep) >= _MIN_KEEP * n:
        xs = [data[x][i] for i in range(n) if keep[i]]
        ys = [data[y][i] for i in range(n) if keep[i]]
        r_kept = _corr(xs, ys)
        held = abs(_corr_t(r_kept, len(xs))) > _T and (r_kept > 0) == positive
        checks.append(
            Check(
                "outliers",
                held,
                "survives IQR outlier removal"
                if held
                else "association vanishes without outliers",
            )
        )
        if not held:
            pivotal = pivotal or "the association is driven by outliers"

    # Spurious regression: if both series are non-stationary (random-walk-like in row
    # order), their levels can correlate from shared drift alone. The association must
    # survive differencing; if the differenced series no longer move together, the
    # level correlation is spurious (Granger-Newbold). Shuffled cross-sectional data
    # has near-zero row-order autocorrelation and never reaches this check.
    if _autocorr1(data[x]) > _RANDOM_WALK and _autocorr1(data[y]) > _RANDOM_WALK:
        dx = [data[x][i] - data[x][i - 1] for i in range(1, n)]
        dy = [data[y][i] - data[y][i - 1] for i in range(1, n)]
        rd = _corr(dx, dy)
        held = abs(_corr_t(rd, len(dx))) > _T and (rd > 0) == positive
        checks.append(
            Check(
                "stationarity",
                held,
                "the association survives differencing (not a shared trend)"
                if held
                else "spurious: both series are non-stationary and the association "
                "vanishes after differencing",
            )
        )
        if not held:
            pivotal = pivotal or (
                f"spurious regression: '{x}' and '{y}' are non-stationary "
                "(random-walk-like) and their association disappears once differenced"
            )

    # A covariate is a confounder if holding it fixed overturns the association:
    # the partial correlation loses significance, flips sign, or collapses toward
    # zero (attenuation catches a spurious link the significance test misses at the
    # margin, where the residual noises correlate by chance).
    confound: str | None = None
    for z in pool:
        partial = _corr(_residual_on(data, x, z), _residual_on(data, y, z))
        overturned = (
            abs(_corr_t(partial, n)) <= _T
            or (partial > 0) != positive
            or abs(partial) < 0.5 * abs(r)
        )
        if overturned:
            confound = z
            break
    if pool:
        held = confound is None
        checks.append(
            Check(
                "confounding",
                held,
                "no observed covariate explains it away"
                if held
                else f"'{confound}' explains the association",
            )
        )
        if not held:
            pivotal = (
                pivotal or f"the association disappears controlling for '{confound}'"
            )

    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    caveats = (
        "correlation is not causation; an association does not establish that one "
        "variable drives the other",
    )
    return VerificationReport(verdict, r, True, tuple(checks), pivotal, caveats)
