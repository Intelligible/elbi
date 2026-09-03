"""Pure-Python numeric helpers shared by the verification gates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._report import _SIGNAL, _T


def _numeric_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Columns whose present values all parse as floats."""
    if not rows:
        return []
    out = []
    for c in rows[0]:
        try:
            [float(r[c]) for r in rows if r.get(c) not in ("", None)]
            out.append(c)
        except (ValueError, TypeError):
            continue
    return out


def _threshold_balanced_accuracy(
    vals: Sequence[float], labs: Sequence[str], a: str, b: str
) -> float:
    """Best balanced accuracy of a single threshold on ``vals`` predicting the label.

    Sweeps every split of ``vals`` and returns the highest balanced accuracy (the mean
    of the two per-class rates, so class imbalance cannot inflate it) for predicting the
    label from which side of the threshold a point falls, in either direction.
    """
    na, nb = labs.count(a), labs.count(b)
    if na == 0 or nb == 0:
        return 0.0
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    a_below = b_below = 0
    best = 0.0
    for i in order:
        if labs[i] == a:
            a_below += 1
        else:
            b_below += 1
        bal = 0.5 * (a_below / na + (nb - b_below) / nb)
        best = max(best, bal, 1.0 - bal)
    return best


def _selecting_baseline(
    rows: Sequence[dict[str, Any]], group: str, exclude: Sequence[str]
) -> str | None:
    """A numeric column on whose threshold the two-level ``group`` is (near-)determined.

    This is the signature of baseline selection, a group defined by an extreme of some
    measurement (``coached = pre_score < 38``), which is what invites regression to the
    mean. Returns the most separating such column when a single threshold predicts the
    group with balanced accuracy at least 0.9, else ``None`` (a randomized or merely
    correlated group does not cross that bar).
    """
    labels = sorted({g for r in rows if (g := str(r.get(group, "")).strip())})
    if len(labels) != 2:
        return None
    a, b = labels
    skip = {c for c in exclude if c} | {group}
    best: str | None = None
    best_acc = 0.0
    for col in _numeric_columns(rows):
        if col in skip:
            continue
        vals: list[float] = []
        labs: list[str] = []
        for r in rows:
            v = _as_float(r.get(col))
            lab = str(r.get(group, "")).strip()
            if v is not None and lab in (a, b):
                vals.append(v)
                labs.append(lab)
        if len(vals) < 40:
            continue
        acc = _threshold_balanced_accuracy(vals, labs, a, b)
        if acc > best_acc:
            best_acc, best = acc, col
    return best if best_acc >= 0.9 else None


def _categorical_columns(
    rows: Sequence[dict[str, Any]], exclude: Sequence[str | None]
) -> list[str]:
    """Low-cardinality columns usable as a subgroup (2-10 distinct present values)."""
    if not rows:
        return []
    skip = {c for c in exclude if c}
    out = []
    for c in rows[0]:
        if c in skip:
            continue
        levels = {str(r.get(c, "")).strip() for r in rows if str(r.get(c, "")).strip()}
        if 2 <= len(levels) <= 10:
            out.append(c)
    return out


def _simpson_reversal(
    rows: Sequence[dict[str, Any]],
    group: str,
    value: str,
    a: str,
    b: str,
    positive: bool,
    candidates: Sequence[str],
) -> tuple[str, str] | None:
    """First (column, level) where the a-vs-b difference significantly reverses."""
    for col in candidates:
        levels = sorted(
            {str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()}
        )
        for level in levels:

            def vals(label: str, lv: str = level, cl: str = col) -> list[float]:
                out = []
                for r in rows:
                    if (
                        str(r.get(group, "")).strip() == label
                        and str(r.get(cl, "")).strip() == lv
                    ):
                        fv = _as_float(r.get(value))
                        if fv is not None:
                            out.append(fv)
                return out

            la, lb = vals(a), vals(b)
            if len(la) >= 10 and len(lb) >= 10:
                dl, tl = _welch(la, lb)
                if abs(tl) > _T and (dl > 0) != positive:
                    return (col, level)
    return None


def _frame(
    rows: Sequence[dict[str, Any]], cols: Sequence[str]
) -> tuple[dict[str, list[float]], int]:
    """Float columns over the rows where every requested column is present."""
    cols = list(dict.fromkeys(cols))
    kept: list[list[float]] = []
    for r in rows:
        try:
            kept.append([float(r[c]) for c in cols])
        except (ValueError, TypeError, KeyError):
            continue
    data = {c: [row[i] for row in kept] for i, c in enumerate(cols)}
    return data, len(kept)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(ai * bi for ai, bi in zip(a, b, strict=True))


def _inv(matrix: list[list[float]]) -> list[list[float]] | None:
    """Inverse of a small square matrix by Gauss-Jordan; None if singular."""
    n = len(matrix)
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [row[n:] for row in aug]


def _ols(
    data: dict[str, list[float]],
    y: str,
    x: str,
    controls: Sequence[str],
    mask: Sequence[bool] | None = None,
) -> tuple[float, float]:
    """OLS of ``y`` on intercept, ``x``, and controls; return (coef of x, its t)."""
    indices = [i for i in range(len(data[y])) if mask is None or mask[i]]
    yv = [data[y][i] for i in indices]
    design = [[1.0] * len(indices), [data[x][i] for i in indices]]
    design += [[data[c][i] for i in indices] for c in controls]
    p = len(design)
    n = len(yv)
    xtx = [[_dot(design[i], design[j]) for j in range(p)] for i in range(p)]
    inv = _inv(xtx)
    if inv is None:
        return 0.0, 0.0
    xty = [_dot(col, yv) for col in design]
    beta = [_dot(inv[i], xty) for i in range(p)]
    resid = [
        yv[k] - math.fsum(beta[i] * design[i][k] for i in range(p)) for k in range(n)
    ]
    sigma2 = _dot(resid, resid) / max(n - p, 1)
    var = sigma2 * inv[1][1]
    se = var**0.5 if var > 0 else math.inf
    return beta[1], (beta[1] / se if se > 0 else 0.0)


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = math.fsum(a) / n, math.fsum(b) / n
    cov = math.fsum((ai - ma) * (bi - mb) for ai, bi in zip(a, b, strict=True))
    va = math.fsum((ai - ma) ** 2 for ai in a)
    vb = math.fsum((bi - mb) ** 2 for bi in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0


def _residual_on(data: dict[str, list[float]], target: str, z: str) -> list[float]:
    """Residuals of ``target`` after regressing it on intercept and ``z``."""
    design = [[1.0] * len(data[z]), data[z]]
    inv = _inv([[_dot(design[i], design[j]) for j in range(2)] for i in range(2)])
    if inv is None:
        return list(data[target])
    xty = [_dot(col, data[target]) for col in design]
    b0, b1 = (_dot(inv[i], xty) for i in range(2))
    return [v - b0 - b1 * zi for v, zi in zip(data[target], data[z], strict=True)]


def _is_unshielded_collider(
    data: dict[str, list[float]], x: str, y: str, z: str
) -> bool:
    """Whether ``z`` is an identifiable collider on ``x`` and ``y`` (``x->z<-y``).

    This is the one causal-role fact observational data can settle: a collider is an
    *unshielded* v-structure: ``x`` and ``y`` are marginally independent but become
    dependent once ``z`` is held fixed. When ``x`` and ``y`` are already marginally
    dependent (``z`` is shielded), a confounder, suppressor, mediator, and shielded
    collider are indistinguishable, so no such claim can be made from the data alone.
    """
    n = len(data[x])
    marg = _corr(data[x], data[y])
    cond = _corr(_residual_on(data, x, z), _residual_on(data, y, z))
    return abs(_corr_t(marg, n)) <= _T and abs(_corr_t(cond, n)) > _T


def _role(data: dict[str, list[float]], x: str, y: str, z: str) -> str:
    """The identifiable role of ``z`` w.r.t. the ``x``-``y`` link, from data alone.

    ``collider`` only when the unshielded v-structure is identifiable (see
    :func:`_is_unshielded_collider`); ``confounder`` when ``x`` and ``y`` are already
    dependent and conditioning on ``z`` shrinks that link (adjusting is defensible);
    ``neutral`` otherwise: including every shielded case whose role the data cannot pin
    down, which callers must therefore not treat as an established collider.
    """
    if _is_unshielded_collider(data, x, y, z):
        return "collider"
    marg = abs(_corr(data[x], data[y]))
    cond = abs(_corr(_residual_on(data, x, z), _residual_on(data, y, z)))
    if cond < marg - _SIGNAL:
        return "confounder"
    return "neutral"


def _iqr_mask(data: dict[str, list[float]], cols: Sequence[str]) -> list[bool]:
    """A row mask dropping points outside 1.5·IQR on any of ``cols``."""
    keep = [True] * len(data[cols[0]])
    for c in cols:
        ordered = sorted(data[c])
        q1, q3 = _quantile(ordered, 0.25), _quantile(ordered, 0.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        keep = [k and lo <= v <= hi for k, v in zip(keep, data[c], strict=True)]
    return keep


def _quantile(ordered: Sequence[float], q: float) -> float:
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _as_float(value: Any) -> float | None:
    """Parse a cell to float, or None when it is not numeric."""
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return None


def _mean_var(xs: Sequence[float]) -> tuple[float, float]:
    """Sample mean and (n-1) variance."""
    n = len(xs)
    m = math.fsum(xs) / n
    return m, math.fsum((x - m) ** 2 for x in xs) / max(n - 1, 1)


def _welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mean difference ``a-b`` and its Welch t-statistic (unequal variances)."""
    ma, va = _mean_var(a)
    mb, vb = _mean_var(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    diff = ma - mb
    return diff, (diff / se if se > 0 else 0.0)


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean difference (pooled), a scale-free effect size."""
    ma, va = _mean_var(a)
    mb, vb = _mean_var(b)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return (ma - mb) / pooled if pooled > 0 else 0.0


def _iqr_keep(xs: Sequence[float]) -> list[bool]:
    """A mask dropping points outside 1.5·IQR of ``xs``."""
    ordered = sorted(xs)
    q1, q3 = _quantile(ordered, 0.25), _quantile(ordered, 0.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return [lo <= x <= hi for x in xs]


def _corr_t(r: float, n: int) -> float:
    """The t-statistic for a correlation ``r`` over ``n`` points."""
    denom = 1.0 - r * r
    if denom <= 0 or n <= 2:
        return math.inf if r != 0 else 0.0
    return r * math.sqrt((n - 2) / denom)


def _autocorr1(xs: Sequence[float]) -> float:
    """Lag-1 autocorrelation; high values mean the points are not independent."""
    n = len(xs)
    if n < 3:
        return 0.0
    m = math.fsum(xs) / n
    num = math.fsum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, n))
    den = math.fsum((x - m) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _fragility(
    data: dict[str, list[float]],
    x: str,
    y: str,
    controls: Sequence[str],
    *,
    base_positive: bool,
    reps: int = 80,
) -> float:
    """Fraction of half-size subsets where the signed effect does not hold."""
    n = len(data[y])
    k = n // 2
    # subset by a content order, not the incoming row order, so the rate is invariant to
    # how the rows happen to be arranged: a deterministic verdict must not depend on
    # input order (ties are between identical rows, so their arrangement cannot matter)
    order = sorted(range(n), key=lambda i: tuple(data[c][i] for c in (y, x, *controls)))
    broke = 0
    for rep in range(reps):
        chosen = {order[p] for p in _subset(n, k, rep)}
        mask = [i in chosen for i in range(n)]
        b, t = _ols(data, y, x, controls, mask=mask)
        if not (abs(t) > _T and (b > 0) == base_positive):
            broke += 1
    return broke / reps


def _subset(n: int, k: int, rep: int) -> set[int]:
    """A ``k``-of-``n`` index subset for resample ``rep``, reproducible without an RNG.

    Ordering by a multiplicative mix of index and ``rep`` gives a different
    pseudo-random half each resample, with no dependence on global RNG state or
    ``PYTHONHASHSEED`` (so the verdict is stable across processes).
    """

    def mix(i: int) -> int:
        return ((i + 1) * (2 * rep + 1) * 2654435761) % 2147483647

    return set(sorted(range(n), key=mix)[:k])


def _p_two_sided(z: float) -> float:
    """Two-sided normal-tail p-value for a z/t statistic (stdlib ``erfc``)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _benjamini_hochberg(pvalues: Sequence[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg discoveries: a mask of which p-values survive FDR ``alpha``.

    Controls the expected false-discovery rate across a family of tests. The ordered
    p-values p(1)..p(m) are compared to the line ``alpha*k/m``; the largest rank whose
    p-value falls below its threshold sets the cutoff, and every test at or below it is
    a discovery. This is the standard correction when many metrics are screened at once.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    cutoff_rank = 0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= alpha * rank / m:
            cutoff_rank = rank
    survive = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank <= cutoff_rank:
            survive[i] = True
    return survive


def _cbrt(v: float) -> float:
    """Real cube root, preserving sign (``v ** (1/3)`` is complex for v < 0)."""
    return math.copysign(abs(v) ** (1.0 / 3.0), v)


def _normal_p(xs: Sequence[float]) -> float | None:
    """D'Agostino-Pearson K² p-value for normality; None when n is too small.

    Combines sample skewness and kurtosis into a statistic that is χ²(2) under
    normality, whose tail is the closed form ``exp(-K²/2)``. Below n = 20 the test
    is unreliable, so normality is left undecided (the caller treats it as unknown).
    """
    n = len(xs)
    if n < 20:
        return None
    m = math.fsum(xs) / n
    m2 = math.fsum((v - m) ** 2 for v in xs) / n
    if m2 <= 0:
        return None
    g1 = (math.fsum((v - m) ** 3 for v in xs) / n) / m2**1.5
    g2 = (math.fsum((v - m) ** 4 for v in xs) / n) / m2**2 - 3.0
    # skewness -> Z1
    mu2 = 6.0 * (n - 2) / ((n + 1) * (n + 3))
    gam2 = (
        36.0 * (n - 7) * (n * n + 2 * n - 5) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    )
    w2 = math.sqrt(2 * gam2 + 4) - 1
    delta = 1.0 / math.sqrt(0.5 * math.log(w2))
    alpha2 = 2.0 / (w2 - 1)
    z1 = delta * math.asinh(g1 / math.sqrt(alpha2 * mu2))
    # kurtosis -> Z2 (Anscombe-Glynn)
    mu1 = -6.0 / (n + 1)
    var2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    gam1 = (6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9))) * math.sqrt(
        6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3))
    )
    a = 6.0 + (8.0 / gam1) * (2.0 / gam1 + math.sqrt(1 + 4.0 / gam1**2))
    xx = (g2 - mu1) / math.sqrt(var2)
    z2 = math.sqrt(9 * a / 2) * (
        1 - 2.0 / (9 * a) - _cbrt((1 - 2.0 / a) / (1 + xx * math.sqrt(2.0 / (a - 4))))
    )
    return math.exp(-(z1 * z1 + z2 * z2) / 2.0)


def _mann_whitney(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mann-Whitney U and its tie-corrected z (rank test, no normality assumed)."""
    n1, n2 = len(a), len(b)
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    ties: list[int] = []
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for kk in range(i, j + 1):
            ranks[kk] = avg
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    r1 = math.fsum(ranks[idx] for idx in range(len(pooled)) if pooled[idx][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    big_n = n1 + n2
    tie = math.fsum(t**3 - t for t in ties)
    var = (n1 * n2 / 12.0) * ((big_n + 1) - tie / (big_n * (big_n - 1)))
    if var <= 0:
        return u, 0.0
    return u, (u - n1 * n2 / 2.0 + 0.5) / math.sqrt(var)


def _chi2_sf(x: float, k: int) -> float:
    """Upper-tail probability of a chi-square with ``k`` degrees of freedom."""
    if x <= 0:
        return 1.0
    return _gamma_q(k / 2.0, x / 2.0)


def _gamma_q(s: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(s, x) (series / continued fraction)."""
    if x < s + 1.0:
        # lower series P(s, x), then complement
        ap = s
        total = 1.0 / s
        delta = total
        for _ in range(300):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    # Lentz continued fraction for Q(s, x)
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (Lentz continued fraction).

    The foundation for exact Student-t and F tail probabilities, replacing the
    normal approximation where degrees of freedom are small.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (helper for :func:`_betai`)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < 1e-15:
            break
    return h


def _t_sf(t: float, df: float) -> float:
    """Two-sided Student-t tail probability for ``t`` with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _f_sf(f: float, d1: float, d2: float) -> float:
    """Upper-tail probability P(F >= f) for an F distribution with ``(d1, d2)`` df."""
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def _inv_normal(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    )
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        return -num / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _ols_full(
    data: dict[str, list[float]],
    y: str,
    cols: Sequence[str],
    mask: Sequence[bool] | None = None,
) -> dict[str, Any] | None:
    """Fit OLS of ``y`` on intercept and ``cols``; return betas, residuals, more.

    The returned mapping carries everything the diagnostics need: ``beta`` (intercept
    first), ``resid``, ``fitted``, ``yv``, the design rows, ``inv`` = (XᵀX)⁻¹, and
    the counts ``n`` and ``p``. None if the design is singular.
    """
    idx = [i for i in range(len(data[y])) if mask is None or mask[i]]
    yv = [data[y][i] for i in idx]
    design = [[1.0] * len(idx)] + [[data[c][i] for i in idx] for c in cols]
    p, n = len(design), len(yv)
    inv = _inv([[_dot(design[i], design[j]) for j in range(p)] for i in range(p)])
    if inv is None:
        return None
    xty = [_dot(col, yv) for col in design]
    beta = [_dot(inv[i], xty) for i in range(p)]
    fitted = [math.fsum(beta[i] * design[i][k] for i in range(p)) for k in range(n)]
    resid = [yv[k] - fitted[k] for k in range(n)]
    return {
        "beta": beta,
        "resid": resid,
        "fitted": fitted,
        "yv": yv,
        "design": design,
        "inv": inv,
        "n": n,
        "p": p,
    }


def _coef_t(fit: dict[str, Any], j: int) -> tuple[float, float]:
    """Coefficient ``j`` and its t-statistic from a fitted model."""
    n, p, inv = fit["n"], fit["p"], fit["inv"]
    s2 = _dot(fit["resid"], fit["resid"]) / max(n - p, 1)
    var = s2 * inv[j][j]
    se = var**0.5 if var > 0 else math.inf
    return fit["beta"][j], (fit["beta"][j] / se if se > 0 else 0.0)


def _vif(data: dict[str, list[float]], target: str, others: Sequence[str]) -> float:
    """Variance inflation factor of ``target`` given the other predictors."""
    if not others:
        return 1.0
    fit = _ols_full(data, target, others)
    if fit is None:
        return math.inf
    rss = _dot(fit["resid"], fit["resid"])
    m = math.fsum(fit["yv"]) / len(fit["yv"])
    tss = math.fsum((v - m) ** 2 for v in fit["yv"])
    r2 = 1.0 - rss / tss if tss > 0 else 1.0
    return 1.0 / (1.0 - r2) if r2 < 1.0 else math.inf


def _breusch_pagan(fit: dict[str, Any], k: int) -> float:
    """Breusch-Pagan (Koenker) p-value: regress squared residuals on predictors."""
    e2 = [r * r for r in fit["resid"]]
    design = fit["design"]
    aux: dict[str, list[float]] = {"_e2": e2}
    cols = []
    for j in range(1, len(design)):
        aux[f"_g{j}"] = design[j]
        cols.append(f"_g{j}")
    aux_fit = _ols_full(aux, "_e2", cols)
    if aux_fit is None:
        return 1.0
    rss = _dot(aux_fit["resid"], aux_fit["resid"])
    m = math.fsum(e2) / len(e2)
    tss = math.fsum((v - m) ** 2 for v in e2)
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return _chi2_sf(fit["n"] * r2, k)


def _robust_t(fit: dict[str, Any], j: int) -> float:
    """Heteroskedasticity-robust (HC0 sandwich) t-statistic for coefficient ``j``."""
    design, inv, n, p = fit["design"], fit["inv"], fit["n"], fit["p"]
    e2 = [r * r for r in fit["resid"]]
    meat = [
        [
            math.fsum(design[a][k] * design[b][k] * e2[k] for k in range(n))
            for b in range(p)
        ]
        for a in range(p)
    ]
    im = [
        [math.fsum(inv[a][c] * meat[c][b] for c in range(p)) for b in range(p)]
        for a in range(p)
    ]
    cov = [
        [math.fsum(im[a][c] * inv[c][b] for c in range(p)) for b in range(p)]
        for a in range(p)
    ]
    se = cov[j][j] ** 0.5 if cov[j][j] > 0 else math.inf
    return fit["beta"][j] / se if se > 0 else 0.0


def _low_influence(fit: dict[str, Any]) -> list[bool]:
    """Mask dropping rows whose Cook's distance exceeds the 4/n rule of thumb."""
    design, inv, n, p = fit["design"], fit["inv"], fit["n"], fit["p"]
    s2 = _dot(fit["resid"], fit["resid"]) / max(n - p, 1)
    thresh = 4.0 / n
    keep = []
    for k in range(n):
        xk = [design[a][k] for a in range(p)]
        h = math.fsum(
            xk[a] * math.fsum(inv[a][b] * xk[b] for b in range(p)) for a in range(p)
        )
        denom = (1.0 - h) ** 2
        cook = (
            (fit["resid"][k] ** 2 / (p * s2)) * (h / denom)
            if s2 > 0 and denom > 0
            else 0.0
        )
        keep.append(cook <= thresh)
    return keep


def _reset_t(
    data: dict[str, list[float]], y: str, cols: Sequence[str], fit: dict[str, Any]
) -> float:
    """Ramsey RESET: t-statistic of the squared fitted values added as a regressor."""
    aug = dict(data)
    aug["_fit2"] = [f * f for f in fit["fitted"]]
    aug_fit = _ols_full(aug, y, [*cols, "_fit2"])
    if aug_fit is None:
        return 0.0
    return _coef_t(aug_fit, len(cols) + 1)[1]
