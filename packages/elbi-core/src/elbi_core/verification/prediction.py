"""Predictive-performance (leakage-aware) gate."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _autocorr1,
    _corr,
    _dot,
    _inv,
    _numeric_columns,
)
from ._report import (
    _RANDOM_WALK,
    Check,
    VerificationReport,
)


def verify_prediction(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    target: str,
    *,
    time_order: str | None = None,
    group: str | None = None,
) -> VerificationReport:
    """Verify a predictive-performance claim by re-evaluating it leakage-free.

    Rather than trust a reported metric, this re-fits a simple model under an honest
    split and reports the held-out skill, after screening the silent failures that
    inflate predictive claims: a single feature that nearly perfectly predicts the
    target on its own (a leaked proxy or target derivative), rows duplicated across the
    split, and, when ``time_order`` or ``group`` is given, a split that lets future rows
    or the same entity appear on both sides. The held-out number it returns is
    leakage-free by construction.

    The verdict is ``unsound`` (a proxy or contamination inflates the claim),
    ``sound`` (honest held-out skill beats the baseline), or ``inconclusive`` (no
    signal beyond the baseline, or too little data to evaluate).
    """
    feats = [f for f in _numeric_columns(rows) if f in features and f != target]
    cols = [target, *feats]
    # Frame target + features as floats, and (aligned to the same kept rows) capture the
    # raw entity (group) and time-order values used only to split honestly. Group and
    # time are not coerced to float, so a string entity id or an ISO date still splits
    # correctly; they never enter the fitted model, only the train/test split.
    data: dict[str, list[float]] = {c: [] for c in cols}
    groups: list[Any] | None = [] if group else None
    times: list[Any] | None = [] if time_order else None
    for r in rows:
        try:
            vec = [float(r[c]) for c in cols]
        except (ValueError, TypeError, KeyError):
            continue
        gval = r.get(group) if group is not None else None
        tval = r.get(time_order) if time_order is not None else None
        if (groups is not None and gval is None) or (
            times is not None and tval is None
        ):
            continue
        for c, v in zip(cols, vec, strict=True):
            data[c].append(v)
        if groups is not None:
            groups.append(gval)
        if times is not None:
            times.append(tval)
    n = len(data[target])
    if n < 40 or not feats:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few rows or features",)
        )
    classify = len({round(v, 9) for v in data[target]}) == 2
    # time-series guard: a strongly autocorrelated target (in row order) means the
    # data is temporal, where a cross-sectional split leaks the future and inflates
    # skill. Refuse to certify rather than report a spurious score; the caller must
    # supply `time_order` (and forecasting belongs in a dedicated temporal check).
    if not time_order and not classify and _autocorr1(data[target]) > _RANDOM_WALK:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (
                "the target is strongly autocorrelated in row order: this looks like "
                "time-series data, where a cross-sectional split leaks the future; "
                "supply `time_order` for a temporal evaluation",
            ),
        )
    checks: list[Check] = []
    pivotal: str | None = None

    # a lone feature that nearly perfectly predicts the target is almost always a
    # leaked proxy (a derivative of the target, or recorded after the outcome).
    proxy = _target_proxy(data, feats, target, classify)
    held = proxy is None
    checks.append(
        Check(
            "target-leakage",
            held,
            "no single feature predicts the target almost perfectly"
            if held
            else f"'{proxy}' alone predicts the target almost perfectly (leak)",
        )
    )
    if not held:
        pivotal = f"'{proxy}' is a leaked proxy for the target; the skill is spurious"

    # split honestly: by time, by entity, or by a deterministic index partition
    train, test = _honest_split(n, times, groups)
    contam = _split_overlap(data, feats, train, test)
    held = contam <= 0.10
    checks.append(
        Check(
            "split-contamination",
            held,
            "train and test rows are disjoint"
            if held
            else f"{contam:.0%} of test rows are duplicated in train",
        )
    )
    if not held:
        pivotal = pivotal or (
            f"{contam:.0%} of test rows also appear in train; the skill is memorized"
        )

    skill, floor = _honest_skill(data, feats, target, train, test, classify)
    real = skill is not None and skill > floor
    if skill is None:
        detail = "could not evaluate held-out skill"
    elif real:
        detail = f"held-out skill {skill:.2f} clears the chance floor {floor:.2f}"
    else:
        detail = f"held-out skill {skill:.2f} is within chance of the baseline"
    checks.append(Check("held-out-skill", real, detail))

    if not all(c.survived for c in checks[:2]):
        verdict = "unsound"
    elif real:
        verdict = "sound"
    else:
        verdict = "inconclusive"
        pivotal = pivotal or "the model has no predictive skill beyond the baseline"
    caveats: tuple[str, ...] = (
        "skill is estimated on this data's distribution; it need not transfer to a "
        "different population or time",
    )
    # The leakage screen tests one feature at a time, so it cannot see a target that is
    # an exact function of several features together (a sum or difference). That shows
    # up as near-perfect held-out skill, so flag it: a caveat, not a veto, since genuine
    # near-perfect prediction is possible and the reader must judge feature provenance.
    if real and skill is not None and skill >= 0.99:
        caveats = (
            *caveats,
            "held-out skill is near-perfect; if the target is an exact function of "
            "several features together (a sum or difference), that is leakage the "
            "single-feature screen cannot see. Confirm every feature is known before "
            "the outcome, not derived from it.",
        )
    return VerificationReport(
        verdict, (skill or 0.0), real, tuple(checks), pivotal, caveats
    )


def _target_proxy(
    data: dict[str, list[float]],
    feats: Sequence[str],
    target: str,
    classify: bool,
) -> str | None:
    """Name a feature that alone predicts the target almost perfectly, else None."""
    yv = data[target]
    for f in feats:
        if classify:
            if _stump_pps(data[f], yv) >= 0.95:
                return f
        elif _corr(data[f], yv) ** 2 >= 0.95:
            return f
    return None


def _stump_pps(values: Sequence[float], labels: Sequence[float]) -> float:
    """Predictive power of a one-feature threshold for a binary label (0 if none).

    Sweeps the best split point and scores accuracy normalized against the majority
    baseline, so a feature that perfectly separates the classes scores near 1.
    """
    pairs = sorted(zip(values, labels, strict=True))
    n = len(pairs)
    ones = math.fsum(lbl for _, lbl in pairs)
    base = max(ones, n - ones) / n
    best = base
    left_ones = 0.0
    for i in range(1, n):
        left_ones += pairs[i - 1][1]
        if pairs[i][0] == pairs[i - 1][0]:
            continue
        # predict majority on each side of the split at i
        correct = max(left_ones, i - left_ones) + max(
            ones - left_ones, (n - i) - (ones - left_ones)
        )
        best = max(best, correct / n)
    return (best - base) / (1.0 - base) if base < 1.0 else 0.0


def _group_bucket(value: Any) -> int:
    """A deterministic 0-9 bucket for an entity id (stable across processes and types).

    ``hash()`` is salted per process, which would make the split non-reproducible; a
    fixed digest of the value's string form keeps whole-entity holdout deterministic
    whether the id is an int, float, or string.
    """
    digest = hashlib.blake2b(repr(value).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10


def _honest_split(
    n: int,
    times: Sequence[Any] | None,
    groups: Sequence[Any] | None,
) -> tuple[list[int], list[int]]:
    """Train/test row indices that leak nothing: by time, by entity, else by index.

    A time split trains on the earliest 70% and tests on the latest 30%, so no future
    row informs the past. An entity split assigns whole entities to one side (the same
    guarantee scikit-learn's ``GroupKFold``/``GroupShuffleSplit`` give), so the model is
    scored only on entities it never trained on. Both fall back to a deterministic index
    partition when they cannot form two non-empty sides.
    """
    if times is not None:
        order = sorted(range(n), key=lambda i: times[i])
        cut = int(n * 0.7)
        if 0 < cut < n:
            return order[:cut], order[cut:]
    if groups is not None:
        # hold out whole entities (~30%, keyed by a stable hash of the id) so the same
        # entity never spans the split and cross-entity generalization is what is scored
        train = [i for i in range(n) if _group_bucket(groups[i]) >= 3]
        test = [i for i in range(n) if _group_bucket(groups[i]) < 3]
        if train and test:
            return train, test
    test = [i for i in range(n) if ((i + 1) * 2654435761) % 10 < 3]
    train = [i for i in range(n) if i not in set(test)]
    return train, test


def _split_overlap(
    data: dict[str, list[float]],
    feats: Sequence[str],
    train: Sequence[int],
    test: Sequence[int],
) -> float:
    """Fraction of test rows whose feature tuple also appears in train."""
    if not test:
        return 0.0

    def key(i: int) -> tuple[float, ...]:
        return tuple(round(data[f][i], 8) for f in feats)

    seen = {key(i) for i in train}
    return math.fsum(1.0 for i in test if key(i) in seen) / len(test)


#: Ridge penalty for the held-out fit, on standardized features. Small, so a
#: well-conditioned problem is essentially unregularized OLS, but strictly positive so a
#: rank-deficient design (collinear features, e.g. a part and its sum) stays invertible
#: and the fit predicts honestly instead of exploding out of sample.
_RIDGE = 1.0


def _honest_skill(
    data: dict[str, list[float]],
    feats: Sequence[str],
    target: str,
    train: Sequence[int],
    test: Sequence[int],
    classify: bool,
) -> tuple[float | None, float]:
    """Held-out skill of a simple model and the chance floor it must clear.

    Fits a ridge-stabilized linear model on the train rows and scores it on the held-out
    test rows: accuracy for a binary target, out-of-sample R² for a continuous one. The
    ridge term (on standardized features) keeps the fit honest when features are
    collinear: a bare OLS on a rank-deficient design produces exploding coefficients and
    a spuriously negative held-out R², reporting "no skill" on data that has it. The
    floor is the level chance alone reaches on this much held-out data, for a binary
    target the majority-class rate plus 3.09 standard errors (about p<0.001), for a
    continuous one a floor that grows with the number of features, so noise does not
    clear it. Returns (skill, floor); skill is None if the fit could not be formed.
    """
    if len(train) < 10 or len(test) < 10:
        return None, 0.0
    k, nt = len(feats), len(test)
    # standardize features on train statistics: ridge needs comparable scales, and a
    # zero-variance feature is dropped to unit scale so it contributes nothing.
    center = {f: math.fsum(data[f][i] for i in train) / len(train) for f in feats}
    scale = {}
    for f in feats:
        var = math.fsum((data[f][i] - center[f]) ** 2 for i in train) / len(train)
        scale[f] = math.sqrt(var) or 1.0

    def z(i: int) -> list[float]:
        return [(data[f][i] - center[f]) / scale[f] for f in feats]

    ymean = math.fsum(data[target][i] for i in train) / len(train)
    ztz = [[0.0] * k for _ in range(k)]
    zty = [0.0] * k
    for i in train:
        zi = z(i)
        yc = data[target][i] - ymean
        for a in range(k):
            zty[a] += zi[a] * yc
            for b in range(a, k):
                ztz[a][b] += zi[a] * zi[b]
    for a in range(k):
        for b in range(a):
            ztz[a][b] = ztz[b][a]
        ztz[a][a] += _RIDGE
    inv = _inv(ztz)
    if inv is None:
        return None, 0.0
    beta = [_dot(inv[a], zty) for a in range(k)]

    def predict(i: int) -> float:
        zi = z(i)
        return ymean + math.fsum(beta[a] * zi[a] for a in range(k))

    if classify:
        ones = math.fsum(data[target][i] for i in test)
        baseline = max(ones, nt - ones) / nt
        correct = math.fsum(
            1.0 for i in test if (1.0 if predict(i) >= 0.5 else 0.0) == data[target][i]
        )
        floor = baseline + 3.09 * math.sqrt(baseline * (1.0 - baseline) / nt)
        return correct / nt, floor
    sse = math.fsum((data[target][i] - predict(i)) ** 2 for i in test)
    sst = math.fsum((data[target][i] - ymean) ** 2 for i in test)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return r2, max(0.06, 4.0 * k / nt)
