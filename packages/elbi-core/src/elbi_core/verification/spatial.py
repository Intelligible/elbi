"""Spatial-pattern gate: is a value's geographic distribution real, not random?

For a question about location, the honest test is not an ordinary regression on latitude
and longitude: spatial dependence biases those coefficients (Anselin, spatial
econometrics). It is whether the value is *spatially autocorrelated*: does it cluster in
space more than chance? The certifying statistic is global Moran's I with a permutation
null (Moran 1950); Getis-Ord Gi* (Getis & Ord 1992) then locates the significant hot and
cold spots: exactly what a price heatmap shows. So a certified spatial finding and the
map are the same object: the map is a view of the verified surface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _frame
from ._report import Check, VerificationReport

_MIN_POINTS = 30
_MAX_POINTS = 2500  # cap the O(n^2) neighbour search; subsample deterministically above
_K = 8  # nearest neighbours defining the spatial weights
_PERMUTATIONS = 199  # permutation null for Moran's I (p resolution ~1/200)
_ALPHA = 0.05
_GI_Z = 1.96  # |z| for a significant Getis-Ord Gi* hot/cold spot
#: Seeded-LCG constants for the deterministic permutation null (no global RNG).
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_MASK64 = (1 << 64) - 1


def _subsample(n: int) -> list[int]:
    """Deterministic indices to keep, so the O(n^2) search stays bounded on large n."""
    if n <= _MAX_POINTS:
        return list(range(n))
    step = n / _MAX_POINTS
    return [int(i * step) for i in range(_MAX_POINTS)]


def _knn(lats: list[float], lons: list[float], k: int) -> list[list[int]]:
    """Each point's k nearest neighbours by planar distance on the coordinates.

    Planar (not great-circle) distance is fine within a study area; the weights only
    need to rank who is near, and a k-nearest-neighbour graph is a standard, connected
    spatial weights structure.
    """
    n = len(lats)
    kk = min(k, n - 1)
    out: list[list[int]] = []
    for i in range(n):
        dists = sorted(
            ((lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2, j)
            for j in range(n)
            if j != i
        )
        out.append([j for _, j in dists[:kk]])
    return out


def _moran(z: list[float], neighbours: list[list[int]], w: float) -> float:
    """Global Moran's I of centred values ``z`` under row-standardised knn weights."""
    denom = sum(zi * zi for zi in z)
    if denom == 0:
        return 0.0
    num = sum(z[i] * w * sum(z[j] for j in neighbours[i]) for i in range(len(z)))
    return num / denom


def _shuffled(values: list[float], seed: int) -> list[float]:
    """A deterministic Fisher-Yates shuffle (seeded LCG) for the permutation null."""
    out = list(values)
    state = (seed * 2 + 1) & _MASK64
    for i in range(len(out) - 1, 0, -1):
        state = (state * _LCG_A + _LCG_C) & _MASK64
        j = (state >> 33) % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def _hot_spots(vals: list[float], neighbours: list[list[int]]) -> int:
    """Count significant Getis-Ord Gi* hot or cold spots (|z| > 1.96)."""
    n = len(vals)
    mean = sum(vals) / n
    s = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    if s == 0:
        return 0
    count = 0
    for i in range(n):
        window = (i, *neighbours[i])
        wc = len(window)
        spread = (n * wc - wc * wc) / (n - 1)
        denom = s * math.sqrt(spread) if spread > 0 else 0.0
        if denom == 0:
            continue
        z = (sum(vals[j] for j in window) - mean * wc) / denom
        if abs(z) > _GI_Z:
            count += 1
    return count


def verify_spatial(
    rows: Sequence[dict[str, Any]], lat: str, long: str, value: str
) -> VerificationReport:
    """Verify a real geographic pattern in ``value`` across ``lat``/``long``.

    Certifies on a significant *positive* global Moran's I under a permutation null:
    ``value`` clusters in space more than random, the unbiased basis for a location
    claim and its map. Reports the Getis-Ord hot/cold-spot count the map displays.
    Inconclusive when the surface is spatially random.
    """
    data, n0 = _frame(rows, [lat, long, value])
    if n0 < _MIN_POINTS or not all(c in data for c in (lat, long, value)):
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"need at least {_MIN_POINTS} located points with a value",),
        )
    keep = _subsample(n0)
    lats = [data[lat][i] for i in keep]
    lons = [data[long][i] for i in keep]
    vals = [data[value][i] for i in keep]
    n = len(vals)
    neighbours = _knn(lats, lons, _K)
    w = 1.0 / min(_K, n - 1)
    mean = sum(vals) / n
    z = [v - mean for v in vals]
    observed = _moran(z, neighbours, w)
    ge = sum(
        1
        for s in range(_PERMUTATIONS)
        if _moran(_shuffled(z, s + 1), neighbours, w) >= observed
    )
    p = (ge + 1) / (_PERMUTATIONS + 1)
    if observed <= 0 or p > _ALPHA:
        detail = f"no significant spatial clustering (I={observed:.3f}, p={p:.3f})"
        return VerificationReport("inconclusive", observed, False, (), None, (detail,))
    checks = (
        Check(
            "morans_i",
            True,
            f"significant spatial autocorrelation (I={observed:.3f}, p={p:.3f})",
        ),
        Check(
            "hot_spots",
            True,
            f"{_hot_spots(vals, neighbours)} significant Getis-Ord hot/cold cells",
        ),
    )
    caveats = (
        "location proxies neighbourhood factors (schools, access, amenities), not a "
        "cause in itself",
    )
    return VerificationReport("sound", observed, True, checks, None, caveats)
