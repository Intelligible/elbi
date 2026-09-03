"""Clustering / segmentation validity gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _numeric_columns
from ._report import Check, VerificationReport

#: Hopkins statistic below which the data shows no more clustering tendency than
#: uniform noise (0.5 = no tendency; toward 1 = clustered).
_HOPKINS = 0.6
#: Mean silhouette below which a partition reflects no substantial structure.
_SILHOUETTE = 0.25


def verify_clusters(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    k: int = 3,
) -> VerificationReport:
    """Verify that a claimed cluster structure is real, not imposed.

    k-means returns k clusters even on uniform or single-blob data, so "we found k
    segments" can be false. This checks cluster *tendency* (the Hopkins statistic
    against a uniform reference) and the silhouette of a k-means partition. The verdict
    is ``sound`` (real, separated clusters), ``unsound`` (no more structure than noise:
    the segments are an artifact of forcing a partition), or ``inconclusive`` (too
    little data).
    """
    feats = [f for f in _numeric_columns(rows) if f in features]
    pts = []
    for r in rows:
        try:
            pts.append([float(r[f]) for f in feats])
        except (KeyError, ValueError, TypeError):
            continue
    if len(pts) < 50 or len(feats) < 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few rows or features to assess clusters",),
        )
    pts = _standardize(pts)
    checks: list[Check] = []

    hop = _hopkins(pts)
    tendency = hop >= _HOPKINS
    checks.append(
        Check(
            "tendency",
            tendency,
            f"Hopkins statistic = {hop:.2f}"
            + ("" if tendency else " (no more clustered than uniform noise)"),
        )
    )

    labels = _kmeans(pts, max(2, k))
    sil = _silhouette(pts, labels)
    separated = sil >= _SILHOUETTE
    checks.append(
        Check(
            "separation",
            separated,
            f"mean silhouette = {sil:.2f}"
            + ("" if separated else " (clusters are not separated)"),
        )
    )

    if all(c.survived for c in checks):
        verdict, pivotal = "sound", None
    else:
        verdict, pivotal = (
            "unsound",
            (
                "the data has no real cluster structure (Hopkins "
                f"{hop:.2f}, silhouette {sil:.2f}); any k segments are imposed "
                "by k-means, not found in the data"
            ),
        )
    caveats = (
        "cluster labels describe structure, not cause; profile them before acting",
    )
    return VerificationReport(verdict, sil, separated, tuple(checks), pivotal, caveats)


def _standardize(pts: list[list[float]]) -> list[list[float]]:
    """Z-score each feature so distances are not dominated by raw scale."""
    n, p = len(pts), len(pts[0])
    out = [row[:] for row in pts]
    for j in range(p):
        col = [row[j] for row in pts]
        m = math.fsum(col) / n
        sd = math.sqrt(math.fsum((v - m) ** 2 for v in col) / n) or 1.0
        for i in range(n):
            out[i][j] = (pts[i][j] - m) / sd
    return out


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True))


def _hopkins(pts: list[list[float]], sample: int = 50) -> float:
    """Hopkins cluster-tendency statistic versus a uniform reference."""
    n, p = len(pts), len(pts[0])
    m = min(sample, n // 2)
    lo = [min(row[j] for row in pts) for j in range(p)]
    hi = [max(row[j] for row in pts) for j in range(p)]
    # deterministic pseudo-random picks, no global RNG
    idx = sorted(range(n), key=lambda i: ((i + 1) * 2654435761) % 2147483647)[:m]
    w = 0.0
    for i in idx:
        w += min(math.sqrt(_dist2(pts[i], pts[j])) for j in range(n) if j != i)
    u = 0.0
    for s in range(m):
        synth = [
            lo[j] + (hi[j] - lo[j]) * (((s + 1) * (j + 7) * 40503) % 997) / 997.0
            for j in range(p)
        ]
        u += min(math.sqrt(_dist2(synth, pts[j])) for j in range(n))
    return u / (u + w) if (u + w) > 0 else 0.5


def _kmeans(pts: list[list[float]], k: int, iters: int = 25) -> list[int]:
    """Lloyd's k-means with a deterministic spread initialization."""
    n = len(pts)
    step = max(1, n // k)
    centers = [pts[(i * step) % n][:] for i in range(k)]
    labels = [0] * n
    for _ in range(iters):
        changed = False
        for i in range(n):
            best = min(range(k), key=lambda c: _dist2(pts[i], centers[c]))
            if best != labels[i]:
                labels[i] = best
                changed = True
        for c in range(k):
            members = [pts[i] for i in range(n) if labels[i] == c]
            if members:
                centers[c] = [
                    math.fsum(row[j] for row in members) / len(members)
                    for j in range(len(pts[0]))
                ]
        if not changed:
            break
    return labels


def _silhouette(
    pts: list[list[float]], labels: Sequence[int], sample: int = 300
) -> float:
    """Mean silhouette over a deterministic subsample of points."""
    n = len(pts)
    idx = sorted(range(n), key=lambda i: ((i + 1) * 2654435761) % 2147483647)[
        : min(sample, n)
    ]
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(labels[i], []).append(i)
    if len(clusters) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in idx:
        own = clusters[labels[i]]
        if len(own) <= 1:
            continue
        a = math.fsum(math.sqrt(_dist2(pts[i], pts[j])) for j in own if j != i) / (
            len(own) - 1
        )
        b = min(
            math.fsum(math.sqrt(_dist2(pts[i], pts[j])) for j in members) / len(members)
            for lab, members in clusters.items()
            if lab != labels[i]
        )
        total += (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        count += 1
    return total / count if count else 0.0
