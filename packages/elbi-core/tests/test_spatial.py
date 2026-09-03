"""The spatial gate certifies a real geographic pattern, not a random one.

For a location question the honest test is spatial autocorrelation (Moran's I), not an
ordinary regression on latitude and longitude. These tests prove the gate certifies a
surface whose value clusters in space, stays inconclusive when the value is spatially
random, and that a latitude/longitude/value claim routes to it through ``verify_all``.
"""

from __future__ import annotations

import random

from elbi_core.verification import verify_all
from elbi_core.verification.spatial import verify_spatial

N = 400  # a 20x20 grid of located cells


def _grid(value_of) -> list[dict[str, str]]:
    rng = random.Random(0)
    rows = []
    for i in range(N):
        r, c = i // 20, i % 20
        rows.append(
            {
                "lat": str(47.2 + 0.01 * r),
                "long": str(-122.4 + 0.01 * c),
                "price": str(value_of(r, c, rng)),
            }
        )
    return rows


def test_spatial_certifies_a_clustered_surface() -> None:
    # price rises smoothly to the north: strong positive spatial autocorrelation
    rows = _grid(lambda r, c, rng: 200_000 + 50_000 * r + rng.gauss(0, 20_000))
    report = verify_spatial(rows, "lat", "long", "price")
    assert report.verdict == "sound"
    assert report.effect > 0.3  # Moran's I well above zero
    assert any(ch.name == "hot_spots" for ch in report.checks)


def test_spatial_is_inconclusive_for_a_random_surface() -> None:
    # price independent of location: no spatial structure
    rows = _grid(lambda r, c, rng: rng.gauss(500_000, 100_000))
    report = verify_spatial(rows, "lat", "long", "price")
    assert report.verdict == "inconclusive"


def test_reverting_the_permutation_null_would_pass_random_as_real() -> None:
    # Guard that the permutation test is what rejects a random surface: its I is small,
    # so a gate that skipped significance and certified any positive I would be wrong.
    rows = _grid(lambda r, c, rng: rng.gauss(500_000, 100_000))
    report = verify_spatial(rows, "lat", "long", "price")
    assert abs(report.effect) < 0.15  # near zero: indistinguishable from noise


def test_geographic_claim_routes_to_the_spatial_gate() -> None:
    rows = _grid(lambda r, c, rng: 200_000 + 50_000 * r + rng.gauss(0, 20_000))
    report = verify_all(rows, latitude="lat", longitude="long", value="price")
    assert report.verdict == "sound"
    assert any(g.name == "spatial" for g in report.ran)
    # the surface is dependent, so the iid-bootstrap stability pass does not apply
    assert not any(g.name == "stability" for g in report.ran)
