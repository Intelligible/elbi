"""Seeded synthetic trap builders, referenced by name from JSON fixtures.

Each generator takes a ``seed`` and a row count ``n`` and returns rows of stringified
cells, the shape the oracle consumes. The constructions are lifted from the existing
gate test helpers (``_world`` in ``test_verification.py``, ``_ab_simpson`` in
``test_advanced_gates.py``, the leakage/rtm/collider builders), so they are proven to
carry the pitfall. Synthetic data is randomised per seed and not memorised by any
model, which is why it is the reliable "a bare LLM misses this" evidence.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .trap import Rows


def _f(value: float) -> str:
    """Stringify a float the way the oracle's parser expects."""
    return repr(round(value, 6))


def simpsons_ab(*, seed: int, n: int = 6000) -> Rows:
    """A/B test where B wins pooled but A wins inside every segment (Simpson's)."""
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        if rng.random() < 0.5:
            variant, segment = "A", ("hard" if rng.random() < 0.7 else "easy")
        else:
            variant, segment = "B", ("easy" if rng.random() < 0.7 else "hard")
        base = 0.5 if segment == "easy" else 0.1
        p = base + (0.10 if variant == "A" else 0.0)
        out.append(
            {
                "variant": variant,
                "segment": segment,
                "converted": str(int(rng.random() < p)),
            }
        )
    return out


def ice_cream_drownings(*, seed: int, n: int = 400) -> Rows:
    """Ice-cream sales and drownings correlate only because temperature drives both."""
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        temperature = rng.gauss(0, 1)
        out.append(
            {
                "ice_cream": _f(temperature + rng.gauss(0, 0.5)),
                "drownings": _f(1.5 * temperature + rng.gauss(0, 0.5)),
                "temperature": _f(temperature),
            }
        )
    return out


def leaky_churn(*, seed: int, n: int = 400) -> Rows:
    """A churn model whose `cancellation_logged` feature is a proxy for the label."""
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        churned = rng.randint(0, 1)
        out.append(
            {
                "churned": str(churned),
                "tenure_months": _f(rng.gauss(24, 8)),
                "monthly_charges": _f(rng.gauss(70, 20)),
                # near-identical to the label: available only after the outcome
                "cancellation_logged": _f(churned + rng.gauss(0, 0.01)),
            }
        )
    return out


def rtm_selected(*, seed: int, n: int = 400) -> Rows:
    """A cohort selected for a low baseline reverts to the mean, no treatment."""
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        latent = rng.gauss(0, 1)
        baseline = latent + rng.gauss(0, 0.7)
        followup = latent + rng.gauss(0, 0.7)
        out.append(
            {
                "baseline": _f(baseline),
                "followup": _f(followup),
                "cohort": "selected" if baseline < -0.43 else "control",
            }
        )
    return out


def berkson_admissions(*, seed: int, n: int = 400) -> Rows:
    """Talent and interview, independent until conditioned on admission (a collider)."""
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        talent = rng.gauss(0, 1)
        interview = rng.gauss(0, 1)
        out.append(
            {
                "talent": _f(talent),
                "interview": _f(interview),
                "admitted": _f(talent + interview + rng.gauss(0, 0.3)),
            }
        )
    return out


def joint_fragile(*, seed: int, n: int = 500) -> Rows:
    """An x-y link that survives each check alone but dies under the combined spec.

    Controlling the confounder 'w' alone or dropping outliers alone each leaves the
    effect standing; only both together (which the multiverse gate tries) erase it.
    """
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        w = rng.gauss(0, 1)
        out.append(
            {
                "x": _f(w + rng.gauss(0, 0.5)),
                "y": _f(w + rng.gauss(0, 0.5)),
                "w": _f(w),
            }
        )
    for i in rng.sample(range(n), max(1, n * 18 // 500)):
        out[i] = {
            "x": _f(6 + rng.gauss(0, 0.3)),
            "y": _f(6 + rng.gauss(0, 0.3)),
            "w": _f(rng.gauss(0, 1)),
        }
    return out


def suppressor(*, seed: int, n: int = 500) -> Rows:
    """An effect that only appears after adjusting for 'z' of ambiguous role.

    Confounder (adjust) or collider (leave)? The rows can't tell, so inconclusive.
    """
    rng = random.Random(seed)
    out: Rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        x = z + rng.gauss(0, 1)
        y = 0.6 * x - 1.5 * z + rng.gauss(0, 0.5)
        out.append({"x": _f(x), "y": _f(y), "z": _f(z)})
    return out


def noise(*, seed: int, n: int = 500) -> Rows:
    """Two independent variables: no real association to certify."""
    rng = random.Random(seed)
    return [{"x": _f(rng.gauss(0, 1)), "y": _f(rng.gauss(0, 1))} for _ in range(n)]


#: Named generators a manifest's ``generated_by.generator`` resolves against.
GENERATORS: dict[str, Callable[..., Rows]] = {
    "simpsons_ab": simpsons_ab,
    "ice_cream_drownings": ice_cream_drownings,
    "leaky_churn": leaky_churn,
    "rtm_selected": rtm_selected,
    "berkson_admissions": berkson_admissions,
    "joint_fragile": joint_fragile,
    "suppressor": suppressor,
    "noise": noise,
}
