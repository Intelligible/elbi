"""The oracle gates a derivation's *conclusion* at certification (Level-2 soundness).

Golden cases verify that a derivation computes the right value; these tests verify the
other half: that a derivation which declares a ``claim`` is certified only when the
verification oracle finds that conclusion sound, so a derivation cannot be served on a
correct computation that supports a wrong inference.
"""

from __future__ import annotations

from elbi_core import (
    Registry,
    RequireSoundConclusion,
    Runner,
    SubprocessExecutor,
    author,
)

_CLEAN = """
def clean_effect(ctx):
    "Dose-response rows with a real effect."
    import random
    rng = random.Random(0)
    out = []
    for _ in range(300):
        x = rng.gauss(0, 1)
        y = 1.2 * x + rng.gauss(0, 0.8)
        out.append({"dose": round(x, 3), "response": round(y, 3)})
    return out
"""

# education and income both driven by parental_wealth: a real computation supporting a
# confounded (unsound) conclusion that education raises income
_CONFOUNDED = """
def confounded(ctx):
    "Education and income, both driven by parental wealth."
    import random
    rng = random.Random(0)
    out = []
    for _ in range(400):
        pw = rng.gauss(0, 1)
        out.append({
            "education": round(pw + rng.gauss(0, 0.5), 3),
            "income": round(1.5 * pw + rng.gauss(0, 0.5), 3),
            "parental_wealth": round(pw, 3),
        })
    return out
"""


def _runner(reg: Registry) -> Runner:
    return Runner(reg, executor=SubprocessExecutor(timeout=60))


def test_sound_conclusion_certifies(registry: Registry) -> None:
    out = author(
        "clean_effect",
        _CLEAN,
        runner=_runner(registry),
        claim={"x": "dose", "y": "response"},
        registry=registry,
    )
    assert out.result.oracle_verdict == "sound"
    assert out.certified and out.derivation.is_certified


def test_unsound_conclusion_blocks_certification(registry: Registry) -> None:
    # the derivation runs fine and computes a correct table, but its declared conclusion
    # is confounded; the oracle must block certification
    out = author(
        "confounded",
        _CONFOUNDED,
        runner=_runner(registry),
        claim={"x": "education", "y": "income"},
        registry=registry,
    )
    assert out.result.oracle_verdict == "unsound"
    assert not out.certified
    assert out.derivation.status == "proposed"


def test_no_claim_keeps_prior_behaviour(registry: Registry) -> None:
    out = author("clean_effect", _CLEAN, runner=_runner(registry), registry=registry)
    assert out.result.oracle_verdict is None
    assert out.certified  # a clean run with no declared claim certifies as before


def test_require_sound_conclusion_policy(registry: Registry) -> None:
    # this strict policy refuses a derivation that declares no verified conclusion
    no_claim = author(
        "clean_effect",
        _CLEAN,
        runner=_runner(registry),
        policy=RequireSoundConclusion(),
        registry=registry,
    )
    assert not no_claim.certified

    sound = author(
        "clean2",
        _CLEAN.replace("clean_effect", "clean2"),
        runner=_runner(registry),
        claim={"x": "dose", "y": "response"},
        policy=RequireSoundConclusion(),
        registry=registry,
    )
    assert sound.certified


def test_claim_is_emitted_in_the_manifest(registry: Registry) -> None:
    out = author(
        "clean_effect",
        _CLEAN,
        runner=_runner(registry),
        claim={"x": "dose", "y": "response"},
        registry=registry,
    )
    assert out.derivation.to_manifest()["claim"] == {"x": "dose", "y": "response"}
