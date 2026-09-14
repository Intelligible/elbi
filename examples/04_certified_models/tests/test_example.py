"""Runs the certified-models example end to end (part of the repo CI suite).

The first two tests are sanity checks on the already-vetted, statically served
model (mirrors examples/01_getting_started's own test shape). The rest are the
actual point of this example, mirroring
packages/elbi-core/tests/test_authoring_oracle.py's pattern: the identical
computation, proposed with two different claims, is certified in one case and
rejected in the other -- proving the oracle catches a leaked feature rather
than merely asserting that it would.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from elbi_cli.mcp_server import build_server
from elbi_core import Registry, Runner, SubprocessExecutor, author, serve
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

PROJECT = Path(__file__).resolve().parent.parent


def _runner() -> Runner:
    registry = Registry()
    discover(PROJECT / "derivations", registry=registry)
    bindings = DataBindings.load(PROJECT / "elbi.dev.yaml")
    return Runner(registry, bindings=bindings, base_dir=PROJECT)


def test_default_risk_scores_every_loan() -> None:
    artifact = _runner().run("default_risk")
    assert len(artifact.value) == 300
    assert {"application_id", "default_probability"} <= artifact.value[0].keys()


def test_default_model_is_opaque_and_internal() -> None:
    artifact = _runner().run("default_model")
    assert artifact.kind == "opaque"
    assert "weight" in artifact.value
    assert "days_past_due" not in artifact.value["features"]


# --- The point: the oracle, not this test, decides which model is trustworthy ---

# Same computation both times; only the claim differs, so a difference in verdict
# is about the claim's validity, not the code. Matches
# fixtures/generate_loans.py's own formula, tuned so a held-out fit clearly clears
# chance on the legitimate features alone.
_PROPOSAL = """
def {name}(ctx):
    "Score default risk from application-time features."
    import random
    rng = random.Random(0)
    out = []
    for _ in range(300):
        income = rng.uniform(30000, 150000)
        debt_ratio = rng.uniform(0.0, 0.6)
        credit_score = rng.uniform(550, 800)
        loan_amount = rng.uniform(2000, 30000)
        loan_to_income = loan_amount / income
        risk = (
            -3.5
            + 10.0 * debt_ratio
            + 8.0 * loan_to_income
            - 0.02 * (credit_score - 550)
        )
        probability = 1.0 / (1.0 + 2.718281828 ** (-risk))
        defaulted = 1 if rng.random() < probability else 0
        # Only ever nonzero *after* a default has already happened -- the leak.
        days_past_due = rng.randint(30, 180) if defaulted else 0
        out.append({{
            "income": round(income, 2),
            "debt_ratio": round(debt_ratio, 3),
            "credit_score": round(credit_score),
            "loan_amount": round(loan_amount, 2),
            "days_past_due": days_past_due,
            "defaulted": defaulted,
        }})
    return out
"""

_LEGITIMATE_FEATURES = ["income", "debt_ratio", "credit_score", "loan_amount"]


def test_clean_claim_certifies() -> None:
    registry = Registry()
    runner = Runner(registry, executor=SubprocessExecutor(timeout=60))
    outcome = author(
        "clean_default_model",
        _PROPOSAL.format(name="clean_default_model"),
        runner=runner,
        serve=serve.table(),
        claim={"target": "defaulted", "features": _LEGITIMATE_FEATURES},
        registry=registry,
    )
    assert outcome.result.oracle_verdict == "sound"
    assert outcome.certified
    assert outcome.derivation.is_served and outcome.derivation.is_certified


def test_leaked_feature_is_rejected() -> None:
    registry = Registry()
    runner = Runner(registry, executor=SubprocessExecutor(timeout=60))
    outcome = author(
        "leaky_default_model",
        _PROPOSAL.format(name="leaky_default_model"),
        runner=runner,
        serve=serve.table(),
        claim={
            "target": "defaulted",
            "features": [*_LEGITIMATE_FEATURES, "days_past_due"],
        },
        registry=registry,
    )
    assert outcome.result.oracle_verdict == "unsound"
    assert not outcome.certified
    assert outcome.derivation.status == "proposed"


def test_unsound_proposal_never_becomes_an_mcp_tool() -> None:
    """The literal claim this example makes, proven at the public boundary
    (AGENTS.md's testing standard): not "the derivation is marked unsound" but
    "an agent connected over MCP cannot see or call it at all".
    """
    registry = Registry()
    runner = Runner(registry, executor=SubprocessExecutor(timeout=60))
    author(
        "leaky_default_model_2",
        _PROPOSAL.format(name="leaky_default_model_2"),
        runner=runner,
        serve=serve.table(),
        claim={
            "target": "defaulted",
            "features": [*_LEGITIMATE_FEATURES, "days_past_due"],
        },
        registry=registry,
    )

    server = build_server(registry, lambda: runner)
    tools = asyncio.run(server.list_tools())
    assert "run_leaky_default_model_2" not in {tool.name for tool in tools}
