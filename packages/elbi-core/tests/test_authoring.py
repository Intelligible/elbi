"""Tests for the agent-authoring loop: propose -> verify -> certify."""

from __future__ import annotations

import pytest

from elbi_core import (
    AutoCertifyOnVerify,
    CertificationPolicy,
    Dataset,
    GoldenCase,
    ManualCertification,
    Registry,
    RequireChecks,
    Runner,
    SubprocessExecutor,
    author,
    certify,
    param,
    propose,
    serve,
    verify,
)
from elbi_core.errors import DerivationError, SpecValidationError

# Two rows in a fixed order: lets us tell ordered from unordered comparison.
_RANKED = """
def ranked(ctx):
    "Rows in a fixed order."
    return [{"id": "a", "score": 2}, {"id": "b", "score": 1}]
"""

# Reads a param, so a golden case can drive it with different inputs.
_SCALED = """
def scaled(ctx):
    "Scale ten by the factor param."
    return [{"v": ctx.param("factor") * 10}]
"""

_SOURCE = """
def revenue(ctx):
    "Total revenue."
    return [{"total": 42}]
"""


def _runner(reg: Registry) -> Runner:
    return Runner(reg, executor=SubprocessExecutor(timeout=30))


def test_propose_registers_agent_proposed(registry: Registry) -> None:
    d = propose("revenue", _SOURCE, serve=serve.table(), registry=registry)
    assert d.origin == "agent"
    assert d.status == "proposed"
    assert d.is_agent_authored and not d.is_certified
    assert d.source == _SOURCE
    assert d.description == "Total revenue."
    assert "revenue" in registry


def test_propose_infers_depends_on_from_derivation_inputs(registry: Registry) -> None:
    up = propose("up", "def up(ctx):\n    return 'u'\n", registry=registry)
    down = propose(
        "down",
        "def down(ctx):\n    return 'd'\n",
        inputs={"u": up},
        registry=registry,
    )
    assert down.depends_on == ("up",)


def test_propose_rejects_syntax_error(registry: Registry) -> None:
    with pytest.raises(DerivationError, match="not valid Python"):
        propose("broken", "def broken(ctx) ->\n", registry=registry)


def test_propose_requires_matching_function_name(registry: Registry) -> None:
    with pytest.raises(DerivationError, match="must define a top-level function"):
        propose("wanted", "def other(ctx):\n    return 1\n", registry=registry)


def test_propose_explicit_description_overrides_docstring(registry: Registry) -> None:
    d = propose("revenue", _SOURCE, description="Custom.", registry=registry)
    assert d.description == "Custom."


def test_propose_without_docstring_has_no_description(registry: Registry) -> None:
    d = propose("plain", "def plain(ctx):\n    return 1\n", registry=registry)
    assert d.description is None


def test_propose_extracts_docstring_past_other_nodes(registry: Registry) -> None:
    # A leading import (not a function) and a helper function before the target
    # exercise the scan for the named function's docstring.
    src = (
        "import math\n\n"
        "def helper():\n    return math.floor(1.5)\n\n"
        "def main(ctx):\n    'Main doc.'\n    return [{'n': helper()}]\n"
    )
    d = propose("main", src, serve=serve.table(), registry=registry)
    assert d.description == "Main doc."


def test_propose_invalid_name_fails_spec(registry: Registry) -> None:
    with pytest.raises(SpecValidationError):
        propose("Bad", "def Bad(ctx):\n    return 1\n", registry=registry)


def test_verify_runs_and_renders(registry: Registry) -> None:
    d = propose("revenue", _SOURCE, serve=serve.table(), registry=registry)
    result = verify(_runner(registry), d)
    assert result.ok
    assert result.matched_prediction is None
    assert result.artifact is not None
    assert "| total |" in (result.rendered or "")


def test_verify_matches_prediction(registry: Registry) -> None:
    d = propose("revenue", _SOURCE, serve=serve.table(), registry=registry)
    result = verify(_runner(registry), d, predicted=[{"total": 42}])
    assert result.ok and result.matched_prediction is True


def test_verify_detects_prediction_mismatch(registry: Registry) -> None:
    d = propose("revenue", _SOURCE, serve=serve.table(), registry=registry)
    result = verify(_runner(registry), d, predicted=[{"total": 0}])
    assert result.matched_prediction is False
    assert result.ok is False


def test_verify_captures_execution_failure(registry: Registry) -> None:
    d = propose(
        "kaboom",
        "def kaboom(ctx):\n    raise ValueError('x')\n",
        serve=serve.text(),
        registry=registry,
    )
    result = verify(_runner(registry), d)
    assert result.ok is False
    assert result.error is not None and "ValueError" in result.error
    assert result.rendered is None


def test_verify_internal_derivation_has_no_render(registry: Registry) -> None:
    d = propose(
        "internal_node",
        "def internal_node(ctx):\n    return [{'total': 1}]\n",
        registry=registry,
    )  # no serve
    result = verify(_runner(registry), d)
    assert result.ok and result.rendered is None


_NONDETERMINISTIC = """
import random

def flaky(ctx):
    "An unseeded draw: a different value on every run, so not reproducible."
    return [{"draw": random.randint(0, 10**9)}]
"""


def test_verify_rejects_nondeterministic_derivation(registry: Registry) -> None:
    # A certified result must be a pure function of its inputs; an unseeded draw is not.
    d = propose("flaky", _NONDETERMINISTIC, serve=serve.table(), registry=registry)
    result = verify(_runner(registry), d)
    assert result.ok is False
    assert result.error is not None and "reproducible" in result.error
    # Opting out of the check no longer blocks it, which proves the rejection comes
    # from the determinism check and not from something else (reverting the check in
    # verify makes this derivation certify, so the assertion above fails).
    assert verify(_runner(registry), d, check_determinism=False).ok


def test_verify_reproducible_derivation_passes(registry: Registry) -> None:
    # A seeded draw reproduces exactly across runs and is not flagged.
    seeded = (
        "import random\n\n"
        "def seeded(ctx):\n"
        '    "A seeded draw: the same value on every run."\n'
        "    return [{'draw': random.Random(0).randint(0, 10**9)}]\n"
    )
    d = propose("seeded", seeded, serve=serve.table(), registry=registry)
    assert verify(_runner(registry), d).ok


def test_certify_promotes_and_reregisters(registry: Registry) -> None:
    d = propose("revenue", _SOURCE, serve=serve.table(), registry=registry)
    certified = certify(d, registry=registry)
    assert certified.is_certified
    assert registry.get("revenue").status == "certified"
    # Provenance is preserved through certification.
    assert certified.origin == "agent"
    assert certified.source == _SOURCE


def test_certify_uses_active_registry_by_default() -> None:
    reg = Registry()
    from elbi_core.registry import use_registry

    with use_registry(reg):
        d = propose(
            "solo", "def solo(ctx):\n    return [{'n': 1}]\n", serve=serve.table()
        )
        certify(d)
    assert reg.get("solo").is_certified


def test_certified_agent_derivation_runs_under_sandbox(registry: Registry) -> None:
    propose("revenue", _SOURCE, serve=serve.table(), registry=registry)
    out = _runner(registry).serve("revenue")
    assert "42" in out


def test_author_auto_certifies_on_clean_verify(registry: Registry) -> None:
    outcome = author(
        "revenue",
        _SOURCE,
        runner=_runner(registry),
        serve=serve.table(),
        registry=registry,
    )
    assert outcome.result.ok
    assert outcome.certified is True
    assert outcome.derivation.is_certified
    assert registry.get("revenue").status == "certified"


def test_author_manual_policy_holds_proposed(registry: Registry) -> None:
    outcome = author(
        "revenue",
        _SOURCE,
        runner=_runner(registry),
        serve=serve.table(),
        policy=ManualCertification(),
        registry=registry,
    )
    assert outcome.result.ok
    assert outcome.certified is False
    assert registry.get("revenue").status == "proposed"


def test_author_does_not_certify_failed_verification(registry: Registry) -> None:
    outcome = author(
        "kaboom",
        "def kaboom(ctx):\n    raise ValueError('x')\n",
        runner=_runner(registry),
        serve=serve.text(),
        registry=registry,
    )
    assert outcome.result.ok is False
    assert outcome.certified is False
    assert registry.get("kaboom").status == "proposed"


def test_author_prediction_mismatch_is_not_certified(registry: Registry) -> None:
    outcome = author(
        "revenue",
        _SOURCE,
        runner=_runner(registry),
        serve=serve.table(),
        predicted=[{"total": 0}],
        registry=registry,
    )
    assert outcome.certified is False
    assert outcome.result.matched_prediction is False


def test_certification_policies_decide() -> None:
    from elbi_core import VerificationResult

    d = propose("p", "def p(ctx):\n    return 1\n", registry=Registry())
    passed = VerificationResult(ok=True)
    failed = VerificationResult(ok=False, error="boom")
    assert isinstance(AutoCertifyOnVerify(), CertificationPolicy)
    assert AutoCertifyOnVerify().should_certify(d, passed) is True
    assert AutoCertifyOnVerify().should_certify(d, failed) is False
    assert ManualCertification().should_certify(d, passed) is False


def test_dataset_input_proposal_partitions(registry: Registry) -> None:
    d = propose(
        "over_sales",
        "def over_sales(ctx):\n    return list(ctx.input('sales').rows)\n",
        inputs={"sales": Dataset("sales")},
        registry=registry,
    )
    assert set(d.dataset_inputs()) == {"sales"}


# --- Golden cases (Gap 2: certification on execution-grounded evidence) ---


def test_verify_passes_all_golden_cases(registry: Registry) -> None:
    d = propose("ranked", _RANKED, serve=serve.table(), registry=registry)
    result = verify(
        _runner(registry),
        d,
        cases=[GoldenCase(expected=[{"id": "a", "score": 2}, {"id": "b", "score": 1}])],
    )
    assert result.ok and result.matched_prediction is True
    assert result.cases_total == 1 and result.cases_passed == 1


def test_verify_fails_when_a_case_mismatches(registry: Registry) -> None:
    d = propose("ranked", _RANKED, serve=serve.table(), registry=registry)
    result = verify(
        _runner(registry),
        d,
        cases=[
            GoldenCase(expected=[{"id": "a", "score": 2}, {"id": "b", "score": 1}]),
            GoldenCase(expected=[{"id": "a", "score": 999}]),
        ],
    )
    assert result.ok is False
    assert result.cases_total == 2 and result.cases_passed == 1


def test_verify_ordered_comparison_is_order_sensitive(registry: Registry) -> None:
    d = propose("ranked", _RANKED, serve=serve.table(), registry=registry)
    # Same rows, wrong order: rejected because order is meaningful by default.
    result = verify(
        _runner(registry),
        d,
        cases=[GoldenCase(expected=[{"id": "b", "score": 1}, {"id": "a", "score": 2}])],
    )
    assert result.ok is False


def test_verify_unordered_comparison_ignores_row_order(registry: Registry) -> None:
    d = propose("ranked", _RANKED, serve=serve.table(), registry=registry)
    result = verify(
        _runner(registry),
        d,
        cases=[
            GoldenCase(
                expected=[{"id": "b", "score": 1}, {"id": "a", "score": 2}],
                unordered=True,
            )
        ],
    )
    assert result.ok is True


def test_verify_case_drives_params(registry: Registry) -> None:
    d = propose(
        "scaled",
        _SCALED,
        serve=serve.table(),
        params={"factor": param.integer()},
        registry=registry,
    )
    result = verify(
        _runner(registry),
        d,
        cases=[
            GoldenCase(expected=[{"v": 30}], params={"factor": 3}),
            GoldenCase(expected=[{"v": 50}], params={"factor": 5}),
        ],
    )
    assert result.ok and result.cases_passed == 2


def test_verify_case_execution_failure_is_captured(registry: Registry) -> None:
    d = propose(
        "scaled",
        _SCALED,
        serve=serve.table(),
        params={"factor": param.integer()},
        registry=registry,
    )
    # An unknown param makes the run fail; verify reports it rather than raising.
    result = verify(
        _runner(registry),
        d,
        cases=[GoldenCase(expected=[{"v": 0}], params={"nope": 1})],
    )
    assert result.ok is False and result.error is not None


def test_require_checks_certifies_only_with_passing_cases(registry: Registry) -> None:
    outcome = author(
        "ranked",
        _RANKED,
        runner=_runner(registry),
        serve=serve.table(),
        cases=[GoldenCase(expected=[{"id": "a", "score": 2}, {"id": "b", "score": 1}])],
        policy=RequireChecks(),
        registry=registry,
    )
    assert outcome.certified is True
    assert registry.get("ranked").is_certified


def test_require_checks_refuses_a_run_only_proposal(registry: Registry) -> None:
    outcome = author(
        "ranked",
        _RANKED,
        runner=_runner(registry),
        serve=serve.table(),
        policy=RequireChecks(),  # no cases: ran, but nothing was checked
        registry=registry,
    )
    assert outcome.result.ok is True
    assert outcome.certified is False
    assert registry.get("ranked").status == "proposed"


def test_require_checks_is_a_certification_policy() -> None:
    assert isinstance(RequireChecks(), CertificationPolicy)
