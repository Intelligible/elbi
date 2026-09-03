"""Unit-tests for the bite runner's logic, without a real mutation sweep.

A real sweep runs in the ``benchmark-bite`` CI job; here the Cosmic Ray calls are
stubbed so the runner's loop (early-stop at the first kill, counting, rejecting a
vacuous trap, skipping a trap with no bite target) is exercised fast and hermetically.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from benchmark import bite
from benchmark.bite import BiteResult, run_bite
from benchmark.loader import load_traps
from cosmic_ray.work_item import TestOutcome as _Outcome
from cosmic_ray.work_item import WorkerOutcome as _WorkerOutcome

SURVIVED = SimpleNamespace(test_outcome=_Outcome.SURVIVED, worker_outcome=None)
KILLED = SimpleNamespace(test_outcome=_Outcome.KILLED, worker_outcome=None)
INCOMPETENT = SimpleNamespace(
    test_outcome=_Outcome.INCOMPETENT, worker_outcome=_WorkerOutcome.EXCEPTION
)


@pytest.fixture
def simpsons_trap():
    (trap,) = [t for t in load_traps() if t.id == "simpsons_ab_reversal"]
    return trap


def _stub(monkeypatch, outcomes):
    """Stub Cosmic Ray so run_bite consumes the given per-mutant outcomes in order."""
    monkeypatch.setattr(bite, "_cosmic_ray", lambda *a, **k: None)
    items = [SimpleNamespace(job_id=str(i), mutations=[]) for i in range(len(outcomes))]
    monkeypatch.setattr(bite, "_work_items", lambda _session: items)
    seq = iter(outcomes)
    monkeypatch.setattr(bite, "mutate_and_test", lambda *a, **k: next(seq))


def test_passed_semantics() -> None:
    live = BiteResult("t", "effect.py", killed=1, survived=3, total=9, skipped=None)
    assert live.passed
    assert not BiteResult("t", "effect.py", 0, 9, 9, skipped=None).passed
    # A trap with no bite target does not gate the build.
    assert BiteResult("t", None, 0, 0, 0, "no bite module declared").passed


def test_stops_at_first_kill(monkeypatch, simpsons_trap) -> None:
    _stub(monkeypatch, [SURVIVED, INCOMPETENT, SURVIVED, KILLED, SURVIVED])
    result = run_bite(simpsons_trap)
    assert result.passed
    assert result.killed == 1
    assert result.survived == 2  # two survivors before the kill; the 5th never runs
    assert result.total == 5


def test_rejects_a_vacuous_trap(monkeypatch, simpsons_trap) -> None:
    _stub(monkeypatch, [SURVIVED, SURVIVED, SURVIVED])
    result = run_bite(simpsons_trap)
    assert not result.passed
    assert result.killed == 0
    assert result.survived == 3


def test_skips_trap_without_bite_target(monkeypatch, simpsons_trap) -> None:
    trap = dataclasses.replace(simpsons_trap, bite_module=None)
    # No stubbing needed: run_bite returns before touching Cosmic Ray.
    result = run_bite(trap)
    assert result.skipped is not None
    assert result.passed


def test_config_scopes_module_and_case() -> None:
    from pathlib import Path

    toml = bite._config_toml(Path("/x/effect.py"), "simpsons_ab_reversal")
    assert 'module-path = "/x/effect.py"' in toml
    assert "benchmark._bitecheck simpsons_ab_reversal" in toml


def test_select_filters_by_module_and_trap() -> None:
    traps = load_traps()
    by_module = bite._select(traps, "effect.py", None)
    assert by_module and all(t.bite_module == "effect.py" for t in by_module)
    (one,) = bite._select(traps, None, "leakage_churn")
    assert one.id == "leakage_churn"


def test_every_bite_module_resolves() -> None:
    # A mistyped bite.module would otherwise surface only in the slow bite job.
    for trap in load_traps():
        if trap.bite_module is not None:
            path = bite.VERIFICATION_DIR / trap.bite_module
            assert path.exists(), f"{trap.id}: bite module missing: {path}"


def test_mutant_order_is_stable() -> None:
    # Ordering must not drift run to run (job ids are fresh UUIDs; keys are stable).
    item = SimpleNamespace(
        job_id="unstable-uuid",
        mutations=[
            SimpleNamespace(
                module_path="effect.py",
                operator_name="op",
                occurrence=3,
                start_pos=(4, 5),
            )
        ],
    )
    assert bite._mutant_key(item) == bite._mutant_key(item)
