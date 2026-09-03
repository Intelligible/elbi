"""Score each trap: run its gate, judge correctness, and compare against the LLM.

Build pass/fail is *only* gate correctness (the verdict the gate returned and, when
pinned, the pivotal reason). The bare-LLM comparison is a report signal: when a trap
has a captured LLM verdict that the gate does not beat, a non-fatal warning is raised,
never a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .trap import Trap, TrapResult


def score(trap: Trap, baseline: dict[str, dict[str, Any]]) -> TrapResult:
    """Run one trap and score it against its expectation and the cached LLM verdict."""
    report = trap.verify(trap.data())
    verdict = report.verdict
    pivotal_text = report.render()
    passed = verdict == trap.expected_verdict and (
        trap.expected_pivotal is None or trap.expected_pivotal in pivotal_text
    )

    # The cached baseline is report-only, so a missing or malformed capture must never
    # crash the run: treat anything but a known verdict string as uncaptured.
    capture = baseline.get(trap.id) or {}
    llm_verdict = capture.get("verdict")
    if llm_verdict not in ("sound", "unsound", "inconclusive"):
        llm_verdict = None
    llm_model = capture.get("model")
    if not isinstance(llm_model, str):
        llm_model = None
    beats_llm = None if llm_verdict is None else verdict != llm_verdict
    # A captured trap the gate does not beat is "saturated" (the bare LLM agrees): low
    # discrimination, a pruning candidate per the IRT/saturation grounding.
    saturated = llm_verdict is not None and not beats_llm
    warning = None
    if saturated:
        warning = (
            f"{trap.id}: bare LLM also returned {llm_verdict!r} (no gate advantage)"
        )
    return TrapResult(
        trap=trap,
        verdict=verdict,
        pivotal_text=pivotal_text,
        passed=passed,
        llm_verdict=llm_verdict,
        llm_model=llm_model,
        beats_llm=beats_llm,
        saturated=saturated,
        warning=warning,
    )


def run_benchmark(
    traps: Sequence[Trap],
    baseline: dict[str, dict[str, Any]],
    on_trap: Callable[[int, int, Trap], None] | None = None,
) -> list[TrapResult]:
    """Score every trap in catalog order.

    ``on_trap(index, total, trap)`` is called before each trap runs, so a caller can
    report progress (the gate verification, not the instant data build, is the wait).
    """
    total = len(traps)
    results: list[TrapResult] = []
    for i, trap in enumerate(traps, start=1):
        if on_trap is not None:
            on_trap(i, total, trap)
        results.append(score(trap, baseline))
    return results
