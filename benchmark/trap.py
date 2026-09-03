"""Runtime types for one trap and its scored outcome.

A ``Trap`` is the runtime object the loader produces from a trap folder's
``manifest.json``: the metadata (including its ground-truth verdict, provenance, and
reviewer sign-off) plus two resolved callables (``data`` builds the rows, ``verify``
runs the gate). ``TrapResult`` is what the harness computes from running one trap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: A dataset as the oracle consumes it: a list of row dicts of stringified cells.
Rows = list[dict[str, str]]


@dataclass(frozen=True)
class Trap:
    """One trap: a dataset plus the gate call and the verdict it must return."""

    id: str  # the trap folder name
    pitfall: str
    description: str
    provenance: str  # "synthetic; ..." or a citation for a real (file-backed) dataset
    added_by: str
    reviewed_by: tuple[str, ...]  # sign-offs; >=2 required for real traps (see tests)
    status: str  # "candidate" | "accepted" (human-vetting only; both run in CI)
    gate: str  # the dispatch key resolved via gates.GATES
    claim: dict[str, Any]  # keyword args (role -> column) passed to the gate
    expected_verdict: str  # the verdict the gate must return for the trap to pass
    expected_pivotal: str | None  # substring that must appear in the report text
    rationale: str  # why this verdict, keyed to RUBRIC.md (auditable ground truth)
    naive_verdict: str  # the hand-written surface conclusion a bare read reaches
    naive_rationale: str  # why that naive read is fooled (report copy)
    bite_module: str | None  # verification/ module the bite gate mutates (or None)
    data: Callable[[], Rows]  # resolved by the loader (generator / file / inline)
    verify: Callable[[Rows], Any]  # resolved by the loader (gate + claim)
    is_real: bool  # file-backed (real) data, vs a seeded generator / inline rows
    fixture: str  # manifest path, for error messages


@dataclass(frozen=True)
class TrapResult:
    """The outcome of running one trap against its expectation and the LLM."""

    trap: Trap
    verdict: str  # what the gate actually returned
    pivotal_text: str  # the rendered report text searched for expected_pivotal
    passed: bool  # gate correctness: the only thing that gates the build
    llm_verdict: str | None  # the cached bare-LLM verdict, or None if uncaptured
    llm_model: str | None  # the model that produced the cached verdict
    beats_llm: bool | None  # gate disagrees with a captured LLM (report signal)
    saturated: bool  # LLM and gate agree -> low discrimination (prune candidate)
    warning: str | None  # set when a captured trap shows no gate advantage
