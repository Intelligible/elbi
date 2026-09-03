"""The verdict of checking a contract: typed, located violations and a report.

The verdict is three-valued (``sound`` / ``unsound`` / ``inconclusive``), the same
vocabulary the verification oracle uses, so a contract folds into the authoring gate
and its attestation exactly like a claim. A contract is a conjunction: every clause must
pass. A clause that cannot be evaluated (a missing column, an empty table) is
``inconclusive``, never a silent pass, so certification is not granted on a check that
did not actually run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SOUND = "sound"
UNSOUND = "unsound"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Violation:
    """One located constraint violation."""

    clause: str
    column: str | None
    row_number: int | None  # one-based; None for a table-level violation
    value: Any
    note: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable record for the attestation."""
        out: dict[str, Any] = {"clause": self.clause, "note": self.note}
        if self.column is not None:
            out["column"] = self.column
        if self.row_number is not None:
            out["rowNumber"] = self.row_number
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass(frozen=True)
class ClauseResult:
    """The outcome of one contract clause (one constraint on one field or the table)."""

    name: str
    verdict: str
    detail: str
    checked: int  # rows the clause was evaluated over (the tolerance denominator)
    violated: int
    sample: tuple[Violation, ...] = ()


@dataclass(frozen=True)
class ContractReport:
    """The combined outcome of checking every clause of a contract."""

    verdict: str
    clauses: tuple[ClauseResult, ...]
    row_count: int
    #: Violations kept per clause for the report and attestation; capped by the checker.
    violations: tuple[Violation, ...] = field(default_factory=tuple)

    @property
    def violated_clauses(self) -> tuple[ClauseResult, ...]:
        """Clauses whose verdict is not ``sound``."""
        return tuple(c for c in self.clauses if c.verdict != SOUND)

    def render(self) -> str:
        """Format the contract verdict as markdown for an agent to read."""
        lines = [f"# Contract: **{self.verdict.upper()}**", ""]
        lines.append(f"{len(self.clauses)} clauses over {self.row_count} rows")
        for clause in self.clauses:
            mark = {SOUND: "ok", UNSOUND: "FAIL", INCONCLUSIVE: "?"}[clause.verdict]
            lines.append(f"- [{mark}] {clause.name}: {clause.detail}")
        if self.violations:
            lines.append("\nsample violations:")
            for violation in self.violations:
                where = (
                    f"row {violation.row_number}"
                    if violation.row_number is not None
                    else "table"
                )
                lines.append(f"- {violation.clause} ({where}): {violation.note}")
        return "\n".join(lines)

    def attestation(self) -> dict[str, Any]:
        """A self-describing record of the contract check, for the served result."""
        return {
            "schema": "elbi.quality/v1",
            "verdict": self.verdict,
            "rowCount": self.row_count,
            "clauses": [
                {
                    "name": clause.name,
                    "verdict": clause.verdict,
                    "detail": clause.detail,
                    "checked": clause.checked,
                    "violated": clause.violated,
                }
                for clause in self.clauses
            ],
            "violations": [v.to_dict() for v in self.violations],
        }


def combine_verdict(clauses: tuple[ClauseResult, ...]) -> str:
    """Combine clause verdicts into the contract verdict (a conjunction).

    Any broken clause makes the contract ``unsound``; failing that, any clause that
    could not be evaluated makes it ``inconclusive``; only when every clause is sound
    (and at least one ran) is the contract ``sound``. An empty clause set is
    ``inconclusive`` -- nothing was actually checked.
    """
    if not clauses:
        return INCONCLUSIVE
    if any(c.verdict == UNSOUND for c in clauses):
        return UNSOUND
    if any(c.verdict == INCONCLUSIVE for c in clauses):
        return INCONCLUSIVE
    return SOUND
