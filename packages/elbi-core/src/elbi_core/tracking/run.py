"""The certified-run record: one execution of a derivation the oracle certified.

A :class:`CertifiedRun` is one entry in a derivation's *result history*. Unlike a
conventional experiment tracker, which records whatever metric the training code
reports, every number here is the verification oracle's own certified estimate, carried
with its verdict, the gates that ran, and the content-addressed identity that makes it
reproducible.

A run is keyed on the derivation's ``derivation_version`` (a hash of its code, params,
and input versions). An identical re-run yields the same version and collapses to the
same record; a change to the data, the code, or the controls yields a new version, so a
derivation's history is the series of its distinct certified versions over time. The
components (``code_version``, ``input_versions``) are kept alongside the composite so
the history can say *what moved* the number between two versions, not merely *that* it
moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..authoring import AuthorOutcome

#: The check tuple carried per gate: (name, verdict, one-line detail).
Check = tuple[str, str, str]


def now_iso() -> str:
    """The current time as an ISO 8601 UTC string (the run's event time)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CertifiedRun:
    """One certified execution of a derivation, keyed on its content-addressed version.

    ``derivation_version`` is the run's identity: an identical re-run yields the same
    version and collapses to one record, while a change to the code, the data, or the
    controls yields a new version and a new record. The record is sealed at
    certification and never mutated. ``estimate`` is the oracle's verified magnitude for
    the claim (``None`` for a run with no scalar estimate, e.g. a data-contract check);
    ``adjusted_for`` names the columns it holds fixed, so it is read as a direct effect
    given those controls. ``code_version`` and ``input_versions`` decompose the
    identity, so a history can attribute a moved estimate to code, data, or controls.
    """

    name: str
    derivation_version: str
    verdict: str
    created_at: str
    data_hash: str | None = None
    estimate: float | None = None
    estimate_label: str | None = None
    adjusted_for: tuple[str, ...] = ()
    claim: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    deps: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()
    skipped: tuple[str, ...] = ()
    code_version: str | None = None
    input_versions: dict[str, str] = field(default_factory=dict)
    question: str = ""
    conversation_id: str | None = None

    @property
    def short_version(self) -> str:
        """The leading 12 characters of the version, for compact display."""
        return self.derivation_version[:12]

    def metrics(self) -> dict[str, float]:
        """The run's numeric summary values.

        A denormalized one-value-per-key view: the certified estimate. These are the
        scalars a history table shows per version.
        """
        out: dict[str, float] = {}
        if self.estimate is not None:
            out["estimate"] = self.estimate
        return out

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable mapping (tuples become lists) for durable storage."""
        return {
            "name": self.name,
            "derivation_version": self.derivation_version,
            "verdict": self.verdict,
            "created_at": self.created_at,
            "data_hash": self.data_hash,
            "estimate": self.estimate,
            "estimate_label": self.estimate_label,
            "adjusted_for": list(self.adjusted_for),
            "claim": dict(self.claim),
            "params": dict(self.params),
            "deps": list(self.deps),
            "checks": [list(c) for c in self.checks],
            "skipped": list(self.skipped),
            "code_version": self.code_version,
            "input_versions": dict(self.input_versions),
            "question": self.question,
            "conversation_id": self.conversation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertifiedRun:
        """Rebuild a run from :meth:`to_dict`'s mapping (lists become tuples)."""
        return cls(
            name=data["name"],
            derivation_version=data["derivation_version"],
            verdict=data["verdict"],
            created_at=data["created_at"],
            data_hash=data.get("data_hash"),
            estimate=data.get("estimate"),
            estimate_label=data.get("estimate_label"),
            adjusted_for=tuple(data.get("adjusted_for") or ()),
            claim=dict(data.get("claim") or {}),
            params=dict(data.get("params") or {}),
            deps=tuple(data.get("deps") or ()),
            checks=tuple(tuple(c) for c in data.get("checks") or ()),
            skipped=tuple(data.get("skipped") or ()),
            code_version=data.get("code_version"),
            input_versions=dict(data.get("input_versions") or {}),
            question=data.get("question", ""),
            conversation_id=data.get("conversation_id"),
        )


def run_from_author(
    outcome: AuthorOutcome,
    question: str = "",
    created_at: str | None = None,
    conversation_id: str | None = None,
) -> CertifiedRun | None:
    """Build a :class:`CertifiedRun` from an author outcome, or ``None`` if uncertified.

    Only a certified derivation becomes a run: an uncertified proposal has no verified
    number to track. The estimate and gates come from the oracle's attestation (or the
    data contract's, for a cleaning step), and the version and its components from the
    verification result. ``created_at`` defaults to now, injectable so a caller can
    supply the store's own timestamp.
    """
    if not outcome.certified:
        return None
    derivation = outcome.derivation
    result = outcome.result
    attestation = result.oracle_attestation or result.contract_attestation or {}
    verdict = result.oracle_verdict or result.contract_verdict or "unverified"
    return CertifiedRun(
        name=derivation.name,
        derivation_version=result.data_version or "",
        verdict=verdict,
        created_at=created_at or now_iso(),
        data_hash=attestation.get("data_hash"),
        estimate=attestation.get("estimate"),
        estimate_label=attestation.get("estimate_label"),
        adjusted_for=tuple(attestation.get("adjusted_for") or ()),
        claim=dict(derivation.claim or {}),
        deps=tuple(derivation.deps or ()),
        checks=tuple(
            (c["name"], c["verdict"], c["detail"])
            for c in attestation.get("checks", [])
        ),
        skipped=tuple(attestation.get("skipped", [])),
        code_version=result.code_version,
        input_versions=dict(result.input_versions or {}),
        question=question,
        conversation_id=conversation_id,
    )
