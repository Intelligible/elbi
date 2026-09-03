"""Verdict report types and the shared decision thresholds."""

from __future__ import annotations

from dataclasses import dataclass

#: A coefficient counts as significant when its |t| exceeds this (normal
#: approximation to p<0.05). The checks ask whether a claim survives perturbation,
#: not for exact p-values.
_T = 1.96
#: A stricter |t| (about p<0.001) for one-shot diagnostics that have no overturn or
#: resampling backstop, so a single test's ~5% false-positive rate does not flag a
#: genuinely sound claim as unsound.
_T_STRICT = 3.29
#: Conditioning on a covariate must change the x-y correlation by more than this to
#: read as confounding (it shrinks) or collider control (it grows); within it the
#: covariate is treated as neutral.
_SIGNAL = 0.05
#: A subset keeping fewer than this fraction of rows is not an honest subset.
_MIN_KEEP = 0.5
#: Lag-1 autocorrelation (in row order) above which a series reads as non-stationary
#: (random-walk-like). Two such series can correlate spuriously from shared drift, so
#: the association must survive differencing. Shuffled cross-sectional data sits near
#: zero and never triggers this; a random walk sits near one.
_RANDOM_WALK = 0.85


@dataclass(frozen=True)
class Check:
    """One soundness check and whether the claim survived it."""

    name: str
    survived: bool
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    """The outcome of verifying a claimed effect of ``x`` on ``y``."""

    verdict: str  # "sound" | "unsound" | "inconclusive"
    effect: float
    significant: bool
    checks: tuple[Check, ...]
    pivotal: str | None
    caveats: tuple[str, ...]

    def render(self) -> str:
        """Format the report as markdown for an agent to read."""
        lines = [
            f"# Verification: **{self.verdict.upper()}**",
            "",
            f"estimate = {self.effect:+.3g}"
            + ("" if self.significant else " (not statistically significant)"),
        ]
        if self.pivotal:
            lines.append(f"\npivotal issue: {self.pivotal}")
        lines.append("\nchecks:")
        for c in self.checks:
            lines.append(f"- [{'ok' if c.survived else 'FAIL'}] {c.name}: {c.detail}")
        if self.caveats:
            lines.append("\ncaveats (resolve with domain knowledge):")
            lines += [f"- {c}" for c in self.caveats]
        return "\n".join(lines)
