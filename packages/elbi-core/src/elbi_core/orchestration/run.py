"""Asset materialization: staleness and dependency-ordered runs over derivations.

A derivation is a software-defined asset. It is *stale* when its content version has
moved since it was last materialized (upstream data or code changed); *materialized*
when the versions match; *never* when it has no materialization on record. Materializing
runs the assets in dependency order, skipping ones already fresh (materializing is
idempotent: an unchanged asset is a cache hit), and retrying a transient failure.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from ..errors import ElbiError
from ..runner import Runner

#: An asset's freshness relative to its last materialization.
AssetStatus = Literal["materialized", "stale", "never"]

#: The outcome of materializing one asset in a run.
StepState = Literal["succeeded", "failed", "skipped"]

#: Returns an asset's last materialized content version, or ``None`` if never.
LastVersionFn = Callable[[str], str | None]

#: Evaluates data-quality checks for a just-materialized asset, returning each check's
#: outcome and whether the asset passed (a failed ``error`` check makes it ``False``).
CheckFn = Callable[[str], "tuple[list[dict[str, object]], bool]"]


@dataclass(frozen=True)
class MaterializationStep:
    """The result of materializing one asset."""

    asset: str
    state: StepState
    data_version: str | None = None
    verdict: str | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0
    #: The step's captured stdout/stderr (in-process execution), for the run-log view.
    logs: str = ""
    #: Data-quality check outcomes for this asset: each a dict of name/severity/expr/
    #: failed/total. An ``error``-severity failure turns the step ``failed``.
    checks: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize for the API/UI."""
        return {
            "asset": self.asset,
            "state": self.state,
            "data_version": self.data_version,
            "verdict": self.verdict,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "logs": self.logs,
            "checks": list(self.checks),
        }


@dataclass(frozen=True)
class RunResult:
    """The result of a materialization run over a set of assets."""

    steps: tuple[MaterializationStep, ...]

    @property
    def ok(self) -> bool:
        """Whether every asset materialized without failure."""
        return all(step.state != "failed" for step in self.steps)

    @property
    def failed(self) -> list[str]:
        """The assets that failed."""
        return [step.asset for step in self.steps if step.state == "failed"]


def asset_status(
    runner: Runner, assets: Sequence[str], last_version: LastVersionFn
) -> dict[str, AssetStatus]:
    """Classify each asset as materialized, stale, or never-materialized."""
    status: dict[str, AssetStatus] = {}
    for name in assets:
        try:
            current = runner.data_version(name)
        except ElbiError:
            continue
        last = last_version(name)
        if last is None:
            status[name] = "never"
        else:
            status[name] = "materialized" if current == last else "stale"
    return status


def materialize(
    runner: Runner,
    ordered: Sequence[str],
    *,
    last_version: LastVersionFn,
    verdict_of: Callable[[str], str | None] | None = None,
    check_asset: CheckFn | None = None,
    max_retries: int = 1,
    on_step: Callable[[MaterializationStep], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    """Materialize ``ordered`` assets (already in dependency order).

    An asset whose content version matches its last materialization is skipped; the rest
    run through ``runner``, retrying up to ``max_retries`` times with backoff on a
    transient failure. ``check_asset`` (optional) runs data-quality checks on a
    freshly-materialized asset; a failed ``error``-severity check turns its step
    ``failed``. ``on_step`` is called as each asset finishes, for live progress and
    persistence. ``should_cancel`` is checked at each asset boundary; when it's true the
    run stops cleanly and the not-yet-started assets are simply omitted (their
    successful upstreams stay materialized, so a later run resumes from where this left
    off: cancellation never leaves a half-written asset).
    """
    steps: list[MaterializationStep] = []
    for name in ordered:
        if should_cancel is not None and should_cancel():
            break
        step = _materialize_one(
            runner,
            name,
            last_version=last_version,
            verdict_of=verdict_of,
            check_asset=check_asset,
            max_retries=max_retries,
            clock=clock,
            sleep=sleep,
        )
        steps.append(step)
        if on_step is not None:
            on_step(step)
    return RunResult(steps=tuple(steps))


def _materialize_one(
    runner: Runner,
    name: str,
    *,
    last_version: LastVersionFn,
    verdict_of: Callable[[str], str | None] | None,
    check_asset: CheckFn | None,
    max_retries: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> MaterializationStep:
    try:
        current = runner.data_version(name)
    except ElbiError as exc:
        return MaterializationStep(name, "failed", error=str(exc))
    if current == last_version(name):
        # Fresh: skip the recompute, but still evaluate checks against the cached output
        # so a newly-added check takes effect (and current data is always re-validated)
        # without forcing a rebuild, like running an asset check on its own.
        checks, checks_ok = _run_checks(check_asset, name)
        return MaterializationStep(
            name,
            "skipped" if checks_ok else "failed",
            data_version=current,
            error=None if checks_ok else _check_failure(checks),
            checks=checks,
        )

    started = clock()
    error: str | None = None
    # Capture the compute's stdout/stderr for the run-log view. This catches output from
    # an in-process (human-authored) derivation; a sandboxed one runs in a subprocess,
    # whose stream the executor surfaces separately, so here only its error is recorded.
    output = io.StringIO()
    for attempt in range(1, max_retries + 2):
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                runner.run(name)
        except Exception as exc:  # a transient failure is worth a retry
            error = str(exc)
            if attempt <= max_retries:
                sleep(_backoff(attempt))
            continue
        checks, checks_ok = _run_checks(check_asset, name)
        return MaterializationStep(
            name,
            "succeeded" if checks_ok else "failed",
            data_version=current,
            verdict=verdict_of(name) if verdict_of else None,
            error=None if checks_ok else _check_failure(checks),
            attempts=attempt,
            duration_ms=_elapsed_ms(started, clock),
            logs=output.getvalue(),
            checks=checks,
        )
    return MaterializationStep(
        name,
        "failed",
        error=error,
        attempts=max_retries + 1,
        duration_ms=_elapsed_ms(started, clock),
        logs=output.getvalue(),
    )


def _run_checks(
    check_asset: CheckFn | None, name: str
) -> tuple[tuple[dict[str, object], ...], bool]:
    """Run an asset's data-quality checks → (outcomes, ok). No hook means it passes."""
    if check_asset is None:
        return (), True
    results, ok = check_asset(name)
    return tuple(results), ok


def _check_failure(checks: tuple[dict[str, object], ...]) -> str:
    """Why the checks blocked this asset, distinguishing broken from violated.

    A check that could not run says nothing about the data, and sending somebody to look
    for bad rows that do not exist is worse than saying nothing at all.
    """
    broken = [str(c.get("name")) for c in checks if c.get("state") == "error"]
    if broken:
        return f"a data-quality check could not run: {', '.join(broken)}"
    return "data-quality check failed"


def _backoff(attempt: int) -> float:
    """Exponential backoff in seconds, capped, for a retry after ``attempt`` tries."""
    return float(min(0.1 * 2 ** (attempt - 1), 5.0))


def _elapsed_ms(started: float, clock: Callable[[], float]) -> int:
    return int((clock() - started) * 1000)
