"""Run a derivation as a durable background job.

Authoring a derivation (sandboxed run, oracle gate, cache, persist) is the expensive,
possibly long step: a real model fit belongs here, not in the interactive scratchpad.
This bridges that step onto the :class:`~elbi.JobRunner` so the agent can launch
it and continue: it content-addresses the request (so an identical derivation dedupes
onto the same job), wraps the app's ``DeriveFn`` as the job's work, and serializes the
outcome for persistence. The heavy compute still runs behind the ``Executor`` seam; this
only changes when the caller waits for it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from elbi_agent import DeriveFn, DeriveOutcome
from elbi_core import Job, JobRunner
from elbi_core.versioning import hash_bytes


def derivation_job_key(
    name: str,
    source: str,
    claim: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    fmt: str,
    deps: Sequence[str],
    datasets: Sequence[str],
) -> str:
    """A content address for a derivation request, so identical work dedupes.

    Keyed by everything that determines the certified result: the source, its claim,
    its contract, output format, dependencies, and the datasets it reads. The name is
    deliberately excluded, so re-running the same analysis under a new name reuses the
    finished job.
    """
    payload = json.dumps(
        {
            "source": source,
            "claim": dict(claim) if claim else None,
            "contract": dict(contract) if contract else None,
            "fmt": fmt,
            "deps": sorted(deps),
            "datasets": sorted(datasets),
        },
        sort_keys=True,
    )
    return hash_bytes(payload.encode("utf-8"))


def outcome_result(outcome: DeriveOutcome) -> dict[str, Any]:
    """The job-result payload for a derivation outcome (JSON-serializable, no rows).

    The certified rows live in the persisted derivation and are fetched by name when a
    visualization needs them, so the durable job row stays small.
    """
    return {
        "certified": outcome.certified,
        "verdict": outcome.verdict,
        "contract_verdict": outcome.contract_verdict,
        "rendered": outcome.rendered,
        "checks": [list(check) for check in outcome.checks],
        "data_hash": outcome.data_hash,
        "detail": outcome.detail,
        "contract_detail": outcome.contract_detail,
        "error": outcome.error,
    }


def submit_derivation_job(
    job_runner: JobRunner,
    derive: DeriveFn,
    *,
    name: str,
    source: str,
    claim: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    fmt: str,
    assumptions: Sequence[str],
    deps: Sequence[str],
    datasets: Sequence[str],
) -> Job:
    """Submit a derivation to run as a background job; return its handle at once.

    The work runs the app's normal authoring path (which certifies, caches, and persists
    on success), so a job that succeeds leaves exactly the same durable derivation a
    synchronous ``derive`` would have. Deduped by :func:`derivation_job_key`.
    """
    key = derivation_job_key(name, source, claim, contract, fmt, deps, datasets)

    def work(progress: Any, cancelled: Any) -> dict[str, Any]:
        progress(f"authoring and verifying {name}")
        return outcome_result(
            derive(name, source, claim, contract, fmt, assumptions, deps)
        )

    return job_runner.submit(key, name, work)
