"""Bridge the chat runtime's ``derive`` to the project's authoring loop and the store.

The chat answers by authoring a derivation. This builds the ``DeriveFn`` the runtime
calls: it runs the same ``author()`` loop the MCP ``propose_derivation`` uses (a
sandboxed run, golden and oracle verification, certification) against the project's
runner and registry, then, on a sound conclusion, lets the runner cache it and persists
it to the native store. Unlike ``propose_derivation`` it does not register an MCP tool:
chat-authored derivations are the app's own durable, cached answers, served natively.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from elbi_agent import DeriveFn, DeriveOutcome
from elbi_cli.project import LoadedProject
from elbi_core import DataContract, Dataset, Serve
from elbi_core import serve as serve_builders
from elbi_core.authoring import ManualCertification, author
from elbi_core.errors import ElbiError
from elbi_core.tracking import run_from_author

from .db import Derivation, Store
from .tracking import emit_configured_run

logger = logging.getLogger("elbi")

#: Cap the rows carried to the client for a visualization: enough for a dense map or
#: scatter, bounded so the stream stays small even on a large dataset.
_VIZ_ROW_CAP = 20000

#: Per-name locks: authoring mutates the shared registry from a threadpool, so
#: same-name runs serialize while distinct names keep authoring in parallel.
_LOCKS_GUARD = threading.Lock()
_NAME_LOCKS: dict[str, threading.Lock] = {}


def _name_lock(name: str) -> threading.Lock:
    """The lock serializing registry mutations for one derivation name."""
    with _LOCKS_GUARD:
        return _NAME_LOCKS.setdefault(name, threading.Lock())


def _serve_for(fmt: str) -> Serve:
    """The serve contract for a format name, defaulting to a table."""
    builders = {
        "table": serve_builders.table,
        "markdown": serve_builders.markdown,
        "json": serve_builders.json,
        "text": serve_builders.text,
    }
    return builders.get(fmt, serve_builders.table)(title=None)


def _output_rows(result: Any) -> tuple[dict[str, Any], ...]:
    """The derivation's certified rows (for a visualization), if it produced a table."""
    artifact = getattr(result, "artifact", None)
    value = getattr(artifact, "value", None)
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return tuple(value[:_VIZ_ROW_CAP])
    return ()


class DeriveFactory(Protocol):
    """Builds a ``DeriveFn`` bound to a conversation id and a question.

    A Protocol rather than a ``Callable`` alias so the two arguments stay named: the
    chat path passes a real conversation, while the promote paths (explore, a notebook
    cell) pass a synthetic one standing for where the source came from.
    """

    def __call__(self, conversation_id: str, question: str) -> DeriveFn:
        """Bind one authoring context and return the ``DeriveFn`` that runs it."""
        ...


def make_derive_factory(
    project: LoadedProject,
    store: Store,
    make_runner: Callable[..., Any],
    dataset_names: Callable[[], list[str]],
) -> DeriveFactory:
    """Build the per-conversation factory the chat endpoint uses to author derivations.

    The returned factory binds a conversation id and question; the ``DeriveFn`` it
    produces authors, certifies, caches, and persists one derivation per call.
    ``make_runner`` and ``dataset_names`` come from the app so an authored derivation
    reads its dataset inputs from the warehouse (the only source of data), and the set
    of bindable datasets is the live warehouse table set.
    """

    def factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(
            name: str,
            source: str,
            claim: Mapping[str, Any] | None,
            contract: Mapping[str, Any] | None,
            fmt: str,
            assumptions: Sequence[str],
            deps: Sequence[str],
        ) -> DeriveOutcome:
            with _name_lock(name):
                # A re-derive of the same name replaces the earlier registry entry;
                # the runner's cache is keyed by content, so an identical re-run
                # still hits it.
                if name in project.registry:
                    project.registry.remove(name)
                try:
                    data_contract = (
                        DataContract.from_manifest(dict(contract)) if contract else None
                    )
                    outcome = author(
                        name,
                        source,
                        runner=make_runner(),
                        serve=_serve_for(fmt),
                        inputs={d: Dataset(d) for d in dataset_names()},
                        claim=dict(claim) if claim else None,
                        contract=data_contract,
                        deps=list(deps),
                        registry=project.registry,
                    )
                except ElbiError as exc:
                    return DeriveOutcome(certified=False, error=str(exc))

            result = outcome.result
            if result.error:
                return DeriveOutcome(certified=False, error=result.error)
            oracle_att = result.oracle_attestation or {}
            contract_att = result.contract_attestation or {}
            # The attestation persisted and returned: the oracle's for a conclusion, the
            # contract's for a cleaning step, or both when a derivation declares each.
            attestation = dict(oracle_att)
            if contract_att:
                attestation["contract"] = contract_att
            checks = tuple(
                (c["name"], c["verdict"], c["detail"])
                for c in oracle_att.get("checks", [])
            )
            data_hash = oracle_att.get("data_hash") or contract_att.get("data_hash")
            verdict = result.oracle_verdict
            rendered = result.rendered or ""

            if outcome.certified:
                # Persist to the project's authored sidecar as well as the app
                # store, so a chat-authored derivation reloads into the registry
                # on restart. That durability is what lets a feature-engineering
                # derivation keep serving as a training source and a scoring
                # transform, not just as this conversation's answer.
                authored_store = getattr(project, "authored_store", None)
                if authored_store is not None:
                    try:
                        authored_store.save(
                            outcome.derivation, attestation=attestation or None
                        )
                    except ElbiError:
                        logger.exception(
                            "failed to persist %s to the authored sidecar", name
                        )
                store.save_derivation(
                    Derivation(
                        name=name,
                        conversation_id=conversation_id,
                        # None (the chat path) falls back to save_derivation's
                        # conversation-row inheritance; promote paths pass the
                        # caller because their conversation ids are synthetic.
                        question=question,
                        source=source,
                        claim_json=json.dumps(dict(claim)) if claim else None,
                        serve_json=json.dumps({"format": fmt, "deps": list(deps)}),
                        verdict=verdict or result.contract_verdict,
                        rendered=rendered,
                        attestation_json=(
                            json.dumps(attestation) if attestation else None
                        ),
                        assumptions_json=(
                            json.dumps(list(assumptions)) if assumptions else None
                        ),
                        data_hash=data_hash,
                    )
                )
                # Record the certified run for the version history and comparison, and
                # export it to a team's MLflow if one is configured (both no-ops on a
                # deterministic re-run: the run collapses to its existing version).
                run = run_from_author(
                    outcome, question=question, conversation_id=conversation_id
                )
                if run is not None and store.append_run(run):
                    emit_configured_run(store, run)
                return DeriveOutcome(
                    certified=True,
                    verdict=verdict,
                    contract_verdict=result.contract_verdict,
                    rendered=rendered,
                    checks=checks,
                    data_hash=data_hash,
                    rows=_output_rows(result),
                    # The oracle's own estimate rides through so the answer reports the
                    # certified number, not a figure the model computed while exploring.
                    estimate=oracle_att.get("estimate"),
                    estimate_label=oracle_att.get("estimate_label"),
                    adjusted_for=tuple(oracle_att.get("adjusted_for") or ()),
                    derivation_version=result.data_version,
                )
            detail = result.oracle_detail or "held for review (not certified)"
            return DeriveOutcome(
                certified=False,
                verdict=verdict,
                contract_verdict=result.contract_verdict,
                rendered=rendered,
                checks=checks,
                detail=detail,
                contract_detail=result.contract_detail or "",
            )

        return derive

    return factory


#: ``DeriveFn`` minus ``assumptions``: proposes and verifies one derivation and
#: returns the verdict view, certifying nothing.
DraftFn = Callable[
    [
        str,
        str,
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
        str,
        Sequence[str],
    ],
    dict[str, Any],
]


def make_draft_factory(
    project: LoadedProject,
    make_runner: Callable[..., Any],
    dataset_names: Callable[[], list[str]],
) -> Callable[[str, str], DraftFn]:
    """Build the factory the Explore workbench uses to *draft* a derivation.

    The ``DraftFn`` proposes and verifies one derivation and reports the verdict,
    certifying and persisting nothing. The proposal must sit in the project registry
    while it verifies (the runner resolves names there), so the draft registers it
    under ``ManualCertification`` and always removes it again, restoring any prior
    holder of the name -- nothing it leaves behind can serve. Certifying is the
    explicit follow-up (the promote path), the one place a derivation becomes durable.
    """

    def factory(conversation_id: str, question: str) -> DraftFn:
        def draft(
            name: str,
            source: str,
            claim: Mapping[str, Any] | None,
            contract: Mapping[str, Any] | None,
            fmt: str,
            deps: Sequence[str],
        ) -> dict[str, Any]:
            with _name_lock(name):
                # Hold any existing holder of the name aside; a preview never
                # clobbers a served artifact.
                prior = project.registry.get(name) if name in project.registry else None
                if prior is not None:
                    project.registry.remove(name)
                try:
                    data_contract = (
                        DataContract.from_manifest(dict(contract)) if contract else None
                    )
                    outcome = author(
                        name,
                        source,
                        runner=make_runner(),
                        serve=_serve_for(fmt),
                        inputs={d: Dataset(d) for d in dataset_names()},
                        claim=dict(claim) if claim else None,
                        contract=data_contract,
                        deps=list(deps),
                        policy=ManualCertification(),
                        registry=project.registry,
                    )
                except ElbiError as exc:
                    return {"ok": False, "error": str(exc)}
                finally:
                    # Whatever the outcome, leave no trace: drop the proposal and
                    # restore the prior holder.
                    if name in project.registry:
                        project.registry.remove(name)
                    if prior is not None:
                        project.registry.register(prior)

            result = outcome.result
            oracle_att = result.oracle_attestation or {}
            checks = [
                [c["name"], c["verdict"], c["detail"]]
                for c in oracle_att.get("checks", [])
            ]
            detail = (
                result.oracle_detail
                or result.contract_detail
                or ("" if result.ok else "held for review (not certified)")
            )
            return {
                # ``ok`` mirrors the certify gate exactly, so the UI enables Certify
                # only when the follow-up click would succeed.
                "ok": bool(result.ok),
                "name": name,
                "certified": False,
                "verdict": result.oracle_verdict,
                "contract_verdict": result.contract_verdict,
                "checks": checks,
                "rendered": result.rendered or "",
                "detail": detail,
                "error": result.error,
            }

        return draft

    return factory
