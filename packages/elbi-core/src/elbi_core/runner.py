"""Local runner: resolve inputs, compute with incremental caching, and serve.

Each derivation has a data version, ``hash(code, params, input versions)``; a
cached result for that version (within any TTL) is reused instead of recomputing.
A derivation input contributes its output version, giving early cutoff: a parent
that recomputes to the same output does not invalidate its children. Results are
held in an in-process session store and an optional persistent
:class:`CacheStore`. Nondeterministic derivations are cached but never drive early
cutoff.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import versioning
from .artifact import Artifact, coerce_artifact
from .cache import (
    CachedResult,
    CachePolicy,
    CacheStore,
    MemoryCacheStore,
    derivation_tag,
)
from .config import DataBindings
from .context import Context
from .data import Table
from .derivation import Derivation
from .errors import CacheError, CycleError, DerivationError, ParamError
from .executor import DEFAULT as DEFAULT_EXECUTOR
from .executor import Executor
from .param import Param
from .registry import Registry
from .semantic_model import SemanticModel

ParamValues = Mapping[str, Any]


class Runner:
    """Executes derivations from a registry, with incremental caching."""

    def __init__(
        self,
        registry: Registry,
        *,
        bindings: DataBindings | None = None,
        base_dir: Path | None = None,
        store: CacheStore | None = None,
        executor: Executor | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._registry = registry
        self._bindings = bindings if bindings is not None else DataBindings()
        self._base_dir = base_dir if base_dir is not None else Path.cwd()
        self._store = store
        self._executor = executor if executor is not None else DEFAULT_EXECUTOR
        self._session = MemoryCacheStore()
        self._clock = clock if clock is not None else time.time

    def run(
        self, name: str, params: ParamValues | None = None, *, fresh: bool = False
    ) -> Artifact:
        """Compute (or reuse the cached) artifact for derivation ``name``.

        ``fresh`` forces a real execution of ``name`` itself, bypassing the cache both
        for the read and the write, so an identical second run does not simply return
        the first's cached value. Its inputs (datasets and upstream derivations) are
        still resolved through the cache; only this derivation's compute re-runs. Used
        to check that a derivation is reproducible before it is certified.
        """
        return self._resolve(name, params or {}, stack=(), fresh=fresh)[0]

    def serve(self, name: str, params: ParamValues | None = None) -> str:
        """Compute ``name`` and render it per its serve contract."""
        derivation = self._registry.get(name)
        if derivation.serve is None:
            raise DerivationError(
                f"derivation {name!r} is internal (no serve contract) and "
                "cannot be served; use run() to consume it"
            )
        artifact = self.run(name, params)
        return derivation.serve.render(artifact)

    def dataset(self, name: str) -> Table:
        """Load a bound dataset by name, independent of any derivation.

        Used to resolve a data contract's referential-integrity targets: a foreign key
        may reference a raw dataset, whose rows are needed to check membership.
        """
        return self._bindings.load_dataset(name, self._base_dir)

    def data_version(self, name: str, params: ParamValues | None = None) -> str:
        """Return the current data version for ``name`` without recomputing it.

        Fingerprints inputs (cheap for files / SQL freshness probes) and composes
        the version. Used by the serving layer to decide cache freshness.
        """
        return self._plan(name, params or {}, stack=()).data_version

    def version_parts(
        self, name: str, params: ParamValues | None = None
    ) -> tuple[str, str, dict[str, str]]:
        """Return ``name``'s data version and the components it composes.

        The tuple is ``(data_version, code_version, input_versions)``: the composite
        content hash, the code's own version, and each input's version by key.
        Decomposing the version lets a caller attribute a change to code versus data
        rather than seeing only that the composite moved. Does not recompute the result.
        """
        plan = self._plan(name, params or {}, stack=())
        return plan.data_version, plan.code_version, dict(plan.input_versions)

    def _resolve(
        self,
        name: str,
        params: ParamValues,
        stack: tuple[str, ...],
        *,
        fresh: bool = False,
    ) -> tuple[Artifact, str]:
        plan = self._plan(name, params, stack)
        now = self._clock()
        if plan.policy.enabled and not fresh:
            cached = self._get(plan.data_version)
            if cached is not None and _fresh(cached, now):
                return cached.artifact, cached.output_version

        resolved_inputs = plan.materialize_inputs()
        artifact = coerce_artifact(
            self._executor.run(plan.derivation, Context(resolved_inputs, plan.params))
        )
        output_version = _output_version(artifact, plan.policy, plan.data_version)

        if plan.policy.enabled and not fresh:
            self._put(
                plan.data_version,
                CachedResult(
                    artifact=artifact,
                    output_version=output_version,
                    computed_at=now,
                    ttl=plan.policy.ttl,
                    # Tag with the derivation name so `delete` can purge its cache.
                    tags=(*plan.policy.tags, derivation_tag(name)),
                ),
            )
        return artifact, output_version

    def _plan(self, name: str, params: ParamValues, stack: tuple[str, ...]) -> _Plan:
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise CycleError(f"dependency cycle detected: {cycle}")
        derivation = self._registry.get(name)
        resolved_params = _resolve_params(derivation, params)

        input_versions: dict[str, str] = {}
        dataset_inputs: dict[str, tuple[str, Table | None]] = {}
        upstream_inputs: dict[str, Artifact] = {}
        semantic_inputs: dict[str, SemanticModel] = {}

        for key, dataset in derivation.dataset_inputs().items():
            version, preloaded = self._bindings.version(dataset.name, self._base_dir)
            input_versions[f"dataset:{key}"] = version
            dataset_inputs[key] = (dataset.name, preloaded)

        for key, upstream in derivation.derivation_inputs().items():
            artifact, output_version = self._resolve(upstream.name, {}, (*stack, name))
            input_versions[f"derivation:{key}"] = output_version
            upstream_inputs[key] = artifact

        for key, model in derivation.semantic_model_inputs().items():
            # The document's content versions the derivation, so editing a definition
            # invalidates the results derived from it.
            input_versions[f"semantic_model:{key}"] = versioning.hash_text(
                model.document_json
            )
            semantic_inputs[key] = model

        policy = derivation.cache
        code = policy.code_version or _code_version(derivation)
        data_version = versioning.derivation_version(
            code=code,
            params=versioning.params_version(resolved_params),
            input_versions=input_versions,
        )

        def materialize_inputs() -> dict[str, Any]:
            resolved: dict[str, Any] = {}
            for key, (dataset_name, preloaded) in dataset_inputs.items():
                resolved[key] = (
                    preloaded
                    if preloaded is not None
                    else self._bindings.load_dataset(dataset_name, self._base_dir)
                )
            resolved.update(upstream_inputs)
            resolved.update(semantic_inputs)
            return resolved

        return _Plan(
            derivation=derivation,
            params=resolved_params,
            data_version=data_version,
            code_version=code,
            input_versions=input_versions,
            policy=policy,
            materialize_inputs=materialize_inputs,
        )

    def _get(self, data_version: str) -> CachedResult | None:
        cached = self._session.get(data_version)
        if cached is not None:
            return cached
        if self._store is not None:
            cached = self._store.get(data_version)
            if cached is not None:
                self._session.put(data_version, cached)
            return cached
        return None

    def _put(self, data_version: str, result: CachedResult) -> None:
        self._session.put(data_version, result)
        if self._store is not None:
            try:
                self._store.put(data_version, result)
            except CacheError as exc:
                warnings.warn(
                    f"could not persist cache for {result.output_version[:12]}: {exc}",
                    stacklevel=2,
                )


class _Plan:
    """A resolved derivation ready to check the cache or compute."""

    def __init__(
        self,
        *,
        derivation: Derivation,
        params: Mapping[str, Any],
        data_version: str,
        code_version: str,
        input_versions: Mapping[str, str],
        policy: CachePolicy,
        materialize_inputs: Callable[[], dict[str, Any]],
    ) -> None:
        self.derivation = derivation
        self.params = params
        self.data_version = data_version
        self.code_version = code_version
        self.input_versions = input_versions
        self.policy = policy
        self.materialize_inputs = materialize_inputs


def _code_version(derivation: Derivation) -> str:
    """The code version of a derivation, by where its logic lives.

    Agent-authored derivations carry their logic as ``source`` (their ``compute``
    refuses to run in-process), so they are fingerprinted from that source;
    human-authored ones are fingerprinted from the compute function object.
    """
    if derivation.source is not None:
        return versioning.source_version(derivation.source)
    return versioning.code_version(derivation.compute)


def _fresh(result: CachedResult, now: float) -> bool:
    """Whether a cached result is within its TTL (content-addressing aside)."""
    if result.ttl is None:
        return True
    return (now - result.computed_at) < result.ttl


def _output_version(artifact: Artifact, policy: CachePolicy, data_version: str) -> str:
    """Return the version a derivation exposes to consumers, for early cutoff.

    Deterministic derivations expose a hash of their output, so an unchanged
    output stops invalidation propagating. Nondeterministic ones expose their data
    version, as do opaque artifacts, whose arbitrary objects have no stable content
    hash; either way consumers recompute only when inputs or code change.
    """
    if not policy.deterministic or artifact.kind == "opaque":
        return data_version
    return versioning.compose("output", versioning.hash_json(artifact.value))


def _resolve_params(derivation: Derivation, supplied: ParamValues) -> dict[str, Any]:
    """Validate supplied params against the declared params and apply defaults."""
    declared: Mapping[str, Param] = derivation.params
    unknown = set(supplied) - set(declared)
    if unknown:
        known = ", ".join(sorted(declared)) or "(none)"
        raise ParamError(
            f"{derivation.name}: unknown parameter(s) {sorted(unknown)}; "
            f"declared: {known}"
        )

    resolved: dict[str, Any] = {}
    for pname, spec in declared.items():
        if pname in supplied:
            try:
                resolved[pname] = spec.coerce(supplied[pname])
            except ValueError as exc:
                raise ParamError(f"{derivation.name}.{pname}: {exc}") from exc
        elif spec.required:
            raise ParamError(f"{derivation.name}: missing required parameter {pname!r}")
        else:
            resolved[pname] = spec.default
    return resolved
