"""Stale-while-revalidate serving for derivations.

An agent gets the last-good answer immediately while an out-of-date derivation
refreshes in the background; concurrent refreshes are de-duplicated, and a failed
refresh keeps serving the last-good value (stale-if-error, RFC 5861). Freshness is
decided by the derivation's data version plus an optional TTL; derivations with
caching disabled are always computed fresh.
"""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from elbi_core import (
    Artifact,
    AuditEvent,
    AuditSink,
    ComponentError,
    DerivationError,
    NullAuditSink,
    Registry,
    Runner,
    Serve,
    stamp_provenance,
    validate_components,
)
from elbi_core.versioning import inputs_version

ParamValues = Mapping[str, Any]

# The per-call cache key. Params are folded to canonical JSON rather than a tuple
# of items, so object/array param values (dicts/lists) stay hashable as keys.
_Key = tuple[str, str]
_T = TypeVar("_T")


def _params_key(params: ParamValues) -> str:
    return json.dumps(dict(params), sort_keys=True, default=str)


@dataclass(frozen=True)
class Served:
    """The last artifact computed for a key, with its version and time.

    The artifact rather than a rendering, so a format change re-renders from what was
    computed instead of recomputing it.
    """

    artifact: Artifact
    data_version: str
    computed_at: float
    # `computed_at` is this layer's TTL clock; `cached_at` is when the runner's cache
    # first computed the version (None without an entry).
    inputs_version: str
    cached_at: float | None


@dataclass(frozen=True)
class ServeOutcome:
    """The result of a serve, with cache status for observability.

    ``text`` is the full render (what a resource read returns); ``preview`` is the
    context-sized inline text for the tool response; ``structured`` is the MCP
    ``structuredContent`` payload -- rows + count for a table, the full component
    objects + count for ``components`` -- else ``None``.
    """

    text: str
    status: str  # "hit" | "stale" | "miss" | "uncached"
    preview: str = ""
    structured: dict[str, Any] | None = None


class Serving:
    """Stale-while-revalidate cache around a :class:`~elbi.Runner`."""

    def __init__(
        self,
        registry: Registry,
        make_runner: Callable[[], Runner],
        *,
        clock: Callable[[], float] | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self._registry = registry
        self._make_runner = make_runner
        self._clock = clock if clock is not None else time.time
        self._last: dict[_Key, Served] = {}
        self._inflight: set[_Key] = set()
        self._audit = audit if audit is not None else NullAuditSink()

    async def serve(self, name: str, params: ParamValues | None = None) -> ServeOutcome:
        """Serve ``name`` from cache or by computing it, and record one audit event.

        Every invocation produces exactly one event: an ``error`` when the run raises,
        otherwise an ``ok`` carrying the content-addressed version that produced the
        answer.
        """
        params = dict(params or {})
        try:
            outcome, version = await self._serve(name, params)
        except Exception as exc:
            self._record(name, params, "error", None, type(exc).__name__)
            raise
        self._record(name, params, "ok", version, None)
        return outcome

    async def _serve(
        self, name: str, params: ParamValues
    ) -> tuple[ServeOutcome, str | None]:
        """The stale-while-revalidate body; returns the outcome and its version.

        The serve contract is resolved first, so an internal derivation is refused
        before a runner is built or the data version computed -- that answer does not
        depend on the result, so computing one would burn the work and take an effect
        derivation's side effect for a call that is refused anyway.
        """
        contract = self._contract(name)
        key: _Key = (name, _params_key(params))
        cache = self._registry.get(name).cache

        if not cache.enabled:
            runner = self._make_runner()
            # Caching is off, so no version is computed for anything else here --
            # except a `components` contract, which needs one to stamp provenance.
            # Read before computing, as the cached path does, so the stamp never
            # names a newer version than the one the artifact was computed under.
            data_version, inputs = (
                await _versions(runner, name, params)
                if contract.format == "components"
                else ("", "")
            )
            artifact = await self._compute(runner, name, params)
            served = Served(artifact, data_version, self._clock(), inputs, None)
            outcome = self._present(name, contract, "uncached", served)
            return outcome, None

        runner = self._make_runner()
        data_version, inputs = await _versions(runner, name, params)
        last = self._last.get(key)

        if last is not None:
            age = self._clock() - last.computed_at
            fresh = last.data_version == data_version and (
                cache.ttl is None or age < cache.ttl
            )
            if fresh:
                # The artifact being served is `last.artifact`, so it is stamped with
                # the version it was actually computed under -- `last.data_version`,
                # which happens to equal `data_version` here, but only because of
                # `fresh`.
                outcome = self._present(name, contract, "hit", last)
                return outcome, data_version
            if cache.expire is None or age < cache.expire:
                # Stale (within the expire ceiling): serve last-good now, refresh
                # in the background (deduped). Bounds stale-if-error to `expire`.
                self._schedule_refresh(key, name, params)
                # Still `last.artifact` under `last.data_version`, which does *not*
                # equal the freshly computed `data_version` here -- that mismatch is
                # exactly what makes this stale. Stamping with the wrong one would
                # claim a served component is current when it is not.
                outcome = self._present(name, contract, "stale", last)
                return outcome, data_version
            # Past `expire`: hard miss: block for fresh, never serve this old.

        artifact = await self._compute(runner, name, params)
        served = await self._served(runner, artifact, data_version, inputs)
        self._last[key] = served
        outcome = self._present(name, contract, "miss", served)
        return outcome, data_version

    async def _compute(
        self, runner: Runner, name: str, params: ParamValues
    ) -> Artifact:
        """Run the derivation off the event loop and return its artifact.

        A ``components`` artifact is validated here, once per computation, so a
        malformed item is refused on the miss that would have cached it and never
        reaches a hit, a stale serve or the search corpus.
        """
        artifact = await _in_thread(lambda: runner.run(name, dict(params)))
        if self._contract(name).format == "components":
            try:
                validate_components(artifact.value)
            except ComponentError as exc:
                raise ComponentError(f"derivation {name!r}: {exc}") from None
        return artifact

    async def _served(
        self, runner: Runner, artifact: Artifact, data_version: str, inputs: str
    ) -> Served:
        cached_at = await _in_thread(lambda: runner.computed_at(data_version))
        return Served(artifact, data_version, self._clock(), inputs, cached_at)

    def _present(
        self, name: str, contract: Serve, status: str, served: Served
    ) -> ServeOutcome:
        """Render an artifact for serving.

        For a ``components`` contract with a known ``data_version``, every item is
        stamped with this derivation's name and *this artifact's* version (the
        value was validated as a list of components when it was computed), plus the
        data it rests on and when that version was first computed. Both come from
        stored state, so serving an unchanged version twice stamps the same thing.
        """
        artifact = served.artifact
        if contract.format == "components":
            artifact = Artifact.components(
                [
                    stamp_provenance(
                        item,
                        derivation=name,
                        derivation_version=served.data_version,
                        inputs_version=served.inputs_version,
                        computed_at=_iso_utc(served.cached_at),
                    )
                    for item in artifact.value
                ]
            )
        return ServeOutcome(
            text=contract.render(artifact),
            status=status,
            preview=contract.preview(artifact),
            structured=contract.structured(artifact),
        )

    def _contract(self, name: str) -> Serve:
        """The derivation's serve contract; an internal derivation has none to serve."""
        contract = self._registry.get(name).serve
        if contract is None:
            raise DerivationError(
                f"derivation {name!r} is internal (no serve contract) and "
                "cannot be served"
            )
        return contract

    def _record(
        self,
        name: str,
        params: ParamValues,
        outcome: str,
        version: str | None,
        error: str | None,
    ) -> None:
        self._audit.record(
            AuditEvent(
                derivation=name,
                timestamp=self._clock(),
                outcome=outcome,
                version=version,
                params=dict(params),
                error=error,
            )
        )

    def _schedule_refresh(self, key: _Key, name: str, params: ParamValues) -> None:
        if key in self._inflight:
            return
        self._inflight.add(key)

        async def refresh() -> None:
            try:
                self._contract(name)  # never refresh what cannot be served
                runner = self._make_runner()
                data_version, inputs = await _versions(runner, name, params)
                # Caches the artifact and not a rendering: this task inherits the
                # context of whichever caller triggered it, so it must persist nothing
                # scoped to them into an entry every caller reads.
                artifact = await self._compute(runner, name, params)
                self._last[key] = await self._served(
                    runner, artifact, data_version, inputs
                )
            except Exception as exc:  # stale-if-error: any failure keeps last-good
                # Keep serving the last-good value, but surface the failure rather
                # than swallowing it silently.
                warnings.warn(
                    f"background refresh of {name!r} failed: {exc}", stacklevel=1
                )
            finally:
                self._inflight.discard(key)

        asyncio.get_running_loop().create_task(refresh())


async def _versions(runner: Runner, name: str, params: ParamValues) -> tuple[str, str]:
    """The composite data version and the hash of its input versions alone."""
    version, _code, inputs = await _in_thread(
        lambda: runner.version_parts(name, params)
    )
    return version, inputs_version(inputs)


def _iso_utc(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(round(timestamp, 6)))


async def _in_thread(thunk: Callable[[], _T]) -> _T:
    """Run a blocking no-arg callable in a thread so the loop is never blocked."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, thunk)
