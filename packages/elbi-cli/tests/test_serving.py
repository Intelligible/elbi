"""Tests for stale-while-revalidate serving."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from elbi_cli.serving import ServeOutcome, Serving
from elbi_core import (
    Artifact,
    AuditEvent,
    AuditSink,
    Context,
    Dataset,
    Registry,
    Runner,
    cache,
    derivation,
    param,
    serve,
)
from elbi_core.config import DataBindings
from elbi_core.errors import DerivationError


class _RecordingSink:
    """An audit sink that keeps events in memory for assertions."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


async def _drain(serving: Serving) -> None:
    for _ in range(200):
        if not serving._inflight:
            return
        await asyncio.sleep(0.005)


def _build(
    tmp_path: Path,
    *,
    never: bool = False,
    fail: list[bool] | None = None,
    audit: AuditSink | None = None,
):
    path = tmp_path / "n.txt"
    path.write_text("2", encoding="utf-8")
    registry = Registry()
    calls: list[int] = []
    fail = fail if fail is not None else [False]

    @derivation(
        inputs={"n": Dataset("n")},
        serve=serve.json(),
        cache=cache.never() if never else cache.auto(),
        registry=registry,
        name="value",
    )
    def value(ctx: Context) -> Artifact:
        calls.append(1)
        if fail[0]:
            raise RuntimeError("boom")
        return Artifact.json(int(ctx.input("n").rows[0]["value"]))

    # The fixture file is a single-column CSV-ish; load via a real CSV.
    (tmp_path / "n.csv").write_text("value\n2\n", encoding="utf-8")
    bindings = DataBindings(bindings={"n": "n.csv"})

    def make_runner() -> Runner:
        return Runner(registry, bindings=bindings, base_dir=tmp_path)

    serving = Serving(registry, make_runner, audit=audit)
    return serving, tmp_path, calls, fail


def test_miss_then_hit(tmp_path: Path) -> None:
    serving, _base, calls, _ = _build(tmp_path)

    async def scenario() -> None:
        first = await serving.serve("value")
        assert first.status == "miss" and first.text == "2"
        second = await serving.serve("value")
        assert second.status == "hit" and second.text == "2"
        assert len(calls) == 1

    asyncio.run(scenario())


def test_stale_serves_old_then_background_refreshes(tmp_path: Path) -> None:
    serving, base, _calls, _ = _build(tmp_path)

    async def scenario() -> None:
        await serving.serve("value")  # miss → caches "2"
        (base / "n.csv").write_text("value\n9\n", encoding="utf-8")  # input changed
        stale = await serving.serve("value")
        assert stale.status == "stale"
        assert stale.text == "2"  # returns the OLD value immediately
        await _drain(serving)  # background refresh completes
        fresh = await serving.serve("value")
        assert fresh.status == "hit"
        assert fresh.text == "9"  # now updated

    asyncio.run(scenario())


def test_uncached_recomputes_every_call(tmp_path: Path) -> None:
    serving, _base, calls, _ = _build(tmp_path, never=True)

    async def scenario() -> None:
        a = await serving.serve("value")
        b = await serving.serve("value")
        assert a.status == "uncached" and b.status == "uncached"
        assert len(calls) == 2

    asyncio.run(scenario())


def test_stale_if_error_keeps_last_good(tmp_path: Path) -> None:
    fail = [False]
    serving, base, _calls, _ = _build(tmp_path, fail=fail)

    async def scenario() -> None:
        await serving.serve("value")  # caches "2"
        (base / "n.csv").write_text("value\n9\n", encoding="utf-8")
        fail[0] = True  # background refresh will raise
        with pytest.warns(UserWarning, match="background refresh"):
            stale = await serving.serve("value")
            assert stale.text == "2"
            await _drain(serving)
        # Refresh failed; last-good ("2") is still served.
        again = await serving.serve("value")
        assert again.text == "2"

    asyncio.run(scenario())


def test_ttl_freshness(tmp_path: Path) -> None:
    (tmp_path / "n.csv").write_text("value\n2\n", encoding="utf-8")
    registry = Registry()

    @derivation(
        inputs={"n": Dataset("n")},
        serve=serve.json(),
        cache=cache.auto(ttl=10),
        registry=registry,
        name="value",
    )
    def value(ctx: Context) -> Artifact:
        return Artifact.json(int(ctx.input("n").rows[0]["value"]))

    bindings = DataBindings(bindings={"n": "n.csv"})
    now = [0.0]
    serving = Serving(
        registry,
        lambda: Runner(registry, bindings=bindings, base_dir=tmp_path),
        clock=lambda: now[0],
    )

    async def scenario() -> None:
        assert (await serving.serve("value")).status == "miss"
        now[0] = 5.0
        assert (await serving.serve("value")).status == "hit"  # within ttl
        now[0] = 20.0
        assert (await serving.serve("value")).status == "stale"  # ttl expired
        await _drain(serving)

    asyncio.run(scenario())


def test_expire_forces_blocking_recompute(tmp_path: Path) -> None:
    (tmp_path / "n.csv").write_text("value\n2\n", encoding="utf-8")
    registry = Registry()
    calls: list[int] = []

    @derivation(
        inputs={"n": Dataset("n")},
        serve=serve.json(),
        cache=cache.auto(ttl=10, expire=100),
        registry=registry,
        name="value",
    )
    def value(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(int(ctx.input("n").rows[0]["value"]))

    bindings = DataBindings(bindings={"n": "n.csv"})
    now = [0.0]
    serving = Serving(
        registry,
        lambda: Runner(registry, bindings=bindings, base_dir=tmp_path),
        clock=lambda: now[0],
    )

    async def scenario() -> None:
        assert (await serving.serve("value")).status == "miss"  # cold
        now[0] = 50.0  # ttl < age < expire → serve stale + background refresh
        assert (await serving.serve("value")).status == "stale"
        await _drain(serving)
        now[0] = 500.0  # age > expire → must NOT serve stale; block and recompute
        assert (await serving.serve("value")).status == "miss"
        assert len(calls) == 3  # cold + background refresh + expire-forced recompute

    asyncio.run(scenario())


def test_refresh_is_deduplicated(tmp_path: Path) -> None:
    serving, base, _calls, _ = _build(tmp_path)

    async def scenario() -> None:
        await serving.serve("value")
        (base / "n.csv").write_text("value\n9\n", encoding="utf-8")
        # Two stale serves in flight should schedule only one refresh.
        await serving.serve("value")
        await serving.serve("value")
        assert len(serving._inflight) <= 1
        await _drain(serving)

    asyncio.run(scenario())


# --- Governance seams (Gap 5: audit trail + authorizer) ---


def test_uncached_serve_audits_without_a_version(tmp_path: Path) -> None:
    sink = _RecordingSink()
    serving, _base, _calls, _ = _build(tmp_path, never=True, audit=sink)

    async def scenario() -> None:
        await serving.serve("value")

    asyncio.run(scenario())
    assert sink.events[0].outcome == "ok"
    assert sink.events[0].version is None


def test_run_error_is_audited_then_reraised(tmp_path: Path) -> None:
    sink = _RecordingSink()
    serving, _base, _calls, _ = _build(tmp_path, never=True, fail=[True], audit=sink)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            await serving.serve("value")

    asyncio.run(scenario())
    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"
    assert sink.events[0].error == "RuntimeError"


def test_serve_records_an_ok_event_with_version(tmp_path: Path) -> None:
    sink = _RecordingSink()
    serving, _base, _calls, _ = _build(tmp_path, audit=sink)

    async def scenario() -> None:
        await serving.serve("value")

    asyncio.run(scenario())
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.derivation == "value"
    assert event.outcome == "ok"
    assert event.version is not None  # the content-addressed provenance


def test_a_cached_serve_renders_the_same_as_an_uncached_one(tmp_path: Path) -> None:
    """A hit renders the cached artifact, so it has to agree with the long way round.

    Every field but the cache status: a hit that rendered differently would be a
    different answer to the same question depending on how recently it was asked.
    """
    serving, _base, _calls, _ = _build(tmp_path)

    async def scenario() -> tuple[ServeOutcome, ServeOutcome]:
        cold = await serving.serve("value")
        warm = await serving.serve("value")
        return cold, warm

    cold, warm = asyncio.run(scenario())

    # The paths first: if both calls missed, everything below proves nothing.
    assert (cold.status, warm.status) == ("miss", "hit")
    assert warm.text == cold.text
    assert warm.preview == cold.preview
    assert warm.structured == cold.structured


def test_structured_params_are_servable_and_keyed(tmp_path: Path) -> None:
    # Regression: object/array params must not break the per-call cache key, which
    # previously hashed param values directly (a dict/list value is unhashable).
    registry = Registry()

    @derivation(
        params={"record": param.object(), "ids": param.array(items="integer")},
        serve=serve.json(),
        registry=registry,
        name="echo",
    )
    def echo(ctx: Context) -> Artifact:
        return Artifact.json({"n": len(ctx.param("record")), "ids": ctx.param("ids")})

    serving = Serving(registry, lambda: Runner(registry, base_dir=tmp_path))

    async def scenario() -> None:
        args = {"record": {"a": 1, "b": 2}, "ids": [3, 1, 2]}
        assert (await serving.serve("echo", args)).status == "miss"
        assert (await serving.serve("echo", args)).status == "hit"  # same params
        other = await serving.serve("echo", {"record": {"a": 1}, "ids": [9]})
        assert other.status == "miss"  # different params → distinct cache key

    asyncio.run(scenario())


# --- Components provenance stamping ---


def _build_components(tmp_path: Path, *, never: bool = False):
    """A components-serving derivation whose one fact echoes the input value, so a
    changed input produces a visibly different component to test staleness against.
    """
    (tmp_path / "n.csv").write_text("value\n2\n", encoding="utf-8")
    registry = Registry()

    @derivation(
        inputs={"n": Dataset("n")},
        serve=serve.components(),
        cache=cache.never() if never else cache.auto(),
        registry=registry,
        name="facts",
    )
    def facts(ctx: Context) -> Artifact:
        value = ctx.input("n").rows[0]["value"]
        return Artifact.components(
            [
                {
                    "id": "test/n_value",
                    "type": "column",
                    "scope": {"dataset": "n"},
                    "statement": f"n is {value}.",
                }
            ]
        )

    bindings = DataBindings(bindings={"n": "n.csv"})
    serving = Serving(
        registry, lambda: Runner(registry, bindings=bindings, base_dir=tmp_path)
    )
    return serving, tmp_path


def _stamped(outcome: ServeOutcome) -> dict:
    return outcome.structured["components"][0]


def test_components_miss_stamps_derivation_and_version(tmp_path: Path) -> None:
    serving, _base = _build_components(tmp_path)

    async def scenario() -> ServeOutcome:
        return await serving.serve("facts")

    outcome = asyncio.run(scenario())
    assert outcome.status == "miss"
    component = _stamped(outcome)
    assert component["provenance"]["derivation"] == "facts"
    assert component["provenance"]["derivation_version"]
    assert "- n is 2." in outcome.text


def test_components_hit_keeps_the_same_stamped_version(tmp_path: Path) -> None:
    serving, _base = _build_components(tmp_path)

    async def scenario() -> tuple[ServeOutcome, ServeOutcome]:
        first = await serving.serve("facts")
        second = await serving.serve("facts")
        return first, second

    first, second = asyncio.run(scenario())
    assert second.status == "hit"
    assert _stamped(second)["provenance"] == _stamped(first)["provenance"]


def test_components_stale_serve_keeps_the_old_version_not_the_new_one(
    tmp_path: Path,
) -> None:
    """A stale serve renders the OLD artifact; stamping it with the freshly computed
    (mismatched) version would claim a served component is current when it is not.
    """
    serving, base = _build_components(tmp_path)

    async def scenario() -> tuple[ServeOutcome, ServeOutcome]:
        first = await serving.serve("facts")  # miss -> caches "n is 2."
        (base / "n.csv").write_text("value\n9\n", encoding="utf-8")
        stale = await serving.serve("facts")
        await _drain(serving)
        return first, stale

    first, stale = asyncio.run(scenario())
    assert stale.status == "stale"
    assert "- n is 2." in stale.text  # still the OLD value
    assert (
        _stamped(stale)["provenance"]["derivation_version"]
        == (_stamped(first)["provenance"]["derivation_version"])
    )


def test_components_uncached_still_gets_a_version_stamped(tmp_path: Path) -> None:
    """Uncached normally skips computing a version at all; `components` is the one
    format that needs it anyway, to stamp provenance.
    """
    serving, _base = _build_components(tmp_path, never=True)

    async def scenario() -> ServeOutcome:
        return await serving.serve("facts")

    outcome = asyncio.run(scenario())
    assert outcome.status == "uncached"
    assert _stamped(outcome)["provenance"]["derivation_version"]


def test_components_author_provided_provenance_is_not_overwritten(
    tmp_path: Path,
) -> None:
    (tmp_path / "n.csv").write_text("value\n2\n", encoding="utf-8")
    registry = Registry()

    @derivation(
        inputs={"n": Dataset("n")},
        serve=serve.components(),
        registry=registry,
        name="facts",
    )
    def facts(ctx: Context) -> Artifact:
        return Artifact.components(
            [
                {
                    "id": "test/rule",
                    "type": "domain_knowledge",
                    "scope": {"dataset": "n"},
                    "statement": "Analysts should double-check n above 100.",
                    "provenance": {"source": "human", "author": "analyst@example.com"},
                }
            ]
        )

    bindings = DataBindings(bindings={"n": "n.csv"})
    serving = Serving(
        registry, lambda: Runner(registry, bindings=bindings, base_dir=tmp_path)
    )

    async def scenario() -> ServeOutcome:
        return await serving.serve("facts")

    outcome = asyncio.run(scenario())
    provenance = _stamped(outcome)["provenance"]
    assert provenance["source"] == "human"
    assert provenance["author"] == "analyst@example.com"
    assert provenance["derivation"] == "facts"  # gap-filled, not overwritten


def test_serving_internal_derivation_is_rejected(tmp_path: Path) -> None:
    registry = Registry()

    @derivation(registry=registry, name="internal")
    def internal(ctx: Context) -> Artifact:  # no serve contract
        return Artifact.json(1)

    serving = Serving(registry, lambda: Runner(registry, base_dir=tmp_path))

    async def scenario() -> None:
        with pytest.raises(DerivationError, match="internal"):
            await serving.serve("internal")

    asyncio.run(scenario())
