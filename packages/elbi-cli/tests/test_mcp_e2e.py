"""End-to-end MCP tests: an in-memory client drives the MCPServer server.

These exercise the whole public path a real agent uses (the client serializes
arguments, the server decodes them, routes through the serving cache and the
runner, runs, and renders) rather than any component in isolation. The
unhashable-dict bug in the serving cache key survived 386 isolated tests because
none called a tool with a structured argument through this path; every param
shape is covered here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from elbi_cli.mcp_server import build_server
from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    Runner,
    derivation,
    param,
    serve,
)


def test_propose_emits_tools_list_changed() -> None:
    """Authoring a derivation notifies the client its tool list changed.

    Without the notification a connected client never learns the new ``run_<name>``
    tool exists in-session. Proven over the real client message path.
    """
    from mcp.types import ToolListChangedNotification

    from elbi_core import RoutingExecutor, SubprocessExecutor

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(
            registry, executor=RoutingExecutor(sandbox=SubprocessExecutor())
        ),
        enable_propose=True,
    )
    seen: list[str] = []

    async def handler(msg: Any) -> None:
        root = getattr(msg, "root", msg)
        if isinstance(root, ToolListChangedNotification):
            seen.append(root.method)

    async def run() -> None:
        # Pinned to the handshake era: `message_handler` observes a notification stream,
        # which is how that era delivers change events.
        async with Client(server, mode="legacy", message_handler=handler) as client:
            await client.call_tool(
                "propose_derivation",
                {"name": "nb", "source": "def nb(ctx):\n    return [{'x': 1}]\n"},
            )
            await client.list_tools()  # round-trip so the notification is delivered

    asyncio.run(run())
    assert seen, "server did not emit tools/list_changed after authoring a derivation"


def test_propose_records_a_certified_run(tmp_path: Path) -> None:
    """A certified proposal is recorded in the run log for the version history.

    Proven over the real client path: the run log is written by the same authoring
    hook the ``dev`` server wires up, so a local certification joins the tracked
    history exactly as it does in the app.
    """
    from elbi_core import RoutingExecutor, SubprocessExecutor
    from elbi_core.tracking import JsonlRunLog

    registry = Registry()
    run_log = JsonlRunLog(tmp_path / "runs.jsonl")
    server = build_server(
        registry,
        lambda: Runner(
            registry, executor=RoutingExecutor(sandbox=SubprocessExecutor())
        ),
        enable_propose=True,
        run_log=run_log,
    )

    async def run() -> None:
        async with Client(server) as client:
            await client.call_tool(
                "propose_derivation",
                {"name": "nb", "source": "def nb(ctx):\n    return [{'x': 1}]\n"},
            )

    asyncio.run(run())
    recorded = run_log.runs(name="nb")
    assert len(recorded) == 1
    # "nb" declares no claim, so nothing was oracle-checked; the run is genuinely
    # recorded (this test's real subject), but its verdict must say so honestly
    # rather than default to a fabricated "sound".
    assert recorded[0].verdict == "unverified"
    assert recorded[0].derivation_version


def _registry() -> Registry:
    """A registry covering every served param shape, plus an opaque-model path."""
    registry = Registry()

    @derivation(
        name="filter_rows",
        params={
            "zipcode": param.string(),
            "limit": param.integer(required=False, default=10),
        },
        serve=serve.table(),
        registry=registry,
    )
    def filter_rows(ctx: Context) -> Artifact:
        return Artifact.table(
            [{"zip": ctx.param("zipcode"), "limit": ctx.param("limit")}]
        )

    @derivation(
        name="score_one",
        params={"record": param.object()},
        serve=serve.json(),
        registry=registry,
    )
    def score_one(ctx: Context) -> Artifact:
        record = ctx.param("record")
        return Artifact.json({"fields": sorted(record), "n": len(record)})

    @derivation(
        name="score_batch",
        params={
            "rows": param.array(items="object"),
            "ids": param.array(items="integer"),
        },
        serve=serve.table(),
        registry=registry,
    )
    def score_batch(ctx: Context) -> Artifact:
        rows = ctx.param("rows")
        return Artifact.table([{"i": i, "fields": len(r)} for i, r in enumerate(rows)])

    @derivation(name="model", registry=registry)  # internal: opaque artifact
    def model(ctx: Context) -> Artifact:
        return Artifact.opaque({"bias": 100.0})

    @derivation(
        name="predict",
        inputs={"model": model},
        params={"features": param.object()},
        serve=serve.json(),
        registry=registry,
    )
    def predict(ctx: Context) -> Artifact:
        bias = ctx.input("model").value["bias"]
        return Artifact.json({"score": bias + float(ctx.param("features")["x"])})

    return registry


def _call(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Build the server, connect an in-memory client, and call one tool."""
    registry = _registry()
    server = build_server(registry, lambda: Runner(registry))

    async def run() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def _text(result: CallToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextContent))


def test_scalar_params_call() -> None:
    result = _call("run_filter_rows", {"zipcode": "98004", "limit": 5})
    assert result.is_error is False, _text(result)
    assert "98004" in _text(result)


def test_object_param_call() -> None:
    # The exact shape that crashed: a dict-valued param through the serve cache.
    result = _call("run_score_one", {"record": {"age": 41, "grade": 9}})
    assert result.is_error is False, _text(result)
    assert "age" in _text(result) and "grade" in _text(result)


def test_array_of_objects_param_call() -> None:
    result = _call(
        "run_score_batch",
        {"rows": [{"a": 1}, {"a": 1, "b": 2}], "ids": [3, 1, 2]},
    )
    assert result.is_error is False, _text(result)
    assert result.structured_content is not None
    assert result.structured_content["row_count"] == 2


def test_model_prediction_end_to_end() -> None:
    # Opaque model artifact + object param, both flowing through serving.
    result = _call("run_predict", {"features": {"x": 42}})
    assert result.is_error is False, _text(result)
    assert "142" in _text(result)  # 100.0 bias + 42


def test_repeated_object_call_is_consistent() -> None:
    # A second identical structured call hits the serve cache key; it must not
    # raise and must return the same answer.
    args = {"record": {"age": 41, "grade": 9}}
    first = _call("run_score_one", args)
    second = _call("run_score_one", args)
    assert first.is_error is False and second.is_error is False
    assert _text(first) == _text(second)


@pytest.mark.filterwarnings("ignore")
def test_model_lifecycle_over_mcp(tmp_path: Path) -> None:
    """Train, list, promote, and predict through the real MCP client path.

    The tools exist only when the optional ml stack is installed (the dev
    environment installs it); the flow proves an MCP agent can leave a registered,
    scoreable model behind and read the registry back.
    """
    import random

    pytest.importorskip("flaml")
    from elbi_core.data import Table

    rng = random.Random(4)
    rows = []
    for _ in range(160):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        label = "1" if x1 + 0.3 * x2 + rng.gauss(0, 0.3) > 0 else "0"
        rows.append({"x1": f"{x1:.4f}", "x2": f"{x2:.4f}", "y": label})
    table = Table(rows=rows)

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        enable_propose=True,
        load_dataset=lambda name: table,
        cache_dir=tmp_path,
    )

    def text(result: CallToolResult) -> str:
        content = result.content[0]
        assert isinstance(content, TextContent)
        return content.text

    async def run() -> None:
        async with Client(server) as client:
            trained = await client.call_tool(
                "train_model",
                {
                    "name": "churn",
                    "dataset": "d",
                    "target": "y",
                    "time_budget": 5,
                },
            )
            report = text(trained)
            assert "Trained and registered" in report
            assert "held-out" in report.lower()

            listed = text(await client.call_tool("list_models", {}))
            assert "churn" in listed and "v1" in listed

            scored = text(
                await client.call_tool(
                    "predict",
                    {"model": "churn", "rows": [{"x1": 3.0, "x2": 1.0}]},
                )
            )
            assert "churn v1 predictions" in scored

            promoted = text(
                await client.call_tool(
                    "promote_model", {"name": "churn", "version": 1, "alias": "prod"}
                )
            )
            assert "@prod" in promoted

            missing = text(
                await client.call_tool("predict", {"model": "nope", "rows": [{}]})
            )
            assert missing.startswith("error:")

    asyncio.run(run())


def test_operate_tools_drive_the_operations_backend() -> None:
    """The operate tools (materialize/run_workflow/run_notebook/backfill/asset_status)
    route through the injected Operations backend and report its result."""

    class _FakeOps:
        def materialize(
            self, *, selection: str, assets: Any, include_downstream: bool
        ) -> dict[str, Any]:
            return {
                "run_id": "r1",
                "ok": True,
                "materialized": ["a"],
                "skipped": [],
                "failed": [],
            }

        def run_workflow(self, name: str) -> dict[str, Any]:
            if name != "wf":
                raise KeyError(f"no workflow named {name!r}")
            return {
                "run_id": "r2",
                "ok": True,
                "steps": [{"id": "build", "state": "succeeded"}],
            }

        def run_notebook(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"ran": 3, "failed": [], "ok": True}

        def backfill(
            self, *, asset: str, param: str, values: list[str]
        ) -> dict[str, Any]:
            return {"run_id": "r3", "ok": True, "partitions": len(values), "failed": []}

        def asset_status(self) -> list[dict[str, Any]]:
            return [{"asset": "a", "status": "materialized", "verdict": "sound"}]

    registry = Registry()
    server = build_server(registry, lambda: Runner(registry), operations=_FakeOps())

    async def run() -> None:
        async with Client(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {
                "materialize",
                "run_workflow",
                "run_notebook",
                "backfill",
                "asset_status",
            } <= tools
            mat = _text(await client.call_tool("materialize", {"selection": "all"}))
            assert "run r1: ok" in mat and "materialized: ['a']" in mat
            wf = _text(await client.call_tool("run_workflow", {"name": "wf"}))
            assert "build=succeeded" in wf
            missing = _text(await client.call_tool("run_workflow", {"name": "nope"}))
            assert "Error" in missing
            nb = _text(await client.call_tool("run_notebook", {"name": "x"}))
            assert "ran 3 cell(s), ok" in nb
            bf = _text(
                await client.call_tool(
                    "backfill", {"asset": "a", "param": "d", "values": ["1", "2"]}
                )
            )
            assert "partitions: 2" in bf
            status = _text(await client.call_tool("asset_status", {}))
            assert "a: materialized [sound]" in status

    asyncio.run(run())


def test_a_blocking_operate_tool_leaves_the_server_answering() -> None:
    """A long tool must not stall every other caller on the same server.

    Tools are ``async def`` and the SDK awaits them on the event loop, so synchronous
    work inside one freezes the whole server until it returns. That is not merely
    unfair: on the 2026-07-28 transport the handler shares a task group with the SSE
    keepalive, so a stalled loop stops the ping, the proxy closes the stream, and the
    run is lost. Two concurrent calls tell the two implementations apart, because only
    an offloaded one lets them overlap.
    """
    import time

    held = 0.4

    class _SlowOps:
        def materialize(
            self, *, selection: str, assets: Any, include_downstream: bool
        ) -> dict[str, Any]:
            time.sleep(held)  # synchronous, exactly like the real backend
            return {"run_id": "r1", "ok": True}

        def run_workflow(self, name: str) -> dict[str, Any]:
            raise KeyError(name)

        def run_notebook(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
            raise KeyError(name)

        def backfill(
            self, *, asset: str, param: str, values: list[str]
        ) -> dict[str, Any]:
            raise KeyError(asset)

        def asset_status(self) -> list[dict[str, Any]]:
            return []

    registry = Registry()
    server = build_server(registry, lambda: Runner(registry), operations=_SlowOps())

    async def run() -> float:
        async with Client(server) as client:
            started = time.perf_counter()
            await asyncio.gather(
                client.call_tool("materialize", {}), client.call_tool("materialize", {})
            )
            return time.perf_counter() - started

    elapsed = asyncio.run(run())
    # Serialized they would cost 2 x held; overlapped, about one. The margin is wide so
    # this measures overlap rather than the machine.
    assert elapsed < held * 1.8, f"calls serialized on the event loop ({elapsed:.2f}s)"


def test_reading_an_unknown_derivation_resource_is_invalid_params() -> None:
    """A resource that does not exist is a bad argument, not a server fault.

    The revision settled this at invalid-params, replacing the separate
    resource-not-found code earlier revisions used. Asserted through a client because
    that is where the mapping happens: the in-process `read_resource` raises the SDK's
    own exception, and only dispatch turns it into the code a client reads.
    """
    import pytest
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS

    registry = _registry()
    server = build_server(registry, lambda: Runner(registry))

    async def run() -> None:
        async with Client(server) as client:
            with pytest.raises(MCPError) as raised:
                await client.read_resource("elbi://derivation/nope")
            assert raised.value.error.code == INVALID_PARAMS
            assert "nope" in str(raised.value.error.message) + str(
                raised.value.error.data
            )


# --- Narrowing over the wire (a row/column policy through the real client) ---


def _roster_project(tmp_path: Path) -> tuple[Registry, Callable[[], Runner]]:
    """A table derivation over a small CSV: the shape a policy can narrow."""
    from elbi_core.config import DataBindings

    (tmp_path / "people.csv").write_text(
        "region,ssn\nEMEA,111111111\nAPAC,222222222\n", encoding="utf-8"
    )
    registry = Registry()

    @derivation(
        inputs={"people": Dataset("people")},
        serve=serve.table(),
        registry=registry,
        name="roster",
    )
    def roster(ctx: Context) -> Artifact:
        return Artifact.table(list(ctx.input("people").rows))

    bindings = DataBindings(bindings={"people": "people.csv"})
    return registry, lambda: Runner(registry, bindings=bindings, base_dir=tmp_path)


class _AuthoredStub:
    """Supplies a stored attestation and certificate, as the authored store would."""

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def read(self, name: str) -> dict[str, Any] | None:
        return self._record if name == "roster" else None


def _signed_proof() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """An issuer, an attestation, and a genuinely signed certificate for `roster`.

    Signed for real because ``build_server`` verifies a stored certificate and silently
    drops one that fails: a hand-built envelope would make a withholding test pass
    because nothing was ever attached.
    """
    from elbi_core.certificate import (
        Certificate,
        CertificateIssuer,
        SigningKey,
    )

    issuer = CertificateIssuer(
        key=SigningKey(bytes(range(32))), issuer="test-certifier"
    )
    certificate = issuer.sign(
        Certificate(
            derivation="roster",
            question="who is in the roster?",
            verdict="supported",
            claim={"kind": "descriptive"},
            estimate=None,
            estimate_label=None,
            adjusted_for=(),
            checks=(),
            skipped=(),
            data_hash="sha256:feed",
            derivation_version="sha256:beef",
            code_version=None,
            input_versions={"people": "sha256:cafe"},
            issued_at="2026-01-01T00:00:00Z",
            issuer="test-certifier",
            public_key=issuer.public_key,
        )
    )
    return issuer, {"verdict": "supported"}, certificate
