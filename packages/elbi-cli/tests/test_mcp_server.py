"""Tests for building the local MCP server from a registry."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from elbi_cli.authored import AuthoredStore
from elbi_cli.mcp_server import (
    _CERTIFICATE_META_KEY,
    build_server,
    resource_uri,
    tool_name,
)
from elbi_core import (
    Artifact,
    Bm25Retriever,
    Context,
    Dataset,
    Registry,
    RoutingExecutor,
    Runner,
    SubprocessExecutor,
    certify,
    derivation,
    load_issuer,
    param,
    propose,
    serve,
    verify_certificate,
)
from elbi_core.config import ColumnSpec, DataBindings, DatasetSpec
from elbi_core.tracking import JsonlRunLog


def _tool_text(result: Any) -> str:
    """Extract text from an MCPServer call_tool result (CallToolResult, blocks)."""
    if hasattr(result, "content"):  # a CallToolResult
        result = result.content
    content = result[0] if isinstance(result, tuple) else result
    blocks = content if isinstance(content, (list, tuple)) else [content]
    return "".join(getattr(b, "text", "") for b in blocks)


def test_uri_and_tool_naming() -> None:
    assert resource_uri("churn_risk") == "elbi://derivation/churn_risk"
    assert tool_name("churn_risk") == "run_churn_risk"


def _server_with_one() -> MCPServer:
    # Pin lexical BM25 so search stays offline and deterministic (the default hybrid
    # retriever would load an embedding model); these tests exercise MCP wiring.
    registry = Registry(retriever=Bm25Retriever())

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:
        """A friendly greeting."""
        return Artifact.text("hi")

    return build_server(registry, lambda: Runner(registry))


def test_build_server_registers_tool_per_derivation() -> None:
    server = _server_with_one()
    tools = asyncio.run(server.list_tools())
    assert "run_hello" in {tool.name for tool in tools}


def test_build_server_registers_resource_per_derivation() -> None:
    server = _server_with_one()
    resources = asyncio.run(server.list_resources())
    assert "elbi://derivation/hello" in {str(r.uri) for r in resources}


def _server_with_params() -> MCPServer:
    registry = Registry()

    @derivation(
        name="query",
        inputs={"d": Dataset("d")},
        params={
            "zipcode": param.string(description="Zip to filter"),
            "max_price": param.integer(required=False, default=0),
        },
        serve=serve.table(),
        registry=registry,
    )
    def query(ctx: Context) -> Artifact:
        return Artifact.table([])

    return build_server(registry, lambda: Runner(registry))


def test_parameterized_tool_exposes_typed_input_schema() -> None:
    server = _server_with_params()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    schema = tools["run_query"].input_schema
    props = schema["properties"]
    assert props["zipcode"]["type"] == "string"
    assert "zipcode" in schema.get("required", [])
    # Optional param with a default is not required.
    assert "max_price" not in schema.get("required", [])


def test_build_server_sets_instructions() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: Runner(registry), instructions="be a careful analyst"
    )
    assert server.instructions == "be a careful analyst"


def test_structured_params_expose_nested_input_schema() -> None:
    registry = Registry()

    @derivation(
        name="score",
        params={
            "record": param.object(description="One feature record"),
            "batch": param.array(items="object"),
            "ids": param.array(items="string"),
        },
        serve=serve.table(),
        registry=registry,
    )
    def score(ctx: Context) -> Artifact:
        return Artifact.table([])

    server = build_server(registry, lambda: Runner(registry))
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    props = tools["run_score"].input_schema["properties"]
    assert props["record"]["type"] == "object"
    assert props["ids"]["type"] == "array"
    assert props["ids"]["items"]["type"] == "string"
    assert props["batch"]["type"] == "array"
    assert props["batch"]["items"]["type"] == "object"


def test_array_param_description_names_element_type() -> None:
    registry = Registry()

    @derivation(
        name="score",
        params={"batch": param.array(items="object", description="rows to score")},
        serve=serve.table(),
        registry=registry,
    )
    def score(ctx: Context) -> Artifact:
        return Artifact.table([])

    server = build_server(registry, lambda: Runner(registry))
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert "batch (array of object)" in tools["run_score"].description


def test_parameterized_derivation_has_no_resource() -> None:
    server = _server_with_params()
    resources = asyncio.run(server.list_resources())
    assert "elbi://derivation/query" not in {str(r.uri) for r in resources}


def test_internal_derivation_is_not_exposed() -> None:
    registry = Registry()

    @derivation(name="hidden", registry=registry)  # no serve → internal
    def hidden(ctx: Context) -> Artifact:
        return Artifact.text("secret")

    @derivation(
        name="shown", inputs={"h": hidden}, serve=serve.text(), registry=registry
    )
    def shown(ctx: Context) -> Artifact:
        return Artifact.text(ctx.input("h").value)

    server = build_server(registry, lambda: Runner(registry))
    tools = {t.name for t in asyncio.run(server.list_tools())}
    resources = {str(r.uri) for r in asyncio.run(server.list_resources())}
    assert "run_shown" in tools
    assert "run_hidden" not in tools  # internal derivation has no tool
    assert "elbi://derivation/hidden" not in resources


def test_parameterized_tool_description_lists_params() -> None:
    server = _server_with_params()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    desc = tools["run_query"].description
    assert "Parameters:" in desc
    assert "zipcode" in desc and "max_price" in desc


def test_call_parameterless_tool_returns_served_output() -> None:
    server = _server_with_one()
    out = asyncio.run(server.call_tool("run_hello", {}))
    assert _tool_text(out) == "hi"


def test_read_parameterless_resource() -> None:
    server = _server_with_one()
    contents = asyncio.run(server.read_resource("elbi://derivation/hello"))
    blocks = list(contents)
    assert blocks[0].content == "hi"


def test_call_parameterized_tool_filters_by_arg(tmp_path: Path) -> None:
    (tmp_path / "d.csv").write_text(
        "zipcode,price\n98001,100\n98002,200\n", encoding="utf-8"
    )
    registry = Registry()

    @derivation(
        name="query",
        inputs={"d": Dataset("d")},
        params={"zipcode": param.string()},
        serve=serve.table(columns=["zipcode", "price"]),
        registry=registry,
    )
    def query(ctx: Context) -> Artifact:
        z = ctx.param("zipcode")
        return Artifact.table([r for r in ctx.input("d").rows if r["zipcode"] == z])

    bindings = DataBindings(bindings={"d": "d.csv"})
    server = build_server(
        registry, lambda: Runner(registry, bindings=bindings, base_dir=tmp_path)
    )

    out = asyncio.run(server.call_tool("run_query", {"zipcode": "98001"}))
    text = _tool_text(out)
    assert "98001" in text
    assert "98002" not in text


def _routing_runner(registry: Registry) -> Runner:
    return Runner(registry, executor=RoutingExecutor(sandbox=SubprocessExecutor()))


def _house_table() -> Any:
    from elbi_core.data import Table

    return Table(
        rows=[
            {"id": "1", "price": "221900.0", "date": "20141013T000000"},
            {"id": "2", "price": "538000.0", "date": "20141209T000000"},
        ]
    )


def test_propose_gated_on_claim_verification(tmp_path: Path) -> None:
    # A derivation that declares a claim is served only if its conclusion verifies
    # sound: a confounded effect is held (not served); a sound one is certified,
    # served, and carries a verification attestation.
    import csv
    import random

    rng = random.Random(0)
    m = 600

    def g(scale: float = 1.0) -> list[float]:
        return [rng.gauss(0, scale) for _ in range(m)]

    def write_csv(name: str, cols: dict[str, list[float]]) -> None:
        names = list(cols)
        with (tmp_path / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(names)
            for i in range(m):
                w.writerow([cols[n][i] for n in names])

    z = g()
    x = [zi + ni for zi, ni in zip(z, g(0.5))]
    write_csv(
        "confounded.csv",
        {
            "x": x,
            "y": [-0.2 * xi + 1.5 * zi + ni for xi, zi, ni in zip(x, z, g(0.5))],
            "z": z,
        },
    )
    xs = g()
    write_csv("sound.csv", {"x": xs, "y": [2 * xi + ni for xi, ni in zip(xs, g())]})

    # the derivation outputs the analytic rows its claim concerns, so what is served
    # is exactly what the oracle verifies
    src = "def eff(ctx):\n    return list(ctx.input('d').rows)\n"
    args = {
        "name": "eff",
        "source": src,
        "inputs": ["d"],
        "claim": {"x": "x", "y": "y"},
    }

    def server_for(filename: str) -> MCPServer:
        registry = Registry()
        bindings = DataBindings(bindings={"d": filename})
        return build_server(
            registry,
            lambda: Runner(
                registry,
                executor=RoutingExecutor(sandbox=SubprocessExecutor()),
                bindings=bindings,
                base_dir=tmp_path,
            ),
            load_dataset=lambda name: bindings.load_dataset(name, tmp_path),
            enable_propose=True,
        )

    confounded = server_for("confounded.csv")
    out = _tool_text(asyncio.run(confounded.call_tool("propose_derivation", args)))
    assert "did not pass verification" in out and "held" in out
    assert "run_eff" not in {t.name for t in asyncio.run(confounded.list_tools())}

    sound = server_for("sound.csv")
    out = _tool_text(asyncio.run(sound.call_tool("propose_derivation", args)))
    assert "certified" in out and "Verified sound" in out
    assert "run_eff" in {t.name for t in asyncio.run(sound.list_tools())}

    # The served result carries a provider-agnostic verification attestation.
    result = asyncio.run(sound.call_tool("run_eff", {}))
    attestation = result.structured_content["verification"]
    assert attestation["verdict"] == "sound"
    assert attestation["claim"] == {"x": "x", "y": "y"}
    assert any(c["name"] == "effect" for c in attestation["checks"])
    assert len(attestation["data_hash"]) == 64  # bare sha256 hex


def _sound_effect_csv(path: Path) -> None:
    import csv
    import random

    rng = random.Random(0)
    m = 400
    xs = [rng.gauss(0, 1) for _ in range(m)]
    ys = [2 * xi + rng.gauss(0, 1) for xi in xs]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for xi, yi in zip(xs, ys):
            writer.writerow([xi, yi])


def _propose_certified_effect(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Stand up a propose-enabled server with an issuer, and certify one effect."""
    _sound_effect_csv(tmp_path / "sound.csv")
    bindings = DataBindings(bindings={"d": "sound.csv"})
    registry = Registry(retriever=Bm25Retriever())
    store = AuthoredStore(tmp_path / ".elbi" / "authored")
    issuer = load_issuer(tmp_path / ".elbi", issuer="tester@example.com")
    server = build_server(
        registry,
        lambda: Runner(
            registry,
            executor=RoutingExecutor(sandbox=SubprocessExecutor()),
            bindings=bindings,
            base_dir=tmp_path,
        ),
        load_dataset=lambda name: bindings.load_dataset(name, tmp_path),
        enable_propose=True,
        authored_store=store,
        run_log=JsonlRunLog(tmp_path / ".elbi" / "runs.jsonl"),
        issuer=issuer,
    )
    args = {
        "name": "eff",
        "source": "def eff(ctx):\n    return list(ctx.input('d').rows)\n",
        "inputs": ["d"],
        "claim": {"x": "x", "y": "y"},
    }
    _tool_text(asyncio.run(server.call_tool("propose_derivation", args)))
    return server, store, registry, bindings


def test_certified_serve_carries_signed_certificate(tmp_path: Path) -> None:
    server, _store, _registry, _bindings = _propose_certified_effect(tmp_path)

    # The tool result carries the signed certificate beside the verification record.
    result = asyncio.run(server.call_tool("run_eff", {}))
    certificate = result.structured_content["certificate"]
    assert result.structured_content["verification"]["verdict"] == "sound"
    cert = verify_certificate(certificate)
    assert cert.derivation == "eff"
    assert cert.verdict == "sound"
    assert cert.issuer == "tester@example.com"

    # The resource read carries the same certificate in its `_meta`, so every serve is
    # accompanied by the proof.
    resources = {r.name: r for r in asyncio.run(server.list_resources())}
    assert resources["eff"].meta[_CERTIFICATE_META_KEY] == certificate


def test_certificate_survives_restart_verbatim(tmp_path: Path) -> None:
    _propose_certified_effect(tmp_path)
    before = AuthoredStore(tmp_path / ".elbi" / "authored").read("eff")
    assert before is not None and "certificate" in before

    # Rebuild the server from a fresh registry, reloading the authored derivation from
    # the sidecar: the persisted certificate is re-served without re-signing.
    fresh = Registry(retriever=Bm25Retriever())
    store = AuthoredStore(tmp_path / ".elbi" / "authored")
    store.load_into(fresh)
    bindings = DataBindings(bindings={"d": "sound.csv"})
    server = build_server(
        fresh,
        lambda: Runner(fresh, bindings=bindings, base_dir=tmp_path),
        authored_store=store,
        issuer=load_issuer(tmp_path / ".elbi", issuer="tester@example.com"),
    )
    resources = {r.name: r for r in asyncio.run(server.list_resources())}
    assert resources["eff"].meta[_CERTIFICATE_META_KEY] == before["certificate"]
    verify_certificate(before["certificate"])  # still valid


def test_reload_drops_a_tampered_stored_certificate(tmp_path: Path) -> None:
    _server, store, registry, _bindings = _propose_certified_effect(tmp_path)
    before = store.read("eff")
    assert before is not None and "certificate" in before

    # Tamper the stored certificate's payload without re-signing: the same shape as
    # an attacker (or disk corruption) editing the sidecar YAML directly, rather than
    # a malformed envelope a parser would already reject.
    tampered = copy.deepcopy(before["certificate"])
    tampered["certificate"]["verdict"] = "unsound"
    store.save(
        registry.get("eff"),
        attestation=before.get("attestation"),
        certificate=tampered,
    )

    fresh = Registry(retriever=Bm25Retriever())
    reload_store = AuthoredStore(tmp_path / ".elbi" / "authored")
    reload_store.load_into(fresh)
    bindings = DataBindings(bindings={"d": "sound.csv"})
    reloaded = build_server(
        fresh,
        lambda: Runner(fresh, bindings=bindings, base_dir=tmp_path),
        authored_store=reload_store,
        issuer=load_issuer(tmp_path / ".elbi", issuer="tester@example.com"),
    )
    resources = {r.name: r for r in asyncio.run(reloaded.list_resources())}
    assert resources["eff"].meta is None


def test_verify_analysis_through_mcp() -> None:
    # Validate the verification oracle through the real MCP tool path: a sound
    # effect passes, a confounded one is caught and the confounder is named.
    import random

    from elbi_core.data import Table

    rng = random.Random(0)
    n = 600

    def gauss(scale: float = 1.0) -> list[float]:
        return [rng.gauss(0, scale) for _ in range(n)]

    def table(**cols: list[float]) -> Table:
        names = list(cols)
        return Table(rows=[{c: str(cols[c][i]) for c in names} for i in range(n)])

    x = gauss()
    sound = table(x=x, y=[2 * xi + ni for xi, ni in zip(x, gauss())])
    z = gauss()
    xc = [zi + ni for zi, ni in zip(z, gauss(0.5))]
    confounded = table(
        x=xc,
        y=[-0.2 * xi + 1.5 * zi + ni for xi, zi, ni in zip(xc, z, gauss(0.5))],
        z=z,
    )
    tables = {"sound": sound, "confounded": confounded}

    registry = Registry()
    server = build_server(
        registry, lambda: Runner(registry), load_dataset=lambda name: tables[name]
    )

    out_sound = _tool_text(
        asyncio.run(
            server.call_tool(
                "verify_analysis", {"dataset": "sound", "x": "x", "y": "y"}
            )
        )
    )
    assert "SOUND" in out_sound

    out_conf = _tool_text(
        asyncio.run(
            server.call_tool(
                "verify_analysis", {"dataset": "confounded", "x": "x", "y": "y"}
            )
        )
    )
    assert "UNSOUND" in out_conf
    assert "'z'" in out_conf  # the confounder is named


def test_describe_dataset_lists_and_describes() -> None:
    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        load_dataset=lambda name: _house_table(),
        dataset_specs=(DatasetSpec(name="house_data"),),
    )
    listing = _tool_text(asyncio.run(server.call_tool("describe_dataset", {})))
    assert "house_data" in listing

    schema = _tool_text(
        asyncio.run(server.call_tool("describe_dataset", {"name": "house_data"}))
    )
    assert "2 rows" in schema
    assert "| id | 1 | integer |" in schema
    assert "| price | 221900.0 | number |" in schema
    assert "| date | 20141013T000000 | string |" in schema


def test_describe_dataset_surfaces_semantic_model() -> None:
    # Author-declared column meaning (description, role, unit, synonyms) shows up,
    # so the agent learns what a column means, not just its type.
    registry = Registry()
    spec = DatasetSpec(
        name="house_data",
        description="One row per sale.",
        columns=(
            ColumnSpec(
                name="price",
                description="Sale price.",
                role="measure",
                unit="USD",
                synonyms=("amount",),
            ),
        ),
    )
    server = build_server(
        registry,
        lambda: Runner(registry),
        load_dataset=lambda name: _house_table(),
        dataset_specs=(spec,),
    )
    out = _tool_text(
        asyncio.run(server.call_tool("describe_dataset", {"name": "house_data"}))
    )
    assert "One row per sale." in out  # dataset description
    assert "Sale price. (measure, USD) aka amount" in out  # column meaning cell


def test_describe_dataset_infers_empty_columns() -> None:
    from elbi_core.data import Table

    table = Table(
        rows=[
            {"id": "1", "price": "221900.0", "note": "", "blank": ""},
            {"id": "2", "price": "", "note": "hi", "blank": ""},
        ]
    )
    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        load_dataset=lambda name: table,
        dataset_specs=(DatasetSpec(name="d"),),
    )
    schema = _tool_text(
        asyncio.run(server.call_tool("describe_dataset", {"name": "d"}))
    )
    assert "| price | 221900.0 | number |  |" in schema  # blank cell skipped
    assert "| note |  | string |  |" in schema
    assert "| blank |  | empty |  |" in schema  # all values empty


def test_describe_dataset_refuses_a_dataset_it_cannot_load() -> None:
    # Naming a dataset the project cannot load is a bad argument, so it leaves as an
    # invalid-params error rather than as a successful result whose text happens to
    # describe a failure. The message still says what went wrong and what could have
    # been named instead, so a model can correct itself without another round trip.
    import pytest
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS

    from elbi_core.errors import DataBindingError

    def boom(name: str) -> Any:
        raise DataBindingError("unbound")

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        load_dataset=boom,
        dataset_specs=(DatasetSpec(name="d"),),
    )
    with pytest.raises(MCPError) as raised:
        asyncio.run(server.call_tool("describe_dataset", {"name": "d"}))

    assert raised.value.error.code == INVALID_PARAMS
    message = raised.value.error.message
    assert "Could not load dataset 'd'" in message
    assert "unbound" in message
    assert "Available datasets: d" in message


def test_train_model_budget_is_bounded_at_both_ends(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """A search budget arriving over the wire cannot exceed the cap or escape it.

    The ceiling alone is not a bound: FLAML reads a negative budget as "no search
    budget" and trains without a time limit, so ``min(time_budget, cap)`` hands an
    unbounded run straight through. Both ends are asserted against what the trainer
    was actually given, since that is the value that decides how long a worker thread
    is held.
    """
    import pytest
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS

    pytest.importorskip("flaml")
    import elbi_core.ml as ml_module
    from elbi_core.data import Table

    budgets: list[float] = []

    class _Report:
        def render(self) -> str:
            return "Trained and registered."

    def spy(rows: Any, **kwargs: Any) -> _Report:
        budgets.append(kwargs["time_budget"])
        return _Report()

    # Patched before the build: `_register_ml` imports the trainer into its closure
    # when the server is constructed, so a later patch would never be seen.
    monkeypatch.setattr(ml_module, "train_automl", spy)

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        enable_propose=True,  # the model tools live on the authoring surface
        load_dataset=lambda name: Table(rows=[{"x": 1, "y": 2}]),
        cache_dir=tmp_path,
    )

    asyncio.run(
        server.call_tool(
            "train_model",
            {"name": "m", "dataset": "d", "target": "y", "time_budget": 1000.0},
        )
    )
    assert budgets == [240.0], "an over-long budget was not capped"

    with pytest.raises(MCPError) as raised:
        asyncio.run(
            server.call_tool(
                "train_model",
                {"name": "m", "dataset": "d", "target": "y", "time_budget": -1.0},
            )
        )
    assert raised.value.error.code == INVALID_PARAMS
    assert "positive" in raised.value.error.message
    # The refusal happened before any training, so the trainer never saw the budget.
    assert budgets == [240.0], "a negative budget reached the trainer"


def test_train_model_refuses_a_dataset_it_cannot_load(tmp_path: Path) -> None:
    # Same reasoning as describe_dataset above: a dataset the project does not have is
    # a bad argument, not a training outcome. A training run that genuinely fails stays
    # an ordinary result, so only the load is refused.
    import pytest
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS

    pytest.importorskip("flaml")
    from elbi_core.errors import DataBindingError

    def boom(name: str) -> Any:
        raise DataBindingError("unbound")

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        enable_propose=True,
        load_dataset=boom,
        cache_dir=tmp_path,
    )
    with pytest.raises(MCPError) as raised:
        asyncio.run(
            server.call_tool(
                "train_model", {"name": "m", "dataset": "nope", "target": "y"}
            )
        )

    assert raised.value.error.code == INVALID_PARAMS
    assert "Could not load dataset 'nope'" in raised.value.error.message
    assert "unbound" in raised.value.error.message


def test_sandbox_environment_tool() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = _tool_text(asyncio.run(server.call_tool("sandbox_environment", {})))
    assert "Python" in out
    assert "standard library" in out


def test_structure_map_surfaces_relationships() -> None:
    # The tool returns the relational skeleton: a derived column and the dependency
    # graph, computed from the data, for the agent to ground on.
    from elbi_core.data import Table

    table = Table(
        rows=[
            {"total": "3", "a": "1", "b": "2"},
            {"total": "7", "a": "3", "b": "4"},
            {"total": "12", "a": "5", "b": "7"},
        ]
    )
    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        load_dataset=lambda name: table,
        dataset_specs=(DatasetSpec(name="d"),),
        enable_propose=True,
    )
    out = _tool_text(asyncio.run(server.call_tool("structure_map", {"name": "d"})))
    assert "# Data structure" in out
    assert "total = a + b" in out  # exact derived column detected


def test_run_code_explores_bound_data() -> None:
    # The exploration loop: the declared datasets are seeded into the session as `data`,
    # and code returns stdout + result.
    registry = Registry()
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        load_dataset=lambda name: _house_table(),
        dataset_specs=(DatasetSpec(name="house"),),
        enable_propose=True,
    )
    out = _tool_text(
        asyncio.run(
            server.call_tool(
                "run_code",
                {
                    "code": "prices = [float(r['price']) for r in data['house']]\n"
                    "print('n', len(prices))\n"
                    "result = max(prices)",
                },
            )
        )
    )
    assert "n 2" in out
    assert "538000" in out  # max of the two bound prices, returned as `result`


def test_run_code_session_persists_variables() -> None:
    # The MCP explore loop is stateful too: a variable from one call is there in the
    # next, matching the app.
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    asyncio.run(server.call_tool("run_code", {"code": "kept = 21"}))
    out = _tool_text(
        asyncio.run(server.call_tool("run_code", {"code": "result = kept * 2"}))
    )
    assert "42" in out


def test_bash_declines_on_the_subprocess_backend() -> None:
    # bash is offered, but on the default subprocess backend it declines (a host shell
    # would run on the user's machine); it needs the docker backend.
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert "bash" in tools
    out = _tool_text(asyncio.run(server.call_tool("bash", {"command": "echo hi"})))
    assert "docker" in out


def test_run_code_returns_traceback_on_error() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = _tool_text(
        asyncio.run(server.call_tool("run_code", {"code": "result = undefined + 1"}))
    )
    assert "NameError" in out  # the traceback comes back for the agent to iterate on


def test_run_code_background_job_and_status() -> None:
    # A long run_code can be launched as a background job on the MCP surface too; its
    # state and output are read back with job_status.
    import re
    import time

    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert {"job_status", "cancel_job"} <= tools
    launched = _tool_text(
        asyncio.run(
            server.call_tool(
                "run_code", {"code": "print('bg mcp')", "background": True}
            )
        )
    )
    assert "Launched background run_code job" in launched
    match = re.search(r"job_[0-9a-f]+", launched)
    assert match is not None
    job_id = match.group(0)
    status = ""
    for _ in range(200):
        status = _tool_text(
            asyncio.run(server.call_tool("job_status", {"job_id": job_id}))
        )
        if any(s in status for s in ("succeeded", "failed", "cancelled")):
            break
        time.sleep(0.02)
    assert "succeeded" in status
    assert "bg mcp" in status  # the job's stdout is reported


def test_job_status_and_cancel_unknown_id() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    assert "no job" in _tool_text(
        asyncio.run(server.call_tool("job_status", {"job_id": "nope"}))
    )
    assert "no job" in _tool_text(
        asyncio.run(server.call_tool("cancel_job", {"job_id": "nope"}))
    )


def test_run_code_workspace_persists_across_calls(tmp_path: Path) -> None:
    # With a scratch dir, the MCP run_code tool carries files across calls, so an agent
    # can save a result and read it back. A derivation still runs hermetically.
    registry = Registry()
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        scratch_dir=tmp_path,
    )
    asyncio.run(server.call_tool("run_code", {"code": "open('m.txt','w').write('K9')"}))
    out = _tool_text(
        asyncio.run(
            server.call_tool("run_code", {"code": "result = open('m.txt').read()"})
        )
    )
    assert "K9" in out  # the second call read what the first wrote


def test_build_server_advertises_tools_list_changed() -> None:
    # The tool list is dynamic (agents author/delete derivations), so the server must
    # advertise `tools.listChanged` on both eras. They derive it differently, so both
    # are asserted: the handshake era builds options with no args (our patch is the only
    # lever), while 2026-07-28 derives it from serving `subscriptions/listen`.
    server = build_server(
        Registry(), lambda: _routing_runner(Registry()), enable_propose=True
    )
    low = server._lowlevel_server

    handshake = low.create_initialization_options().capabilities
    assert handshake.tools is not None
    assert handshake.tools.list_changed is True

    modern = low.get_capabilities(protocol_version="2026-07-28")
    assert modern.tools is not None
    assert modern.tools.list_changed is True


def test_propose_response_is_self_sufficient() -> None:
    # The result must come back inline so the agent answers directly without
    # depending on the just-authored run tool appearing in its tool list.
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = _tool_text(
        asyncio.run(
            server.call_tool(
                "propose_derivation",
                {
                    "name": "smoke",
                    "source": "def smoke(ctx):\n    return [{'v': 42}]\n",
                },
            )
        )
    )
    assert "Result:" in out
    assert "report it directly" in out
    assert "42" in out  # the computed value is present in the response


def test_delete_derivation_removes_agent_tool() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "probe", "source": "def probe(ctx):\n    return [{'x': 1}]\n"},
        )
    )
    assert "run_probe" in {t.name for t in asyncio.run(server.list_tools())}

    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "probe"}))
    )
    assert "Deleted 'probe'" in out
    assert "probe" not in registry
    assert "run_probe" not in {t.name for t in asyncio.run(server.list_tools())}


def test_delete_derivation_removes_unserved_proposal() -> None:
    from elbi_core import ManualCertification

    registry = Registry()
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        certification=ManualCertification(),
    )
    asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "held", "source": "def held(ctx):\n    return [{'x': 1}]\n"},
        )
    )
    # Held (not served), so there is no run tool to unregister; delete still works.
    assert "run_held" not in {t.name for t in asyncio.run(server.list_tools())}
    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "held"}))
    )
    assert "Deleted 'held'" in out
    assert "held" not in registry


def test_delete_derivation_protects_human_authored() -> None:
    registry = Registry()

    @derivation(name="kept", serve=serve.text(), registry=registry)
    def kept(ctx: Context) -> Artifact:
        return Artifact.text("x")

    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "kept"}))
    )
    assert "Refusing to delete" in out
    assert "kept" in registry


def test_delete_derivation_unknown_name() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "ghost"}))
    )
    assert "Cannot delete 'ghost'" in out


def test_delete_derivation_trashes_when_on_delete_is_wired(tmp_path: Path) -> None:
    from elbi_cli.authored import AuthoredStore

    registry = Registry()
    authored_store = AuthoredStore(tmp_path / "authored")
    trashed_names: list[str] = []

    def on_delete(name: str) -> bool:
        # The served app's hook returns whether the database stamp took; the
        # trash path (sidecar moved aside, restorable) only happens on True.
        trashed_names.append(name)
        return True

    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        authored_store=authored_store,
        on_delete=on_delete,
    )
    asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "probe", "source": "def probe(ctx):\n    return [{'x': 1}]\n"},
        )
    )
    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "probe"}))
    )
    assert "Moved 'probe' to trash" in out
    assert trashed_names == ["probe"]
    # Stops running either way: gone from the registry and its tool unserved.
    assert "probe" not in registry
    assert "run_probe" not in {t.name for t in asyncio.run(server.list_tools())}
    # Trashed, not erased: the sidecar moved aside rather than being removed.
    assert authored_store.read("probe") is None
    assert authored_store.restore("probe") is True


def test_delete_falls_back_to_permanent_when_the_trash_stamp_does_not_take(
    tmp_path: Path,
) -> None:
    """A falsy on_delete means no governed row was stamped -- say so, honestly.

    Reproduces the verified live gap (IP-27 Finding 2): the tool replied
    "Moved to trash ... Restore it to bring it back" while no trash listing
    held the name and restore 404'd. The truthful outcome for a row-less
    derivation is the permanent path: "Deleted" wording, sidecar removed
    outright (not moved aside), nothing left to restore.
    """
    from elbi_cli.authored import AuthoredStore

    registry = Registry()
    authored_store = AuthoredStore(tmp_path / "authored")
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        authored_store=authored_store,
        on_delete=lambda name: False,  # a held proposal / legacy probe: no row
    )
    asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "probe", "source": "def probe(ctx):\n    return [{'x': 1}]\n"},
        )
    )
    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "probe"}))
    )
    assert "Deleted 'probe'" in out
    assert "trash" not in out.lower()
    assert "probe" not in registry
    # Removed outright: no live sidecar and no trashed one to bring back.
    assert authored_store.read("probe") is None
    assert authored_store.restore("probe") is False


def test_delete_survives_a_hook_that_does_its_own_teardown(tmp_path: Path) -> None:
    """The served app's real hook unregisters the derivation itself.

    Source for this stub's behavior: ``serve.py``'s ``derivation_on_trash``,
    which calls ``_stop_derivation_runtime`` (registry.remove + _unregister)
    and moves the sidecar before returning True. A delete path that assumes it
    removes the entry first raises on the second removal -- caught live on
    2026-08-21, where the trash stamp landed but the tool answered with an
    error instead of the trash message.
    """
    from elbi_cli.authored import AuthoredStore

    registry = Registry()
    authored_store = AuthoredStore(tmp_path / "authored")

    def on_delete(name: str) -> bool:
        registry.remove(name)  # what the real hook does, before we get here
        authored_store.trash(name)
        return True

    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        authored_store=authored_store,
        on_delete=on_delete,
    )
    asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "probe", "source": "def probe(ctx):\n    return [{'x': 1}]\n"},
        )
    )
    out = _tool_text(
        asyncio.run(server.call_tool("delete_derivation", {"name": "probe"}))
    )
    assert "Moved 'probe' to trash" in out
    assert "probe" not in registry
    assert "run_probe" not in {t.name for t in asyncio.run(server.list_tools())}
    # Still trashed rather than erased, so a restore stays possible.
    assert authored_store.restore("probe") is True


def test_propose_reports_a_certified_derivation_to_on_author(
    tmp_path: Path,
) -> None:
    """The served app's on_author hook receives the fields the store row needs.

    The hook's record is what serve.py persists as the governed DB row (the
    fix for IP-27 Finding 2's root cause: MCP-authored derivations previously
    existed only in the live registry and sidecar, so /api/derivations and
    trash never saw them).
    """
    from elbi_cli.authored import AuthoredStore

    registry = Registry()
    authored = []
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        authored_store=AuthoredStore(tmp_path / "authored"),
        on_author=lambda name, record: authored.append((name, record)),
    )
    source = "def probe(ctx):\n    return [{'x': 1}]\n"
    asyncio.run(
        server.call_tool("propose_derivation", {"name": "probe", "source": source})
    )
    assert [name for name, _ in authored] == ["probe"]
    record = authored[0][1]
    assert record["source"] == source
    assert record["format"] == "table"
    assert record["question"] == "Proposed via MCP"
    # No claim was declared, so there is no oracle verdict to carry.
    assert record["verdict"] is None
    assert record["claim"] is None


def test_proposed_derivation_is_not_served() -> None:
    registry = Registry()
    propose(
        "draft",
        "def draft(ctx):\n    return [{'a': 1}]\n",
        serve=serve.table(),
        registry=registry,
    )
    server = build_server(registry, lambda: Runner(registry))
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert "run_draft" not in tools  # proposed → never served


def test_certified_agent_derivation_is_served() -> None:
    registry = Registry()
    propose(
        "draft",
        "def draft(ctx):\n    return [{'a': 1}]\n",
        serve=serve.table(),
        registry=registry,
    )
    certify(registry.get("draft"), registry=registry)
    server = build_server(registry, lambda: _routing_runner(registry))
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert "run_draft" in tools
    out = asyncio.run(server.call_tool("run_draft", {}))
    assert "| a |" in _tool_text(out)


def test_table_tool_emits_preview_structured_and_resource_link() -> None:
    registry = Registry()

    @derivation(name="rows", serve=serve.table(max_cells=2), registry=registry)
    def rows(ctx: Context) -> Artifact:
        return Artifact.table([{"n": i} for i in range(10)])

    server = build_server(registry, lambda: Runner(registry))
    result = asyncio.run(server.call_tool("run_rows", {}))

    # structuredContent carries the full result; the inline text is a preview.
    assert result.structured_content == {
        "rows": [{"n": i} for i in range(10)],
        "row_count": 10,
    }
    assert "Preview: first 2 of 10 rows" in _tool_text(result)
    links = [b for b in result.content if getattr(b, "type", "") == "resource_link"]
    assert len(links) == 1
    assert str(links[0].uri) == resource_uri("rows")


def test_text_tool_has_no_structured_content_or_link() -> None:
    server = _server_with_one()  # the "hello" text derivation
    result = asyncio.run(server.call_tool("run_hello", {}))
    assert result.structured_content is None
    assert all(getattr(b, "type", "") != "resource_link" for b in result.content)
    assert _tool_text(result) == "hi"


def test_search_derivations_tool_returns_matches() -> None:
    server = _server_with_one()
    out = asyncio.run(server.call_tool("search_derivations", {"query": "greeting"}))
    text = _tool_text(out)
    assert "hello" in text
    miss = asyncio.run(server.call_tool("search_derivations", {"query": "zzz"}))
    assert "No derivations match" in _tool_text(miss)


def _server_with_components() -> MCPServer:
    registry = Registry(retriever=Bm25Retriever())

    @derivation(name="facts", serve=serve.components(), registry=registry)
    def facts(ctx: Context) -> Artifact:
        return Artifact.components(
            [
                {
                    "id": "test/discount_threshold",
                    "type": "threshold_rule",
                    "scope": {"dataset": "customers"},
                    "statement": "Discounts above 20% correlate with higher churn.",
                },
                {
                    "id": "test/tenure_shape",
                    "type": "model_component",
                    "scope": {"dataset": "customers"},
                    "statement": "Tenure has a protective, nonlinear effect on churn.",
                },
            ]
        )

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:  # a non-components derivation, must be ignored
        return Artifact.text("hi")

    return build_server(registry, lambda: Runner(registry))


def test_search_components_tool_returns_matches_with_provenance() -> None:
    server = _server_with_components()
    out = asyncio.run(
        server.call_tool("search_components", {"query": "discount churn"})
    )
    text = _tool_text(out)
    assert "Discounts above 20%" in text
    matches = out.structured_content["components"]
    assert matches[0]["provenance"]["derivation"] == "facts"
    assert matches[0]["provenance"]["derivation_version"]


def test_search_components_tool_ignores_non_components_derivations() -> None:
    server = _server_with_components()
    out = asyncio.run(server.call_tool("search_components", {"query": "hi hello"}))
    assert out.structured_content["components"] == []
    assert "No components match" in _tool_text(out)


def test_search_components_tool_registered_on_every_server() -> None:
    server = _server_with_one()  # no components-format derivation at all
    tools = asyncio.run(server.list_tools())
    assert "search_components" in {tool.name for tool in tools}
    out = asyncio.run(server.call_tool("search_components", {"query": "anything"}))
    assert out.structured_content["components"] == []


def test_propose_persists_to_store_and_delete_removes_it(tmp_path: Path) -> None:
    registry = Registry()
    store = AuthoredStore(tmp_path / "authored")
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        authored_store=store,
        cache_dir=tmp_path / "cache",  # delete also prunes the derivation's cache
    )
    src = 'def revenue(ctx):\n    "Total."\n    return [{"v": 1}]\n'
    asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "revenue", "source": src, "format": "table"},
        )
    )
    # Persisted to the sidecar, reloadable into a fresh registry across a restart.
    assert store.load_into(Registry()) == ("revenue",)

    asyncio.run(server.call_tool("delete_derivation", {"name": "revenue"}))
    assert store.load_into(Registry()) == ()  # delete drops the persisted record too


def test_propose_tool_present_only_when_enabled() -> None:
    server = _server_with_one()  # enable_propose defaults to False
    assert "propose_derivation" not in {
        t.name for t in asyncio.run(server.list_tools())
    }

    registry = Registry()
    enabled = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    assert "propose_derivation" in {t.name for t in asyncio.run(enabled.list_tools())}


def test_propose_derivation_tool_auto_certifies_and_serves() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = asyncio.run(
        server.call_tool(
            "propose_derivation",
            {
                "name": "revenue",
                "source": "def revenue(ctx):\n    return [{'total': 42}]\n",
                "format": "table",
            },
        )
    )
    text = _tool_text(out)
    assert "Authored and certified 'revenue'" in text
    # Default policy auto-certifies a clean verify → served live in this session.
    assert "run_revenue" in {t.name for t in asyncio.run(server.list_tools())}
    served = asyncio.run(server.call_tool("run_revenue", {}))
    assert "| total |" in _tool_text(served)


def test_propose_derivation_tool_manual_policy_holds() -> None:
    from elbi_core import ManualCertification

    registry = Registry()
    server = build_server(
        registry,
        lambda: _routing_runner(registry),
        enable_propose=True,
        certification=ManualCertification(),
    )
    out = asyncio.run(
        server.call_tool(
            "propose_derivation",
            {
                "name": "revenue",
                "source": "def revenue(ctx):\n    return [{'total': 42}]\n",
                "format": "table",
            },
        )
    )
    text = _tool_text(out)
    assert "held for review" in text
    assert "elbi certify revenue" in text
    # Held: not served.
    assert "run_revenue" not in {t.name for t in asyncio.run(server.list_tools())}


def test_propose_derivation_tool_reports_sandbox_failure() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = asyncio.run(
        server.call_tool(
            "propose_derivation",
            {
                "name": "kaboom",
                "source": "def kaboom(ctx):\n    raise ValueError('x')\n",
                "format": "text",
            },
        )
    )
    assert "failed to run in the sandbox" in _tool_text(out)


def test_propose_derivation_tool_rejects_bad_proposal() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = asyncio.run(
        server.call_tool(
            "propose_derivation",
            {"name": "mismatch", "source": "def other(ctx):\n    return 1\n"},
        )
    )
    assert "rejected" in _tool_text(out)


def test_propose_derivation_tool_rejects_unknown_format() -> None:
    registry = Registry()
    server = build_server(
        registry, lambda: _routing_runner(registry), enable_propose=True
    )
    out = asyncio.run(
        server.call_tool(
            "propose_derivation",
            {
                "name": "x",
                "source": "def x(ctx):\n    return 1\n",
                "format": "csv",
            },
        )
    )
    assert "unknown serve format" in _tool_text(out)


def test_every_tool_declares_what_it_does_to_the_world() -> None:
    """The protocol's hints, over the whole surface.

    An unannotated tool is not neutral: a client must assume it is destructive and
    non-idempotent, which puts a gate that only computes a statistic behind the same
    confirmation as a delete. Codex CLI drives its approval prompts from these, so the
    ones that read must say so.
    """
    from elbi_core.data import Table

    registry = Registry()

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:
        return Artifact.text("hi")

    server = build_server(
        registry,
        lambda: Runner(registry),
        enable_propose=True,
        load_dataset=lambda name: Table(rows=[{"x": 1.0, "y": 2.0}]),
        dataset_specs=(DatasetSpec(name="d"),),
    )
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert tools, "no tools registered"
    unannotated = [name for name, t in tools.items() if t.annotations is None]
    assert unannotated == [], f"tools with no behaviour hints: {unannotated}"

    # A derivation's own tool serves a certified result; the oracle's gates compute one.
    assert tools["run_hello"].annotations.read_only_hint is True
    assert tools["verify_regression"].annotations.read_only_hint is True
    # ...and the ones that change things say so, deletion loudest.
    assert tools["propose_derivation"].annotations.read_only_hint is False
    assert tools["delete_derivation"].annotations.destructive_hint is True
    assert tools["run_code"].annotations.read_only_hint is False


def test_every_tool_schema_is_valid_json_schema_2020_12() -> None:
    """The draft the 2026-07-28 revision requires, over the whole tool surface.

    The draft is asserted rather than inferred: the SDK emits no ``$schema`` key, so a
    dialect-detecting validator would fall back to its own default and agree with
    whatever it was handed. Naming the metaschema is what makes this a conformance
    check.

    A ``run_<name>`` tool for a parameterized derivation is the one that matters most.
    Its schema is written nowhere: it is synthesized from a fabricated ``__signature__``
    (see ``_build_param_tool``), so it is where an invalid document would come from.
    """
    from jsonschema import Draft202012Validator

    from elbi_core.data import Table

    registry = Registry()

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:
        return Artifact.text("hi")

    @derivation(
        name="query",
        params={
            "zipcode": param.string(description="Zip to filter"),
            "record": param.object(description="One feature record"),
            "ids": param.array(items="string", required=False),
        },
        serve=serve.table(),
        registry=registry,
    )
    def query(ctx: Context) -> Artifact:
        return Artifact.table([])

    server = build_server(
        registry,
        lambda: Runner(registry),
        enable_propose=True,
        load_dataset=lambda name: Table(rows=[{"x": 1.0, "y": 2.0}]),
        dataset_specs=(DatasetSpec(name="d"),),
    )
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    # Guard against passing vacuously: both schema-generation paths must be present,
    # the synthesized signature and an ordinary one carrying an optional list.
    assert "run_query" in tools, "no parameterized tool registered"
    assert "verify_regression" in tools

    for tool in tools.values():
        Draft202012Validator.check_schema(tool.input_schema)
        # Only a tool with a structured return declares one.
        if tool.output_schema is not None:
            Draft202012Validator.check_schema(tool.output_schema)


def test_the_warehouse_tools_answer_only_from_the_callers_catalog() -> None:
    """A caller reaches the tables the catalog returns, and no others.

    The catalog callable is resolved per call, and the tools treat what it returns as
    the whole truth: `sample_rows` checks the name against the catalog it was just
    handed rather than passing it down, so naming a table that was not listed cannot
    read rows from it.
    """
    import pytest
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS

    entitled = True
    sampled: list[tuple[str, int]] = []

    def catalog() -> dict[str, Any]:
        if not entitled:
            return {"tables": []}
        return {
            "tables": [
                {
                    "name": "sales",
                    "rows": 2,
                    "columns": [
                        {"name": "customer_id", "type": "VARCHAR"},
                        {"name": "amount", "type": "BIGINT"},
                    ],
                }
            ]
        }

    def sample(name: str, limit: int) -> list[dict[str, Any]]:
        sampled.append((name, limit))
        return [{"customer_id": "c1", "amount": 100}]

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        warehouse_schema=catalog,
        warehouse_sample=sample,
    )

    described = _tool_text(asyncio.run(server.call_tool("describe_warehouse", {})))
    assert "1 warehouse table(s):" in described
    assert "- sales (2 rows): customer_id:VARCHAR, amount:BIGINT" in described
    schema = _tool_text(
        asyncio.run(server.call_tool("table_schema", {"name": "sales"}))
    )
    assert '"table": "sales"' in schema
    assert "c1" in _tool_text(
        asyncio.run(server.call_tool("sample_rows", {"name": "sales"}))
    )

    entitled = False
    assert "no tables yet" in _tool_text(
        asyncio.run(server.call_tool("describe_warehouse", {}))
    )
    for tool in ("table_schema", "sample_rows"):
        with pytest.raises(MCPError) as raised:
            asyncio.run(server.call_tool(tool, {"name": "sales"}))
        assert raised.value.error.code == INVALID_PARAMS
        assert "No warehouse table named 'sales'" in raised.value.error.message

    # The refusal has to land above the read: reaching the sampler at all would return
    # rows from a table the catalog had just declined to admit existed.
    assert sampled == [("sales", 10)], sampled


def test_sample_rows_bounds_the_row_count_it_asks_for() -> None:
    """A row count arriving over the wire is clamped before it reaches the warehouse.

    The floor matters as much as the ceiling. The value becomes a LIMIT, and a
    warehouse reads zero or fewer as "no limit", so a ceiling on its own would let a
    caller pull a whole table into a model's context by asking for none of it.
    """
    asked: list[int] = []

    def catalog() -> dict[str, Any]:
        return {"tables": [{"name": "sales", "rows": 1, "columns": []}]}

    def sample(name: str, limit: int) -> list[dict[str, Any]]:
        asked.append(limit)
        return []

    registry = Registry()
    server = build_server(
        registry,
        lambda: Runner(registry),
        warehouse_schema=catalog,
        warehouse_sample=sample,
    )
    for requested in (0, -5, 1000):
        asyncio.run(
            server.call_tool("sample_rows", {"name": "sales", "limit": requested})
        )
    assert asked == [1, 1, 100]

    # A server with no sampler does not advertise `sample_rows` at all, rather than
    # advertising one that can only decline: a listed tool that always refuses reads to
    # an agent as broken, and it retries instead of planning around the absence. The
    # schema tools stay, so such a server still describes the data it cannot sample.
    schema_only = build_server(
        registry, lambda: Runner(registry), warehouse_schema=catalog
    )
    assert "sample_rows" not in {t.name for t in asyncio.run(schema_only.list_tools())}
    assert "1 warehouse table(s):" in _tool_text(
        asyncio.run(schema_only.call_tool("describe_warehouse", {}))
    )
