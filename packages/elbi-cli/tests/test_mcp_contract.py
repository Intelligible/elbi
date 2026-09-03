"""End-to-end tests for the data-cleaning MCP tools over the in-memory server.

Drives the real server surface: the read-only profile/suggest/verify tools, and
``propose_derivation`` gating certification on a declared data contract.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from elbi_cli.mcp_server import build_server
from elbi_core import Registry, RoutingExecutor, Runner, SubprocessExecutor
from elbi_core.config import DataBindings


def _tool_text(result: Any) -> str:
    if hasattr(result, "content"):
        result = result.content
    content = result[0] if isinstance(result, tuple) else result
    blocks = content if isinstance(content, (list, tuple)) else [content]
    return "".join(getattr(b, "text", "") for b in blocks)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _server(tmp_path: Path, filename: str) -> MCPServer:
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


def test_contract_tools_are_registered(tmp_path: Path) -> None:
    _write_csv(tmp_path / "d.csv", [{"id": "1"}])
    server = _server(tmp_path, "d.csv")
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"profile_dataset", "suggest_contract", "verify_contract"} <= names


def test_profile_suggest_and_verify_round_trip(tmp_path: Path) -> None:
    rows = [{"id": str(i), "grade": ["A", "B", "C"][i % 3]} for i in range(1, 61)]
    _write_csv(tmp_path / "d.csv", rows)
    server = _server(tmp_path, "d.csv")

    profile = _tool_text(
        asyncio.run(server.call_tool("profile_dataset", {"dataset": "d"}))
    )
    assert '"inferredType": "integer"' in profile

    suggested = _tool_text(
        asyncio.run(server.call_tool("suggest_contract", {"dataset": "d"}))
    )
    contract = json.loads(suggested)
    assert contract["kind"] == "DataContract"

    # The suggested contract certifies the data it was drawn from.
    verdict = _tool_text(
        asyncio.run(
            server.call_tool("verify_contract", {"dataset": "d", "contract": contract})
        )
    )
    assert "SOUND" in verdict


def test_verify_contract_reports_located_violations(tmp_path: Path) -> None:
    _write_csv(tmp_path / "d.csv", [{"id": "1"}, {"id": "1"}])  # duplicate id
    server = _server(tmp_path, "d.csv")
    contract = {
        "specVersion": "1.0",
        "kind": "DataContract",
        "fields": [{"name": "id", "type": "integer", "constraints": {"unique": True}}],
    }
    out = _tool_text(
        asyncio.run(
            server.call_tool("verify_contract", {"dataset": "d", "contract": contract})
        )
    )
    assert "UNSOUND" in out and "id.unique" in out


def test_propose_is_gated_on_the_data_contract(tmp_path: Path) -> None:
    _write_csv(tmp_path / "d.csv", [{"id": "1"}, {"id": "2"}, {"id": "2"}])
    contract = {
        "specVersion": "1.0",
        "kind": "DataContract",
        "fields": [
            {
                "name": "id",
                "type": "integer",
                "constraints": {"required": True, "unique": True},
            }
        ],
        "table": {"primaryKey": ["id"]},
    }

    # A derivation that passes the raw (duplicated) rows through violates the contract.
    passthrough = "def clean(ctx):\n    return list(ctx.input('d').rows)\n"
    server = _server(tmp_path, "d.csv")
    held = _tool_text(
        asyncio.run(
            server.call_tool(
                "propose_derivation",
                {
                    "name": "clean",
                    "source": passthrough,
                    "inputs": ["d"],
                    "contract": contract,
                },
            )
        )
    )
    assert "did not satisfy its data contract" in held and "held" in held
    assert "run_clean" not in {t.name for t in asyncio.run(server.list_tools())}

    # A derivation that dedupes on id satisfies the contract, so it is certified.
    dedupe = (
        "def clean(ctx):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for r in ctx.input('d').rows:\n"
        "        if r['id'] not in seen:\n"
        "            seen.add(r['id'])\n"
        "            out.append(r)\n"
        "    return out\n"
    )
    server2 = _server(tmp_path, "d.csv")
    certified = _tool_text(
        asyncio.run(
            server2.call_tool(
                "propose_derivation",
                {
                    "name": "clean",
                    "source": dedupe,
                    "inputs": ["d"],
                    "contract": contract,
                },
            )
        )
    )
    assert "certified" in certified
    assert "run_clean" in {t.name for t in asyncio.run(server2.list_tools())}
