"""``run()`` in a notebook cell, through the real app and the project's own
derivations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

_DERIVATIONS = '''\
from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, Param, derivation, serve


@derivation(inputs={"sales": Dataset("sales")}, serve=serve.table(max_rows=1))
def big_sales(ctx: Context) -> Artifact:
    """Sales rows above 10."""
    return Artifact.table(
        [{"id": r["id"], "amount": int(r["amount"])}
         for r in ctx.input("sales").rows if int(r["amount"]) > 10]
    )


@derivation(
    inputs={"sales": Dataset("sales")},
    params={"floor": Param("integer", required=False, default=0)},
    serve=serve.json(),
)
def total(ctx: Context) -> Artifact:
    """Total of amounts at or above a floor."""
    floor = ctx.param("floor")
    amounts = [int(r["amount"]) for r in ctx.input("sales").rows]
    return Artifact.json({"total": sum(a for a in amounts if a >= floor)})


@derivation(inputs={"sales": Dataset("sales")})
def helper(ctx: Context) -> Artifact:
    """Internal step, no serve contract."""
    return Artifact.json({"n": 1})
'''


def _outputs(http: TestClient, notebook_id: str, cell: str, source: str) -> str:
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": source})
    response = http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]})
    frames = [
        json.loads(line[len("data:") :])
        for line in response.text.splitlines()
        if line.startswith("data:") and "[DONE]" not in line
    ]
    return json.dumps([f for f in frames if f.get("event") == "output"])


def test_run_computes_the_projects_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    root = tmp_path / "proj"
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(
        "id,amount\na,5\nb,20\nc,30\n", encoding="utf-8"
    )
    (root / "elbi.yaml").write_text(
        "project: t\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    (root / "derivations").mkdir()
    (root / "derivations" / "d.py").write_text(_DERIVATIONS, encoding="utf-8")

    app = build(root, with_mcp=False)
    with TestClient(app) as http:
        notebook_id = http.post("/api/notebooks", json={"name": "R"}).json()["id"]
        cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
        run: Any = lambda src: _outputs(http, notebook_id, cell, src)  # noqa: E731

        # A table is every row (20+30), not the contract's max_rows=1 agent-sized cap
        # (which would give 20).
        assert '"text/plain": "50"' in run("sum(r['amount'] for r in run('big_sales'))")
        # Params travel as JSON and reach ctx.param: 5+20+30, then only 30.
        assert '"text/plain": "55"' in run("run('total')['total']")
        assert '"text/plain": "30"' in run("run('total', floor=25)['total']")
        assert "nope" in run("run('nope')")
        assert "'helper' is internal" in run("run('helper')")
