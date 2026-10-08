"""Every path that records a derivation writes the same store row.

A derivation reaches the store from a ``derivations/*.py`` file, from chat, from an MCP
proposal and from an Explore promotion. Each test drives one of those paths through the
real app and reads the result back over ``GET /api/derivations/{name}``, so a field one
path forgets to copy shows up in what the app serves.
"""

from __future__ import annotations

import os
import time
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from elbi import create_app
from elbi.authoring import make_derive_factory
from elbi.db import open_store
from elbi_core import Registry, Runner, SubprocessExecutor

_SALES = "customer_id,amount\nc1,100\nc2,5\nc3,40\n"

_TABLE_MANIFEST = {
    "format": "table",
    "title": "Churn risk",
    "columns": ["customer_id", "amount", "risk"],
    "maxRows": 50,
    "maxCells": 2000,
}


# ``__CLAIM__`` is replaced with the decorator's claim keyword, or with nothing.
_CHURN_RISK = '''\
from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(
        title="Churn risk", columns=["customer_id", "amount", "risk"], max_rows=50
    ),
    __CLAIM__
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores derived from recent sales activity."""
    return Artifact.table(
        [
            {
                "customer_id": row["customer_id"],
                "amount": float(row["amount"]),
                "risk": round(1.0 / (1.0 + float(row["amount"])), 4),
            }
            for row in ctx.input("sales").rows
        ]
    )
'''


def _project(root: Path) -> Path:
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "elbi.yaml").write_text(
        "project: t\n"
        "sources:\n"
        "  - name: sales\n"
        "    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    (root / "derivations").mkdir()
    (root / "derivations" / "churn_risk.py").write_text(
        _CHURN_RISK.replace("__CLAIM__", 'claim={"x": "amount", "y": "risk"},'),
        encoding="utf-8",
    )
    return root


def test_repo_derivation_claim_reaches_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim declared in a derivation file is on the derivation's detail."""
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    app = build(_project(tmp_path / "proj"), with_mcp=False)
    with TestClient(app) as http:
        detail = http.get("/api/derivations/churn_risk")
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["claim"] == {"x": "amount", "y": "risk"}
        assert body["origin"] == "repo"
        # A file derivation is never oracle-run in the app, so it has no verdict.
        assert body["verdict"] is None
        assert body["serve"] == _TABLE_MANIFEST


def _edit(path: Path, text: str) -> None:
    """Rewrite ``path`` the way a later edit would, with a newer modification time.

    Python reuses a cached ``.pyc`` when the source's size and modification time (to
    the second) both match, so a same-size rewrite within a second of the import would
    reload stale code.
    """
    path.write_text(text, encoding="utf-8")
    later = time.time() + 10
    os.utime(path, (later, later))


def test_reload_picks_up_a_claim_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing a file's claim and reloading (``elbi sync``) updates the detail."""
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    project = _project(tmp_path / "proj")
    derivation_file = project / "derivations" / "churn_risk.py"
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        _edit(
            derivation_file,
            _CHURN_RISK.replace("__CLAIM__", 'claim={"x": "risk", "y": "amount"},'),
        )
        reloaded = http.post("/api/project/reload")
        assert reloaded.status_code == 200, reloaded.text
        detail = http.get("/api/derivations/churn_risk").json()
        assert detail["claim"] == {"x": "risk", "y": "amount"}

        _edit(derivation_file, _CHURN_RISK.replace("__CLAIM__", ""))
        reloaded = http.post("/api/project/reload")
        assert reloaded.status_code == 200, reloaded.text
        assert http.get("/api/derivations/churn_risk").json()["claim"] is None


def test_explore_promotion_records_the_full_table_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted warehouse query records the table settings it is served with."""
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    app = build(_project(tmp_path / "proj"), with_mcp=False)
    with TestClient(app) as http:
        promoted = http.post(
            "/api/explore/promote",
            json={
                "name": "top_customer",
                "sql": "SELECT customer_id FROM sales ORDER BY amount DESC LIMIT 1",
                "source_id": "warehouse",
            },
        )
        assert promoted.status_code == 200, promoted.text
        assert promoted.json()["ok"] is True, promoted.json()

        detail = http.get("/api/derivations/top_customer").json()
        assert detail["serve"] == {"format": "table", "maxRows": 100, "maxCells": 2000}
        assert detail["claim"] is None
        assert detail["origin"] == "human"


# A clean dose-response effect the oracle certifies sound.
_CLEAN = """
def clean_effect(ctx):
    "Dose-response rows with a real effect."
    import random
    rng = random.Random(0)
    out = []
    for _ in range(300):
        x = rng.gauss(0, 1)
        y = 1.2 * x + rng.gauss(0, 0.8)
        out.append({"dose": round(x, 3), "response": round(y, 3)})
    return out
"""


class _NoClient:
    def step(self, transcript: object, tools: Sequence[object]) -> object:
        raise NotImplementedError


def _chat_detail(tmp_path: Path, fmt: str) -> dict[str, Any]:
    """Author ``clean_effect`` through the chat bridge in ``fmt``; return its detail."""
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = Registry()
    project = types.SimpleNamespace(
        registry=registry,
        config=types.SimpleNamespace(datasets=[]),
        make_runner=lambda: Runner(registry, executor=SubprocessExecutor(timeout=60)),
    )
    derive = make_derive_factory(
        project,
        store,
        make_runner=project.make_runner,
        dataset_names=lambda: [],
    )("c1", "does dose affect response?")
    outcome = derive(
        "clean_effect", _CLEAN, {"x": "dose", "y": "response"}, None, fmt, [], []
    )
    assert outcome.certified, outcome.error
    app = create_app(load_datasets=lambda: {}, client=_NoClient(), store=store)
    with TestClient(app) as http:
        detail = http.get("/api/derivations/clean_effect")
        assert detail.status_code == 200, detail.text
        body: dict[str, Any] = detail.json()
        return body


def test_chat_derivation_records_the_full_manifest(tmp_path: Path) -> None:
    """A chat derivation records its claim and the table settings it is served with."""
    detail = _chat_detail(tmp_path, "table")
    assert detail["claim"] == {"x": "dose", "y": "response"}
    assert detail["origin"] == "agent"
    assert detail["serve"] == {"format": "table", "maxRows": 100, "maxCells": 2000}


def test_chat_derivation_in_an_unknown_format_records_a_table(tmp_path: Path) -> None:
    """A format the chat bridge doesn't support is recorded as a table.

    The bridge serves table, markdown, json and text in their own format, and any other
    format, here ``"csv"``, as a table.
    """
    detail = _chat_detail(tmp_path, "csv")
    assert detail["serve"]["format"] == "table"
