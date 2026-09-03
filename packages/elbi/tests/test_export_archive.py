"""Round-trip tests for `elbi export` / `import`: the workspace archive (IP-31).

Two real apps, each built from its own project directory via
``elbi.serve.build`` -- the same harness ``test_config_sync.py`` uses -- so the
export engine runs in-process against real ASGI apps rather than a live server. App A
gets a certified (registry-status) derivation, a dashboard, and a metric bound to it;
``export`` writes an archive; ``import`` applies it to a clean app B; a ``plan`` against
B is then asserted to be a no-op for every non-certificate surface, which is round-trip
equality for definitions (AC-3).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elbi.serve import build
from elbi_cli import config_sync
from elbi_cli.commands.export_cmd import _export, _extract

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("cryptography")

_CHURN_DERIVATION = '''\
"""Per-customer churn-risk scores from the sales dataset."""

from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", columns=["customer_id", "risk"], max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores derived from recent sales activity."""
    rows = ctx.input("sales").rows
    scored = [
        {"customer_id": r["customer_id"], "risk": round(1.0 / (1.0 + r["amount"]), 4)}
        for r in rows
    ]
    return Artifact.table(scored)
'''

_DASHBOARD = {
    "specVersion": "1.0",
    "kind": "Dashboard",
    "name": "risk",
    "title": "Risk",
    "pages": [
        {
            "name": "main",
            "widgets": [
                {
                    "id": "w1",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "bind": {"derivation": "churn_risk"},
                }
            ],
        }
    ],
}

_METRIC = {
    "name": "risk_metric",
    "type": "simple",
    "source": "churn_risk",
    "measure": {"agg": "sum", "column": "risk"},
}


def _seed_feature_store(client) -> None:  # type: ignore[no-untyped-def]
    """Register one entity and one feature view bound to the certified derivation.

    Without this the round trip below never writes ``features/store.yaml``, which is
    exactly how a ``pull`` emitting a manifest ``sync`` rejects survived a suite that
    otherwise round-trips every surface.
    """
    entity = client.post(
        "/api/features/entities",
        json={"name": "customer", "join_key": "customer_id", "value_type": "string"},
    )
    assert entity.status_code == 200, entity.text
    view = client.post(
        "/api/features/views",
        json={
            "name": "customer_risk",
            "entities": ["customer"],
            "source": "churn_risk",
            "features": [{"name": "risk"}],
            "description": "Per-customer churn-risk score.",
        },
    )
    assert view.status_code == 200, view.text


def _project(root: Path) -> Path:
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text("customer_id,amount\nc1,100\nc2,5\n")
    (root / "elbi.yaml").write_text(
        "project: t\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n"
    )
    (root / "derivations").mkdir()
    (root / "derivations" / "churn_risk.py").write_text(_CHURN_DERIVATION)
    return root


@pytest.fixture
def app_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh_a'}")
    root = _project(tmp_path / "proj_a")
    with TestClient(build(root, with_mcp=False)) as client:
        yield root, client


@pytest.fixture
def app_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh_b'}")
    root = _project(tmp_path / "proj_b")
    with TestClient(build(root, with_mcp=False)) as client:
        yield root, client


def _non_certificate_diff(client, root: Path) -> list[config_sync.Change]:  # type: ignore[no-untyped-def]
    return [
        c
        for c in config_sync.plan(client, root)
        if c.surface != "certificates" and c.action != "unchanged"
    ]


def test_export_then_import_round_trips_definitions(  # type: ignore[no-untyped-def]
    app_a, app_b, tmp_path: Path
) -> None:
    root_a, client_a = app_a
    _root_b, client_b = app_b

    created = client_a.post("/api/dashboards", json=_DASHBOARD)
    assert created.status_code == 200, created.text
    assert client_a.post("/api/metrics", json=_METRIC).status_code == 200
    _seed_feature_store(client_a)

    archive = tmp_path / "workspace.zip"
    _export(client_a, root_a, archive)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "derivations/churn_risk.py" in names
    assert "features/store.yaml" in names
    # Sync state describes this archive's *source*; carrying it makes the import below
    # diff a fresh target against the wrong baseline and skip surfaces silently.
    assert not [n for n in names if n.startswith(".elbi/")], names
    # Compiled bytecode is not portable evidence.
    assert not [n for n in names if "__pycache__" in n], names
    assert any(n.startswith("records/derivations/") for n in names)
    assert any(n.startswith("records/dashboards/") for n in names)
    assert any(n.startswith("records/metrics/") for n in names)

    extracted = _extract(archive, tmp_path / "extract")
    defs = [s for s in config_sync.surfaces() if s.name != "certificates"]
    config_sync.sync(client_b, extracted, surfaces_=defs)

    assert _non_certificate_diff(client_b, extracted) == []


def test_import_actually_lands_the_feature_view_on_the_target(  # type: ignore[no-untyped-def]
    app_a, app_b, tmp_path: Path
) -> None:
    """The target must really hold the view -- not merely report nothing to do.

    ``plan`` was the thing that lied: with the source's sync state in the archive it
    reported "repo and app match" for a surface the target had never received. So this
    asserts against B's own state rather than against a diff of it.
    """
    root_a, client_a = app_a
    _root_b, client_b = app_b
    _seed_feature_store(client_a)

    archive = tmp_path / "features.zip"
    _export(client_a, root_a, archive)
    extracted = _extract(archive, tmp_path / "extract_features")
    assert not (extracted / ".elbi").exists()

    defs = [s for s in config_sync.surfaces() if s.name != "certificates"]
    config_sync.sync(client_b, extracted, surfaces_=defs)

    views = client_b.get("/api/features/views").json()
    assert [v["name"] for v in views] == ["customer_risk"]
    assert [f["name"] for f in views[0]["features"]] == ["risk"]


def test_import_drops_sync_state_left_by_an_older_archive(tmp_path: Path) -> None:
    """An archive written before export learned to omit it must still import cleanly."""
    stale = tmp_path / "stale.zip"
    with zipfile.ZipFile(stale, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema": "elbi.export/v1"}))
        zf.writestr(
            ".elbi/state.json",
            json.dumps({"surfaces": {"features": {"store": "deadbeef"}}}),
        )
    extracted = _extract(stale, tmp_path / "out_stale")
    assert not (extracted / ".elbi").exists()


def test_import_refuses_an_unknown_schema_major(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema": "elbi.export/v99"}))
    with pytest.raises(config_sync.SyncError, match="not supported"):
        _extract(bogus, tmp_path / "out")


def test_import_refuses_a_zip_slip_member(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema": "elbi.export/v1"}))
        zf.writestr("../escape.py", "pwned = True\n")
    with pytest.raises(config_sync.SyncError, match="escapes"):
        _extract(evil, tmp_path / "out")


def test_records_are_written_but_never_replayed_into_the_target(  # type: ignore[no-untyped-def]
    app_a, app_b, tmp_path: Path
) -> None:
    root_a, client_a = app_a
    _root_b, client_b = app_b
    archive = tmp_path / "workspace2.zip"
    _export(client_a, root_a, archive)
    extracted = _extract(archive, tmp_path / "extract2")

    assert (extracted / "records" / "derivations" / "churn_risk.json").exists()

    defs = [s for s in config_sync.surfaces() if s.name != "certificates"]
    config_sync.sync(client_b, extracted, surfaces_=defs)

    history = client_b.get("/api/derivations/churn_risk/history").json()
    assert history == []
