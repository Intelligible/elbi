"""Analytics-as-code sync/pull round-trips against a real app over its HTTP API.

Drives the ``elbi_cli.config_sync`` engine through httpx's in-process ASGI
transport (no network server): write repo files, ``sync`` them into the app, read them
back through the app's API, ``pull`` into a second repo, and assert the round-trip is a
no-op. Queries and notebooks are used because they need no certification gate, so they
exercise the engine cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")

from elbi_cli import config_sync

_MINIMAL_IPYNB = {
    "cells": [
        {"cell_type": "markdown", "source": "# Demo", "metadata": {}},
        {
            "cell_type": "code",
            "source": "x = 1 + 1\nx",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


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


def _project(root: Path) -> Path:
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text("customer_id,amount\nc1,100\nc2,5\n")
    (root / "elbi.yaml").write_text(
        "project: t\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n"
    )
    # A human-authored derivation is certified by default, so it is a valid target for
    # a monitor (which watches only verified numbers).
    (root / "derivations").mkdir()
    (root / "derivations" / "churn_risk.py").write_text(_CHURN_DERIVATION)
    return root


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # TestClient is an httpx.Client subclass that drives the ASGI app synchronously, so
    # the sync engine (which expects an httpx.Client) runs in-process against the real
    # app: exactly what a remote httpx.Client(base_url=...) does in production.
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh'}")
    from fastapi.testclient import TestClient

    from elbi.serve import build

    app = build(_project(tmp_path / "proj"), with_mcp=False)
    with TestClient(app) as c:
        yield c


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sync_creates_then_is_idempotent(client: httpx.Client, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "queries/top_spenders.sql", "SELECT customer_id FROM sales\n")
    import json

    _write(repo, "notebooks/demo.ipynb", json.dumps(_MINIMAL_IPYNB))

    # First sync creates both objects.
    applied = config_sync.sync(client, repo)
    actions = {(c.surface, c.name): c.action for c in applied}
    assert actions[("queries", "top_spenders")] == "create"
    assert actions[("notebooks", "demo")] == "create"

    # They are live on the app.
    names = {q["name"] for q in client.get("/api/explore/queries").json()}
    assert "top_spenders" in names
    nb_names = {n["name"] for n in client.get("/api/notebooks").json()}
    assert "demo" in nb_names

    # Canonicalize the repo from the app (pull), then a second sync is a no-op; the
    # contract is idempotency on the canonical form the app round-trips to.
    config_sync.pull(client, repo)
    again = config_sync.sync(client, repo)
    assert again == []
    plan = [c for c in config_sync.plan(client, repo) if c.action != "unchanged"]
    assert plan == []


def test_syncing_a_changed_notebook_keeps_the_same_notebook(
    client: httpx.Client, tmp_path: Path
) -> None:
    """Pushing an edit must update the notebook, not replace it with a copy.

    This is the whole engine, not the route: ``sync`` used to delete the notebook and
    import the file as a new one, which produced the right cells and dropped everything
    else the row carried. The id is asserted because the folder rides on the same row,
    so a notebook that comes back as a new row at the root has lost where it was filed.
    """
    import json

    repo = tmp_path / "repo"
    _write(repo, "notebooks/demo.ipynb", json.dumps(_MINIMAL_IPYNB))
    config_sync.sync(client, repo)
    config_sync.pull(client, repo)
    created = client.get("/api/notebooks").json()
    assert len(created) == 1
    notebook_id = created[0]["id"]
    created_at = created[0]["created_at"]

    # put it in a folder, as somebody sharing their work would
    folder_id = client.post("/api/notebooks/folders", json={"name": "team"}).json()[
        "id"
    ]
    client.post(f"/api/notebooks/{notebook_id}/move", json={"folder_id": folder_id})

    # edit the file the way a person would, then push
    document = json.loads((repo / "notebooks" / "demo.ipynb").read_text())
    document["cells"].append(
        {
            "cell_type": "code",
            "source": "y = 2",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }
    )
    _write(repo, "notebooks/demo.ipynb", json.dumps(document))
    applied = config_sync.sync(client, repo)
    assert [(c.surface, c.name, c.action) for c in applied] == [
        ("notebooks", "demo", "update")
    ]

    after = client.get("/api/notebooks").json()
    assert len(after) == 1
    assert after[0]["id"] == notebook_id, "the notebook was replaced, not updated"
    assert after[0]["folder_id"] == folder_id, "the sync moved it out of its folder"
    assert after[0]["created_at"] == created_at
    # and the edit did land
    view = client.get(f"/api/notebooks/{notebook_id}").json()
    assert [c["source"] for c in view["cells"]][-1] == "y = 2"


def test_pull_round_trips(client: httpx.Client, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo,
        "queries/revenue.sql",
        "-- source: warehouse\nSELECT sum(amount) FROM sales\n",
    )
    import json

    _write(repo, "notebooks/eda.ipynb", json.dumps(_MINIMAL_IPYNB))
    config_sync.sync(client, repo)

    # Pull into a fresh repo, then a sync from that repo must be a no-op.
    out = tmp_path / "pulled"
    written = config_sync.pull(client, out)
    got = {(c.surface, c.name) for c in written}
    assert ("queries", "revenue") in got
    assert ("notebooks", "eda") in got
    assert (out / "queries" / "revenue.sql").exists()
    assert (out / "notebooks" / "eda.ipynb").exists()

    residual = [c for c in config_sync.plan(client, out) if c.action != "unchanged"]
    assert residual == []  # pull then sync is a no-op


def test_cli_sync_and_schema(
    client: httpx.Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI verbs are thin wrappers over the engine; drive them through Typer's runner
    # with the in-process client injected, to guard the command wiring + exit codes.
    import contextlib

    from typer.testing import CliRunner

    from elbi_cli.app import app
    from elbi_cli.commands import config_cmd

    @contextlib.contextmanager
    def _fake_client(url, token, **kwargs):
        yield client

    monkeypatch.setattr(config_cmd, "client_for", _fake_client)
    repo = tmp_path / "repo"
    _write(repo, "queries/q.sql", "SELECT 1\n")

    runner = CliRunner()
    synced = runner.invoke(app, ["sync", "-C", str(repo)])
    assert synced.exit_code == 0, synced.output
    assert "q" in {q["name"] for q in client.get("/api/explore/queries").json()}

    schema = runner.invoke(app, ["schema"])
    assert schema.exit_code == 0, schema.output
    assert "sales" in schema.output  # the declared warehouse table shows up


def test_cli_plan_pull_and_the_archive_round_trip(
    client: httpx.Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive `plan`, `pull`, `export` and `import` through Typer.

    The engine below them is covered above; this is the wiring. A command whose body
    never runs in a test can print a key the payload stopped carrying and nothing
    notices until someone runs it.
    """
    import contextlib

    from typer.testing import CliRunner

    from elbi_cli.app import app
    from elbi_cli.commands import config_cmd, export_cmd

    @contextlib.contextmanager
    def _fake_client(url, token, **kwargs):
        yield client

    monkeypatch.setattr(config_cmd, "client_for", _fake_client)
    monkeypatch.setattr(export_cmd, "client_for", _fake_client)
    runner = CliRunner()

    repo = tmp_path / "planrepo"
    _write(repo, "queries/planned.sql", "SELECT 2\n")

    planned = runner.invoke(app, ["plan", "-C", str(repo)])
    assert planned.exit_code == 0, planned.output
    assert "planned" in planned.output

    assert runner.invoke(app, ["sync", "-C", str(repo)]).exit_code == 0
    settled = runner.invoke(app, ["plan", "-C", str(repo)])
    assert settled.exit_code == 0, settled.output
    assert "planned" not in settled.output  # nothing left to do

    pulled_into = tmp_path / "pulled"
    pulled_into.mkdir()
    pulled = runner.invoke(app, ["pull", "-C", str(pulled_into)])
    assert pulled.exit_code == 0, pulled.output
    assert (pulled_into / "queries" / "planned.sql").exists()

    archive = tmp_path / "workspace.zip"
    exported = runner.invoke(app, ["export", "--out", str(archive), "-C", str(repo)])
    assert exported.exit_code == 0, exported.output
    assert archive.exists() and archive.stat().st_size > 0

    extracted = tmp_path / "imported"
    extracted.mkdir()
    imported = runner.invoke(app, ["import", str(archive), "-C", str(extracted)])
    assert imported.exit_code == 0, imported.output


_DASHBOARD = {
    "specVersion": "1.0",
    "kind": "Dashboard",
    "title": "Sales",
    "pages": [
        {
            "name": "main",
            "widgets": [
                {
                    "id": "revenue_table",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "bind": {"derivation": "revenue", "params": {}},
                }
            ],
        }
    ],
}


def test_dashboards_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    _write(repo, "dashboards/sales.yaml", yaml.safe_dump(_DASHBOARD))
    created = config_sync.sync(client, repo)
    assert ("dashboards", "sales") in {(c.surface, c.name) for c in created}
    assert "sales" in {d["name"] for d in client.get("/api/dashboards").json()}

    # Canonicalize then confirm idempotency (create draft, no certified binding needed).
    config_sync.pull(client, repo)
    residual = [c for c in config_sync.plan(client, repo) if c.action != "unchanged"]
    assert residual == []


def test_prune_deletes_app_objects_absent_from_repo(
    client: httpx.Client, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "queries/keep.sql", "SELECT 1\n")
    _write(repo, "queries/drop.sql", "SELECT 2\n")
    config_sync.sync(client, repo)
    assert {q["name"] for q in client.get("/api/explore/queries").json()} == {
        "keep",
        "drop",
    }
    # Remove one file, sync with prune → the app drops it too.
    (repo / "queries" / "drop.sql").unlink()
    config_sync.sync(client, repo, prune=True)
    assert {q["name"] for q in client.get("/api/explore/queries").json()} == {"keep"}


def test_a_plan_reports_only_what_its_own_sync_would_do(
    client: httpx.Client, tmp_path: Path
) -> None:
    """A plan is a promise about the next command, so its flags must match it.

    Removals happen only under ``--prune``, so listing them in a plain plan overstates
    the damage -- and a plan that names deletions the sync will not perform is one a
    careful operator stops and reads the source over.
    """
    repo = tmp_path / "repo"
    _write(repo, "queries/keep.sql", "SELECT 1\n")
    _write(repo, "queries/extra.sql", "SELECT 2\n")
    config_sync.sync(client, repo)
    (repo / "queries" / "extra.sql").unlink()

    plain = [c for c in config_sync.plan(client, repo) if c.action == "delete"]
    assert plain == [], "a plain plan promised a deletion sync would not make"
    pruning = [
        c for c in config_sync.plan(client, repo, prune=True) if c.action == "delete"
    ]
    assert [c.name for c in pruning] == ["extra"]

    # ...and the plain sync leaves it alone, which is what the plain plan said.
    config_sync.sync(client, repo)
    assert "extra" in {q["name"] for q in client.get("/api/explore/queries").json()}


def test_schemas_are_valid_and_catch_bad_files(tmp_path: Path) -> None:
    import jsonschema

    schemas = config_sync.schemas()
    assert set(schemas) == {
        "metrics",
        "dashboards",
        "features",
        "monitors",
        "schedules",
        "workflows",
        "checks",
        "models",
        "queries",
        "notebooks",
        "certificates",
    }
    for spec in schemas.values():  # each is a well-formed Draft 2020-12 schema
        jsonschema.Draft202012Validator.check_schema(spec)

    # A malformed monitor (unknown method, missing target) is a validation error.
    repo = tmp_path / "repo"
    _write(repo, "monitors/bad.yaml", "target_kind: derivation\nmethod: bogus\n")
    problems = config_sync.validate(repo)
    assert any("method" in p for p in problems)
    assert any("target" in p for p in problems)


def test_schema_json_command_writes_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from typer.testing import CliRunner

    from elbi_cli.app import app

    result = CliRunner().invoke(app, ["schema", "--json", "-C", str(tmp_path)])
    assert result.exit_code == 0, result.output
    written = tmp_path / ".elbi" / "schemas"
    assert (written / "metrics.json").exists()
    assert (written / "models.json").exists()
    metric = json.loads((written / "metrics.json").read_text())
    assert "name" not in metric["$defs"]["metric"]["required"]  # file stem supplies it


_NEW_DERIVATION = '''\
"""High-value customers flagged from the sales dataset."""

from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, derivation


@derivation(inputs={"sales": Dataset("sales")})
def high_value(ctx: Context) -> Artifact:
    """Customers whose single order exceeds 50."""
    rows = [
        {"customer_id": r["customer_id"], "amount": r["amount"]}
        for r in ctx.input("sales").rows
        if r["amount"] > 50
    ]
    return Artifact.table(rows)
'''


def test_repo_derivations_are_in_the_db_list(
    client: httpx.Client, tmp_path: Path
) -> None:
    # The UI's Derivations page reads /api/derivations (the store), never the code tree.
    # A repo-authored derivation is mirrored into the store at boot so it lists there.
    listed = {d["name"]: d for d in client.get("/api/derivations").json()}
    assert "churn_risk" in listed
    assert listed["churn_risk"]["origin"] == "repo"
    # Its detail is reachable (project-global, no owner) and renders without a verdict.
    detail = client.get("/api/derivations/churn_risk")
    assert detail.status_code == 200, detail.text
    assert detail.json()["verdict"] is None

    # Add a file + reload → it appears in the store list; delete + reload → it's gone.
    proj = tmp_path / "proj"
    (proj / "derivations" / "high_value.py").write_text(_NEW_DERIVATION)
    client.post("/api/project/reload")
    assert "high_value" in {d["name"] for d in client.get("/api/derivations").json()}
    (proj / "derivations" / "high_value.py").unlink()
    client.post("/api/project/reload")
    remaining = {d["name"] for d in client.get("/api/derivations").json()}
    assert "high_value" not in remaining


def test_reload_picks_up_added_and_removed_derivations(
    client: httpx.Client, tmp_path: Path
) -> None:
    # The app was built from tmp_path/proj, which ships one derivation (churn_risk).
    derivs = {
        c["name"]
        for c in client.get("/api/catalog").json()
        if c["type"] == "derivation"
    }
    assert derivs == {"churn_risk"}

    # Add a new derivation source file on disk, then reload (what `sync` triggers).
    proj = tmp_path / "proj"
    (proj / "derivations" / "high_value.py").write_text(_NEW_DERIVATION)
    added = client.post("/api/project/reload")
    assert added.status_code == 200, added.text
    assert "high_value" in added.json()["derivations"]["added"]
    live = {
        c["name"]
        for c in client.get("/api/catalog").json()
        if c["type"] == "derivation"
    }
    assert live == {"churn_risk", "high_value"}  # new derivation is live, no restart

    # Delete a source file → reload removes it from the live registry.
    (proj / "derivations" / "churn_risk.py").unlink()
    removed = client.post("/api/project/reload").json()
    assert "churn_risk" in removed["derivations"]["removed"]
    remaining = {
        c["name"]
        for c in client.get("/api/catalog").json()
        if c["type"] == "derivation"
    }
    assert remaining == {"high_value"}


def _round_trips(client: httpx.Client, repo: Path) -> None:
    """sync → pull → sync is a no-op (idempotent on the pulled canonical form)."""
    config_sync.sync(client, repo)
    config_sync.pull(client, repo)
    residual = [c for c in config_sync.plan(client, repo) if c.action != "unchanged"]
    assert residual == [], residual


def test_features_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    store = {
        "entities": [{"name": "customer", "joinKey": "customer_id"}],
        "featureViews": [
            {
                "name": "customer_risk",
                "entities": ["customer"],
                "source": "churn_risk",
                "features": [{"name": "risk"}],
            }
        ],
    }
    _write(repo, "features/store.yaml", yaml.safe_dump(store))
    config_sync.sync(client, repo)
    entities = {e["name"] for e in client.get("/api/features/entities").json()}
    views = {v["name"] for v in client.get("/api/features/views").json()}
    assert "customer" in entities
    assert "customer_risk" in views
    _round_trips(client, repo)


def test_schedules_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    schedule = {"selection": "stale", "mode": "cron", "cron": "0 * * * *"}
    _write(repo, "schedules/nightly.yaml", yaml.safe_dump(schedule))
    config_sync.sync(client, repo)
    names = {s["name"] for s in client.get("/api/orchestration/schedules").json()}
    assert "nightly" in names
    _round_trips(client, repo)


def test_notebook_schedule_and_deps_round_trip(
    client: httpx.Client, tmp_path: Path
) -> None:
    import json

    repo = tmp_path / "repo"
    ipynb = {
        **_MINIMAL_IPYNB,
        "metadata": {
            "elbi": {
                "deps": ["pandas"],
                "schedule": {
                    "enabled": True,
                    "mode": "interval",
                    "interval_hours": 24,
                    "params": {},
                },
            }
        },
    }
    _write(repo, "notebooks/nightly.ipynb", json.dumps(ipynb))
    config_sync.sync(client, repo)
    nb_id = next(
        n["id"] for n in client.get("/api/notebooks").json() if n["name"] == "nightly"
    )
    view = client.get(f"/api/notebooks/{nb_id}").json()
    assert view["schedule"]["mode"] == "interval"
    assert view["schedule"]["interval_hours"] == 24
    assert view["deps"] == ["pandas"]
    _round_trips(client, repo)


def test_running_a_notebook_does_not_make_the_next_sync_conflict(
    client: httpx.Client, tmp_path: Path
) -> None:
    """The edit -> sync -> run -> edit loop must survive its second turn.

    A run records per-cell telemetry (which side of the data boundary the cell landed
    on, and the queries it pushed, with their timings) into the cell's own metadata.
    That is regenerated state, but it used to reach the fingerprint the repo is compared
    against, so a notebook that had *run* was reported as changed in the app -- and the
    ``sync`` refused, demanding ``--force``. Timings differ between runs, so re-running
    identical code was enough to trigger it.
    """
    import json

    repo = tmp_path / "repo"
    _write(repo, "notebooks/loop.ipynb", json.dumps(_MINIMAL_IPYNB))
    config_sync.sync(client, repo)
    config_sync.pull(client, repo)

    notebook_id = next(
        n["id"] for n in client.get("/api/notebooks").json() if n["name"] == "loop"
    )
    view = client.get(f"/api/notebooks/{notebook_id}").json()
    code_cell = next(c for c in view["cells"] if c["cell_type"] == "code")

    # What a run persists, via the same store call the run path uses.
    store = client.app.state.store  # type: ignore[attr-defined]
    store.save_cell_result(
        code_cell["id"],
        json.dumps([{"output_type": "stream", "name": "stdout", "text": "2\n"}]),
        1,
        data_mode="pushed_down",
        queries=[
            {"sql": "select 1", "duration_ms": 3094.9, "rows": 1, "truncated": False}
        ],
    )

    assert [
        c.action for c in config_sync.plan(client, repo) if c.surface == "notebooks"
    ] == ["unchanged"], "the run alone was reported as an app-side change"

    # ...and a real edit still pushes without --force.
    document = json.loads((repo / "notebooks" / "loop.ipynb").read_text())
    document["cells"].append(
        {
            "cell_type": "code",
            "source": "y = 2",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }
    )
    _write(repo, "notebooks/loop.ipynb", json.dumps(document))
    applied = config_sync.sync(client, repo)
    assert ("notebooks", "loop", "update") in [
        (c.surface, c.name, c.action) for c in applied
    ]


def test_strip_keeps_authored_cell_metadata() -> None:
    """Only what a run wrote is dropped; a ``parameters`` tag is the notebook's own."""
    stripped = config_sync._strip_notebook(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "n = 1",
                    "outputs": [{"output_type": "stream", "text": "x"}],
                    "execution_count": 3,
                    "metadata": {
                        "tags": ["parameters"],
                        "elbi": {
                            "data_mode": "pushed_down",
                            "queries": [{"sql": "select 1", "duration_ms": 12.3}],
                            "pinned": True,
                        },
                    },
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )
    cell = stripped["cells"][0]
    assert cell["outputs"] == [] and cell["execution_count"] is None
    assert cell["metadata"]["tags"] == ["parameters"]
    assert cell["metadata"]["elbi"] == {"pinned": True}


def test_strip_drops_our_key_when_a_run_wrote_all_of_it() -> None:
    """A notebook that has run and one that has not must serialise identically."""
    ran = config_sync._strip_notebook(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "n = 1",
                    "outputs": [],
                    "execution_count": 1,
                    "metadata": {"elbi": {"data_mode": "materialised", "queries": []}},
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )
    assert ran["cells"][0]["metadata"] == {}


def test_workflows_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    workflow = {
        "steps": [
            {"id": "build", "selection": "all"},
            {
                "id": "publish",
                "selection": "all",
                "dependsOn": ["build"],
                "runIf": "all_success",
            },
        ]
    }
    _write(repo, "workflows/nightly.yaml", yaml.safe_dump(workflow))
    config_sync.sync(client, repo)
    remote = client.get("/api/orchestration/workflows").json()
    assert {w["name"] for w in remote} == {"nightly"}
    assert [s["id"] for s in remote[0]["steps"]] == ["build", "publish"]
    assert remote[0]["steps"][1]["runIf"] == "all_success"
    _round_trips(client, repo)


def test_checks_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    check = {
        "asset": "big_spenders",
        "name": "nonneg",
        "expr": "total_spend >= 0",
        "severity": "error",
    }
    _write(repo, "checks/big_spenders.nonneg.yaml", yaml.safe_dump(check))
    config_sync.sync(client, repo)
    remote = client.get("/api/orchestration/checks").json()
    assert [(c["asset"], c["name"], c["severity"]) for c in remote] == [
        ("big_spenders", "nonneg", "error")
    ]
    _round_trips(client, repo)


def test_models_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    policy = {
        "source_kind": "dataset",
        "dataset": "sales",
        "target": "amount",
        "mode": "interval",
        "interval_hours": 24.0,
    }
    _write(repo, "models/spend.yaml", yaml.safe_dump(policy))
    config_sync.sync(client, repo)
    got = client.get("/api/registry/models/spend/retrain").json()
    assert got["configured"] and got["dataset"] == "sales" and got["target"] == "amount"
    _round_trips(client, repo)


def test_monitors_round_trip(client: httpx.Client, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    monitor = {
        "target_kind": "derivation",
        "target": "churn_risk",
        "method": "mad",
        "sensitivity": 3.0,
        "window": 30,
        "interval_hours": 1.0,
    }
    _write(repo, "monitors/risk.yaml", yaml.safe_dump(monitor))
    config_sync.sync(client, repo)
    assert "risk" in {m["name"] for m in client.get("/api/monitors").json()}
    _round_trips(client, repo)


def _cli_run(tc: Any, monkeypatch: pytest.MonkeyPatch):
    """Invoke `elbi run ...` against a shared in-process app (starlette's
    TestClient bridges sync↔ASGI); a no-op ``__exit__`` keeps the lifespan alive across
    commands, since each command opens ``with _client(...)``."""
    from typer.testing import CliRunner

    from elbi_cli.app import app as cli_app
    from elbi_cli.commands import run_cmd

    class _Ctx:
        def __enter__(self) -> Any:
            return tc

        def __exit__(self, *exc: Any) -> bool:
            return False

    monkeypatch.setattr(run_cmd, "_client", lambda url, token: _Ctx())
    return CliRunner(), cli_app


def test_run_cli_materialize_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh'}")
    from fastapi.testclient import TestClient

    from elbi.serve import build

    app = build(_project(tmp_path / "proj"), with_mcp=False)
    with TestClient(app) as tc:
        runner, cli_app = _cli_run(tc, monkeypatch)
        res = runner.invoke(cli_app, ["run", "materialize", "--select", "all"])
        assert res.exit_code == 0, res.output
        assert "churn_risk" in res.output
        res = runner.invoke(cli_app, ["run", "status"])
        assert res.exit_code == 0 and "churn_risk" in res.output


def test_run_cli_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh'}")
    from fastapi.testclient import TestClient

    from elbi.serve import build

    app = build(_project(tmp_path / "proj"), with_mcp=False)
    with TestClient(app) as tc:
        tc.post(
            "/api/orchestration/workflows",
            json={"name": "wf", "steps": [{"id": "build", "selection": "all"}]},
        )
        runner, cli_app = _cli_run(tc, monkeypatch)
        res = runner.invoke(cli_app, ["run", "workflow", "wf"])
        assert res.exit_code == 0, res.output
        assert "build" in res.output and "succeeded" in res.output
        res = runner.invoke(cli_app, ["run", "workflow", "nope"])
        assert res.exit_code == 1


def test_a_notebook_can_be_scheduled_on_a_calendar_time(
    client: httpx.Client, tmp_path: Path
) -> None:
    """An interval cannot say "the 1st at 06:00", and 720 hours drifts off it."""
    import json

    repo = tmp_path / "repo"
    ipynb = {
        **_MINIMAL_IPYNB,
        "metadata": {
            "elbi": {
                "schedule": {
                    "enabled": True,
                    "mode": "cron",
                    "cron": "0 6 1 * *",
                    "timezone": "America/Phoenix",
                    "params": {},
                }
            }
        },
    }
    _write(repo, "notebooks/monthly.ipynb", json.dumps(ipynb))
    config_sync.sync(client, repo)
    nb_id = next(
        n["id"] for n in client.get("/api/notebooks").json() if n["name"] == "monthly"
    )
    view = client.get(f"/api/notebooks/{nb_id}").json()
    assert view["schedule"]["mode"] == "cron"
    assert view["schedule"]["cron"] == "0 6 1 * *"
    assert view["schedule"]["timezone"] == "America/Phoenix"
    _round_trips(client, repo)
