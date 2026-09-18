"""Integration tests for the notebook API over a real store and kernel.

These drive the HTTP surface with a TestClient (the boundary the browser uses), so they
cover the wiring end to end: creating and editing cells, a reactive run that streams
outputs over SSE and cascades to dependents, staleness after an edit, a parameterized
rerun, ``.ipynb`` export, and promoting a cell to a derivation. The kernel is the real
subprocess kernel; the LLM client is a stub, since notebooks never call it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import open_store
from elbi.notebooks import NotebookService
from elbi_agent import DeriveOutcome
from elbi_core.sandbox import ComputeProfile, ComputeProfiles


def _profiles(**overrides: object) -> ComputeProfiles:
    """A one-profile menu, so a test can size the kernel it is about to start."""
    return ComputeProfiles(
        profiles=(ComputeProfile(name="test", **overrides),), default="test"
    )


DATASETS = {"nums": [{"x": 1}, {"x": 2}, {"x": 3}]}


def _sse(response: Any) -> list[dict[str, Any]]:
    """Parse a run response's SSE ``data:`` frames into events (dropping [DONE])."""
    events: list[dict[str, Any]] = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                events.append(json.loads(payload))
    return events


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, str]]]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    promoted: dict[str, str] = {}

    def derive_factory(conversation_id: str, question: str):  # type: ignore[no-untyped-def]
        def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
            promoted["name"] = name
            return DeriveOutcome(
                certified=True, verdict="sound", rendered=f"promoted {name}"
            )

        return derive

    service = NotebookService(
        store=store,
        load_datasets=lambda: DATASETS,
        profiles=_profiles(max_runtime=15),
        derive_factory=derive_factory,
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        notebook_service=service,
    )
    with TestClient(app) as http:
        yield http, promoted
    service.close()


@pytest.fixture
def compute_client(tmp_path: Path) -> Iterator[tuple[TestClient, Any]]:
    """An app whose menu has a restricted profile and a very short idle timeout."""
    store = open_store(f"sqlite:{tmp_path / 'compute.db'}")
    profiles = ComputeProfiles(
        profiles=(
            ComputeProfile(name="small", cpu="2", memory="4Gi", max_runtime=15),
            ComputeProfile(name="large", cpu="8", memory="32Gi", max_runtime=15),
            ComputeProfile(name="gpu", gpu=1, max_runtime=15),
        ),
        default="small",
    )
    recorded: list[tuple[str, str, float, str]] = []
    service = NotebookService(
        store=store,
        load_datasets=lambda: DATASETS,
        profiles=profiles,
        on_session_end=lambda nb, profile, seconds, kind: recorded.append(
            (nb, profile.name, seconds, kind)
        ),
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        notebook_service=service,
    )
    with TestClient(app) as http:
        yield http, (service, recorded)
    service.close()


def test_create_add_and_reactive_run(client: tuple[TestClient, dict[str, str]]) -> None:
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    view = http.get(f"/api/notebooks/{notebook_id}").json()
    first = view["cells"][0]["id"]
    http.put(
        f"/api/notebooks/{notebook_id}/cells/{first}",
        json={"source": "total = sum(r['x'] for r in data['nums'])\ntotal"},
    )
    second = http.post(
        f"/api/notebooks/{notebook_id}/cells",
        json={"source": "doubled = total * 2\ndoubled"},
    ).json()["id"]

    # Running the first cell reactively cascades to the dependent second cell.
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [first]})
    )
    started = [e["cell"] for e in events if e["event"] == "cell_start"]
    assert started == [first, second]
    results = {
        e["cell"]: e["output"]["data"]["text/plain"]
        for e in events
        if e["event"] == "output" and e["output"]["output_type"] == "execute_result"
    }
    assert results[first] == "6"
    assert results[second] == "12"


def test_cells_can_be_reordered_and_deleted(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """Two routes the editor calls on every drag and every delete, with no test.

    Order is the document, so getting it wrong silently reorders someone's notebook.
    """
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "Order"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    second = http.post(
        f"/api/notebooks/{notebook_id}/cells", json={"source": "b = 2"}
    ).json()["id"]
    third = http.post(
        f"/api/notebooks/{notebook_id}/cells", json={"source": "c = 3"}
    ).json()["id"]

    def order() -> list[str]:
        return [
            c["id"] for c in http.get(f"/api/notebooks/{notebook_id}").json()["cells"]
        ]

    assert order() == [first, second, third]
    reordered = http.post(
        f"/api/notebooks/{notebook_id}/cells/reorder",
        json={"order": [third, first, second]},
    )
    assert reordered.status_code == 200, reordered.text
    assert order() == [third, first, second]

    deleted = http.delete(f"/api/notebooks/{notebook_id}/cells/{first}")
    assert deleted.status_code == 200, deleted.text
    assert order() == [third, second]


def test_variables_report_what_the_kernel_holds(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """The variables pane. Empty before a run, and naming the user's own names after."""
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "Vars"}).json()["id"]
    cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    assert http.get(f"/api/notebooks/{notebook_id}/variables").json() == []

    http.put(
        f"/api/notebooks/{notebook_id}/cells/{cell}",
        json={"source": "rows = [1, 2, 3]"},
    )
    _sse(http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]}))
    names = {
        v["name"] for v in http.get(f"/api/notebooks/{notebook_id}/variables").json()
    }
    assert "rows" in names


def test_relock_reports_success_and_names_a_missing_notebook(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """Re-resolving reports through the body, not the status code.

    A caller that only checked the status would treat "notebook not found" as a fresh
    lock, so the shape of both answers is the contract worth pinning.
    """
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "Lock"}).json()["id"]

    relocked = http.post(f"/api/notebooks/{notebook_id}/relock")
    assert relocked.status_code == 200, relocked.text
    assert relocked.json() == {"ok": True, "lock": []}

    missing = http.post("/api/notebooks/absent/relock")
    assert missing.status_code == 200
    assert missing.json() == {"ok": False, "error": "notebook not found"}


def test_interrupting_an_idle_notebook_is_harmless(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """Interrupt is a button a person mashes, so it has to be safe when nothing runs."""
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "Idle"}).json()["id"]
    assert http.post(f"/api/notebooks/{notebook_id}/interrupt").status_code == 200
    assert http.get(f"/api/notebooks/{notebook_id}").status_code == 200


def test_opening_a_derivation_as_a_notebook_over_http(tmp_path: Path) -> None:
    """The route, as against the service call the test further down makes.

    A missing derivation is a 404 rather than an empty notebook.
    """
    from elbi.db import Derivation

    store = open_store(f"sqlite:{tmp_path / 'fromderiv.db'}")
    store.save_derivation(
        Derivation(name="avg_x", source="def avg_x(ctx):\n    return []", question="?")
    )
    service = NotebookService(store=store, load_datasets=lambda: DATASETS)
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        notebook_service=service,
    )
    try:
        with TestClient(app) as http:
            assert http.post("/api/notebooks/from-derivation/absent").status_code == 404

            created = http.post("/api/notebooks/from-derivation/avg_x")
            assert created.status_code == 200, created.text
            cells = http.get(f"/api/notebooks/{created.json()['id']}").json()["cells"]
            assert cells[0]["cell_type"] == "markdown"
            assert "def avg_x(ctx)" in cells[1]["source"]
    finally:
        service.close()


def test_edit_marks_downstream_stale(client: tuple[TestClient, dict[str, str]]) -> None:
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{first}", json={"source": "a = 1"})
    second = http.post(
        f"/api/notebooks/{notebook_id}/cells", json={"source": "b = a + 1"}
    ).json()["id"]
    _sse(http.post(f"/api/notebooks/{notebook_id}/run", json={"run_all": True}))

    # Edit the upstream cell but do not run it, then run the downstream cell. The edited
    # cell is now stale (its source changed since it ran), and the downstream cell is
    # stale too because its input (the edited cell) is out of date.
    http.put(f"/api/notebooks/{notebook_id}/cells/{first}", json={"source": "a = 100"})
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [second]})
    )
    stale = next(e["cells"] for e in events if e["event"] == "stale")
    assert set(stale) == {first, second}


def test_parameterized_run_overrides_defaults(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(
        f"/api/notebooks/{notebook_id}/cells/{first}",
        json={"source": "alpha = 0.1", "metadata": {"tags": ["parameters"]}},
    )
    http.post(
        f"/api/notebooks/{notebook_id}/cells",
        json={"source": "scaled = alpha * 100\nscaled"},
    )
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"params": {"alpha": 0.5}})
    )
    results = [
        e["output"]["data"]["text/plain"]
        for e in events
        if e["event"] == "output" and e["output"]["output_type"] == "execute_result"
    ]
    assert "50.0" in results


def test_export_is_valid_nbformat(client: tuple[TestClient, dict[str, str]]) -> None:
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    payload = http.get(f"/api/notebooks/{notebook_id}/export").json()
    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] == 5
    assert isinstance(payload["cells"], list)


def _notebook_with_a_real_output(http: TestClient) -> tuple[str, str]:
    """A notebook whose one cell has actually run, so it holds a stored output."""
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    cell_id = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(
        f"/api/notebooks/{notebook_id}/cells/{cell_id}",
        json={"source": "secret_revenue = 1111 * 1111\nsecret_revenue"},
    )
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"run_all": True})
    )
    rendered = [
        e["output"]["data"]["text/plain"]
        for e in events
        if e["event"] == "output" and e["output"]["output_type"] == "execute_result"
    ]
    assert "1234321" in rendered, "the cell must really have produced an output"
    return notebook_id, cell_id


def _outputs_in(payload: dict[str, Any]) -> list[Any]:
    """Every output across a payload's code cells."""
    return [o for cell in payload["cells"] for o in cell.get("outputs", [])]


def test_export_strips_outputs_by_default(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """A cell output is a rendering of warehouse rows, and this is the route pull reads.

    So the default must not carry them off the deployment. nbstripout exists for this,
    naming sensitive information among its reasons; Databricks refuses to commit .ipynb
    output from a Git folder until an administrator enables it.
    """
    http, _ = client
    notebook_id, _ = _notebook_with_a_real_output(http)
    payload = http.get(f"/api/notebooks/{notebook_id}/export").json()
    assert _outputs_in(payload) == []
    # the rendered value appears nowhere: it is not a literal in the source, so finding
    # it in the file could only mean an output came along
    assert "1234321" not in json.dumps(payload)
    # and the source is still there: stripping outputs is not stripping the notebook
    assert "secret_revenue" in json.dumps(payload)


def test_asking_for_outputs_is_not_enough(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """Whether data may leave is the deployment's decision, not the caller's."""
    http, _ = client
    notebook_id, _ = _notebook_with_a_real_output(http)
    payload = http.get(
        f"/api/notebooks/{notebook_id}/export", params={"outputs": "true"}
    ).json()
    assert _outputs_in(payload) == []
    assert "1234321" not in json.dumps(payload)


def test_a_deployment_can_permit_outputs(
    client: tuple[TestClient, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permitted and asked for: someone downloading their own work, deliberately."""
    http, _ = client
    notebook_id, _ = _notebook_with_a_real_output(http)
    monkeypatch.setenv("NOTEBOOK_EXPORT_OUTPUTS", "1")
    payload = http.get(
        f"/api/notebooks/{notebook_id}/export", params={"outputs": "true"}
    ).json()
    assert _outputs_in(payload) != []
    assert "1234321" in json.dumps(payload)
    # and permission alone still does not volunteer them
    default = http.get(f"/api/notebooks/{notebook_id}/export").json()
    assert _outputs_in(default) == []


def test_a_stripped_export_still_round_trips(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    """What pull writes must be what sync can push back, or the repo form is useless."""
    http, _ = client
    notebook_id, _ = _notebook_with_a_real_output(http)
    payload = http.get(f"/api/notebooks/{notebook_id}/export").json()
    created = http.post(
        "/api/notebooks/import", json={"name": "round-tripped", "ipynb": payload}
    )
    assert created.status_code == 200, created.text
    reimported = http.get(f"/api/notebooks/{created.json()['id']}").json()
    assert any("secret_revenue" in c["source"] for c in reimported["cells"])


def test_promote_cell_authors_the_function(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, promoted = client
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(
        f"/api/notebooks/{notebook_id}/cells/{first}",
        json={"source": "def avg_x(ctx):\n    return [{'avg': 2}]"},
    )
    result = http.post(f"/api/notebooks/{notebook_id}/cells/{first}/promote").json()
    assert result["certified"] is True
    assert result["name"] == "avg_x"
    assert promoted["name"] == "avg_x"  # the derivation name came from the function


def test_promote_rejects_a_non_derivation_cell(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    notebook_id = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{first}", json={"source": "x = 1"})
    result = http.post(f"/api/notebooks/{notebook_id}/cells/{first}/promote").json()
    assert result["ok"] is False
    assert "ctx" in result["error"]


# Environment management. The resolver is stubbed so these run offline and
# deterministically; a real ``uv pip compile`` is covered by the SDK env tests.
@pytest.fixture
def env_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    import elbi.notebooks as notebooks

    # Fake lock: pretend each spec resolves to a pinned version, so composition and
    # persistence are exercised without hitting the network.
    monkeypatch.setattr(
        notebooks,
        "resolve_lock",
        lambda deps, **_: [f"{str(d).split('==')[0]}==9.9" for d in deps],
    )
    store = open_store(f"sqlite:{tmp_path / 'env.db'}")
    store.set_config(
        notebooks.BASE_ENV_SETTING,
        json.dumps([{"name": "DS", "deps": ["numpy"]}]),
    )
    service = notebooks.NotebookService(
        store=store, load_datasets=lambda: DATASETS, profiles=_profiles(max_runtime=15)
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        notebook_service=service,
    )
    with TestClient(app) as http:
        yield http
    service.close()


def test_set_environment_composes_base_and_locks(env_client: TestClient) -> None:
    http = env_client
    notebook_id = http.post("/api/notebooks", json={"name": "E"}).json()["id"]
    result = http.put(
        f"/api/notebooks/{notebook_id}", json={"deps": ["pandas"], "base_env": "DS"}
    ).json()
    assert result["ok"] is True
    env = http.get(f"/api/notebooks/{notebook_id}").json()["environment"]
    assert env["base_env"] == "DS"
    assert env["locked"] is True
    # The lock is the base environment's package plus the notebook's own.
    assert "numpy==9.9" in env["lock"]
    assert "pandas==9.9" in env["lock"]


def test_pip_install_cell_adds_to_the_environment(env_client: TestClient) -> None:
    http = env_client
    notebook_id = http.post("/api/notebooks", json={"name": "E"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(
        f"/api/notebooks/{notebook_id}/cells/{first}",
        json={"source": "%pip install toml"},
    )
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [first]})
    )
    assert any(e["event"] == "installing" for e in events)
    assert any(e["event"] == "installed" for e in events)
    assert "toml" in http.get(f"/api/notebooks/{notebook_id}").json()["deps"]


def test_base_environments_admin_api(env_client: TestClient) -> None:
    http = env_client
    http.put(
        "/api/settings/notebook-environments",
        json={"environments": [{"name": "ML", "deps": ["scikit-learn"]}]},
    )
    envs = http.get("/api/settings/notebook-environments").json()
    assert {e["name"] for e in envs} == {"ML"}
    assert envs[0]["deps"] == ["scikit-learn"]


def test_notebook_from_model_requires_the_ml_extra(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    # The test app has no model service, so the training-run bridge degrades.
    http, _ = client
    assert http.post("/api/notebooks/from-model/anything").status_code == 503


def test_create_from_derivation_scaffolds_a_notebook(tmp_path: Path) -> None:
    from elbi.db import Derivation

    store = open_store(f"sqlite:{tmp_path / 'd.db'}")
    store.save_derivation(
        Derivation(name="avg_x", source="def avg_x(ctx):\n    return []", question="?")
    )
    service = NotebookService(store=store, load_datasets=lambda: {})
    try:
        notebook_id = service.create_from_derivation("avg_x")
        assert notebook_id is not None
        cells = service.view(notebook_id)["cells"]
        assert cells[0]["cell_type"] == "markdown"
        assert cells[1]["cell_type"] == "code"
        assert "def avg_x(ctx)" in cells[1]["source"]
    finally:
        service.close()


def test_folder_crud_over_http(client: tuple[TestClient, dict[str, str]]) -> None:
    http, _ = client
    root = http.post("/api/notebooks/folders", json={"name": "Marketing"}).json()["id"]
    child = http.post(
        "/api/notebooks/folders", json={"name": "Q3", "parent_id": root}
    ).json()["id"]

    folders = http.get("/api/notebooks/folders").json()
    assert {f["name"]: f["parent_id"] for f in folders} == {
        "Marketing": None,
        "Q3": root,
    }

    # A notebook created in a folder reports that folder in its summary.
    created = http.post("/api/notebooks", json={"name": "Report", "folder_id": child})
    nb = created.json()["id"]
    summary = next(n for n in http.get("/api/notebooks").json() if n["id"] == nb)
    assert summary["folder_id"] == child

    # Rename, then move the notebook up to the root.
    renamed = http.put(f"/api/notebooks/folders/{child}", json={"name": "Q4"})
    assert renamed.status_code == 200
    moved = http.post(f"/api/notebooks/{nb}/move", json={"folder_id": None})
    assert moved.status_code == 200
    summary = next(n for n in http.get("/api/notebooks").json() if n["id"] == nb)
    assert summary["folder_id"] is None


def test_move_into_descendant_is_conflict(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    a = http.post("/api/notebooks/folders", json={"name": "A"}).json()["id"]
    b = http.post("/api/notebooks/folders", json={"name": "B", "parent_id": a}).json()[
        "id"
    ]
    resp = http.put(f"/api/notebooks/folders/{a}", json={"parent_id": b})
    assert resp.status_code == 409


def test_delete_nonempty_folder_needs_recursive(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    folder = http.post("/api/notebooks/folders", json={"name": "Keep"}).json()["id"]
    http.post("/api/notebooks", json={"name": "Inside", "folder_id": folder})

    assert http.delete(f"/api/notebooks/folders/{folder}").status_code == 409
    assert http.get("/api/notebooks/folders").json()  # still there

    recursive = http.delete(f"/api/notebooks/folders/{folder}?recursive=true")
    assert recursive.status_code == 200
    assert http.get("/api/notebooks/folders").json() == []
    assert http.get("/api/notebooks").json() == []


def test_unknown_folder_on_create_is_404(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    assert (
        http.post("/api/notebooks", json={"name": "x", "folder_id": "nope"}).status_code
        == 404
    )


def test_complete_and_inspect_over_http(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    nb = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    first = http.get(f"/api/notebooks/{nb}").json()["cells"][0]["id"]
    http.put(
        f"/api/notebooks/{nb}/cells/{first}",
        json={"source": "import math\nvalue = 5"},
    )
    _sse(http.post(f"/api/notebooks/{nb}/run", json={"cells": [first]}))

    completion = http.post(
        f"/api/notebooks/{nb}/complete", json={"code": "val", "cursor_pos": 3}
    ).json()
    assert "value" in completion["matches"]

    inspection = http.post(
        f"/api/notebooks/{nb}/inspect", json={"code": "math.sqrt", "cursor_pos": 6}
    ).json()
    assert inspection["found"] is True
    assert "sqrt" in inspection["data"]["text/plain"]


def test_completion_is_empty_without_a_running_kernel(
    client: tuple[TestClient, dict[str, str]],
) -> None:
    http, _ = client
    nb = http.post("/api/notebooks", json={"name": "T"}).json()["id"]
    completion = http.post(
        f"/api/notebooks/{nb}/complete", json={"code": "va", "cursor_pos": 2}
    ).json()
    assert completion["matches"] == []
    # No kernel means nothing to deliver input to.
    assert http.post(f"/api/notebooks/{nb}/input", json={"value": "x"}).json() == {
        "ok": False
    }


@pytest.fixture
def lazy_client(tmp_path: Path) -> Iterator[tuple[TestClient, list[dict[str, Any]]]]:
    """A notebook service on the lazy data path, recording what the kernel asked for.

    The resolver stands in for the warehouse: it answers a table read by name and a
    ``sql`` request with a trivial aggregate, which is enough to show the app wiring
    carries both and that nothing is fetched before a cell asks.
    """
    store = open_store(f"sqlite:{tmp_path / 'lazy.db'}")
    asked: list[dict[str, Any]] = []

    def resolver(sql: str, table: str, limit: int) -> dict[str, Any]:
        asked.append({"sql": sql, "table": table, "limit": limit})
        if table:
            rows = DATASETS.get(table)
            if rows is None:
                return {"error": f"no such table {table!r}"}
            return {"columns": ["x"], "rows": rows, "truncated": False}
        return {"columns": ["total"], "rows": [{"total": 6}], "truncated": False}

    service = NotebookService(
        store=store,
        load_datasets=lambda: (_ for _ in ()).throw(  # must never be called
            AssertionError("the lazy path must not load every dataset")
        ),
        dataset_names=lambda: list(DATASETS),
        query_resolver=resolver,
        profiles=_profiles(max_runtime=15),
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        notebook_service=service,
    )
    with TestClient(app) as http:
        yield http, asked
    service.close()


def _run_cell(http: TestClient, notebook_id: str, cell: str, source: str) -> list[Any]:
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": source})
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]})
    )
    return [e for e in events if e["event"] == "output"]


def test_lazy_path_defers_fetches_and_serves_sql(
    lazy_client: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    http, asked = lazy_client
    notebook_id = http.post("/api/notebooks", json={"name": "L"}).json()["id"]
    first = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]

    # Starting the kernel and listing the bound datasets must fetch nothing: the whole
    # point: kernel start no longer costs whatever the bound data happens to weigh.
    outputs = _run_cell(http, notebook_id, first, "sorted(data)")
    assert any("nums" in json.dumps(o) for o in outputs)
    assert asked == []

    # data['nums'] fetches exactly that table, by name, never as generated SQL.
    outputs = _run_cell(http, notebook_id, first, "sum(r['x'] for r in data['nums'])")
    assert any("6" in json.dumps(o) for o in outputs)
    assert [a["table"] for a in asked] == ["nums"]
    assert all(a["sql"] == "" for a in asked)

    # sql() reaches the resolver as SQL, and the result comes back bounded.
    asked.clear()
    outputs = _run_cell(
        http, notebook_id, first, "sql('select sum(x) total from nums')[0]['total']"
    )
    assert any("6" in json.dumps(o) for o in outputs)
    assert len(asked) == 1
    assert asked[0]["table"] == "" and "select sum(x)" in asked[0]["sql"]


# -- compute profiles -----------------------------------------------------------


def test_the_menu_is_offered_but_not_editable(
    compute_client: tuple[TestClient, Any],
) -> None:
    http, _ = compute_client
    body = http.get("/api/compute/profiles").json()
    assert body["default"] == "small"
    names = [p["name"] for p in body["profiles"]]
    assert names == ["small", "large", "gpu"]
    small = body["profiles"][0]
    # The cost estimate is what a governance decision is made against, so it travels
    # with the profile rather than being recomputed by every caller.
    assert small["costPerHour"] > 0
    assert small["version"]
    # Infrastructure defines the menu. There is no route to change it, because a field
    # the UI offered but could not save would be worse than its absence.
    assert http.put("/api/compute/profiles", json={}).status_code == 405


def test_a_notebook_carries_its_profile(compute_client: tuple[TestClient, Any]) -> None:
    http, _ = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    assert (
        http.get(f"/api/compute/notebooks/{notebook_id}").json()["profile"]["name"]
        == "small"
    )

    assert (
        http.put(
            f"/api/notebooks/{notebook_id}", json={"compute_profile": "large"}
        ).status_code
        == 200
    )
    state = http.get(f"/api/compute/notebooks/{notebook_id}").json()
    assert state["profile"]["name"] == "large"
    assert state["profile"]["memory"] == "32Gi"
    assert state["status"] == "idle"

    # A name that is not on the menu is refused while the person is still looking at the
    # picker, rather than at kernel start.
    bad = http.put(f"/api/notebooks/{notebook_id}", json={"compute_profile": "huge"})
    assert bad.status_code == 400
    assert "unknown compute profile" in bad.json()["detail"]


def test_a_running_kernel_keeps_the_definition_it_started_with(
    compute_client: tuple[TestClient, Any],
) -> None:
    # Databricks' compliance idea: after a policy changes, compute created under it
    # "aren't automatically updated". An admin who tightened a limit needs telling.
    http, (service, _) = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": "1 + 1"})
    _sse(http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]}))

    state = http.get(f"/api/compute/notebooks/{notebook_id}").json()
    assert state["status"] == "running"
    assert state["drift"] is None

    # Resize the profile underneath the live kernel.
    service._profiles = ComputeProfiles(
        profiles=(ComputeProfile(name="small", cpu="2", memory="1Gi", max_runtime=15),),
        default="small",
    )
    drifted = http.get(f"/api/compute/notebooks/{notebook_id}").json()
    assert "Restart the kernel" in drifted["drift"]
    assert drifted["startedWith"]["memory"] == "4Gi"
    assert drifted["profile"]["memory"] == "1Gi"

    # Restarting is what applies it, rather than the change killing live work.
    http.post(f"/api/notebooks/{notebook_id}/restart")
    assert http.get(f"/api/compute/notebooks/{notebook_id}").json()["drift"] is None


def test_a_finished_session_is_attributed(
    compute_client: tuple[TestClient, Any],
) -> None:
    # Usage history cannot be backfilled, so it is recorded from the first session.
    http, (_, recorded) = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": "1 + 1"})
    _sse(http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]}))
    http.post(f"/api/notebooks/{notebook_id}/restart")
    assert [(nb, profile) for nb, profile, _, _ in recorded] == [(notebook_id, "small")]
    assert recorded[0][2] >= 0


def test_a_profile_that_disappeared_is_reported_not_absorbed(
    compute_client: tuple[TestClient, Any],
) -> None:
    # Falling back keeps the notebook openable, but silently running something other
    # than what it asked for is how a resized limit goes unnoticed.
    http, (service, _) = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    http.put(f"/api/notebooks/{notebook_id}", json={"compute_profile": "large"})

    service._profiles = ComputeProfiles(
        profiles=(ComputeProfile(name="small", max_runtime=15),), default="small"
    )
    state = http.get(f"/api/compute/notebooks/{notebook_id}").json()
    assert state["profile"]["name"] == "small"
    assert "unknown compute profile 'large'" in state["unavailable"]


def test_changing_what_a_notebook_runs_on_is_audited(
    compute_client: tuple[TestClient, Any],
) -> None:
    # Anyone who can edit a notebook can change its size, and "who moved this onto the
    # GPU tier" is a question with a cost attached. The usage record says what ran; the
    # audit entry says who decided it would.
    http, _ = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    http.put(f"/api/notebooks/{notebook_id}", json={"compute_profile": "large"})
    events = http.get("/api/audit").json()
    entries = [e for e in events if e["action"] == "notebook.compute_profile"]
    assert entries and entries[0]["targetId"] == notebook_id
    assert entries[0]["verdict"] == "large"


# -- warm pool ------------------------------------------------------------------


def _pool_service(tmp_path: Path, size: int) -> Any:
    """A service whose single profile pre-warms ``size`` kernels."""
    import elbi.notebooks as notebooks

    store = open_store(f"sqlite:{tmp_path / 'warm.db'}")
    return notebooks.NotebookService(
        store=store,
        load_datasets=lambda: DATASETS,
        profiles=ComputeProfiles(
            profiles=(
                ComputeProfile(name="warm", warm_pool_size=size, max_runtime=15),
            ),
            default="warm",
        ),
    )


def test_a_pooled_kernel_serves_the_next_session(tmp_path: Path) -> None:
    import time as clock

    service = _pool_service(tmp_path, 2)
    try:
        profile = service.profiles.get("warm")
        # Nothing is warm until something asks, so a deployment that never opens a
        # notebook never pays for the pool.
        assert service._warm.depth() == {}

        assert service._warm.take(profile, ()) is None  # first ask, pool empty
        for _ in range(100):
            if service._warm.depth().get("warm") == 2:
                break
            clock.sleep(0.05)
        assert service._warm.depth() == {"warm": 2}

        taken = service._warm.take(profile, ())
        assert taken is not None and taken.alive
        taken.close()
    finally:
        service.close()


def test_a_resized_profile_invalidates_what_is_waiting(tmp_path: Path) -> None:
    # A pooled kernel was built to a definition. Handing one out after the definition
    # changed would apply the old size while reporting the new one, which is the exact
    # drift the profile version exists to prevent.
    import time as clock

    service = _pool_service(tmp_path, 1)
    try:
        profile = service.profiles.get("warm")
        service._warm.take(profile, ())
        for _ in range(100):
            if service._warm.depth().get("warm") == 1:
                break
            clock.sleep(0.05)

        resized = ComputeProfile(name="warm", memory="1Gi", warm_pool_size=1)
        assert resized.version != profile.version
        assert service._warm.take(resized, ()) is None
    finally:
        service.close()


def test_a_different_dependency_set_is_a_different_pool(tmp_path: Path) -> None:
    service = _pool_service(tmp_path, 1)
    try:
        profile = service.profiles.get("warm")
        service._warm.take(profile, ())
        # A kernel's environment is fixed at start, so one warmed without dependencies
        # cannot serve a notebook that declared some.
        assert service._warm.take(profile, ("polars==1.0.0",)) is None
    finally:
        service.close()


def test_pooling_is_off_unless_a_profile_asks(tmp_path: Path) -> None:
    service = _pool_service(tmp_path, 0)
    try:
        profile = service.profiles.get("warm")
        assert service._warm.take(profile, ()) is None
        assert service._warm.depth() == {}
    finally:
        service.close()


def test_a_notebook_may_shorten_its_idle_timeout_but_not_extend_it(
    compute_client: tuple[TestClient, Any],
) -> None:
    # Holding a kernel is a cost someone else pays, so the profile is the cap and a
    # notebook's own setting can only tighten it.
    http, (service, _) = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    cap = service.profiles.get("small").idle_timeout

    state = http.get(f"/api/compute/notebooks/{notebook_id}").json()
    assert state["idleTimeout"] == cap
    assert state["idleTimeoutCap"] == cap

    http.put(f"/api/notebooks/{notebook_id}", json={"metadata": {"idle_timeout": 300}})
    assert (
        http.get(f"/api/compute/notebooks/{notebook_id}").json()["idleTimeout"] == 300
    )

    # Asking for more than the profile allows gets the profile's value, not the request.
    http.put(
        f"/api/notebooks/{notebook_id}", json={"metadata": {"idle_timeout": cap * 10}}
    )
    assert (
        http.get(f"/api/compute/notebooks/{notebook_id}").json()["idleTimeout"] == cap
    )

    # Nonsense is ignored rather than turned into a zero-second timeout that reaps a
    # kernel the moment it starts.
    http.put(
        f"/api/notebooks/{notebook_id}", json={"metadata": {"idle_timeout": "soon"}}
    )
    assert (
        http.get(f"/api/compute/notebooks/{notebook_id}").json()["idleTimeout"] == cap
    )


def test_a_scheduled_run_always_enforces_the_current_profile(
    compute_client: tuple[TestClient, Any],
) -> None:
    """Batch enforces at once, because nothing is watching it.

    D16's rule: an interactive session keeps the definition it started with, so an admin
    resizing a profile does not destroy work in progress. A scheduled run has no such
    claim on anyone's attention, so it starts fresh and therefore always picks up the
    current definition.
    """
    http, (service, _) = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": "1 + 1"})
    _sse(http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]}))

    started_with = service._live[notebook_id].profile_version
    service._profiles = ComputeProfiles(
        profiles=(ComputeProfile(name="small", memory="1Gi", max_runtime=15),),
        default="small",
    )
    assert http.get(f"/api/compute/notebooks/{notebook_id}").json()["drift"]

    service.run_scheduled(notebook_id, {})
    # The kernel it ran on is the new definition, not the one the session held.
    assert service._live[notebook_id].profile_version != started_with
    assert http.get(f"/api/compute/notebooks/{notebook_id}").json()["drift"] is None


def test_an_environment_past_end_of_support_stops_taking_new_work(
    env_client: TestClient,
) -> None:
    """A lock says what you got; it does not say when it stops being maintained.

    Databricks pairs each serverless environment version with a dated end of support
    and, past it, stops offering that version for new work while existing workloads keep
    running. The asymmetry is the mechanism: expiry has to be plannable, not a removal.
    """
    http = env_client
    http.put(
        "/api/settings/notebook-environments",
        json={
            "environments": [
                {"name": "DS", "deps": ["numpy"], "version": "1.0"},
                {
                    "name": "Legacy",
                    "deps": ["numpy"],
                    "version": "0.9",
                    "end_of_support": "2020-01-01",
                },
            ]
        },
    )
    # Only the supported one is offered for new work.
    offered = http.get("/api/settings/notebook-environments").json()
    assert [e["name"] for e in offered] == ["DS"]
    # An admin still has to be able to see and manage the expired one.
    everything = http.get("/api/settings/notebook-environments?all=true").json()
    assert [(e["name"], e["expired"]) for e in everything] == [
        ("DS", False),
        ("Legacy", True),
    ]

    notebook_id = http.post("/api/notebooks", json={"name": "E"}).json()["id"]
    refused = http.put(
        f"/api/notebooks/{notebook_id}", json={"deps": [], "base_env": "Legacy"}
    ).json()
    assert refused["ok"] is False
    assert "end of support on 2020-01-01" in refused["error"]

    # A notebook already on it keeps its packages: expiry is not removal.
    http.put(f"/api/notebooks/{notebook_id}", json={"deps": [], "base_env": "DS"})
    assert (
        http.get(f"/api/notebooks/{notebook_id}").json()["environment"]["base_env"]
        == "DS"
    )


def test_a_malformed_support_date_is_refused_not_treated_as_expiry(
    env_client: TestClient,
) -> None:
    # Treating an unparseable date as an expiry would take an environment out of service
    # over a typo; accepting it silently would look like no support window at all.
    bad = env_client.put(
        "/api/settings/notebook-environments",
        json={"environments": [{"name": "DS", "deps": [], "end_of_support": "soon"}]},
    )
    assert bad.status_code == 400
    assert "must be a date" in bad.json()["detail"]


def test_a_shared_profile_puts_two_notebooks_in_one_kernel(tmp_path: Path) -> None:
    """Opt-in, and only worth it where nothing sensitive is in the process.

    One process holds one identity, so a shared kernel cannot carry per-notebook
    credentials and its namespace is reachable by everyone using it, which is why this
    is a profile flag and not a default. Databricks made their equivalent legacy and off
    by default for the same reason.
    """
    import elbi.notebooks as notebooks

    store = open_store(f"sqlite:{tmp_path / 'shared.db'}")
    service = notebooks.NotebookService(
        store=store,
        load_datasets=lambda: DATASETS,
        profiles=ComputeProfiles(
            profiles=(
                ComputeProfile(name="scratch", shared=True, max_runtime=15),
                ComputeProfile(name="own", max_runtime=15),
            ),
            default="scratch",
        ),
    )
    try:
        first = store.create_notebook(name="A")
        second = store.create_notebook(name="B")
        # Two notebooks, one interpreter: that is the whole point, and the hazard.
        runtime_a = service._runtime(first, ())
        runtime_b = service._runtime(second, ())
        assert runtime_a is runtime_b
        assert list(service._live) == ["profile:scratch"]

        # An unshared profile keeps them apart, and every lookup agrees about which key
        # to use; a restart must not kill a kernel an interrupt could not find.
        store.update_notebook(second, compute_profile="own")
        runtime_c = service._runtime(second, ())
        assert runtime_c is not runtime_a
        assert service.kernel_status(second) == "running"
        service.restart(second)
        assert service.kernel_status(second) == "idle"
        assert service.kernel_status(first) == "running"
    finally:
        service.close()


def test_a_scheduled_run_is_attributed_separately_from_a_person(
    compute_client: tuple[TestClient, Any],
) -> None:
    # Databricks bills all-purpose and job compute to different SKUs and does "not
    # recommend" running production jobs on all-purpose compute. Without the split, a
    # scheduled 3am rerun and an abandoned session look identical in a cost report and
    # only one of them is worth chasing.
    http, (service, recorded) = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": "1 + 1"})

    _sse(http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]}))
    service.run_scheduled(notebook_id, {})
    service.restart(notebook_id)

    kinds = [kind for *_rest, kind in recorded]
    assert kinds == ["interactive", "batch"]


def test_the_queries_a_cell_pushed_down_reach_the_editor(
    compute_client: tuple[TestClient, Any],
) -> None:
    http, _ = compute_client
    notebook_id = http.post("/api/notebooks", json={"name": "C"}).json()["id"]
    cell = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]["id"]
    http.put(f"/api/notebooks/{notebook_id}/cells/{cell}", json={"source": "1 + 1"})
    events = _sse(
        http.post(f"/api/notebooks/{notebook_id}/run", json={"cells": [cell]})
    )

    # No warehouse wired in this fixture, so no queries -- but the field is present on
    # the status event and persisted, which is what the editor reads.
    status = [e for e in events if e.get("event") == "status"]
    assert status and status[0]["queries"] == []
    saved = http.get(f"/api/notebooks/{notebook_id}").json()["cells"][0]
    assert "queries" not in (saved["metadata"].get("elbi") or {})


def test_a_derivation_and_its_upstream_share_one_preamble(tmp_path: Path) -> None:
    """The context a module gave them is hoisted once, not repeated per cell.

    Each stored source is self-contained — it carries the imports and constants its
    module gave it — so seeding a derivation beside its upstream would otherwise print
    the same import block in every cell.
    """
    from elbi.db import Derivation

    store = open_store(f"sqlite:{tmp_path / 'd.db'}")
    context = "from collections import defaultdict\nLIMIT = 3\n\n"
    store.save_derivation(
        Derivation(
            name="upstream",
            source=context + "@derivation()\ndef upstream(ctx):\n    return LIMIT",
            question="?",
        )
    )
    store.save_derivation(
        Derivation(
            name="downstream",
            source=context
            + (
                "@derivation(inputs={'u': upstream})\n"
                "def downstream(ctx):\n    return LIMIT"
            ),
            question="?",
        )
    )
    service = NotebookService(store=store, load_datasets=lambda: {})
    try:
        notebook_id = service.create_from_derivation("downstream")
        sources = [c["source"] for c in service.view(notebook_id)["cells"]]
        joined = "\n".join(sources)

        assert joined.count("from collections import defaultdict") == 1
        assert joined.count("LIMIT = 3") == 1
        # And the upstream is still defined before the cell that reads it.
        up = next(i for i, s in enumerate(sources) if "def upstream" in s)
        down = next(i for i, s in enumerate(sources) if "def downstream" in s)
        assert up < down
    finally:
        service.close()


def test_derivations_that_reference_each_other_still_terminate(tmp_path: Path) -> None:
    """A cycle in the store must not seed a cell per hop."""
    from elbi.db import Derivation

    store = open_store(f"sqlite:{tmp_path / 'd.db'}")
    store.save_derivation(
        Derivation(
            name="a",
            source="@derivation(inputs={'b': b})\ndef a(ctx): ...",
            question="?",
        )
    )
    store.save_derivation(
        Derivation(
            name="b",
            source="@derivation(inputs={'a': a})\ndef b(ctx): ...",
            question="?",
        )
    )
    service = NotebookService(store=store, load_datasets=lambda: {})
    try:
        notebook_id = service.create_from_derivation("a")
        assert len(service.view(notebook_id)["cells"]) < 10
    finally:
        service.close()


def test_a_derivation_notebook_ends_on_something_that_runs_it(tmp_path: Path) -> None:
    """Defining a derivation displays nothing, so the scaffold has to call it.

    A decorated `def` is a statement: a notebook has no value to echo, which leaves
    someone editing a derivation unable to see what their edit did.
    """
    from elbi.db import Derivation

    store = open_store(f"sqlite:{tmp_path / 'd.db'}")
    store.save_derivation(
        Derivation(
            name="totals",
            source="@derivation()\ndef totals(ctx):\n    return Artifact.table([])",
            question="?",
        )
    )
    service = NotebookService(store=store, load_datasets=lambda: {})
    try:
        notebook_id = service.create_from_derivation("totals")
        last = service.view(notebook_id)["cells"][-1]["source"]

        assert last.rstrip().endswith("run(totals)")
        # It must resolve inputs the way the runner does, not assume there are none.
        assert "isinstance(spec, Dataset)" in last
        assert "Table(data[spec.name])" in last
    finally:
        service.close()
