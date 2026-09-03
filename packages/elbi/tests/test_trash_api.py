"""Integration tests for soft delete, trash, restore, and immediate erasure.

Drives the HTTP surface with a TestClient, the boundary a browser uses, so these cover
the whole path: a DELETE that trashes by default, the /api/trash listing and its
restore/erase actions, and the ``?permanent=true`` bypass.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import Store, open_store
from elbi.explore import ExploreService
from elbi.notebooks import NotebookService
from elbi.orchestration import OrchestrationService
from elbi_core import Registry, Runner
from elbi_core.sandbox import ComputeProfile, ComputeProfiles


class _NoClient:
    def step(self, transcript: object, tools: object) -> object:
        raise NotImplementedError


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'trash.db'}")


def _client(
    store: Store, *, explore_service: ExploreService | None = None
) -> TestClient:
    registry = Registry()
    notebook_service = NotebookService(
        store=store,
        load_datasets=lambda: {"d": []},
        profiles=ComputeProfiles(
            profiles=(ComputeProfile(name="test", max_runtime=15),), default="test"
        ),
    )
    orchestration_service = OrchestrationService(
        store=store,
        registry_provider=lambda: registry,
        make_runner=lambda: Runner(registry),
    )
    return TestClient(
        create_app(
            load_datasets=lambda: {"d": []},
            client=_NoClient(),
            store=store,
            notebook_service=notebook_service,
            orchestration_service=orchestration_service,
            explore_service=explore_service,
            # Faithful stand-ins for serve.py's hooks: the same store calls,
            # minus registry/sidecar side effects these tests do not assert.
            derivation_on_trash=store.trash_derivation,
            derivation_on_restore=store.restore_derivation,
            derivation_on_erase=store.erase_derivation,
        )
    )


# -- notebook: trash, list, restore, intact ------------------------------------


def test_deleting_a_notebook_trashes_it_and_restore_brings_it_back(
    store: Store,
) -> None:
    notebook_id = store.create_notebook("Q3 plan")
    store.add_cell(notebook_id, cell_type="code", source="print(1)")
    with _client(store) as http:
        assert http.delete(f"/api/notebooks/{notebook_id}").status_code == 200

        # Gone from every normal surface.
        assert http.get("/api/notebooks").json() == []
        assert http.get(f"/api/notebooks/{notebook_id}").status_code == 404

        # But present in trash, with when.
        trash = http.get("/api/trash").json()
        assert len(trash) == 1
        entry = trash[0]
        assert entry["type"] == "notebook"
        assert entry["id"] == notebook_id
        assert entry["deletedAt"]

        # Restoring brings it back with its cells intact.
        restored = http.post(f"/api/trash/notebook/{notebook_id}/restore")
        assert restored.status_code == 200
        assert http.get("/api/trash").json() == []
        again = http.get(f"/api/notebooks/{notebook_id}")
        assert again.status_code == 200
        # create_notebook seeds one empty cell; add_cell added the second.
        cells = again.json()["cells"]
        assert len(cells) == 2
        assert any(c["source"] == "print(1)" for c in cells)


def test_permanent_query_param_bypasses_trash(store: Store) -> None:
    notebook_id = store.create_notebook("Scratch")
    with _client(store) as http:
        assert (
            http.delete(f"/api/notebooks/{notebook_id}?permanent=true").status_code
            == 200
        )
        assert http.get("/api/trash").json() == []
        # Nothing to restore: it never went to trash.
        assert (
            http.post(f"/api/trash/notebook/{notebook_id}/restore").status_code == 404
        )


def test_erase_from_the_trash_endpoint_is_permanent(store: Store) -> None:
    notebook_id = store.create_notebook("Old draft")
    with _client(store) as http:
        http.delete(f"/api/notebooks/{notebook_id}")
        assert http.delete(f"/api/trash/notebook/{notebook_id}").status_code == 200
        assert http.get("/api/trash").json() == []
        assert (
            http.post(f"/api/trash/notebook/{notebook_id}/restore").status_code == 404
        )


# -- folder subtree -------------------------------------------------------------


def test_trashing_a_folder_takes_its_notebooks_with_it_and_restore_brings_both_back(
    store: Store,
) -> None:
    folder_id = store.create_folder("Reports")
    notebook_id = store.create_notebook("Report A", folder_id=folder_id)
    with _client(store) as http:
        resp = http.delete(f"/api/notebooks/folders/{folder_id}?recursive=true")
        assert resp.status_code == 200
        assert http.get(f"/api/notebooks/{notebook_id}").status_code == 404
        trash = http.get("/api/trash").json()
        # Only the folder is a trash root; its notebook is represented by it.
        assert [item["type"] for item in trash] == ["folder"]

        assert http.post(f"/api/trash/folder/{folder_id}/restore").status_code == 200
        assert http.get(f"/api/notebooks/{notebook_id}").status_code == 200


def test_a_nonempty_folder_still_409s_without_recursive(store: Store) -> None:
    folder_id = store.create_folder("Reports")
    store.create_notebook("Report A", folder_id=folder_id)
    with _client(store) as http:
        assert http.delete(f"/api/notebooks/folders/{folder_id}").status_code == 409


# -- audit: trash, restore, and erase all emit events, with no content ----------


def test_trash_restore_and_erase_all_emit_audit_events_without_content(
    store: Store,
) -> None:
    notebook_id = store.create_notebook("Secret plan")
    with _client(store) as http:
        http.delete(f"/api/notebooks/{notebook_id}")
        http.post(f"/api/trash/notebook/{notebook_id}/restore")
        http.delete(f"/api/notebooks/{notebook_id}?permanent=true")

    events = store.list_audit()
    actions = [e.action for e in events]
    assert "notebook.trash" in actions
    assert "notebook.restore" in actions
    assert "notebook.erase" in actions
    for event in events:
        # Metadata only: no field on AuditEvent can carry a notebook's content, and
        # the ones that could carry a name/title do not.
        assert "Secret plan" not in (event.target_id or "")
        assert "Secret plan" not in (event.verdict or "")


# -- metric: name-keyed create-over-trashed -------------------------------------


def test_trashing_an_already_trashed_metric_stays_true_and_does_not_re_audit(
    store: Store,
) -> None:
    store.upsert_metric(name="mrr", manifest_json="{}", source=None)
    assert store.trash_metric("mrr") is True
    # Trashing it again is a no-op, not a failure -- it still exists for this
    # caller, so the contract ("did it exist for this caller") stays True.
    assert store.trash_metric("mrr") is True
    actions = [e.action for e in store.list_audit()]
    assert actions.count("metric.trash") == 1


def test_creating_a_metric_over_a_trashed_one_succeeds_and_erases_the_old_row(
    store: Store,
) -> None:
    store.upsert_metric(name="mrr", manifest_json="{}", source=None)
    store.trash_metric("mrr")
    # A fresh create with the same name is not blocked by the trashed row.
    store.upsert_metric(name="mrr", manifest_json='{"v": 2}', source=None)
    live = store.get_metric("mrr")
    assert live is not None and live.manifest_json == '{"v": 2}'
    # The trashed row is gone rather than sitting under the reused name.
    assert all(item["id"] != "mrr" for item in store.list_trash())


# -- promoted derivations: trash, restore, erase ---------------------------------


def _explore_with_stub_factory(store: Store) -> ExploreService:
    """An ExploreService whose derive factory persists like the real bridge.

    Mirrors authoring.py's certified path in one line, so the route -> service ->
    factory thread is exercised end to end without the sandbox.
    """
    from typing import Any

    from elbi.db import Derivation
    from elbi_agent import DeriveOutcome

    def factory(conversation_id: str, question: str) -> Any:
        def derive(*args: Any, **kwargs: Any) -> DeriveOutcome:
            name, source = args[0], args[1]
            store.save_derivation(
                Derivation(
                    name=name,
                    conversation_id=conversation_id,
                    question=question,
                    source=source,
                    verdict="sound",
                    rendered="ok",
                )
            )
            return DeriveOutcome(certified=True, verdict="sound", rendered="ok")

        return derive

    return ExploreService(
        store=store,
        dataset_names=lambda: ["sales"],
        resolve_source=lambda name: [{"x": 1}],
        read_schema=lambda name: (["x"], 1),
        derive_factory=factory,
    )


def test_a_promoted_derivation_can_be_trashed_restored_and_erased(store: Store) -> None:
    with _client(store, explore_service=_explore_with_stub_factory(store)) as http:
        promoted = http.post(
            "/api/explore/promote",
            json={"name": "sales_peek", "sql": "select * from sales"},
        )
        assert promoted.status_code == 200 and promoted.json()["ok"] is True
        assert store.get_derivation("sales_peek") is not None

        assert http.delete("/api/derivations/sales_peek").status_code == 200
        assert [e["id"] for e in http.get("/api/trash").json()] == ["sales_peek"]
        assert http.post("/api/trash/derivation/sales_peek/restore").status_code == 200
        assert (
            http.delete("/api/derivations/sales_peek?permanent=true").status_code == 200
        )
        assert store.get_derivation("sales_peek") is None
