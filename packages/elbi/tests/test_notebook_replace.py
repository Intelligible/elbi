"""Updating a notebook from a file must keep the notebook, not replace it with a copy.

``elbi sync`` used to push a changed notebook by deleting it and importing the file as
a new one. The cells came out right, so it looked correct, and it quietly threw away
everything else the row carried -- its id, its folder, and its history.

Each test here pins one thing that was lost. They are written against the HTTP surface
because that is what the CLI drives.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app, notebooks
from elbi.db import Store, open_store

OWNER = "00uAlice"
COLLEAGUE = "00uBob"


@pytest.fixture
def store(tmp_path: Any) -> Iterator[Store]:
    opened = open_store(f"sqlite:{tmp_path / 'replace.db'}")
    yield opened
    opened.close()


@pytest.fixture
def http(store: Store) -> Iterator[TestClient]:
    service = notebooks.NotebookService(store=store, load_datasets=lambda: {})
    app = create_app(
        load_datasets=lambda: {},
        client=MagicMock(),
        store=store,
        notebook_service=service,
    )
    with TestClient(app) as client:
        yield client
    service.close()


def _ipynb(*sources: str) -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": source,
                "outputs": [],
                "execution_count": None,
            }
            for source in sources
        ],
    }


def _create(http: TestClient, name: str = "Analysis") -> str:
    return str(http.post("/api/notebooks", json={"name": name}).json()["id"])


# -- what the bug destroyed ------------------------------------------------------
def test_the_notebook_keeps_its_id(http: TestClient) -> None:
    """Anything referring to it -- a dashboard tile, a lineage edge, a shared link --
    points at this id."""
    notebook_id = _create(http)
    response = http.put(
        f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": _ipynb("print(1)")}
    )
    assert response.status_code == 200, response.text
    assert response.json()["id"] == notebook_id
    listed = http.get("/api/notebooks").json()
    assert [n["id"] for n in listed] == [notebook_id]


def test_the_notebook_keeps_its_folder(http: TestClient, store: Store) -> None:
    """A notebook that comes back at the root has lost where it was filed, silently."""
    folder_id = http.post("/api/notebooks/folders", json={"name": "finance"}).json()[
        "id"
    ]
    notebook_id = _create(http)
    http.post(f"/api/notebooks/{notebook_id}/move", json={"folder_id": folder_id})
    assert http.get("/api/notebooks").json()[0]["folder_id"] == folder_id

    http.put(f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": _ipynb("print(1)")})

    after = http.get("/api/notebooks").json()
    assert len(after) == 1
    assert after[0]["folder_id"] == folder_id


def test_the_notebook_keeps_its_creation_time(http: TestClient) -> None:
    notebook_id = _create(http)
    before = http.get("/api/notebooks").json()[0]["created_at"]
    http.put(f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": _ipynb("print(1)")})
    assert http.get("/api/notebooks").json()[0]["created_at"] == before


# -- and it still does the job it was replacing ----------------------------------
def test_the_cells_are_replaced(http: TestClient) -> None:
    notebook_id = _create(http)
    http.put(
        f"/api/notebooks/{notebook_id}/ipynb",
        json={"ipynb": _ipynb("first", "second")},
    )
    view = http.get(f"/api/notebooks/{notebook_id}").json()
    assert [c["source"] for c in view["cells"]] == ["first", "second"]

    http.put(f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": _ipynb("only")})
    view = http.get(f"/api/notebooks/{notebook_id}").json()
    assert [c["source"] for c in view["cells"]] == ["only"]


def test_cells_get_fresh_ids(http: TestClient) -> None:
    """The store keys cells globally, so reusing the file's ids collides with whatever
    the file was exported from."""
    notebook_id = _create(http)
    http.put(f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": _ipynb("a", "b")})
    first = [c["id"] for c in http.get(f"/api/notebooks/{notebook_id}").json()["cells"]]
    other = _create(http, "Second")
    http.put(f"/api/notebooks/{other}/ipynb", json={"ipynb": _ipynb("a", "b")})
    second = [c["id"] for c in http.get(f"/api/notebooks/{other}").json()["cells"]]
    assert not set(first) & set(second)


def test_a_rename_travels_with_the_contents(http: TestClient) -> None:
    """A file's name is the notebook's name in the repo, so renaming the file renames
    the notebook rather than pushing cells and leaving the old name behind."""
    notebook_id = _create(http, "Old name")
    http.put(
        f"/api/notebooks/{notebook_id}/ipynb",
        json={"ipynb": _ipynb("x"), "name": "New name"},
    )
    assert http.get("/api/notebooks").json()[0]["name"] == "New name"


def test_declared_packages_are_applied(http: TestClient, store: Store) -> None:
    """``metadata.elbi`` configures the notebook, the same as on import."""
    notebook_id = _create(http)
    document = _ipynb("x")
    document["metadata"] = {"elbi": {"deps": ["pandas"]}}
    http.put(f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": document})
    row = store.get_notebook(notebook_id)
    assert row is not None
    assert json.loads(row.deps_json or "[]") == ["pandas"]


# -- refusals --------------------------------------------------------------------
def test_a_missing_notebook_is_a_404(http: TestClient) -> None:
    response = http.put("/api/notebooks/nope/ipynb", json={"ipynb": _ipynb("x")})
    assert response.status_code == 404


def test_a_body_with_no_document_is_refused(http: TestClient) -> None:
    notebook_id = _create(http)
    response = http.put(f"/api/notebooks/{notebook_id}/ipynb", json={"name": "x"})
    assert response.status_code == 400


def test_a_document_that_is_not_an_object_is_refused(http: TestClient) -> None:
    notebook_id = _create(http)
    response = http.put(
        f"/api/notebooks/{notebook_id}/ipynb", json={"ipynb": "not a document"}
    )
    assert response.status_code == 400


# -- the service, directly -------------------------------------------------------
def test_replace_reports_a_missing_notebook(store: Store) -> None:
    service = notebooks.NotebookService(store=store, load_datasets=lambda: {})
    try:
        assert service.replace_ipynb("does-not-exist", _ipynb("x")) is False
    finally:
        service.close()


def test_import_still_creates(store: Store) -> None:
    """The create path is unchanged: a notebook the repo has and the app does not."""
    service = notebooks.NotebookService(store=store, load_datasets=lambda: {})
    try:
        notebook_id = service.import_ipynb(_ipynb("x"), name="Fresh")
        row = store.get_notebook(notebook_id)
        assert row is not None
        assert row.name == "Fresh"
    finally:
        service.close()
