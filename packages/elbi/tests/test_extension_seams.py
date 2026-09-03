"""The seams an extension attaches to, exercised through the real routes.

These are the app's extension contract. Nothing in this repository consumes them, which
is exactly why they are tested here: a seam with no in-repo caller is a seam that gets
renamed during an unrelated refactor, and the breakage then surfaces somewhere else
entirely, long afterwards, as an extension that silently stops applying.

Each test drives a *fake* implementation through the routes a real one would attach to,
so it checks the contract rather than any particular extension.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from elbi import create_app, extensions
from elbi.db import Store, open_store
from elbi_core.errors import ConfigError


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = open_store(f"sqlite:{tmp_path / 'seams.db'}")
    yield opened
    opened.close()


def _app(store: Store) -> FastAPI:
    return create_app(load_datasets=lambda: {}, store=store)


def test_an_extension_can_mount_routes_and_reads_the_services_it_needs(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The context carries the app, the store and the warehouse, and routes stick.

    Mounted before the single-page app's catch-all, which would otherwise shadow
    anything an extension added.
    """
    seen: dict[str, Any] = {}

    class _Extension:
        def install_extension(self, context: extensions.ExtensionContext) -> None:
            seen["store"] = context.store
            seen["warehouse"] = context.warehouse

            @context.app.get("/api/from-an-extension")
            async def _added() -> dict[str, bool]:
                return {"mounted": True}

    monkeypatch.setattr(extensions, "load_all", lambda: [("fake", _Extension())])
    with TestClient(_app(store)) as http:
        assert http.get("/api/from-an-extension").json() == {"mounted": True}
    assert seen["store"] is store
    assert "warehouse" in seen


def test_one_failing_extension_does_not_stop_the_app_from_serving(
    store: Store, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An extension that raises is logged and skipped; the rest still install.

    The alternative is an unbootable app whose only symptom is a traceback from a
    package the operator may not know is installed.
    """

    class _Broken:
        def install_extension(self, context: extensions.ExtensionContext) -> None:
            raise RuntimeError("misconfigured")

    class _Fine:
        def install_extension(self, context: extensions.ExtensionContext) -> None:
            @context.app.get("/api/still-here")
            async def _added() -> dict[str, bool]:
                return {"ok": True}

    monkeypatch.setattr(
        extensions, "load_all", lambda: [("broken", _Broken()), ("fine", _Fine())]
    )
    with TestClient(_app(store)) as http:
        assert http.get("/api/still-here").json() == {"ok": True}
        assert http.get("/health").status_code == 200
    assert "broken" in caplog.text


def test_nothing_is_installed_when_no_extension_is_present(store: Store) -> None:
    """The shipped default: an empty group, and no route it would have added."""
    assert extensions.available() == []
    with TestClient(_app(store)) as http:
        assert http.get("/api/from-an-extension").status_code == 404


def test_a_store_hands_out_its_own_sessions(store: Store) -> None:
    """``reading``/``writing`` are how an extension queries its own tables.

    This store's sessions, not a second engine: a writer outside its lock would contend
    with the app's on SQLite, which permits one writer per file.
    """
    from elbi.db import Setting

    with store.writing() as session:
        assert session.get_bind() is store._engine
        session.add(Setting(key="theme", value="dark"))
        session.commit()

    with store.reading() as session:
        assert session.get_bind() is store._engine
        assert session.get(Setting, "theme") is not None


def test_an_extension_supplies_the_class_the_store_is_built_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam that has to resolve before there is a store to hand anybody.

    An extension adding tables and queries of its own serves them from the one engine,
    so the class is settled at the point the store is opened rather than mounted onto
    one already built.
    """
    from sqlmodel import Field, SQLModel

    class Extra(SQLModel, table=True):
        __tablename__ = "extension_table"
        key: str = Field(primary_key=True)

    class _Bigger(Store):
        store_class_marker = True

    class _Supplier:
        store_class = _Bigger

        def install_extension(self, context: extensions.ExtensionContext) -> None:
            pass

    monkeypatch.setattr(extensions, "load_all", lambda: [("supplier", _Supplier())])
    opened = open_store(f"sqlite:{tmp_path / 'supplied.db'}")
    try:
        assert isinstance(opened, _Bigger)
        # Loading the extension is what registers its tables, and that has to happen
        # before the schema is created or they are simply absent.
        with opened.writing() as session:
            session.add(Extra(key="k"))
            session.commit()
    finally:
        opened.close()
        SQLModel.metadata.remove(SQLModel.metadata.tables["extension_table"])


def test_an_explicit_store_class_beats_the_one_an_extension_supplies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller naming the class means it, so the extension does not override it."""

    class _Supplied(Store):
        pass

    class _Asked(Store):
        pass

    class _Supplier:
        store_class = _Supplied

        def install_extension(self, context: extensions.ExtensionContext) -> None:
            pass

    monkeypatch.setattr(extensions, "load_all", lambda: [("supplier", _Supplier())])
    opened = open_store(f"sqlite:{tmp_path / 'asked.db'}", store_class=_Asked)
    try:
        assert isinstance(opened, _Asked)
    finally:
        opened.close()


def test_two_extensions_supplying_a_store_class_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only one class can build the store, and the sort order must not decide which."""

    class _One:
        store_class = Store

        def install_extension(self, context: extensions.ExtensionContext) -> None:
            pass

    monkeypatch.setattr(extensions, "load_all", lambda: [("a", _One()), ("b", _One())])
    with pytest.raises(ConfigError, match="only one can build the store"):
        extensions.store_class()


def test_no_extension_leaves_the_shipped_store_class(store: Store) -> None:
    """The default install: nothing supplies one, and the plain store is what opens."""
    assert extensions.store_class() is None
    assert type(store) is Store


def test_an_extension_supplies_the_classes_the_services_are_built_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the same seam: a service this package constructs, not mounts.

    An extension narrowing what a service does cannot wait to be handed one already
    built, so it names a subclass and the build resolves it. Matched on what the
    subclass inherits from rather than by name, so nothing keeps a string in step.
    """
    from elbi.notebooks import NotebookService
    from elbi.warehouse.service import WarehouseService

    class _Narrower(NotebookService):
        pass

    class _Supplier:
        service_classes = (_Narrower,)

        def install_extension(self, context: extensions.ExtensionContext) -> None:
            pass

    monkeypatch.setattr(extensions, "load_all", lambda: [("supplier", _Supplier())])
    assert extensions.service_class(NotebookService) is _Narrower
    # A service nothing replaced keeps the shipped class.
    assert extensions.service_class(WarehouseService) is WarehouseService


def test_two_extensions_replacing_one_service_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only one class can build a service, and the sort order must not decide which."""
    from elbi.notebooks import NotebookService

    class _Theirs(NotebookService):
        pass

    class _Supplier:
        service_classes = (_Theirs,)

        def install_extension(self, context: extensions.ExtensionContext) -> None:
            pass

    monkeypatch.setattr(
        extensions, "load_all", lambda: [("a", _Supplier()), ("b", _Supplier())]
    )
    with pytest.raises(ConfigError, match="only one can build it"):
        extensions.service_class(NotebookService)


def test_no_extension_leaves_every_shipped_service_class(store: Store) -> None:
    """The default install: each service is built from the class shipped here."""
    from elbi.dashboards import DashboardService
    from elbi.notebooks import NotebookService

    for base in (NotebookService, DashboardService):
        assert extensions.service_class(base) is base


def test_a_store_contributes_its_own_housekeeping(tmp_path: Path) -> None:
    """A store with cutoff-based rows of its own gets them swept by the same pass.

    Under the built-in contract: report how many rows went, and be safe to re-run.
    """
    from elbi.maintenance import run_once

    swept = 0

    class _Sweeping(Store):
        def housekeeping(self) -> dict[str, Any]:
            def sweep() -> int:
                nonlocal swept
                swept += 1
                return 7

            return {"widgets_pruned": sweep}

    opened = open_store(f"sqlite:{tmp_path / 'sweep.db'}", store_class=_Sweeping)
    try:
        result = run_once(opened)
        assert result["widgets_pruned"] == 7
        # The built-in tasks still run beside it.
        assert "trash_purged" in result
        assert run_once(opened)["widgets_pruned"] == 7
        assert swept == 2
    finally:
        opened.close()


def test_a_failing_contributed_task_does_not_stop_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Each task is isolated, contributed ones included."""
    from elbi.maintenance import run_once

    class _Broken(Store):
        def housekeeping(self) -> dict[str, Any]:
            def fail() -> int:
                raise RuntimeError("no")

            return {"widgets_pruned": fail}

    opened = open_store(f"sqlite:{tmp_path / 'broken.db'}", store_class=_Broken)
    try:
        result = run_once(opened)
        assert "widgets_pruned" not in result
        assert "trash_purged" in result
    finally:
        opened.close()
    assert "widgets_pruned" in caplog.text


def test_a_stored_rendering_is_withheld_when_the_app_says_to(store: Store) -> None:
    """Asked per request off the app, so something installed later can answer it.

    A rendering is text: a narrowing that arrives after it was computed cannot be
    applied to it, so the question is whether to serve it at all.
    """
    from elbi.db import Derivation

    store.save_derivation(
        Derivation(name="roster", question="q", source="s", rendered="| ssn |\n| 1 |")
    )
    app = _app(store)
    with TestClient(app) as http:
        assert "| ssn |" in (
            http.get("/api/derivations/roster").json()["rendered"] or ""
        )
        app.state.withhold_rendering = lambda name: name == "roster"
        withheld = http.get("/api/derivations/roster").json()
        assert withheld["rendered"] is None
        # Withheld, not hidden.
        assert withheld["name"] == "roster"


def test_the_constructor_argument_seeds_the_withholding_answer(store: Store) -> None:
    """A caller passing one directly is unaffected by where it is now read from."""
    from elbi.db import Derivation

    store.save_derivation(
        Derivation(name="roster", question="q", source="s", rendered="| ssn |\n| 1 |")
    )
    app = create_app(
        load_datasets=lambda: {}, store=store, withhold_rendering=lambda _name: True
    )
    with TestClient(app) as http:
        assert http.get("/api/derivations/roster").json()["rendered"] is None
