"""Data-source connectors: crypto round-trip, connection resolution, and REST CRUD."""

from __future__ import annotations

import contextlib
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elbi import create_app, crypto, datasources
from elbi.db import DataSource, open_store


class _NoClient:
    def step(
        self, transcript: object, tools: Sequence[object]
    ) -> object:  # pragma: no cover
        raise NotImplementedError


def _make_app(tmp: str, name: str):
    store = open_store(f"sqlite:{Path(tmp) / name}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_NoClient(), store=store)
    return app, store


def _sqlite_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.execute("create table t (x integer)")
    con.executemany("insert into t values (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()


def test_crypto_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SECRET_KEY", "topsecret")
    token = crypto.encrypt("hunter2")
    assert token != "hunter2"
    assert crypto.decrypt(token) == "hunter2"
    assert crypto.is_configured()


def test_crypto_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    assert not crypto.is_configured()
    with pytest.raises(RuntimeError):
        crypto.encrypt("x")


def test_connection_url_test_and_load(tmp_path: object) -> None:
    db = str(Path(str(tmp_path)) / "data.db")
    _sqlite_db(db)
    src = DataSource(name="s", kind="sqlite", database=db)
    assert datasources.connection_url(src) == f"sqlite:///{db}"
    assert datasources.test_connection(src) == {"ok": True}
    rows = datasources.load_rows(src, table="t", max_rows=10)
    assert [r["x"] for r in rows] == [1, 2, 3]


def test_test_connection_reports_failure() -> None:
    # An unsupported kind fails to build a URL; test_connection reports it, not raises.
    bad = DataSource(name="b", kind="mongodb", host="h")
    result = datasources.test_connection(bad)
    assert result["ok"] is False and "mongodb" in result["error"]


def test_datasource_crud(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SECRET_KEY", "k")
    app, store = _make_app(str(tmp_path), "ds.db")
    with TestClient(app) as http:
        created = http.post(
            "/api/data-sources",
            json={
                "name": "wh",
                "kind": "postgres",
                "host": "h",
                "port": 5432,
                "database": "db",
                "username": "u",
                "secret": "pw",
            },
        ).json()
        assert created["secretSet"] is True and "secret" not in created
        sid = created["id"]
        # the secret is stored encrypted, not in the clear
        stored = store.get_data_source(sid)
        assert stored.secret != "pw"  # type: ignore[union-attr]
        assert crypto.decrypt(stored.secret) == "pw"  # type: ignore[union-attr,arg-type]
        # list masks the secret
        assert [s["name"] for s in http.get("/api/data-sources").json()] == ["wh"]
        # update without a secret keeps the stored one
        http.put(f"/api/data-sources/{sid}", json={"host": "h2"})
        again = store.get_data_source(sid)
        assert again.host == "h2"  # type: ignore[union-attr]
        assert crypto.decrypt(again.secret) == "pw"  # type: ignore[union-attr,arg-type]
        # duplicate name rejected; delete works
        assert (
            http.post(
                "/api/data-sources", json={"name": "wh", "kind": "sqlite"}
            ).status_code
            == 400
        )
        assert http.delete(f"/api/data-sources/{sid}").status_code == 200
        assert http.get(f"/api/data-sources/{sid}").status_code == 404


def test_secret_requires_app_key(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    app, _ = _make_app(str(tmp_path), "nokey.db")
    with TestClient(app) as http:
        r = http.post(
            "/api/data-sources",
            json={"name": "x", "kind": "postgres", "secret": "pw"},
        )
        assert r.status_code == 400


def test_preview_and_test_endpoints(tmp_path: object) -> None:
    db = str(Path(str(tmp_path)) / "d.db")
    _sqlite_db(db)
    app, _ = _make_app(str(tmp_path), "prev.db")
    with TestClient(app) as http:
        v = http.post(
            "/api/data-sources", json={"name": "s", "kind": "sqlite", "database": db}
        ).json()
        assert http.post(f"/api/data-sources/{v['id']}/test", json={}).json() == {
            "ok": True
        }
        preview = http.post(
            f"/api/data-sources/{v['id']}/preview", json={"table": "t"}
        ).json()
        assert preview["count"] == 3 and preview["rows"][0]["x"] == 1


def test_an_unsaved_connection_can_be_tested_before_it_is_saved(
    tmp_path: object,
) -> None:
    """The dialog's Test button posts the form, not a stored source.

    Nothing exercised this path, so the payload-to-source conversion it does on the way
    in was covered only through the saved-source route, which skips it.
    """
    db = str(Path(str(tmp_path)) / "unsaved.db")
    _sqlite_db(db)
    app, store = _make_app(str(tmp_path), "unsaved-app.db")
    with TestClient(app) as http:
        ok = http.post(
            "/api/data-sources/test",
            json={"name": "probe", "kind": "sqlite", "database": db},
        )
        assert ok.status_code == 200 and ok.json() == {"ok": True}

        bad = http.post(
            "/api/data-sources/test",
            json={"name": "probe", "kind": "sqlite", "database": "/nope/missing.db"},
        )
        assert bad.status_code == 200 and bad.json()["ok"] is False

    # Testing is not saving: neither call may leave a source behind.
    assert store.list_data_sources() == []


def test_secrets_round_trip_masked(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save, list and delete a secret. The value never comes back out.

    Listing is what the settings page reads, and it had a test only by accident of
    another one calling it; saving and deleting had none at all.
    """
    monkeypatch.setenv("APP_SECRET_KEY", "k")
    app, _ = _make_app(str(tmp_path), "secrets.db")
    with TestClient(app) as http:
        saved = http.post(
            "/api/secrets",
            json={"name": "api_key", "value": "hunter2", "description": "for the API"},
        )
        assert saved.status_code == 200 and saved.json() == {"name": "api_key"}

        listed = http.get("/api/secrets").json()
        assert listed == [{"name": "api_key", "description": "for the API"}]
        assert "hunter2" not in str(listed)

        assert http.request("DELETE", "/api/secrets/api_key").status_code == 200
        assert http.get("/api/secrets").json() == []
        assert http.request("DELETE", "/api/secrets/api_key").status_code == 404


def test_saving_a_secret_needs_a_name_and_a_key(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_SECRET_KEY", "k")
    app, _ = _make_app(str(tmp_path), "secrets-guard.db")
    with TestClient(app) as http:
        assert http.post("/api/secrets", json={"value": "v"}).status_code == 400
        monkeypatch.delenv("APP_SECRET_KEY", raising=False)
        refused = http.post("/api/secrets", json={"name": "n", "value": "v"})
        assert refused.status_code == 400
        assert "APP_SECRET_KEY" in refused.json()["detail"]


# -- dlt's outbound telemetry --------------------------------------------------------


def test_dlt_telemetry_is_off_before_the_library_can_report_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dlt enables anonymous usage tracking by default; this deployment must not.

    Left alone, dlt posts environment metadata to a dltHub endpoint the first time it is
    used. The payload carries no customer data, but an unannounced outbound connection
    is a finding in any security review, it contradicts the deployment docs' promise
    that the only outbound calls go to the configured LLM endpoint, and in an air-gapped
    install it is a failed connection somebody has to explain.

    Asserted at the socket, not just on the setting, because the setting only matters if
    it is applied before dlt loads and initialises its tracker.
    """
    import socket

    from elbi.warehouse.sources import _dlt

    monkeypatch.delenv("RUNTIME__DLTHUB_TELEMETRY", raising=False)

    dialed: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        dialed.append(address)
        raise OSError("no outbound connections in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)

    # Constructing the source is what pulls dlt in; the config need not be reachable.
    with contextlib.suppress(Exception):
        _dlt.rest_api_source(
            {"client": {"base_url": "https://example.invalid"}, "resources": ["x"]}
        )

    monkeypatch.undo()

    from dlt.common.configuration.resolve import resolve_configuration
    from dlt.common.configuration.specs import RuntimeConfiguration

    assert resolve_configuration(RuntimeConfiguration()).dlthub_telemetry is False
    assert not [a for a in dialed if "scalevector" in str(a) or "dlthub" in str(a)]


def test_no_connector_reaches_dlt_except_through_the_one_door() -> None:
    """A third call site that imports dlt directly would silently re-enable telemetry.

    The setting has to be applied before the library is imported, so the import itself
    is the thing that must stay in one place. Cheaper to enforce than to rediscover.
    """
    warehouse = Path(__file__).resolve().parents[1] / "src" / "elbi" / "warehouse"
    offenders = [
        f"{path.relative_to(warehouse)}:{n}"
        for path in warehouse.rglob("*.py")
        if path.name != "_dlt.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.match(r"\s*(from dlt[. ]|import dlt\b)", line)
    ]
    assert not offenders, (
        f"import dlt through warehouse.sources._dlt instead: {offenders}"
    )


def test_a_failed_connection_test_masks_the_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dialect is free to quote the URL it was handed, and that URL holds the secret.

    The driver is stubbed because the ones packaged here happen not to echo the URL;
    what is under test is that the message is masked when one does.
    """
    monkeypatch.setenv("APP_SECRET_KEY", "topsecret")
    source = DataSource(
        name="warehouse",
        kind="postgres",
        host="db.internal",
        port=5432,
        database="analytics",
        username="reader",
        secret=crypto.encrypt("hunter2"),
    )

    def _echoes_the_url(url: str, **kwargs: object) -> object:
        raise RuntimeError(f"could not connect using {url}")

    monkeypatch.setattr(datasources, "create_engine", _echoes_the_url)
    result = datasources.test_connection(source)
    assert result["ok"] is False
    assert "hunter2" not in result["error"]
    assert "db.internal" in result["error"]
