"""Upgrading the MLflow tracking store's schema when the app starts.

An image carrying a newer MLflow meets a ``mlflow.db`` written by the older one, and
MLflow refuses an out-of-date schema outright. These build a store genuinely one
migration behind (MLflow's initial tables, then its own migrations up to the revision
before head) and check what starting the app does to it.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from alembic import command
from alembic.script import ScriptDirectory
from mlflow.store.db import utils as db_utils
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase

from elbi.ml import upgrade_tracking_store


def _uri(path: Path) -> str:
    return f"sqlite:///{path}"


def _revision(path: Path) -> str | None:
    engine = db_utils.create_sqlalchemy_engine(_uri(path))
    try:
        return db_utils._get_schema_version(engine)
    finally:
        engine.dispose()


def _head() -> str:
    return db_utils._get_latest_schema_revision()


@pytest.fixture(scope="module")
def behind_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A store one migration behind head, built once and copied per test."""
    path = tmp_path_factory.mktemp("mlflow") / "behind.db"
    config = db_utils._get_alembic_config(_uri(path))
    previous = ScriptDirectory.from_config(config).get_revision(_head()).down_revision
    assert isinstance(previous, str)
    engine = db_utils.create_sqlalchemy_engine(_uri(path))
    try:
        InitialBase.metadata.create_all(engine)
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, previous)
    finally:
        engine.dispose()
    return path, previous


@pytest.fixture
def behind(tmp_path: Path, behind_template: tuple[Path, str]) -> tuple[Path, str]:
    template, previous = behind_template
    path = tmp_path / "mlflow.db"
    path.write_bytes(template.read_bytes())
    return path, previous


def test_a_store_behind_is_upgraded_to_head_after_a_backup(
    behind: tuple[Path, str],
) -> None:
    path, previous = behind
    with pytest.raises(Exception, match="out-of-date database schema"):
        engine = db_utils.create_sqlalchemy_engine(_uri(path))
        try:
            db_utils._verify_schema(engine)
        finally:
            engine.dispose()

    upgrade_tracking_store(_uri(path))

    assert _revision(path) == _head()
    backup = path.with_name(f"mlflow.db.{previous}.bak")
    assert _revision(backup) == previous


def test_a_current_store_is_left_alone(behind: tuple[Path, str]) -> None:
    path, _ = behind
    upgrade_tracking_store(_uri(path))
    for stale in path.parent.glob("*.bak"):
        stale.unlink()
    before = path.stat().st_mtime_ns

    upgrade_tracking_store(_uri(path))

    assert path.stat().st_mtime_ns == before
    assert list(path.parent.glob("*.bak")) == []


def test_an_earlier_backup_of_the_same_revision_is_kept(
    behind: tuple[Path, str],
) -> None:
    """A retry after a failed attempt must not replace the pristine copy."""
    path, previous = behind
    backup = path.with_name(f"mlflow.db.{previous}.bak")
    backup.write_bytes(b"the first copy")

    upgrade_tracking_store(_uri(path))

    assert backup.read_bytes() == b"the first copy"
    assert _revision(path) == _head()


def test_a_missing_store_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "mlflow.db"
    upgrade_tracking_store(_uri(path))
    assert not path.exists()


def test_an_empty_store_is_left_for_mlflow_to_initialize(tmp_path: Path) -> None:
    path = tmp_path / "mlflow.db"
    sqlite3.connect(path).close()
    upgrade_tracking_store(_uri(path))
    assert _revision(path) is None
    assert list(tmp_path.glob("*.bak")) == []


@pytest.mark.parametrize(
    "uri",
    [
        "http://mlflow.internal:5000",
        "postgresql+psycopg2://u:p@db.internal/mlflow",
        "file:///tmp/mlruns",
    ],
)
def test_a_store_that_is_not_a_local_sqlite_file_is_not_touched(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object) -> None:
        raise AssertionError("must not open a store it does not own")

    monkeypatch.setattr(db_utils, "create_sqlalchemy_engine", refuse)
    upgrade_tracking_store(uri)


def test_a_failed_upgrade_is_logged_not_raised(
    behind: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path, previous = behind

    def broken(engine: object) -> None:
        raise RuntimeError("migration exploded")

    monkeypatch.setattr(db_utils, "_upgrade_db", broken)
    with caplog.at_level(logging.ERROR, logger="elbi"):
        upgrade_tracking_store(_uri(path))

    assert _revision(path) == previous
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    message = record.getMessage()
    assert "could not upgrade the MLflow tracking store" in message
    assert f"mlflow.db.{previous}.bak" in message
    assert f"mlflow db upgrade {_uri(path)}" in message


_SALES = "customer_id,amount\nc1,100\nc2,5\n"


def test_serving_a_project_upgrades_its_store_before_use(
    tmp_path: Path, behind_template: tuple[Path, str]
) -> None:
    """``elbi serve`` upgrades the store under the project cache as it builds."""
    pytest.importorskip("deltalake")
    pytest.importorskip("duckdb")
    from elbi.serve import build
    from elbi_cli.project import load_project

    root = tmp_path / "project"
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "elbi.yaml").write_text(
        "project: upgrade\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    cache_dir = load_project(root).cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = cache_dir / "mlflow.db"
    store.write_bytes(behind_template[0].read_bytes())

    build(root, with_mcp=False)

    assert _revision(store) == _head()
