"""Opening a database an earlier version wrote.

`create_all` only ever adds, so a column this version's models no longer declare
survives in an older database. A nullable leftover is dead weight; one declared NOT
NULL rejects every insert the current code writes, because that code no longer supplies
it -- a store that opens and then cannot be written to. No test caught that, because
every other test starts from an empty file.

What is retired is worked out by comparing the database against the models, so the last
test here is the one that matters: it retires a column no list ever named.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

from elbi.db import Derivation, open_store


def _legacy(path: Path) -> sqlite3.Connection:
    """A database with the shape the previous version wrote.

    Built by taking the current schema and putting the retired columns back, rather than
    by hand: a hand-written table would be missing whatever else has been added since,
    and would then be failing for a reason this module is not about.
    """
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine)
    engine.dispose()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE budget")
    conn.executescript(
        """
        ALTER TABLE derivation ADD COLUMN retired_note TEXT;
        ALTER TABLE derivation ADD COLUMN retired_kind TEXT NOT NULL DEFAULT 'x';
        ALTER TABLE derivation ADD COLUMN retired_at TEXT;
        CREATE INDEX ix_derivation_retired_note ON derivation (retired_note);
        CREATE TABLE budget (
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            max_budget FLOAT NOT NULL DEFAULT 0.0,
            window TEXT NOT NULL DEFAULT '30d',
            window_start TIMESTAMP NOT NULL DEFAULT '2026-01-01 00:00:00',
            spend FLOAT NOT NULL DEFAULT 0.0,
            PRIMARY KEY (subject_type, subject_id)
        );
        """
    )
    return conn


def test_a_retired_column_is_dropped_and_the_store_stays_writable(
    tmp_path: Path,
) -> None:
    """The failure this exists for: an insert refused by a column nobody supplies."""
    path = tmp_path / "old.db"
    conn = _legacy(path)
    conn.execute(
        "INSERT INTO derivation"
        " (name, question, source, verdict, narrative, rendered, origin,"
        " created_at, retired_note)"
        " VALUES ('legacy', 'q', 's', 'sound', 'n', 'r', 'repo',"
        " '2026-01-01 00:00:00', 'alice')"
    )
    conn.commit()
    conn.close()

    store = open_store(f"sqlite:{path}")
    # The store's own engine, not a second one: an engine opened inline here is
    # collected at an arbitrary later moment, and CPython 3.13 raises "unclosed
    # database" from `sqlite3.Connection.__del__` inside whichever test is running then.
    columns = {c["name"] for c in inspect(store._engine).get_columns("derivation")}
    assert not columns & {"retired_note", "retired_kind", "retired_at"}

    # The row an earlier version wrote is still there, and a new one can be written --
    # which the NOT NULL leftover refused before the column was retired.
    assert store.get_derivation("legacy") is not None
    store.save_derivation(Derivation(name="fresh", question="q", source="s"))
    assert store.get_derivation("fresh") is not None
    store.close()


def test_a_retired_key_column_rebuilds_the_table_and_carries_the_cap(
    tmp_path: Path,
) -> None:
    """A cap used to be held per subject, so the key changed rather than a column going.

    No ``ALTER`` expresses that and SQLite refuses to drop a key column at all, so the
    table is rebuilt -- and the instance-wide cap carries over, being the one whose
    meaning survives having a single caller.
    """
    path = tmp_path / "old.db"
    conn = _legacy(path)
    conn.execute(
        "INSERT INTO budget (subject_type, subject_id, max_budget, window, spend)"
        " VALUES ('global', '', 250.0, '30d', 3.5)"
    )
    conn.execute(
        "INSERT INTO budget (subject_type, subject_id, max_budget, window, spend)"
        " VALUES ('user', 'alice', 10.0, '7d', 1.0)"
    )
    conn.commit()
    conn.close()

    store = open_store(f"sqlite:{path}")
    budget = store.get_budget()
    assert budget is not None
    assert (budget.max_budget, budget.window, budget.spend) == (250.0, "30d", 3.5)
    store.close()


def test_a_column_nobody_wrote_down_is_retired_too(tmp_path: Path) -> None:
    """The reason this is derived rather than listed.

    A hardcoded list retires exactly what somebody remembered to add to it, and a column
    left off is the one that breaks a store nobody can then write to. Here the column is
    invented by the test, so no list could have named it.
    """
    path = tmp_path / "unlisted.db"
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine)
    engine.dispose()

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE metric ADD COLUMN forgotten TEXT NOT NULL DEFAULT 'x'")
    conn.commit()
    conn.close()

    store = open_store(f"sqlite:{path}")
    columns = {c["name"] for c in inspect(store._engine).get_columns("metric")}
    assert "forgotten" not in columns


def test_opening_twice_changes_nothing_the_second_time(tmp_path: Path) -> None:
    """Every upgrade step runs on every open, so each has to be a no-op once applied."""
    path = tmp_path / "old.db"
    conn = _legacy(path)
    conn.execute(
        "INSERT INTO derivation"
        " (name, question, source, narrative, rendered, origin, created_at)"
        " VALUES ('legacy', 'q', 's', 'n', 'r', 'repo', '2026-01-01 00:00:00')"
    )
    conn.commit()
    conn.close()

    open_store(f"sqlite:{path}").close()
    store = open_store(f"sqlite:{path}")
    assert store.get_derivation("legacy") is not None
    store.close()
