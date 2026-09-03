"""The native store: round-trips on SQLite, and the DB_URI translation for both
backends. Exercised at the public boundary: open a store, write, read it back.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlmodel import create_engine

from elbi.db import (
    Derivation,
    Store,
    migrate,
    open_store,
    sqlalchemy_url,
)


@pytest.fixture
def store(tmp_path) -> Iterator[Store]:
    with open_store(f"sqlite:{tmp_path / 'test.db'}") as opened:
        yield opened


def test_migrate_is_idempotent_and_atomic(tmp_path) -> None:
    """A second migration emits no DDL, and the whole thing is one transaction.

    The single transaction is the load-bearing part: the advisory lock that makes
    concurrent boots safe is transaction-scoped, so it only covers the
    inspect-then-alter sequence while everything shares one connection.
    """
    engine = create_engine(sqlalchemy_url(f"sqlite:{tmp_path / 'm.db'}"))
    statements: list[str] = []
    connections: set[int] = set()
    transactions: list[int] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, *_args) -> None:
        statements.append(statement.strip().upper())
        connections.add(id(conn))

    @event.listens_for(engine, "begin")
    def _began(conn) -> None:
        transactions.append(1)

    try:
        migrate(engine)
        ddl = [s for s in statements if s.startswith(("CREATE TABLE", "ALTER TABLE"))]
        assert ddl, "the first migration must create the schema"
        assert len(connections) == 1
        assert len(transactions) == 1

        statements.clear()
        migrate(engine)
        kinds = ("CREATE TABLE", "ALTER TABLE")
        assert [s for s in statements if s.startswith(kinds)] == []
    finally:
        # Leaving the pool open leaks a sqlite connection, and the ResourceWarning its
        # finalizer raises is reported against whichever test happens to be running.
        engine.dispose()


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URI"),
    reason="needs a disposable Postgres; set TEST_POSTGRES_URI to run",
)
def test_concurrent_migrate_on_postgres_does_not_race() -> None:
    """Replicas booting together must all succeed, not fight over the same DDL.

    Without the advisory lock this fails: concurrent ``create_all`` raises a unique
    violation on ``pg_type_typname_nsp_index``, and concurrent ``ALTER TABLE`` on the
    same column raises a duplicate-column error.
    """
    uri = os.environ["TEST_POSTGRES_URI"]
    from sqlalchemy import text

    reset = create_engine(sqlalchemy_url(uri))
    with reset.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    reset.dispose()

    def boot() -> None:
        engine = create_engine(sqlalchemy_url(uri))
        try:
            migrate(engine)
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=8) as pool:
        # Collect every exception, not just the first: partial failure stays legible.
        errors = [f.exception() for f in [pool.submit(boot) for _ in range(8)]]
    assert [e for e in errors if e is not None] == []


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URI"),
    reason="needs a disposable Postgres; set TEST_POSTGRES_URI to run",
)
def test_migrating_an_older_postgres_database_forward() -> None:
    """Every added column must apply to a database that predates it, on Postgres too.

    The upgrade path is the one this misses otherwise: a fresh boot has ``create_all``
    make the column, so the ``ALTER TABLE`` never runs and a default that only SQLite
    accepts looks fine. Dropping the columns first is what an older database is, and it
    is the only way this reaches the statement a deployment actually executes.

    Postgres is stricter than SQLite about a default matching its column's type, which
    is the whole point of running this here: a default one dialect takes silently, the
    other refuses.
    """
    uri = os.environ["TEST_POSTGRES_URI"]
    from sqlalchemy import inspect, text

    engine = create_engine(sqlalchemy_url(uri))
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
        migrate(engine)
    finally:
        engine.dispose()

    # A row written by the older version, through the store rather than by hand, so the
    # columns it fills are whatever the code actually fills.
    with open_store(uri) as older:
        notebook_id = older.create_notebook("Q3 plan")

    # What an older database is: the table without the columns a later version adds.
    added = {
        "conversation": ["profile", "prompt_tokens", "summary_turns"],
        "notebook": ["lock_json", "folder_id", "compute_profile", "copied_from"],
        "compute_usage": ["kind"],
        # The split structure a scheduled retrain must reproduce: nullable VARCHAR, so
        # an older row simply has no entity column rather than a bad default.
        "retrainpolicy": ["groups"],
        "orchestration_run": ["parent_run_id", "workflow_id", "workflow_steps_json"],
    }
    engine = create_engine(sqlalchemy_url(uri))
    try:
        with engine.begin() as conn:
            for table, columns in added.items():
                for column in columns:
                    conn.execute(
                        text(f'ALTER TABLE "{table}" DROP COLUMN IF EXISTS {column}')
                    )

        migrate(engine)

        inspector = inspect(engine)
        for table, columns in added.items():
            present = {c["name"] for c in inspector.get_columns(table)}
            missing = set(columns) - present
            assert not missing, f"{table} is missing {missing}"

        # The row that predates the columns survives the ALTER rather than being
        # rewritten by it.
        with engine.connect() as conn:
            name = conn.execute(
                text("SELECT name FROM notebook WHERE id = :id"), {"id": notebook_id}
            ).scalar()
        assert name == "Q3 plan"
    finally:
        engine.dispose()


def test_db_uri_translation() -> None:
    assert sqlalchemy_url("sqlite:./elbi.db") == "sqlite:///./elbi.db"
    assert sqlalchemy_url("elbi.db") == "sqlite:///elbi.db"
    assert (
        sqlalchemy_url("postgres://u:p@host:5432/db")
        == "postgresql+psycopg://u:p@host:5432/db"
    )
    assert (
        sqlalchemy_url("postgresql://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    )


def test_job_store_round_trips_and_dedupes(store: Store) -> None:
    from elbi_core import Job

    jobs = store.job_store()
    jobs.create(Job(id="j1", key="k", label="train", state="queued", created_at=1.0))
    assert jobs.get("j1").state == "queued"  # type: ignore[union-attr]
    updated = jobs.update("j1", state="succeeded", result={"r2": 0.9})
    assert updated.state == "succeeded" and updated.result == {"r2": 0.9}
    assert jobs.find_by_key("k").id == "j1"  # type: ignore[union-attr]
    assert jobs.get("missing") is None
    assert [j.id for j in jobs.list()] == ["j1"]


def test_jobs_survive_a_reopen(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from elbi_core import Job

    uri = f"sqlite:{tmp_path / 'jobs.db'}"
    store_a = open_store(uri)
    store_a.job_store().create(
        Job(id="persisted", key="k", label="x", state="running", created_at=1.0)
    )
    # A fresh store over the same database still sees the job (durable across restart).
    reopened = open_store(uri).job_store().get("persisted")
    assert reopened is not None and reopened.state == "running"


def test_job_runner_over_the_db_store_persists_results(store: Store) -> None:
    from elbi_core import JobRunner

    runner = JobRunner(store=store.job_store(), max_workers=2)
    try:
        job = runner.submit("train:v1", "ebm", lambda progress, cancelled: {"r2": 0.91})
        done = runner.wait(job.id, timeout=5)
        assert done is not None and done.state == "succeeded"
        assert store.job_store().get(job.id).result == {"r2": 0.91}  # type: ignore[union-attr]
    finally:
        runner.close()


def test_conversation_reuse_by_client_id(store: Store) -> None:
    # The client supplies a stable id; the server continues that same conversation
    # across turns instead of minting a new one.
    cid = store.create_conversation("train a model", conversation_id="chat-123")
    assert cid == "chat-123"
    assert store.get_conversation("chat-123") is not None
    assert store.get_conversation("never-seen") is None


def test_derivations_for_conversation(store: Store) -> None:
    store.create_conversation("t", conversation_id="c1")
    store.save_derivation(
        Derivation(name="a", conversation_id="c1", question="q", verdict="sound")
    )
    store.save_derivation(
        Derivation(name="b", conversation_id="c2", question="q", verdict="sound")
    )
    names = [d.name for d in store.derivations_for("c1")]
    assert names == ["a"]  # only this conversation's derivations


def test_conversation_and_messages_round_trip(store: Store) -> None:
    cid = store.create_conversation("effect of x on y")
    store.add_message(cid, "user", content="What is the effect of x on y?")
    store.add_message(
        cid,
        "assistant",
        content="x raises y.",
        tools=[{"name": "derive", "arguments": {}}],
        result={"verified": True, "verdict": "sound"},
    )

    convos, next_page_id = store.search_conversations()
    assert [c.id for c in convos] == [cid]
    assert next_page_id is None  # only one, so no next page

    msgs = store.messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "What is the effect of x on y?"
    assert json.loads(msgs[1].tools_json)[0]["name"] == "derive"
    assert json.loads(msgs[1].result_json or "{}")["verified"] is True


def test_conversations_paginate_by_recent_activity(store: Store) -> None:
    # Five conversations; a page of 2 returns a cursor, and ordering is by recent
    # activity: adding a message to the oldest lifts it to the front.
    ids = [store.create_conversation(f"c{i}") for i in range(5)]
    first, cursor = store.search_conversations(limit=2)
    assert len(first) == 2 and cursor == "2"  # offset cursor for the next page
    second, cursor2 = store.search_conversations(limit=2, page_id=cursor)
    assert len(second) == 2 and cursor2 == "4"
    last, cursor3 = store.search_conversations(limit=2, page_id=cursor2)
    assert len(last) == 1 and cursor3 is None  # exhausted

    # Recent activity ordering: touch the oldest conversation, it rises to the top.
    store.add_message(ids[0], "user", content="revive me")
    front, _ = store.search_conversations(limit=1)
    assert front[0].id == ids[0]


def test_derivation_persist_and_list(store: Store) -> None:
    store.save_derivation(
        Derivation(
            name="x_on_y_effect",
            question="What is the effect of x on y?",
            source="def x_on_y_effect(ctx): ...",
            claim_json=json.dumps({"x": "x", "y": "y"}),
            verdict="sound",
            narrative="x raises y.",
            rendered="| slope |\n| 0.4 |",
            data_hash="abc123",
        )
    )

    got = store.get_derivation("x_on_y_effect")
    assert got is not None
    assert got.verdict == "sound"
    assert got.data_hash == "abc123"
    assert [d.name for d in store.list_derivations()] == ["x_on_y_effect"]


def test_derivation_save_replaces_same_name(store: Store) -> None:
    store.save_derivation(Derivation(name="d", narrative="first", verdict="sound"))
    store.save_derivation(
        Derivation(name="d", narrative="second", verdict="inconclusive")
    )

    got = store.get_derivation("d")
    assert got is not None
    assert got.narrative == "second"
    assert got.verdict == "inconclusive"
    assert len(store.list_derivations()) == 1


def test_config_upsert(store: Store) -> None:
    assert store.get_config("model") is None
    store.set_config("model", "claude-opus-4-8")
    assert store.get_config("model") == "claude-opus-4-8"
    store.set_config("model", "claude-sonnet-5")
    assert store.get_config("model") == "claude-sonnet-5"


def test_store_persists_across_reopen(tmp_path) -> None:
    path = f"sqlite:{tmp_path / 'persist.db'}"
    first = open_store(path)
    first.save_derivation(Derivation(name="kept", narrative="durable", verdict="sound"))

    reopened = open_store(path)
    got = reopened.get_derivation("kept")
    assert got is not None
    assert got.narrative == "durable"


def test_sqlite_store_uses_wal(store: Store) -> None:
    # WAL lets a read run while a write is in flight; without _enable_sqlite_wal the
    # journal mode would be the blocking default.
    from sqlalchemy import text

    with store._engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal"


def test_concurrent_writes_all_persist(store: Store) -> None:
    # A background-job write and request writes never hit the single-writer database at
    # once (the shared lock), and WAL plus a busy timeout absorb contention: every write
    # lands, none is lost, and none raises "database is locked".
    import threading

    cid = store.create_conversation("c")

    def add(i: int) -> None:
        store.add_message(cid, "user", content=f"m{i}")

    threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.messages(cid)) == 20


def test_delete_conversation_removes_it_and_its_messages(store: Store) -> None:
    # The conversation and its turns are gone, and a second delete reports it no longer
    # exists.
    cid = store.create_conversation("analysis", conversation_id="c-del")
    store.add_message(cid, "user", content="how does x affect y?")
    store.add_message(cid, "assistant", content="it raises it.")
    assert len(store.messages(cid)) == 2

    assert store.delete_conversation(cid) is True
    assert store.get_conversation(cid) is None
    assert store.messages(cid) == []  # its turns went with it
    assert store.delete_conversation(cid) is False  # already gone


def test_delete_conversation_keeps_authored_derivations(store: Store) -> None:
    # Derivations are durable, certified artifacts surfaced independently, so deleting
    # the conversation that authored one does not cascade to it.
    cid = store.create_conversation("analysis", conversation_id="c-keep")
    store.save_derivation(
        Derivation(name="x_on_y", conversation_id=cid, verdict="sound")
    )
    assert store.delete_conversation(cid) is True
    assert store.get_derivation("x_on_y") is not None


def test_rename_conversation(store: Store) -> None:
    cid = store.create_conversation("old title", conversation_id="c-rename")
    assert store.rename_conversation(cid, "new title") is True
    assert store.get_conversation(cid).title == "new title"  # type: ignore[union-attr]
    assert store.rename_conversation("nope", "x") is False


def test_search_conversations_filters_by_title(store: Store) -> None:
    store.create_conversation("housing prices analysis", conversation_id="c-a")
    store.create_conversation("customer churn", conversation_id="c-b")
    store.create_conversation("Housing market trends", conversation_id="c-c")
    rows, _ = store.search_conversations(query="housing")
    names = {c.id for c in rows}
    assert names == {"c-a", "c-c"}  # case-insensitive title contains
    rows_all, _ = store.search_conversations()
    assert len(rows_all) == 3  # no query returns everything


def test_llm_profile_crud(store: Store) -> None:
    from elbi.db import LlmProfile

    store.save_profile(LlmProfile(name="OpenAI", model="openai/gpt-5", api_key="sk"))
    store.save_profile(LlmProfile(name="Claude", model="anthropic/x"))
    assert {p.name for p in store.list_profiles()} == {"OpenAI", "Claude"}
    assert store.get_profile("OpenAI").model == "openai/gpt-5"  # type: ignore[union-attr]
    # replace-by-name
    store.save_profile(LlmProfile(name="OpenAI", model="openai/gpt-5-mini"))
    assert store.get_profile("OpenAI").model == "openai/gpt-5-mini"  # type: ignore[union-attr]
    assert store.delete_profile("Claude") is True
    assert store.delete_profile("Claude") is False
    assert [p.name for p in store.list_profiles()] == ["OpenAI"]


def test_conversation_carries_a_profile(store: Store) -> None:
    cid = store.create_conversation("t", conversation_id="c1", profile="OpenAI")
    assert store.get_conversation(cid).profile == "OpenAI"  # type: ignore[union-attr]
    store.set_conversation_profile(cid, "Claude")
    assert store.get_conversation(cid).profile == "Claude"  # type: ignore[union-attr]


def test_add_usage_accumulates(store: Store) -> None:
    cid = store.create_conversation("t", conversation_id="c1")
    store.add_usage(cid, 100, 50, 0.01)
    store.add_usage(cid, 20, 10, 0.002)
    c = store.get_conversation(cid)
    assert c is not None
    assert c.prompt_tokens == 120 and c.completion_tokens == 60
    assert abs((c.cost or 0.0) - 0.012) < 1e-9


def test_set_feedback(store: Store) -> None:
    cid = store.create_conversation("t", conversation_id="c1")
    mid = store.add_message(cid, "assistant", content="ans")
    assert store.set_feedback(mid, "down") is True
    assert store.messages(cid)[0].feedback == "down"
    assert store.set_feedback("nope", "up") is False


def test_delete_messages_from(store: Store) -> None:
    cid = store.create_conversation("t", conversation_id="c1")
    m1 = store.add_message(cid, "user", content="q1")
    store.add_message(cid, "assistant", content="a1")
    m3 = store.add_message(cid, "user", content="q2")
    store.add_message(cid, "assistant", content="a2")
    # dropping from the second question removes it and its answer, keeping the first
    assert store.delete_messages_from(cid, m3) == 2
    assert [m.content for m in store.messages(cid)] == ["q1", "a1"]
    # dropping from the first question clears the whole conversation
    assert store.delete_messages_from(cid, m1) == 2
    assert store.messages(cid) == []
    assert store.delete_messages_from(cid, "nope") == 0


def test_window_seconds_parsing() -> None:
    from elbi.db import _window_seconds

    assert _window_seconds("24h") == 24 * 3600
    assert _window_seconds("30d") == 30 * 86400
    assert _window_seconds("1mo") == 30 * 86400
    assert _window_seconds("garbage") == 30 * 86400  # default


def test_setting_a_budget_again_replaces_the_one_before_it(store: Store) -> None:
    """One row: a second cap must not sit beside the first and disagree with it."""
    store.set_budget(10.0, "30d")
    store.add_spend(4.0)
    store.set_budget(5.0, "1d")

    budget = store.get_budget()
    assert budget is not None
    assert (budget.max_budget, budget.window) == (5.0, "1d")
    assert budget.spend == 0.0, "a new cap starts its window fresh"


def test_budget_spend_and_over(store: Store) -> None:
    store.set_budget(1.0, "30d")
    assert store.is_over_budget() is False
    store.add_spend(0.6)
    assert store.is_over_budget() is False
    store.add_spend(0.5)  # 1.1 >= 1.0
    assert store.is_over_budget() is True


def test_budget_elapsed_window_resets(store: Store) -> None:
    # A window of zero seconds is always elapsed, so spend resets and never blocks.
    store.set_budget(1.0, "0d")
    store.add_spend(5.0)
    assert store.is_over_budget() is False


def test_folder_crud(store: Store) -> None:
    root = store.create_folder("Marketing")
    child = store.create_folder("Q3", parent_id=root)

    folders = store.list_folders()
    assert {f["name"] for f in folders} == {"Marketing", "Q3"}
    q3 = next(f for f in folders if f["id"] == child)
    assert q3["parent_id"] == root

    store.rename_folder(child, "Q4")
    renamed = store.get_folder(child)
    assert renamed is not None and renamed.name == "Q4"


def test_move_folder_rejects_cycle(store: Store) -> None:
    a = store.create_folder("A")
    b = store.create_folder("B", parent_id=a)
    # Moving A under its own descendant B (or under itself) must be refused.
    assert store.move_folder(a, b) is False
    assert store.move_folder(a, a) is False
    assert store.get_folder(a).parent_id is None
    # A legitimate move still works.
    c = store.create_folder("C")
    assert store.move_folder(a, c) is True
    assert store.get_folder(a).parent_id == c


def test_delete_folder_empty_vs_recursive(store: Store) -> None:
    parent = store.create_folder("Parent")
    child = store.create_folder("Child", parent_id=parent)
    nb = store.create_notebook("Inside", folder_id=child)

    # Non-recursive delete of a non-empty folder is refused and changes nothing.
    assert store.delete_folder(parent, recursive=False) is False
    assert store.get_folder(parent) is not None
    assert store.get_notebook(nb) is not None

    # Recursive delete removes the whole subtree and the notebooks within it.
    assert store.delete_folder(parent, recursive=True) is True
    assert store.get_folder(parent) is None
    assert store.get_folder(child) is None
    assert store.get_notebook(nb) is None

    # An empty folder deletes without recursion.
    empty = store.create_folder("Empty")
    assert store.delete_folder(empty, recursive=False) is True
    assert store.get_folder(empty) is None


def test_notebook_folder_placement(store: Store) -> None:
    folder = store.create_folder("Home things")
    nb = store.create_notebook("Report", folder_id=folder)
    summary = next(s for s in store.list_notebooks() if s["id"] == nb)
    assert summary["folder_id"] == folder

    # Move it back to the root, then into the folder again.
    store.move_notebook(nb, None)
    assert store.get_notebook(nb).folder_id is None
    store.move_notebook(nb, folder)
    assert store.get_notebook(nb).folder_id == folder


def test_deleting_a_conversation_removes_its_messages_first(tmp_path: Path) -> None:
    """The child rows must reach the database before the parent row is deleted.

    Asserted as statement order rather than as an end state, because the end state is
    reached either way on SQLite (which does not enforce foreign keys by default), while
    Postgres rejects the transaction outright when the parent goes first. The unit of
    work orders deletes from mapper relationships, and there is none between these two
    tables, so without an explicit flush the order is arbitrary: it passed locally and
    returned 500 on a real deployment.
    """
    store = open_store(f"sqlite:{tmp_path / 'order.db'}")
    try:
        conversation_id = store.create_conversation("doomed")
        store.add_message(conversation_id, "user", content="hello")
        store.add_message(conversation_id, "assistant", content="hi")

        statements: list[str] = []

        def record(conn, cursor, statement, *args) -> None:  # type: ignore[no-untyped-def]
            if statement.lstrip().upper().startswith("DELETE"):
                statements.append(" ".join(statement.split())[:40])

        event.listen(store._engine, "before_cursor_execute", record)
        try:
            assert store.delete_conversation(conversation_id) is True
        finally:
            event.remove(store._engine, "before_cursor_execute", record)

        targets = [s.split()[2] for s in statements]
        assert "message" in targets, f"no message delete was emitted: {statements}"
        assert "conversation" in targets, f"no conversation delete: {statements}"
        assert targets.index("message") < targets.index("conversation"), (
            f"the parent row was deleted before its children: {statements}"
        )
    finally:
        store.close()


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URI"),
    reason="needs a disposable Postgres; set TEST_POSTGRES_URI to run",
)
def test_deleting_a_conversation_on_postgres_where_the_key_is_enforced() -> None:
    """The same delete, against a database that actually checks the foreign key."""
    from sqlalchemy import text

    uri = os.environ["TEST_POSTGRES_URI"]
    reset = create_engine(sqlalchemy_url(uri))
    with reset.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    reset.dispose()

    store = open_store(uri)
    try:
        conversation_id = store.create_conversation("doomed")
        store.add_message(conversation_id, "user", content="hello")
        store.add_message(conversation_id, "assistant", content="hi")

        assert store.delete_conversation(conversation_id) is True
        assert store.get_conversation(conversation_id) is None
        assert store.messages(conversation_id) == []
    finally:
        store.close()


# -- Derivation trash, restore, and erasure -----------------------------


def test_save_derivation_over_a_trashed_row_erases_it_first(store: Store) -> None:
    store.save_derivation(Derivation(name="probe", origin="agent", question="v1"))
    assert store.trash_derivation("probe") is True
    # A fresh save with the same name is not blocked by the trashed row.
    store.save_derivation(Derivation(name="probe", origin="agent", question="v2"))
    live = store.get_derivation("probe")
    assert live is not None and live.question == "v2"
    actions = [e.action for e in store.list_audit()]
    assert "derivation.erase" in actions


def test_derivation_origin_and_repo_refusal(store: Store) -> None:
    store.save_derivation(Derivation(name="agent_one", origin="agent"))
    store.sync_repo_derivations([Derivation(name="repo_one", origin="repo")])

    assert store.derivation_origin("agent_one") == "agent"
    assert store.derivation_origin("repo_one") == "repo"
    assert store.derivation_origin("missing") is None

    # trash_derivation itself does not special-case origin -- that refusal is
    # the API route's job, made possible by derivation_origin. This just
    # confirms a normal, non-repo row still trashes cleanly on its own.
    assert store.trash_derivation("agent_one") is True
    assert store.get_derivation("agent_one") is None
