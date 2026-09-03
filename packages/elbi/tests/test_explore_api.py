"""Integration tests for the exploration API (SQL workbench + profiling).

Drive the HTTP surface with a TestClient over a real store and an in-memory dataset
loader: browse the catalog, run SQL over bound datasets through DuckDB, profile a
dataset and a query result, save/list/delete queries, and promote a query to a
derivation (through a stub authoring factory). The LLM client is a stub; explore never
calls it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from elbi import create_app
from elbi.db import Derivation, open_store
from elbi.explore import ExploreService
from elbi.warehouse.service import WarehouseError
from elbi_agent import DeriveOutcome
from elbi_core.errors import ElbiError

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

DATASETS: dict[str, list[dict[str, Any]]] = {
    "sales": [
        {"region": "west", "amount": 10},
        {"region": "west", "amount": 30},
        {"region": "east", "amount": 20},
    ],
    "regions": [
        {"region": "west", "manager": "ana"},
        {"region": "east", "manager": "bo"},
    ],
}


class _Captured:
    """Records the last derivation authored, for the promote wiring test."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.source: str | None = None
        self.deps: list[str] = []
        self.external: tuple[str, str, str] | None = None
        self.drafted: str | None = None
        self.claim: Any = None  # claim the certify path received
        self.drafted_claim: Any = None  # claim the draft path received


class _FakeClient:
    """A minimal LLMClient stub: bare SQL for nl2sql, JSON for the claim-aware draft.

    ``sql``/``claim`` are instance state so a test can vary the model's answer (no
    claim asserted, empty SQL) without leaking into other tests.
    """

    sql = "SELECT region FROM sales ORDER BY region"

    def __init__(self) -> None:
        self.sql = type(self).sql
        self.claim: dict[str, Any] | None = {"x": "region", "y": "amount"}

    def step(self, transcript: Any, tools: Any) -> Any:
        if "JSON object" in (getattr(transcript, "system", "") or ""):
            return SimpleNamespace(
                text=json.dumps({"sql": self.sql, "claim": self.claim})
            )
        return SimpleNamespace(text=f"```sql\n{self.sql}\n```")


@pytest.fixture
def captured() -> _Captured:
    return _Captured()


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


@pytest.fixture
def client(
    tmp_path: Path, captured: _Captured, fake_client: _FakeClient
) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")

    # Names starting with "reject_" stand in for drafts that fail verification.
    def _rejected(name: str) -> bool:
        return name.startswith("reject_")

    def factory(conversation_id: str, question: str) -> Any:
        def derive(
            name: str,
            source: str,
            claim: Any,
            contract: Any,
            fmt: str,
            assumptions: Any,
            deps: Any,
        ) -> DeriveOutcome:
            captured.name = name
            captured.source = source
            captured.deps = list(deps)
            captured.claim = claim
            if _rejected(name):
                return DeriveOutcome(
                    certified=False, verdict="unsound", detail="rejected"
                )
            # Persist like the real bridge, so certified derivations (and only
            # they) reach /api/derivations.
            store.save_derivation(
                Derivation(
                    name=name,
                    conversation_id=conversation_id,
                    question=question,
                    source=source,
                    verdict="sound",
                    rendered="ok",
                    data_hash="h",
                )
            )
            return DeriveOutcome(certified=True, verdict="sound", rendered="ok")

        return derive

    def draft_factory(conversation_id: str, question: str) -> Any:
        def draft(
            name: str,
            source: str,
            claim: Any,
            contract: Any,
            fmt: str,
            deps: Any,
        ) -> dict[str, Any]:
            captured.drafted = name
            captured.drafted_claim = claim
            ok = not _rejected(name)
            return {
                "ok": ok,
                "name": name,
                "certified": False,
                "verdict": "sound" if ok else "unsound",
                "contract_verdict": None,
                "checks": [["effect", "sound" if ok else "unsound", "detail"]],
                "rendered": "| region |\n| west |",
                "detail": "" if ok else "the effect did not survive controls",
                "error": None,
            }

        return draft

    def promote_external(name: str, sql: str, source_id: str) -> dict[str, Any]:
        captured.external = (name, sql, source_id)
        return {"ok": True, "name": name, "certified": True, "verdict": None}

    service = ExploreService(
        store=store,
        dataset_names=lambda: list(DATASETS),
        resolve_source=lambda name: list(DATASETS[name]),
        read_schema=lambda name: (
            list(DATASETS[name][0]),
            len(DATASETS[name]),
        ),
        derive_factory=factory,
        draft_factory=draft_factory,
        promote_external=promote_external,
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=fake_client,
        store=store,
        explore_service=service,
    )
    with TestClient(app) as http:
        yield http


def test_sources_lists_bound_datasets_target(client: TestClient) -> None:
    response = client.get("/api/explore/sources")
    assert response.status_code == 200
    sources = response.json()
    assert sources[0] == {"id": None, "name": "Bound datasets", "kind": "project"}


def test_catalog_reports_dataset_columns(client: TestClient) -> None:
    response = client.get("/api/explore/catalog")
    assert response.status_code == 200
    tables = {t["name"]: t for t in response.json()["tables"]}
    assert set(tables) == {"sales", "regions"}
    assert [c["name"] for c in tables["sales"]["columns"]] == ["region", "amount"]
    assert tables["sales"]["rows"] == 3


def test_query_runs_sql_over_bound_datasets(client: TestClient) -> None:
    response = client.post(
        "/api/explore/query",
        json={
            "sql": "SELECT region, sum(amount) AS total FROM sales "
            "GROUP BY region ORDER BY region"
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["columns"] == ["region", "total"]
    assert body["rows"] == [
        {"region": "east", "total": 20},
        {"region": "west", "total": 40},
    ]
    assert body["truncated"] is False


def test_query_joins_across_datasets(client: TestClient) -> None:
    response = client.post(
        "/api/explore/query",
        json={
            "sql": "SELECT r.manager, sum(s.amount) AS total FROM sales s "
            "JOIN regions r ON r.region = s.region "
            "GROUP BY r.manager ORDER BY r.manager"
        },
    )
    assert response.status_code == 200
    assert response.json()["rows"] == [
        {"manager": "ana", "total": 40},
        {"manager": "bo", "total": 20},
    ]


def test_query_row_cap_flags_truncation(client: TestClient) -> None:
    response = client.post(
        "/api/explore/query",
        json={"sql": "SELECT amount FROM sales ORDER BY amount", "max_rows": 2},
    )
    body = response.json()
    assert body["truncated"] is True
    assert len(body["rows"]) == 2


def test_bad_sql_is_a_400(client: TestClient) -> None:
    response = client.post("/api/explore/query", json={"sql": "SELECT * FROM nope"})
    assert response.status_code == 400


def test_empty_query_is_a_400(client: TestClient) -> None:
    assert client.post("/api/explore/query", json={"sql": "  "}).status_code == 400


def test_profile_a_dataset(client: TestClient) -> None:
    response = client.post("/api/explore/profile", json={"dataset": "sales"})
    assert response.status_code == 200
    body = response.json()
    assert body["rowCount"] == 3
    columns = {c["name"]: c for c in body["columns"]}
    assert columns["region"]["distinct"] == 2
    assert columns["amount"]["inferredType"] == "integer"
    assert columns["region"]["isUnique"] is False


def test_profile_a_query_result(client: TestClient) -> None:
    response = client.post(
        "/api/explore/profile",
        json={"sql": "SELECT DISTINCT region FROM sales"},
    )
    assert response.status_code == 200
    assert response.json()["rowCount"] == 2


def test_nl2sql_drafts_sql_from_a_prompt(client: TestClient) -> None:
    response = client.post(
        "/api/explore/nl2sql",
        json={"prompt": "regions in order"},
    )
    assert response.status_code == 200
    # The markdown fence the model wrapped its answer in is stripped to bare SQL.
    assert response.json()["sql"] == "SELECT region FROM sales ORDER BY region"


def test_nl2sql_requires_a_prompt(client: TestClient) -> None:
    assert client.post("/api/explore/nl2sql", json={"prompt": " "}).status_code == 400


def test_saved_query_roundtrip(client: TestClient) -> None:
    created = client.post(
        "/api/explore/queries",
        json={"name": "top regions", "sql": "SELECT * FROM sales"},
    ).json()
    assert created["name"] == "top regions"
    query_id = created["id"]

    listed = client.get("/api/explore/queries").json()
    assert [q["id"] for q in listed] == [query_id]

    fetched = client.get(f"/api/explore/queries/{query_id}").json()
    assert fetched["sql"] == "SELECT * FROM sales"

    # Saving with the same id updates in place rather than creating a second row.
    updated = client.post(
        "/api/explore/queries",
        json={"id": query_id, "name": "top regions", "sql": "SELECT region FROM sales"},
    ).json()
    assert updated["id"] == query_id
    assert len(client.get("/api/explore/queries").json()) == 1

    assert client.delete(f"/api/explore/queries/{query_id}").status_code == 200
    assert client.get("/api/explore/queries").json() == []
    assert client.get(f"/api/explore/queries/{query_id}").status_code == 404


def test_promote_authors_a_derivation(client: TestClient, captured: _Captured) -> None:
    response = client.post(
        "/api/explore/promote",
        json={
            "name": "west_totals",
            "sql": "SELECT region, sum(amount) AS total FROM sales GROUP BY region",
        },
    )
    assert response.status_code == 200
    assert response.json()["certified"] is True
    # The generated source runs the query over the referenced dataset and declares the
    # engine deps the sandbox provisions.
    assert captured.name == "west_totals"
    assert captured.source is not None
    assert "query_datasets" in captured.source
    assert 'ctx.input("sales").rows' in captured.source
    assert "regions" not in captured.source  # only the referenced dataset is bound
    assert captured.deps == ["duckdb", "pyarrow"]


def test_promote_rejects_invalid_name(client: TestClient) -> None:
    response = client.post(
        "/api/explore/promote",
        json={"name": "Bad Name", "sql": "SELECT * FROM sales"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "invalid derivation name" in body["error"]


def test_promote_external_delegates_to_trusted_path(
    client: TestClient, captured: _Captured
) -> None:
    response = client.post(
        "/api/explore/promote",
        json={"name": "ext_totals", "sql": "SELECT 1", "source_id": "src-1"},
    )
    assert response.status_code == 200
    assert response.json()["certified"] is True
    # External-source promotion goes through the trusted in-process path, not the
    # sandboxed derive_factory used for bound datasets.
    assert captured.external == ("ext_totals", "SELECT 1", "src-1")
    assert captured.source is None


# -- the one-click NL -> draft -> verdict -> certify flow ----------------------
def _derivation_names(client: TestClient) -> set[str]:
    """The names the serving surface exposes (certified derivations only)."""
    return {d["name"] for d in client.get("/api/derivations").json()}


def test_draft_reports_a_verdict_without_certifying(client: TestClient) -> None:
    response = client.post(
        "/api/explore/draft",
        json={
            "name": "west_regions",
            "sql": "SELECT region FROM sales ORDER BY region",
            "flow_id": "f1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["certified"] is False
    assert body["verdict"] == "sound"
    assert body["sql"] == "SELECT region FROM sales ORDER BY region"
    assert body["flowId"] == "f1"  # responses are camelCased at the API boundary
    # A draft is verify-only: it certifies nothing, so it never reaches serving.
    assert "west_regions" not in _derivation_names(client)


def test_draft_from_a_prompt_generates_sql_then_verifies(client: TestClient) -> None:
    response = client.post(
        "/api/explore/draft",
        json={"name": "regions", "prompt": "regions in order"},
    )
    assert response.status_code == 200
    body = response.json()
    # The prompt is grounded into SQL by the model, then that SQL is drafted + verified.
    assert body["sql"] == _FakeClient.sql
    assert body["certified"] is False
    assert body["verdict"] == "sound"


def test_draft_prompt_wins_over_leftover_editor_sql(client: TestClient) -> None:
    response = client.post(
        "/api/explore/draft",
        json={
            "name": "prompted",
            "sql": "SELECT 1 AS example",  # the editor placeholder rides along
            "prompt": "regions in order",
        },
    )
    assert response.status_code == 200
    # The typed question is what gets drafted, not the leftover editor text.
    assert response.json()["sql"] == _FakeClient.sql


def test_draft_prompt_extracts_a_claim_for_the_oracle(
    client: TestClient, captured: _Captured
) -> None:
    body = client.post(
        "/api/explore/draft",
        json={"name": "claimed", "prompt": "does region drive amount?"},
    ).json()
    # The one-completion draft returns SQL plus the claim, and the claim reaches the
    # authoring loop so the oracle rules on the draft, not just the computation.
    assert body["claim"] == {"x": "region", "y": "amount"}
    assert captured.drafted_claim == {"x": "region", "y": "amount"}


def test_certify_carries_the_claim_through(
    client: TestClient, captured: _Captured
) -> None:
    client.post(
        "/api/explore/certify",
        json={
            "name": "claimed_certify",
            "sql": "SELECT region, amount FROM sales ORDER BY region",
            "claim": {"x": "region", "y": "amount"},
        },
    )
    # The certified derivation keeps the claim, so its verdict is the oracle's.
    assert captured.claim == {"x": "region", "y": "amount"}


def test_draft_prompt_without_a_relationship_stays_claimless(
    client: TestClient, fake_client: _FakeClient, captured: _Captured
) -> None:
    fake_client.claim = None  # the model asserts nothing statistical
    body = client.post(
        "/api/explore/draft",
        json={"name": "plain", "prompt": "top regions by amount"},
    ).json()
    # No claim is invented; the flow degrades to computation-only verification.
    assert body["claim"] is None
    assert captured.drafted_claim is None


def test_draft_accepts_an_explicit_claim_with_sql(
    client: TestClient, captured: _Captured
) -> None:
    client.post(
        "/api/explore/draft",
        json={
            "name": "api_claimed",
            "sql": "SELECT region, amount FROM sales ORDER BY region",
            "claim": {"x": "region", "y": "amount"},
        },
    )
    # An API caller can declare the claim directly, without the NL path.
    assert captured.drafted_claim == {"x": "region", "y": "amount"}


def test_draft_ignores_a_malformed_claim(
    client: TestClient, captured: _Captured
) -> None:
    response = client.post(
        "/api/explore/draft",
        json={
            "name": "malformed",
            "sql": "SELECT region FROM sales",
            "claim": "region drives amount",  # not the roles object
        },
    )
    # A malformed claim degrades to a claimless draft rather than an error.
    assert response.status_code == 200
    assert captured.drafted_claim is None


def test_draft_400s_when_the_model_returns_no_sql(
    client: TestClient, fake_client: _FakeClient
) -> None:
    fake_client.sql = ""
    response = client.post(
        "/api/explore/draft",
        json={"name": "empty", "prompt": "anything at all"},
    )
    assert response.status_code == 400
    assert "could not draft" in response.json()["detail"]


def test_drafted_name_serves_only_after_certify(client: TestClient) -> None:
    client.post(
        "/api/explore/draft",
        json={"name": "gated", "sql": "SELECT region FROM sales"},
    )
    # The serving surface has no such derivation until the human certifies.
    assert client.get("/api/derivations/gated").status_code == 404
    client.post(
        "/api/explore/certify",
        json={"name": "gated", "sql": "SELECT region FROM sales"},
    )
    assert client.get("/api/derivations/gated").status_code == 200


@given(text=st.text(max_size=300))
def test_parse_draft_never_raises_and_never_invents_columns(text: str) -> None:
    from elbi.nl2sql import _parse_draft

    tables = [{"name": "t", "columns": [{"name": "a"}, {"name": "b"}]}]
    sql, claim = _parse_draft(text, tables)
    assert isinstance(sql, str)
    if claim is not None:
        # Any claim that survives parsing names only real columns.
        assert {claim["x"], claim["y"]} <= {"a", "b"}
        assert set(claim.get("controls", [])) <= {"a", "b"}


def test_warehouse_error_is_a_domain_error() -> None:
    """Catalog skipping and draft refusal both depend on this subclassing."""
    assert issubclass(WarehouseError, ElbiError)


def test_catalog_skips_a_deleted_warehouse_table(tmp_path: Path) -> None:
    """One unreadable (just-deleted) table never hides the rest of the catalog."""
    store = open_store(f"sqlite:{tmp_path / 'skip.db'}")

    def read_schema(name: str) -> tuple[list[str], int]:
        if name == "ghost":
            raise WarehouseError(f"warehouse table {name!r} not found")
        return (["region"], 3)

    service = ExploreService(
        store=store,
        dataset_names=lambda: ["sales", "ghost"],
        resolve_source=lambda name: [],
        read_schema=read_schema,
    )
    assert [t["name"] for t in service.catalog()["tables"]] == ["sales"]


def test_draft_and_certify_refuse_a_deleted_table(tmp_path: Path) -> None:
    """A table deleted mid-session is a clean refusal, never a server error."""
    store = open_store(f"sqlite:{tmp_path / 'gone.db'}")

    def resolve(name: str) -> Any:
        raise WarehouseError(f"warehouse table {name!r} not found")

    service = ExploreService(
        store=store,
        dataset_names=lambda: ["ghost"],
        resolve_source=resolve,
        read_schema=lambda name: ([], 0),
        derive_factory=lambda c, q: lambda *a: DeriveOutcome(certified=True),
        draft_factory=lambda c, q: lambda *a: {"ok": True},
    )
    drafted = service.draft("gone", "SELECT region FROM ghost")
    assert drafted["ok"] is False
    assert "not found" in drafted["error"]
    promoted = service.promote("gone", "SELECT region FROM ghost")
    assert promoted["ok"] is False
    assert "not found" in promoted["error"]


def test_parse_draft_degrades_to_bare_sql() -> None:
    from elbi.nl2sql import _parse_draft

    sql, claim = _parse_draft("SELECT 1 AS one", [])
    assert sql == "SELECT 1 AS one" and claim is None


def test_parse_draft_drops_a_claim_over_unknown_columns() -> None:
    from elbi.nl2sql import _parse_draft

    tables = [{"name": "t", "columns": [{"name": "a"}]}]
    sql, claim = _parse_draft(
        json.dumps({"sql": "SELECT a FROM t", "claim": {"x": "a", "y": "ghost"}}),
        tables,
    )
    assert sql == "SELECT a FROM t" and claim is None


def test_parse_draft_cleans_fenced_sql_inside_the_json() -> None:
    from elbi.nl2sql import _parse_draft

    sql, _claim = _parse_draft(
        json.dumps({"sql": "```sql\nSELECT a FROM t;\n```", "claim": None}), []
    )
    assert sql == "SELECT a FROM t"


def test_parse_draft_drops_a_degenerate_self_claim() -> None:
    from elbi.nl2sql import _parse_draft

    tables = [{"name": "t", "columns": [{"name": "a"}, {"name": "b"}]}]
    # A column cannot drive itself, and x/y never double as their own controls.
    _sql, claim = _parse_draft(
        json.dumps({"sql": "SELECT a FROM t", "claim": {"x": "a", "y": "a"}}), tables
    )
    assert claim is None
    _sql, claim = _parse_draft(
        json.dumps(
            {
                "sql": "SELECT a, b FROM t",
                "claim": {"x": "a", "y": "b", "controls": ["a", "b"]},
            }
        ),
        tables,
    )
    assert claim == {"x": "a", "y": "b"}  # overlapping controls dropped


def test_certify_refuses_a_claim_on_an_external_source(
    client: TestClient, captured: _Captured
) -> None:
    body = client.post(
        "/api/explore/certify",
        json={
            "name": "ext_claimed",
            "sql": "SELECT 1",
            "source_id": "src-1",
            "claim": {"x": "region", "y": "amount"},
        },
    ).json()
    # The trusted external path runs no oracle: refuse rather than certify around
    # a claim that would go silently unverified.
    assert body["ok"] is False
    assert body["certified"] is False
    assert "bound-dataset" in body["error"]
    assert captured.external is None  # never delegated


def test_certify_certifies_a_reviewed_draft(client: TestClient) -> None:
    draft = client.post(
        "/api/explore/draft",
        json={
            "name": "region_totals",
            "sql": "SELECT region FROM sales",
            "flow_id": "f2",
        },
    ).json()
    assert draft["ok"] is True and draft["certified"] is False
    assert "region_totals" not in _derivation_names(client)

    certified = client.post(
        "/api/explore/certify",
        json={
            "name": "region_totals",
            "sql": "SELECT region FROM sales",
            "flow_id": "f2",
        },
    ).json()
    assert certified["certified"] is True
    # One click later, the certified derivation is on the serving surface.
    assert "region_totals" in _derivation_names(client)


def test_rejected_draft_is_never_certified_or_served(client: TestClient) -> None:
    draft = client.post(
        "/api/explore/draft",
        json={"name": "reject_bogus", "sql": "SELECT region FROM sales"},
    ).json()
    assert draft["ok"] is False
    assert draft["verdict"] == "unsound"

    # Even if a client posts the certify call for a rejected draft, it is never
    # certified and never reaches the serving surface.
    certified = client.post(
        "/api/explore/certify",
        json={"name": "reject_bogus", "sql": "SELECT region FROM sales"},
    ).json()
    assert certified["certified"] is False
    assert "reject_bogus" not in _derivation_names(client)


def test_draft_rejects_an_invalid_name(client: TestClient) -> None:
    body = client.post(
        "/api/explore/draft",
        json={"name": "Bad Name", "sql": "SELECT region FROM sales"},
    ).json()
    assert body["ok"] is False
    assert "invalid derivation name" in body["error"]


def test_draft_rejects_an_external_source(client: TestClient) -> None:
    body = client.post(
        "/api/explore/draft",
        json={"name": "ext_draft", "sql": "SELECT 1", "source_id": "src-1"},
    ).json()
    assert body["ok"] is False
    assert "bound-dataset" in body["error"]


def test_draft_requires_a_name(client: TestClient) -> None:
    response = client.post("/api/explore/draft", json={"sql": "SELECT 1"})
    assert response.status_code == 400


def test_draft_requires_a_prompt_or_sql(client: TestClient) -> None:
    response = client.post("/api/explore/draft", json={"name": "x"})
    assert response.status_code == 400


def test_loop_latency_is_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        client.post(
            "/api/explore/draft",
            json={"name": "logged", "sql": "SELECT region FROM sales", "flow_id": "f3"},
        )
        client.post(
            "/api/explore/certify",
            json={"name": "logged", "sql": "SELECT region FROM sales", "flow_id": "f3"},
        )

    draft_logs = [r for r in caplog.records if r.getMessage() == "explore.draft"]
    certify_logs = [r for r in caplog.records if r.getMessage() == "explore.certify"]
    assert len(draft_logs) == 1
    assert len(certify_logs) == 1
    # End-to-end loop latency is recorded, correlated across the two calls by flow_id.
    assert draft_logs[0].flow_id == "f3"
    assert isinstance(draft_logs[0].draft_ms, float)
    assert isinstance(draft_logs[0].verify_ms, float)
    assert certify_logs[0].flow_id == "f3"
    assert isinstance(certify_logs[0].certify_ms, float)


def test_a_table_synced_this_session_is_queryable_and_promotable(
    client: TestClient, captured: _Captured
) -> None:
    """The dataset namespace is read per call, not captured when the app was built.

    Syncing a warehouse source adds a table while the app is running. Held as a list
    snapshotted at construction, that table was absent from the schema browser and,
    worse, silently unbound when a query naming it was promoted -- the generated
    derivation bound nothing, so the sandbox refused it as an unknown table. Nothing
    caught it, because every test built its service with the namespace already final.
    """
    DATASETS["shipments"] = [{"region": "west", "parcels": 4}]
    try:
        catalog = client.get("/api/explore/catalog").json()
        assert "shipments" in {t["name"] for t in catalog["tables"]}

        promoted = client.post(
            "/api/explore/promote",
            json={
                "name": "parcels_by_region",
                "sql": "select region, sum(parcels) from shipments group by region",
            },
        )
        assert promoted.status_code == 200, promoted.text
        assert promoted.json()["certified"] is True
        # The binding is what the sandbox reads; without it the table is unknown there.
        assert 'ctx.input("shipments")' in (captured.source or "")
    finally:
        DATASETS.pop("shipments", None)
