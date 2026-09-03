"""``GET /api/search``: the one endpoint the palette and the agent both read.

Driven through the app rather than the index, because what goes wrong here is at the
seam: a filter applied to the wrong side of retrieval, a facet counting the wrong
population, a snippet carrying text the caller may not read.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("duckdb")

from sqlmodel import Session

from elbi.app import create_app
from elbi.db import MetricRow, Secret, Store, open_store
from elbi.search.build import SearchBuilder
from elbi.search.index import SearchIndex
from elbi.search.schema import SearchDoc
from elbi.search.sources import Sources

DIM = 4


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


@pytest.fixture
def index(tmp_path: Path) -> Iterator[SearchIndex]:
    ix = SearchIndex(tmp_path / "search.duckdb", dim=DIM, embedding_model="stub")
    ix.open()
    yield ix
    ix.close()


@pytest.fixture
def client(store: Store, index: SearchIndex) -> Iterator[TestClient]:
    app = create_app(
        load_datasets=dict,
        store=store,
        search_builder=SearchBuilder(index, Sources(store=store)),
    )
    with TestClient(app) as http:
        yield http


def _metric(entity_id: str, title: str, **kw: object) -> SearchDoc:
    kw.setdefault("entity_type", "metric")
    return SearchDoc(entity_id=entity_id, title=title, **kw)  # type: ignore[arg-type]


def _metric_row(store: Store, name: str) -> None:
    """A metric row in the store, so the index pass has something to read."""
    with Session(store._engine) as session:
        session.add(MetricRow(name=name, manifest_json="{}", source=""))
        session.commit()


def _seed(index: SearchIndex, store: Store, docs: list[SearchDoc]) -> None:
    """Index documents."""
    index.upsert(docs)


# --- The result row -----------------------------------------------------------------


def test_a_result_says_what_it_is_and_where_to_go(
    client: TestClient, index: SearchIndex
) -> None:
    """AC1: type, name, a snippet, verdict where applicable, and a navigable target."""
    index.upsert(
        [
            _metric(
                "rev",
                "revenue by region",
                body="Total revenue grouped by sales region and cohort.",
                verdict="sound",
                route="/metrics/rev",
            )
        ]
    )

    item = client.get("/api/search", params={"q": "cohort"}).json()["items"][0]

    assert item["type"] == "metric"
    assert item["name"] == "revenue by region"
    assert item["verdict"] == "sound"
    assert item["route"] == "/metrics/rev"
    assert "cohort" in item["snippet"]


def test_a_snippet_moves_to_the_terms_that_matched(
    client: TestClient, index: SearchIndex
) -> None:
    """A truncated description reads the same for every query; a snippet does not."""
    body = "Opening prose. " * 20 + "Then we filtered to Q3 and computed churn."
    index.upsert([_metric("long", "quarterly notes", body=body)])

    hit = client.get("/api/search", params={"q": "churn"}).json()["items"][0]

    assert "churn" in hit["snippet"]
    assert len(hit["snippet"]) < len(body)


def test_an_empty_query_returns_nothing_rather_than_everything(
    client: TestClient, index: SearchIndex
) -> None:
    index.upsert([_metric("rev", "revenue")])

    assert client.get("/api/search", params={"q": "  "}).json()["items"] == []


# --- Authorization, through the endpoint --------------------------------------------


def test_a_secret_is_not_searchable_by_content_or_by_name(
    client: TestClient, store: Store, index: SearchIndex
) -> None:
    """AC6: a column declared never-index stays out of the built document.

    Built through the store and a real build, so this asserts the extractor never asks
    for the column rather than that a hand-written document happened to omit it.
    """
    store.save_secret(
        Secret(
            name="stripe_key",
            value="sk-live-4242-do-not-index",
            description="the production stripe key",
        )
    )
    SearchBuilder(index, Sources(store=store)).build()

    for query in ("sk-live-4242-do-not-index", "stripe_key"):
        body = client.get("/api/search", params={"q": query}).json()
        assert body["items"] == [], f"{query!r} reached the index"


# --- Facets and paging ---------------------------------------------------------------


def test_a_verdict_filter_returns_only_that_verdict(
    client: TestClient, index: SearchIndex
) -> None:
    """AC7."""
    index.upsert(
        [
            _metric("good", "revenue north", verdict="sound"),
            _metric("bad", "revenue south", verdict="unsound"),
        ]
    )

    body = client.get("/api/search", params={"q": "revenue", "verdict": "sound"}).json()

    assert [i["id"] for i in body["items"]] == ["good"]
    assert all(i["verdict"] == "sound" for i in body["items"])


def test_a_facet_does_not_narrow_its_own_counts(
    client: TestClient, index: SearchIndex
) -> None:
    """A value's count says what choosing it would give, so it survives being chosen.

    Applying the current verdict to the verdict facet collapses every other bucket to
    zero, and the caller can no longer see what switching to it would return.
    """
    index.upsert(
        [
            _metric("good", "revenue north", verdict="sound"),
            _metric("bad", "revenue south", verdict="unsound"),
        ]
    )

    body = client.get("/api/search", params={"q": "revenue", "verdict": "sound"}).json()

    assert body["facets"]["verdict"] == {"sound": 1, "unsound": 1}


def test_facets_count_what_the_query_matched(
    client: TestClient, store: Store, index: SearchIndex
) -> None:
    """A bucket that returns nothing when clicked is worse than no bucket at all."""
    index.upsert(
        [
            _metric("rev", "revenue by cohort"),
            SearchDoc(
                entity_type="derivation",
                entity_id="churn",
                title="churn_risk",
            ),
        ]
    )

    body = client.get("/api/search", params={"q": "cohort"}).json()

    assert body["facets"]["entity_type"] == {"metric": 1}


def test_results_page_with_the_conversations_cursor(
    client: TestClient, index: SearchIndex
) -> None:
    """AC9, and the same envelope: an offset cursor, exhausted by returning ``None``."""
    index.upsert([_metric(f"m{i}", f"revenue {i}") for i in range(5)])

    first = client.get("/api/search", params={"q": "revenue", "limit": 2}).json()
    assert len(first["items"]) == 2 and first["next_page_id"] == "2"

    second = client.get(
        "/api/search", params={"q": "revenue", "limit": 2, "page_id": "2"}
    ).json()
    assert len(second["items"]) == 2 and second["next_page_id"] == "4"

    last = client.get(
        "/api/search", params={"q": "revenue", "limit": 2, "page_id": "4"}
    ).json()
    assert len(last["items"]) == 1 and last["next_page_id"] is None

    seen = {i["id"] for page in (first, second, last) for i in page["items"]}
    assert len(seen) == 5, "a page boundary dropped or repeated a result"


def test_the_cursor_field_is_spelled_the_way_conversations_spells_it(
    client: TestClient, index: SearchIndex
) -> None:
    """Two endpoints naming one cursor differently is a trap for a single client."""
    index.upsert([_metric("rev", "revenue")])

    body = client.get("/api/search", params={"q": "revenue"}).json()

    assert "next_page_id" in body
    assert "nextPageId" not in body


def test_search_is_unavailable_rather_than_broken_without_an_index(
    store: Store,
) -> None:
    """``serve`` treats an unopenable index as a degradation; the route says so."""
    app = create_app(load_datasets=dict, store=store)
    with TestClient(app) as http:
        assert http.get("/api/search", params={"q": "revenue"}).status_code == 503


def test_a_facet_counts_entities_not_documents(
    client: TestClient, store: Store, index: SearchIndex
) -> None:
    """Results collapse to one per entity, so a count of documents names an unreachable
    number.

    A derivation is indexed one document per field. Counting rows put "3" beside a facet
    that two results could ever come back from, which is the kind of number that makes a
    reader distrust the rest of the page.
    """
    index.upsert(
        [
            SearchDoc(
                entity_type="derivation",
                entity_id="churn_risk",
                title="churn_risk",
                body=part,
                chunk_index=i,
            )
            for i, part in enumerate(("churn scores", "churn by cohort", "churn notes"))
        ]
    )

    body = client.get("/api/search", params={"q": "churn"}).json()

    assert len(body["items"]) == 1
    assert body["facets"]["entity_type"] == {"derivation": 1}


def test_a_query_that_tokenizes_to_nothing_counts_nothing(
    client: TestClient, index: SearchIndex
) -> None:
    """``###`` is a query the caller typed and got no results for.

    Empty terms meant "no query" to the facet statement, which then counted the whole
    reachable corpus beside an empty result list.
    """
    index.upsert([_metric(f"m{i}", f"revenue {i}") for i in range(5)])

    body = client.get("/api/search", params={"q": "###"}).json()

    assert body["items"] == []
    assert body["facets"]["entity_type"] == {}


def test_paging_reaches_past_the_candidate_window(
    client: TestClient, index: SearchIndex
) -> None:
    """The window caps the ranking statement, so a fixed one strands every deeper page.

    Worse than slow: the facets have no such cap, so the page advertised a total the
    cursor could never walk to.
    """
    index.upsert([_metric(f"m{i}", f"revenue {i}") for i in range(240)])

    seen: set[str] = set()
    cursor: str | None = None
    for _ in range(10):
        params = {"q": "revenue", "limit": 50}
        if cursor:
            params["page_id"] = cursor
        page = client.get("/api/search", params=params).json()
        seen |= {i["id"] for i in page["items"]}
        cursor = page["next_page_id"]
        if cursor is None:
            break

    assert len(seen) == 240, "a page past the default window was unreachable"
    assert cursor is None


def test_a_page_fills_past_an_entity_that_fills_the_window(
    client: TestClient, store: Store, index: SearchIndex
) -> None:
    """Chunks collapse to one result *after* fusion, so documents are not results.

    An entity with enough chunks to fill the candidate window left the page holding one
    item and no cursor, while entities that matched sat below the cut. The window has to
    widen until the page is full or the rankings are spent.
    """
    _metric_row(store, "loud")
    _metric_row(store, "quiet")
    chunky = [
        _metric("loud", "revenue chatter", chunk_index=n, body="revenue " * 5)
        for n in range(400)
    ]
    _seed(index, store, [*chunky, _metric("quiet", "revenue summary")])

    body = client.get(
        "/api/search",
        params={"q": "revenue", "limit": 2},
        headers={"x-test-user": "alice"},
    ).json()

    assert {item["id"] for item in body["items"]} == {"loud", "quiet"}, (
        f"the chunk-heavy entity crowded the window: {body['items']}"
    )


def test_a_facet_counts_a_result_the_dense_half_found(
    store: Store, index: SearchIndex
) -> None:
    """A semantic hit shares no word with the query, so postings do not know it.

    Counted from the lexical side alone, such a result appeared in the page and in no
    bucket, and the numbers beside the results stopped describing them.
    """
    _metric_row(store, "hers")
    vector = [1.0] + [0.0] * (DIM - 1)
    index.upsert([_metric("hers", "margin study")], [vector])

    lexical_only = index.facet_counts("entity_type", query="revenue")
    both = index.facet_counts("entity_type", query="revenue", vector=vector)

    assert lexical_only == [], "the query shares no word with it, so postings miss it"
    assert both == [("metric", 1)], "the dense half found it and the facet did not"


def test_the_route_asks_the_dense_half_too(
    store: Store, index: SearchIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Hybrid-ranked" has to mean both halves ran.

    Passing no vector makes ``search`` fall through to BM25 in silence, so the endpoint
    reads as working while every stored embedding and the HNSW index over them go
    unread -- and a query that means the right thing without sharing a word finds
    nothing. Asserted on the call rather than on the ranking, because a stub embedder
    that happens to agree with BM25 would hide it.
    """

    class _Embedder:
        def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
            return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    builder = SearchBuilder(index, Sources(store=store), _Embedder())
    seen: dict[str, object] = {}
    original = index.search

    def _record(*args: object, **kw: object) -> object:
        seen.update(kw)
        return original(*args, **kw)

    monkeypatch.setattr(index, "search", _record)
    app = create_app(load_datasets=dict, store=store, search_builder=builder)
    with TestClient(app) as http:
        http.get("/api/search", params={"q": "revenue"})

    assert seen.get("vector") is not None, "the dense half was never asked"


def test_an_untrusted_index_serves_no_row_and_no_body(
    store: Store, index: SearchIndex
) -> None:
    """Ranking refused already; the readers past it did not.

    ``set_trusted(False)`` is the last rung of the fail-closed ladder and means nothing
    here may be served. A caller holding ids from an earlier request reached the rows
    straight out of ``hydrate``.
    """
    _metric_row(store, "hers")
    _seed(index, store, [_metric("hers", "revenue by region", body="the margin fell")])
    assert index.hydrate(["metric:hers#0"])

    index.set_trusted(False)

    assert index.search("revenue") == []
    assert index.hydrate(["metric:hers#0"]) == []


def test_a_snippet_finds_a_term_inside_a_compound_name() -> None:
    """The ranker splits ``churn_risk`` into two terms; the snippet took only the first.

    So a query the ranking matched on ``risk`` found no span here, and a long text fell
    back to opening at its start rather than at the hit. Offsets have to come from the
    same tokenizer the ranking uses.
    """
    from elbi.search.snippet import snippet

    padding = "unrelated filler text " * 40
    text = f"{padding}the churn_risk cohort moved sharply"

    result = snippet(text, "risk")

    assert "churn_risk" in result, f"the snippet opened away from the match: {result!r}"
