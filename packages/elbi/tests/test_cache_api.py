"""``POST /api/cache/invalidate``: busting the derivation cache over HTTP.

Driven through the real route over a real on-disk cache store, so what is checked is
which entries are gone afterwards, not which function was called. The last tests build
the app the way ``elbi serve`` does, to hold the route to the project's own cache.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from elbi import create_app, extensions
from elbi.db import Store, open_store
from elbi_core import Artifact
from elbi_core.cache import CachedResult, LocalCacheStore, derivation_tag


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = open_store(f"sqlite:{tmp_path / 'app.db'}")
    yield opened
    opened.close()


def _put(cache: LocalCacheStore, key: str, *tags: str) -> None:
    cache.put(
        key,
        CachedResult(
            artifact=Artifact.table([{"key": key}]),
            output_version=key,
            computed_at=time.time(),
            tags=tags,
        ),
    )


@pytest.fixture
def cache(tmp_path: Path) -> LocalCacheStore:
    cache = LocalCacheStore(tmp_path / "cache")
    _put(cache, "revenue", "finance", derivation_tag("revenue"))
    _put(cache, "churn", derivation_tag("churn"))
    _put(cache, "headcount", "hr", derivation_tag("headcount"))
    return cache


@pytest.fixture
def client(store: Store, cache: LocalCacheStore) -> Iterator[TestClient]:
    app = create_app(
        load_datasets=dict, store=store, invalidate_cache=cache.invalidate_tag
    )
    with TestClient(app) as http:
        yield http


def _cached(cache: LocalCacheStore) -> set[str]:
    return {
        key for key in ("revenue", "churn", "headcount") if cache.get(key) is not None
    }


def test_a_tag_removes_only_the_entries_carrying_it(
    client: TestClient, cache: LocalCacheStore
) -> None:
    response = client.post("/api/cache/invalidate", json={"tag": "finance"})
    assert response.status_code == 200, response.text
    assert response.json() == {"removed": 1}
    assert _cached(cache) == {"churn", "headcount"}


def test_a_derivation_name_removes_that_derivations_entries(
    client: TestClient, cache: LocalCacheStore
) -> None:
    response = client.post("/api/cache/invalidate", json={"derivation": "churn"})
    assert response.json() == {"removed": 1}
    assert _cached(cache) == {"revenue", "headcount"}


def test_tag_and_derivation_together_remove_either(
    client: TestClient, cache: LocalCacheStore
) -> None:
    response = client.post(
        "/api/cache/invalidate", json={"tag": "hr", "derivation": "churn"}
    )
    assert response.json() == {"removed": 2}
    assert _cached(cache) == {"revenue"}


def test_an_entry_matched_twice_is_counted_once(
    client: TestClient, cache: LocalCacheStore
) -> None:
    response = client.post(
        "/api/cache/invalidate", json={"tag": "finance", "derivation": "revenue"}
    )
    assert response.json() == {"removed": 1}
    assert _cached(cache) == {"churn", "headcount"}


def test_an_unknown_tag_removes_nothing(
    client: TestClient, cache: LocalCacheStore
) -> None:
    response = client.post("/api/cache/invalidate", json={"tag": "nope"})
    assert response.json() == {"removed": 0}
    assert _cached(cache) == {"revenue", "churn", "headcount"}


@pytest.mark.parametrize(
    "body",
    [{}, {"tag": ""}, {"tag": "  "}, {"derivation": 3}, {"tag": ["a"]}, ["finance"]],
)
def test_a_request_naming_nothing_usable_is_refused_and_clears_nothing(
    client: TestClient, cache: LocalCacheStore, body: object
) -> None:
    """The whole cache is never the fallback for a request that names no target."""
    response = client.post("/api/cache/invalidate", json=body)
    assert response.status_code == 400
    assert _cached(cache) == {"revenue", "churn", "headcount"}


def test_each_invalidation_is_audited(client: TestClient, store: Store) -> None:
    client.post("/api/cache/invalidate", json={"tag": "hr", "derivation": "churn"})
    events = [
        (e.action, e.target_type, e.target_id)
        for e in store.list_audit()
        if e.action == "cache.invalidate"
    ]
    assert sorted(events) == [
        ("cache.invalidate", "cache", "derivation:churn"),
        ("cache.invalidate", "cache", "hr"),
    ]


def test_without_a_cache_the_route_says_so(store: Store) -> None:
    with TestClient(create_app(load_datasets=dict, store=store)) as http:
        response = http.post("/api/cache/invalidate", json={"tag": "finance"})
    assert response.status_code == 503


def test_an_installed_authenticator_gates_the_route(
    store: Store, cache: LocalCacheStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route sits inside the app's middleware stack, not beside it.

    The app ships no authenticator of its own; a deployment installs one through the
    extension seam (or in front of the app). Whatever is installed must see this
    route, so an unauthenticated caller cannot bust the cache by going around it.
    """

    class _BearerExtension:
        def install_extension(self, context: extensions.ExtensionContext) -> None:
            @context.app.middleware("http")
            async def _require_bearer(
                request: Request, call_next: Callable[[Request], Awaitable[Response]]
            ) -> Response:
                if request.headers.get("authorization") != "Bearer s3cret":
                    return Response(status_code=401)
                return await call_next(request)

    monkeypatch.setattr(
        extensions, "load_all", lambda: [("bearer", _BearerExtension())]
    )
    app = create_app(
        load_datasets=dict, store=store, invalidate_cache=cache.invalidate_tag
    )
    with TestClient(app) as http:
        refused = http.post("/api/cache/invalidate", json={"tag": "finance"})
        assert refused.status_code == 401
        assert _cached(cache) == {"revenue", "churn", "headcount"}
        allowed = http.post(
            "/api/cache/invalidate",
            json={"tag": "finance"},
            headers={"Authorization": "Bearer s3cret"},
        )
    assert allowed.json() == {"removed": 1}
    assert _cached(cache) == {"churn", "headcount"}


_SALES = "customer_id,amount\nc1,100\nc2,5\n"


def _served(root: Path) -> FastAPI:
    """The app as ``elbi serve`` assembles it, over a one-source project."""
    pytest.importorskip("deltalake")
    pytest.importorskip("duckdb")
    from elbi.serve import build

    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "elbi.yaml").write_text(
        "project: cachebust\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    return build(root, with_mcp=False)


def test_the_served_app_invalidates_the_projects_own_cache(tmp_path: Path) -> None:
    from elbi_cli.project import load_project

    root = tmp_path / "project"
    app = _served(root)
    cache = LocalCacheStore(load_project(root).cache_dir)
    _put(cache, "revenue", derivation_tag("revenue"))
    _put(cache, "churn", derivation_tag("churn"))
    with TestClient(app) as http:
        response = http.post("/api/cache/invalidate", json={"derivation": "revenue"})
    assert response.json() == {"removed": 1}
    assert cache.get("revenue") is None
    assert cache.get("churn") is not None
