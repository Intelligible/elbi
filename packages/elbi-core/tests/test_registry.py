"""Tests for the Registry."""

from __future__ import annotations

import pytest

from elbi_core import (
    Bm25Retriever,
    Context,
    DuplicateDerivationError,
    Registry,
    UnknownDerivationError,
    derivation,
    serve,
)


@pytest.fixture
def bm25_registry() -> Registry:
    """A registry pinned to lexical BM25, for exercising lexical ranking offline.

    The default retriever is hybrid, whose semantic half would load an embedding
    model; these ranking tests assert BM25 behavior, so they pin it directly.
    """
    return Registry(retriever=Bm25Retriever())


def test_register_and_get(registry: Registry) -> None:
    @derivation(serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    assert registry.get("a") is a
    assert "a" in registry
    assert registry.names() == ["a"]
    assert len(registry) == 1
    assert list(registry) == [a]


def test_duplicate_name_raises(registry: Registry) -> None:
    @derivation(name="dup", serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    with pytest.raises(DuplicateDerivationError):

        @derivation(name="dup", serve=serve.text(), registry=registry)
        def b(ctx: Context) -> str:
            return "b"


def test_unknown_name_raises(registry: Registry) -> None:
    with pytest.raises(UnknownDerivationError, match="no derivation named"):
        registry.get("missing")


def test_clear(registry: Registry) -> None:
    @derivation(serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    registry.clear()
    assert len(registry) == 0


def test_reregister_same_object_is_idempotent(registry: Registry) -> None:
    @derivation(name="x", serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    registry.register(a)  # same object, no error
    assert len(registry) == 1


def test_replace_overwrites_without_error(registry: Registry) -> None:
    @derivation(name="x", serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    from dataclasses import replace

    promoted = replace(a, description="changed")
    registry.replace(promoted)  # same name, different object: no error
    assert registry.get("x").description == "changed"
    assert len(registry) == 1


def test_remove_deletes_and_unknown_raises(registry: Registry) -> None:
    @derivation(name="x", serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    registry.remove("x")
    assert "x" not in registry
    with pytest.raises(UnknownDerivationError, match="no derivation named"):
        registry.remove("x")


def _searchable(registry: Registry) -> None:
    @derivation(name="churn_risk", serve=serve.text(), registry=registry)
    def churn_risk(ctx: Context) -> str:
        """Per-customer churn risk scores."""
        return "a"

    @derivation(name="revenue_by_region", serve=serve.text(), registry=registry)
    def revenue_by_region(ctx: Context) -> str:
        """Total revenue grouped by region."""
        return "b"


def test_search_ranks_name_match_first(bm25_registry: Registry) -> None:
    _searchable(bm25_registry)
    results = bm25_registry.search("revenue")
    assert results[0].name == "revenue_by_region"


def test_search_matches_description(bm25_registry: Registry) -> None:
    _searchable(bm25_registry)
    names = {d.name for d in bm25_registry.search("customer")}
    assert names == {"churn_risk"}


def test_search_empty_query_returns_nothing(bm25_registry: Registry) -> None:
    _searchable(bm25_registry)
    assert bm25_registry.search("   ") == []


def test_search_respects_limit(bm25_registry: Registry) -> None:
    _searchable(bm25_registry)
    # Both match "by"/"scores"? Use a shared term and a limit of 1.
    assert len(bm25_registry.search("region revenue churn", limit=1)) == 1
