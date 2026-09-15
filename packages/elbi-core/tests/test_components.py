"""Tests for ORC component validation and provenance stamping.

Unlike ``derivation.schema.json``, this package's ``component.schema.json`` is
somebody else's standard (OpenReasoningComponents): it is vendored here for
validation only, with no published counterpart under ``spec/`` (see
``tests/test_spec_vendoring.py``), so there is no canonical-drift test here --
the same reason there is none for ``osi.schema.json``.
"""

from __future__ import annotations

import pytest

from elbi_core import (
    ComponentError,
    is_valid_component,
    search_components,
    stamp_provenance,
    validate_component,
)

VALID_COMPONENT = {
    "id": "intelligible/customers_discount",
    "type": "column",
    "scope": {"dataset": "customer_churn"},
    "statement": "discount is a float column ranging from 0% to 35%.",
}


def test_validate_component_accepts_valid() -> None:
    validate_component(VALID_COMPONENT)  # does not raise
    assert is_valid_component(VALID_COMPONENT) is True


def test_validate_component_rejects_and_reports() -> None:
    bad = {"id": "no-namespace", "type": "column", "scope": {"dataset": "d"}}
    with pytest.raises(ComponentError) as excinfo:
        validate_component(bad)
    assert "statement" in str(excinfo.value) or "id" in str(excinfo.value)
    assert is_valid_component(bad) is False


def test_stamp_provenance_fills_gap() -> None:
    stamped = stamp_provenance(
        VALID_COMPONENT, derivation="customers_discount", derivation_version="abc123"
    )
    assert stamped["provenance"] == {
        "derivation": "customers_discount",
        "derivation_version": "abc123",
    }
    # The input is not mutated.
    assert "provenance" not in VALID_COMPONENT


def test_stamp_provenance_preserves_other_fields() -> None:
    """Stamping fills only the two gap fields; a human author/source is kept."""
    authored = {
        **VALID_COMPONENT,
        "provenance": {"source": "human", "author": "analyst@example.com"},
    }
    stamped = stamp_provenance(authored, derivation="x", derivation_version="y")
    assert stamped["provenance"] == {
        "source": "human",
        "author": "analyst@example.com",
        "derivation": "x",
        "derivation_version": "y",
    }


CORPUS = [
    {
        "id": "a/discount",
        "type": "column",
        "scope": {"dataset": "d"},
        "statement": "discount is a float column ranging from 0% to 35%.",
    },
    {
        "id": "a/churn_threshold",
        "type": "threshold_rule",
        "scope": {"dataset": "d"},
        "statement": "Customers receiving discounts above 20% churn more often.",
    },
    {
        "id": "a/tenure_shape",
        "type": "model_component",
        "scope": {"dataset": "d"},
        "statement": "Tenure has a nonlinear protective effect on churn.",
    },
]


def test_search_components_ranks_by_bm25_over_statement() -> None:
    results = search_components("discount churn", CORPUS, limit=5)
    assert results
    assert results[0]["id"] == "a/churn_threshold"  # matches both query terms


def test_search_components_abstains_on_empty_query_or_corpus() -> None:
    assert search_components("", CORPUS) == []
    assert search_components("discount", []) == []


def test_search_components_respects_limit() -> None:
    results = search_components("customers discount churn tenure", CORPUS, limit=1)
    assert len(results) == 1


class _FakeEmbedder:
    """Embeds by a fixed lookup so fusion can be tested deterministically."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts):
        return [self._vectors[text] for text in texts]


def test_search_components_fuses_with_an_embedder() -> None:
    # No lexical overlap with any statement, but the fake embedder places the query
    # closest to tenure_shape's vector.
    vectors = {
        "protective effect of age on retention": [1.0, 0.0],
        CORPUS[0]["statement"]: [0.0, 1.0],
        CORPUS[1]["statement"]: [0.1, 0.9],
        CORPUS[2]["statement"]: [0.95, 0.05],
    }
    results = search_components(
        "protective effect of age on retention",
        CORPUS,
        limit=1,
        embedder=_FakeEmbedder(vectors),
    )
    assert results[0]["id"] == "a/tenure_shape"


def test_stamp_provenance_does_not_overwrite_existing_derivation() -> None:
    """A component that already names its own derivation is left alone: this only
    fills gaps, it never overwrites a producer's own claim.
    """
    authored = {
        **VALID_COMPONENT,
        "provenance": {"derivation": "other_system_job", "derivation_version": "v9"},
    }
    stamped = stamp_provenance(authored, derivation="x", derivation_version="y")
    assert stamped["provenance"] == {
        "derivation": "other_system_job",
        "derivation_version": "v9",
    }
