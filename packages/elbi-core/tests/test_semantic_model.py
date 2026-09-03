"""Tests for SemanticModel, the governed semantic model bound as an input."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from elbi_core import SemanticModel
from elbi_core.errors import MetricError
from elbi_core.metrics import Measure
from elbi_core.metrics.osi import OSI_VERSION


def _document() -> dict[str, Any]:
    """A fresh conformant OSI document, so a mutating test cannot leak."""
    return {
        "version": OSI_VERSION,
        "semantic_model": [
            {
                "name": "sales_semantics",
                "datasets": [
                    {
                        "name": "sales",
                        "source": "sales",
                        "fields": [
                            {
                                "name": "region",
                                "expression": {
                                    "dialects": [
                                        {"dialect": "ANSI_SQL", "expression": "region"}
                                    ]
                                },
                                "dimension": {"is_time": False},
                            }
                        ],
                    }
                ],
                "metrics": [
                    {
                        "name": "total_amount",
                        "expression": {
                            "dialects": [
                                {"dialect": "ANSI_SQL", "expression": "SUM(amount)"}
                            ]
                        },
                    }
                ],
            }
        ],
    }


def test_from_osi_takes_its_name_from_the_document() -> None:
    assert SemanticModel.from_osi(_document()).name == "sales_semantics"


def test_name_comes_from_the_first_model_not_the_last() -> None:
    # Several models: the name is the first one's, while every model's metrics import.
    document = _document()
    second = {
        "name": "second_semantics",
        "datasets": [{"name": "refunds", "source": "refunds", "fields": []}],
        "metrics": [
            {
                "name": "total_refunds",
                "expression": {
                    "dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(amount)"}]
                },
            }
        ],
    }
    document["semantic_model"].append(second)

    model = SemanticModel.from_osi(document)
    assert model.name == "sales_semantics"
    assert {m.name for m in model.metrics.metrics} == {
        "total_amount",
        "total_refunds",
    }


def test_explicit_name_overrides_the_document() -> None:
    model = SemanticModel.from_osi(_document(), name="governed_sales")
    assert model.name == "governed_sales"


def test_from_osi_parses_the_definitions() -> None:
    model = SemanticModel.from_osi(_document())
    metric = model.metrics.metric("total_amount")
    assert metric is not None
    assert metric.measure == Measure(agg="sum", column="amount")
    assert metric.dimensions == ("region",)


def test_from_osi_rejects_a_nonconformant_document() -> None:
    document = _document()
    document["version"] = "9.9.9"
    with pytest.raises(MetricError, match="not schema-conformant"):
        SemanticModel.from_osi(document)


def test_from_osi_rejects_a_document_naming_no_model() -> None:
    with pytest.raises(ValueError, match="names no semantic model"):
        SemanticModel.from_osi({"version": OSI_VERSION, "semantic_model": []})


def test_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        SemanticModel.from_osi(_document(), name="")


def test_document_is_a_defensive_copy() -> None:
    # The document versions every derivation reading it, so a caller must not be able
    # to mutate this model through the property.
    model = SemanticModel.from_osi(_document())
    model.document["semantic_model"][0]["name"] = "tampered"
    assert model.document["semantic_model"][0]["name"] == "sales_semantics"


def test_documents_differing_only_in_key_order_are_equal() -> None:
    # Canonical JSON is what makes one definition produce one cache entry.
    plain = _document()
    reordered = {"semantic_model": plain["semantic_model"], "version": plain["version"]}
    assert SemanticModel.from_osi(plain) == SemanticModel.from_osi(reordered)


def test_is_frozen_and_hashable() -> None:
    model = SemanticModel.from_osi(_document())
    assert hash(model) == hash(SemanticModel.from_osi(_document()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.name = "other"  # type: ignore[misc]
