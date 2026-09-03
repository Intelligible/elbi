"""Tests for the DashboardSpec: manifest round-tripping and validation.

A dashboard is a declarative artifact, so the invariants that matter are that a valid
manifest round-trips through the dataclasses unchanged, and that the checks the JSON
Schema cannot express (unique ids, references that resolve, and the per-widget-type
shape) are actually enforced. Each negative case reverts one such rule and asserts the
specific violation is reported.
"""

from __future__ import annotations

import pytest

from elbi_core import DashboardSpec
from elbi_core.dashboard import is_valid_dashboard, validate_dashboard
from elbi_core.errors import SpecValidationError


def _manifest() -> dict:
    return {
        "specVersion": "1.0",
        "kind": "Dashboard",
        "name": "revenue_health",
        "title": "Revenue Health",
        "variables": [
            {
                "name": "region",
                "type": "string",
                "control": "multiselect",
                "default": ["west"],
                "options": {"derivation": "regions", "column": "region"},
            }
        ],
        "pages": [
            {
                "name": "overview",
                "title": "Overview",
                "widgets": [
                    {
                        "id": "region_filter",
                        "type": "filter",
                        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 2},
                        "variable": "region",
                    },
                    {
                        "id": "mrr",
                        "type": "metric",
                        "gridPos": {"x": 0, "y": 2, "w": 6, "h": 4},
                        "bind": {
                            "derivation": "monthly_revenue",
                            "params": {"region": "$region"},
                        },
                        "viz": {"field": "mrr", "format": "currency"},
                        "interactions": {
                            "drillThrough": {
                                "target": "page:detail",
                                "carry": ["region"],
                            }
                        },
                    },
                ],
            },
            {
                "name": "detail",
                "widgets": [
                    {
                        "id": "rows",
                        "type": "table",
                        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 10},
                        "bind": {"derivation": "revenue_rows"},
                    }
                ],
            },
        ],
    }


def test_manifest_round_trips() -> None:
    spec = DashboardSpec.from_manifest(_manifest())
    assert spec.name == "revenue_health"
    assert spec.variable("region").control == "multiselect"
    assert spec.page("overview").title == "Overview"
    # A parsed-then-serialized manifest re-validates and re-parses to an equal spec.
    assert DashboardSpec.from_manifest(spec.to_manifest()) == spec


def test_derivation_names_covers_binds_and_dynamic_options() -> None:
    spec = DashboardSpec.from_manifest(_manifest())
    assert set(spec.derivation_names()) == {
        "regions",
        "monthly_revenue",
        "revenue_rows",
    }


def test_is_valid_dashboard_accepts_the_example() -> None:
    assert is_valid_dashboard(_manifest())


def test_duplicate_widget_id_rejected() -> None:
    manifest = _manifest()
    manifest["pages"][1]["widgets"][0]["id"] = "mrr"  # collides with page 1
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("not unique" in message for message in err.value.messages)


def test_param_referencing_unknown_variable_rejected() -> None:
    manifest = _manifest()
    manifest["pages"][0]["widgets"][1]["bind"]["params"] = {"region": "$unknown"}
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("unknown variable $unknown" in m for m in err.value.messages)


def test_filter_widget_requires_known_variable() -> None:
    manifest = _manifest()
    manifest["pages"][0]["widgets"][0]["variable"] = "nope"
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("unknown variable 'nope'" in m for m in err.value.messages)


def test_data_widget_requires_bind() -> None:
    manifest = _manifest()
    del manifest["pages"][1]["widgets"][0]["bind"]
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("requires 'bind'" in m for m in err.value.messages)


def test_text_widget_may_not_bind() -> None:
    manifest = _manifest()
    manifest["pages"][1]["widgets"].append(
        {
            "id": "note",
            "type": "text",
            "gridPos": {"x": 0, "y": 11, "w": 24, "h": 2},
            "content": "hello",
            "bind": {"derivation": "revenue_rows"},
        }
    )
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("may not have 'bind'" in m for m in err.value.messages)


def test_drill_through_to_missing_page_rejected() -> None:
    manifest = _manifest()
    through = manifest["pages"][0]["widgets"][1]["interactions"]["drillThrough"]
    through["target"] = "page:ghost"
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("target page 'ghost' not found" in m for m in err.value.messages)


def test_variable_scope_must_be_global_or_a_page() -> None:
    manifest = _manifest()
    manifest["variables"][0]["scope"] = "nowhere"
    with pytest.raises(SpecValidationError) as err:
        validate_dashboard(manifest)
    assert any("scope 'nowhere'" in m for m in err.value.messages)


def test_schema_rejects_unknown_top_level_field() -> None:
    manifest = _manifest()
    manifest["unexpected"] = True
    assert not is_valid_dashboard(manifest)


def test_missing_pages_rejected() -> None:
    manifest = _manifest()
    manifest["pages"] = []
    assert not is_valid_dashboard(manifest)
