"""Tests for resolving a dashboard's widgets against variable state.

The resolver turns ``$name`` references and viewer selections into the concrete params
each widget's derivation runs with, then runs them. The cases pin the substitution
rules (reference, default fallback, ``$$`` escape), that a broken tile is captured
rather than raised, and that dynamic filter options come from a derivation's distinct
column values.
"""

from __future__ import annotations

import pytest

from elbi_core import (
    Artifact,
    Context,
    Registry,
    Runner,
    derivation,
    param,
    serve,
)
from elbi_core.dashboard import (
    DashboardSpec,
    Variable,
    initial_variable_state,
    resolve_page,
    resolve_widget,
)
from elbi_core.dashboard.resolve import (
    resolve_options,
    resolve_params,
    resolve_value,
)
from elbi_core.registry import use_registry


@pytest.fixture
def runner(registry: Registry) -> Runner:
    with use_registry(registry):

        @derivation(
            params={"region": param.string(required=False, default="all")},
            serve=serve.table(),
        )
        def sales_by_region(ctx: Context) -> Artifact:
            region = ctx.param("region")
            rows = [
                {"region": "west", "revenue": 10},
                {"region": "east", "revenue": 20},
            ]
            if region != "all":
                rows = [r for r in rows if r["region"] == region]
            return Artifact.table(rows)

        @derivation(serve=serve.table())
        def regions(ctx: Context) -> Artifact:
            return Artifact.table(
                [
                    {"region": "west", "name": "West"},
                    {"region": "east", "name": "East"},
                    {"region": "west", "name": "West"},  # duplicate; deduped
                ]
            )

    return Runner(registry)


def _spec() -> DashboardSpec:
    return DashboardSpec.from_manifest(
        {
            "specVersion": "1.0",
            "kind": "Dashboard",
            "name": "sales",
            "variables": [
                {
                    "name": "region",
                    "type": "string",
                    "default": "all",
                    "options": {"derivation": "regions", "column": "region"},
                }
            ],
            "pages": [
                {
                    "name": "main",
                    "widgets": [
                        {
                            "id": "table",
                            "type": "table",
                            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                            "bind": {
                                "derivation": "sales_by_region",
                                "params": {"region": "$region"},
                            },
                        }
                    ],
                }
            ],
        }
    )


def test_resolve_value_reference_default_and_escape() -> None:
    spec = _spec()
    # An unset reference falls back to the variable's declared default.
    assert resolve_value("$region", {}, spec) == "all"
    # A supplied selection wins.
    assert resolve_value("$region", {"region": "west"}, spec) == "west"
    # "$$" escapes a literal leading dollar; other values pass through untouched.
    assert resolve_value("$$region", {}, spec) == "$region"
    assert resolve_value(42, {}, spec) == 42


def test_initial_state_is_declared_defaults() -> None:
    assert initial_variable_state(_spec()) == {"region": "all"}


def test_resolve_params_substitutes_the_selection() -> None:
    spec = _spec()
    widget = spec.page("main").widgets[0]
    assert resolve_params(spec, widget, {"region": "east"}) == {"region": "east"}


def test_resolve_widget_runs_the_bound_derivation(runner: Runner) -> None:
    spec = _spec()
    widget = spec.page("main").widgets[0]
    data = resolve_widget(runner, spec, widget, {"region": "west"})
    assert data.error is None
    assert data.kind == "table"
    assert data.value == [{"region": "west", "revenue": 10}]
    assert data.data_version  # a content version was captured for client caching


def test_resolve_page_uses_defaults_when_unset(runner: Runner) -> None:
    spec = _spec()
    results = resolve_page(runner, spec, "main", initial_variable_state(spec))
    assert len(results) == 1
    assert {r["region"] for r in results[0].value} == {"west", "east"}


def test_unknown_derivation_is_captured_not_raised(runner: Runner) -> None:
    broken = DashboardSpec.from_manifest(
        {
            "specVersion": "1.0",
            "kind": "Dashboard",
            "name": "broken",
            "pages": [
                {
                    "name": "main",
                    "widgets": [
                        {
                            "id": "t",
                            "type": "table",
                            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                            "bind": {"derivation": "does_not_exist"},
                        }
                    ],
                }
            ],
        }
    )
    widget = broken.page("main").widgets[0]
    data = resolve_widget(runner, broken, widget, {})
    assert data.value is None
    assert data.error is not None


def test_opaque_artifact_is_not_renderable(registry: Registry) -> None:
    with use_registry(registry):

        @derivation()
        def a_model(ctx: Context) -> Artifact:
            return Artifact.opaque(object())

    spec = DashboardSpec.from_manifest(
        {
            "specVersion": "1.0",
            "kind": "Dashboard",
            "name": "m",
            "pages": [
                {
                    "name": "main",
                    "widgets": [
                        {
                            "id": "t",
                            "type": "table",
                            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                            "bind": {"derivation": "a_model"},
                        }
                    ],
                }
            ],
        }
    )
    widget = spec.page("main").widgets[0]
    data = resolve_widget(Runner(registry), spec, widget, {})
    assert data.error is not None
    assert "opaque" in data.error


def test_dynamic_options_are_distinct_with_labels(runner: Runner) -> None:
    variable = Variable.from_manifest(
        {
            "name": "region",
            "type": "string",
            "options": {
                "derivation": "regions",
                "column": "region",
                "labelColumn": "name",
            },
        }
    )
    options = resolve_options(runner, variable)
    assert options == [
        {"value": "west", "label": "West"},
        {"value": "east", "label": "East"},
    ]


def test_static_options_pass_through(registry: Registry) -> None:
    variable = Variable.from_manifest(
        {
            "name": "tier",
            "type": "string",
            "options": {"values": ["gold", {"value": "silver", "label": "Silver"}]},
        }
    )
    # Static options never touch the runner, but the signature takes one.
    options = resolve_options(Runner(registry), variable)
    assert options == [
        {"value": "gold", "label": "gold"},
        {"value": "silver", "label": "Silver"},
    ]


def _metric_dashboard() -> DashboardSpec:
    """A dashboard whose one tile binds a semantic-layer metric, not a derivation."""
    return DashboardSpec.from_manifest(
        {
            "specVersion": "1.0",
            "kind": "Dashboard",
            "name": "revenue_board",
            "pages": [
                {
                    "name": "overview",
                    "widgets": [
                        {
                            "id": "rev",
                            "type": "table",
                            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 6},
                            "bind": {"metric": "revenue", "groupBy": ["region"]},
                        }
                    ],
                }
            ],
        }
    )


def test_metric_bound_widget_resolves_via_metric_resolver(runner: Runner) -> None:
    spec = _metric_dashboard()
    assert spec.metric_names() == ("revenue",)
    assert spec.derivation_names() == ()

    calls: list[tuple[str, list[str], str | None]] = []

    def metric_resolver(name, group_by, grain, filters):
        calls.append((name, list(group_by), grain))
        return [{"region": "west", "revenue": 40}, {"region": "east", "revenue": 20}]

    results = resolve_page(
        runner, spec, "overview", {}, metric_resolver=metric_resolver
    )
    assert calls == [("revenue", ["region"], None)]
    assert results[0].kind == "table"
    assert results[0].value == [
        {"region": "west", "revenue": 40},
        {"region": "east", "revenue": 20},
    ]


def test_metric_bound_widget_without_resolver_errors(runner: Runner) -> None:
    spec = _metric_dashboard()
    results = resolve_page(runner, spec, "overview", {})
    assert results[0].error is not None
    assert "not resolvable" in results[0].error
