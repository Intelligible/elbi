"""Dashboards: declarative presentation surfaces over certified derivations.

A dashboard binds widgets to derivations; it holds no data logic of its own. This
package defines the spec (:class:`DashboardSpec` and its parts), validation against
the bundled Dashboard Spec schema, and the :mod:`resolve` layer that turns a
dashboard plus a viewer's variable selections into rendered widget data through a
:class:`~elbi.runner.Runner`.
"""

from __future__ import annotations

from .resolve import (
    MetricResolver,
    WidgetData,
    initial_variable_state,
    resolve_options,
    resolve_page,
    resolve_params,
    resolve_value,
    resolve_widget,
)
from .spec import (
    DASHBOARD_SPEC_VERSION,
    Bind,
    CrossFilter,
    DashboardSpec,
    DrillDown,
    DrillThrough,
    GridPos,
    Interactions,
    Options,
    Page,
    Variable,
    Widget,
    is_valid_dashboard,
    load_dashboard_schema,
    validate_dashboard,
)

__all__ = [
    "DASHBOARD_SPEC_VERSION",
    "Bind",
    "CrossFilter",
    "DashboardSpec",
    "DrillDown",
    "DrillThrough",
    "GridPos",
    "Interactions",
    "MetricResolver",
    "Options",
    "Page",
    "Variable",
    "Widget",
    "WidgetData",
    "initial_variable_state",
    "is_valid_dashboard",
    "load_dashboard_schema",
    "resolve_options",
    "resolve_page",
    "resolve_params",
    "resolve_value",
    "resolve_widget",
    "validate_dashboard",
]
