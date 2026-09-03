"""Resolving a dashboard's widgets against the current variable state.

The resolver is the boundary between the declarative spec and a live :class:`Runner`.
Given a dashboard, a page, and the viewer's current variable selections, it computes
each data-bound widget's concrete parameters (substituting ``$name`` references with
variable values) and runs the bound derivation to produce the rows the widget draws.

It is deliberately pure of any web or storage concern: it takes a ``Runner`` and
returns data. The serving layer wraps it with authorization, auditing, attestation
lookup, and stale-while-revalidate caching; none of that belongs here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ElbiError
from ..runner import Runner
from .spec import DashboardSpec, Variable, Widget, _variable_ref

#: Resolve a metric binding to rows: (metric, group_by, grain, filters) -> list[dict].
#: The dashboard core has no metric engine; the serving layer injects this.
MetricResolver = Callable[
    [str, Sequence[str], str | None, Sequence[Mapping[str, Any]]], list[dict[str, Any]]
]


@dataclass(frozen=True)
class WidgetData:
    """The resolved data for one data-bound widget.

    Exactly one of ``value`` (populated) or ``error`` (a message) is meaningful. On
    success ``kind`` is the bound derivation's artifact kind (``table``, ``json``,
    ``markdown``, ``text``) and ``value`` its rendered value; ``data_version`` is the
    content version the widget's data was computed at, for client-side caching.
    """

    widget_id: str
    derivation: str
    params: dict[str, Any] = field(default_factory=dict)
    kind: str | None = None
    value: Any = None
    data_version: str | None = None
    error: str | None = None


def initial_variable_state(spec: DashboardSpec) -> dict[str, Any]:
    """The variable values a dashboard opens with: each variable's declared default."""
    return {v.name: v.default for v in spec.variables if v.default is not None}


def resolve_value(value: Any, state: dict[str, Any], spec: DashboardSpec) -> Any:
    """Resolve one bind-param value against the variable state.

    ``"$name"`` resolves to the current value of variable ``name``: the viewer's
    selection, or the variable's default when unset. ``"$$"``-prefixed strings are
    literals whose leading dollar sign is escaped. Every other value is a literal.
    """
    if isinstance(value, str) and value.startswith("$$"):
        return value[1:]
    ref = _variable_ref(value)
    if ref is None:
        return value
    if ref in state:
        return state[ref]
    variable = spec.variable(ref)
    return variable.default if variable is not None else None


def resolve_params(
    spec: DashboardSpec, widget: Widget, state: dict[str, Any]
) -> dict[str, Any]:
    """The concrete parameters for ``widget`` given the current variable ``state``."""
    if widget.bind is None:
        return {}
    return {
        key: resolve_value(value, state, spec)
        for key, value in widget.bind.params.items()
    }


def resolve_widget(
    runner: Runner,
    spec: DashboardSpec,
    widget: Widget,
    state: dict[str, Any],
    *,
    metric_resolver: MetricResolver | None = None,
) -> WidgetData:
    """Compute one data-bound widget's data through ``runner`` (or the metric resolver).

    A failure (unknown, uncertified, or a compute error) is captured as ``error`` rather
    than raised, so one broken tile never blanks the whole dashboard.
    """
    if widget.bind is None:
        return WidgetData(widget.id, "", error="widget has no data binding")
    if widget.bind.is_metric:
        return _resolve_metric_widget(widget, metric_resolver)
    name = widget.bind.derivation or ""
    params = resolve_params(spec, widget, state)
    try:
        artifact = runner.run(name, params)
    except ElbiError as exc:
        return WidgetData(widget.id, name, params=params, error=str(exc))
    if artifact.kind == "opaque":
        return WidgetData(
            widget.id,
            name,
            params=params,
            error=f"derivation {name!r} produces an opaque artifact, not renderable",
        )
    try:
        data_version = runner.data_version(name, params)
    except ElbiError:
        data_version = None
    return WidgetData(
        widget.id,
        name,
        params=params,
        kind=artifact.kind,
        value=artifact.value,
        data_version=data_version,
    )


def _resolve_metric_widget(
    widget: Widget, metric_resolver: MetricResolver | None
) -> WidgetData:
    """Resolve a metric-bound widget through the injected metric resolver."""
    bind = widget.bind
    if bind is None or bind.metric is None:  # the caller checks this; belt and braces
        return WidgetData(widget.id, "", error="widget has no metric binding")
    if metric_resolver is None:
        return WidgetData(
            widget.id, bind.metric, error="metric bindings are not resolvable here"
        )
    try:
        rows = metric_resolver(bind.metric, bind.group_by, bind.grain, bind.filters)
    except ElbiError as exc:
        return WidgetData(widget.id, bind.metric, error=str(exc))
    return WidgetData(widget.id, bind.metric, kind="table", value=rows)


def resolve_page(
    runner: Runner,
    spec: DashboardSpec,
    page_name: str,
    state: dict[str, Any],
    *,
    metric_resolver: MetricResolver | None = None,
) -> list[WidgetData]:
    """Resolve every data-bound widget on a page against the variable ``state``.

    Raises:
        KeyError: if ``page_name`` is not a page of the dashboard.
    """
    page = spec.page(page_name)
    if page is None:
        raise KeyError(page_name)
    return [
        resolve_widget(runner, spec, widget, state, metric_resolver=metric_resolver)
        for widget in page.widgets
        if widget.is_data_bound
    ]


def resolve_options(runner: Runner, variable: Variable) -> list[dict[str, Any]]:
    """The selectable ``{"value", "label"}`` options for a filter control.

    A static option list is returned as declared. A dynamic list runs the backing
    derivation and takes the distinct, non-null values of its column in first-seen
    order, with an optional label column.
    """
    options = variable.options
    if options is None:
        return []
    if options.derivation is None or options.column is None:
        return [_option_record(item) for item in (options.values or ())]

    artifact = runner.run(options.derivation)
    rows = artifact.value if artifact.kind == "table" else []
    seen: set[Any] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        if options.column not in row:
            continue
        value = row[options.column]
        if value is None or value in seen:
            continue
        seen.add(value)
        label = row.get(options.label_column) if options.label_column else value
        records.append({"value": value, "label": str(label)})
    return records


def _option_record(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return {"value": item["value"], "label": str(item.get("label", item["value"]))}
    return {"value": item, "label": str(item)}
