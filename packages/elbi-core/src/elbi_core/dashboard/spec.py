"""The DashboardSpec: the declarative manifest of a dashboard.

A dashboard is a presentation surface, not a source of data. Each of its widgets
*binds* to a derivation by name and supplies parameters; the derivation is where the
computation and its verification live. Wiring is explicit: a widget consumes a
dashboard variable only by referencing it as ``$name`` in its bind params, so a spec
is reviewable the way code is, and an agent can author one against derivations that
are already certified.

On construction from a manifest the spec is validated against the bundled Dashboard
Spec schema, and then against a handful of cross-referential invariants the schema
cannot express (unique ids, variables and pages that references resolve to, and the
per-widget-type shape). An ill-formed dashboard fails loudly here rather than at
render time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

from ..errors import SpecValidationError

#: The Dashboard Spec version this SDK implements (MAJOR.MINOR).
DASHBOARD_SPEC_VERSION = "1.0"

#: The value types a variable selection produces.
VARIABLE_TYPES = ("string", "integer", "number", "boolean", "date")

#: The input controls a variable may render.
CONTROLS = ("dropdown", "multiselect", "search", "date", "daterange", "range", "toggle")

#: Every widget kind.
WIDGET_TYPES = ("metric", "chart", "map", "table", "text", "filter")

#: Widget kinds that bind to a derivation for their data.
DATA_WIDGET_TYPES = ("metric", "chart", "map", "table")


@dataclass(frozen=True)
class GridPos:
    """A widget's placement on the page grid.

    The page is divided into ``columns`` columns (24 by default); ``x``/``w`` are in
    those column units and ``y``/``h`` in grid rows, matching the layout convention
    common to code-defined dashboards.
    """

    x: int
    y: int
    w: int
    h: int

    def to_manifest(self) -> dict[str, int]:
        """Serialize to a spec ``gridPos`` object."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> GridPos:
        """Parse a spec ``gridPos`` object."""
        return cls(
            x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"])
        )


@dataclass(frozen=True)
class Bind:
    """A widget's data binding: a certified derivation, or a semantic-layer metric.

    A derivation binding names a ``derivation`` and the ``params`` passed to it (a param
    value is a literal, or ``"$name"`` referencing a dashboard variable). A metric
    binding names a ``metric`` and how to slice it (``group_by`` dimensions, a time
    ``grain``, and ``filters``); the metric aggregates a certified derivation, so a
    metric tile shows a verified number too. Exactly one of ``derivation`` or ``metric``
    is set.
    """

    derivation: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metric: str | None = None
    group_by: tuple[str, ...] = ()
    grain: str | None = None
    filters: tuple[dict[str, Any], ...] = ()

    @property
    def is_metric(self) -> bool:
        """Whether this binding resolves a metric rather than a derivation."""
        return self.metric is not None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``bind`` object, omitting empty fields."""
        if self.metric is not None:
            out: dict[str, Any] = {"metric": self.metric}
            if self.group_by:
                out["groupBy"] = list(self.group_by)
            if self.grain is not None:
                out["grain"] = self.grain
            if self.filters:
                out["filters"] = [dict(f) for f in self.filters]
            return out
        out = {"derivation": self.derivation}
        if self.params:
            out["params"] = dict(self.params)
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Bind:
        """Parse a spec ``bind`` object (a derivation or a metric binding)."""
        if data.get("metric") is not None:
            return cls(
                metric=str(data["metric"]),
                group_by=tuple(str(g) for g in data.get("groupBy", ())),
                grain=str(data["grain"]) if data.get("grain") is not None else None,
                filters=tuple(dict(f) for f in data.get("filters", ())),
            )
        return cls(
            derivation=str(data["derivation"]),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class Options:
    """Where a filter control's selectable options come from.

    Either a static ``values`` list (scalars, or ``{"value", "label"}`` records), or
    the distinct values of ``column`` in the rows of a certified ``derivation``.
    """

    values: tuple[Any, ...] | None = None
    derivation: str | None = None
    column: str | None = None
    label_column: str | None = None

    @property
    def is_dynamic(self) -> bool:
        """Whether options come from a derivation rather than a static list."""
        return self.derivation is not None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``options`` object."""
        if self.derivation is not None:
            out: dict[str, Any] = {"derivation": self.derivation, "column": self.column}
            if self.label_column is not None:
                out["labelColumn"] = self.label_column
            return out
        return {"values": list(self.values or ())}

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Options:
        """Parse a spec ``options`` object."""
        if "derivation" in data:
            return cls(
                derivation=str(data["derivation"]),
                column=str(data["column"]),
                label_column=_opt_str(data.get("labelColumn")),
            )
        return cls(values=tuple(data.get("values", ())))


@dataclass(frozen=True)
class Variable:
    """A dashboard-scoped value a viewer supplies through a filter control.

    Widgets consume it by referencing ``$name`` in their bind params. ``applies_to``
    and ``immune`` are advisory to the renderer: they govern which widgets a
    cross-filter selection propagates to, not whether an explicit ``$name`` reference
    resolves (it always does).
    """

    name: str
    type: str
    control: str = "dropdown"
    label: str | None = None
    default: Any = None
    options: Options | None = None
    scope: str = "global"
    applies_to: str | tuple[str, ...] = "*"
    immune: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``variable`` object, omitting defaulted fields."""
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.control != "dropdown":
            out["control"] = self.control
        if self.label is not None:
            out["label"] = self.label
        if self.default is not None:
            out["default"] = self.default
        if self.options is not None:
            out["options"] = self.options.to_manifest()
        if self.scope != "global":
            out["scope"] = self.scope
        if self.applies_to != "*":
            out["appliesTo"] = list(self.applies_to)
        if self.immune:
            out["immune"] = list(self.immune)
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Variable:
        """Parse a spec ``variable`` object."""
        applies = data.get("appliesTo", "*")
        options = data.get("options")
        return cls(
            name=str(data["name"]),
            type=str(data["type"]),
            control=str(data.get("control", "dropdown")),
            label=_opt_str(data.get("label")),
            default=data.get("default"),
            options=Options.from_manifest(options) if options is not None else None,
            scope=str(data.get("scope", "global")),
            applies_to="*" if applies == "*" else tuple(str(w) for w in applies),
            immune=tuple(str(w) for w in data.get("immune", ())),
        )


@dataclass(frozen=True)
class CrossFilter:
    """On selecting a mark, set variables from fields of the selected datum."""

    emit: dict[str, str]


@dataclass(frozen=True)
class DrillThrough:
    """On selection, navigate to a detail target carrying variable state."""

    target: str
    carry: tuple[str, ...] = ()


@dataclass(frozen=True)
class DrillDown:
    """Step through a hierarchy of grouping columns by rebinding a param."""

    hierarchy: tuple[str, ...]
    param: str


@dataclass(frozen=True)
class Interactions:
    """The interactive behaviors attached to a widget."""

    cross_filter: CrossFilter | None = None
    drill_through: DrillThrough | None = None
    drill_down: DrillDown | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``interactions`` object, omitting unset behaviors."""
        out: dict[str, Any] = {}
        if self.cross_filter is not None:
            out["crossFilter"] = {"emit": dict(self.cross_filter.emit)}
        if self.drill_through is not None:
            through: dict[str, Any] = {"target": self.drill_through.target}
            if self.drill_through.carry:
                through["carry"] = list(self.drill_through.carry)
            out["drillThrough"] = through
        if self.drill_down is not None:
            out["drillDown"] = {
                "hierarchy": list(self.drill_down.hierarchy),
                "param": self.drill_down.param,
            }
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Interactions:
        """Parse a spec ``interactions`` object."""
        cross = data.get("crossFilter")
        through = data.get("drillThrough")
        down = data.get("drillDown")
        return cls(
            cross_filter=CrossFilter(emit=dict(cross["emit"])) if cross else None,
            drill_through=(
                DrillThrough(
                    target=str(through["target"]),
                    carry=tuple(str(v) for v in through.get("carry", ())),
                )
                if through
                else None
            ),
            drill_down=(
                DrillDown(
                    hierarchy=tuple(str(c) for c in down["hierarchy"]),
                    param=str(down["param"]),
                )
                if down
                else None
            ),
        )


@dataclass(frozen=True)
class Widget:
    """A single tile on a page.

    ``metric``/``chart``/``map``/``table`` widgets carry a :class:`Bind`; a ``text``
    widget carries markdown ``content``, or a :class:`Bind` naming a derivation that
    returns markdown; a ``filter`` widget names the ``variable`` whose control it
    renders.
    """

    id: str
    type: str
    grid_pos: GridPos
    title: str | None = None
    bind: Bind | None = None
    viz: dict[str, Any] | None = None
    content: str | None = None
    variable: str | None = None
    interactions: Interactions | None = None

    @property
    def is_data_bound(self) -> bool:
        """Whether this widget draws its data from a derivation.

        A ``text`` tile usually carries its own prose, but naming a derivation makes it
        the one governed escape hatch from the fixed widget types: a bespoke visual
        belongs in a derivation returning markdown, which has to be run to be shown.
        """
        return self.type in DATA_WIDGET_TYPES or (
            self.type == "text" and self.bind is not None
        )

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``widget`` object, omitting unset fields."""
        out: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "gridPos": self.grid_pos.to_manifest(),
        }
        if self.title is not None:
            out["title"] = self.title
        if self.bind is not None:
            out["bind"] = self.bind.to_manifest()
        if self.viz is not None:
            out["viz"] = self.viz
        if self.content is not None:
            out["content"] = self.content
        if self.variable is not None:
            out["variable"] = self.variable
        if self.interactions is not None:
            interactions = self.interactions.to_manifest()
            if interactions:
                out["interactions"] = interactions
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Widget:
        """Parse a spec ``widget`` object."""
        bind = data.get("bind")
        interactions = data.get("interactions")
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            grid_pos=GridPos.from_manifest(data["gridPos"]),
            title=_opt_str(data.get("title")),
            bind=Bind.from_manifest(bind) if bind is not None else None,
            viz=data.get("viz"),
            content=_opt_str(data.get("content")),
            variable=_opt_str(data.get("variable")),
            interactions=(
                Interactions.from_manifest(interactions)
                if interactions is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Page:
    """A grid of widgets."""

    name: str
    widgets: tuple[Widget, ...]
    title: str | None = None
    columns: int = 24

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``page`` object."""
        out: dict[str, Any] = {
            "name": self.name,
            "widgets": [w.to_manifest() for w in self.widgets],
        }
        if self.title is not None:
            out["title"] = self.title
        if self.columns != 24:
            out["columns"] = self.columns
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Page:
        """Parse a spec ``page`` object."""
        return cls(
            name=str(data["name"]),
            widgets=tuple(Widget.from_manifest(w) for w in data.get("widgets", ())),
            title=_opt_str(data.get("title")),
            columns=int(data.get("columns", 24)),
        )


@dataclass(frozen=True)
class DashboardSpec:
    """A declarative dashboard over certified derivations.

    Construct from a validated manifest with :meth:`from_manifest`, or build
    programmatically and call :meth:`validate`. Both check the JSON Schema and the
    cross-referential invariants it cannot express.
    """

    name: str
    pages: tuple[Page, ...]
    title: str | None = None
    description: str | None = None
    theme: str = "auto"
    variables: tuple[Variable, ...] = ()
    refresh_interval: str = "off"

    def variable(self, name: str) -> Variable | None:
        """The variable named ``name``, or ``None``."""
        return next((v for v in self.variables if v.name == name), None)

    def page(self, name: str) -> Page | None:
        """The page named ``name``, or ``None``."""
        return next((p for p in self.pages if p.name == name), None)

    def derivation_names(self) -> tuple[str, ...]:
        """Every distinct derivation this dashboard binds, in first-seen order.

        Includes widget bindings and the derivations that back dynamic filter options.
        The set a publish gate must confirm are certified.
        """
        seen: dict[str, None] = {}
        for variable in self.variables:
            if variable.options is not None and variable.options.derivation:
                seen.setdefault(variable.options.derivation, None)
        for page in self.pages:
            for widget in page.widgets:
                if widget.bind is not None and widget.bind.derivation is not None:
                    seen.setdefault(widget.bind.derivation, None)
        return tuple(seen)

    def metric_names(self) -> tuple[str, ...]:
        """Every distinct metric this dashboard binds, in first-seen order."""
        seen: dict[str, None] = {}
        for page in self.pages:
            for widget in page.widgets:
                if widget.bind is not None and widget.bind.metric is not None:
                    seen.setdefault(widget.bind.metric, None)
        return tuple(seen)

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a full, spec-conformant Dashboard manifest."""
        out: dict[str, Any] = {
            "specVersion": DASHBOARD_SPEC_VERSION,
            "kind": "Dashboard",
            "name": self.name,
            "pages": [p.to_manifest() for p in self.pages],
        }
        if self.title is not None:
            out["title"] = self.title
        if self.description is not None:
            out["description"] = self.description
        if self.theme != "auto":
            out["theme"] = self.theme
        if self.variables:
            out["variables"] = [v.to_manifest() for v in self.variables]
        if self.refresh_interval != "off":
            out["refresh"] = {"interval": self.refresh_interval}
        return out

    def validate(self) -> None:
        """Validate this dashboard against the spec and its structural invariants.

        Raises:
            SpecValidationError: if the dashboard is not conformant.
        """
        validate_dashboard(self.to_manifest())

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> DashboardSpec:
        """Parse and validate a Dashboard manifest.

        Raises:
            SpecValidationError: if the manifest does not conform to the spec.
        """
        validate_dashboard(data)
        refresh = data.get("refresh", {})
        return cls(
            name=str(data["name"]),
            pages=tuple(Page.from_manifest(p) for p in data["pages"]),
            title=_opt_str(data.get("title")),
            description=_opt_str(data.get("description")),
            theme=str(data.get("theme", "auto")),
            variables=tuple(
                Variable.from_manifest(v) for v in data.get("variables", ())
            ),
            refresh_interval=str(refresh.get("interval", "off")),
        )


@lru_cache(maxsize=1)
def load_dashboard_schema() -> dict[str, Any]:
    """Return the bundled Dashboard JSON Schema as a dict."""
    resource = files("elbi_core") / "spec" / "dashboard.schema.json"
    schema: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return schema


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = load_dashboard_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_dashboard(manifest: dict[str, Any]) -> None:
    """Validate a Dashboard manifest against the spec.

    Structural conformance is checked against the JSON Schema; the cross-referential
    invariants the schema cannot express (unique names, resolvable references, and the
    per-widget-type shape) are checked here.

    Raises:
        SpecValidationError: if the manifest does not conform, carrying one
            human-readable message per violation.
    """
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(e.path))
    messages = [_format_error(error) for error in errors]
    # The reference checks below assume the shape the schema guarantees, so only run
    # them once the manifest is structurally valid.
    if not messages:
        messages.extend(_reference_errors(manifest))
    if messages:
        raise SpecValidationError(messages)


def is_valid_dashboard(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest conforms to the spec, without raising."""
    try:
        validate_dashboard(manifest)
    except SpecValidationError:
        return False
    return True


def _reference_errors(manifest: dict[str, Any]) -> list[str]:
    """Invariants the JSON Schema cannot express: uniqueness and resolvable refs."""
    messages: list[str] = []
    variables = {str(v["name"]): v for v in manifest.get("variables", [])}
    page_names = {str(p["name"]) for p in manifest["pages"]}

    messages.extend(
        _duplicate_errors("variable name", manifest.get("variables", []), "name")
    )
    messages.extend(_duplicate_errors("page name", manifest["pages"], "name"))

    widget_ids: set[str] = set()
    for page in manifest["pages"]:
        for widget in page.get("widgets", []):
            wid = str(widget["id"])
            if wid in widget_ids:
                messages.append(f"widget id {wid!r} is not unique across the dashboard")
            widget_ids.add(wid)

    for name, variable in variables.items():
        scope = str(variable.get("scope", "global"))
        if scope != "global" and scope not in page_names:
            messages.append(
                f"variables/{name}: scope {scope!r} is not 'global' or a page name"
            )

    for page in manifest["pages"]:
        for widget in page.get("widgets", []):
            messages.extend(_widget_errors(page, widget, variables, page_names))

    return messages


def _widget_errors(
    page: dict[str, Any],
    widget: dict[str, Any],
    variables: dict[str, Any],
    page_names: set[str],
) -> list[str]:
    wid = str(widget["id"])
    wtype = str(widget["type"])
    where = f"pages/{page['name']}/widgets/{wid}"
    messages: list[str] = []

    if wtype in DATA_WIDGET_TYPES and "bind" not in widget:
        messages.append(f"{where}: a {wtype!r} widget requires 'bind'")
    if wtype == "filter" and "bind" in widget:
        messages.append(f"{where}: a 'filter' widget may not have 'bind'")
    # A text tile's body is its own prose or a derivation returning markdown. Both
    # would leave which one renders decided somewhere other than the spec.
    if wtype == "text":
        has = ("content" in widget, "bind" in widget)
        if has == (False, False):
            messages.append(f"{where}: a 'text' widget requires 'content' or 'bind'")
        elif has == (True, True):
            messages.append(
                f"{where}: a 'text' widget takes 'content' or 'bind', not both"
            )
    if wtype == "filter":
        var = widget.get("variable")
        if var is None:
            messages.append(f"{where}: a 'filter' widget requires 'variable'")
        elif str(var) not in variables:
            messages.append(f"{where}: unknown variable {str(var)!r}")

    bind = widget.get("bind")
    if bind is not None:
        for value in bind.get("params", {}).values():
            ref = _variable_ref(value)
            if ref is not None and ref not in variables:
                messages.append(f"{where}: param references unknown variable ${ref}")

    interactions = widget.get("interactions", {})
    cross = interactions.get("crossFilter")
    if cross is not None:
        for var in cross["emit"]:
            if str(var) not in variables:
                messages.append(f"{where}: crossFilter emits unknown variable {var!r}")
    through = interactions.get("drillThrough")
    if through is not None:
        kind, _, target = str(through["target"]).partition(":")
        if kind == "page" and target not in page_names:
            messages.append(f"{where}: drillThrough target page {target!r} not found")
        for var in through.get("carry", []):
            if str(var) not in variables:
                messages.append(
                    f"{where}: drillThrough carries unknown variable {var!r}"
                )
    down = interactions.get("drillDown")
    if (
        down is not None
        and bind is not None
        and str(down["param"]) not in bind.get("params", {})
    ):
        messages.append(
            f"{where}: drillDown param {down['param']!r} is not a bind param"
        )

    return messages


def _duplicate_errors(label: str, items: list[Any], key: str) -> list[str]:
    seen: set[str] = set()
    messages: list[str] = []
    for item in items:
        value = str(item.get(key, ""))
        if value in seen:
            messages.append(f"duplicate {label}: {value!r}")
        seen.add(value)
    return messages


def _variable_ref(value: Any) -> str | None:
    """The variable name a param value references, or ``None`` for a literal.

    ``"$name"`` references variable ``name``; ``"$$"`` (and any ``"$$..."``) is a
    literal that happens to start with a dollar sign, not a reference.
    """
    if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        return value[1:]
    return None


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
