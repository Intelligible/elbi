"""The metric spec: named semantic-layer metrics over certified derivations.

A metric pairs an aggregation (a *measure*) with the *dimensions* it can be sliced by,
so one definition yields the same number everywhere it is queried. The vocabulary
follows the industry model (dbt MetricFlow, OSI): measures with an aggregation function,
categorical and time dimensions, grains, and metric types (simple, ratio). Because a
metric's ``source`` is a pre-joined certified derivation, there are no multi-dataset
joins or relationships to express. The set is validated against the bundled Metric Spec
schema on construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

from ..errors import SpecValidationError

#: The Metric Spec version this SDK implements (MAJOR.MINOR).
METRIC_SPEC_VERSION = "1.0"

#: Aggregation functions a measure may use (MetricFlow-aligned names).
AGGREGATIONS = ("sum", "average", "count", "count_distinct", "min", "max", "median")

#: Time grains a time dimension may roll up to.
GRAINS = ("hour", "day", "week", "month", "quarter", "year")

#: Row-filter comparison operators.
FILTER_OPS = ("eq", "ne", "lt", "le", "gt", "ge", "in")


@dataclass(frozen=True)
class Measure:
    """An aggregation over a column of the source rows.

    ``column`` is omitted only for ``agg="count"`` (which counts rows).
    """

    agg: str
    column: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``measure`` object."""
        out: dict[str, Any] = {"agg": self.agg}
        if self.column is not None:
            out["column"] = self.column
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Measure:
        """Parse a spec ``measure`` object."""
        return cls(agg=str(data["agg"]), column=_opt_str(data.get("column")))


@dataclass(frozen=True)
class TimeDimension:
    """A time column and the grain to roll it up to."""

    column: str
    grain: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``timeDimension`` object."""
        out: dict[str, Any] = {"column": self.column}
        if self.grain is not None:
            out["grain"] = self.grain
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> TimeDimension:
        """Parse a spec ``timeDimension`` object."""
        return cls(column=str(data["column"]), grain=_opt_str(data.get("grain")))


@dataclass(frozen=True)
class Filter:
    """A row filter applied before aggregation: column, comparison, and value."""

    column: str
    op: str
    value: Any

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``filter`` object."""
        return {"column": self.column, "op": self.op, "value": self.value}

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Filter:
        """Parse a spec ``filter`` object."""
        return cls(column=str(data["column"]), op=str(data["op"]), value=data["value"])


#: Display-format kinds a metric value can carry (Cube's vocabulary).
FORMAT_KINDS = ("number", "currency", "percent", "duration")


@dataclass(frozen=True)
class Format:
    """How a metric's value is displayed, never how it is computed.

    Follows Cube's model (the de-facto standard, since OSI and dbt define no format
    field): ``kind`` selects the unit style; ``currency`` is an ISO 4217 code used only
    when ``kind`` is ``currency``; ``precision`` fixes the decimal places; ``notation``
    is ``standard`` / ``compact`` / ``scientific``; ``pattern`` is a d3-format string
    that, when set, overrides the rest (the Looker ``value_format`` escape hatch). The
    fields map one-to-one onto the frontend's ``Intl.NumberFormat``. A ``percent``
    metric stores the fraction (``0.12``), which the formatter renders as ``12%``.
    """

    kind: str = "number"
    currency: str | None = None
    precision: int | None = None
    notation: str | None = None
    pattern: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``format`` object (omitting unset fields)."""
        out: dict[str, Any] = {"kind": self.kind}
        if self.currency is not None:
            out["currency"] = self.currency
        if self.precision is not None:
            out["precision"] = self.precision
        if self.notation is not None:
            out["notation"] = self.notation
        if self.pattern is not None:
            out["pattern"] = self.pattern
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Format:
        """Parse a spec ``format`` object."""
        precision = data.get("precision")
        return cls(
            kind=str(data.get("kind", "number")),
            currency=_opt_str(data.get("currency")),
            precision=int(precision) if precision is not None else None,
            notation=_opt_str(data.get("notation")),
            pattern=_opt_str(data.get("pattern")),
        )


@dataclass(frozen=True)
class Metric:
    """A named metric over a certified derivation.

    A ``simple`` metric aggregates its ``measure`` over ``source``'s rows, sliceable by
    ``dimensions`` and ``time_dimension``, narrowed by ``filters``. A ``ratio`` metric
    divides its ``numerator`` metric by its ``denominator`` metric (both defined in the
    same set), grouped the same way.
    """

    name: str
    type: str
    source: str | None = None
    measure: Measure | None = None
    numerator: str | None = None
    denominator: str | None = None
    expr: str | None = None
    input_metrics: tuple[str, ...] = ()
    input_metric: str | None = None
    window: int | None = None
    period_agg: str | None = None
    dimensions: tuple[str, ...] = ()
    time_dimension: TimeDimension | None = None
    filters: tuple[Filter, ...] = ()
    label: str | None = None
    description: str | None = None
    format: Format | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``metric`` object."""
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.label is not None:
            out["label"] = self.label
        if self.description is not None:
            out["description"] = self.description
        if self.format is not None:
            out["format"] = self.format.to_manifest()
        if self.source is not None:
            out["source"] = self.source
        if self.measure is not None:
            out["measure"] = self.measure.to_manifest()
        if self.numerator is not None:
            out["numerator"] = self.numerator
        if self.denominator is not None:
            out["denominator"] = self.denominator
        if self.expr is not None:
            out["expr"] = self.expr
        if self.input_metrics:
            out["inputMetrics"] = list(self.input_metrics)
        if self.input_metric is not None:
            out["inputMetric"] = self.input_metric
        if self.window is not None:
            out["window"] = self.window
        if self.period_agg is not None:
            out["periodAgg"] = self.period_agg
        if self.dimensions:
            out["dimensions"] = list(self.dimensions)
        if self.time_dimension is not None:
            out["timeDimension"] = self.time_dimension.to_manifest()
        if self.filters:
            out["filters"] = [f.to_manifest() for f in self.filters]
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Metric:
        """Parse a spec ``metric`` object."""
        measure = data.get("measure")
        time_dimension = data.get("timeDimension")
        return cls(
            name=str(data["name"]),
            type=str(data["type"]),
            source=_opt_str(data.get("source")),
            measure=Measure.from_manifest(measure) if measure else None,
            numerator=_opt_str(data.get("numerator")),
            denominator=_opt_str(data.get("denominator")),
            expr=_opt_str(data.get("expr")),
            input_metrics=tuple(str(m) for m in data.get("inputMetrics", ())),
            input_metric=_opt_str(data.get("inputMetric")),
            window=int(data["window"]) if data.get("window") is not None else None,
            period_agg=_opt_str(data.get("periodAgg")),
            dimensions=tuple(str(d) for d in data.get("dimensions", ())),
            time_dimension=(
                TimeDimension.from_manifest(time_dimension) if time_dimension else None
            ),
            filters=tuple(Filter.from_manifest(f) for f in data.get("filters", ())),
            label=_opt_str(data.get("label")),
            description=_opt_str(data.get("description")),
            format=Format.from_manifest(data["format"]) if data.get("format") else None,
        )


@dataclass(frozen=True)
class MetricSet:
    """A registry of metrics.

    Construct from a validated manifest with :meth:`from_manifest`, or build
    programmatically and call :meth:`validate`.
    """

    metrics: tuple[Metric, ...] = field(default_factory=tuple)

    def metric(self, name: str) -> Metric | None:
        """The metric named ``name``, or ``None``."""
        return next((m for m in self.metrics if m.name == name), None)

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a full, spec-conformant MetricSet manifest."""
        out: dict[str, Any] = {"specVersion": METRIC_SPEC_VERSION, "kind": "MetricSet"}
        if self.metrics:
            out["metrics"] = [m.to_manifest() for m in self.metrics]
        return out

    def validate(self) -> None:
        """Validate this set against the spec.

        Raises:
            SpecValidationError: if the set is not conformant.
        """
        validate_metric_set(self.to_manifest())

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> MetricSet:
        """Parse and validate a MetricSet manifest.

        Raises:
            SpecValidationError: if the manifest does not conform to the spec.
        """
        validate_metric_set(data)
        return cls(
            metrics=tuple(Metric.from_manifest(m) for m in data.get("metrics", ()))
        )


@lru_cache(maxsize=1)
def load_metric_schema() -> dict[str, Any]:
    """Return the bundled Metric JSON Schema as a dict."""
    resource = files("elbi_core") / "spec" / "metric.schema.json"
    schema: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return schema


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = load_metric_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_metric_set(manifest: dict[str, Any]) -> None:
    """Validate a MetricSet manifest against the spec and its invariants.

    Checks JSON-Schema conformance, then the cross-references the schema cannot express:
    unique metric names, and every ratio metric's numerator and denominator resolving to
    a metric defined in the same set.

    Raises:
        SpecValidationError: with one message per violation.
    """
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(e.path))
    messages = [_format_error(error) for error in errors]
    if not messages:
        messages.extend(_reference_errors(manifest))
    if messages:
        raise SpecValidationError(messages)


def is_valid_metric_set(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest conforms, without raising."""
    try:
        validate_metric_set(manifest)
    except SpecValidationError:
        return False
    return True


def _reference_errors(manifest: dict[str, Any]) -> list[str]:
    metrics = manifest.get("metrics", [])
    messages = _duplicate_errors("metric name", metrics, "name")
    defined = {str(m["name"]) for m in metrics}
    for metric in metrics:
        for role, ref in _operand_refs(metric):
            if str(ref) not in defined:
                messages.append(
                    f"metrics/{metric['name']}: {role} {ref!r} is not a defined metric"
                )
    return messages


def _operand_refs(metric: dict[str, Any]) -> list[tuple[str, str]]:
    """Every other metric a metric references, as ``(role, name)`` pairs by type."""
    kind = metric.get("type")
    if kind == "ratio":
        return [
            (role, str(metric[role]))
            for role in ("numerator", "denominator")
            if metric.get(role) is not None
        ]
    if kind == "derived":
        return [("input metric", str(m)) for m in metric.get("inputMetrics", ())]
    if kind == "cumulative" and metric.get("inputMetric") is not None:
        return [("input metric", str(metric["inputMetric"]))]
    return []


def _duplicate_errors(label: str, items: list[Any], key: str) -> list[str]:
    seen: set[str] = set()
    messages: list[str] = []
    for item in items:
        value = str(item.get(key, ""))
        if value in seen:
            messages.append(f"duplicate {label}: {value!r}")
        seen.add(value)
    return messages


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
