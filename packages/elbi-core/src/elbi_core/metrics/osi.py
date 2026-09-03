"""Import and export the Open Semantic Interchange (OSI) format.

`OSI <https://open-semantic-interchange.org/>`_ is the vendor-neutral, Apache-2.0
standard for exchanging semantic models across analytics, BI, and AI tools. This adapter
maps between a :class:`~elbi.metrics.spec.MetricSet` and an OSI semantic model,
so elbi metrics can be shared with, or seeded from, any OSI-speaking tool.

Export emits standard OSI (datasets with dimension fields, metrics as aggregation
expressions, and ``ai_context`` for agent grounding) validated against the bundled OSI
JSON Schema, so what we emit is conformant OSI, not merely OSI-shaped. It also carries
the exact elbi definition in a ``COMMON`` ``custom_extensions`` entry (OSI's
``vendor_name`` is a fixed enum, so a non-listed producer namespaces itself inside the
entry's ``data``), giving a lossless round-trip while staying readable to other tools.
Import restores that extension when present; otherwise it maps a foreign model
best-effort, parsing simple ``AGG(column)`` metric expressions.

OSI is pre-1.0 (this targets the released 0.1.1 schema): the core metric object has no
type/format/filter fields, so those richer semantics ride in the extension rather than
the core, per the spec's own guidance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import jsonschema

from ..errors import MetricError
from .spec import Measure, Metric, MetricSet, TimeDimension


@dataclass(frozen=True)
class _Dataset:
    """An OSI dataset mapped to a source derivation, with its dimension fields."""

    source: str
    dimensions: tuple[str, ...]
    time_field: str | None


#: The released OSI spec version this adapter targets and validates against.
OSI_VERSION = "0.1.1"

#: OSI's ``vendor_name`` is a fixed enum; a non-listed producer uses ``COMMON`` and
#: namespaces its payload with ``_PRODUCER`` inside the extension's ``data``.
_VENDOR = "COMMON"

#: Marks our payload inside a ``COMMON`` extension, so import restores only our own.
_PRODUCER = "elbi"

#: SQL dialect label used for emitted expressions.
_DIALECT = "ANSI_SQL"

#: elbi aggregation to its SQL function (count/count_distinct handled apart).
_AGG_SQL = {
    "sum": "SUM",
    "average": "AVG",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
}

#: SQL function name back to an elbi aggregation, for importing foreign models.
_SQL_AGG = {
    "SUM": "sum",
    "AVG": "average",
    "MIN": "min",
    "MAX": "max",
    "MEDIAN": "median",
}

#: A simple ``AGG(col)``, ``COUNT(*)``, or ``COUNT(DISTINCT col)`` expression.
_AGG_RE = re.compile(
    r"^\s*(?P<fn>[A-Za-z_]+)\s*\(\s*(?P<distinct>DISTINCT\s+)?(?P<col>\*|\"[^\"]+\"|[A-Za-z_][\w]*)\s*\)\s*$",
    re.IGNORECASE,
)


def to_osi(metric_set: MetricSet, *, model_name: str = "elbi") -> dict[str, Any]:
    """Export a metric set as an OSI semantic-model document.

    Each source derivation becomes an OSI dataset whose fields are the metrics'
    dimensions; each metric becomes an OSI metric with an aggregation expression, plus a
    vendor extension carrying its exact definition for a lossless round-trip.
    """
    datasets: dict[str, dict[str, dict[str, bool]]] = {}
    for metric in metric_set.metrics:
        if metric.type != "simple" or metric.source is None:
            continue
        fields = datasets.setdefault(metric.source, {})
        for dimension in metric.dimensions:
            fields.setdefault(dimension, {"is_time": False})
        if metric.time_dimension is not None:
            fields[metric.time_dimension.column] = {"is_time": True}

    model: dict[str, Any] = {
        "name": model_name,
        "datasets": [
            _osi_dataset(source, fields) for source, fields in datasets.items()
        ],
        "metrics": [_osi_metric(metric, metric_set) for metric in metric_set.metrics],
    }
    document = {"version": OSI_VERSION, "semantic_model": [model]}
    validate_osi(document)  # emit conformant OSI, not merely OSI-shaped
    return document


def validate_osi(document: dict[str, Any]) -> None:
    """Validate ``document`` against the bundled OSI JSON Schema.

    Raises:
        MetricError: if it does not conform (e.g. an empty model, OSI requires at least
            one dataset, or an unknown ``vendor_name``), with the offending path.
    """
    try:
        jsonschema.validate(document, _osi_schema())
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise MetricError(
            f"OSI document is not schema-conformant at {location}: {exc.message}"
        ) from exc


def _osi_schema() -> dict[str, Any]:
    """The bundled OSI JSON Schema (the released 0.1.1 core spec)."""
    resource = files("elbi_core") / "spec" / "osi.schema.json"
    schema: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return schema


def from_osi(
    document: dict[str, Any], *, source_for: dict[str, str] | None = None
) -> MetricSet:
    """Import an OSI semantic-model document as a metric set.

    When a metric carries the elbi vendor extension (an OSI produced by
    :func:`to_osi`), its exact definition is restored. Otherwise the metric is mapped
    best-effort: its ``AGG(column)`` expression becomes a measure, the model's dataset
    fields become dimensions, and ``source_for`` (or the dataset name) names the source
    derivation. A metric may carry a ``dataset`` tag naming the dataset it belongs to;
    OSI has no such field, so the tag is an elbi convenience and is excluded
    from the conformance check rather than rejected by it.

    Raises:
        MetricError: if ``document`` is not schema-conformant OSI, or if a foreign
            metric's expression is not a simple aggregation and no vendor extension is
            present to fall back on.
    """
    validate_osi(_without_dataset_tags(document))
    models = document.get("semantic_model") or []
    metrics: list[Metric] = []
    for model in models:
        datasets = _dataset_index(model, source_for)
        for entry in model.get("metrics", []):
            restored = _from_extension(entry)
            if restored is not None:
                metrics.append(restored)
                continue
            metrics.append(_import_foreign(entry, datasets))
    metric_set = MetricSet(metrics=tuple(metrics))
    metric_set.validate()
    return metric_set


def _without_dataset_tags(document: dict[str, Any]) -> dict[str, Any]:
    """Return ``document`` with the non-standard metric ``dataset`` tag removed.

    OSI closes the metric object to additional properties, so the tag has to be
    excluded before validating. Only the containers on the path to the metrics are
    rebuilt, leaving the caller's document untouched and everything else shared.

    A document malformed enough that the metrics cannot be walked is returned as it
    came, so :func:`validate_osi` is the one place that judges and phrases the fault.
    """
    try:
        models = [
            {
                **model,
                "metrics": [
                    {key: value for key, value in metric.items() if key != "dataset"}
                    for metric in model["metrics"]
                ],
            }
            if model.get("metrics")
            else model
            for model in document["semantic_model"]
        ]
    except (AttributeError, TypeError, KeyError):
        return document
    return {**document, "semantic_model": models}


def _osi_dataset(source: str, fields: dict[str, dict[str, bool]]) -> dict[str, Any]:
    return {
        "name": source,
        "source": source,
        "ai_context": f"Certified derivation {source!r} (the verified source).",
        "fields": [
            {
                "name": name,
                "expression": {"dialects": [{"dialect": _DIALECT, "expression": name}]},
                "dimension": {"is_time": meta["is_time"]},
            }
            for name, meta in fields.items()
        ],
    }


def _osi_metric(metric: Metric, metric_set: MetricSet) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": metric.name,
        "expression": {
            "dialects": [
                {"dialect": _DIALECT, "expression": _metric_sql(metric, metric_set)}
            ]
        },
        # The exact definition, namespaced inside a COMMON extension, for a lossless
        # round-trip (OSI's core carries no metric type/format/filter fields).
        "custom_extensions": [
            {
                "vendor_name": _VENDOR,
                "data": json.dumps(
                    {"producer": _PRODUCER, "metric": metric.to_manifest()}
                ),
            }
        ],
    }
    if metric.description is not None:
        out["description"] = metric.description
    ai_context = _ai_context(metric)
    if ai_context is not None:
        out["ai_context"] = ai_context
    return out


def _ai_context(metric: Metric) -> dict[str, Any] | None:
    """OSI ``ai_context``: instructions from the description, label as a synonym.

    The agent-grounding hints that are OSI's distinctive feature.
    """
    context: dict[str, Any] = {}
    if metric.description is not None:
        context["instructions"] = metric.description
    if metric.label is not None and metric.label != metric.name:
        context["synonyms"] = [metric.label]
    return context or None


def _metric_sql(metric: Metric, metric_set: MetricSet) -> str:
    if metric.type == "ratio":
        numerator = metric_set.metric(metric.numerator or "")
        denominator = metric_set.metric(metric.denominator or "")
        top = (
            _measure_sql(numerator.measure) if numerator and numerator.measure else "1"
        )
        bottom = (
            _measure_sql(denominator.measure)
            if denominator and denominator.measure
            else "1"
        )
        return f"({top}) / NULLIF({bottom}, 0)"
    if metric.type == "derived":
        # A metric-name arithmetic expression; the vendor extension carries the exact
        # definition, so this stays a readable (if non-executable) OSI representation.
        return metric.expr or "1"
    if metric.type == "cumulative":
        return metric.input_metric or "1"
    return _measure_sql(metric.measure) if metric.measure is not None else "1"


def _measure_sql(measure: Measure) -> str:
    if measure.agg == "count":
        return "COUNT(*)"
    if measure.agg == "count_distinct":
        return f'COUNT(DISTINCT "{measure.column}")'
    return f'{_AGG_SQL[measure.agg]}("{measure.column}")'


def _dataset_index(
    model: dict[str, Any], source_for: dict[str, str] | None
) -> dict[str, _Dataset]:
    """Map each OSI dataset name to its source derivation and dimension fields."""
    index: dict[str, _Dataset] = {}
    for dataset in model.get("datasets", []):
        name = str(dataset["name"])
        fields = dataset.get("fields", [])
        index[name] = _Dataset(
            source=(source_for or {}).get(name, name),
            dimensions=tuple(
                str(f["name"])
                for f in fields
                if not f.get("dimension", {}).get("is_time", False)
            ),
            time_field=next(
                (
                    str(f["name"])
                    for f in fields
                    if f.get("dimension", {}).get("is_time", False)
                ),
                None,
            ),
        )
    return index


def _from_extension(entry: dict[str, Any]) -> Metric | None:
    """Restore our exact metric from its ``COMMON`` extension, if one is present."""
    for extension in entry.get("custom_extensions", []):
        if extension.get("vendor_name") != _VENDOR:
            continue
        try:
            data = json.loads(extension["data"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        if isinstance(data, dict) and data.get("producer") == _PRODUCER:
            return Metric.from_manifest(data["metric"])
    return None


def _import_foreign(entry: dict[str, Any], datasets: dict[str, _Dataset]) -> Metric:
    name = str(entry["name"])
    agg, column = _parse_aggregation(name, _first_expression(entry))
    dataset = _dataset_for(name, entry, datasets)
    return Metric(
        name=name,
        type="simple",
        source=dataset.source,
        measure=Measure(agg=agg, column=column),
        dimensions=dataset.dimensions,
        time_dimension=(
            TimeDimension(column=dataset.time_field) if dataset.time_field else None
        ),
        description=_opt_str(entry.get("description")),
    )


def _dataset_for(
    name: str, entry: dict[str, Any], datasets: dict[str, _Dataset]
) -> _Dataset:
    """The dataset a foreign metric belongs to: its ``dataset`` tag, or the sole one."""
    tagged = entry.get("dataset")
    if tagged is not None:
        if str(tagged) not in datasets:
            raise MetricError(
                f"OSI metric {name!r} names dataset {tagged!r}, which the model omits"
            )
        return datasets[str(tagged)]
    if len(datasets) == 1:
        return next(iter(datasets.values()))
    raise MetricError(
        f"cannot import OSI metric {name!r}: the model has several datasets and the "
        "metric does not name one, so its source is ambiguous; tag the metric with a "
        "'dataset', pass source_for, or import a model with the vendor extension"
    )


def _first_expression(entry: dict[str, Any]) -> str:
    dialects = entry.get("expression", {}).get("dialects", [])
    if not dialects:
        raise MetricError(f"OSI metric {entry.get('name')!r} has no expression")
    return str(dialects[0].get("expression", ""))


def _parse_aggregation(name: str, expression: str) -> tuple[str, str | None]:
    match = _AGG_RE.match(expression)
    if match is None:
        raise MetricError(
            f"cannot import OSI metric {name!r}: expression {expression!r} is not a "
            "simple aggregation (AGG(column)); only simple metrics import without a "
            "vendor extension"
        )
    fn = match.group("fn").upper()
    column = match.group("col").strip('"')
    if fn == "COUNT":
        if match.group("distinct"):
            return "count_distinct", column
        return "count", None if column == "*" else column
    if fn not in _SQL_AGG:
        raise MetricError(
            f"cannot import OSI metric {name!r}: unknown aggregation {fn!r}"
        )
    return _SQL_AGG[fn], column


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
