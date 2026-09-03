"""Semantic-layer metrics over certified derivations.

A metric is a named aggregation (a *measure*) over a certified derivation's rows, sliced
by declared *dimensions* and time *grains* -- defined once, so the same number is served
everywhere. The vocabulary follows the industry model (dbt MetricFlow, OSI); a metric
set imports from and exports to the OSI standard. Resolution compiles a metric to one
GROUP BY run through the DuckDB query engine, so its number rides on the verified rows.
"""

from .osi import OSI_VERSION, from_osi, to_osi
from .resolve import MetricResult, SourceLoader, compile_metric, resolve_metric
from .spec import (
    AGGREGATIONS,
    FILTER_OPS,
    FORMAT_KINDS,
    GRAINS,
    METRIC_SPEC_VERSION,
    Filter,
    Format,
    Measure,
    Metric,
    MetricSet,
    TimeDimension,
    is_valid_metric_set,
    load_metric_schema,
    validate_metric_set,
)

__all__ = [
    "AGGREGATIONS",
    "FILTER_OPS",
    "FORMAT_KINDS",
    "GRAINS",
    "METRIC_SPEC_VERSION",
    "OSI_VERSION",
    "Filter",
    "Format",
    "Measure",
    "Metric",
    "MetricResult",
    "MetricSet",
    "SourceLoader",
    "TimeDimension",
    "compile_metric",
    "from_osi",
    "is_valid_metric_set",
    "load_metric_schema",
    "resolve_metric",
    "to_osi",
    "validate_metric_set",
]
