"""Resolve a metric to grouped, aggregated rows over its certified source.

A metric definition (an aggregation plus the dimensions it may be sliced by) is compiled
to one ``GROUP BY`` query and run through :func:`elbi.query_datasets` (DuckDB),
so the number is computed the same way every time and rides on the certified source's
verified rows. A simple metric aggregates its measure; a ratio metric divides its
numerator metric by its denominator, grouped identically.

Identifiers (columns, aggregation, grain) come from the validated definition and a fixed
vocabulary, and are checked against the source's actual columns; filter values are bound
as query parameters. So resolving a metric is not a SQL-injection surface.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from ..errors import MetricError
from ..query import query_datasets
from .spec import GRAINS, Filter, Measure, Metric, MetricSet

Row = dict[str, Any]

#: A callable that returns a source derivation's rows, by derivation name.
SourceLoader = Callable[[str], list[Row]]

#: Measure aggregation to the DuckDB function (count/count_distinct handled apart).
_AGG_FN = {
    "sum": "sum",
    "average": "avg",
    "min": "min",
    "max": "max",
    "median": "median",
}

#: Filter comparison operator to SQL (``in`` is expanded separately).
_OP_SQL = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


@dataclass(frozen=True)
class MetricResult:
    """A resolved metric: the output columns (group keys, then the metric) and rows."""

    columns: list[str]
    rows: list[Row]


def resolve_metric(
    metric_set: MetricSet,
    name: str,
    load_source: SourceLoader,
    *,
    group_by: Sequence[str] = (),
    grain: str | None = None,
    filters: Sequence[Filter] = (),
    max_rows: int = 10_000,
) -> MetricResult:
    """Resolve metric ``name`` in ``metric_set`` to grouped, aggregated rows.

    ``group_by`` slices the metric by the named dimensions (each must be one the metric
    declares); ``grain`` overrides the time dimension's grain; ``filters`` narrows rows
    beyond the metric's own filters. ``load_source`` returns a source derivation's rows.

    Raises:
        MetricError: if the metric or a referenced column is unknown, a requested
            dimension is not declared, or the query fails.
    """
    metric = metric_set.metric(name)
    if metric is None:
        raise MetricError(f"unknown metric {name!r}")
    if grain is not None and grain not in GRAINS:
        raise MetricError(
            f"unknown grain {grain!r}; expected one of {', '.join(GRAINS)}"
        )
    if metric.type == "ratio":
        return _resolve_ratio(
            metric_set, metric, load_source, group_by, grain, filters, max_rows
        )
    if metric.type == "derived":
        return _resolve_derived(
            metric_set, metric, load_source, group_by, grain, filters, max_rows
        )
    if metric.type == "cumulative":
        return _resolve_cumulative(
            metric_set, metric, load_source, group_by, grain, filters, max_rows
        )
    return _resolve_simple(metric, load_source, group_by, grain, filters, max_rows)


def _resolve_simple(
    metric: Metric,
    load_source: SourceLoader,
    group_by: Sequence[str],
    grain: str | None,
    extra_filters: Sequence[Filter],
    max_rows: int,
) -> MetricResult:
    if metric.source is None or metric.measure is None:  # guaranteed by the spec
        raise MetricError(f"metric {metric.name!r} is missing a source or measure")
    rows = load_source(metric.source)
    available = _columns(rows)
    out_columns, group_parts, select_parts = _simple_parts(
        metric, group_by, grain, available
    )
    sql, params = _assemble_sql(
        metric, group_parts, select_parts, (*metric.filters, *extra_filters), available
    )
    result = query_datasets({"src": rows}, sql, params=params, max_rows=max_rows)
    return MetricResult(columns=out_columns, rows=result.rows)


def _simple_parts(
    metric: Metric,
    group_by: Sequence[str],
    grain: str | None,
    available: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Build a simple metric's SELECT/GROUP BY fragments and output columns.

    Shared by resolution and compilation. Each requested dimension is checked against
    the metric's declared set and the source's columns; the time dimension truncates to
    the effective grain.
    """
    if metric.measure is None:  # pragma: no cover - guarded by the caller
        raise MetricError(f"metric {metric.name!r} is missing a measure")
    time_column = metric.time_dimension.column if metric.time_dimension else None
    effective_grain = grain or (
        metric.time_dimension.grain if metric.time_dimension else None
    )
    allowed = set(metric.dimensions) | ({time_column} if time_column else set())

    select_parts: list[str] = []
    group_parts: list[str] = []
    out_columns: list[str] = []
    for dimension in group_by:
        if dimension not in allowed:
            raise MetricError(
                f"cannot group metric {metric.name!r} by {dimension!r}; "
                f"declared dimensions: {', '.join(sorted(allowed)) or '(none)'}"
            )
        _require_column(metric, dimension, available)
        if dimension == time_column and effective_grain:
            # Cast to timestamp so a string/date time column truncates; grain is from
            # the fixed GRAINS vocabulary, so interpolating it is safe.
            expr = (
                f"date_trunc('{effective_grain}', CAST({_qi(dimension)} AS TIMESTAMP))"
            )
        else:
            expr = _qi(dimension)
        select_parts.append(f"{expr} AS {_qi(dimension)}")
        group_parts.append(expr)
        out_columns.append(dimension)

    if metric.measure.column is not None:
        _require_column(metric, metric.measure.column, available)
    select_parts.append(f"{_value_expr(metric.measure)} AS {_qi(metric.name)}")
    out_columns.append(metric.name)
    return out_columns, group_parts, select_parts


def compile_metric(
    metric_set: MetricSet,
    name: str,
    load_source: SourceLoader,
    *,
    group_by: Sequence[str] = (),
    grain: str | None = None,
    filters: Sequence[Filter] = (),
) -> str:
    """Return the SQL a simple metric query compiles to, without running it.

    The trust/debug affordance every semantic layer exposes (Cube's "Generated SQL",
    dbt's ``--explain``): the exact query the resolver would run for this request, with
    ``?`` placeholders for the bound filter values. Only simple metrics compile to a
    single query; ratio/derived/cumulative are composed from several and report that.

    Raises:
        MetricError: if the metric is unknown or a referenced dimension/column is not.
    """
    metric = metric_set.metric(name)
    if metric is None:
        raise MetricError(f"unknown metric {name!r}")
    if metric.type != "simple":
        return f"-- compiled SQL is shown for simple metrics only (not {metric.type})"
    if metric.source is None:  # pragma: no cover - guaranteed by the spec
        raise MetricError(f"metric {name!r} is missing a source")
    available = _columns(load_source(metric.source))
    _, group_parts, select_parts = _simple_parts(metric, group_by, grain, available)
    sql, _params = _assemble_sql(
        metric, group_parts, select_parts, (*metric.filters, *filters), available
    )
    return sql


def _assemble_sql(
    metric: Metric,
    group_parts: Sequence[str],
    select_parts: Sequence[str],
    filters: Sequence[Filter],
    available: set[str],
) -> tuple[str, list[Any]]:
    """Build the ``GROUP BY`` SQL for a simple metric and its bound parameters.

    The static keywords and the (quoted-identifier) fragments are separate, and every
    value rides in ``params``, so there is no dynamic SQL literal to interpolate
    unsafely. Shared by resolution (which runs it) and compilation (which returns it,
    the way Cube/dbt expose a metric's generated SQL as a trust/debug affordance).
    """
    where_sql, params = _where(metric, filters, available)
    clauses = ["SELECT", ", ".join(select_parts), "FROM src"]
    if where_sql:
        clauses += ["WHERE", where_sql]
    if group_parts:
        grouped = ", ".join(group_parts)
        clauses += ["GROUP BY", grouped, "ORDER BY", grouped]
    return " ".join(clauses), params


def _resolve_ratio(
    metric_set: MetricSet,
    metric: Metric,
    load_source: SourceLoader,
    group_by: Sequence[str],
    grain: str | None,
    filters: Sequence[Filter],
    max_rows: int,
) -> MetricResult:
    if metric.numerator is None or metric.denominator is None:  # guaranteed by the spec
        raise MetricError(f"ratio metric {metric.name!r} is missing an operand")
    numerator = resolve_metric(
        metric_set,
        metric.numerator,
        load_source,
        group_by=group_by,
        grain=grain,
        filters=filters,
        max_rows=max_rows,
    )
    denominator = resolve_metric(
        metric_set,
        metric.denominator,
        load_source,
        group_by=group_by,
        grain=grain,
        filters=filters,
        max_rows=max_rows,
    )
    keys = list(group_by)

    def key(row: Row) -> tuple[Any, ...]:
        return tuple(row[k] for k in keys)

    denominator_by_key = {key(row): row[metric.denominator] for row in denominator.rows}
    out_rows: list[Row] = []
    for row in numerator.rows:
        top = row[metric.numerator]
        bottom = denominator_by_key.get(key(row))
        ratio = top / bottom if bottom not in (None, 0) and top is not None else None
        out_rows.append({**{k: row[k] for k in keys}, metric.name: ratio})
    return MetricResult(columns=[*keys, metric.name], rows=out_rows)


def _resolve_derived(
    metric_set: MetricSet,
    metric: Metric,
    load_source: SourceLoader,
    group_by: Sequence[str],
    grain: str | None,
    filters: Sequence[Filter],
    max_rows: int,
) -> MetricResult:
    """Evaluate a derived metric's expression over its input metrics, per group."""
    tree = _parse_expr(metric)
    keys = list(group_by)

    def key(row: Row) -> tuple[Any, ...]:
        return tuple(row[k] for k in keys)

    per_input: dict[str, dict[tuple[Any, ...], Any]] = {}
    for name in metric.input_metrics:
        resolved = resolve_metric(
            metric_set,
            name,
            load_source,
            group_by=group_by,
            grain=grain,
            filters=filters,
            max_rows=max_rows,
        )
        per_input[name] = {key(row): row[name] for row in resolved.rows}

    all_keys = {k for values in per_input.values() for k in values}
    out_rows: list[Row] = []
    for k in sorted(all_keys, key=lambda t: tuple(str(v) for v in t)):
        values = {name: per_input[name].get(k) for name in metric.input_metrics}
        computed = (
            None if any(v is None for v in values.values()) else _eval(tree, values)
        )
        out_rows.append(
            {**{keys[i]: k[i] for i in range(len(keys))}, metric.name: computed}
        )
    return MetricResult(columns=[*keys, metric.name], rows=out_rows)


def _resolve_cumulative(
    metric_set: MetricSet,
    metric: Metric,
    load_source: SourceLoader,
    group_by: Sequence[str],
    grain: str | None,
    filters: Sequence[Filter],
    max_rows: int,
) -> MetricResult:
    """Aggregate a metric over a trailing time window (or cumulatively to date)."""
    if metric.input_metric is None:  # guaranteed by the spec
        raise MetricError(
            f"cumulative metric {metric.name!r} is missing an input metric"
        )
    input_metric = metric_set.metric(metric.input_metric)
    if input_metric is None or input_metric.time_dimension is None:
        raise MetricError(
            f"cumulative metric {metric.name!r} needs an input metric with a time "
            "dimension to accumulate over"
        )
    time_column = input_metric.time_dimension.column
    grouped = [time_column, *[g for g in group_by if g != time_column]]
    resolved = resolve_metric(
        metric_set,
        metric.input_metric,
        load_source,
        group_by=grouped,
        grain=grain,
        filters=filters,
        max_rows=max_rows,
    )
    partition_keys = [g for g in grouped if g != time_column]
    partitions: dict[tuple[Any, ...], list[Row]] = {}
    for row in resolved.rows:
        partitions.setdefault(tuple(row[k] for k in partition_keys), []).append(row)

    period_agg = metric.period_agg or "sum"
    out_rows: list[Row] = []
    for rows in partitions.values():
        ordered = sorted(rows, key=lambda r: str(r[time_column]))
        for position, row in enumerate(ordered):
            start = 0 if metric.window is None else max(0, position - metric.window + 1)
            window = ordered[start : position + 1]
            values = [
                r[metric.input_metric]
                for r in window
                if r[metric.input_metric] is not None
            ]
            out = {time_column: row[time_column]}
            out.update({k: row[k] for k in partition_keys})
            out[metric.name] = _period_value(period_agg, values)
            out_rows.append(out)
    return MetricResult(columns=[*grouped, metric.name], rows=out_rows)


#: Arithmetic operators a derived metric expression may use.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _parse_expr(metric: Metric) -> ast.Expression:
    if not metric.expr:  # guaranteed by the spec
        raise MetricError(f"derived metric {metric.name!r} is missing an expression")
    try:
        return ast.parse(metric.expr, mode="eval")
    except SyntaxError as exc:
        raise MetricError(f"metric {metric.name!r}: invalid expression: {exc}") from exc


def _eval(tree: ast.Expression, values: dict[str, Any]) -> float | None:
    """Evaluate a parsed arithmetic expression over metric ``values`` (no ``eval``)."""
    return _eval_node(tree.body, values)


def _eval_node(node: ast.AST, values: dict[str, Any]) -> float | None:
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        return float(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _eval_node(node.operand, values)
        if operand is None:
            return None
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise MetricError(f"expression references unknown metric {node.id!r}")
        value = values[node.id]
        return None if value is None else float(value)
    raise MetricError("expression allows only + - * / over metric names and numbers")


def _period_value(period_agg: str, values: list[Any]) -> float | None:
    """Aggregate a window of values for a cumulative metric."""
    numbers = [float(v) for v in values]
    if not numbers:
        return None
    if period_agg == "average":
        return fmean(numbers)
    if period_agg == "min":
        return min(numbers)
    if period_agg == "max":
        return max(numbers)
    return sum(numbers)


def _value_expr(measure: Measure) -> str:
    """The SQL aggregation expression for a measure."""
    if measure.agg == "count":
        return "count(*)"
    if measure.agg == "count_distinct":
        return f"count(distinct {_qi(_require(measure))})"
    return f"{_AGG_FN[measure.agg]}({_qi(_require(measure))})"


def _require(measure: Measure) -> str:
    if measure.column is None:  # guaranteed by the spec for non-count aggregations
        raise MetricError(f"aggregation {measure.agg!r} needs a column")
    return measure.column


def _where(
    metric: Metric, filters: Sequence[Filter], available: set[str]
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for f in filters:
        _require_column(metric, f.column, available)
        if f.op == "in":
            values = list(f.value) if isinstance(f.value, (list, tuple)) else [f.value]
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{_qi(f.column)} IN ({placeholders})")
            params.extend(values)
        else:
            clauses.append(f"{_qi(f.column)} {_OP_SQL[f.op]} ?")
            params.append(f.value)
    return " AND ".join(clauses), params


def _require_column(metric: Metric, column: str, available: set[str]) -> None:
    if column not in available:
        raise MetricError(
            f"metric {metric.name!r} references column {column!r}, "
            f"absent from its source rows"
        )


def _columns(rows: list[Row]) -> set[str]:
    return {key for row in rows for key in row}


def _qi(name: str) -> str:
    """Quote an identifier, neutralizing embedded quotes."""
    return '"' + name.replace('"', '""') + '"'
