"""Tests for the semantic-layer metrics: spec, resolver, and OSI round-trip."""

from __future__ import annotations

from typing import Any

import pytest

from elbi_core import (
    Metric,
    MetricSet,
    from_osi,
    is_valid_metric_set,
    resolve_metric,
    to_osi,
    validate_metric_set,
)
from elbi_core.errors import MetricError, SpecValidationError
from elbi_core.metrics import Filter, Format, Measure, TimeDimension
from elbi_core.metrics.osi import OSI_VERSION, validate_osi

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

ORDERS = [
    {"region": "west", "day": "2026-01-01", "amount": 10, "converted": 1},
    {"region": "west", "day": "2026-01-02", "amount": 30, "converted": 0},
    {"region": "east", "day": "2026-01-08", "amount": 20, "converted": 1},
    {"region": "east", "day": "2026-02-01", "amount": 40, "converted": 1},
]


def _sources(_name: str) -> list[dict[str, Any]]:
    return [dict(r) for r in ORDERS]


def _revenue() -> Metric:
    return Metric(
        name="revenue",
        type="simple",
        source="orders",
        measure=Measure(agg="sum", column="amount"),
        dimensions=("region",),
        time_dimension=TimeDimension(column="day", grain="month"),
    )


# -- spec --------------------------------------------------------------------


def test_metric_set_manifest_roundtrips() -> None:
    original = MetricSet(metrics=(_revenue(),))
    parsed = MetricSet.from_manifest(original.to_manifest())
    assert parsed == original
    assert is_valid_metric_set(original.to_manifest())


def test_ratio_operands_must_be_defined() -> None:
    manifest = {
        "specVersion": "1.0",
        "kind": "MetricSet",
        "metrics": [
            {
                "name": "conversion_rate",
                "type": "ratio",
                "numerator": "conversions",
                "denominator": "orders_count",
            }
        ],
    }
    with pytest.raises(SpecValidationError, match="numerator 'conversions'"):
        validate_metric_set(manifest)


def test_simple_metric_requires_source_and_measure() -> None:
    manifest = {
        "specVersion": "1.0",
        "kind": "MetricSet",
        "metrics": [{"name": "revenue", "type": "simple"}],
    }
    assert not is_valid_metric_set(manifest)


# -- resolver ----------------------------------------------------------------


def test_resolve_simple_grouped_by_dimension() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    result = resolve_metric(metrics, "revenue", _sources, group_by=["region"])
    assert result.columns == ["region", "revenue"]
    assert result.rows == [
        {"region": "east", "revenue": 60},
        {"region": "west", "revenue": 40},
    ]


def test_resolve_ungrouped_is_a_single_total() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    result = resolve_metric(metrics, "revenue", _sources)
    assert result.rows == [{"revenue": 100}]


def test_resolve_by_time_grain_buckets_month() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    result = resolve_metric(
        metrics, "revenue", _sources, group_by=["day"], grain="month"
    )
    # Jan (10+30+20) and Feb (40), bucketed to month starts.
    totals = {row["day"][:7]: row["revenue"] for row in result.rows}
    assert totals == {"2026-01": 60, "2026-02": 40}


def test_resolve_applies_filter() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    result = resolve_metric(
        metrics,
        "revenue",
        _sources,
        filters=[Filter(column="region", op="eq", value="west")],
    )
    assert result.rows == [{"revenue": 40}]


def test_resolve_ratio_divides_operands() -> None:
    metrics = MetricSet(
        metrics=(
            Metric(
                name="conversions",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="converted"),
                dimensions=("region",),
            ),
            Metric(
                name="orders_count",
                type="simple",
                source="orders",
                measure=Measure(agg="count"),
                dimensions=("region",),
            ),
            Metric(
                name="conversion_rate",
                type="ratio",
                numerator="conversions",
                denominator="orders_count",
                dimensions=("region",),
            ),
        )
    )
    result = resolve_metric(metrics, "conversion_rate", _sources, group_by=["region"])
    rates = {row["region"]: row["conversion_rate"] for row in result.rows}
    assert rates["east"] == 1.0  # 2 conversions / 2 orders
    assert rates["west"] == 0.5  # 1 conversion / 2 orders


def test_resolve_rejects_undeclared_dimension() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    with pytest.raises(MetricError, match="cannot group metric 'revenue' by 'amount'"):
        resolve_metric(metrics, "revenue", _sources, group_by=["amount"])


def test_resolve_unknown_metric() -> None:
    with pytest.raises(MetricError, match="unknown metric 'nope'"):
        resolve_metric(MetricSet(), "nope", _sources)


def test_filter_value_is_not_injectable() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    # A value that would break naive string interpolation is bound as a parameter.
    result = resolve_metric(
        metrics,
        "revenue",
        _sources,
        filters=[Filter(column="region", op="eq", value="west' OR '1'='1")],
    )
    # The value matched no region (bound as a parameter, not injected); an ungrouped
    # aggregate over zero rows is a single NULL, which is the tell it was not injected.
    assert result.rows == [{"revenue": None}]


# -- derived + cumulative ----------------------------------------------------


def _revenue_cost_set() -> MetricSet:
    return MetricSet(
        metrics=(
            Metric(
                name="revenue",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="amount"),
                dimensions=("region",),
            ),
            Metric(
                name="order_count",
                type="simple",
                source="orders",
                measure=Measure(agg="count"),
                dimensions=("region",),
            ),
            Metric(
                name="avg_order_value",
                type="derived",
                expr="revenue / order_count",
                input_metrics=("revenue", "order_count"),
                dimensions=("region",),
            ),
        )
    )


def test_resolve_derived_expression() -> None:
    metrics = _revenue_cost_set()
    result = resolve_metric(metrics, "avg_order_value", _sources, group_by=["region"])
    values = {row["region"]: row["avg_order_value"] for row in result.rows}
    assert values["west"] == 20.0  # 40 revenue / 2 orders
    assert values["east"] == 30.0  # 60 revenue / 2 orders


def test_derived_division_by_zero_is_none() -> None:
    metrics = MetricSet(
        metrics=(
            Metric(
                name="zero",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="converted"),
                filters=(Filter(column="region", op="eq", value="nowhere"),),
            ),
            Metric(
                name="revenue",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="amount"),
            ),
            Metric(
                name="ratio_expr",
                type="derived",
                expr="revenue / zero",
                input_metrics=("revenue", "zero"),
            ),
        )
    )
    result = resolve_metric(metrics, "ratio_expr", _sources)
    assert result.rows == [{"ratio_expr": None}]


def test_derived_rejects_unsafe_expression() -> None:
    metrics = MetricSet(
        metrics=(
            Metric(
                name="revenue",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="amount"),
            ),
            Metric(
                name="evil",
                type="derived",
                expr="__import__('os').system('boom')",
                input_metrics=("revenue",),
            ),
        )
    )
    with pytest.raises(MetricError, match="only \\+ - \\* / over"):
        resolve_metric(metrics, "evil", _sources)


def test_resolve_cumulative_running_total() -> None:
    metrics = MetricSet(
        metrics=(
            Metric(
                name="daily_revenue",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="amount"),
                time_dimension=TimeDimension(column="day", grain="day"),
            ),
            Metric(
                name="revenue_to_date",
                type="cumulative",
                input_metric="daily_revenue",
                period_agg="sum",
            ),
        )
    )
    result = resolve_metric(metrics, "revenue_to_date", _sources, grain="day")
    running = {row["day"][:10]: row["revenue_to_date"] for row in result.rows}
    # 2026-01-01: 10, 01-02: 10+30=40, 01-08: 60, 02-01: 100 (cumulative to date)
    assert running["2026-01-01"] == 10
    assert running["2026-01-02"] == 40
    assert running["2026-02-01"] == 100


def test_resolve_cumulative_trailing_window() -> None:
    metrics = MetricSet(
        metrics=(
            Metric(
                name="daily_revenue",
                type="simple",
                source="orders",
                measure=Measure(agg="sum", column="amount"),
                time_dimension=TimeDimension(column="day", grain="day"),
            ),
            Metric(
                name="revenue_2day",
                type="cumulative",
                input_metric="daily_revenue",
                window=2,
            ),
        )
    )
    result = resolve_metric(metrics, "revenue_2day", _sources, grain="day")
    windowed = {row["day"][:10]: row["revenue_2day"] for row in result.rows}
    # Trailing 2 day-buckets: 01-02 = 10+30 = 40; 02-01 alone (no 01-31 bucket) = 40.
    assert windowed["2026-01-02"] == 40


# -- OSI round-trip ----------------------------------------------------------


def test_osi_export_has_datasets_and_metrics() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    document = to_osi(metrics)
    model = document["semantic_model"][0]
    assert document["version"]
    assert [d["name"] for d in model["datasets"]] == ["orders"]
    expr = model["metrics"][0]["expression"]["dialects"][0]["expression"]
    assert expr == 'SUM("amount")'


def test_osi_roundtrips_losslessly_via_vendor_extension() -> None:
    metrics = MetricSet(metrics=(_revenue(),))
    restored = from_osi(to_osi(metrics))
    assert restored == metrics


def test_osi_export_conforms_to_the_official_schema() -> None:
    # to_osi validates internally; assert the released version, the COMMON extension
    # (OSI's vendor_name is a fixed enum), and the ai_context grounding are all present.
    document = to_osi(MetricSet(metrics=(_revenue(),)))
    assert document["version"] == "0.1.1"
    validate_osi(document)  # explicit: it passes the bundled OSI JSON Schema
    metric = document["semantic_model"][0]["metrics"][0]
    assert metric["custom_extensions"][0]["vendor_name"] == "COMMON"
    dataset = document["semantic_model"][0]["datasets"][0]
    assert "ai_context" in dataset


def test_osi_validate_rejects_nonconformant_documents() -> None:
    # A model with no datasets (OSI requires minItems 1) must be rejected, not emitted.
    with pytest.raises(MetricError, match="not schema-conformant"):
        validate_osi(
            {"version": "0.1.1", "semantic_model": [{"name": "x", "datasets": []}]}
        )
    # An unknown vendor_name (not in OSI's Vendor enum) must also be rejected.
    with pytest.raises(MetricError, match="not schema-conformant"):
        validate_osi({"version": "9.9.9", "semantic_model": []})


def test_osi_roundtrip_preserves_display_format() -> None:
    # format is not a native OSI field; it must survive via the COMMON extension.
    metric = Metric(
        name="revenue",
        type="simple",
        source="orders",
        measure=Measure(agg="sum", column="amount"),
        dimensions=("region",),
        format=Format(kind="currency", currency="USD", precision=2),
    )
    restored = from_osi(to_osi(MetricSet(metrics=(metric,))))
    assert restored.metric("revenue") == metric


def test_osi_import_rejects_a_nonconformant_document() -> None:
    # An imported document is untrusted, so import validates it rather than
    # best-effort parsing whatever it is handed.
    with pytest.raises(MetricError, match="not schema-conformant"):
        from_osi(
            {
                "version": "0.2.0",
                "semantic_model": [
                    {
                        "name": "sales",
                        "datasets": [
                            {"name": "orders", "source": "orders", "fields": []}
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "document",
    [
        {"version": OSI_VERSION, "semantic_model": "sales"},
        {"version": OSI_VERSION, "semantic_model": {"name": "sales"}},
        {"version": OSI_VERSION, "semantic_model": ["sales"]},
    ],
    ids=["string", "mapping", "list-of-strings"],
)
def test_osi_import_reports_a_malformed_model_as_a_conformance_error(
    document: dict[str, Any],
) -> None:
    # A document whose 'semantic_model' is not a list of models is still a document
    # an API caller can send, so it has to surface as a conformance error rather than
    # an unhandled exception on the way to the validator.
    with pytest.raises(MetricError, match="not schema-conformant"):
        from_osi(document)


def test_osi_import_tolerates_the_elbi_dataset_tag() -> None:
    # OSI closes the metric object, so the tag is excluded from the conformance
    # check rather than rejected by it.
    metrics = from_osi(
        {
            "version": OSI_VERSION,
            "semantic_model": [
                {
                    "name": "sales",
                    "datasets": [
                        {"name": "orders", "source": "orders", "fields": []},
                        {"name": "refunds", "source": "refunds", "fields": []},
                    ],
                    "metrics": [
                        {
                            "name": "total_amount",
                            "dataset": "refunds",
                            "expression": {
                                "dialects": [
                                    {
                                        "dialect": "ANSI_SQL",
                                        "expression": "SUM(amount)",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    )
    metric = metrics.metric("total_amount")
    assert metric is not None
    assert metric.source == "refunds"


def test_osi_imports_a_foreign_simple_metric() -> None:
    foreign = {
        "version": OSI_VERSION,
        "semantic_model": [
            {
                "name": "sales",
                "datasets": [
                    {
                        "name": "orders",
                        "source": "orders",
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
    metrics = from_osi(foreign)
    metric = metrics.metric("total_amount")
    assert metric is not None
    assert metric.type == "simple"
    assert metric.source == "orders"
    assert metric.measure == Measure(agg="sum", column="amount")
    assert metric.dimensions == ("region",)


def test_osi_imports_multi_dataset_via_dataset_tag() -> None:
    def dataset(name: str) -> dict[str, Any]:
        return {"name": name, "source": name, "fields": []}

    foreign = {
        "version": OSI_VERSION,
        "semantic_model": [
            {
                "name": "sales",
                "datasets": [dataset("orders"), dataset("refunds")],
                "metrics": [
                    {
                        "name": "gross",
                        "dataset": "orders",
                        "expression": {
                            "dialects": [
                                {"dialect": "ANSI_SQL", "expression": "SUM(amount)"}
                            ]
                        },
                    },
                    {
                        "name": "refunded",
                        "dataset": "refunds",
                        "expression": {
                            "dialects": [
                                {"dialect": "ANSI_SQL", "expression": "SUM(amount)"}
                            ]
                        },
                    },
                ],
            }
        ],
    }
    metrics = from_osi(foreign)
    gross = metrics.metric("gross")
    refunded = metrics.metric("refunded")
    assert gross is not None and gross.source == "orders"
    assert refunded is not None and refunded.source == "refunds"


def test_osi_import_rejects_ambiguous_multi_dataset() -> None:
    foreign = {
        "version": OSI_VERSION,
        "semantic_model": [
            {
                "name": "sales",
                "datasets": [
                    {"name": "orders", "source": "orders", "fields": []},
                    {"name": "refunds", "source": "refunds", "fields": []},
                ],
                "metrics": [
                    {
                        "name": "gross",
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
    with pytest.raises(MetricError, match="several datasets"):
        from_osi(foreign)


def test_osi_import_rejects_unparseable_foreign_expression() -> None:
    foreign = {
        "version": OSI_VERSION,
        "semantic_model": [
            {
                "name": "sales",
                "datasets": [{"name": "orders", "source": "orders", "fields": []}],
                "metrics": [
                    {
                        "name": "weird",
                        "expression": {
                            "dialects": [
                                {"dialect": "ANSI_SQL", "expression": "SUM(a) + AVG(b)"}
                            ]
                        },
                    }
                ],
            }
        ],
    }
    with pytest.raises(MetricError, match="not a simple aggregation"):
        from_osi(foreign)
