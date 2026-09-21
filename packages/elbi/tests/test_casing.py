"""The camelCase API boundary: field-name conversion with the correctness exemptions."""

from __future__ import annotations

from elbi.casing import camelize, snakeify


def test_camelize_rewrites_field_names() -> None:
    assert camelize({"target_kind": "metric", "min_value": 1}) == {
        "targetKind": "metric",
        "minValue": 1,
    }


def test_camelize_is_recursive() -> None:
    assert camelize({"a_b": {"c_d": [{"e_f": 1}]}}) == {"aB": {"cD": [{"eF": 1}]}}


def test_camelize_leaves_tabular_data_verbatim() -> None:
    # Keys under `rows`/`variables` are user column/variable names, not API fields.
    got = camelize({"row_count": 2, "rows": [{"total_spend": 10, "region": "west"}]})
    assert got == {"rowCount": 2, "rows": [{"total_spend": 10, "region": "west"}]}


def test_camelize_leaves_run_history_cells_verbatim() -> None:
    # `cells` is keyed by asset name (e.g. `clean_sales`), not API field names, so the
    # run-history matrix can still look up a cell by the asset's real name.
    got = camelize({"run_id": "r1", "cells": {"clean_sales": "skipped"}})
    assert got == {"runId": "r1", "cells": {"clean_sales": "skipped"}}


def test_camelize_leaves_mime_and_dotted_keys_untouched() -> None:
    # nbformat/MIME-bundle keys are not snake_case field names and must not change.
    assert camelize({"text/plain": "x", "a.b": 1, "already": 2}) == {
        "text/plain": "x",
        "a.b": 1,
        "already": 2,
    }


def test_snakeify_is_the_inverse_for_field_names() -> None:
    original = {"targetKind": "metric", "minValue": 1, "nested": {"createdAt": "t"}}
    assert snakeify(original) == {
        "target_kind": "metric",
        "min_value": 1,
        "nested": {"created_at": "t"},
    }
    assert camelize(snakeify(original)) == original


def test_round_trip_preserves_data_and_fields() -> None:
    payload = {
        "source_certified": True,
        "time_dimension": {"column": "day", "grain": "month"},
        "rows": [{"total_spend": 1}],
    }
    assert snakeify(camelize(payload)) == payload


def test_camelize_leaves_an_artifact_value_verbatim() -> None:
    """A dashboard tile's rows are a derivation's output, keyed by user column names.

    Camelizing them renamed `mrr_usd` to `mrrUsd` on the wire, so a widget's
    `viz.field` (written in the spec as the derivation writes it) matched nothing and
    the tile rendered a dash.
    """
    payload = {"widget_id": "mrr", "value": [{"mrr_usd": 300.0, "is_fixed": True}]}

    assert camelize(payload) == {
        "widgetId": "mrr",
        "value": [{"mrr_usd": 300.0, "is_fixed": True}],
    }


def test_camelize_leaves_a_scalar_value_alone_either_way() -> None:
    """The other `value` fields are scalars, which the transform never rewrote."""
    assert camelize({"value": 3, "some_field": 1}) == {"value": 3, "someField": 1}


def test_snakeify_leaves_an_artifact_value_verbatim() -> None:
    """The inverse direction: a filter's value must not be snake-cased into data."""
    payload = {"widgetId": "mrr", "value": [{"mrrUsd": 300.0}]}

    assert snakeify(payload) == {"widget_id": "mrr", "value": [{"mrrUsd": 300.0}]}
