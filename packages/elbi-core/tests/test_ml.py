"""The ml extra's public boundary: train, register, promote, load, and score.

The training tests run real FLAML searches (tiny ``max_iter`` budgets so they are
fast and reproducible) against a real SQLite-backed MLflow store, because the
behavior under test is the integration contract itself: a train call must leave a
registered, loadable, scoreable model behind, with held-out metrics and the oracle's
verdict recorded. The scoring-protocol tests are pure and property-tested.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elbi_core.errors import ModelError
from elbi_core.ml import (
    ModelRegistry,
    parse_invocations,
    predictions_payload,
    train_automl,
)

# Third-party training stacks warn freely (convergence, dtypes, serialization);
# these tests assert behavior, not warning hygiene, so warnings stay warnings.
pytestmark = pytest.mark.filterwarnings("ignore")


def _classification_rows(n: int = 240, seed: int = 3) -> list[dict[str, str]]:
    """String-valued rows (as datasets load) with a strong, clean binary signal."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        segment = rng.choice(["a", "b"])
        label = "1" if x1 + 0.2 * x2 + rng.gauss(0, 0.3) > 0 else "0"
        rows.append(
            {"x1": f"{x1:.4f}", "x2": f"{x2:.4f}", "segment": segment, "y": label}
        )
    return rows


def _regression_rows(n: int = 200, seed: int = 5) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        y = 3.0 * x1 - 2.0 * x2 + rng.gauss(0, 0.5)
        rows.append({"x1": f"{x1:.4f}", "x2": f"{x2:.4f}", "y": f"{y:.4f}"})
    return rows


def _noise_rows(n: int = 160, seed: int = 9) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return [
        {
            "x1": f"{rng.gauss(0, 1):.4f}",
            "x2": f"{rng.gauss(0, 1):.4f}",
            "y": str(rng.randint(0, 1)),
        }
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One registry with two trained versions of the same model (lgbm, then rf)."""
    root = tmp_path_factory.mktemp("mlstore")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    rows = _classification_rows()
    common: dict[str, Any] = {
        "name": "churn",
        "target": "y",
        "tracking_uri": uri,
        "artifact_root": root / "mlartifacts",
        "time_budget": 60.0,
        "seed": 7,
    }
    first = train_automl(rows, estimator_list=["lgbm"], max_iter=3, **common)
    second = train_automl(rows, estimator_list=["rf"], max_iter=2, **common)
    return {"uri": uri, "rows": rows, "first": first, "second": second}


def test_first_sound_version_is_auto_promoted(trained: dict[str, Any]) -> None:
    first, second = trained["first"], trained["second"]
    assert first.version == 1 and second.version == 2
    assert first.oracle_verdict == "sound"
    assert first.champion is True  # the first sound version earns the alias
    assert second.champion is False  # a later version never displaces it silently
    assert first.model_uri == "models:/churn/1"
    assert first.best_estimator == "lgbm"
    assert first.task == "classification"
    assert first.target == "y" and set(first.features) == {"x1", "x2", "segment"}


def test_reported_metrics_are_holdout(trained: dict[str, Any]) -> None:
    report = trained["first"]
    assert {"accuracy", "f1"} <= set(report.metrics)
    assert all(0.0 <= v <= 1.0 for k, v in report.metrics.items() if k != "log_loss")
    # the strong signal must actually be learnable, or the metrics are not real
    assert report.metrics["accuracy"] > 0.8
    # an 80/20 split: the holdout the metrics come from is the untouched fifth
    assert report.n_test == pytest.approx(len(trained["rows"]) * 0.2, abs=2)
    assert report.n_train + report.n_test == len(trained["rows"])
    rendered = report.render()
    assert "held-out" in rendered.lower()
    assert "champion" in rendered


def test_registry_lists_models_and_versions(trained: dict[str, Any]) -> None:
    registry = ModelRegistry(trained["uri"])
    (model,) = registry.models()
    assert model.name == "churn"
    assert model.latest_version == 2
    assert model.champion_version == 1
    versions = registry.versions("churn")
    assert [v.version for v in versions] == [2, 1]  # newest first
    by_version = {v.version: v for v in versions}
    assert "champion" in by_version[1].aliases
    assert by_version[1].params["best_estimator"] == "lgbm"
    assert "holdout_accuracy" in by_version[1].metrics
    assert by_version[1].tags["elbi.oracle_verdict"] == "sound"


def test_bare_name_serves_the_champion_and_scores(trained: dict[str, Any]) -> None:
    registry = ModelRegistry(trained["uri"])
    assert registry.resolve("churn") == 1  # champion, not newest
    model = registry.load("churn")
    frame, params = parse_invocations(
        {
            "dataframe_records": [
                {"x1": 3.0, "x2": 1.0, "segment": "a"},
                {"x1": -3.0, "x2": -1.0, "segment": "b"},
            ]
        }
    )
    assert params is None
    payload = predictions_payload(model.predict(frame))
    assert [int(p) for p in payload["predictions"]] == [1, 0]
    # loading is cached per resolved version: the same object comes back
    assert registry.load("churn") is model


def test_promote_moves_the_champion(trained: dict[str, Any]) -> None:
    registry = ModelRegistry(trained["uri"])
    registry.promote("churn", 2)
    try:
        assert registry.resolve("churn") == 2
        assert registry.resolve("churn", "champion") == 2
        assert registry.resolve("churn", "1") == 1  # digits are a version number
    finally:
        registry.promote("churn", 1)  # restore for the other module tests
    assert registry.resolve("churn") == 1
    with pytest.raises(ModelError):
        registry.promote("churn", 99)
    with pytest.raises(ModelError):
        registry.resolve("churn", "no_such_alias")
    with pytest.raises(ModelError):
        registry.versions("nonexistent")


def test_uncertified_signal_is_registered_but_not_promoted(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        _noise_rows(),
        name="noise",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=60.0,
        estimator_list=["lgbm"],
        max_iter=2,
        seed=7,
    )
    assert report.oracle_verdict != "sound"
    assert report.champion is False
    registry = ModelRegistry(uri)
    (model,) = registry.models()
    assert model.champion_version is None  # no silent champion over noise
    # ...and nothing serves it by default. Registering a version and deploying one are
    # two acts: the alias is the second, and only a sound verdict earns it. Resolving to
    # the newest version instead would have served precisely the models that failed the
    # gate.
    with pytest.raises(ModelError, match="no @champion"):
        registry.resolve("noise")
    assert registry.resolve("noise", "1") == 1  # naming it explicitly still works


def test_auto_task_resolves_regression(tmp_path: Path) -> None:
    report = train_automl(
        _regression_rows(),
        name="value",
        target="y",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "mlartifacts",
        time_budget=60.0,
        estimator_list=["lgbm"],
        max_iter=10,
        seed=7,
    )
    assert report.task == "regression"
    assert {"r2", "rmse", "mae"} <= set(report.metrics)
    assert report.metrics["r2"] > 0.7  # the linear signal must be learnable


def test_train_validates_its_inputs(tmp_path: Path) -> None:
    rows = _classification_rows(n=40)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with pytest.raises(ModelError, match="target column"):
        train_automl(rows, name="m", target="missing", tracking_uri=uri)
    with pytest.raises(ModelError, match="feature columns"):
        train_automl(rows, name="m", target="y", features=["nope"], tracking_uri=uri)
    with pytest.raises(ModelError, match="cannot also be a feature"):
        train_automl(rows, name="m", target="y", features=["y", "x1"], tracking_uri=uri)
    with pytest.raises(ModelError, match="unsupported task"):
        train_automl(rows, name="m", target="y", task="ranking", tracking_uri=uri)


# -- the MLflow /invocations payload protocol ------------------------------------


def test_all_four_framings_parse_to_the_same_frame() -> None:
    records = [{"a": 1.0, "b": "x"}, {"a": 2.0, "b": "y"}]
    split = {"columns": ["a", "b"], "data": [[1.0, "x"], [2.0, "y"]]}
    frames = [
        parse_invocations({"dataframe_records": records})[0],
        parse_invocations({"dataframe_split": split})[0],
        parse_invocations({"instances": records})[0],
        parse_invocations({"inputs": {"a": [1.0, 2.0], "b": ["x", "y"]}})[0],
    ]
    reference = frames[0][["a", "b"]].values.tolist()
    for frame in frames[1:]:
        assert frame[["a", "b"]].values.tolist() == reference


def test_exactly_one_framing_is_required() -> None:
    with pytest.raises(ModelError, match="exactly one"):
        parse_invocations({})
    # naming the offending keys is part of the contract: the caller sent both
    with pytest.raises(ModelError, match="got instances, inputs"):
        parse_invocations({"instances": [[1]], "inputs": [[1]]})
    with pytest.raises(ModelError, match="JSON object"):
        parse_invocations([1, 2])
    with pytest.raises(ModelError, match="no rows"):
        parse_invocations({"dataframe_records": []})
    with pytest.raises(ModelError, match="params"):
        parse_invocations({"instances": [[1]], "params": [1]})
    # a framing pandas cannot build becomes the protocol's own error, not a crash
    with pytest.raises(ModelError, match="malformed 'dataframe_split'"):
        parse_invocations({"dataframe_split": {"columns": ["a"], "data": 7}})


def test_tensor_framings_accept_bare_arrays() -> None:
    frame, _ = parse_invocations({"instances": [[1.0, 2.0], [3.0, 4.0]]})
    assert frame.shape == (2, 2)  # positional columns, one row per instance
    assert frame.values.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    single, _ = parse_invocations({"instances": [{"a": 1.0}]})
    assert single.to_dict(orient="records") == [{"a": 1.0}]


def test_params_pass_through() -> None:
    _, params = parse_invocations(
        {"dataframe_records": [{"a": 1}], "params": {"temperature": 0}}
    )
    assert params == {"temperature": 0}


def test_predictions_payload_is_json_safe() -> None:
    import json

    import numpy as np
    import pandas as pd

    payloads = [
        (np.array([1, 0]), {"predictions": [1, 0]}),
        (pd.Series([0.5]), {"predictions": [0.5]}),
        (pd.DataFrame({"p": [1]}), {"predictions": [{"p": 1}]}),
        ([np.int64(3), "a"], {"predictions": [3, "a"]}),
        (np.float64(0.25), {"predictions": 0.25}),
    ]
    for predictions, expected in payloads:
        payload = predictions_payload(predictions)
        assert payload == expected
        # "JSON-safe" is the contract: numpy scalars must actually serialize
        json.dumps(payload)


_CELL = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.text(min_size=0, max_size=8),
)


@given(
    st.lists(
        st.fixed_dictionaries({"a": _CELL, "b": _CELL, "c": _CELL}),
        min_size=1,
        max_size=8,
    )
)
def test_records_roundtrip_through_the_protocol(rows: list[dict[str, Any]]) -> None:
    # Whatever the cells hold, parsing records then reading them back preserves
    # every value; and the split framing of the same data parses identically.
    frame, _ = parse_invocations({"dataframe_records": rows})
    assert frame.to_dict(orient="records") == rows
    split = {"columns": list(frame.columns), "data": frame.values.tolist()}
    again, _ = parse_invocations({"dataframe_split": split})
    assert again.to_dict(orient="records") == rows


def test_classification_refuses_absurd_cardinality(tmp_path: Path) -> None:
    # A continuous target forced into classification would crawl through a giant
    # softmax until the budget dies with no model; refusing it names the fix.
    rows = _regression_rows(n=200)
    with pytest.raises(ModelError, match="distinct values"):
        train_automl(
            rows,
            name="m",
            target="y",
            task="classification",
            tracking_uri=f"sqlite:///{tmp_path / 'm.db'}",
        )


def test_training_refuses_an_unusable_target(tmp_path: Path) -> None:
    rows = [{"x1": str(i), "y": ""} for i in range(50)]
    with pytest.raises(ModelError, match="usable rows"):
        train_automl(
            rows,
            name="m",
            target="y",
            tracking_uri=f"sqlite:///{tmp_path / 'm.db'}",
        )


def test_schema_parse_casts_json_numbers(trained: dict[str, Any]) -> None:
    # JSON has one number type; a payload sending 3 for a double column must cast
    # to the model's schema exactly as the reference scoring server does, and a
    # missing required column must be the protocol's error, not a crash.
    registry = ModelRegistry(trained["uri"])
    model = registry.load("churn")
    schema = model.metadata.get_input_schema()
    frame, _ = parse_invocations(
        {"dataframe_records": [{"x1": 3, "x2": 1, "segment": "a"}]}, schema=schema
    )
    assert str(frame["x1"].dtype) == "float64"
    # the cast frame must actually score (this exact shape used to 500)
    assert len(model.predict(frame)) == 1


def test_trials_are_logged_as_child_runs_with_lineage(tmp_path: Path) -> None:
    # Each search trial must appear as a nested run under the training run (the
    # MLflow UI leaderboard), and the run and version must carry the data lineage.
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    train_automl(
        _classification_rows(n=120),
        name="lead",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=60.0,
        max_iter=4,
        seed=7,
        dataset="demo",
    )
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("lead")
    runs = client.search_runs([experiment.experiment_id], max_results=200)
    parents = [r for r in runs if r.data.tags.get("elbi.trial") != "true"]
    trials = [r for r in runs if r.data.tags.get("elbi.trial") == "true"]
    assert len(parents) == 1 and len(trials) >= 1
    parent_id = parents[0].info.run_id
    assert all(t.data.tags["mlflow.parentRunId"] == parent_id for t in trials)
    assert any(t.data.tags.get("elbi.best_trial") == "true" for t in trials)
    assert all("validation_loss" in t.data.metrics for t in trials)
    assert parents[0].data.tags["elbi.dataset"] == "demo"
    data_hash = parents[0].data.tags["elbi.data_hash"]
    assert len(data_hash) == 64
    version = client.get_model_version("lead", "1")
    assert version.tags["elbi.data_hash"] == data_hash


def test_evaluation_and_explanation_artifacts_ride_on_the_run(
    trained: dict[str, Any],
) -> None:
    # A training run must arrive with its evidence attached: MLflow's evaluation
    # plots, the feature importances, and a SHAP beeswarm.
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=trained["uri"])
    artifacts = {a.path for a in client.list_artifacts(trained["first"].run_id)}
    assert {
        "confusion_matrix.png",
        "roc_curve_plot.png",
        "feature_importance.csv",
        "shap_beeswarm.png",
    } <= artifacts
    run = client.get_run(trained["first"].run_id)
    assert "f1_score" in run.data.metrics  # evaluate()'s own per-run metrics


def test_glass_box_artifacts_and_model_card(trained: dict[str, Any]) -> None:
    # The run carries the data profile and an editable reproduction script, and
    # the registered version's description IS its model card.
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=trained["uri"])
    artifacts = {a.path for a in client.list_artifacts(trained["first"].run_id)}
    assert {"data_profile.md", "training_code.py", "model_card.md"} <= artifacts
    version = client.get_model_version("churn", "1")
    assert version.description.startswith("# Model card: churn v1")
    assert "Held-out metrics" in version.description
    assert "sound" in version.description  # the oracle verdict is in the card


def test_ts_forecast_trains_and_serves_future_periods(tmp_path: Path) -> None:
    # Forecasting: temporal holdout (never a shuffle), forecast-gate verdict, and
    # a registered model that predicts from future timestamps.
    import datetime as dt
    import math

    # The gate scores MASE against a *last-value* naive baseline, so the model's edge
    # is the weekly shape naive cannot anticipate. A weak seasonal amplitude leaves
    # MASE close enough to 1 that the platform's own boosting numerics decide the
    # verdict, which is what made this test flaky on macOS. A pronounced, quiet cycle
    # with barely any trend to extrapolate puts the margin beyond that noise.
    rng = random.Random(0)
    rows = []
    for i in range(150):
        value = 10 + 0.02 * i + 6 * math.sin(i * 2 * math.pi / 7) + rng.gauss(0, 0.25)
        rows.append(
            {
                "day": (dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat(),
                "y": f"{value:.3f}",
            }
        )
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        rows,
        name="sales_fc",
        target="y",
        task="ts_forecast",
        time_col="day",
        horizon=14,
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        # The verdict depends on how far the search got, so the clock must never be
        # what ends it: at 60s a loaded runner finished fewer iterations, fit worse,
        # and returned "inconclusive". max_iter bounds this search instead, and is
        # generous enough that a slower learner still gets found. The budget stays a
        # number only because FLAML's time-series task cannot take -1 for unbounded.
        time_budget=600.0,
        max_iter=12,
        seed=7,
    )
    assert report.task == "ts_forecast"
    assert {"rmse", "mae", "mape"} <= set(report.metrics)
    assert report.metrics["mape"] < 0.2  # the seasonal signal must be learnable
    assert report.oracle_verdict == "sound"  # the forecast gate, not prediction
    assert report.champion is True

    import pandas as pd

    model = ModelRegistry(uri).load("sales_fc")
    future = pd.DataFrame({"day": pd.date_range("2024-05-30", periods=3, freq="D")})
    assert len(model.predict(future)) == 3

    with pytest.raises(ModelError, match="time_col"):
        train_automl(rows, name="bad", target="y", task="ts_forecast", tracking_uri=uri)
    with pytest.raises(ModelError, match="horizon"):
        train_automl(
            rows,
            name="bad",
            target="y",
            task="ts_forecast",
            time_col="day",
            horizon=100,
            tracking_uri=uri,
        )


def test_data_drift_separates_shifted_from_stable() -> None:
    # The drift verdict must move with the data: a shifted feature drifts, the
    # same distribution does not, and too-thin traffic is refused, not guessed.
    from elbi_core.ml import data_drift

    rng = random.Random(0)
    reference = [{"x1": rng.gauss(0, 1), "x2": rng.gauss(5, 2)} for _ in range(400)]
    shifted = [{"x1": rng.gauss(3, 1), "x2": rng.gauss(5, 2)} for _ in range(200)]
    report = data_drift(reference, shifted)
    by_column = {c.column: c for c in report.columns}
    assert by_column["x1"].drifted is True
    assert by_column["x2"].drifted is False
    assert report.n_drifted == 1 and report.share_drifted == 0.5
    assert report.dataset_drift is True  # half the columns moved
    assert "<html" in report.html.lower()

    stable = [{"x1": rng.gauss(0, 1), "x2": rng.gauss(5, 2)} for _ in range(200)]
    calm = data_drift(reference, stable)
    assert calm.n_drifted == 0 and calm.dataset_drift is False

    with pytest.raises(ModelError, match="at least"):
        data_drift(reference, shifted[:5])


def test_autogluon_engine_trains_registers_and_serves(tmp_path: Path) -> None:
    # The accuracy-first engine through the same lifecycle: leaderboard child
    # runs, engine lineage, a pyfunc-served predictor, and the champion policy.
    pytest.importorskip("autogluon.tabular")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        _classification_rows(),
        name="ag",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=60.0,
        estimator_list=["lgbm"],
        seed=7,
        engine="autogluon",
        dataset="demo",
    )
    assert report.metrics["accuracy"] > 0.8
    assert report.oracle_verdict == "sound" and report.champion is True
    assert "WeightedEnsemble" in report.best_estimator or report.best_estimator

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(report.run_id)
    assert run.data.params["engine"] == "autogluon"
    experiment = client.get_experiment_by_name("ag")
    trials = [
        r
        for r in client.search_runs([experiment.experiment_id], max_results=200)
        if r.data.tags.get("elbi.trial") == "true"
    ]
    assert trials and any(r.data.tags.get("elbi.best_trial") == "true" for r in trials)
    assert all("score_val" in r.data.metrics for r in trials)
    artifacts = {a.path for a in client.list_artifacts(report.run_id)}
    assert "leaderboard.csv" in artifacts
    assert "reference.csv" in artifacts  # drift monitoring works for AG models too

    # the registered pyfunc predictor loads and scores through the registry
    registry = ModelRegistry(uri)
    model = registry.load("ag")
    frame, _ = parse_invocations(
        {
            "dataframe_records": [
                {"x1": 3.0, "x2": 1.0, "segment": "a"},
                {"x1": -3.0, "x2": -1.0, "segment": "b"},
            ]
        },
        schema=model.metadata.get_input_schema(),
    )
    assert [int(v) for v in model.predict(frame)] == [1, 0]

    with pytest.raises(ModelError, match="unknown engine"):
        train_automl(
            _classification_rows(n=40),
            name="bad",
            target="y",
            tracking_uri=uri,
            engine="h2o",
        )
    with pytest.raises(ModelError, match="unknown estimators"):
        train_automl(
            _classification_rows(n=60),
            name="bad",
            target="y",
            tracking_uri=uri,
            engine="autogluon",
            estimator_list=["prophet"],
        )


def test_optuna_engine_tunes_registers_and_serves(tmp_path: Path) -> None:
    # The explicit Bayesian tuner through the same lifecycle: a TPE study over
    # gradient boosting, trials as child runs, a sklearn-served pipeline, the
    # champion policy, and the oracle gate that is the same regardless of engine.
    pytest.importorskip("optuna")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        _classification_rows(),
        name="opt",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=8.0,
        seed=7,
        engine="optuna",
        dataset="demo",
    )
    assert report.metrics["accuracy"] > 0.8
    assert report.oracle_verdict == "sound" and report.champion is True
    assert "Optuna" in report.best_estimator
    # the tuned hyperparameters were actually searched, not left at defaults
    assert {"learning_rate", "max_depth", "n_estimators"} <= set(report.best_config)

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    assert client.get_run(report.run_id).data.params["engine"] == "optuna"
    experiment = client.get_experiment_by_name("opt")
    trials = [
        r
        for r in client.search_runs([experiment.experiment_id], max_results=500)
        if r.data.tags.get("elbi.trial") == "true"
    ]
    assert trials and all("cv_score" in r.data.metrics for r in trials)
    assert any(r.data.tags.get("elbi.best_trial") == "true" for r in trials)

    # the registered sklearn pipeline loads and scores through the registry
    registry = ModelRegistry(uri)
    model = registry.load("opt")
    frame, _ = parse_invocations(
        {
            "dataframe_records": [
                {"x1": 3.0, "x2": 1.0, "segment": "a"},
                {"x1": -3.0, "x2": -1.0, "segment": "b"},
            ]
        }
    )
    assert [int(v) for v in model.predict(frame)] == [1, 0]

    # regression, via an explicit LightGBM choice
    reg = train_automl(
        _regression_rows(),
        name="opt_reg",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=8.0,
        seed=7,
        engine="optuna",
        estimator_list=["lgbm"],
    )
    assert reg.metrics["r2"] > 0.8 and reg.task == "regression"

    # the tuner takes exactly one supported gradient booster
    with pytest.raises(ModelError, match="one gradient booster"):
        train_automl(
            _classification_rows(n=60),
            name="bad",
            target="y",
            tracking_uri=uri,
            engine="optuna",
            estimator_list=["lgbm", "xgboost"],
        )


def test_optuna_engine_does_not_certify_noise(tmp_path: Path) -> None:
    # The oracle gate is engine-independent: it screens the data, not the model, so a
    # tuner that fits noise still must not earn the champion alias.
    pytest.importorskip("optuna")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        _noise_rows(),
        name="opt_noise",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=6.0,
        seed=7,
        engine="optuna",
    )
    assert report.oracle_verdict != "sound"
    assert report.champion is False


def test_ensemble_engine_blends_registers_and_serves(
    tmp_path: Path, monkeypatch
) -> None:
    # The cross-model blend through the same lifecycle: a diverse base set, greedy
    # weighted selection, base models as child runs, and a sklearn-served blend.
    # The optional TabICL member is force-skipped here (cap 0) so the test stays fast
    # and offline; its in-ensemble path is covered by the opt-in TabICL test.
    from elbi_core.ml import ensemble_engine

    monkeypatch.setattr(ensemble_engine, "_TABICL_MAX_ROWS", 0)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        _regression_rows(),
        name="ens",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        time_budget=30.0,
        seed=7,
        engine="ensemble",
        dataset="demo",
    )
    assert report.metrics["r2"] > 0.8
    assert report.oracle_verdict == "sound" and report.champion is True
    assert "ensemble" in report.best_estimator
    # the blend weights are a normalized distribution over the chosen base families
    assert report.best_config and abs(sum(report.best_config.values()) - 1.0) < 1e-6

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    assert client.get_run(report.run_id).data.params["engine"] == "ensemble"
    experiment = client.get_experiment_by_name("ens")
    bases = [
        r
        for r in client.search_runs([experiment.experiment_id], max_results=500)
        if r.data.tags.get("elbi.trial") == "true"
    ]
    assert bases  # each base family logged as a child run with its validation score

    registry = ModelRegistry(uri)
    model = registry.load("ens")
    frame, _ = parse_invocations(
        {"dataframe_records": [{"x1": 2.0, "x2": -1.0}, {"x1": -2.0, "x2": 1.0}]}
    )
    preds = model.predict(frame)
    assert len(preds) == 2 and preds[0] > preds[1]  # y = 3*x1 - 2*x2

    # an unknown base family is refused up front
    with pytest.raises(ModelError, match="unknown base learners"):
        train_automl(
            _regression_rows(n=60),
            name="bad",
            target="y",
            tracking_uri=uri,
            engine="ensemble",
            estimator_list=["prophet"],
        )


def test_tabicl_engine_guards_cleanly(tmp_path: Path, monkeypatch) -> None:
    # The foundation-model engine never crashes raw: forecasting and oversized data
    # both surface as a ModelError with a clear next step, before any weight download.
    from elbi_core.ml import tabicl_engine

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    # A foundation model cannot forecast; the guard fires before any heavy import.
    with pytest.raises(ModelError, match="does not support ts_forecast"):
        train_automl(
            _regression_rows(n=60),
            name="tc_fc",
            target="y",
            tracking_uri=uri,
            engine="tabicl",
            task="ts_forecast",
            time_col="x1",
            horizon=5,
        )

    # TabICL's size ceiling is enforced with guidance rather than a slow crawl.
    monkeypatch.setattr(tabicl_engine, "_MAX_ROWS", 10)
    with pytest.raises(ModelError, match="TabICL handles up to"):
        train_automl(
            _classification_rows(n=60),
            name="tc_big",
            target="y",
            tracking_uri=uri,
            engine="tabicl",
        )


@pytest.mark.skipif(
    not os.environ.get("ELBI_TEST_TABICL"),
    reason="TabICL end-to-end downloads weights and is slow on CPU; "
    "opt in with ELBI_TEST_TABICL=1",
)
def test_tabicl_engine_trains_registers_and_serves(tmp_path: Path) -> None:
    # The open foundation model through the full lifecycle: a single-forward-pass
    # model that registers, loads, and scores like any other engine, and is gated by
    # the same oracle. Opt-in because it downloads pretrained weights on first use.
    pytest.importorskip("tabicl")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    report = train_automl(
        _classification_rows(),
        name="tc",
        target="y",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        seed=7,
        engine="tabicl",
        dataset="demo",
    )
    assert report.metrics["accuracy"] > 0.8
    assert "TabICL" in report.best_estimator
    registry = ModelRegistry(uri)
    model = registry.load("tc")
    frame, _ = parse_invocations(
        {"dataframe_records": [{"x1": 3.0, "x2": 1.0, "segment": "a"}]},
        schema=model.metadata.get_input_schema(),
    )
    assert len(model.predict(frame)) == 1


def test_a_time_ordered_holdout_is_the_future(tmp_path: Path) -> None:
    """A model applied forward must be measured forward.

    A shuffled split lets the search see rows dated after its test rows, which scores a
    time machine rather than a model.
    """
    rows = [{"t": i, "x": float(i % 7), "y": float(i) + (i % 7)} for i in range(120)]
    report = train_automl(
        rows,
        name="ordered",
        target="y",
        time_col="t",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "mlartifacts",
        time_budget=10.0,
        estimator_list=["lgbm"],
        max_iter=2,
        seed=7,
    )
    # The ordering column is an index, not evidence: it must not become a feature.
    assert "t" not in report.features
    assert report.n_train + report.n_test == len(rows)
    assert report.n_test > 0


def test_a_repeated_entity_cannot_straddle_the_holdout(tmp_path: Path) -> None:
    """Scoring a model on entities it trained on is leakage by another name.

    ``groups`` names the entity, as AutoGluon's does, and the column is excluded from
    the features because a model that can see an identifier memorises it.
    """
    import pandas as pd

    from elbi_core.ml.training import _split

    frame = pd.DataFrame(
        [{"house": i // 2, "x": float(i), "y": float(i) * 3} for i in range(60)]
    )
    x_train, x_test, _y_train, _y_test = _split(
        frame, "y", ["x"], "regression", 7, groups="house"
    )
    trained = set(frame.loc[x_train.index, "house"])
    tested = set(frame.loc[x_test.index, "house"])
    assert trained & tested == set(), "an entity was scored against itself"
    assert len(x_test) > 0


def test_a_classifier_is_judged_on_its_own_decisions(tmp_path: Path) -> None:
    """The {target, features} claim asks if the data predicts, not if the model does.

    At a 6% base rate a model that answers "no" to everything scores 94% and finds no
    churner at all -- a fact about the model, invisible to a check over the dataset.
    The classifier gate has to see the holdout predictions to say so, which is how the
    forecast gate already works.
    """
    rng = random.Random(5)
    rows = []
    for _ in range(1200):
        tenure = rng.randint(1, 48)
        tickets = rng.randint(0, 9)
        logit = -3.4 - 0.05 * tenure + 0.32 * tickets
        churned = int(rng.random() < 1 / (1 + math.exp(-logit)))
        rows.append(
            {
                "tenure": str(tenure),
                "tickets": str(tickets),
                "churned": str(churned),
            }
        )
    prevalence = sum(int(r["churned"]) for r in rows) / len(rows)
    assert 0.02 < prevalence < 0.15, prevalence

    report = train_automl(
        rows,
        target="churned",
        features=("tenure", "tickets"),
        task="classification",
        time_budget=5.0,
        name="churn_probe",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=str(tmp_path / "artifacts"),
    )
    # Whatever AutoML settles on, the verdict must be about the model's decisions: the
    # reason names accuracy against the majority baseline or the ranking, not the
    # dataset's predictability.
    assert report.oracle_verdict in ("sound", "inconclusive")
    assert any(
        word in report.oracle_detail.lower()
        for word in ("accuracy", "majority", "ranking", "roc", "class")
    ), report.oracle_detail


def test_a_small_holdout_does_not_count_against_a_model(tmp_path: Path) -> None:
    """Too few rows to judge is not evidence of no skill, and must not block promotion.

    The classifier gate needs 40 rows; a small dataset's holdout has fewer. Reading
    its "inconclusive" as a verdict on the model would refuse to promote a first
    version that is perfectly good -- which is what a separable 160-row set is.
    """
    rng = random.Random(4)
    rows = []
    for _ in range(160):
        x1, x2 = rng.gauss(0, 1), rng.gauss(0, 1)
        label = "1" if x1 + 0.3 * x2 + rng.gauss(0, 0.3) > 0 else "0"
        rows.append({"x1": f"{x1:.4f}", "x2": f"{x2:.4f}", "y": label})

    report = train_automl(
        rows,
        target="y",
        features=("x1", "x2"),
        task="classification",
        time_budget=5.0,
        name="small_holdout",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=str(tmp_path / "artifacts"),
    )
    assert report.oracle_verdict == "sound", report.oracle_detail
    assert report.champion is True, "a sound first version deploys"
