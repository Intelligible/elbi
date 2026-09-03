"""AutoGluon as the maximum-accuracy AutoML engine (the ``autogluon`` extra).

FLAML stays the default (budget-aware, light, fast enough for a chat turn);
AutoGluon is the opt-in engine for accuracy-first training, bringing the
multi-layer stack ensembling that tops the tabular benchmarks. The install is
tabular-only with the tree learners, so torch never comes along.

MLflow has no native AutoGluon flavor, so the fitted ``TabularPredictor`` is
logged as a pyfunc model wrapping the predictor's saved directory as artifacts,
the pattern AutoGluon's own deployment guidance uses; the wrapper class lives
at module top level so the pyfunc loader can import it by path wherever the
model is loaded back.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import mlflow

from ..errors import ModelError
from .engine import EngineResult

#: Cap on leaderboard rows logged as child runs; a deep stack should not turn the
#: tracking store into a giant table.
_MAX_TRIAL_RUNS = 500

#: FLAML-style estimator names mapped to AutoGluon model keys, so one
#: ``estimator_list`` vocabulary works for both engines.
_ESTIMATOR_KEYS = {
    "lgbm": "GBM",
    "xgboost": "XGB",
    "rf": "RF",
    "extra_tree": "XT",
    "catboost": "CAT",
}

#: Metric aliases: what our surfaces call it, mapped to AutoGluon's name where
#: they differ. Unmapped names pass through (AutoGluon validates them).
_METRIC_ALIASES = {
    "rmse": "root_mean_squared_error",
    "mape": "mean_absolute_percentage_error",
}


def require_autogluon() -> None:
    """Raise :class:`ModelError` unless the AutoGluon engine is importable."""
    try:
        import autogluon.tabular  # noqa: F401
    except ImportError as exc:
        raise ModelError(
            "engine='autogluon' requires the autogluon extra: "
            "pip install 'elbi[autogluon]'"
        ) from exc


def fit_autogluon(
    train_df: Any,
    *,
    target: str,
    task: str,
    time_budget: float,
    metric: str | None,
    ensemble: bool,
    estimator_list: Sequence[str],
    scratch: Path,
) -> Any:
    """Fit a ``TabularPredictor`` under ``scratch`` and return it.

    ``ensemble`` selects the ``best`` preset (deeper stacking, the engine's
    reason to exist); the default ``medium`` preset trades some accuracy for
    speed. ``estimator_list`` restricts the searched families using the same
    names FLAML uses.
    """
    require_autogluon()
    from autogluon.tabular import TabularPredictor

    if task == "classification":
        problem_type = "binary" if train_df[target].nunique() == 2 else "multiclass"
    else:
        problem_type = "regression"
    fit_kwargs: dict[str, Any] = {
        "time_limit": max(1.0, time_budget),
        "presets": "best" if ensemble else "medium",
    }
    if estimator_list:
        unknown = [e for e in estimator_list if e not in _ESTIMATOR_KEYS]
        if unknown:
            raise ModelError(
                "unknown estimators for the autogluon engine: "
                + ", ".join(unknown)
                + f" (supported: {', '.join(sorted(_ESTIMATOR_KEYS))})"
            )
        fit_kwargs["hyperparameters"] = {_ESTIMATOR_KEYS[e]: {} for e in estimator_list}
    predictor = TabularPredictor(
        label=target,
        path=str(scratch / "autogluon"),
        problem_type=problem_type,
        eval_metric=_METRIC_ALIASES.get(metric, metric) if metric else None,
        verbosity=0,
    )
    try:
        return predictor.fit(train_df, **fit_kwargs)
    except Exception as exc:
        raise ModelError(f"autogluon training failed: {exc}") from exc


class AutoGluonAdapter:
    """The fitted predictor behind the engine-neutral predict surface.

    Training's shared plumbing (held-out metrics, SHAP over the model's own
    predict) talks to ``predict``/``predict_proba`` returning arrays, which is
    what FLAML's object already offers; this gives AutoGluon the same shape.
    """

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def predict(self, data: Any) -> Any:
        """Predicted labels/values as an array."""
        return self.predictor.predict(data).to_numpy()

    def predict_proba(self, data: Any) -> Any:
        """Class probabilities as an array."""
        return self.predictor.predict_proba(data).to_numpy()


class AutoGluonPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """Serve a saved ``TabularPredictor`` from the logged artifacts directory.

    Top-level so the pyfunc loader imports it by path wherever the model loads
    back; the predictor itself loads once per process in ``load_context``.
    """

    def load_context(self, context: Any) -> None:
        """Load the predictor from the artifacts directory, once per process."""
        from autogluon.tabular import TabularPredictor

        self._predictor = TabularPredictor.load(context.artifacts["predictor"])

    def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
        """Score a DataFrame of feature rows."""
        return self._predictor.predict(model_input).to_numpy()


def fit(
    *,
    x_train: Any,
    y_train: Any,
    target: str,
    task: str,
    time_budget: float,
    metric: str | None,
    ensemble: bool,
    estimator_list: Sequence[str],
    scratch: Path,
) -> EngineResult:
    """Fit AutoGluon and return the engine-neutral result the trainer expects.

    The predictor's saved directory (under ``scratch``, alive for the trainer's
    logging) rides as pyfunc artifacts, since MLflow has no native AutoGluon flavor.
    """
    import pandas as pd

    predictor = fit_autogluon(
        pd.concat([x_train, y_train], axis=1),
        target=target,
        task=task,
        time_budget=time_budget,
        metric=metric,
        ensemble=ensemble,
        estimator_list=estimator_list,
        scratch=scratch,
    )

    def log_model(
        *, name: str, signature: Any, input_example: Any, pip_requirements: list[str]
    ) -> Any:
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=AutoGluonPyfunc(),
            artifacts={"predictor": predictor.path},
            signature=signature,
            input_example=input_example,
            registered_model_name=name,
            pip_requirements=pip_requirements,
        )

    def log_trials(experiment_id: str, parent_run_id: str) -> None:
        _log_ag_trials(predictor, experiment_id, parent_run_id)

    return EngineResult(
        model=AutoGluonAdapter(predictor),
        best_estimator=str(predictor.model_best),
        best_config={"presets": "best" if ensemble else "medium"},
        log_model=log_model,
        training_code=_training_code(target, list(x_train.columns), ensemble),
        sklearn_estimator=None,
        log_trials=log_trials,
        search_best_loss=None,
    )


def _log_ag_trials(predictor: Any, experiment_id: str, parent_run_id: str) -> None:
    """Log AutoGluon's leaderboard as nested child runs (the trial leaderboard).

    One child per trained model with its validation score and times, plus the
    leaderboard itself as a CSV artifact on the parent, so the MLflow UI reads
    the same way it does for a FLAML search. Fail-soft: leaderboard evidence must
    never fail a good training run.
    """
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    try:
        board = predictor.leaderboard()
    except Exception:
        return
    client = MlflowClient()
    with suppress(Exception):
        client.log_text(parent_run_id, board.to_csv(index=False), "leaderboard.csv")
    best = getattr(predictor, "model_best", None)
    for _, row in board.head(_MAX_TRIAL_RUNS).iterrows():
        try:
            model_name = str(row["model"])
            run = client.create_run(
                experiment_id,
                run_name=f"trial-{model_name}",
                tags={
                    "mlflow.parentRunId": parent_run_id,
                    "elbi.trial": "true",
                    "elbi.best_trial": str(model_name == best).lower(),
                },
            )
            timestamp = int(run.info.start_time)
            metrics = [
                Metric(key, float(row[column]), timestamp, 0)
                for key, column in (
                    ("score_val", "score_val"),
                    ("fit_time", "fit_time"),
                    ("pred_time_val", "pred_time_val"),
                )
                if row.get(column) == row.get(column)  # drop NaN cells
            ]
            params = [
                Param("learner", model_name),
                Param("stack_level", str(row.get("stack_level", ""))),
            ]
            client.log_batch(run.info.run_id, metrics=metrics, params=params)
            client.set_terminated(run.info.run_id, status="FINISHED")
        except Exception:  # noqa: S112 - one bad row must not lose the rest
            continue


def _training_code(target: str, features: list[str], ensemble: bool) -> str:
    """The glass-box reproduction script for an AutoGluon training."""
    preset = "best" if ensemble else "medium"
    return f'''"""Winning configuration from the AutoML search (autogluon engine).

AutoGluon retrains its full model portfolio and ensembles them; the script
below reproduces the search that selected this model rather than one frozen
estimator, which is how AutoGluon is meant to be rerun.
"""

import pandas as pd
from autogluon.tabular import TabularPredictor

data = pd.read_csv("training_data.csv")  # your dataset here
features = {features!r}
target = {target!r}

train = data[features + [target]]
predictor = TabularPredictor(label=target).fit(
    train, presets={preset!r}, time_limit=600
)
print(predictor.leaderboard())
'''
