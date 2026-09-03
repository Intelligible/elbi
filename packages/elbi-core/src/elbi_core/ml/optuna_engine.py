"""Optuna as an explicit Bayesian tuner over gradient boosting (engine="optuna").

Where FLAML runs its own budget-aware search and AutoGluon stacks a portfolio, this
engine is the standard modern hyperparameter-optimization loop written out in the
open: a Tree-structured Parzen Estimator (TPE) study searches a gradient-boosting
model's hyperparameters, a median pruner abandons unpromising trials early, and each
trial is scored by cross-validation (never a single split, which would overfit the
tuning to one fold). Every trial is logged as a nested MLflow child run, so the run
table reads as a leaderboard the same way the other engines' do; the best
configuration is refit on the full training split as the model that is registered.

The searched learner is XGBoost by default, or LightGBM via ``estimator_list=["lgbm"]``
-- both come with the ``ml`` extra. Categorical columns are one-hot encoded in a
pipeline so the served model accepts raw feature rows exactly as trained.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import ModelError
from .engine import EngineResult

#: The gradient-boosting learners this engine can tune, by our estimator vocabulary.
_LEARNERS = ("xgboost", "lgbm")

#: Trials to run before the wall-clock budget stops the study, whichever comes first.
#: A ceiling, not a target: the ``time_budget`` is normally what ends the search.
_MAX_TRIALS = 200

#: Cross-validation folds for the trial objective.
_CV_FOLDS = 5


def require_optuna() -> None:
    """Raise :class:`ModelError` unless Optuna is importable."""
    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise ModelError(
            "engine='optuna' requires optuna: pip install 'elbi[ml]'"
        ) from exc


def fit(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    feature_names: list[str],
    time_budget: float,
    metric: str | None,
    seed: int,
    estimator_list: Sequence[str],
) -> EngineResult:
    """Tune a gradient-boosting model with an Optuna TPE study; return the result."""
    require_optuna()
    import optuna

    learner = _resolve_learner(estimator_list)
    is_classification = task == "classification"
    scorer = _Scorer(task, metric, y_train)

    # A label encoder is fitted once over the full training labels so every fold and
    # the final refit share one class ordering; XGBoost/LightGBM want integer classes.
    encoder = _LabelState(y_train) if is_classification else None
    splitter = _splitter(task, y_train, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        params = _suggest(trial, learner)
        scores: list[float] = []
        # StratifiedKFold reads y to balance folds; KFold ignores the second argument.
        for step, (tr, va) in enumerate(splitter.split(x_train, y_train)):
            estimator = _build_estimator(learner, params, x_train, is_classification)
            x_tr, x_va = x_train.iloc[tr], x_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
            estimator.fit(x_tr, _encode(encoder, y_tr))
            scores.append(scorer.score(estimator, x_va, y_va, encoder))
            # Report the running mean so the pruner can stop a clearly poor trial
            # partway through its folds rather than paying for all five.
            trial.report(sum(scores) / len(scores), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(scores) / len(scores)

    study.optimize(
        objective, n_trials=_MAX_TRIALS, timeout=max(1.0, time_budget), n_jobs=1
    )
    if study.best_trial is None:  # pragma: no cover - optimize guarantees one
        raise ModelError("the Optuna study completed no trial within the budget")

    best_params = study.best_params
    model = _OptunaModel(
        _build_estimator(learner, best_params, x_train, is_classification),
        encoder,
    )
    model.fit(x_train, y_train)

    def log_model(
        *, name: str, signature: Any, input_example: Any, pip_requirements: list[str]
    ) -> Any:
        import mlflow

        return mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=name,
            pip_requirements=pip_requirements,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    def log_trials(experiment_id: str, parent_run_id: str) -> None:
        _log_optuna_trials(study, learner, experiment_id, parent_run_id)

    return EngineResult(
        model=model,
        best_estimator=f"{learner} (Optuna TPE, {len(study.trials)} trials)",
        best_config=best_params,
        log_model=log_model,
        training_code=_training_code(learner, best_params, feature_names, task),
        # The tuned learner sits behind a one-hot preprocessor, so its importances
        # index transformed columns, not the raw features; SHAP over the model's own
        # predict is the honest attribution and runs for every engine.
        sklearn_estimator=None,
        log_trials=log_trials,
        search_best_loss=-float(study.best_value),
    )


def _resolve_learner(estimator_list: Sequence[str]) -> str:
    """The gradient booster to tune (default XGBoost); validate an explicit choice."""
    if not estimator_list:
        return "xgboost"
    chosen = list(estimator_list)
    if len(chosen) != 1 or chosen[0] not in _LEARNERS:
        raise ModelError(
            "engine='optuna' tunes one gradient booster; set estimator_list to "
            f"exactly one of {', '.join(_LEARNERS)}"
        )
    return chosen[0]


def _suggest(trial: Any, learner: str) -> dict[str, Any]:
    """One trial's hyperparameters. Ranges that span orders of magnitude are log-scaled.

    The search space is the well-worn one for gradient boosting: tree shape, row and
    column subsampling, and L1/L2 regularization, tuned on a log scale where the useful
    values span several orders of magnitude (a linear range would never find them).
    """
    space = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    }
    if learner == "xgboost":
        space["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        space["gamma"] = trial.suggest_float("gamma", 1e-8, 5.0, log=True)
    else:  # lgbm
        space["num_leaves"] = trial.suggest_int("num_leaves", 15, 255)
        space["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
    return space


def _build_estimator(
    learner: str, params: dict[str, Any], x_train: Any, is_classification: bool
) -> Any:
    """A one-hot-preprocessing pipeline around the tuned gradient booster.

    Object-dtype columns are one-hot encoded (unknown categories at serve time are
    ignored, not an error); numeric columns pass through. The pipeline is what gets
    fitted and served, so it accepts raw feature rows exactly as they were trained.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical = [c for c in x_train.columns if x_train[c].dtype == object]
    numeric = [c for c in x_train.columns if c not in categorical]
    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    booster = _booster(learner, params, is_classification)
    return Pipeline([("prep", prep), ("model", booster)])


def _booster(learner: str, params: dict[str, Any], is_classification: bool) -> Any:
    """Construct the underlying XGBoost/LightGBM estimator for the task."""
    if learner == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if is_classification else XGBRegressor
        return cls(**params, n_jobs=1, verbosity=0, tree_method="hist")
    from lightgbm import LGBMClassifier, LGBMRegressor

    cls = LGBMClassifier if is_classification else LGBMRegressor
    return cls(**params, n_jobs=1, verbose=-1)


class _LabelState:
    """A fitted label encoder shared across folds and the refit (classification)."""

    def __init__(self, y: Any) -> None:
        from sklearn.preprocessing import LabelEncoder

        self.encoder = LabelEncoder().fit(y)

    def encode(self, y: Any) -> Any:
        return self.encoder.transform(y)

    def decode(self, codes: Any) -> Any:
        return self.encoder.inverse_transform(codes)


def _encode(state: _LabelState | None, y: Any) -> Any:
    """Integer-encode labels for the booster; pass regression targets through."""
    return state.encode(y) if state is not None else y


class _OptunaModel:
    """The served model: the fitted pipeline, with label decoding for classification.

    Kept at module scope so cloudpickle can round-trip it wherever the model is loaded
    back. ``predict`` returns labels in the data's own values (not the booster's
    integer codes); ``predict_proba`` returns class probabilities for a classifier.
    """

    def __init__(self, pipeline: Any, label_state: _LabelState | None) -> None:
        self.pipeline = pipeline
        self.label_state = label_state

    def fit(self, x: Any, y: Any) -> _OptunaModel:
        self.pipeline.fit(x, _encode(self.label_state, y))
        return self

    def predict(self, x: Any) -> Any:
        preds = self.pipeline.predict(x)
        if self.label_state is not None:
            return self.label_state.decode(preds)
        return preds

    def predict_proba(self, x: Any) -> Any:
        return self.pipeline.predict_proba(x)

    def score(self, x: Any, y: Any, sample_weight: Any = None) -> float:
        """The task's default score (accuracy or R^2), the sklearn estimator contract.

        Present so a loaded model answers ``score`` like any sklearn estimator, which
        is what MLflow's default evaluator calls for its baseline metric. ``score``
        takes an optional ``sample_weight`` to match the sklearn signature.
        """
        from sklearn.metrics import accuracy_score, r2_score

        pred = self.predict(x)
        if self.label_state is not None:
            return float(accuracy_score(y, pred, sample_weight=sample_weight))
        return float(r2_score(y, pred, sample_weight=sample_weight))


class _Scorer:
    """Higher-is-better score of one fitted model on a validation fold.

    Mirrors the held-out metric vocabulary the trainer reports, always oriented so the
    study maximizes: error metrics are negated. ``metric=None`` picks the task's
    default (accuracy for classification, R^2 for regression).
    """

    def __init__(self, task: str, metric: str | None, y: Any) -> None:
        self.task = task
        self.n_classes = int(y.nunique()) if task == "classification" else 0
        self.metric = metric or ("accuracy" if task == "classification" else "r2")

    def score(
        self, estimator: Any, x_val: Any, y_val: Any, encoder: _LabelState | None
    ) -> float:
        from sklearn import metrics as sk

        if self.task == "regression":
            pred = estimator.predict(x_val)
            if self.metric == "rmse":
                return -float(sk.root_mean_squared_error(y_val, pred))
            if self.metric == "mae":
                return -float(sk.mean_absolute_error(y_val, pred))
            return float(sk.r2_score(y_val, pred))
        # Decode integer class codes back to the data's own labels before scoring.
        pred = _OptunaModel(estimator, encoder).predict(x_val)
        if self.metric in ("roc_auc", "log_loss"):
            proba = estimator.predict_proba(x_val)
            classes = encoder.encoder.classes_ if encoder is not None else None
            y_codes = encoder.encode(y_val) if encoder is not None else y_val
            if self.metric == "log_loss":
                labels = list(range(len(classes))) if classes is not None else None
                return -float(sk.log_loss(y_codes, proba, labels=labels))
            if self.n_classes == 2:
                return float(sk.roc_auc_score(y_codes, proba[:, 1]))
            return float(sk.roc_auc_score(y_codes, proba, multi_class="ovr"))
        if self.metric == "f1":
            average = "binary" if self.n_classes == 2 else "macro"
            return float(sk.f1_score(y_val, pred, average=average))
        return float(sk.accuracy_score(y_val, pred))


def _splitter(task: str, y: Any, seed: int) -> Any:
    """A stratified K-fold splitter for classification, plain K-fold otherwise.

    Folds are clamped down when the data is small or a class is rare, so a stratified
    split never asks for more folds than the smallest class can fill.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if task == "classification":
        folds = max(2, min(_CV_FOLDS, int(y.value_counts().min())))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    folds = max(2, min(_CV_FOLDS, len(y) // 2))
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _log_optuna_trials(
    study: Any, learner: str, experiment_id: str, parent_run_id: str
) -> None:
    """Log each Optuna trial as a nested MLflow child run (the study leaderboard).

    One child per trial with its suggested hyperparameters and cross-validated score,
    the best trial flagged, so the run table reads as a leaderboard. Fail-soft: trial
    bookkeeping must never fail the training that produced a good model.
    """
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    client = MlflowClient()
    best_number = study.best_trial.number
    for trial in study.trials[:_MAX_TRIALS]:
        try:
            state = trial.state.name
            run = client.create_run(
                experiment_id,
                run_name=f"trial-{trial.number:03d}-{learner}",
                tags={
                    "mlflow.parentRunId": parent_run_id,
                    "elbi.trial": "true",
                    "elbi.best_trial": str(trial.number == best_number).lower(),
                    "elbi.trial_state": state,
                },
            )
            params = [Param("learner", learner)] + [
                Param(str(k), str(v)[:500]) for k, v in trial.params.items()
            ]
            metrics = []
            if trial.value is not None:
                metrics.append(
                    Metric("cv_score", float(trial.value), int(run.info.start_time), 0)
                )
            client.log_batch(run.info.run_id, metrics=metrics, params=params)
            client.set_terminated(run.info.run_id, status="FINISHED")
        except Exception:  # noqa: S112 - one bad trial must not lose the rest
            continue


def _training_code(
    learner: str, params: dict[str, Any], features: list[str], task: str
) -> str:
    """A standalone script reproducing the tuned configuration, as editable code."""
    imports = (
        "from xgboost import XGBClassifier, XGBRegressor"
        if learner == "xgboost"
        else "from lightgbm import LGBMClassifier, LGBMRegressor"
    )
    cls = (
        ("XGBClassifier" if learner == "xgboost" else "LGBMClassifier")
        if task == "classification"
        else ("XGBRegressor" if learner == "xgboost" else "LGBMRegressor")
    )
    rendered = ",\n    ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    metric_line = (
        "from sklearn.metrics import accuracy_score as score"
        if task == "classification"
        else "from sklearn.metrics import r2_score as score"
    )
    return f'''"""Best configuration from the Optuna TPE study, as editable code.

Optuna tuned {learner} by cross-validated Bayesian search; the hyperparameters below
are the best trial's. Object (categorical) feature columns should be one-hot encoded
before fitting, as the trained pipeline does.
"""

import pandas as pd
{imports}
from sklearn.model_selection import train_test_split
{metric_line}

data = pd.read_csv("training_data.csv")  # your dataset here
features = {features!r}
target = "TARGET"  # your target column

X_train, X_test, y_train, y_test = train_test_split(
    pd.get_dummies(data[features]), data[target], test_size=0.2, random_state=7
)
model = {cls}(
    {rendered}
)
model.fit(X_train, y_train)
print("held-out score:", score(y_test, model.predict(X_test)))
'''
