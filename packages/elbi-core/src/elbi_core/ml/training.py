"""AutoML training recorded in MLflow: the standard loop, implemented exactly.

The trainer is deliberately unoriginal. FLAML (budget-aware AutoML over the standard
tabular learners) selects the model and its hyperparameters within a wall-clock
budget; MLflow records the run (params, metrics, the FLAML search log); the fitted
model is logged with a signature and an input example and registered in the MLflow
Model Registry; promotion is an alias (``champion``), MLflow 3's replacement for
registry stages. What elbi adds is its own gate: the same verification oracle
that certifies a ``derive`` conclusion first checks that leakage-free held-out skill
exists in the data at all, and only a sound signal earns the champion alias
automatically. Every metric reported to the caller is computed on a held-out split
the search never saw, never the search's own validation score.

This module needs the ``ml`` extra (``pip install 'elbi[ml]'``); every import
of the stack is deferred to call time so the core package stays import-light.
"""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ModelError
from ..verification import verify_all, verify_classification
from ..versioning import hash_json
from .engine import EngineResult

#: Pinned per-model dependencies recorded with every logged model, so a served model
#: reinstalls the exact stack that trained it (MLflow's documented best practice over
#: letting requirements be inferred). The list spans every engine; a distribution that
#: is not installed is skipped, so a FLAML model carries no torch pin it never used.
_PIN_DISTRIBUTIONS = (
    "mlflow",
    "flaml",
    "scikit-learn",
    "pandas",
    "numpy",
    "lightgbm",
    "xgboost",
    "autogluon.tabular",
    "catboost",
    "optuna",
    "tabicl",
    "torch",
)

#: The AutoML engines. ``flaml`` (budget-aware, the default) and ``autogluon``
#: (accuracy-first stack ensembling) select a model and tune it; ``optuna`` is an
#: explicit Bayesian tuner over gradient boosting; ``tabicl`` is an open, pretrained
#: tabular foundation model that predicts in one forward pass with no per-dataset
#: training; ``ensemble`` blends a diverse base set by greedy weighted selection.
#: Only ``flaml`` supports ``ts_forecast``.
ENGINES = ("flaml", "autogluon", "optuna", "tabicl", "ensemble")

#: A numeric target with at most this many distinct values is treated as a class
#: label under ``task="auto"`` (a 0/1 outcome or a small ordinal code, not a measure).
_AUTO_CLASSIFICATION_MAX_LEVELS = 10

#: Refuse classification beyond this many classes: it is almost always a continuous
#: or identifier column picked by mistake, and every trial would crawl through a
#: giant softmax until the budget dies with no model.
_MAX_CLASSIFICATION_LEVELS = 100

#: Rows of training features kept as the drift-monitoring reference sample.
_REFERENCE_ROWS = 2000

#: The forecast gate scores nothing shorter than this window; a holdout tail is
#: stretched to it (series permitting) so short-horizon models still get judged.
_FORECAST_GATE_MIN = 20

#: Held-out share for the final evaluation split. FLAML does its own validation
#: inside the training split; this split exists so reported metrics come from data
#: no part of the search ever saw.
_HOLDOUT_SHARE = 0.2


@contextmanager
def scoped_tracking(uri: str) -> Iterator[None]:
    """Point MLflow's fluent API at ``uri``, restoring the prior state afterwards.

    ``mlflow.set_tracking_uri`` writes both a module global and the
    ``MLFLOW_TRACKING_URI`` environment variable; leaking either would silently
    redirect every later MLflow caller in the process (the app resolves the env
    variable ahead of its own setting), so both are put back on exit.
    """
    import mlflow

    prior_env = os.environ.get("MLFLOW_TRACKING_URI")
    prior_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(prior_uri)
        if prior_env is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = prior_env


def require_ml() -> None:
    """Raise :class:`ModelError` unless the optional ML stack is importable."""
    try:
        import flaml.automl  # noqa: F401
        import mlflow  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ModelError(
            "model training requires the 'ml' extra: pip install 'elbi[ml]'"
        ) from exc


@dataclass(frozen=True)
class TrainingReport:
    """The durable record of one AutoML training run.

    ``metrics`` are held-out (computed on the split the search never saw);
    ``oracle_verdict`` is the verification oracle's independent judgement that
    leakage-free predictive skill exists in the data, the same ``prediction`` gate a
    ``derive`` with a ``{target, features}`` claim runs. ``champion`` records whether
    this version was auto-promoted (only a first, sound version is).
    """

    name: str
    version: int
    run_id: str
    model_uri: str
    task: str
    target: str
    features: tuple[str, ...]
    best_estimator: str
    best_config: dict[str, Any]
    metric_optimized: str
    metrics: dict[str, float]
    n_train: int
    n_test: int
    time_budget: float
    oracle_verdict: str
    oracle_detail: str
    champion: bool

    def render(self) -> str:
        """The report as compact markdown, for the chat and MCP surfaces."""
        metrics = "\n".join(
            f"| {key} | {value:.4f} |" for key, value in sorted(self.metrics.items())
        )
        promoted = (
            "promoted to `@champion` (first sound version)"
            if self.champion
            else "not promoted; use promote_model to make it the champion"
        )
        return (
            f"Trained and registered **{self.name}** version {self.version} "
            f"({self.task}, target `{self.target}`).\n"
            f"- best model: {self.best_estimator} "
            f"(AutoML over a {self.time_budget:.0f}s budget, "
            f"optimizing {self.metric_optimized})\n"
            f"- oracle signal check: {self.oracle_verdict} ({self.oracle_detail})\n"
            f"- registry: {promoted}\n\n"
            f"Held-out metrics ({self.n_test} rows the search never saw):\n\n"
            f"| metric | value |\n| --- | --- |\n{metrics}"
        )


def train_automl(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    target: str,
    features: Sequence[str] = (),
    task: str = "auto",
    time_budget: float = 60.0,
    metric: str | None = None,
    tracking_uri: str,
    artifact_root: str | Path | None = None,
    seed: int = 7,
    estimator_list: Sequence[str] = (),
    max_iter: int | None = None,
    ensemble: bool = False,
    dataset: str | None = None,
    dataset_kind: str = "dataset",
    time_col: str | None = None,
    horizon: int | None = None,
    groups: str | None = None,
    engine: str = "flaml",
) -> TrainingReport:
    """Run the full train-track-register loop over ``rows`` and return its report.

    ``rows`` are the dataset's records (string-valued columns are coerced to numbers
    where every value parses). ``features`` defaults to every column except the
    target. ``task`` is ``classification``, ``regression``, or ``auto`` (numeric
    targets with many levels regress, everything else classifies). ``metric`` is any
    FLAML metric name; the default lets FLAML pick the task's standard one.
    ``estimator_list`` and ``max_iter`` bound the search (``max_iter`` makes a run
    reproducible where a wall-clock budget is not); both default to FLAML's own
    behavior. ``ensemble`` asks FLAML for a stacked ensemble over the searched
    learners as the final model. The registered model lands in the registry behind
    ``tracking_uri``; ``artifact_root`` anchors new experiments' artifacts (kept
    out of the working directory, where MLflow would otherwise put them).

    ``engine`` picks the AutoML engine. ``flaml`` (the default) is budget-aware,
    light, and fast enough for an interactive turn; ``autogluon`` is the
    accuracy-first engine (multi-layer stack ensembling; ``ensemble`` maps to its
    ``best`` preset); ``optuna`` is an explicit Bayesian tuner (a TPE study with
    median pruning) over gradient boosting; ``tabicl`` is an open, pretrained
    tabular foundation model that predicts in a single forward pass with no
    per-dataset training (best on small-to-mid data, no ``ts_forecast``); and
    ``ensemble`` blends a diverse base set by greedy weighted selection. Everything
    around the engine (the oracle gate, registry, lineage, evidence artifacts) is
    engine-neutral.

    ``task="ts_forecast"`` trains a forecaster instead: ``time_col`` names the
    timestamp column and ``horizon`` how many trailing periods to hold out and
    predict ahead; the split is temporal (the last ``horizon`` rows, never a
    shuffle), the reported metrics are forecast errors on that tail, and the
    oracle's check is its forecast gate over the held-out tail rather than the
    tabular prediction gate.

    ``dataset`` is the training source's name and ``dataset_kind`` what it is: a
    bound ``dataset``, or a certified feature ``derivation`` (the platform's
    feature pipeline). Both are recorded (with a content hash of ``rows``) as run
    and version tags, so the model's lineage back to the exact feature code and
    data that trained it is queryable rather than tribal knowledge; a derivation
    source also tags ``elbi.feature_derivation``, which is what
    the skew-proof raw scoring path resolves. Every trial
    of the search is also logged as a nested child run, which is what makes the
    MLflow UI's run table a leaderboard instead of a single opaque row.
    """
    require_ml()
    import mlflow
    import pandas as pd
    from mlflow.models import infer_signature

    if engine not in ENGINES:
        raise ModelError(
            f"unknown engine {engine!r}: expected one of {', '.join(ENGINES)}"
        )
    if task == "ts_forecast" and engine != "flaml":
        raise ModelError(
            f"engine={engine!r} does not support ts_forecast; "
            "forecasting uses the flaml engine"
        )

    frame = _coerce_numeric(pd.DataFrame.from_records(list(rows)))
    if target not in frame.columns:
        raise ModelError(f"target column {target!r} is not in the data")
    feature_names = list(features) or [
        c for c in frame.columns if c not in (target, time_col, groups)
    ]
    if groups:
        if groups not in frame.columns:
            raise ModelError(f"groups column {groups!r} is not in the data")
        # An entity key is an identifier, and a model that can see one memorises it.
        feature_names = [c for c in feature_names if c != groups]
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ModelError(f"feature columns not in the data: {', '.join(missing)}")
    if target in feature_names:
        raise ModelError(f"target {target!r} cannot also be a feature")

    # A row without a usable target teaches nothing and breaks several learners;
    # drop them up front (an empty string is missing too, the way text-loaded
    # data spells null), and refuse when nothing (or almost nothing) remains
    # rather than searching over garbage until the budget expires.
    if frame[target].dtype == object:
        frame[target] = frame[target].replace("", None)
    usable = frame.dropna(subset=[target])
    n_dropped = len(frame) - len(usable)
    if len(usable) < 20:
        raise ModelError(
            f"target {target!r} has only {len(usable)} usable rows"
            + (f" ({n_dropped} rows have no target value)" if n_dropped else "")
            + "; training needs at least 20"
        )
    frame = usable

    # The content identity of the training data, tagged on the run and the
    # registered version: lineage back to the exact rows, not just a dataset name.
    data_hash = hash_json([dict(r) for r in rows])

    resolved_task = _resolve_task(task, frame[target])
    if resolved_task == "classification":
        levels = int(frame[target].nunique())
        if levels < 2:
            raise ModelError(
                f"target {target!r} has a single value; there is nothing to learn"
            )
        if levels > _MAX_CLASSIFICATION_LEVELS:
            raise ModelError(
                f"target {target!r} has {levels} distinct values, which is not a "
                "classification problem; use task='regression' for a continuous "
                "target, or pick a categorical target"
            )
    fit_period: int | None = None
    if resolved_task == "ts_forecast":
        frame, split_horizon, fit_period = _forecast_frame(
            frame, target, time_col, horizon
        )
        train_df = frame.iloc[:-split_horizon]
        test_df = frame.iloc[-split_horizon:]
        x_train = train_df.drop(columns=[target])
        x_test = test_df.drop(columns=[target])
        y_train = train_df[target]
        y_test = test_df[target]
    else:
        x_train, x_test, y_train, y_test = _split(
            frame, target, feature_names, resolved_task, seed, time_col, groups
        )

    # The oracle's independent signal check. For tabular tasks it is the same
    # prediction gate `derive` runs on a {target, features} claim, over the
    # numeric feature subset; for a forecaster the check is deferred to the
    # forecast gate over the held-out tail (computed after the fit, below). It
    # does not veto the training; it decides auto-promotion.
    if resolved_task == "ts_forecast":
        oracle_verdict, oracle_detail = "inconclusive", "pending forecast gate"
    else:
        oracle_verdict, oracle_detail = _oracle_signal(
            list(rows), frame, target, feature_names
        )

    with tempfile.TemporaryDirectory() as scratch:
        result = _fit_engine(
            engine,
            task=resolved_task,
            x_train=x_train,
            y_train=y_train,
            target=target,
            feature_names=feature_names,
            time_budget=time_budget,
            metric=metric,
            seed=seed,
            estimator_list=estimator_list,
            max_iter=max_iter,
            ensemble=ensemble,
            fit_period=fit_period if resolved_task == "ts_forecast" else None,
            scratch=Path(scratch),
        )
        predictions = result.model.predict(x_test)
        holdout = _holdout_metrics(
            resolved_task, result.model, x_test, y_test, predictions
        )
        if resolved_task == "ts_forecast":
            oracle_verdict, oracle_detail = _forecast_oracle(y_test, predictions)
        elif resolved_task == "classification":
            judged = _classification_oracle(result.model, x_test, y_test, predictions)
            if judged is not None:
                oracle_verdict, oracle_detail = judged

        with scoped_tracking(tracking_uri):
            experiment_id = _experiment_id(name, artifact_root)
            with mlflow.start_run(experiment_id=experiment_id) as active:
                mlflow.log_params(
                    {
                        "task": resolved_task,
                        "target": target,
                        "features": ",".join(feature_names),
                        "time_budget": time_budget,
                        "metric": metric or "auto",
                        "seed": seed,
                        "ensemble": ensemble,
                        "engine": engine,
                        "best_estimator": result.best_estimator,
                        "best_config": repr(result.best_config),
                        "n_train": len(x_train),
                        "n_test": len(x_test),
                    }
                )
                mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items()})
                if result.search_best_loss is not None:
                    mlflow.log_metric("search_best_loss", result.search_best_loss)
                tags = {
                    "elbi.oracle_verdict": oracle_verdict,
                    "elbi.oracle_detail": oracle_detail[:250],
                    "elbi.task": resolved_task,
                    "elbi.data_hash": data_hash,
                    "elbi.source_kind": dataset_kind,
                    "mlflow.source.name": "elbi",
                }
                if dataset:
                    tags["elbi.dataset"] = dataset
                    if dataset_kind == "derivation":
                        tags["elbi.feature_derivation"] = dataset
                mlflow.set_tags(tags)
                if result.log_trials is not None:
                    result.log_trials(experiment_id, active.info.run_id)
                # Signature and input example are logged per MLflow's serving best
                # practice; requirements are pinned, not inferred, so a served model
                # reinstalls the exact stack that trained it. The flavor is the
                # engine's own (a native sklearn flavor, or a pyfunc wrapper for an
                # engine MLflow has no flavor for).
                with warnings.catch_warnings():
                    # MLflow warns that untyped predict() hints disable hint-based
                    # validation; the explicit signature above is what validates
                    # serving input, so the warning is only noise.
                    warnings.filterwarnings("ignore", message=".*Any type hint.*")
                    info = result.log_model(
                        name=name,
                        signature=infer_signature(x_test, predictions),
                        input_example=x_test.head(3),
                        pip_requirements=_pip_requirements(),
                    )
                # Evaluation and explainability artifacts (per-class metrics,
                # ROC/confusion plots, attributions) ride on the same run, the
                # way a managed AutoML run arrives with its evidence attached,
                # plus the glass-box pieces: a data-exploration profile and a
                # standalone script reproducing the winning configuration.
                _log_evaluation(info.model_uri, x_test, y_test, target, resolved_task)
                _log_explanations(
                    result.model, result.sklearn_estimator, x_train, x_test
                )
                try:
                    # The training-feature sample every later drift check
                    # compares production traffic against.
                    mlflow.log_text(
                        x_train.sample(
                            min(len(x_train), _REFERENCE_ROWS), random_state=0
                        ).to_csv(index=False),
                        "reference.csv",
                    )
                    mlflow.log_text(
                        _data_profile([dict(r) for r in rows], feature_names, target),
                        "data_profile.md",
                    )
                    mlflow.log_text(result.training_code, "training_code.py")
                except Exception as exc:  # glass-box evidence is best-effort
                    mlflow.set_tag("elbi.glass_box", f"skipped: {exc}"[:250])
                run_id = active.info.run_id

    version = int(info.registered_model_version)
    version_tags = {
        "elbi.oracle_verdict": oracle_verdict,
        "elbi.oracle_detail": oracle_detail[:250],
        "elbi.data_hash": data_hash,
        "elbi.source_kind": dataset_kind,
    }
    if dataset:
        version_tags["elbi.dataset"] = dataset
        if dataset_kind == "derivation":
            version_tags["elbi.feature_derivation"] = dataset
    _tag_version(tracking_uri, name, version, version_tags)
    champion = _promote_first_sound(tracking_uri, name, version, oracle_verdict)
    report = TrainingReport(
        name=name,
        version=version,
        run_id=run_id,
        model_uri=f"models:/{name}/{version}",
        task=resolved_task,
        target=target,
        features=tuple(feature_names),
        best_estimator=result.best_estimator,
        best_config=result.best_config,
        metric_optimized=metric or "auto",
        metrics=holdout,
        n_train=len(x_train),
        n_test=len(x_test),
        time_budget=time_budget,
        oracle_verdict=oracle_verdict,
        oracle_detail=oracle_detail,
        champion=champion,
    )
    _publish_model_card(tracking_uri, report, dataset, data_hash, dataset_kind)
    return report


def _fit_engine(
    engine: str,
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    target: str,
    feature_names: list[str],
    time_budget: float,
    metric: str | None,
    seed: int,
    estimator_list: Sequence[str],
    max_iter: int | None,
    ensemble: bool,
    fit_period: int | None,
    scratch: Path,
) -> EngineResult:
    """Fit the chosen engine and return its engine-neutral result.

    Each engine module owns how it fits, logs, and reproduces its model; this only
    routes to the right one. The engine modules are imported at call time so an
    engine's optional stack (AutoGluon, Optuna, torch, jax) is never imported for a
    run that does not use it.
    """
    if engine == "flaml":
        return _fit_flaml(
            task=task,
            x_train=x_train,
            y_train=y_train,
            target=target,
            feature_names=feature_names,
            time_budget=time_budget,
            metric=metric,
            seed=seed,
            estimator_list=estimator_list,
            max_iter=max_iter,
            ensemble=ensemble,
            fit_period=fit_period,
            scratch=scratch,
        )
    if engine == "autogluon":
        from .autogluon_engine import fit as fit_autogluon

        return fit_autogluon(
            x_train=x_train,
            y_train=y_train,
            target=target,
            task=task,
            time_budget=time_budget,
            metric=metric,
            ensemble=ensemble,
            estimator_list=estimator_list,
            scratch=scratch,
        )
    if engine == "optuna":
        from .optuna_engine import fit as fit_optuna

        return fit_optuna(
            task=task,
            x_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            time_budget=time_budget,
            metric=metric,
            seed=seed,
            estimator_list=estimator_list,
        )
    if engine == "tabicl":
        from .tabicl_engine import fit as fit_tabicl

        return fit_tabicl(task=task, x_train=x_train, y_train=y_train, seed=seed)
    if engine == "ensemble":
        from .ensemble_engine import fit as fit_ensemble

        return fit_ensemble(
            task=task,
            x_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            time_budget=time_budget,
            metric=metric,
            seed=seed,
            estimator_list=estimator_list,
        )
    raise ModelError(f"unknown engine {engine!r}")


def _fit_flaml(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    target: str,
    feature_names: list[str],
    time_budget: float,
    metric: str | None,
    seed: int,
    estimator_list: Sequence[str],
    max_iter: int | None,
    ensemble: bool,
    fit_period: int | None,
    scratch: Path,
) -> EngineResult:
    """The default engine: FLAML's budget-aware AutoML over the tabular learners.

    The whole fitted ``AutoML`` object is the model: it carries FLAML's data
    transformer, so the served model accepts raw feature rows exactly as trained,
    logged via the native sklearn flavor. Cloudpickle (not MLflow's skops default)
    because the AutoML wrapper is not a plain sklearn estimator; the model is only
    ever loaded back into this same pinned environment.
    """
    import mlflow
    import pandas as pd
    from flaml.automl import AutoML

    search_log = scratch / "flaml.log"
    automl = AutoML()
    fit_kwargs: dict[str, Any] = {
        "task": task,
        "time_budget": time_budget,
        "metric": metric or "auto",
        "seed": seed,
        "log_file_name": str(search_log),
        # MLflow logging is done explicitly by the trainer, so the record is one
        # run with a registered model rather than FLAML's own run-per-trial layout.
        "mlflow_logging": False,
        "verbose": 0,
    }
    if estimator_list:
        fit_kwargs["estimator_list"] = list(estimator_list)
    elif task == "ts_forecast":
        # Restrict the default forecast search to always-available learners;
        # FLAML's full list includes estimators behind optional packages.
        fit_kwargs["estimator_list"] = ["lgbm", "xgboost", "rf", "extra_tree"]
    if max_iter is not None:
        fit_kwargs["max_iter"] = max_iter
    if ensemble:
        fit_kwargs["ensemble"] = True
    if task == "ts_forecast":
        fit_kwargs["period"] = fit_period
        automl.fit(
            dataframe=pd.concat([x_train, y_train], axis=1), label=target, **fit_kwargs
        )
    else:
        automl.fit(X_train=x_train, y_train=y_train, **fit_kwargs)
    if automl.model is None:
        raise ModelError(
            f"AutoML completed no trial within the {time_budget:.0f}s budget on "
            f"{len(x_train)} rows x {len(feature_names)} features. Either the first "
            "trial needs longer than the budget on data this size (raise "
            "time_budget), or every trial errored (check the target's values and "
            "types, or drop unusual feature columns)"
        )

    def log_model(
        *, name: str, signature: Any, input_example: Any, pip_requirements: list[str]
    ) -> Any:
        return mlflow.sklearn.log_model(
            automl,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=name,
            pip_requirements=pip_requirements,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    def log_trials(experiment_id: str, parent_run_id: str) -> None:
        if search_log.exists():
            mlflow.log_artifact(str(search_log))
            _log_trials(search_log, experiment_id, parent_run_id)

    return EngineResult(
        model=automl,
        best_estimator=str(automl.best_estimator),
        best_config=dict(automl.best_config),
        log_model=log_model,
        training_code=_training_code(automl, task, target, feature_names),
        sklearn_estimator=getattr(automl.model, "estimator", None),
        log_trials=log_trials,
        search_best_loss=float(automl.best_loss),
    )


def _publish_model_card(
    tracking_uri: str,
    report: TrainingReport,
    dataset: str | None,
    data_hash: str,
    dataset_kind: str = "dataset",
) -> None:
    """Write the version's model card: an artifact, and the version description.

    The card is the standard documentation a model is expected to carry in 2026:
    what it predicts, what trained it, how it scored on held-out data, and what
    the verification oracle concluded, with the platform's honest limitations
    spelled out. Setting it as the registry description makes it readable in the
    MLflow UI's version page without opening artifacts. Fail-soft: documentation
    must never fail the training it documents.
    """
    from contextlib import suppress

    from mlflow import MlflowClient

    metrics = "\n".join(
        f"| {key} | {value:.4f} |" for key, value in sorted(report.metrics.items())
    )
    if dataset and dataset_kind == "derivation":
        source = f"the certified feature derivation `{dataset}`"
    elif dataset:
        source = f"`{dataset}`"
    else:
        source = "the provided rows"
    promoted = (
        "This version was auto-promoted to the champion alias."
        if report.champion
        else "This version was not auto-promoted."
    )
    card = f"""# Model card: {report.name} v{report.version}

## Overview

{report.task.capitalize()} model predicting `{report.target}` from
{len(report.features)} features, selected by AutoML (best learner:
{report.best_estimator}) within a {report.time_budget:.0f}s search budget.

## Training data

Trained on {source}: {report.n_train} training rows, {report.n_test} held-out
rows. Data content hash `{data_hash[:16]}...` (the exact rows are recoverable
from lineage, not just the dataset's name).

## Held-out metrics

| metric | value |
| --- | --- |
{metrics}

## Verification

The oracle's independent signal check (leakage-screened held-out skill on the
numeric features): **{report.oracle_verdict}** ({report.oracle_detail}).
{promoted}

## Intended use and limitations

Scores records shaped like the training columns via the MLflow scoring
protocol. Metrics were estimated on this data's distribution; they need not
transfer to a different population or time. Retrain or re-evaluate before
relying on it over drifted data.
"""
    with suppress(Exception):
        client = MlflowClient(tracking_uri=tracking_uri)
        client.log_text(report.run_id, card, "model_card.md")
        client.update_model_version(report.name, str(report.version), card[:4900])


def _oracle_signal(
    rows: list[Mapping[str, Any]], frame: Any, target: str, features: list[str]
) -> tuple[str, str]:
    """The prediction gate's own verdict and detail for this training's claim.

    The composite verdict would read sound when the gate merely did not apply (only
    the screens ran), so promotion keys on the gate itself. When no numeric feature
    exists for it, the verdict is ``inconclusive`` with the reason, never a silent
    pass.
    """
    import pandas as pd

    numeric = [f for f in features if pd.api.types.is_numeric_dtype(frame[f])]
    if not numeric:
        return (
            "inconclusive",
            "the oracle's prediction gate needs at least one numeric feature",
        )
    report = verify_all([dict(r) for r in rows], target=target, features=numeric)
    for gate in report.ran:
        if gate.name == "prediction":
            return gate.verdict, gate.detail
    reasons = dict(report.skipped)
    return "inconclusive", reasons.get(
        "prediction", "the oracle's prediction gate did not apply"
    )


def _coerce_numeric(frame: Any) -> Any:
    """Convert columns whose every non-null value parses as a number.

    Datasets arrive string-valued (CSV, SQL text); a column is numeric only when
    nothing is lost by the conversion, so mixed or categorical columns stay strings
    for FLAML's own categorical handling.
    """
    import pandas as pd

    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted
    return frame


def _resolve_task(task: str, target: Any) -> str:
    """Resolve ``auto`` to a concrete FLAML task from the target's shape."""
    import pandas as pd

    if task in ("classification", "regression", "ts_forecast"):
        return task
    if task != "auto":
        raise ModelError(
            f"unsupported task {task!r}: expected classification, regression, "
            "ts_forecast, or auto"
        )
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    if target.nunique() <= _AUTO_CLASSIFICATION_MAX_LEVELS:
        return "classification"
    return "regression"


def _split(
    frame: Any,
    target: str,
    features: list[str],
    task: str,
    seed: int,
    time_col: str | None = None,
    groups: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    """A held-out split for the final evaluation, honouring time and group structure.

    A shuffled split is only valid when the rows are exchangeable, and the two ways they
    commonly are not each have a standard answer:

    ``time_col``
        The model will predict the future, so the holdout has to *be* the future.
        Sorting and taking the tail is what ``TimeSeriesSplit`` does at its last fold,
        and it is the only honest measurement of a model applied forward.

    ``groups``
        Repeated observations of one entity (a house sold twice, a patient seen twice)
        must not straddle the split, or the model is scored on entities it trained on.
        ``GroupShuffleSplit`` keeps each group whole. The column names the entity and is
        excluded from the features, AutoGluon's ``groups`` convention.

    Both together sort by time and cut at the first boundary keeping groups whole, since
    no primitive does both: scikit-learn's ``TimeSeriesSplit`` ignores ``groups``.

    Stratification applies only to a plain shuffled classification split; it is
    meaningless once the holdout is defined by time.
    """
    x, y = frame[features], frame[target]
    if time_col:
        ordered = frame.sort_values(time_col, kind="stable")
        cut = max(1, int(len(ordered) * (1 - _HOLDOUT_SHARE)))
        if groups:
            # Walk the boundary forward while it would split one entity in two.
            keys = ordered[groups].to_numpy()
            while cut < len(keys) and keys[cut] == keys[cut - 1]:
                cut += 1
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
        if test.empty:  # every row in one group, or one timestamp: nothing to hold out
            raise ModelError(
                f"a time-ordered holdout on {time_col!r} leaves no test rows; the data "
                "may span a single period"
            )
        return train[features], test[features], train[target], test[target]
    if groups:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=_HOLDOUT_SHARE, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(x, y, groups=frame[groups]))
        return x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    from sklearn.model_selection import train_test_split

    # Stratify a classification split so rare classes appear on both sides; fall back
    # when a class is too small to split at all.
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    split: tuple[Any, Any, Any, Any] = train_test_split(
        x, y, test_size=_HOLDOUT_SHARE, random_state=seed, stratify=stratify
    )
    return split


def _forecast_frame(
    frame: Any, target: str, time_col: str | None, horizon: int | None
) -> tuple[Any, int, int]:
    """Validate and order a forecasting frame; return (frame, holdout, period).

    The frame comes back sorted by the (parsed) time column with the timestamp
    first, the way FLAML's forecaster expects it. The holdout tail is at least
    the oracle's minimum scoring window (when the series affords it) even for a
    short deployment horizon, so the forecast gate can actually judge skill; the
    fit period stays the caller's horizon.
    """
    import pandas as pd

    if not time_col or horizon is None:
        raise ModelError(
            "task='ts_forecast' needs `time_col` (the timestamp column) and "
            "`horizon` (periods to forecast ahead)"
        )
    if time_col not in frame.columns:
        raise ModelError(f"time column {time_col!r} is not in the data")
    horizon = int(horizon)
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.assign(**{time_col: parsed}).dropna(subset=[time_col])
    frame = frame.sort_values(time_col).reset_index(drop=True)
    if horizon < 1 or horizon * 2 > len(frame):
        raise ModelError(
            f"horizon must be between 1 and half the series length "
            f"({len(frame)} usable rows)"
        )
    holdout = horizon
    if len(frame) >= _FORECAST_GATE_MIN * 2:
        holdout = max(horizon, _FORECAST_GATE_MIN)
    ordered = [
        time_col,
        *[c for c in frame.columns if c != time_col and c != target],
        target,
    ]
    return frame[ordered], holdout, horizon


def _classification_oracle(
    model: Any, x_test: Any, actuals: Any, predictions: Any
) -> tuple[str, str] | None:
    """The oracle's classifier gate over the fitted model's own holdout decisions.

    The {target, features} claim asks whether the *data* is predictable, which a
    classifier can pass while deciding nothing: at a 6% base rate a model that answers
    "no" to everything scores 94% and never finds a positive. That is a fact about the
    model, so it needs the model's own predictions -- the same way the forecast gate
    judges a forecaster on its held-out tail rather than on the series.

    Returns ``None`` when the gate cannot judge -- a multiclass target, or a holdout
    too small for it -- leaving the dataset-level verdict in place. That distinction
    matters: "this model decides nothing" must block promotion, while "there were not
    enough held-out rows to tell" is not evidence against the model and must not.
    Scores come from ``predict_proba`` where the estimator has it: ranking is the half
    of the evidence a threshold cannot spoil.
    """
    labels = {str(v) for v in dict.fromkeys(list(actuals))}
    if labels - {"0", "1", "0.0", "1.0", "True", "False"} or len(labels) < 2:
        return None
    gate_rows: list[dict[str, Any]] = [
        {"y_true": int(float(a)), "y_pred": int(float(p))}
        for a, p in zip(actuals, predictions, strict=True)
    ]
    scored = None
    try:
        proba = model.predict_proba(x_test)
    except (AttributeError, NotImplementedError, ValueError):
        proba = None
    if proba is not None:
        for row, p in zip(gate_rows, proba, strict=True):
            row["y_score"] = float(p[1]) if len(p) > 1 else float(p[0])
        scored = "y_score"
    report = verify_classification(gate_rows, "y_true", y_pred="y_pred", y_score=scored)
    if not report.checks:  # it declined for want of rows, not for want of skill
        return None
    detail = report.pivotal or "; ".join(c.detail for c in report.checks)
    return report.verdict, detail


def _forecast_oracle(actuals: Any, forecasts: Any) -> tuple[str, str]:
    """The oracle's forecast gate over the held-out tail.

    The same check a ``derive`` with a {time, actual, forecast} claim runs: is
    this forecast actually useful against the naive baselines? Its verdict is
    what decides auto-promotion for a forecaster.
    """
    # The gate wants a numeric time index; the tail is already time-ordered, so
    # its position is the index (the timestamps themselves stay on the model).
    gate_rows = [
        {"t": i, "actual": float(a), "forecast": float(f)}
        for i, (a, f) in enumerate(zip(actuals, forecasts, strict=True))
    ]
    report = verify_all(gate_rows, time="t", actual="actual", forecast="forecast")
    for gate in report.ran:
        if gate.name == "forecast":
            return gate.verdict, gate.detail
    reasons = dict(report.skipped)
    return "inconclusive", reasons.get(
        "forecast", "the oracle's forecast gate did not apply"
    )


def _holdout_metrics(
    task: str, automl: Any, x_test: Any, y_test: Any, predictions: Any
) -> dict[str, float]:
    """The task's standard metrics on the held-out split.

    Probability-based metrics (ROC AUC, log loss) are best-effort: a tiny held-out
    split can miss a class entirely, which makes them undefined rather than zero, so
    they are simply omitted in that case.
    """
    from sklearn import metrics as sk

    if task == "ts_forecast":
        forecast_scores: dict[str, float] = {
            "rmse": float(sk.root_mean_squared_error(y_test, predictions)),
            "mae": float(sk.mean_absolute_error(y_test, predictions)),
        }
        actuals = [float(v) for v in y_test]
        if all(abs(v) > 1e-9 for v in actuals):
            errors = [
                abs(float(p) - a) / abs(a)
                for p, a in zip(predictions, actuals, strict=True)
            ]
            forecast_scores["mape"] = float(sum(errors) / len(errors))
        return forecast_scores
    if task == "regression":
        return {
            "r2": float(sk.r2_score(y_test, predictions)),
            "rmse": float(sk.root_mean_squared_error(y_test, predictions)),
            "mae": float(sk.mean_absolute_error(y_test, predictions)),
        }
    classes = sorted(set(y_test))
    average = "binary" if len(classes) == 2 else "macro"
    scores: dict[str, float] = {
        "accuracy": float(sk.accuracy_score(y_test, predictions)),
        "f1": float(
            sk.f1_score(y_test, predictions, average=average, pos_label=classes[-1])
        ),
    }
    try:
        probabilities = automl.predict_proba(x_test)
        scores["log_loss"] = float(sk.log_loss(y_test, probabilities))
        if len(classes) == 2:
            scores["roc_auc"] = float(sk.roc_auc_score(y_test, probabilities[:, 1]))
    except (ValueError, AttributeError, IndexError):
        pass
    return scores


#: Sample caps for explanation artifacts: attribution quality saturates quickly
#: and a permutation explainer over the full data would dwarf the training budget.
_SHAP_BACKGROUND_ROWS = 100
_SHAP_EXPLAIN_ROWS = 200


def _data_profile(rows: list[dict[str, Any]], features: list[str], target: str) -> str:
    """The training data's exploration report, as a markdown artifact.

    The per-column profile (completeness, distinct counts, ranges, top values)
    plus the warnings a reviewer would raise before trusting a model: columns
    with heavy missingness, near-constant columns, and identifier-like columns
    whose cardinality tracks the row count.
    """
    from ..quality import profile_columns

    used = [target, *features]
    profiles = [p.to_dict() for p in profile_columns(rows) if p.name in used]
    lines = [
        "# Data profile",
        "",
        f"{len(rows)} rows; target `{target}`; {len(features)} features.",
        "",
        "| column | type | completeness | distinct | range | top value |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    warnings_out: list[str] = []
    for p in profiles:
        top = p["topValues"][0]["value"] if p["topValues"] else ""
        span = f"{p['minimum']} .. {p['maximum']}" if p["minimum"] is not None else ""
        lines.append(
            f"| {p['name']} | {p['inferredType']} | {p['completeness']:.0%} "
            f"| {p['distinct']} | {span} | {top} |"
        )
        if p["completeness"] < 0.9:
            warnings_out.append(
                f"`{p['name']}` is only {p['completeness']:.0%} complete"
            )
        if p["distinct"] <= 1:
            warnings_out.append(f"`{p['name']}` is constant")
        if p["name"] != target and p["distinct"] >= 0.95 * len(rows) > 20:
            warnings_out.append(
                f"`{p['name']}` looks like an identifier "
                f"({p['distinct']} distinct values)"
            )
    if warnings_out:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings_out]
    return "\n".join(lines)


def _training_code(automl: Any, task: str, target: str, features: list[str]) -> str:
    """A standalone script reproducing the winning model configuration.

    The glass-box artifact: the searched-and-chosen estimator, spelled out as
    plain code a person can read, edit, and rerun, rather than an opaque fitted
    object. Honest about its one gap: FLAML label-encodes categorical columns
    internally, so the script notes that preprocessing where it applies.
    """
    estimator = automl.model.estimator
    cls = type(estimator)
    get_params = getattr(estimator, "get_params", None)
    if get_params is None:
        # A wrapped forecaster has no sklearn params surface; reproduce through
        # FLAML itself, pinned to the winning learner and configuration.
        return (
            '"""Winning configuration from the AutoML forecast search."""\n\n'
            "from flaml.automl import AutoML\n\n"
            "# Fit with FLAML pinned to the winning learner and config:\n"
            f"# learner: {automl.best_estimator}\n"
            f"# config: {automl.best_config!r}\n"
            f"# task: {task!r}; target: {target!r}; features: {features!r}\n"
        )
    params = {
        k: v
        for k, v in get_params().items()
        if v is not None and k not in ("n_jobs", "verbose", "verbosity")
    }
    rendered = ",\n    ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    metric_line = (
        "from sklearn.metrics import accuracy_score as score"
        if task == "classification"
        else "from sklearn.metrics import r2_score as score"
    )
    return f'''"""Winning configuration from the AutoML search, as editable code.

Chosen by FLAML ({automl.best_estimator}); hyperparameters below are the fitted
estimator's own. FLAML integer-codes string/categorical feature columns before
fitting; apply an equivalent encoding if your data has any.
"""

import pandas as pd
from {cls.__module__} import {cls.__name__}
from sklearn.model_selection import train_test_split
{metric_line}

data = pd.read_csv("training_data.csv")  # your dataset here
features = {features!r}
target = {target!r}

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=7
)
model = {cls.__name__}(
    {rendered}
)
model.fit(X_train, y_train)
print("held-out score:", score(y_test, model.predict(X_test)))
'''


def _log_evaluation(
    model_uri: str, x_test: Any, y_test: Any, target: str, task: str
) -> None:
    """Log MLflow's standard evaluation artifacts for the held-out split.

    ``mlflow.models.evaluate`` with the default evaluator produces the evidence a
    reviewer expects next to a model (per-class metrics, ROC and precision-recall
    curves, a confusion matrix) on the active run. Explainability is logged
    separately (:func:`_log_explanations`), so the evaluator's own SHAP pass is
    disabled. Fail-soft: evaluation evidence must never fail a good training run;
    a skip is recorded as a tag instead.
    """
    import mlflow

    data = x_test.copy()
    data[target] = list(y_test)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.models.evaluate(
                model=model_uri,
                data=data,
                targets=target,
                model_type="classifier" if task == "classification" else "regressor",
                evaluator_config={"log_model_explainability": False},
            )
    except Exception as exc:  # evidence is best-effort by design
        mlflow.set_tag("elbi.evaluation", f"skipped: {exc}"[:250])


def _log_explanations(
    model: Any, sklearn_estimator: Any, x_train: Any, x_test: Any
) -> None:
    """Log feature importances and a SHAP beeswarm for the fitted model.

    Importances come from ``sklearn_estimator`` when the engine exposes one with
    them (tree models); engines whose model is not a single such estimator (the
    foundation models, the blended ensemble) simply skip that artifact.
    Attributions come from a model-agnostic permutation explainer over the model's
    own predict, so any internal preprocessing is part of the explained function
    rather than a mismatch. Sampled hard: this is a readable picture, not an audit
    of every row. Fail-soft like the evaluation artifacts.
    """
    import mlflow

    try:
        importances = getattr(sklearn_estimator, "feature_importances_", None)
        if importances is not None:
            names = list(x_train.columns) or [f"f{i}" for i in range(len(importances))]
            ranked = sorted(
                zip(names, (float(v) for v in importances), strict=False),
                key=lambda pair: pair[1],
                reverse=True,
            )
            csv = "feature,importance\n" + "\n".join(
                f"{name},{value}" for name, value in ranked
            )
            mlflow.log_text(csv, "feature_importance.csv")
    except Exception as exc:  # best-effort, like the evaluation
        mlflow.set_tag("elbi.feature_importance", f"skipped: {exc}"[:250])

    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        background = x_train.sample(
            min(len(x_train), _SHAP_BACKGROUND_ROWS), random_state=0
        )
        sample = x_test.sample(min(len(x_test), _SHAP_EXPLAIN_ROWS), random_state=0)
        # The permutation explainer does arithmetic on feature values, which a
        # string category cannot survive; explain an integer-coded view instead
        # and decode back to raw rows before every model call, so the explained
        # function is still exactly the served model.
        encoded_background, encoded_sample, decode = _shap_codec(background, sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.Explainer(
                lambda arr: model.predict(decode(arr)), encoded_background
            )
            values = explainer(encoded_sample)
            values.feature_names = list(sample.columns)
            plt.figure()
            shap.plots.beeswarm(values, show=False)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_beeswarm.png")
            plt.close("all")
    except Exception as exc:  # best-effort, like the evaluation
        mlflow.set_tag("elbi.shap", f"skipped: {exc}"[:250])


def _shap_codec(background: Any, sample: Any) -> tuple[Any, Any, Any]:
    """Integer-code categorical columns for SHAP, with a decoder to raw rows.

    Categories are fitted over both frames so no code is unseen; numeric columns
    pass through. The decoder rebuilds a DataFrame in the original dtypes, which
    is what the model's own preprocessing expects.
    """
    import numpy as np
    import pandas as pd

    columns = list(background.columns)
    categories: dict[str, list[Any]] = {}
    for column in columns:
        if background[column].dtype == object:
            values = pd.concat([background[column], sample[column]])
            categories[column] = sorted(values.dropna().unique().tolist())

    def encode(frame: Any) -> Any:
        parts = []
        for column in columns:
            if column in categories:
                mapping = {v: i for i, v in enumerate(categories[column])}
                parts.append(frame[column].map(mapping).astype(float).to_numpy())
            else:
                parts.append(frame[column].astype(float).to_numpy())
        return np.column_stack(parts)

    def decode(array: Any) -> Any:
        data = {}
        for index, column in enumerate(columns):
            values = array[:, index]
            if column in categories:
                cats = categories[column]
                codes = np.clip(np.round(values).astype(int), 0, len(cats) - 1)
                data[column] = [cats[c] for c in codes]
            else:
                data[column] = values
        return pd.DataFrame(data, columns=columns)

    return encode(background), encode(sample), decode


def _experiment_id(name: str, artifact_root: str | Path | None) -> str:
    """The id of the MLflow experiment ``name``, created under ``artifact_root``.

    An explicit artifact location keeps model artifacts under the project's own
    cache directory instead of an ``mlruns/`` folder in whatever the process's
    working directory happens to be.
    """
    import mlflow

    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    location = None
    if artifact_root is not None:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        location = root.resolve().as_uri()
    return str(mlflow.create_experiment(name, artifact_location=location))


def _tag_version(
    tracking_uri: str, name: str, version: int, tags: Mapping[str, str]
) -> None:
    """Record the oracle's verdict and data lineage on the registered version.

    The training run carries the full record, but the registry is what a person
    browses when choosing a version to promote, so the verdict and provenance
    must be legible there without following the run.
    """
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)


#: Cap on trials logged as child runs; a runaway search should not turn the
#: tracking store into a million-row table.
_MAX_TRIAL_RUNS = 500


def _log_trials(search_log: Path, experiment_id: str, parent_run_id: str) -> None:
    """Log every search trial as a nested child run of the training run.

    This is what makes the MLflow UI's experiment table a leaderboard: each
    trial appears with its learner, config, and validation loss under the parent
    run, exactly the layout an AutoML user expects. Parsed from FLAML's own
    search log so the record is FLAML's, not a reconstruction. Fail-soft: a
    malformed log line (or a struggling tracking store) must never fail the
    training that produced a perfectly good model.
    """
    import json

    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    client = MlflowClient()
    best_id: int | None = None
    trials: list[dict[str, Any]] = []
    for line in search_log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "curr_best_record_id" in record:
            best_id = record["curr_best_record_id"]
        elif "record_id" in record:
            trials.append(record)
    for trial in trials[:_MAX_TRIAL_RUNS]:
        try:
            learner = str(trial.get("learner", "?"))
            run = client.create_run(
                experiment_id,
                run_name=f"trial-{trial['record_id']:03d}-{learner}",
                tags={
                    "mlflow.parentRunId": parent_run_id,
                    "elbi.trial": "true",
                    "elbi.best_trial": str(trial["record_id"] == best_id).lower(),
                },
            )
            timestamp = int(run.info.start_time)
            metrics = [
                Metric(key, float(trial[key]), timestamp, 0)
                for key in ("validation_loss", "trial_time", "wall_clock_time")
            ]
            params = [Param("learner", learner)] + [
                Param(str(k), str(v)[:500])
                for k, v in (trial.get("config") or {}).items()
            ]
            client.log_batch(run.info.run_id, metrics=metrics, params=params)
            client.set_terminated(run.info.run_id, status="FINISHED")
        except Exception:  # noqa: S112 - one bad trial line must not lose the rest
            continue


def _promote_first_sound(
    tracking_uri: str, name: str, version: int, verdict: str
) -> bool:
    """Alias the first sound version as ``champion``; later promotion is explicit.

    Auto-promoting only a *sound* first version keeps the registry's champion
    meaningful: a model registered over data the oracle could not certify predictive
    signal in stays unaliased until a person promotes it deliberately.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    if verdict != "sound":
        return False
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_model_version_by_alias(name, "champion")
        return False  # a champion already exists; replacing it is an explicit act
    except MlflowException:
        client.set_registered_model_alias(name, "champion", str(version))
        return True


def _pip_requirements() -> list[str]:
    """Exact pins of the training stack, recorded with the logged model."""
    pins: list[str] = []
    for distribution in _PIN_DISTRIBUTIONS:
        try:
            pins.append(f"{distribution}=={importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return pins
