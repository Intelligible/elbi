"""Cross-model ensembling by greedy weighted selection (engine="ensemble").

The result that reshaped tabular benchmarking is that no single model family wins:
a diverse blend of gradient-boosted trees, and -- where they help -- a tabular
foundation model, reliably beats the best individual model. This engine builds that
blend the way the benchmarks' own ensembles do, with Caruana ensemble selection: fit a
diverse base set, then greedily add base models (with replacement) to a running
weighted average, each step choosing the one that most improves a held-out score. The
resulting integer counts become the blend weights, so a strong model can be weighted up
and a weak one dropped to zero, with no risk of a bad member dragging the ensemble down.

The base set spans distinct families -- LightGBM, XGBoost, CatBoost, random forest, and
extremely randomized trees -- all installed with the ``ml``/``autogluon`` extras, plus
TabICL (the open tabular foundation model) when its extra is installed and the data is
within its limits. The selection score is computed on a validation split the base models
never trained on; the selected members are then refit on the full data for the ensemble.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..errors import ModelError
from .engine import EngineResult

#: Base learner families, by our estimator vocabulary. TabICL is added separately,
#: only when its optional stack is importable and the data fits its limits.
_FAMILIES = ("lgbm", "xgboost", "catboost", "rf", "extra_tree")

#: Greedy selection rounds. Each round adds one base model (with replacement) to the
#: blend; the prefix with the best validation score wins, so extra rounds cannot hurt.
_SELECTION_ROUNDS = 30

#: Validation share held out of training for the selection score.
_VAL_SHARE = 0.25

#: TabICL's practical ceilings as an ensemble member; above these it is left out of
#: the base set (it trains on the fit split and again on the full data).
_TABICL_MAX_ROWS = 50000
_TABICL_MAX_FEATURES = 1000


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
    """Build a greedy-weighted blend of a diverse base set; return the result."""
    is_classification = task == "classification"
    label_state = _LabelState(y_train) if is_classification else None
    y_codes = (
        label_state.encode(y_train) if label_state is not None else y_train.to_numpy()
    )

    transformer = _fit_transformer(x_train)
    x_all = transformer.transform(x_train)
    x_all = _dense(x_all)

    fit_idx, val_idx = _split_indices(len(x_train), y_codes, task, seed)
    x_fit, x_val = x_all[fit_idx], x_all[val_idx]
    y_fit, y_val = y_codes[fit_idx], y_codes[val_idx]

    families = _resolve_families(estimator_list)
    scorer = _Scorer(task, metric, y_codes)

    # Fit each base model on the fit split and cache its validation predictions once,
    # so greedy selection is pure arithmetic over cached arrays, not repeated fitting.
    val_preds: dict[str, Any] = {}
    val_scores: dict[str, float] = {}
    for key in families:
        model = _base_model(key, task, seed)
        try:
            model.fit(x_fit, y_fit)
        except Exception as exc:  # a family that cannot fit this data is skipped
            val_scores[key] = float("nan")
            _SKIPPED[key] = str(exc)[:200]
            continue
        val_preds[key] = _predict_for_blend(model, x_val, is_classification)
        val_scores[key] = scorer.score(val_preds[key], y_val)
    _maybe_add_tabicl(
        val_preds,
        val_scores,
        scorer,
        x_fit,
        y_fit,
        x_val,
        y_val,
        is_classification,
        x_all,
    )
    if not val_preds:
        raise ModelError(
            "engine='ensemble' could not fit any base model on this data; "
            "try engine='flaml' or check the feature columns"
        )

    weights = _greedy_select(val_preds, scorer, y_val)

    # Refit the selected members on the full training data for the served ensemble.
    members: dict[str, Any] = {}
    for key in weights:
        model = (
            _tabicl_model(is_classification)
            if key == "tabicl"
            else _base_model(key, task, seed)
        )
        model.fit(x_all, y_codes)
        members[key] = model

    ensemble = _EnsembleModel(transformer, members, weights, task, label_state)
    best_score = scorer.score(
        _blend([val_preds[k] for k in weights], list(weights.values())), y_val
    )

    def log_model(
        *, name: str, signature: Any, input_example: Any, pip_requirements: list[str]
    ) -> Any:
        import mlflow

        return mlflow.sklearn.log_model(
            ensemble,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=name,
            pip_requirements=pip_requirements,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    def log_trials(experiment_id: str, parent_run_id: str) -> None:
        _log_base_scores(val_scores, weights, experiment_id, parent_run_id)

    summary = ", ".join(
        f"{k} {round(w, 2)}" for k, w in sorted(weights.items(), key=lambda kv: -kv[1])
    )
    return EngineResult(
        model=ensemble,
        best_estimator=f"weighted ensemble ({summary})",
        best_config={k: round(v, 4) for k, v in weights.items()},
        log_model=log_model,
        training_code=_training_code(weights, feature_names, task),
        sklearn_estimator=None,
        log_trials=log_trials,
        search_best_loss=-float(best_score),
    )


#: Base families skipped at fit time, with the reason (surfaced in the trial log).
_SKIPPED: dict[str, str] = {}


def _resolve_families(estimator_list: Sequence[str]) -> list[str]:
    """The base families to fit: the full diverse set, or an explicit subset."""
    if not estimator_list:
        return list(_FAMILIES)
    unknown = [e for e in estimator_list if e not in _FAMILIES]
    if unknown:
        raise ModelError(
            "unknown base learners for the ensemble engine: "
            + ", ".join(unknown)
            + f" (supported: {', '.join(_FAMILIES)})"
        )
    return list(estimator_list)


def _base_model(key: str, task: str, seed: int) -> Any:
    """One base estimator of the given family for the task."""
    is_clf = task == "classification"
    if key == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if is_clf else LGBMRegressor
        return cls(n_estimators=300, n_jobs=1, verbose=-1, random_state=seed)
    if key == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if is_clf else XGBRegressor
        return cls(
            n_estimators=300,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
            random_state=seed,
        )
    if key == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if is_clf else CatBoostRegressor
        return cls(
            iterations=300, verbose=0, random_seed=seed, allow_writing_files=False
        )
    if key == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(n_estimators=300, n_jobs=1, random_state=seed)
    if key == "extra_tree":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        cls = ExtraTreesClassifier if is_clf else ExtraTreesRegressor
        return cls(n_estimators=300, n_jobs=1, random_state=seed)
    raise ModelError(f"unknown base family {key!r}")


def _maybe_add_tabicl(
    val_preds: dict[str, Any],
    val_scores: dict[str, float],
    scorer: _Scorer,
    x_fit: Any,
    y_fit: Any,
    x_val: Any,
    y_val: Any,
    is_classification: bool,
    x_all: Any,
) -> None:
    """Fold TabICL into the base set when its stack is present and the data fits.

    The foundation model is genuinely different from the trees, which is exactly the
    diversity greedy selection can exploit; but it is only worth its cost within its
    practical size limits, so oversized data simply leaves it out (no error).
    """
    try:
        from .tabicl_engine import require_tabicl

        require_tabicl()
    except ModelError:
        return
    if len(x_all) > _TABICL_MAX_ROWS or x_all.shape[1] > _TABICL_MAX_FEATURES:
        _SKIPPED["tabicl"] = (
            f"data {x_all.shape} exceeds TabICL limits "
            f"({_TABICL_MAX_ROWS} rows x {_TABICL_MAX_FEATURES} features)"
        )
        return
    try:
        model = _tabicl_model(is_classification)
        model.fit(x_fit, y_fit)
        val_preds["tabicl"] = _predict_for_blend(model, x_val, is_classification)
        val_scores["tabicl"] = scorer.score(val_preds["tabicl"], y_val)
    except Exception as exc:  # a foundation-model failure must not sink the ensemble
        _SKIPPED["tabicl"] = str(exc)[:200]


def _tabicl_model(is_classification: bool) -> Any:
    """A TabICL estimator on the best available device (used as an ensemble member)."""
    from .tabicl_engine import make_estimator

    return make_estimator(is_classification)


def _fit_transformer(x_train: Any) -> Any:
    """A fitted one-hot(+passthrough) transformer shared by every base model.

    Object columns are one-hot encoded (unknown categories ignored at serve time);
    numeric columns pass through. Fitting it once and feeding every base model the same
    numeric matrix keeps the members comparable and the served transform singular.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    categorical = [c for c in x_train.columns if x_train[c].dtype == object]
    numeric = [c for c in x_train.columns if c not in categorical]
    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    return transformer.fit(x_train)


def _dense(matrix: Any) -> Any:
    """A dense float array (OneHotEncoder is sparse; the tree learners want dense)."""
    import numpy as np

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _split_indices(n: int, y_codes: Any, task: str, seed: int) -> tuple[Any, Any]:
    """Train/validation index split for selection, stratified for classification."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    stratify = None
    if task == "classification":
        _, counts = np.unique(y_codes, return_counts=True)
        if counts.min() >= 2:
            stratify = y_codes
    fit_idx, val_idx = train_test_split(
        idx, test_size=_VAL_SHARE, random_state=seed, stratify=stratify
    )
    return fit_idx, val_idx


def _predict_for_blend(model: Any, x: Any, is_classification: bool) -> Any:
    """A base model's blendable prediction: class probabilities or regression values."""
    if is_classification:
        return model.predict_proba(x)
    import numpy as np

    return np.asarray(model.predict(x), dtype=float)


def _blend(preds: list[Any], weights: list[float]) -> Any:
    """The weighted average of cached base predictions (proba matrices or vectors)."""
    import numpy as np

    total = float(sum(weights))
    stacked = np.array([w / total for w in weights])[:, None]
    if preds[0].ndim == 2:
        stacked = stacked[:, :, None]
    return np.sum(np.stack(preds) * stacked, axis=0)


def _greedy_select(
    val_preds: dict[str, Any], scorer: _Scorer, y_val: Any
) -> dict[str, float]:
    """Caruana greedy selection with replacement; returns normalized blend weights.

    Starts from the single best base model, then each round adds whichever member most
    improves the blended validation score. The prefix that scored best across all rounds
    wins, so the search cannot overfit into a worse ensemble than the best single.
    """
    keys = [k for k, p in val_preds.items() if p is not None]
    best_single = max(keys, key=lambda k: scorer.score(val_preds[k], y_val))
    chosen = [best_single]
    best_score = scorer.score(val_preds[best_single], y_val)
    best_prefix = list(chosen)
    for _ in range(_SELECTION_ROUNDS):
        scored = []
        for k in keys:
            trial = [*chosen, k]
            blended = _blend([val_preds[c] for c in trial], [1.0] * len(trial))
            scored.append((scorer.score(blended, y_val), k))
        step_score, step_key = max(scored, key=lambda sk: sk[0])
        chosen.append(step_key)
        if step_score > best_score:
            best_score = step_score
            best_prefix = list(chosen)
    counts = Counter(best_prefix)
    total = sum(counts.values())
    return {k: counts[k] / total for k in counts}


class _LabelState:
    """A fitted label encoder mapping the data's classes to codes 0..K-1."""

    def __init__(self, y: Any) -> None:
        from sklearn.preprocessing import LabelEncoder

        self.encoder = LabelEncoder().fit(y)

    def encode(self, y: Any) -> Any:
        return self.encoder.transform(y)

    def decode(self, codes: Any) -> Any:
        return self.encoder.inverse_transform(codes)


class _Scorer:
    """Higher-is-better score of a blended prediction against validation labels."""

    def __init__(self, task: str, metric: str | None, y_codes: Any) -> None:
        import numpy as np

        self.task = task
        self.n_classes = int(np.unique(y_codes).size) if task == "classification" else 0
        self.metric = metric or ("accuracy" if task == "classification" else "r2")

    def score(self, blended: Any, y_val: Any) -> float:
        import numpy as np
        from sklearn import metrics as sk

        if self.task == "regression":
            if self.metric == "rmse":
                return -float(sk.root_mean_squared_error(y_val, blended))
            if self.metric == "mae":
                return -float(sk.mean_absolute_error(y_val, blended))
            return float(sk.r2_score(y_val, blended))
        labels = list(range(self.n_classes))
        if self.metric == "log_loss":
            return -float(sk.log_loss(y_val, blended, labels=labels))
        if self.metric == "roc_auc":
            if self.n_classes == 2:
                return float(sk.roc_auc_score(y_val, blended[:, 1]))
            return float(
                sk.roc_auc_score(y_val, blended, multi_class="ovr", labels=labels)
            )
        pred = np.argmax(blended, axis=1)
        if self.metric == "f1":
            average = "binary" if self.n_classes == 2 else "macro"
            return float(sk.f1_score(y_val, pred, average=average))
        return float(sk.accuracy_score(y_val, pred))


class _EnsembleModel:
    """The served blend: shared transformer, refit members, and their weights.

    Module-scoped so cloudpickle round-trips it wherever the model loads back.
    ``predict`` returns labels in the data's own values (classification) or blended
    values (regression); ``predict_proba`` returns the blended class probabilities.
    """

    def __init__(
        self,
        transformer: Any,
        members: dict[str, Any],
        weights: dict[str, float],
        task: str,
        label_state: _LabelState | None,
    ) -> None:
        self.transformer = transformer
        self.members = members
        self.weights = weights
        self.task = task
        self.label_state = label_state

    def _blended(self, x: Any) -> Any:
        xt = _dense(self.transformer.transform(x))
        keys = list(self.members)
        preds = [
            _predict_for_blend(self.members[k], xt, self.task == "classification")
            for k in keys
        ]
        return _blend(preds, [self.weights[k] for k in keys])

    def predict(self, x: Any) -> Any:
        import numpy as np

        blended = self._blended(x)
        if self.task == "classification":
            # A classification ensemble is always built with a label state (fit sets it
            # iff the task is classification); guard the invariant before decoding.
            if self.label_state is None:
                raise ModelError("classification ensemble is missing its label state")
            codes = np.argmax(blended, axis=1)
            return self.label_state.decode(codes)
        return blended

    def predict_proba(self, x: Any) -> Any:
        return self._blended(x)

    def score(self, x: Any, y: Any, sample_weight: Any = None) -> float:
        from sklearn.metrics import accuracy_score, r2_score

        pred = self.predict(x)
        if self.task == "classification":
            return float(accuracy_score(y, pred, sample_weight=sample_weight))
        return float(r2_score(y, pred, sample_weight=sample_weight))


def _log_base_scores(
    val_scores: dict[str, float],
    weights: dict[str, float],
    experiment_id: str,
    parent_run_id: str,
) -> None:
    """Log each base model's validation score and its blend weight as child runs."""
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    client = MlflowClient()
    for key, score in val_scores.items():
        try:
            run = client.create_run(
                experiment_id,
                run_name=f"base-{key}",
                tags={
                    "mlflow.parentRunId": parent_run_id,
                    "elbi.trial": "true",
                    "elbi.best_trial": str(weights.get(key, 0) > 0).lower(),
                },
            )
            metrics = []
            if not math.isnan(score):
                metrics.append(
                    Metric("val_score", float(score), int(run.info.start_time), 0)
                )
            params = [
                Param("family", key),
                Param("weight", str(round(weights.get(key, 0.0), 4))),
            ]
            if key in _SKIPPED:
                params.append(Param("skipped", _SKIPPED[key]))
            client.log_batch(run.info.run_id, metrics=metrics, params=params)
            client.set_terminated(run.info.run_id, status="FINISHED")
        except Exception:  # noqa: S112 - one bad row must not lose the rest
            continue


def _training_code(weights: dict[str, float], features: list[str], task: str) -> str:
    """A description of the blend and its weights, as a readable artifact."""
    members = "\n".join(
        f"#   {k}: weight {round(w, 4)}"
        for k, w in sorted(weights.items(), key=lambda kv: -kv[1])
    )
    return f'''"""Greedy-weighted ensemble (engine="ensemble").

A diverse base set was fit and blended by Caruana ensemble selection on a held-out
validation split. The final blend weights ({task}):

{members}

To reproduce: fit each family on the (one-hot encoded) features below, then average
their predicted {"class probabilities" if task == "classification" else "values"}
with the weights above. The trained ensemble does exactly this behind one predict().
"""

features = {features!r}
'''
