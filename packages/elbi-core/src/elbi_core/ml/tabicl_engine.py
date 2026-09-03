"""TabICL as an open, pretrained tabular foundation model (engine="tabicl").

TabICL (Inria's soda team) predicts tabular targets by in-context learning: the training
rows are fed to a pretrained transformer as context and the query rows are answered in a
single forward pass, with no per-dataset training. It is the current state of the art on
the TabArena living benchmark, beating tuned gradient boosting on most datasets, and --
unlike the other foundation models -- it is genuinely open: BSD-3-licensed code *and*
weights with no use-case restriction, so it is safe in a commercial product, and the
checkpoints download from Hugging Face on first use with no token or license step.

It fits the small-to-mid regime it was trained for best (hundreds to tens of thousands
of rows, up to a few hundred features), scaling to larger data with more time.
Categorical columns are one-hot encoded and class labels handled in a wrapper, so the
served model takes raw feature rows and returns labels in the data's own values. This
module is also the ensemble engine's foundation-model member via :func:`make_estimator`.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..errors import ModelError
from .engine import EngineResult

#: TabICL's practical ceiling here. It was pretrained on up to ~48k rows and generalizes
#: beyond that, but a single forward pass over an unbounded table is refused with
#: guidance toward the search engines instead.
_MAX_ROWS = 100000
_MAX_FEATURES = 1000

#: On CPU the forward pass is only quick for smaller data; above this a note explains a
#: slow run rather than leaving it mysterious.
_CPU_ROW_ADVISORY = 5000


def require_tabicl() -> None:
    """Raise :class:`ModelError` unless TabICL is importable."""
    try:
        import tabicl  # noqa: F401
    except ImportError as exc:
        raise ModelError(
            "engine='tabicl' requires the tabicl extra: pip install 'elbi[tabicl]'"
        ) from exc


def _device() -> str:
    """The best available torch device: the GPU when present, else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_estimator(is_classification: bool, device: str | None = None) -> Any:
    """A bare TabICL estimator on the chosen device (or the best available one).

    This is the ensemble engine's TabICL member, which is fed pre-encoded numeric
    arrays and integer class codes; it does no preprocessing of its own.
    """
    require_tabicl()
    from tabicl import TabICLClassifier, TabICLRegressor

    device = device or _device()
    cls = TabICLClassifier if is_classification else TabICLRegressor
    return cls(device=device)


def fit(*, task: str, x_train: Any, y_train: Any, seed: int) -> EngineResult:
    """Fit a TabICL model (context load, no training) and return the result."""
    require_tabicl()
    is_classification = task == "classification"

    n_rows, n_features = len(x_train), x_train.shape[1]
    if n_rows > _MAX_ROWS or n_features > _MAX_FEATURES:
        raise ModelError(
            f"TabICL handles up to {_MAX_ROWS} rows and {_MAX_FEATURES} features; "
            f"this data is {n_rows} rows x {n_features} features. Sample the data "
            "down, or use engine='autogluon'/'ensemble' for data this size"
        )
    device = _device()
    note = ""
    if device == "cpu" and n_rows > _CPU_ROW_ADVISORY:
        note = f" (running on CPU with {n_rows} rows; a GPU is faster for TabICL)"

    model = _TabICLModel(
        _pipeline(x_train, is_classification, device),
        _LabelState(y_train) if is_classification else None,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
    except Exception as exc:
        raise ModelError(f"TabICL training failed: {exc}") from exc

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

    kind = "classifier" if is_classification else "regressor"
    return EngineResult(
        model=model,
        best_estimator=f"TabICL foundation model ({kind}, device={device}){note}",
        best_config={"model": "tabicl", "device": device, "n_context_rows": n_rows},
        log_model=log_model,
        training_code=_training_code(is_classification),
        sklearn_estimator=None,
        log_trials=None,
        search_best_loss=None,
    )


def _pipeline(x_train: Any, is_classification: bool, device: str) -> Any:
    """A one-hot(+passthrough) preprocessor feeding a bare TabICL estimator.

    Object columns are one-hot encoded (unknown categories ignored at serve time),
    numeric columns pass through, so the served model accepts raw feature rows.
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
    estimator = make_estimator(is_classification, device)
    return Pipeline([("prep", prep), ("model", estimator)])


class _LabelState:
    """A fitted label encoder mapping the data's classes to codes 0..K-1."""

    def __init__(self, y: Any) -> None:
        from sklearn.preprocessing import LabelEncoder

        self.encoder = LabelEncoder().fit(y)

    def encode(self, y: Any) -> Any:
        return self.encoder.transform(y)

    def decode(self, codes: Any) -> Any:
        return self.encoder.inverse_transform(codes)


class _TabICLModel:
    """The served TabICL model: preprocessing pipeline plus label decoding.

    Module-scoped so cloudpickle round-trips it wherever the model loads back.
    ``predict`` returns labels in the data's own values (classification) or values
    (regression); ``predict_proba`` returns class probabilities for a classifier.
    """

    def __init__(self, pipeline: Any, label_state: _LabelState | None) -> None:
        self.pipeline = pipeline
        self.label_state = label_state

    def fit(self, x: Any, y: Any) -> _TabICLModel:
        codes = self.label_state.encode(y) if self.label_state is not None else y
        self.pipeline.fit(x, codes)
        return self

    def predict(self, x: Any) -> Any:
        preds = self.pipeline.predict(x)
        if self.label_state is not None:
            return self.label_state.decode(preds)
        return preds

    def predict_proba(self, x: Any) -> Any:
        return self.pipeline.predict_proba(x)

    def score(self, x: Any, y: Any, sample_weight: Any = None) -> float:
        from sklearn.metrics import accuracy_score, r2_score

        pred = self.predict(x)
        if self.label_state is not None:
            return float(accuracy_score(y, pred, sample_weight=sample_weight))
        return float(r2_score(y, pred, sample_weight=sample_weight))


def _training_code(is_classification: bool) -> str:
    """A standalone script reproducing the TabICL model, as editable code."""
    cls = "TabICLClassifier" if is_classification else "TabICLRegressor"
    metric_line = (
        "from sklearn.metrics import accuracy_score as score"
        if is_classification
        else "from sklearn.metrics import r2_score as score"
    )
    return f'''"""TabICL foundation model: predict in one forward pass, no training.

TabICL loads the training rows as context and answers the query rows from it; the
open (BSD-3) pretrained weights download from Hugging Face on first use, no token
needed. Object (categorical) feature columns should be one-hot encoded first, as
the trained pipeline does.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tabicl import {cls}
{metric_line}

data = pd.read_csv("training_data.csv")  # your dataset here
features = [...]  # your feature columns
target = "TARGET"  # your target column

X_train, X_test, y_train, y_test = train_test_split(
    pd.get_dummies(data[features]), data[target], test_size=0.2, random_state=7
)
model = {cls}(device="cpu")  # use device="cuda" on a GPU
model.fit(X_train, y_train)
print("held-out score:", score(y_test, model.predict(X_test)))
'''
