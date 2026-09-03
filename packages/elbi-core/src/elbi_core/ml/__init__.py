"""Predictive models as first-class, certified artifacts (the optional ``ml`` extra).

Three deliberately standard pieces, with MLflow as the system of record throughout:

- :mod:`.training` runs FLAML AutoML under a budget, evaluates on a held-out split,
  logs the run and the fitted model to MLflow, and registers it in the Model
  Registry; the verification oracle's ``prediction`` gate decides whether the first
  version earns the ``champion`` alias automatically.
- :mod:`.registry` is the typed ``MlflowClient`` view the platform's surfaces share:
  list models and versions, promote an alias, load a version for scoring.
- :mod:`.scoring` implements MLflow's ``/invocations`` payload protocol, so a served
  model is callable by anything that already speaks to an MLflow scoring server.

Everything imports its stack lazily; without the extra installed, calls raise
:class:`elbi.errors.ModelError` naming the install command.
"""

from .batch import BatchScoreReport, batch_score
from .drift import MIN_DRIFT_ROWS, ColumnDrift, DriftReport, data_drift
from .registry import (
    CHAMPION,
    ModelRegistry,
    ModelVersionInfo,
    RegisteredModelInfo,
)
from .scoring import parse_invocations, predictions_payload
from .training import ENGINES, TrainingReport, require_ml, train_automl

__all__ = [
    "CHAMPION",
    "ENGINES",
    "MIN_DRIFT_ROWS",
    "BatchScoreReport",
    "ColumnDrift",
    "DriftReport",
    "ModelRegistry",
    "ModelVersionInfo",
    "RegisteredModelInfo",
    "TrainingReport",
    "batch_score",
    "data_drift",
    "parse_invocations",
    "predictions_payload",
    "require_ml",
    "train_automl",
]
