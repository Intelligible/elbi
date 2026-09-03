"""The engine-neutral result that the AutoML engines hand back to the trainer.

Every engine (FLAML, AutoGluon, the Optuna tuner, the TabICL tabular foundation
model, and the cross-model ensemble) fits a model its own way, but the
plumbing around it is identical: hold-out metrics, the oracle's signal check, the
MLflow run, the evaluation and explainability evidence, lineage, the registry, and
promotion. Each engine expresses its differences through this one object, so the
trainer never special-cases an engine by name.

``model`` is the engine-neutral predict surface (``predict`` returning an array, and
``predict_proba`` for a classifier). ``log_model`` is invoked inside the active MLflow
run to log-and-register the model with the flavor the engine needs (a native sklearn
flavor, or a pyfunc wrapper for engines MLflow has no flavor for). ``training_code`` is
the glass-box reproduction script. ``sklearn_estimator`` is the underlying fitted
estimator when one exists (for tree-model feature importances); it is ``None`` for
engines whose model is not a single sklearn estimator. ``log_trials`` records the
search/leaderboard as nested child runs when the engine has one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class LogModel(Protocol):
    """Log-and-register the fitted model inside the active MLflow run.

    Called with the shared serving metadata (an inferred signature, a small input
    example, and the pinned requirements); returns MLflow's ``ModelInfo`` so the
    trainer can read back the registered version and model URI.
    """

    def __call__(
        self,
        *,
        name: str,
        signature: Any,
        input_example: Any,
        pip_requirements: list[str],
    ) -> Any:
        """Log-and-register the model, returning MLflow's ``ModelInfo``."""
        ...


@dataclass(frozen=True)
class EngineResult:
    """What an engine returns to the trainer after fitting one model."""

    model: Any
    best_estimator: str
    best_config: dict[str, Any]
    log_model: LogModel
    training_code: str
    sklearn_estimator: Any = None
    log_trials: Callable[[str, str], None] | None = None
    search_best_loss: float | None = None
