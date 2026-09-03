"""elbi-agent: the verifying analysis runtime.

An LLM drives a tool loop over a dataset and can only report a directional effect
through the verification gate, so a returned effect is always one that passed the
soundness checks. The LLM is injected via :class:`LLMClient`; the loop entry point
is :func:`answer`.
"""

from __future__ import annotations

from .condense import condense_turns
from .llm import LLMClient, Step, StepDelta, ToolCall, ToolSpec, Transcript, Usage
from .runtime import (
    AnswerResult,
    BatchScoreFn,
    DeriveFn,
    DeriveOutcome,
    Event,
    ListModelsFn,
    PredictFn,
    PromoteFn,
    TrainFn,
    Workspace,
    answer,
    stream,
)

__version__ = "0.1.0"

__all__ = [
    "AnswerResult",
    "BatchScoreFn",
    "DeriveFn",
    "DeriveOutcome",
    "Event",
    "LLMClient",
    "ListModelsFn",
    "PredictFn",
    "PromoteFn",
    "Step",
    "StepDelta",
    "ToolCall",
    "ToolSpec",
    "TrainFn",
    "Transcript",
    "Usage",
    "Workspace",
    "__version__",
    "answer",
    "condense_turns",
    "stream",
]
