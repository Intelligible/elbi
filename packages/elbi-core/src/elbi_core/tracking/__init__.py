"""Result history: a durable, comparable record of a derivation's certified versions.

Every time a derivation is certified, it produces a number: the verification oracle's
estimate for the claim. This package keeps those results as a derivation's *history*, so
you can see how a certified answer changed as the data, the code, or the controls
changed, and attribute the change to its cause. Unlike a conventional experiment
tracker, which faithfully records whatever metric the training code reports, every
number here is the oracle's verified estimate, so the history is a record of trustworthy
results.

Because a derivation is deterministic and content-addressed, a version's identity is its
``derivation_version``: an identical re-run collapses to the same version, and a change
to an input yields a new one. A derivation's history is the series of its distinct
certified versions over time (:class:`JsonlRunLog`), and :func:`changed_dimensions` says
what moved the estimate between two of them.
:func:`~elbi.tracking.mlflow.emit_run` optionally exports a certified version to
a team's existing MLflow.
"""

from __future__ import annotations

from .history import changed_dimensions, with_changes
from .log import JsonlRunLog, RunLog
from .run import CertifiedRun, now_iso, run_from_author

__all__ = [
    "CertifiedRun",
    "JsonlRunLog",
    "RunLog",
    "changed_dimensions",
    "now_iso",
    "run_from_author",
    "with_changes",
]
