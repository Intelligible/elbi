"""Notebooks: an authoring surface whose outputs become certified derivations.

A notebook here is the human-facing twin of the agent's exploration loop (a reactive,
cell-based interpreter for iterating on data), not a deliverable in itself. Its value is
what leaves it: a working cell is promoted to a governed, cached derivation, and a
training cell registers a model. The building blocks are a nbformat-compatible document
(:mod:`~elbi.notebook.document`), static dataflow analysis for reactive execution
(:mod:`~elbi.notebook.dependencies`), a persistent kernel with rich output
(:mod:`~elbi.notebook.kernel`), and papermill-style parameterization
(:mod:`~elbi.notebook.parameters`).
"""

from __future__ import annotations

from .dependencies import CellDeps, DependencyGraph, analyze_code
from .document import (
    Cell,
    Notebook,
    display_data,
    error_output,
    execute_result,
    new_id,
    stream_output,
)
from .environment import resolve_lock
from .kernel import (
    MAX_OUTPUT_BYTES,
    CommListener,
    Kernel,
    OutputSink,
    QueryResolver,
    SubprocessKernel,
)
from .parameters import (
    INJECTED_TAG,
    PARAMETERS_TAG,
    apply_overrides,
    declared_parameters,
    find_parameters_cell,
)

__all__ = [
    "INJECTED_TAG",
    "MAX_OUTPUT_BYTES",
    "PARAMETERS_TAG",
    "Cell",
    "CellDeps",
    "CommListener",
    "DependencyGraph",
    "Kernel",
    "Notebook",
    "OutputSink",
    "QueryResolver",
    "SubprocessKernel",
    "analyze_code",
    "apply_overrides",
    "declared_parameters",
    "display_data",
    "error_output",
    "execute_result",
    "find_parameters_cell",
    "new_id",
    "resolve_lock",
    "stream_output",
]
