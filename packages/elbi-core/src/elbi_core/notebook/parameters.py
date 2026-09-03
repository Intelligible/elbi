"""Parameterized runs: override a notebook's inputs without editing it.

One code cell may be tagged ``parameters`` and hold default assignments (``alpha =
0.1``, ``start = "2026-01-01"``). A run can then supply overrides, and we splice a
synthesized cell of just those assignments immediately after the defaults, so the
overrides win when execution reaches them: the papermill model, reimplemented over our
own document so it needs no extra dependency and composes with reactive execution.

The honest limitation is papermill's: overrides are plain re-assignments, not a
recompute. If the parameters cell also derives ``twice = alpha * 2``, overriding
``alpha`` alone does not update ``twice`` under sequential injection, but under this
notebook's reactive engine the injected cell *defines* the parameter names, so its
dependents (including a later ``twice``) are re-run exactly as if the value had been
edited by hand.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from .document import Cell, Notebook

PARAMETERS_TAG = "parameters"
INJECTED_TAG = "injected-parameters"


def find_parameters_cell(notebook: Notebook) -> Cell | None:
    """The first cell tagged ``parameters``, or ``None`` if the notebook has none."""
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.has_tag(PARAMETERS_TAG):
            return cell
    return None


def declared_parameters(cell: Cell) -> dict[str, Any]:
    """Read the default parameter names and literal values from a parameters cell.

    Only top-level ``name = <literal>`` assignments are read (via
    :func:`ast.literal_eval`); a computed default is reported with value ``None`` so the
    name is still known to be a parameter without executing anything.
    """
    defaults: dict[str, Any] = {}
    try:
        tree = ast.parse(cell.source)
    except SyntaxError:
        return defaults
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    defaults[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    defaults[target.id] = None
    return defaults


def render_injected_source(overrides: Mapping[str, Any]) -> str:
    """Render overrides as a block of ``name = <repr>`` assignments.

    Values are emitted with :func:`repr`, a valid Python literal for the JSON types a
    parameter carries (str, int, float, bool, None, list, dict); a value whose ``repr``
    does not round-trip would not be a sensible parameter and is rejected upstream.
    """
    lines = ["# Injected parameters"]
    lines += [f"{name} = {value!r}" for name, value in overrides.items()]
    return "\n".join(lines)


def injected_cell(overrides: Mapping[str, Any]) -> Cell:
    """Build the synthesized parameters cell (tagged ``injected-parameters``)."""
    return Cell(
        id=INJECTED_TAG,
        cell_type="code",
        source=render_injected_source(overrides),
        metadata={"tags": [INJECTED_TAG]},
    )


def apply_overrides(notebook: Notebook, overrides: Mapping[str, Any]) -> Notebook:
    """Return a copy of ``notebook`` with an injected-parameters cell spliced in.

    The injected cell goes immediately after the ``parameters`` cell, or at the top when
    there is none, matching papermill, so its assignments take effect before any cell
    that consumes them. The original notebook is left untouched.
    """
    if not overrides:
        return notebook
    cells = list(notebook.cells)
    injected = injected_cell(overrides)
    params_cell = find_parameters_cell(notebook)
    if params_cell is None:
        cells.insert(0, injected)
    else:
        index = cells.index(params_cell)
        # Replace a stale injected cell from a prior run rather than stacking a new one.
        if index + 1 < len(cells) and cells[index + 1].has_tag(INJECTED_TAG):
            cells[index + 1] = injected
        else:
            cells.insert(index + 1, injected)
    return Notebook(
        cells=cells,
        metadata=dict(notebook.metadata),
        nbformat=notebook.nbformat,
        nbformat_minor=notebook.nbformat_minor,
    )
