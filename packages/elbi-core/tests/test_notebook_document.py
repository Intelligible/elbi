"""Tests for the notebook document model: ``.ipynb`` round-tripping and parameters.

The document is the interchange format, so the invariants that matter are the ones a
foreign tool relies on: a round-trip preserves cells and outputs, an older or malformed
notebook is repaired (unique cell ids, string-or-list source), and outputs clear on a
restart. The parameters cases pin the papermill-style injection the reruns build on.
"""

from __future__ import annotations

from elbi_core.notebook import Cell, Notebook, apply_overrides
from elbi_core.notebook.parameters import (
    declared_parameters,
    find_parameters_cell,
)


def test_ipynb_round_trip_preserves_cells_and_outputs() -> None:
    nb = Notebook(
        cells=[
            Cell(cell_type="markdown", source="# Title"),
            Cell(
                cell_type="code",
                source="x = 1\nx",
                execution_count=1,
                outputs=[
                    {
                        "output_type": "execute_result",
                        "execution_count": 1,
                        "data": {"text/plain": "1"},
                        "metadata": {},
                    }
                ],
            ),
        ]
    )
    restored = Notebook.from_ipynb(nb.to_ipynb())
    assert [c.cell_type for c in restored.cells] == ["markdown", "code"]
    assert restored.cells[1].source == "x = 1\nx"
    assert restored.cells[1].outputs[0]["output_type"] == "execute_result"
    assert restored.nbformat == 4
    assert restored.nbformat_minor == 5


def test_source_accepts_list_of_lines() -> None:
    # nbformat allows source as a list of lines (the common on-disk form); it joins.
    cell = Cell.from_ipynb({"cell_type": "code", "source": ["a = 1\n", "b = 2"]})
    assert cell.source == "a = 1\nb = 2"


def test_missing_and_duplicate_ids_are_repaired() -> None:
    nb = Notebook.from_ipynb(
        {
            "nbformat": 4,
            "nbformat_minor": 4,  # pre-id notebook
            "cells": [
                {"cell_type": "code", "source": "a = 1"},
                {"cell_type": "code", "source": "b = 2", "id": "dup"},
                {"cell_type": "code", "source": "c = 3", "id": "dup"},
            ],
        }
    )
    ids = [c.id for c in nb.cells]
    assert all(ids)  # every cell got an id
    assert len(set(ids)) == 3  # the collision was resolved


def test_clear_outputs_resets_code_cells() -> None:
    nb = Notebook(cells=[Cell(source="x", execution_count=3, outputs=[{"o": 1}])])
    nb.clear_outputs()
    assert nb.cells[0].outputs == []
    assert nb.cells[0].execution_count is None


def test_apply_overrides_injects_after_parameters_cell() -> None:
    nb = Notebook(
        cells=[
            Cell(source="alpha = 0.1", metadata={"tags": ["parameters"]}),
            Cell(cell_type="code", source="print(alpha)"),
        ]
    )
    applied = apply_overrides(nb, {"alpha": 0.9})
    sources = [c.source for c in applied.cells]
    assert sources[0] == "alpha = 0.1"
    assert "alpha = 0.9" in sources[1]  # injected right after the parameters cell
    assert sources[2] == "print(alpha)"


def test_apply_overrides_replaces_a_stale_injected_cell() -> None:
    nb = Notebook(cells=[Cell(source="alpha = 0.1", metadata={"tags": ["parameters"]})])
    once = apply_overrides(nb, {"alpha": 1})
    twice = apply_overrides(once, {"alpha": 2})
    injected = [c for c in twice.cells if c.has_tag("injected-parameters")]
    assert len(injected) == 1  # not stacked
    assert "alpha = 2" in injected[0].source


def test_declared_parameters_reads_literals() -> None:
    cell = Cell(source="alpha = 0.1\nname = 'demo'\ncomputed = alpha * 2")
    nb = Notebook(cells=[Cell(source="alpha = 0.1", metadata={"tags": ["parameters"]})])
    assert find_parameters_cell(nb) is nb.cells[0]
    params = declared_parameters(cell)
    assert params["alpha"] == 0.1
    assert params["name"] == "demo"
    assert params["computed"] is None  # a computed default is known but has no literal
