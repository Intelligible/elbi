"""The notebook document model: an ``.ipynb``-compatible cell list, in memory.

A notebook is a thin, standard object, the Jupyter ``nbformat`` v4.5 schema, so one
authored here opens in Jupyter, VS Code, or nbviewer unchanged, and one exported from
those tools imports here. We keep the model deliberately close to the on-disk format
(four output shapes, a MIME-bundle ``data`` dict, per-cell ids) rather than inventing a
parallel representation, and repair the schema's rules on read so a malformed or older
notebook is upgraded cleanly instead of crashing a kernel later.

This module is pure data: it defines the cells and outputs, reads and writes ``.ipynb``
JSON, and mints stable cell ids. The stateful side (running a cell, capturing its
output) is the kernel's job (:mod:`elbi.notebook.kernel`); an output object here is
exactly what the kernel produces and what the browser renders.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

NBFORMAT = 4
NBFORMAT_MINOR = 5

CellType = Literal["code", "markdown", "raw"]

#: nbformat 4.5 cell-id charset and length: ``[A-Za-z0-9-_]`` up to 64 chars, unique per
#: notebook. Ids anchor comments, links, and diffs, so they are stable across saves.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def new_id() -> str:
    """Mint a fresh, schema-valid cell id (a short URL-safe token)."""
    return secrets.token_urlsafe(9).replace("=", "")[:16]


def _as_text(value: Any) -> str:
    """Normalize a nbformat multiline string (string or list of lines) to a string."""
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return "" if value is None else str(value)


def _split_lines(text: str) -> list[str]:
    """Split into nbformat's list-of-lines form (newlines kept, no trailing empty)."""
    return text.splitlines(keepends=True)


@dataclass
class Cell:
    """One notebook cell: code, markdown, or raw.

    ``outputs`` and ``execution_count`` are meaningful only for code cells (empty and
    ``None`` otherwise). ``outputs`` holds nbformat output objects verbatim (``stream``,
    ``display_data``, ``execute_result``, ``error``), so the same list round-trips to
    ``.ipynb`` and streams to the browser without translation.
    """

    id: str = field(default_factory=new_id)
    cell_type: CellType = "code"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    execution_count: int | None = None

    @property
    def tags(self) -> list[str]:
        """The cell's tag list (where ``parameters`` marks the parameters cell)."""
        tags = self.metadata.get("tags", [])
        return [str(tag) for tag in tags] if isinstance(tags, list) else []

    def has_tag(self, tag: str) -> bool:
        """Whether ``tag`` is present in the cell's tag list."""
        return tag in self.tags

    def to_ipynb(self) -> dict[str, Any]:
        """Serialize to a nbformat 4.5 cell object."""
        cell: dict[str, Any] = {
            "cell_type": self.cell_type,
            "id": self.id,
            "metadata": self.metadata,
            "source": _split_lines(self.source),
        }
        if self.cell_type == "code":
            cell["execution_count"] = self.execution_count
            cell["outputs"] = self.outputs
        return cell

    @classmethod
    def from_ipynb(cls, data: Mapping[str, Any]) -> Cell:
        """Read a nbformat cell, tolerating list-or-string source and a missing id."""
        cell_type = str(data.get("cell_type", "code"))
        if cell_type not in ("code", "markdown", "raw"):
            cell_type = "raw"
        raw_id = str(data.get("id", "") or "")
        cell_id = raw_id if _ID_RE.match(raw_id) else new_id()
        metadata = dict(data.get("metadata", {}) or {})
        outputs = list(data.get("outputs", []) or []) if cell_type == "code" else []
        count = data.get("execution_count") if cell_type == "code" else None
        return cls(
            id=cell_id,
            cell_type=cell_type,  # type: ignore[arg-type]
            source=_as_text(data.get("source", "")),
            metadata=metadata,
            outputs=outputs,
            execution_count=int(count) if isinstance(count, int) else None,
        )


@dataclass
class Notebook:
    """A notebook document: ordered cells plus nbformat metadata.

    This is the interchange and in-memory representation. Application-level facts about
    a
    notebook (its name, kernel dependencies, schedule) live in the app's own
    store;
    ``metadata`` here carries only what the ``.ipynb`` format defines (kernelspec,
    language_info, and any namespaced extras, preserved on round-trip).
    """

    cells: list[Cell] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    nbformat: int = NBFORMAT
    nbformat_minor: int = NBFORMAT_MINOR

    def code_cells(self) -> list[Cell]:
        """The code cells, in document order (the ones that participate in dataflow)."""
        return [cell for cell in self.cells if cell.cell_type == "code"]

    def dependency_pairs(self) -> list[tuple[str, str]]:
        """``(id, source)`` per code cell: the input to a :class:`DependencyGraph`."""
        return [(cell.id, cell.source) for cell in self.code_cells()]

    def clear_outputs(self) -> None:
        """Drop every code cell's outputs and reset its counter (restart & clear)."""
        for cell in self.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

    def to_ipynb(self) -> dict[str, Any]:
        """Serialize to a nbformat 4.5 notebook object (a valid ``.ipynb`` payload)."""
        metadata = dict(self.metadata)
        metadata.setdefault(
            "kernelspec",
            {"name": "python3", "display_name": "Python 3", "language": "python"},
        )
        metadata.setdefault(
            "language_info",
            {"name": "python", "mimetype": "text/x-python", "file_extension": ".py"},
        )
        return {
            "nbformat": NBFORMAT,
            "nbformat_minor": NBFORMAT_MINOR,
            "metadata": metadata,
            "cells": [cell.to_ipynb() for cell in self.cells],
        }

    @classmethod
    def from_ipynb(cls, data: Mapping[str, Any]) -> Notebook:
        """Read a nbformat notebook, minting ids and de-duplicating them if needed.

        Older notebooks (minor < 5) carry no cell ids and duplicates can arrive from
        copy/paste; both are repaired so every cell ends up with a unique, valid id.
        """
        cells = [Cell.from_ipynb(cell) for cell in data.get("cells", []) or []]
        _ensure_unique_ids(cells)
        return cls(
            cells=cells,
            metadata=dict(data.get("metadata", {}) or {}),
            nbformat=int(data.get("nbformat", NBFORMAT) or NBFORMAT),
            nbformat_minor=NBFORMAT_MINOR,
        )


def _ensure_unique_ids(cells: Iterable[Cell]) -> None:
    """Give every cell a unique id in place, replacing collisions with fresh ones."""
    seen: set[str] = set()
    for cell in cells:
        if cell.id in seen or not _ID_RE.match(cell.id):
            cell.id = new_id()
        seen.add(cell.id)


# -- output constructors -------------------------------------------------------------
# The four nbformat output shapes, as factory functions. The kernel worker builds these
# in the child; these mirror them for code that assembles outputs host-side (a promoted
# cell, a synthesized parameters echo), so the shape is defined in exactly one place.


def stream_output(name: str, text: str) -> dict[str, Any]:
    """A ``stream`` output: text written to stdout or stderr."""
    return {"output_type": "stream", "name": name, "text": text}


def execute_result(data: Mapping[str, Any], execution_count: int) -> dict[str, Any]:
    """An ``execute_result``: the REPL value of a cell's last expression."""
    return {
        "output_type": "execute_result",
        "execution_count": execution_count,
        "data": dict(data),
        "metadata": {},
    }


def display_data(
    data: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A ``display_data`` output: a MIME bundle shown mid-cell (display/figure)."""
    return {
        "output_type": "display_data",
        "data": dict(data),
        "metadata": dict(metadata or {}),
    }


def error_output(ename: str, evalue: str, traceback: Sequence[str]) -> dict[str, Any]:
    """An ``error`` output: an exception's class, message, and formatted traceback."""
    return {
        "output_type": "error",
        "ename": ename,
        "evalue": evalue,
        "traceback": list(traceback),
    }
