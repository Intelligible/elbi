"""The notebook capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.notebooks.NotebookService` so the agent operates the same notebook
surface a human does: create, run (reading outputs back so it can self-correct),
inspect, promote a cell to a certified derivation, and schedule. Results are concise and
model-shaped, outputs are truncated so a large table or print cannot flood the context,
and errors name the next move. Creating and running are autonomous (sandboxed);
promoting a cell goes through the same certification gate ``derive`` does, so the gate,
not the agent, authorizes the durable artifact.

Everything the agent does shows up in the UI to inspect, edit, or take over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .notebooks import NotebookService

#: Cap each rendered cell output so a big DataFrame or runaway print stays readable and
#: cheap in context; the full output is always visible to the user in the UI.
_MAX_OUTPUT_CHARS = 2000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n…[truncated]"


def _render_output(output: Mapping[str, Any]) -> str:
    """Render one nbformat output as short text the model can read and act on."""
    kind = output.get("output_type")
    if kind == "stream":
        return _truncate(str(output.get("text", "")).rstrip())
    if kind == "error":
        trace = "\n".join(str(line) for line in output.get("traceback", []))
        return _truncate(trace or f"{output.get('ename')}: {output.get('evalue')}")
    data = output.get("data", {}) or {}
    if "text/plain" in data:
        text = data["text/plain"]
        return _truncate("".join(text) if isinstance(text, list) else str(text))
    if "text/html" in data:
        return "<HTML output (e.g. a DataFrame): rendered in the UI>"
    if any(str(mime).startswith("image/") for mime in data):
        return "<image output: rendered in the UI>"
    if any("vega" in str(mime) for mime in data):
        return "<chart: rendered in the UI>"
    return "<output rendered in the UI>"


class NotebookAgent:
    """Bind a :class:`NotebookService` into the agent's ``nb_*`` string capabilities."""

    def __init__(self, service: NotebookService) -> None:
        self._service = service

    def list(self) -> str:
        """List the notebooks that exist."""
        summaries = self._service.summaries()
        if not summaries:
            return "No notebooks yet. Create one with write_notebook."
        lines = [
            f"- {s['name']} (id {s['id']}, {s['cell_count']} cells)" for s in summaries
        ]
        return "Notebooks:\n" + "\n".join(lines)

    def read(self, notebook_id: str) -> str:
        """Render a notebook's cells, outputs, environment, and schedule."""
        view = self._service.view(notebook_id)
        if view is None:
            return f"No notebook with id {notebook_id!r}."
        parts = [f"# Notebook: {view['name']} (id {view['id']})"]
        env = view["environment"]
        if view["deps"] or env["base_env"]:
            base = f"base '{env['base_env']}' + " if env["base_env"] else ""
            parts.append(f"Environment: {base}{', '.join(view['deps']) or '(none)'}")
        if view["schedule"]:
            parts.append(f"Schedule: {view['schedule']}")
        conflicts = view["graph"].get("conflicts")
        if conflicts:
            parts.append(f"⚠ names defined by more than one cell: {conflicts}")
        for cell in view["cells"]:
            count = cell["execution_count"]
            marker = f"[{count}]" if count is not None else "[ ]"
            header = f"\n## cell {cell['id']} ({cell['cell_type']}) {marker}"
            parts.append(header)
            parts.append(_truncate(cell["source"]))
            outputs = cell["outputs"]
            if outputs:
                parts.append(
                    "output:\n" + "\n".join(_render_output(o) for o in outputs)
                )
        return "\n".join(parts)

    def write(
        self,
        notebook_id: str | None,
        name: str,
        cells: Sequence[Mapping[str, Any]],
        deps: Sequence[str] | None,
    ) -> str:
        """Create a notebook (no id) or replace one's cells; set deps if given."""
        if notebook_id is None:
            notebook_id = self._service.create(name)
        elif self._service.view(notebook_id) is None:
            return f"No notebook with id {notebook_id!r}."
        self._service.set_cells(notebook_id, cells)
        note = ""
        if deps is not None:
            result = self._service.set_environment(notebook_id, deps=list(deps))
            note = (
                f" Environment locked ({len(result.get('lock') or [])} packages)."
                if result.get("ok")
                else f" Environment note: {result.get('error')}."
            )
        saved = self._service.view(notebook_id) or {}
        cell_ids = [c["id"] for c in saved.get("cells", [])]
        return (
            f"Saved '{name}' (id {notebook_id}): {len(cells)} cells.{note}\n"
            f"Cell ids: {', '.join(cell_ids)}\n"
            f"Run it with run_notebook to see outputs. URL: /notebooks/{notebook_id}."
        )

    def run(self, notebook_id: str, cell_ids: Sequence[str] | None) -> str:
        """Run the notebook (all cells or the given ids); render outputs per cell."""
        if self._service.view(notebook_id) is None:
            return f"No notebook with id {notebook_id!r}."
        if cell_ids:
            events = self._service.run_events(notebook_id, list(cell_ids))
        else:
            events = self._service.run_all_events(notebook_id)
        # Collect streamed outputs per cell so the render reads like a run log.
        outputs: dict[str, list[str]] = {}
        status: dict[str, str] = {}
        order: list[str] = []
        installed: list[str] = []
        for event in events:
            kind = event["event"]
            if kind == "cell_start" and event["cell"] not in order:
                order.append(event["cell"])
            elif kind == "output":
                outputs.setdefault(event["cell"], []).append(
                    _render_output(event["output"])
                )
            elif kind == "status":
                status[event["cell"]] = event["status"]
            elif kind == "installed":
                installed = event.get("packages", [])
        lines = []
        if installed:
            lines.append(f"Installed into the environment: {', '.join(installed)}")
        for cell_id in order:
            rendered = "\n".join(outputs.get(cell_id, [])) or "(no output)"
            lines.append(f"cell {cell_id}: {status.get(cell_id, 'done')}:\n{rendered}")
        if not order:
            return "Nothing ran (no code cells)."
        failed = [c for c, s in status.items() if s not in ("ok")]
        summary = (
            "All cells ran."
            if not failed
            else f"{len(failed)} cell(s) failed: fix with write_notebook and re-run."
        )
        return summary + "\n\n" + "\n\n".join(lines)

    def promote_cell(self, notebook_id: str, cell_id: str) -> str:
        """Promote a cell's ``def <name>(ctx)`` to a certified derivation (gated)."""
        result = self._service.promote_cell(notebook_id, cell_id)
        if result.get("certified"):
            return (
                f"Certified derivation '{result['name']}'"
                + (f" (verdict {result['verdict']})" if result.get("verdict") else "")
                + ". It is now durable, cached, and available to train models on."
            )
        return "Not certified: " + str(
            result.get("error") or result.get("detail") or result.get("verdict")
        )

    def schedule(
        self,
        notebook_id: str,
        mode: str,
        interval_hours: float | None,
        dataset: str | None,
    ) -> str:
        """Set a notebook's rerun schedule (interval or on dataset change)."""
        if self._service.view(notebook_id) is None:
            return f"No notebook with id {notebook_id!r}."
        if mode not in ("interval", "on_data_change"):
            return "mode must be 'interval' or 'on_data_change'."
        schedule: dict[str, Any] = {"enabled": True, "mode": mode}
        if mode == "interval":
            schedule["interval_hours"] = interval_hours or 24
            detail = f"every {schedule['interval_hours']}h"
        else:
            if not dataset:
                return "on_data_change needs a `dataset`."
            schedule["dataset"] = dataset
            detail = f"when dataset '{dataset}' changes"
        self._service.set_schedule(notebook_id, schedule)
        return f"Scheduled notebook {notebook_id} to rerun {detail}."
