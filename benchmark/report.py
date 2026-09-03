"""Render the per-trap results as a markdown report and a JSON document.

Both reuse the gate's own ``verdict`` and rendered text; no new report primitive is
introduced. The markdown groups traps by pitfall for a reader; the JSON is the
machine-readable artifact CI uploads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .trap import TrapResult

PITFALL_ORDER = (
    "simpsons",
    "confounding",
    "leakage",
    "rtm",
    "multiverse",
    "selection-berkson",
)

#: Appended to the report so a reader knows what each column means. Wrapped in a
#: <details> block so it renders as a click-to-expand section on GitHub.
_COLUMN_LEGEND = [
    "<details>",
    "<summary>Columns</summary>",
    "",
    "- **trap** - the trap id (its folder name under `traps/`).",
    "- **kind** - `synthetic` (seeded generator) or `real` (a committed dataset; needs "
    ">=2 reviewers to be accepted).",
    "- **status** - `✅ accepted` (reviewed) or `⚠️ candidate` (provisional, not yet "
    "reviewed - do not rely on its verdict). Both run in CI.",
    "- **gate verdict** - what the verification oracle actually returned.",
    "- **expected** - the ground-truth verdict from the manifest, per `RUBRIC.md`.",
    "- **naive** - the hand-written surface conclusion a bare read reaches "
    "(the trap's intended wrong answer).",
    "- **bare LLM** - what the cached bare model actually returned "
    "(report-only), or `uncaptured`.",
    "- **beats LLM** - `yes` if the gate verdict differs from the bare LLM's; "
    "report-only, never affects `result`. `-` means uncaptured.",
    "- **result** - `PASS` when the gate returned `expected` (and, if pinned, "
    "the pivotal reason). The only column that gates the build.",
    "",
    "A trap flagged **saturated** passed but the bare LLM matched the gate "
    "(`beats LLM = no`): still a regression net, not proof it beats a bare model.",
    "",
    "</details>",
    "",
]


def _pivotal_ok(result: TrapResult) -> bool:
    pivotal = result.trap.expected_pivotal
    return pivotal is None or pivotal in result.pivotal_text


def _llm_cell(result: TrapResult) -> str:
    if result.llm_verdict is None:
        return "uncaptured"
    model = result.llm_model or "?"
    return f"{result.llm_verdict} ({model})"


def _status_cell(status: str) -> str:
    """A visually distinct status marker: candidates are flagged provisional."""
    return "⚠️ candidate" if status == "candidate" else "✅ accepted"


def to_json(results: Sequence[TrapResult]) -> dict[str, Any]:
    """Build the machine-readable report document."""
    warnings = [r.warning for r in results if r.warning]
    beat = sum(1 for r in results if r.beats_llm)
    captured = sum(1 for r in results if r.llm_verdict is not None)
    models = sorted({r.llm_model for r in results if r.llm_model})
    saturated = [r.trap.id for r in results if r.saturated]
    return {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "real": sum(1 for r in results if r.trap.is_real),
            "synthetic": sum(1 for r in results if not r.trap.is_real),
            "llm_captured": captured,
            "llm_models": models,
            "beats_llm": beat,
            "saturated": len(saturated),
            "warnings": len(warnings),
        },
        "flagged_for_pruning": saturated,
        "warnings": warnings,
        "traps": [
            {
                "id": r.trap.id,
                "pitfall": r.trap.pitfall,
                "kind": "real" if r.trap.is_real else "synthetic",
                "status": r.trap.status,
                "gate": r.trap.gate,
                "provenance": r.trap.provenance,
                "description": r.trap.description,
                "expected_verdict": r.trap.expected_verdict,
                "gate_verdict": r.verdict,
                "expected_pivotal": r.trap.expected_pivotal,
                "pivotal_ok": _pivotal_ok(r),
                "passed": r.passed,
                "naive_verdict": r.trap.naive_verdict,
                "llm_verdict": r.llm_verdict,
                "llm_model": r.llm_model,
                "beats_llm": r.beats_llm,
                "saturated": r.saturated,
            }
            for r in results
        ],
    }


def render_markdown(results: Sequence[TrapResult]) -> str:
    """Render the per-trap pass/fail report, grouped by pitfall."""
    doc = to_json(results)
    summary = doc["summary"]
    baseline = (
        f"bare-LLM baseline: {', '.join(summary['llm_models'])} "
        f"(captured for {summary['llm_captured']}/{summary['total']}, "
        f"gate beats it on {summary['beats_llm']})"
        if summary["llm_models"]
        else "no bare-LLM baseline captured yet (run `benchmark.run --refresh-llm`)"
    )
    lines = [
        "# Statistical-pitfall benchmark",
        "",
        f"**{summary['passed']}/{summary['total']} traps pass** "
        f"(gate returns the expected verdict) - "
        f"{summary['synthetic']} synthetic, {summary['real']} real.",
        "",
        baseline + f"; {summary['saturated']} saturated.",
        "",
    ]
    candidates = [r.trap.id for r in results if r.trap.status == "candidate"]
    if candidates:
        lines.append(
            "> ⚠️ **Candidate traps - results are provisional.** Not yet reviewed and "
            "accepted; do not rely on their verdicts. A reviewer should check each "
            "against `RUBRIC.md`, following the review steps in `benchmark/README.md`, "
            "then set `status: accepted`:"
        )
        lines += [f"> - `{tid}`" for tid in candidates]
        lines.append("")
    if doc["warnings"]:
        lines.append("## Flagged for pruning (saturated: bare LLM matched the gate)")
        lines += [f"- {w}" for w in doc["warnings"]]
        lines.append("")

    by_pitfall: dict[str, list[TrapResult]] = {}
    for r in results:
        by_pitfall.setdefault(r.trap.pitfall, []).append(r)
    ordered = [p for p in PITFALL_ORDER if p in by_pitfall]
    ordered += [p for p in by_pitfall if p not in PITFALL_ORDER]

    for pitfall in ordered:
        lines.append(f"## {pitfall}")
        lines.append("")
        headers = [
            "trap",
            "kind",
            "status",
            "gate verdict",
            "expected",
            "naive",
            "bare LLM",
            "beats LLM",
            "result",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in by_pitfall[pitfall]:
            beats = "-" if r.beats_llm is None else ("yes" if r.beats_llm else "no")
            result = "PASS" if r.passed else "FAIL"
            kind = "real" if r.trap.is_real else "synthetic"
            lines.append(
                f"| {r.trap.id} | {kind} | {_status_cell(r.trap.status)} | "
                f"{r.verdict} | {r.trap.expected_verdict} | {r.trap.naive_verdict} | "
                f"{_llm_cell(r)} | {beats} | {result} |"
            )
        lines.append("")

    lines += _COLUMN_LEGEND
    return "\n".join(lines)
