"""The bare-LLM baseline: a cached read path (CI) and an opt-in refresh path.

The read path loads a committed ``llm_baseline.json`` and is pure file I/O, so it runs
free and deterministically on every PR. The refresh path re-queries a bare model (no
verification tool, so it is not the gated runtime) and rewrites the cache for a human
to review and commit. Refresh needs an API key and never runs on a PR leg.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .loader import BENCHMARK_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from elbi_agent.llm import LLMClient

    from .trap import Rows, Trap

BASELINE_PATH = BENCHMARK_DIR / "llm_baseline.json"

#: Rows sampled into the prompt; a bare reader judges from a sample, not 6000 rows.
_SAMPLE_ROWS = 300
#: Bumped when the baseline prompt changes, so stale captures are visible in the diff.
PROMPT_VERSION = 1

_SYSTEM = (
    "You are a data analyst. Given a dataset sample and a claimed relationship, decide "
    "whether the claim is statistically SOUND (well supported) or UNSOUND (an artefact "
    "of a pitfall such as confounding, a collider, Simpson's paradox, leakage, or "
    "regression to the mean). Reason briefly, then end with a final line exactly: "
    "'VERDICT: sound' or 'VERDICT: unsound' or 'VERDICT: inconclusive'."
)


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict[str, Any]]:
    """Return the cached ``trap id -> capture`` map, or empty if none is committed."""
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): dict(v) for k, v in loaded.items()}


def _sample(rows: Rows) -> Rows:
    return rows[:_SAMPLE_ROWS]


def _prompt(trap: Trap, rows: Rows) -> str:
    sample = _sample(rows)
    columns = list(sample[0].keys()) if sample else []
    header = ",".join(columns)
    body = "\n".join(",".join(r.get(c, "") for c in columns) for r in sample)
    note = (
        f"(showing {len(sample)} of {len(rows)} rows)"
        if len(rows) > len(sample)
        else ""
    )
    return (
        f"Claim to judge (gate={trap.gate}, roles={trap.claim}):\n"
        f"{trap.description}\n\n"
        f"Dataset {note}:\n{header}\n{body}\n"
    )


def _parse_verdict(text: str) -> str:
    for line in reversed(text.splitlines()):
        low = line.strip().lower()
        if low.startswith("verdict:"):
            token = low.split(":", 1)[1].strip()
            if token in ("sound", "unsound", "inconclusive"):
                return token
    return "inconclusive"


def query_llm(client: LLMClient, trap: Trap, rows: Rows) -> tuple[str, str]:
    """Ask a bare model for a verdict on one trap; return (verdict, excerpt)."""
    from elbi_agent.llm import Transcript

    transcript = Transcript(system=_SYSTEM)
    transcript.add_user_text(_prompt(trap, rows))
    step = client.step(transcript, ())
    text = step.text or ""
    # Normalise newlines + em/en dashes so committed excerpts stay clean.
    excerpt = text.strip().replace(chr(10), " ")
    excerpt = excerpt.replace(chr(0x2014), "-").replace(chr(0x2013), "-")
    excerpt = excerpt[:280]
    return _parse_verdict(text), excerpt


def refresh(
    traps: Sequence[Trap],
    client: LLMClient,
    *,
    model: str,
    captured_at: str,
    path: Path = BASELINE_PATH,
    on_trap: Callable[[int, int, Trap], None] | None = None,
    max_workers: int = 4,
) -> dict[str, dict[str, Any]]:
    """Re-query the bare model for every trap and rewrite the cached baseline file.

    The per-trap calls run concurrently (they are independent network requests), so a
    slow reasoning model does not serialise into minutes. ``on_trap(done, total, trap)``
    fires as each capture completes, for progress. The file is written with sorted keys,
    so its order is stable regardless of completion order.
    """

    def _capture(trap: Trap) -> tuple[str, dict[str, Any]]:
        verdict, excerpt = query_llm(client, trap, trap.data())
        return trap.id, {
            "verdict": verdict,
            "model": model,
            "captured_at": captured_at,
            "prompt_version": PROMPT_VERSION,
            "transcript_excerpt": excerpt,
        }

    baseline: dict[str, dict[str, Any]] = {}
    total = len(traps)
    workers = max(1, min(max_workers, total or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_capture, trap): trap for trap in traps}
        for done, future in enumerate(as_completed(futures), start=1):
            trap_id, entry = future.result()
            baseline[trap_id] = entry
            if on_trap is not None:
                on_trap(done, total, futures[future])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return baseline
