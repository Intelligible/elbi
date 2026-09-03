"""Command-line entry point: run the benchmark, or refresh the bare-LLM baseline.

``python -m benchmark.run`` runs every trap against the *cached* baseline, writes the
report, prints a summary, and exits non-zero if any trap fails (gate correctness).
``python -m benchmark.run --refresh-llm`` re-queries a bare model and rewrites the
cached baseline for review; it never runs on a PR. It defaults to the product's Claude
model; ``--provider litellm --model openai/gpt-5`` (etc.) benchmarks the gates against a
different bare model. The provider's own API key must be set in the environment.
``python -m benchmark.run --dump-data DIR`` materialises every trap's dataset to CSV
for inspection. ``python -m benchmark.run --regenerate-data`` rewrites each synthetic
trap's committed data.jsonl from its ``generated_by`` generator (an authoring step).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

from .harness import run_benchmark
from .llm_baseline import BASELINE_PATH, load_baseline, refresh
from .loader import BENCHMARK_DIR, generate_rows, load_traps, manifest_paths
from .report import render_markdown, to_json

REPORT_DIR = BENCHMARK_DIR / "report"
#: The bare-LLM baseline model when none is passed. Pinned explicitly (rather than
#: inheriting the agent package's default) so the benchmark baseline stays stable.
DEFAULT_MODEL = "claude-sonnet-5"


def _write_report(results: list[object], report_dir: Path) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    doc = to_json(results)  # type: ignore[arg-type]
    (report_dir / "report.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "report.md").write_text(
        render_markdown(results) + "\n",  # type: ignore[arg-type]
        encoding="utf-8",
    )
    return doc


def _run(report_dir: Path) -> int:
    traps = load_traps()
    baseline = load_baseline()

    def _progress(i: int, total: int, trap: Any) -> None:
        # Progress goes to stderr so it stays out of the report printed on stdout.
        print(f"[{i}/{total}] verifying {trap.id} ...", file=sys.stderr, flush=True)

    results = run_benchmark(traps, baseline, on_trap=_progress)
    _write_report(list(results), report_dir)
    print(render_markdown(results))
    print(f"\nreport written to {report_dir}")
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\nFAILED: {len(failed)} trap(s) did not return the expected verdict:")
        for r in failed:
            print(
                f"  - {r.trap.id} ({r.trap.fixture}): got {r.verdict!r}, "
                f"expected {r.trap.expected_verdict!r}"
            )
        return 1
    return 0


def _make_client(provider: str, model: str | None) -> tuple[Any, str]:
    """Build the bare-model client for the chosen provider; return (client, model id).

    ``anthropic`` uses the product's default Claude model unless ``model`` overrides it.
    ``litellm`` needs an explicit ``provider/model`` string (e.g. ``openai/gpt-5``) and
    reaches any LiteLLM-supported backend.
    """
    if provider == "anthropic":
        from elbi_agent.anthropic_client import AnthropicClient

        chosen = model or DEFAULT_MODEL
        return AnthropicClient(model=chosen), chosen
    if provider == "litellm":
        if not model:
            raise SystemExit("--provider litellm requires --model, e.g. openai/gpt-5")
        from elbi_agent.litellm_client import LiteLLMClient

        return LiteLLMClient(model=model), model
    raise SystemExit(f"unknown provider {provider!r}")


def _refresh(provider: str, model: str | None) -> int:
    # The anthropic default needs ANTHROPIC_API_KEY; a litellm backend uses whatever key
    # env var its provider expects, so only the default path is guarded here.
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; skipping the bare-LLM refresh.")
        return 2
    client, model_id = _make_client(provider, model)
    captured_at = datetime.date.today().isoformat()
    traps = load_traps()
    print(
        f"querying {len(traps)} traps against bare {model_id} (concurrent); "
        "a reasoning model can take a while per trap ...",
        file=sys.stderr,
        flush=True,
    )

    def _progress(done: int, total: int, trap: Any) -> None:
        print(f"[{done}/{total}] captured {trap.id}", file=sys.stderr, flush=True)

    baseline = refresh(
        traps, client, model=model_id, captured_at=captured_at, on_trap=_progress
    )
    print(
        f"refreshed {len(baseline)} bare-LLM captures with {model_id} "
        f"at {BASELINE_PATH}; review the diff before committing."
    )
    return 0


def _dump_data(out_dir: Path) -> int:
    """Materialise every trap's rows to ``<out_dir>/<id>.csv`` for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traps = load_traps()
    for trap in traps:
        rows = trap.data()
        path = out_dir / f"{trap.id}.csv"
        if rows:
            # union of keys in first-seen order, so a ragged dataset does not raise
            fieldnames = list({k: None for row in rows for k in row})
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(rows)
        print(f"  {trap.id}: {len(rows)} rows -> {path}")
    print(f"materialized {len(traps)} datasets to {out_dir}")
    return 0


def _regenerate_data() -> int:
    """Rewrite each synthetic trap's committed data.jsonl from its `generated_by` block.

    Generators are authoring tools: run this after adding or changing one, then commit
    the regenerated data.jsonl. Real (file-backed, no `generated_by`) traps are skipped.
    """
    count = 0
    for path in manifest_paths():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "generated_by" not in manifest or "file" not in manifest["data"]:
            continue
        rows = generate_rows(manifest)
        out = path.parent / manifest["data"]["file"]
        out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"  {path.parent.name}: {len(rows)} rows -> {out}")
        count += 1
    print(f"regenerated {count} synthetic dataset(s); review the diff and commit.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the run, refresh, dump, or regenerate path."""
    parser = argparse.ArgumentParser(
        description="Statistical-pitfall benchmark harness."
    )
    parser.add_argument(
        "--refresh-llm",
        action="store_true",
        help="re-query a bare model and rewrite the cached baseline (needs a key)",
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "litellm"),
        default="anthropic",
        help="bare-LLM provider for --refresh-llm (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model for --refresh-llm; a Claude id for anthropic (default: "
        f"{DEFAULT_MODEL}), or a 'provider/model' string for litellm (openai/gpt-5)",
    )
    parser.add_argument(
        "--dump-data",
        type=Path,
        nargs="?",
        const=REPORT_DIR / "data",
        default=None,
        help="materialise every trap's dataset to CSV (default dir: report/data)",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="rewrite each synthetic trap's committed data.jsonl from its generator",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="where to write report.md and report.json",
    )
    args = parser.parse_args(argv)
    if args.refresh_llm:
        return _refresh(args.provider, args.model)
    if args.regenerate_data:
        return _regenerate_data()
    if args.dump_data is not None:
        return _dump_data(args.dump_data)
    return _run(args.report_dir)


if __name__ == "__main__":
    sys.exit(main())
