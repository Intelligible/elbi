"""Replay the CausalDS official rounds through the SHIPPED reconcile_effect, offline.

For every elbi-arm task whose agent wrote its answer through the prototype's submit_answer,
take that call's inputs (treatment, outcome, adjustment set, x0, x1, estimate, interval), run
elbi_cli.mcp_server._reconcile_effect on the rows the prototype used (the calibration file's
true values when the scene has one, else the data), and write its final answer in place of
the prototype's. Every other answer is the recorded one. Then grade base, the recorded elbi
arm, and the replayed arm with CausalDS's own grade_exam. No agent is run.

  uv run python replay_shipped.py      # with elbi's Python; then grade_replay.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
from elbi_cli.mcp_server import _reconcile_effect  # noqa: E402

items = json.loads((HERE / "exam_official.json").read_text())["items"]


def submit_call(task: Path):
    call, results = None, {}
    for line in (task / "transcript.jsonl").read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (ev.get("message") or {}).get("content") if isinstance(ev, dict) else None
        for c in content if isinstance(content, list) else []:
            if c.get("type") == "tool_use" and c.get("name", "").endswith("submit_answer"):
                call = c
            elif c.get("type") == "tool_result":
                results[c.get("tool_use_id")] = c.get("content")
    if call is None:
        return None, None
    res = results.get(call["id"])
    text = res[0].get("text", "") if isinstance(res, list) and res else str(res or "")
    return call["input"], text


def replay(task: Path, inp: dict) -> tuple[dict | None, str]:
    if not inp.get("identifiable", True) or inp.get("ci_lower") is None:
        return None, "not identifiable or no interval: recorded answer kept"
    scene = next((task / "scenes").iterdir())
    calib = scene / "calibration.parquet"
    rows = pd.read_parquet(calib if calib.exists() else scene / "data.parquet")
    cols = [inp["treatment"], inp["outcome"], *(inp.get("adjust") or [])]
    if any(c not in rows.columns for c in cols):
        return None, "variables not in the rows: recorded answer kept"
    d = rows[cols].apply(pd.to_numeric, errors="coerce").dropna()
    with threadpool_limits(limits=4):
        out = json.loads(_reconcile_effect(
            d.to_dict("records"), inp["treatment"], inp["outcome"], float(inp["ate"]),
            list(inp.get("adjust") or []), "ate", float(inp.get("x0", 0.0)), float(inp.get("x1", 1.0)),
            float(inp["ci_lower"]), float(inp["ci_upper"]), False, False))
    if out.get("final") is None:
        # Withheld: outside every elbi model. The agent would have had to confirm (kept as
        # its own answer) or change it; replay assumes it confirmed.
        return {"ate": float(inp["ate"]), "ci_lower": float(inp["ci_lower"]),
                "ci_upper": float(inp["ci_upper"]), "alpha": 0.05}, "WITHHELD (assumed confirmed)"
    f = out["final"]
    return {"ate": f["estimate"], "ci_lower": f["ci_lower"], "ci_upper": f["ci_upper"], "alpha": 0.05,
            "method": "agent estimate, combined with elbi's"}, out.get("combined", "")


for r in (1, 2):
    elbi = HERE / f"runs_official_r{r}" / "elbi"
    out_root = HERE / f"runs_official_r{r}" / "elbi_shipped"
    print(f"round {r}", flush=True)
    for i, ex in enumerate(items, 1):
        task = elbi / f"t{i:03d}"
        src_dir = task / "answers"
        dst_dir = out_root / f"t{i:03d}" / "answers"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob("*") if src_dir.exists() else []:
            shutil.copy(f, dst_dir / f.name)
        if not (task / "transcript.jsonl").exists():
            continue
        inp, _ = submit_call(task)
        if inp is None:
            continue
        ans, how = replay(task, inp)
        target = Path(inp["answer_file"]).name
        old = json.loads((src_dir / target).read_text()) if (src_dir / target).exists() else {}
        if ans is not None:
            (dst_dir / target).write_text(json.dumps(ans))
        fmt = lambda a: "-" if not a or a.get("ci_lower") is None else f"{a['ate']:.4g} [{a['ci_lower']:.4g}, {a['ci_upper']:.4g}]"
        print(f"  t{i:03d} agent {inp.get('ate')} [{inp.get('ci_lower')}, {inp.get('ci_upper')}]  "
              f"prototype {fmt(old)}  shipped {fmt(ans)}  {how}", flush=True)
