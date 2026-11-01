"""Replay the held-out v2 run through the shipped reconcile_effect, offline.

For each task, take the agent's last submit_answer call (its answer, data, x, y, adjustment
set, estimand, flags), run elbi_cli.mcp_server._reconcile_effect on the same rows, write the
point the way the prototype did (_format_like), and grade it with QRData's published rule.
No agent is run: this checks whether the shipped code reproduces the prototype's combination.

  RUNS=runs_heldout uv run python replay_shipped.py   # reads RUNS/{base,elbi2}/tNNN
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from elbi_cli.mcp_server import _reconcile_effect  # noqa: E402

src = (HERE / "upstream/benchmark/eval.py").read_text().split("data = json.load")[0]
ns: dict = {}
exec(src, ns)
calc_acc = ns["calc_acc"]
NUM = re.compile(r"-?\d+\.?\d*")

items = json.loads((HERE / "exam_heldout.json").read_text())["items"]
ROOT = HERE / os.environ.get("RUNS", "runs_heldout")


def ok(pred, it):
    return calc_acc([{"pred": pred, "answer": it["answer"], "meta_data": it["meta_data"]}]) == 1.0


def fmt_like(answer, value):
    a = answer.strip()
    pct = a.endswith("%")
    m = NUM.search((a[:-1] if pct else a).replace(",", ""))
    dec = len(m.group().split(".")[1]) if m and "." in m.group() else 0
    return f"{value * 100 if pct else value:.{dec}f}" + ("%" if pct else "")


def num(s):
    m = NUM.search(s.replace(",", ""))
    if not m:
        return None
    v = float(m.group())
    return v / 100 if s.strip().endswith("%") else v


def last_call(task):
    call = None
    for line in (task / "transcript.jsonl").read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") if isinstance(ev, dict) else None
        for c in (msg or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name", "").endswith("submit_answer"):
                call = c["input"]
    return call


def remap(path, task):
    """Data paths point at the task dir the run used; resolve through the view."""
    p = Path(path)
    if p.exists():
        return p
    alt = task.resolve() / "data" / p.name
    return alt if alt.exists() else p


rows_out = []
for i, it in enumerate(items, 1):
    task = ROOT / "elbi2" / f"t{i:03d}"
    pred = (task / "answers/answer.txt").read_text().strip() if (task / "answers/answer.txt").exists() else ""
    base = ROOT / "base" / f"t{i:03d}" / "answers/answer.txt"
    base_pred = base.read_text().strip() if base.exists() else ""
    rec = {"t": i, "proto": pred, "proto_ok": ok(pred, it), "base_ok": ok(base_pred, it), "shipped": pred,
           "path": "no effect check"}
    call = last_call(task) if (task / "transcript.jsonl").exists() else None
    if call and call.get("check") == "effect":
        a = num(call["answer"])
        df = pd.read_csv(remap(call["data_file"], task))
        given = list(call.get("given") or [])
        d = df[[call["x"], call["y"], *given]].apply(pd.to_numeric, errors="coerce").dropna()
        x0, x1 = float(call.get("x0", 0.0)), float(call.get("x1", 1.0))
        with threadpool_limits(limits=4):
            out = json.loads(_reconcile_effect(
                d.to_dict("records"), call["x"], call["y"], a, given, call.get("estimand", "ate").lower(),
                x0, x1, None, None, bool(call.get("procedure_specified")), bool(call.get("confirm"))))
        rec["dr"] = out["elbi_estimates"].get("doubly_robust")
        rec["path"] = out.get("combined") or out.get("note") or out.get("problem", "")[:40]
        if out["final"] is not None:
            rec["shipped"] = fmt_like(call["answer"], out["final"]["estimate"])
        else:
            rec["shipped"] = call["answer"]
            rec["path"] = "WITHHELD: " + rec["path"]  # agent would have had to confirm or change
        rec["agent"] = call["answer"]
        rec["agent_ok"] = ok(call["answer"], it)
    rec["shipped_ok"] = ok(rec["shipped"], it)
    rows_out.append(rec)
    flag = "" if rec["shipped_ok"] == rec["proto_ok"] else ("  <-- FIXED" if rec["shipped_ok"] else "  <-- BROKE")
    print(f"t{i:03d} gold {it['answer'][:9]:9} proto {pred[:9]:9} {'Y' if rec['proto_ok'] else '.'} "
          f"shipped {rec['shipped'][:9]:9} {'Y' if rec['shipped_ok'] else '.'}  {rec['path'][:45]}{flag}", flush=True)

n = len(rows_out)
print(f"\nbase {sum(r['base_ok'] for r in rows_out)}/{n}, prototype v2 {sum(r['proto_ok'] for r in rows_out)}/{n}, "
      f"shipped replay {sum(r['shipped_ok'] for r in rows_out)}/{n}")
diff = [r for r in rows_out if r["shipped_ok"] != r["proto_ok"]]
print(f"tasks where shipped and prototype grade differently: {len(diff)}")
withheld = [r["t"] for r in rows_out if r["path"].startswith("WITHHELD")]
print(f"withheld by shipped range check (agent's answer kept above; would need confirm): {withheld}")
(HERE / os.environ.get("OUT", "replay_shipped.json")).write_text(json.dumps(rows_out, indent=1, default=float))
