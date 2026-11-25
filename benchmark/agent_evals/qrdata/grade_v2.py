"""Grade the held-out run (prereg_v2.md): arms base, elbi (v1 tool), elbi2 (v2 tool), with
QRData's published rule. Primary: elbi2 - base paired accuracy. Also elbi2 - elbi, elbi - base,
the mechanism (answers the v2 combination changed, and whether for better), and, exploratory,
what writing elbi's doubly robust estimate alone would have scored.

  RUNS=runs_heldout EXAM=exam_heldout.json python grade_v2.py
"""

import json
import os
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "upstream/benchmark/eval.py").read_text().split("data = json.load")[0]
ns = {}
exec(src, ns)  # extract_first_number, calc_acc exactly as published
calc_acc = ns["calc_acc"]

items = json.loads((HERE / os.environ.get("EXAM", "exam_heldout.json")).read_text())["items"]
ROOT = HERE / os.environ.get("RUNS", "runs_heldout")
ARMS = [a for a in ("base", "elbi", "elbi2") if (ROOT / a).exists()]


def ok(pred, it):
    return calc_acc([{"pred": pred, "answer": it["answer"], "meta_data": it["meta_data"]}]) == 1.0


def cost(tr):
    usd = 0.0
    if tr.exists():
        for line in tr.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "result":
                usd += ev.get("total_cost_usd") or 0.0
    return usd


def fmt_like(answer, value):
    a = answer.strip()
    pct = a.endswith("%")
    m = re.search(r"-?\d+\.?\d*", (a[:-1] if pct else a).replace(",", ""))
    dec = len(m.group().split(".")[1]) if m and "." in m.group() else 0
    return f"{value * 100 if pct else value:.{dec}f}" + ("%" if pct else "")


res = {}
for arm in ARMS:
    res[arm] = []
    for i, it in enumerate(items, 1):
        d = ROOT / arm / f"t{i:03d}"
        a = d / "answers" / "answer.txt"
        pred = a.read_text().strip() if a.exists() else ""
        log = d / "elbi_check.json"
        chk = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        tr = (d / "transcript.jsonl").read_text() if (d / "transcript.jsonl").exists() else ""
        last = chk[-1] if chk else {}
        rec = {"ok": ok(pred, it), "pred": pred, "ran": a.exists(), "usd": cost(d / "transcript.jsonl"),
               "used": '"name":"mcp__check__submit_answer"' in tr,
               "refused": len(re.findall(r'wrote\\*"\s*:\s*null', tr)) > 0, "check": last.get("check")}
        comb = last.get("combined")
        if comb:
            rec["combined"] = (ok(comb["yours"], it), ok(comb["written"], it))
        dr = (last.get("effect") or {}).get("elbi_estimates", {}).get("doubly_robust")
        if dr is not None and last.get("answer"):
            rec["dr_alone_ok"] = ok(fmt_like(last["answer"], dr), it)
        res[arm].append(rec)

both = [i for i in range(len(items)) if all(res[a][i]["ran"] for a in ARMS)]
print(f"{len(both)} tasks run in all of {ARMS}")
for i in both:
    it = items[i]
    print(f"t{i+1:03d} gold {it['answer'][:9]:9} " + " | ".join(
        f"{a} {res[a][i]['pred'][:9]:9} {'Y' if res[a][i]['ok'] else '.'}"
        + (f" {'C' + ''.join('Y' if v else '.' for v in res[a][i]['combined'])}" if res[a][i].get('combined') else "")
        + (" R" if res[a][i]["refused"] else "") for a in ARMS) + f"  {it['question'][:50]!r}")
for a in ARMS:
    r = [res[a][i] for i in both]
    print(f"{a}: accuracy {sum(x['ok'] for x in r)}/{len(r)}, cost ${sum(x['usd'] for x in r):.2f}"
          + (f"; used tool {sum(x['used'] for x in r)}, refused {sum(x['refused'] for x in r)}, effect checks {sum(x['check'] == 'effect' for x in r)}" if a != "base" else ""))


def paired(x, y):
    d = [int(res[x][i]["ok"]) - int(res[y][i]["ok"]) for i in both]
    rnd = random.Random(0)
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(10000))
    print(f"{x} - {y}: {sum(d)/len(d):+.3f} (95% CI {bs[250]:+.3f}, {bs[9750]:+.3f}); better {d.count(1)}, worse {d.count(-1)}")


for x, y in (("elbi2", "base"), ("elbi2", "elbi"), ("elbi", "base")):
    if x in res and y in res:
        paired(x, y)
if "elbi2" in res:
    c = [res["elbi2"][i]["combined"] for i in both if res["elbi2"][i].get("combined")]
    print(f"elbi2 combination applied {len(c)}x: agent's own right {sum(a for a, _ in c)}, written right {sum(b for _, b in c)}; "
          f"fixed {sum(b and not a for a, b in c)}, broke {sum(a and not b for a, b in c)}")
    dr = [res["elbi2"][i] for i in both if "dr_alone_ok" in res["elbi2"][i]]
    print(f"exploratory: elbi's doubly robust estimate alone right {sum(x['dr_alone_ok'] for x in dr)}/{len(dr)} "
          f"where computed (agent's final there: {sum(x['ok'] for x in dr)}/{len(dr)})")
