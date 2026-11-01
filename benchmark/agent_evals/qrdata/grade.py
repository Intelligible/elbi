"""Grade the pilot with QRData's own rule (upstream/benchmark/eval.py: numbers within 3%
relative, choices by prefix match), paired base vs elbi, plus cost and how elbi's check acted.

  RUNS=runs_pilot python grade.py
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

items = json.loads((HERE / os.environ.get("EXAM", "exam_pilot.json")).read_text())["items"]
ROOT = HERE / os.environ.get("RUNS", "runs_pilot")


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


res = {}
for arm in ("base", "elbi"):
    res[arm] = []
    for i, it in enumerate(items, 1):
        d = ROOT / arm / f"t{i:03d}"
        a = d / "answers" / "answer.txt"
        pred = a.read_text().strip() if a.exists() else ""
        ok = calc_acc([{"pred": pred, "answer": it["answer"], "meta_data": it["meta_data"]}]) == 1.0
        log = d / "elbi_check.json"
        chk = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        tr = (d / "transcript.jsonl").read_text() if (d / "transcript.jsonl").exists() else ""
        res[arm].append({"ok": ok, "pred": pred, "ran": d.exists(), "usd": cost(d / "transcript.jsonl"),
                         "calls": tr.count('"name":"mcp__check__submit_answer"'),
                         "refused": len(re.findall(r'wrote\\*"\s*:\s*null', tr)), "check": chk[-1]["check"] if chk else None,
                         "overridden": bool(chk and chk[-1]["problems"])})

both = [i for i in range(len(items)) if res["base"][i]["ran"] and res["elbi"][i]["ran"]]
print(f"{len(both)} tasks run in both arms")
for i in both:
    it = items[i]
    b, e = res["base"][i], res["elbi"][i]
    flag = "" if b["ok"] == e["ok"] else ("  <- elbi better" if e["ok"] else "  <- elbi worse")
    print(f"t{i+1:03d} {it['stratum']:28} gold {it['answer'][:10]:10} base {b['pred'][:10]:10} {'Y' if b['ok'] else '.'} | "
          f"elbi {e['pred'][:10]:10} {'Y' if e['ok'] else '.'} calls {e['calls']} refused {e['refused']} check {e['check']}{flag}")

for arm in res:
    r = [res[arm][i] for i in both]
    print(f"{arm}: accuracy {sum(x['ok'] for x in r)}/{len(r)}, cost ${sum(x['usd'] for x in r):.2f}")
for s in sorted({items[i]["stratum"] for i in both}):
    ix = [i for i in both if items[i]["stratum"] == s]
    print(f"  {s:28} base {sum(res['base'][i]['ok'] for i in ix)}/{len(ix)}  elbi {sum(res['elbi'][i]['ok'] for i in ix)}/{len(ix)}")
d = [res["elbi"][i]["ok"] - res["base"][i]["ok"] for i in both]
if d:
    rnd = random.Random(0)
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(10000))
    print(f"elbi - base accuracy: {sum(d)/len(d):+.3f} (95% CI {bs[250]:+.3f}, {bs[9750]:+.3f}); "
          f"elbi better on {d.count(1)}, worse on {d.count(-1)}")
e = [res["elbi"][i] for i in both]
print(f"elbi arm: used submit_answer {sum(x['calls'] > 0 for x in e)}/{len(e)}; refused at least once {sum(x['refused'] > 0 for x in e)}; "
      f"final kept over a problem {sum(x['overridden'] for x in e)}; checks {[x['check'] for x in e].count('effect')} effect, "
      f"{[x['check'] for x in e].count('dependence')} dependence")
