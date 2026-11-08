"""Build the two QRData exams from a QRData checkout (upstream/benchmark/QRData.json).

exam_pilot.json: 40 questions stratified by (first keyword, question type), seed 0.
exam_heldout.json: every Causality/numerical question not in the pilot (49).
The question ids are in exam_*_qids.json; the build checks against them.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = json.loads((HERE / "upstream/benchmark/QRData.json").read_text())
QUOTA = {("Causality", "multiple_choice"): 10, ("Causality", "numerical"): 10,
         ("Statistics", "numerical"): 10, ("Statistics", "multiple_choice"): 5,
         ("Social science", "numerical"): 5}

rng = random.Random(0)
items = []
for stratum, k in QUOTA.items():
    pool = [i for i, q in enumerate(Q) if (q["meta_data"]["keywords"][0], q["meta_data"]["question_type"]) == stratum]
    for i in sorted(rng.sample(pool, k)):
        items.append({"qid": i, "stratum": "/".join(stratum), **Q[i]})
(HERE / "exam_pilot.json").write_text(json.dumps({"seed": 0, "quota": {"/".join(s): k for s, k in QUOTA.items()}, "items": items}, indent=1))
pilot = {x["qid"] for x in items}
held = [
    {"qid": i, "stratum": "Causality/numerical", **q}
    for i, q in enumerate(Q)
    if q["meta_data"]["keywords"][0] == "Causality"
    and q["meta_data"]["question_type"] == "numerical"
    and i not in pilot
]
(HERE / "exam_heldout.json").write_text(json.dumps({"note": "all Causality/numerical questions not in the pilot", "items": held}, indent=1))
for name, built in (("pilot", items), ("heldout", held)):
    want = json.loads((HERE / f"exam_{name}_qids.json").read_text())
    assert [x["qid"] for x in built] == want, f"{name} exam differs from the one that was run"
print(len(items), len(held))
