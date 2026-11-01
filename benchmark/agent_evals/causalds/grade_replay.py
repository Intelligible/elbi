"""Official CausalDSScore for rounds 1-2: base, the recorded elbi arm, and elbi_shipped (the
recorded submissions replayed through the shipped reconcile_effect by replay_shipped.py).

  python grade_replay.py      # with CausalDS's Python (it imports causalds)
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = Path(os.environ.get("CAUSALDS_DIR", HERE / "upstream"))
sys.path.insert(0, str(C))
from causalds.exam_builder import Exam, ExamItem  # noqa: E402
from causalds.grader import grade_exam  # noqa: E402

BENCH = C / "data" / "benchmark" / "main"
FIELDS = set(ExamItem.__dataclass_fields__)
items = [ExamItem(**{k: v for k, v in it.items() if k in FIELDS})
         for it in json.loads((HERE / "exam_official.json").read_text())["items"]]
KEYS = ["overall_causal_ds_score", "overall_pass_rate", "overall_causal_ds_nrel_component"]

rows = {}
for r in (1, 2):
    for arm in ("base", "elbi", "elbi_shipped"):
        root = HERE / f"runs_official_r{r}" / arm
        if not root.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            ans = Path(tmp)
            n = 0
            for i, ex in enumerate(items, 1):
                src = root / f"t{i:03d}" / "answers" / ex.answer_filename()
                if src.exists():
                    shutil.copy(src, ans / ex.answer_filename())
                    n += 1
            exam = Exam(exam_id="official", items=items, benchmark_dir=BENCH, seed=0)
            s = grade_exam(exam, ans, BENCH).to_dict()["summary"]
        rows[(r, arm)] = s
        print(f"round {r} {arm:12} answered {n:3}/100  CausalDSScore {s.get(KEYS[0], float('nan')):.4f}  "
              f"pass rate {s.get(KEYS[1], float('nan')):.3f}  "
              f"interval mean nIS {s.get('overall_causal_ds_nrel_component_parts', {}).get('interval_mean', float('nan')):.3f}")
for arm in ("base", "elbi", "elbi_shipped"):
    v = [rows[(r, arm)][KEYS[0]] for r in (1, 2) if (r, arm) in rows and KEYS[0] in rows[(r, arm)]]
    if v:
        print(f"{arm:12} mean CausalDSScore over {len(v)} rounds: {sum(v)/len(v):.4f}  (rounds: {', '.join(f'{x:.4f}' for x in v)})")
