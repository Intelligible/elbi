"""Build exam_official.json: CausalDS's released 100-task realistic exam
(upstream/data/benchmark/main/exam.json), with each item told where its scene lives."""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = Path(os.environ.get("CAUSALDS_DIR", HERE / "upstream"))
bench = C / "data" / "benchmark" / "main"
exam = json.loads((bench / "exam.json").read_text())
for item in exam["items"]:
    item["_bench"] = str(bench)
(HERE / "exam_official.json").write_text(json.dumps(exam, indent=1))
print(len(exam["items"]))
