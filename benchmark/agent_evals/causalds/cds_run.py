"""Minimal CausalDS A/B: headless Claude Code with and without elbi's front door.

Each task runs in a fresh directory holding only that scene's public files and an empty
answers/ folder, with CausalDS's own task prompt (exam_builder.build_single_task_prompt)
and the rules from its agent config. Their harness's /workspace paths become this run's
directory, the same in both arms. Arm "facts" adds one MCP tool, plan_analysis (facts
only), pointed at the scene's data.parquet. Grading: grade.py with CausalDS's grader.
"""

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# A clone of github.com/andleb/causalds with its own venv (see README).
C = Path(os.environ.get("CAUSALDS_DIR", HERE / "upstream"))
PLAN = HERE.parent / "prototype" / "plan_server.py"
# Python for the elbi MCP server: needs elbi_core, mcp, pandas and scikit-learn.
ELBI_PY = os.environ.get("ELBI_PY", sys.executable)
sys.path.insert(0, str(C))
from causalds.exam_builder import ExamItem, build_single_task_prompt  # noqa: E402

SCENES = C / "data/benchmark_runs/dry/workspace/scenes"
EXAM = HERE / os.environ.get("EXAM", "exam_official.json")
MODEL = os.environ.get("ARM_MODEL", "claude-opus-5-5")
ARMS = os.environ.get("ARMS", "base facts").split()
WORKERS = int(os.environ.get("WORKERS", "4"))
TIMEOUT_S = 30 * 60
FIELDS = set(ExamItem.__dataclass_fields__)

# CausalDS's benchmark_agent.yaml system rules, minus the harness-specific "exactly one
# bash call per response" and heredoc conventions.
RULES = """You are an expert data scientist solving causal reasoning tasks.
You have Python 3.11 (`python3`) with: pandas, numpy, scipy, scikit-learn, statsmodels,
pyarrow, matplotlib, seaborn, networkx, xgboost. Do not assume unlisted packages are installed.
Measurement columns sharing a stem (e.g. `variable1`, `variable2`, ...) measure the same single
conceptual variable and are not separate causal variables.
The grader reads only the answers folder named below. When done, verify your answer file exists
at the requested path.

"""


def run_one(arm: str, idx: int, item: dict) -> dict:
    home = HERE / os.environ.get("RUNS", "runs") / arm / f"t{idx:03d}"
    ans_name = ExamItem(**{k: v for k, v in item.items() if k in FIELDS}).answer_filename()
    done = home / "answers" / ans_name
    if done.exists():
        return {"arm": arm, "task": idx, "status": "cached"}
    if home.exists():
        shutil.rmtree(home)
    (home / "answers").mkdir(parents=True)
    scene = item["scene_id"]
    if "_bench" in item:
        item["_bench"] = str(C / "data" / "benchmark" / Path(item["_bench"]).name)
        # A generated scene: story plus the files of this task's observation variant.
        dst = home / "scenes" / scene
        shutil.copytree(Path(item["_bench"]) / "scenes" / scene / "variants" / item["observation_variant"], dst)
        shutil.copy(Path(item["_bench"]) / "scenes" / scene / "story.md", dst / "story.md")
        (dst / "tasks.json").unlink(missing_ok=True)  # the task catalog is not given to the agent
    else:
        shutil.copytree(SCENES / scene, home / "scenes" / scene)
    ex = ExamItem(**{k: v for k, v in item.items() if k in FIELDS})
    prompt = RULES + build_single_task_prompt(ex, task_index=idx).replace("/workspace/", f"{home}/")

    mcp = {"mcpServers": {}}
    tools = ["Bash", "Read", "Write", "Edit"]
    if arm == "effect_hint":
        # How an elbi user works: they ask the agent to use elbi. One sentence, nothing else.
        prompt += ("\n\nYou have elbi's data-science tools (mcp__plan__plan_analysis and "
                   "mcp__plan__estimate_effect). Use them to check your estimate before you answer.\n")
    if arm == "submit_hint":
        prompt += ("\n\nYou have elbi's data-science tools (mcp__plan__plan_analysis and "
                   "mcp__plan__submit_answer). Write your final answer with submit_answer.\n")
    if arm == "elbi":
        # Same tools as submit_hint; the hint says when submit_answer applies, since the full
        # exam has many task types.
        prompt += ("\n\nYou have elbi's data-science tools (mcp__plan__plan_analysis and "
                   "mcp__plan__submit_answer). If the task asks for an average effect with a "
                   "confidence interval, write your final answer with submit_answer.\n")
    if arm == "claim":
        prompt += ("\n\nYou have elbi's data-science tools (mcp__plan__plan_analysis and "
                   "mcp__plan__submit_effect_claim). Write your final answer with submit_effect_claim.\n")
    if arm in ("facts", "effect", "effect_hint", "submit_hint", "elbi", "claim"):
        data = home / "scenes" / scene / "data.parquet"
        if data.exists():
            mcp = {"mcpServers": {"plan": {
                "type": "stdio",
                "command": ELBI_PY,
                "args": [str(PLAN)],
                "env": {"PLAN_DATA_CSV": str(data), "PLAN_PRESCRIBE": "0",
                        "PLAN_SUBMIT": "1" if arm in ("submit_hint", "elbi") else "0", "PLAN_CLAIM": "1" if arm == "claim" else "0", "OMP_NUM_THREADS": "2"},
            }}}
            tools.append("mcp__plan")
    (home / "mcp.json").write_text(json.dumps(mcp))

    env = dict(os.environ, PATH=f"{C / '.venv/bin'}:{os.environ['PATH']}")
    cmd = ["claude", "-p", prompt, "--model", MODEL,
           "--output-format", "stream-json", "--verbose",
           "--strict-mcp-config", "--mcp-config", str(home / "mcp.json"),
           "--allowedTools", *tools,
           "--disallowedTools", "WebFetch", "WebSearch", "Task",
           "--setting-sources", "project", "--no-session-persistence"]
    t0 = time.time()
    try:
        with open(home / "transcript.jsonl", "w") as out, open(home / "stderr.log", "w") as err:
            proc = subprocess.run(cmd, cwd=home, stdout=out, stderr=err, env=env, timeout=TIMEOUT_S)
        status = f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    status += "" if done.exists() else ", no answer"
    return {"arm": arm, "task": idx, "status": status, "secs": round(time.time() - t0)}


def main() -> None:
    items = json.loads(EXAM.read_text())["items"]
    only = {int(x) for x in sys.argv[1:]}
    jobs = [(arm, i, it) for i, it in enumerate(items, 1) for arm in ARMS if not only or i in only]
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for r in pool.map(lambda j: run_one(*j), jobs):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
