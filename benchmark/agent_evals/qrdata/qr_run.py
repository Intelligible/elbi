"""QRData pilot: headless Claude Code, base vs elbi (check_server.submit_answer).

Each task runs in a fresh directory holding only the question's data files (data/) and an
empty answers/ folder. Same prompt in both arms; the elbi arm adds one MCP tool and one
sentence. Grading: grade.py with QRData's own accuracy rule.

  RUNS=runs_pilot ARMS="base elbi" WORKERS=4 python qr_run.py
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
# The agent's Python (pandas, numpy, scipy, scikit-learn, statsmodels, ...); CausalDS's venv
# serves. Prepended to PATH for the agent and used to re-run its scripts.
PY_ENV = Path(os.environ.get("AGENT_BIN", HERE.parent / "causalds" / "upstream" / ".venv" / "bin"))
# Python for the check servers: needs mcp, pandas and scikit-learn.
ELBI_PY = os.environ.get("ELBI_PY", sys.executable)
EXAM = HERE / os.environ.get("EXAM", "exam_pilot.json")
MODEL = os.environ.get("ARM_MODEL", "claude-opus-5-5")
ARMS = os.environ.get("ARMS", "base elbi").split()
WORKERS = int(os.environ.get("WORKERS", "4"))
TIMEOUT_S = 30 * 60

RULES = """You are an expert data scientist answering a quantitative question from data.
You have Python 3.11 (`python3`) with: pandas, numpy, scipy, scikit-learn, statsmodels,
pyarrow, matplotlib, seaborn, networkx, xgboost. Do not assume unlisted packages are installed.
Analyze the data with code; do not guess.

"""
ELBI_HINT = ("\nYou have elbi's tool mcp__check__submit_answer. Write your final answer with "
             "submit_answer instead of writing the answer file yourself.\n")


def prompt_for(item, home):
    files = ", ".join(f"data/{f}" for f in item["data_files"])
    p = (RULES + f"Data description: {item['data_description']}\n\nData files (in {home}): {files}\n\n"
         f"Question: {item['question']}\n")
    if item["meta_data"]["question_type"] == "multiple_choice":
        p += f"\nOptions: {item['meta_data']['multiple_choices']}\n"
        fmt = "the chosen option exactly as listed (e.g. the letter), nothing else"
    else:
        fmt = "the number only, rounded as the question asks, nothing else"
    return p + f"\nWrite your final answer to {home}/answers/answer.txt: {fmt}.\n"


def run_one(arm, idx, item):
    home = HERE / os.environ.get("RUNS", "runs_pilot") / arm / f"t{idx:03d}"
    done = home / "answers" / "answer.txt"
    if done.exists():
        return {"arm": arm, "task": idx, "status": "cached"}
    if home.exists():
        shutil.rmtree(home)
    (home / "answers").mkdir(parents=True)
    (home / "data").mkdir()
    for f in item["data_files"]:
        shutil.copy(HERE / "data" / f, home / "data" / f)
    prompt = prompt_for(item, home)
    tools = ["Bash", "Read", "Write", "Edit"]
    mcp = {"mcpServers": {}}
    if arm in ("elbi", "elbi2"):
        prompt += ELBI_HINT
        server = {"elbi": "check_server.py", "elbi2": "check_server_v2.py"}[arm]
        mcp = {"mcpServers": {"check": {
            "type": "stdio", "command": ELBI_PY, "args": [str(HERE / server)],
            "env": {"CHECK_HOME": str(home), "CHECK_PY": str(PY_ENV / "python"), "OMP_NUM_THREADS": "2"}}}}
        tools.append("mcp__check")
    (home / "mcp.json").write_text(json.dumps(mcp))
    env = dict(os.environ, PATH=f"{PY_ENV}:{os.environ['PATH']}")
    cmd = ["claude", "-p", prompt, "--model", MODEL, "--output-format", "stream-json", "--verbose",
           "--strict-mcp-config", "--mcp-config", str(home / "mcp.json"), "--allowedTools", *tools,
           "--disallowedTools", "WebFetch", "WebSearch", "Task", "--setting-sources", "project",
           "--no-session-persistence"]
    t0 = time.time()
    try:
        with open(home / "transcript.jsonl", "w") as out, open(home / "stderr.log", "w") as err:
            proc = subprocess.run(cmd, cwd=home, stdout=out, stderr=err, env=env, timeout=TIMEOUT_S)
        status = f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    status += "" if done.exists() else ", no answer"
    return {"arm": arm, "task": idx, "status": status, "secs": round(time.time() - t0)}


if __name__ == "__main__":
    items = json.loads(EXAM.read_text())["items"]
    only = {int(x) for x in sys.argv[1:]}
    jobs = [(a, i, it) for i, it in enumerate(items, 1) for a in ARMS if not only or i in only]
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for r in pool.map(lambda j: run_one(*j), jobs):
            print(json.dumps(r), flush=True)
