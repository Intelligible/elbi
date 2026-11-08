"""elbi's answer check for point answers, v2 (QRData). One tool, submit_answer.

v2 changes from v1 (check_server_v1_frozen.py), both decided after the pilot and before the
held-out runs: (1) effect estimates come from effect_dr (binary-outcome spline fixed; plus a
cross-fitted doubly robust estimate); (2) when the question asks for the effect itself rather
than the output of a named procedure, elbi writes the equal-weight combination of the agent's
estimate and its doubly robust estimate: the point-answer form of the CausalDS rule, where two
independent analyses equally believed are combined.

v1 description follows.

The CausalDS result came from two things: elbi analysed the question independently, and
elbi wrote the final answer, so the second analysis could not be skipped. For a single
number or a choice there is no interval to combine, so here elbi re-derives what it can and
refuses to write an answer it cannot reproduce or that its own analysis contradicts, until
the agent has looked at the disagreement:

  reproduce   re-run the agent's analysis script from scratch on the original files and
              compare what it prints with the submitted answer (always)
  effect      for an average effect: re-estimate with three g-computation outcome models
              and a bootstrap; flag an answer outside their range (effect_check.assess)
  dependence  for a claimed relation between two variables: marginal and partial
              association tests, reported with the answer (informational, never blocks)

Nothing here names a QRData dataset or question. Run with elbi's Python.
Env: CHECK_HOME (task directory), CHECK_PY (python used to re-run scripts).
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOME = os.environ["CHECK_HOME"]
PY = os.environ.get("CHECK_PY", sys.executable)
ANSWER = os.path.join(HOME, "answers", "answer.txt")
mcp = MCPServer("elbi-check", instructions=(
    "elbi checks a final answer before it is written. Write your final answer with "
    "submit_answer, passing the script that computes it."))
NUM = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?%?", re.I)


def _num(s):
    m = NUM.search(str(s).replace(",", ""))
    if not m:
        return None
    t = m.group()
    return float(t[:-1]) / 100 if t.endswith("%") else float(t)


def _same(answer, printed):
    """Whether the script's printed result supports the answer (numbers: equal up to the
    answer's rounding; text: same leading token)."""
    a, p = _num(answer), _num(printed)
    if a is not None and p is not None and not re.match(r"^[a-z]\b", answer.strip(), re.I):
        dec = len(answer.split(".")[1].rstrip("%")) if "." in answer else 0
        tol = 0.5 * 10 ** -dec * (0.01 if answer.strip().endswith("%") else 1) + 1e-9
        return abs(a - p) <= max(tol, 1e-6 * abs(a))
    tok = lambda s: re.sub(r"[^a-z0-9]", "", str(s).strip().lower().split()[0]) if str(s).strip() else ""
    return tok(answer) == tok(printed)


def _load(data_file):
    p = data_file if os.path.isabs(data_file) else os.path.join(HOME, data_file)
    return pd.read_csv(p)


def _dependence(df, x, y, given):
    from scipy import stats
    cols = [x, y, *given]
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    out = {"n": len(d)}
    r, p = stats.spearmanr(d[x], d[y])
    out["marginal_spearman"] = {"rho": float(r), "p": float(p)}
    if given:
        Z = np.column_stack([np.ones(len(d)), d[list(given)].to_numpy(float)])
        res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        rx, ry = res(d[x].to_numpy(float)), res(d[y].to_numpy(float))
        r, p = stats.pearsonr(rx, ry)
        out["partial_correlation_given"] = {"given": list(given), "r": float(r), "p": float(p)}
    return out


def _format_like(answer, value):
    """`value` written with the answer's decimals, as a percentage if the answer was one."""
    a = answer.strip()
    pct = a.endswith("%")
    body = a[:-1] if pct else a
    m = NUM.search(body.replace(",", ""))
    dec = len(m.group().split(".")[1]) if m and "." in m.group() else 0
    return f"{value * 100 if pct else value:.{dec}f}" + ("%" if pct else "")


@mcp.tool()
def submit_answer(answer: str, script: str, check: str = "none", data_file: str = "",
                  x: str = "", y: str = "", given: list[str] | None = None,
                  x0: float = 0.0, x1: float = 1.0, estimand: str = "ate",
                  procedure_specified: bool = False, confirm: bool = False, reason: str = "") -> str:
    """Write your final answer through elbi, instead of writing the answer file yourself.

    answer: exactly what should be graded (a number, or the chosen option).
    script: path to the Python script that computes the answer from the data files and
      prints it as its last line. elbi re-runs it from scratch and checks it prints the answer.
    check: what elbi should also verify independently, when it applies:
      "effect": the answer is an average effect of x on y (from x0 to x1) adjusting for
        `given`, over everyone (estimand "ate"), the treated, x == x1 ("att"), or the
        untreated, x == x0 ("atc"); elbi re-estimates it with three outcome models and,
        for a two-valued treatment, a doubly robust estimator.
      "dependence": the answer rests on whether x and y are related (given `given`);
        elbi reports marginal and partial association tests.
      "none": no independent check applies.
    data_file: the CSV the check uses (for "effect" and "dependence").
    procedure_specified: true if the question asks for the result of a specific method
      (e.g. "use linear regression", "with propensity score matching"); false if it asks
      for the effect itself. For an effect itself, elbi's answer combines your estimate and
      its own doubly robust estimate with equal weight; for a named procedure it only checks.

    If the script does not reproduce the answer, or elbi's effect estimate contradicts it,
    nothing is written and elbi says why. Look into the disagreement; then fix the answer,
    or resubmit with confirm=true and a reason to keep it."""
    given = list(given or [])
    script_p = script if os.path.isabs(script) else os.path.join(HOME, script)
    report = {}
    problems = []
    combined_value = None
    try:
        run = subprocess.run([PY, script_p], cwd=HOME, capture_output=True, text=True, timeout=600)
        lines = [l for l in run.stdout.strip().splitlines() if l.strip()]
        printed = lines[-1] if lines else ""
        report["script_printed"] = printed
        if run.returncode != 0:
            problems.append(f"script failed on a fresh run: {run.stderr.strip()[-600:]}")
        elif not _same(answer, printed):
            problems.append(f"script's last printed line ({printed!r}) does not match the answer ({answer!r})")
    except Exception as e:  # noqa: BLE001
        problems.append(f"could not re-run the script: {e}")

    if check in ("effect", "dependence"):
        try:
            df = _load(data_file)
            missing = [v for v in [x, y, *given] if v not in df.columns]
            if missing:
                problems.append(f"not columns of {data_file}: {missing}")
            elif check == "dependence":
                report["dependence"] = _dependence(df, x, y, given)
            else:
                import effect_dr as ed
                rows = df[[x, y, *given]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
                e = estimand.lower()
                if e not in ("ate", "att", "atc"):
                    raise ValueError(f"estimand must be ate, att or atc, not {estimand!r}")
                est, se = ed.g_estimates(rows, x, y, given, float(x0), float(x1), e)
                two_valued = set(np.round(rows[x].unique(), 9)) <= {round(float(x0), 9), round(float(x1), 9)}
                if two_valued:
                    est["doubly_robust"], se["doubly_robust"] = ed.aipw(rows, x, y, given, float(x0), float(x1), e)
                lo = min(est[k] - 2 * se[k] for k in est if np.isfinite(est[k]))
                hi = max(est[k] + 2 * se[k] for k in est if np.isfinite(est[k]))
                report["effect"] = {"elbi_estimates": est, "elbi_standard_errors": se,
                                    "range": [lo, hi], "rows": len(rows), "estimand": e}
                a = _num(answer)
                if a is None or not lo <= a <= hi:
                    problems.append(f"answer {answer} is outside elbi's estimates of the effect ({lo:.4g} to {hi:.4g})")
                elif two_valued and not procedure_specified:
                    combined_value = (a + est["doubly_robust"]) / 2
        except Exception as e:  # noqa: BLE001
            problems.append(f"elbi's {check} check failed: {e}")

    if problems and not confirm:
        return json.dumps({"wrote": None, "problems": problems, **report,
                           "next": "Investigate; then fix the answer, or resubmit with confirm=true and a reason."}, indent=1)
    final = answer.strip()
    if combined_value is not None and not problems:
        final = _format_like(answer, combined_value)
        report["combined"] = {"yours": answer, "elbi_doubly_robust": report["effect"]["elbi_estimates"]["doubly_robust"],
                              "written": final, "rule": "equal-weight mean of two independent estimates"}
    os.makedirs(os.path.dirname(ANSWER), exist_ok=True)
    with open(ANSWER, "w") as f:
        f.write(final + "\n")
    with open(os.path.join(HOME, "elbi_check.json"), "a") as f:
        f.write(json.dumps({"answer": answer, "check": check, "problems": problems, "confirm": confirm,
                            "reason": reason, **report}) + "\n")
    return json.dumps({"wrote": ANSWER, "answer": final, "overridden": problems if problems else None, **report,
                       "note": "This is the final answer; do not rewrite the file."}, indent=1)


if __name__ == "__main__":
    mcp.run()
