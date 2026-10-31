"""Prototype "front door" for elbi: one tool, plan_analysis, called before any analysis.

It wraps elbi's own structure analysis (elbi_core.analyze_structure, the engine behind
the structure_map tool) and adds generic checks for the decisions agents skipped on the
BLADE dev datasets: repeated units and nesting, measurement groups, exposure/rate
columns, outcome shape and outliers, recodes, and missingness. Nothing here names a
BLADE dataset or column; every rule is ordinary statistical practice.

Run with elbi's Python (it imports elbi_core). The data path comes from PLAN_DATA_CSV.
"""

import json
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from elbi_core import analyze_structure

DATA = os.environ["PLAN_DATA_CSV"]

SUBMIT = os.environ.get("PLAN_SUBMIT") == "1"  # submit_answer replaces estimate_effect
CLAIM = os.environ.get("PLAN_CLAIM") == "1"  # submit_effect_claim replaces submit_answer
REUSE = os.environ.get("PLAN_REUSE") == "1"  # session 2: reuse a saved typed analysis
MEMORY = os.environ.get("PLAN_MEMORY", "")  # where session 1 leaves its analysis
STORE = os.environ.get("PLAN_STORE", "")  # "typed": claim + fitted models; "code": runnable script
INSTRUCTIONS = (
    "elbi plans data analyses. Before writing any analysis code for a research question, "
    "call plan_analysis once with the question. It returns the dataset's structure and "
    "the analysis decisions an expert would weigh (unit of analysis, composites, rates, "
    "outcome transforms, model family), computed from the data. "
    + ("For an average-effect estimate with an interval, write the final answer with "
       "submit_answer: it checks your estimate against elbi's own and writes the answer file."
       if SUBMIT else
       "For any average-effect estimate with an interval, also call estimate_effect and "
       "reconcile it with your own.")
)
mcp = MCPServer("elbi-plan", instructions=INSTRUCTIONS)

ID_HINT = re.compile(r"(^|_)(id|code|name|key)$|id$", re.I)
EXPOSURE_WORDS = {"games", "matches", "trials", "total", "students", "exposure", "visits", "days", "years", "attempts", "sessions"}


def _stem(col: str) -> str:
    """Name with trailing digits/rater suffixes removed, to find parallel measures."""
    return re.sub(r"[\W_]*(\d+|[ab])$", "", col.lower())


def _tokens(text: str) -> set[str]:
    """Words in a column name or question, splitting camelCase and snake_case."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


def _related(col: str, q: set[str]) -> bool:
    """Whether a column shares a word (or a word's prefix) with the question."""
    return any(a.startswith(b) or b.startswith(a) for a in _tokens(col) for b in q)


def _is_unit_id(s: pd.Series, n: int) -> bool:
    """Integer codes 0/1..k with every code repeated: an identifier of a repeated unit."""
    x = s.dropna()
    if not len(x) or not (x % 1 == 0).all():
        return False
    k = x.nunique()
    return 20 <= k and n / k >= 2 and x.min() == 1 and x.max() == k


def _section(title: str, lines: list[str]) -> str:
    return f"## {title}\n" + ("\n".join(f"- {x}" for x in lines) if lines else "- none found")


@mcp.tool()
def plan_analysis(question: str) -> str:
    """Call this FIRST, before writing analysis code: plans the analysis of the dataset for
    a research question. Returns the data's structure (identifiers, derived and collinear
    columns, likely confounders, from elbi) and the decisions to make: unit of analysis
    and nesting, composite measures, rates with an exposure, outcome transforms and
    outliers, recodes, missing data, and the model family with defensible alternatives."""
    df = pd.read_parquet(DATA) if DATA.endswith(".parquet") else pd.read_csv(DATA)
    n = len(df)
    sample = df.sample(min(n, 20000), random_state=0) if n > 20000 else df
    rows = sample.where(sample.notna(), None).to_dict("records")
    structure = analyze_structure(rows, list(df.columns)).render()
    # elbi flags near-unique columns as identifiers, which wrongly includes continuous
    # measurements (every value of a uniform draw is unique). Keep only non-float ones.
    fixed = []
    for line in structure.splitlines():
        if line.startswith("Identifiers"):
            head, _, names = line.partition(": ")
            keep = [c for c in names.split(", ") if c in df.columns and not pd.api.types.is_float_dtype(df[c])]
            if not keep:
                continue
            line = f"{head}: {', '.join(keep)}"
        fixed.append(line)
    structure = "\n".join(fixed)

    num = df.select_dtypes("number")
    q = _tokens(question)
    # Question-related columns first, so the relevant findings survive truncation.
    order = sorted(df.columns, key=lambda c: not _related(c, q))

    # Repeated units: a column whose values repeat across rows marks a unit observed
    # many times (people, places, raters) -> choose the unit, cluster or model it.
    units, unit_cols = [], []
    for c in df.columns:
        k = df[c].nunique(dropna=True)
        named_id = ID_HINT.search(c) or df[c].dtype == object or _is_unit_id(df[c], n)
        if 1 < k < n and named_id and n / k >= 2:
            if k <= 12 and df[c].dtype == object:
                continue  # a small categorical, handled as a recode below
            units.append(f"`{c}`: {k} distinct values, ~{n / k:.1f} rows each")
            unit_cols.append(c)
    pairs = len(units) >= 2
    unit_notes = units[:6]
    if units:
        unit_notes.append(
            "Rows repeat these units, so observations are not independent. Decide the unit "
            "the question is about; either aggregate to it (e.g. one row per unit with "
            "totals/means) or keep rows and use a mixed-effects model with a random "
            "intercept per unit (or cluster-robust errors)."
        )
    if pairs:
        unit_notes.append(
            "Two or more repeating units: rows may be pairs (dyads) of units. Consider "
            "collapsing to the unit in the question before modeling."
        )

    # Measurement groups: parallel numeric columns that measure one construct.
    groups: dict[str, list[str]] = {}
    for c in num.columns:
        groups.setdefault(_stem(c), []).append(c)
    meas, composites = [], []
    for stem, cols in groups.items():
        if len(cols) >= 2 and stem:
            composites.append((f"{stem}_mean", cols))
            corr = num[cols].corr().values[np.triu_indices(len(cols), 1)]
            meas.append(
                f"{', '.join(f'`{c}`' for c in cols)} look like parallel measures of "
                f"'{stem}' (mean pairwise r = {np.nanmean(corr):.2f}). Combine them: mean "
                "(the usual choice), and consider rescaled, rounded or binarized versions."
            )
    for a, b in combinations(num.columns, 2):
        if len(meas) >= 6:
            break
        if _stem(a) != _stem(b) and num[a].nunique() > 3 and abs(r := num[[a, b]].corr().iloc[0, 1]) > 0.85:
            if os.path.commonprefix([a.lower(), b.lower()]).__len__() >= 4:
                meas.append(f"`{a}` and `{b}` look like two measurements of one construct (r = {r:.2f}): combine them (mean), or use one and check the other.")
                composites.append((f"{os.path.commonprefix([a, b]).rstrip('_')}_mean", [a, b]))
            else:
                meas.append(f"`{a}` and `{b}` are highly correlated (r = {r:.2f}): don't enter both as regressors.")

    # Exposure: a count alongside the opportunity it comes from -> a rate or an offset.
    ints = [c for c in order if c in num.columns and (num[c].dropna() % 1 == 0).all() and num[c].min() >= 0]
    exposure, rate_pairs = [], []
    for e in [c for c in ints if any(w in t for t in _tokens(c) for w in EXPOSURE_WORDS) and not _is_unit_id(df[c], n)]:
        for c in ints:
            if c != e and num[c].max() > 0 and (num[c] <= num[e]).mean() > 0.98 and num[c].mean() < num[e].mean():
                exposure.append(f"`{c}` is bounded by `{e}`: model the rate `{c}`/`{e}` (e.g. {c}_per_{e}), or a count model with log(`{e}`) as an offset.")
                rate_pairs.append((c, e))
    exposure = exposure[:6]

    # Outcome shape: candidates are numeric columns named in the question, else all.
    named = [c for c in num.columns if _related(c, q)] or list(num.columns)
    shape, skewed, extreme, binary = [], [], {}, []
    for c in named[:8]:
        x = num[c].dropna()
        if x.nunique() <= 2:
            shape.append(f"`{c}` is binary: logistic regression if it is the outcome.")
            binary.append(c)
            continue
        sk = float(x.skew())
        line = f"`{c}`: skew {sk:.1f}"
        if c in ints and x.mean() < 50:
            disp = x.var() / x.mean() if x.mean() > 0 else float("nan")
            line += f", count with variance/mean {disp:.1f}"
            if disp > 1.5:
                line += " (overdispersed: negative binomial over Poisson)"
            if (x == 0).mean() > 0.5:
                line += f", {100 * (x == 0).mean():.0f}% zeros"
        if abs(sk) > 1 and x.min() >= 0:
            line += "; skewed: try log(x+1) or Box-Cox as the outcome"
            skewed.append(c)
        z = (x - x.mean()) / x.std()
        out = z[abs(z) > 4]
        if len(out):
            extreme[c] = list(out.index[:5])
            line += f"; {len(out)} extreme value(s) (|z|>4, rows {list(out.index[:3])}): check results with and without them"
        shape.append(line)

    # Recodes and missing data.
    recodes = [
        f"`{c}` ({df[c].nunique()} levels: {', '.join(map(str, df[c].dropna().unique()[:4]))}): code as indicator(s)."
        for c in df.columns
        if df[c].dtype == object and 2 <= df[c].nunique() <= 6
    ][:8]
    missing = [
        f"`{c}`: {100 * df[c].isna().mean():.0f}% missing"
        for c in df.columns
        if df[c].isna().mean() > 0.01
    ][:8]
    if missing:
        missing.append("Drop rows missing the analysis variables (say so), or impute; check the result doesn't depend on it.")

    model = [
        "Continuous outcome: OLS; add the confounders above as controls; with repeated units, a mixed-effects model (random intercept per unit) or cluster-robust errors.",
        "Binary outcome: logistic regression (mixed-effects logistic with repeated units).",
        "Count outcome: Poisson, or negative binomial if overdispersed; with an exposure, use a rate or an offset.",
        "Write down the main specification, then the defensible alternatives (transformed outcome, with/without outliers, composite variants, clustered vs mixed) and check the effect holds across them.",
    ]

    # PLAN_PRESCRIBE=0: facts only. The agent picks the outcome and model; guessing them
    # from the question's words picked the wrong outcome on held-out data.
    if _calibration_path() and any(len(v) >= 2 for v in groups.values()):
        meas.append("A calibration file holds true values for some rows, and some variables are measured "
                    "only through several noisy columns. For effect estimates and intervals, call "
                    "estimate_effect: single-method intervals here ignore measurement error.")
    if os.environ.get("PLAN_PRESCRIBE", "1") == "0":
        main = ("## Your call\nChoose the outcome, predictor and model from the question; "
                "use the facts above to pick transformations, controls and the model family.")
    else:
        main = _main_spec(q, df, unit_cols, composites, rate_pairs, skewed, extreme, binary)

    return "\n\n".join([
        f"# Analysis plan for: {question}",
        f"{n} rows x {df.shape[1]} columns.",
        "## Structure (from elbi)\n" + structure,
        _section("Unit of analysis and nesting", unit_notes),
        _section("Measurement groups (composites)", meas),
        _section("Rates and exposure", exposure),
        _section("Outcome and variable shape", shape),
        _section("Recodes", recodes),
        _section("Missing data", missing),
        _section("Model family and alternatives", model),
        main,
    ])


def _overlap(col: str, q: set[str]) -> int:
    return sum(any(a.startswith(b) or b.startswith(a) for b in q) for a in _tokens(col))


def _main_spec(q, df, unit_cols, composites, rate_pairs, skewed, extreme, binary) -> str:
    """One concrete, conventional main specification built from the findings above,
    plus named alternatives. Conventional on purpose: the simplest defensible model
    first, the more elaborate ones as checks."""
    steps, alts = [], []
    for name, cols in composites:
        if name.rstrip("_mean"):
            steps.append(f"Build `{name}` = mean of {', '.join(f'`{c}`' for c in cols)}.")
            alts.append(f"`{name}` binarized at its midpoint, or one measure alone.")
    focus = sorted(unit_cols, key=lambda c: -_overlap(c, q))
    unit = focus[0] if focus and _overlap(focus[0], q) else None
    collapse = unit is not None and len(unit_cols) >= 2
    # The outcome: the count with an exposure that best matches the question, else the
    # question-related column.
    rated = sorted(rate_pairs, key=lambda p: -_overlap(p[0], q))
    outcome = None
    if rated and _overlap(rated[0][0], q):
        c, e = rated[0]
        if collapse:
            steps.append(f"Collapse to one row per `{unit}`: sum `{c}` and `{e}`, average the other per-row measures (and the composites).")
            alts.append(f"Keep the original rows and fit a mixed-effects model with a random intercept per `{unit}`.")
        outcome = f"{c}_per_{e}"
        steps.append(f"Outcome: `{outcome}` = `{c}` / `{e}`.")
        alts.append(f"Poisson or negative binomial on `{c}` with log(`{e}`) as an offset.")
        alts.append(f"Logistic regression on whether `{c}` > 0.")
    else:
        cands = sorted([c for c in skewed + binary], key=lambda c: -_overlap(c, q))
        if cands and _overlap(cands[0], q):
            c = cands[0]
            if c in binary:
                outcome = c
                alts.append("Report odds ratios; check separation.")
            else:
                outcome = f"log_{c}"
                steps.append(f"Outcome: `{outcome}` = log(`{c}` + 1), because `{c}` is strongly skewed.")
                alts.append(f"Box-Cox transform of `{c}`; or a negative binomial model on `{c}` itself.")
    if outcome is None and skewed:
        listed = ", ".join(f"`{c}`" for c in skewed[:4])
        steps.append(f"If the outcome is one of the strongly skewed columns ({listed}), model log(x + 1) of it.")
        alts.append("Box-Cox instead of log; or a negative binomial model on the raw count.")
        for c in skewed[:4]:
            if c in extreme:
                alts.append(f"If `{c}` is the outcome, refit without its extreme rows ({extreme[c]}).")
    for c, rows in extreme.items():
        if outcome and c in outcome:
            alts.append(f"Refit without the extreme rows of `{c}` ({rows}).")
    if unit_cols and not collapse:
        u = unit_cols[0]
        model = f"Mixed-effects linear model with a random intercept per `{u}` (smf.mixedlm(..., groups=df['{u}']))"
        alts.append(f"OLS with cluster-robust errors by `{u}`.")
    elif outcome in binary:
        model = "Logistic regression (smf.logit)"
    else:
        model = "OLS (smf.ols)"
    steps.append(
        f"Model: {model} of {f'`{outcome}`' if outcome else 'the outcome'} on the independent "
        "variable, with the confounders from the structure map as controls."
    )
    alts.append("Same model without controls, to show how much they move the effect.")
    return (
        "## Recommended main specification\n"
        "Follow this unless the data rule it out; it is the conventional analysis.\n"
        + "\n".join(f"{i}. {x}" for i, x in enumerate(steps, 1))
        + "\n\nDefensible alternatives to check it against:\n"
        + "\n".join(f"- {x}" for x in alts)
    )


def _calibration_path() -> str | None:
    """A calibration file next to the data: a subset of rows that also holds true values."""
    cand = os.path.join(os.path.dirname(DATA), "calibration.parquet")
    return cand if os.path.exists(cand) else None


def estimate_effect(treatment: str, outcome: str, adjust: list[str], x0: float, x1: float) -> str:
    """An independent second estimate of the average effect of changing `treatment` from x0
    to x1 on `outcome`, adjusting for `adjust` (which you choose from the causal story),
    with a bootstrap 95% interval. Use it to check your own estimate, especially when some
    variables are measured only through several noisy columns (Name1, Name2, ...) and a
    calibration file gives their true values for some rows: it fits a robust spline outcome
    model to those true values. Name variables by concept (the column stem before digits).
    It does not decide identifiability: if the effect cannot be identified, say so instead."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import latent_effect as le
    data = pd.read_parquet(DATA) if DATA.endswith(".parquet") else pd.read_csv(DATA)
    cpath = _calibration_path()
    rows = pd.read_parquet(cpath) if cpath else data
    missing = [c for c in [treatment, outcome, *adjust] if c not in rows.columns]
    if missing:
        return (f"Unknown variable(s): {missing}. Use concept names as they appear in "
                f"{'the calibration file' if cpath else 'the data'}: {list(rows.columns)}")
    r = le.run_calibration_spline(rows, treatment, outcome, list(adjust), float(x0), float(x1))
    source = ("the calibration rows' true values" if cpath else "the data as given")
    return json.dumps({
        "estimate": r["estimate"], "ci_95": r["ci"], "based_on": source,
        "how_to_use": ("Compare with your own estimate and interval. If the two intervals overlap, "
                       "report yours. If they do not overlap, the effect is less certain than either "
                       "suggests: report an interval spanning both, with the estimate between them."),
    }, indent=1)


def _known_variables(cols, calib_cols) -> set[str]:
    """Concept names usable as variables: plain columns, proxy stems (Name1, Name2 -> Name),
    and the calibration file's true-value columns."""
    stems = {re.sub(r"\d+$", "", c) for c in cols if re.search(r"\d+$", c)}
    return set(cols) | stems | set(calib_cols)


def submit_answer(answer_file: str, identifiable: bool, treatment: str = "", outcome: str = "",
                  adjust: list[str] | None = None, x0: float = 0.0, x1: float = 1.0,
                  ate: float | None = None, ci_lower: float | None = None,
                  ci_upper: float | None = None) -> str:
    """Write your final average-effect answer through elbi. Call it once, after your own
    analysis, instead of writing the answer file yourself.

    If you concluded the effect is not identifiable, pass identifiable=false (other fields
    optional): it writes {"ate": null}. Otherwise pass the treatment, outcome and adjustment
    set you used (concept names; for variables measured by several columns Name1, Name2, ...
    use the stem Name), the levels x0 and x1, and your estimate and 95% interval.

    elbi first checks that the named variables exist in this data; if they do not, it writes
    nothing and tells you why. Then it re-estimates the effect with three outcome models and
    a bootstrap, and combines its analysis with yours, equally weighted: where they agree the
    interval stays narrow, where they disagree it spans both. It returns what it wrote and why.
    It can take a couple of minutes on large data."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    path = answer_file if os.path.isabs(answer_file) else os.path.join(os.getcwd(), answer_file)
    if not identifiable:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"ate": None}, f)
        return json.dumps({"wrote": path, "answer": {"ate": None}})
    if None in (ate, ci_lower, ci_upper) or not treatment or not outcome:
        return "Nothing written: pass treatment, outcome, ate, ci_lower and ci_upper (or identifiable=false)."
    import effect_check as ec
    data = pd.read_parquet(DATA) if DATA.endswith(".parquet") else pd.read_csv(DATA)
    cpath = _calibration_path()
    calib = pd.read_parquet(cpath) if cpath else None
    adjust = list(adjust or [])
    known = _known_variables(list(data.columns), list(calib.columns) if calib is not None else [])
    unknown = [v for v in [treatment, outcome, *adjust] if v not in known]
    if unknown:
        return json.dumps({"wrote": None, "problem": (
            f"Not in this data: {unknown}. Known variables: {sorted(known)}. If the question names "
            "variables this dataset does not contain, the data may not match the question: say so "
            "rather than answering with a number from other variables.")}, indent=1)
    rows = calib if calib is not None else data
    missing = [v for v in [treatment, outcome, *adjust] if v not in rows.columns]
    if missing:
        return json.dumps({"wrote": None, "problem": (
            f"{missing} are measured only through noisy columns and the calibration file lacks "
            "their true values, so elbi cannot check this estimate.")}, indent=1)
    r = ec.assess(rows, treatment, outcome, adjust, x0, x1, proxy=calib is not None)
    # Two analyses, equally believed: yours, and elbi's (its three outcome models share its
    # half). The answer is the central 95% of that combined uncertainty.
    yours = (0.5, (ci_lower + ci_upper) / 2, max(ci_upper - ci_lower, 1e-12) / (2 * 1.959964))
    elbi_lo, elbi_hi = ec.mixture_interval(ec.elbi_components(r))
    final_lo, final_hi = ec.mixture_interval([yours, *ec.elbi_components(r, weight=0.5)])
    answer = {"ate": float(min(max(ate, final_lo), final_hi)), "ci_lower": float(final_lo),
              "ci_upper": float(final_hi), "alpha": 0.05, "method": "agent estimate, combined with elbi's"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(answer, f)
    return json.dumps({
        "wrote": path, "answer": answer,
        "your_interval": [ci_lower, ci_upper], "elbi_interval": [elbi_lo, elbi_hi],
        "elbi_estimates": r["estimates"], "elbi_standard_errors": r["se"], "based_on": "calibration rows' true values" if calib is not None else "the data",
        "note": ("Your analysis and elbi's are combined with equal weight: where they agree the interval "
                 "stays narrow, where they disagree it spans both. This is the final answer; do not "
                 "rewrite the file."),
    }, indent=1)


def submit_effect_claim(answer_file: str, strategy: str, treatment: str, outcome: str,
                        graph: list[list[str]], unobserved: list[str] | None = None,
                        adjust: list[str] | None = None, mediators: list[str] | None = None,
                        x0: float = 0.0, x1: float = 1.0, ate: float | None = None,
                        ci_lower: float | None = None, ci_upper: float | None = None) -> str:
    """Write your final average-effect answer through elbi, stating how it is identified.
    Call it once, after your own analysis, instead of writing the answer file yourself.

    Declare the causal graph you assumed from the story: `graph` is a list of [cause, effect]
    pairs over conceptual variables (for variables measured by several columns Name1, Name2,
    ... use the stem Name), and `unobserved` names graph variables with no data. Then
    `strategy`: "backdoor" (give `adjust`, the set you adjusted for), "frontdoor" (give
    `mediators`), or "not_identifiable" (writes {"ate": null}). For an identified effect,
    also give x0, x1, and your estimate and 95% interval.

    elbi checks that your strategy identifies the effect in your declared graph; if not, it
    writes nothing and says why, so you can fix the analysis or the declaration. Otherwise it
    re-estimates the same quantity by the same strategy with three outcome models, combines
    that with your estimate at equal weight, and writes the answer. It can take a few minutes."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import effect_claim as cl
    import effect_check as ec
    path = answer_file if os.path.isabs(answer_file) else os.path.join(os.getcwd(), answer_file)
    if strategy not in cl.STRATEGIES:
        return f"Nothing written: strategy must be one of {cl.STRATEGIES}."
    if strategy == "not_identifiable":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"ate": None}, f)
        return json.dumps({"wrote": path, "answer": {"ate": None}})
    if None in (ate, ci_lower, ci_upper):
        return "Nothing written: pass ate, ci_lower and ci_upper for an identified effect."
    data = pd.read_parquet(DATA) if DATA.endswith(".parquet") else pd.read_csv(DATA)
    cpath = _calibration_path()
    calib = pd.read_parquet(cpath) if cpath else None
    rows = calib if calib is not None else data
    claim = cl.EffectClaim(treatment=treatment, outcome=outcome, estimand="ate", x0=float(x0), x1=float(x1),
                           strategy=strategy, adjust=tuple(adjust or ()), mediators=tuple(mediators or ()),
                           graph=tuple(tuple(e) for e in graph), latent=tuple(unobserved or ()),
                           data_hash=cl.data_hash(rows))
    ok, why = cl.check_identification(claim)
    if not ok:
        return json.dumps({"wrote": None, "problem": f"Your {strategy} strategy does not identify the effect "
                           f"in the graph you declared: {why}."}, indent=1)
    needed = [treatment, outcome, *claim.adjust, *claim.mediators]
    missing = [v for v in needed if v not in rows.columns]
    if missing:
        return json.dumps({"wrote": None, "problem": f"No values for {missing} in "
                           f"{'the calibration file' if calib is not None else 'the data'}; known: {list(rows.columns)}"}, indent=1)
    res, fa = cl.estimate(claim, rows)
    _store(claim, res, fa)
    yours = (0.5, (ci_lower + ci_upper) / 2, max(ci_upper - ci_lower, 1e-12) / (2 * 1.959964))
    comps = [(0.5 / len(res["estimates"]), res["estimates"][k], res["se"][k]) for k in res["estimates"]]
    lo, hi = ec.mixture_interval([yours, *comps])
    answer = {"ate": float(min(max(ate, lo), hi)), "ci_lower": float(lo), "ci_upper": float(hi),
              "alpha": 0.05, "method": f"agent {strategy} estimate, combined with elbi's"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(answer, f)
    return json.dumps({"wrote": path, "answer": answer, "identification": why,
                       "elbi_estimates": res["estimates"], "elbi_standard_errors": res["se"],
                       "note": "This is the final answer; do not rewrite the file."}, indent=1)


def _store(claim, res, fa):
    """Leave session 1's checked analysis for a later session, typed or as code."""
    if not MEMORY or not STORE:
        return
    import dataclasses
    import pickle
    os.makedirs(MEMORY, exist_ok=True)
    spec = dataclasses.asdict(claim)
    if STORE == "typed":
        with open(os.path.join(MEMORY, "analysis.pkl"), "wb") as f:
            pickle.dump(fa, f)
        with open(os.path.join(MEMORY, "analysis.json"), "w") as f:
            json.dump({"claim": spec, "result": res}, f, indent=1)
    elif STORE == "code":
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(MEMORY, "elbi_analysis.py"), "w") as f:
            f.write(f'''"""elbi's check of the earlier analysis on this dataset, as runnable code.

Re-runs elbi's estimator for the claim below (three outcome models, bootstrap SEs). Edit the
claim (e.g. the estimand) and run with: python3 elbi_analysis.py <rows.parquet>
Estimands: "ate", or "ett" (averaged over rows with treatment == x1). Strategies: backdoor,
frontdoor (ate only). The rows used before were {"calibration.parquet" if spec else ""} if present, else data.parquet.
"""
import json, sys
sys.path.insert(0, {here!r})
import pandas as pd
import effect_claim as cl

CLAIM = {spec!r}
CLAIM["graph"] = tuple(map(tuple, CLAIM["graph"]))

if __name__ == "__main__":
    rows = pd.read_parquet(sys.argv[1])
    c = cl.EffectClaim(**{{**CLAIM, "data_hash": cl.data_hash(rows)}})
    res, _ = cl.estimate(c, rows)
    print(json.dumps(res, indent=1))
''')


def saved_analysis() -> str:
    """The analysis an earlier session saved for this dataset: what was estimated (estimand,
    treatment levels), by which identification strategy and adjustment or mediator set,
    under which declared causal graph, on which rows (data hash), with elbi's estimates."""
    path = os.path.join(MEMORY, "analysis.json")
    return open(path).read() if os.path.exists(path) else "No saved analysis for this dataset."


def reuse_saved_analysis(estimand: str) -> str:
    """Answer a different estimand ("ate" or "ett", the effect on the treated) from the saved
    analysis by re-averaging its fitted models over the new target population, without
    refitting. elbi refuses, and says why, when reuse would not be valid: the data changed,
    the strategy does not identify the new estimand, or reuse is unsupported for it. Returns
    per-model estimates, bootstrap SEs, and their combined 95% interval."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pickle
    import effect_check as ec
    import effect_claim as cl
    path = os.path.join(MEMORY, "analysis.pkl")
    if not os.path.exists(path):
        return "No saved analysis for this dataset."
    with open(path, "rb") as f:
        fa = pickle.load(f)
    cpath = _calibration_path()
    rows = pd.read_parquet(cpath) if cpath else (pd.read_parquet(DATA) if DATA.endswith(".parquet") else pd.read_csv(DATA))
    res, why = cl.retarget(fa, estimand, rows)
    if not res:
        return json.dumps({"reused": False, "reason": why}, indent=1)
    lo, hi = ec.mixture_interval([(1 / len(res["estimates"]), res["estimates"][k], res["se"][k]) for k in res["estimates"]])
    return json.dumps({"reused": True, "estimand": estimand, "estimates": res["estimates"], "se": res["se"],
                       "combined_95": [lo, hi], "how": why, "claim": {k: v for k, v in
                       __import__("dataclasses").asdict(fa.claim).items() if k != "graph"}}, indent=1)


if REUSE:
    mcp.tool()(saved_analysis)
    mcp.tool()(reuse_saved_analysis)
else:
    mcp.tool()(submit_effect_claim if CLAIM else submit_answer if SUBMIT else estimate_effect)

if __name__ == "__main__":
    mcp.run()
