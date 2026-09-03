"""Precondition-driven verification selection (a Tea-style applicability engine).

Choosing which checks a question needs is not a routing problem: with a fixed catalog of
cheap, deterministic checks, the right design (Simonsohn-style suites; Tea's constraint
solver, Jun et al. 2019) is for every check to *declare its preconditions* (the column
roles it needs and the data shape those columns must have) and for the engine to keep
exactly the checks whose preconditions hold, then run them all. There is no intent
classifier and no model in the loop: a check self-selects when the data supports it, so
a model-evaluation question reaches the classification check because the data has
predictions and labels, not because something guessed the intent. The caller (an agent,
upstream) only maps the question onto column roles; the selection and the verdict stay
deterministic and auditable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ._numerics import _as_float, _selecting_baseline
from ._report import VerificationReport
from .calibration import verify_calibration
from .classification import verify_classification
from .clustering import verify_clusters
from .comparison import verify_comparison
from .correlation import verify_correlation
from .counts import verify_counts
from .did import verify_did
from .distribution import verify_normality
from .effect import verify_effect
from .equivalence import verify_equivalence
from .experiment import verify_experiment
from .fairness import verify_fairness
from .forecast import verify_forecast
from .glm import verify_logistic
from .groups import verify_groups
from .hazards import verify_proportional_hazards
from .iv import verify_iv
from .leakage import verify_leakage
from .missingness import verify_missingness
from .overlap import verify_overlap
from .prediction import verify_prediction
from .proportions import verify_proportions
from .rdd import verify_rdd
from .regression import verify_regression
from .reliability import verify_reliability
from .rtm import verify_rtm
from .screen import verify_screen
from .sensitivity import verify_sensitivity
from .spatial import verify_spatial
from .survival import verify_survival
from .trend import verify_trend

Rows = Sequence[dict[str, Any]]
Claim = dict[str, Any]


@dataclass(frozen=True)
class GateResult:
    """One check that ran, and what it concluded."""

    name: str
    verdict: str  # "sound" | "unsound" | "inconclusive"
    detail: str
    estimate: float | None = None  # the gate's own quantity (a coefficient, lift, …)


@dataclass(frozen=True)
class CompositeReport:
    """The combined outcome of running every applicable check for a claim."""

    verdict: str  # "sound" | "unsound" | "invalid" | "inconclusive"
    ran: tuple[GateResult, ...]
    skipped: tuple[tuple[str, str], ...]  # (check, why it does not apply)
    #: The certifying gate's own estimate (a coefficient, a mean difference, a held-out
    #: skill) and a unit-aware human description of it, surfaced so a serving layer
    #: reports the number the oracle computed rather than one the model wrote. Both are
    #: None unless the verdict is ``sound``. ``adjusted_for`` names the columns the
    #: estimate holds fixed, so it is presented as a direct effect given them, never as
    #: an unconditional total effect (the Table 2 fallacy).
    estimate: float | None = None
    estimate_label: str | None = None
    adjusted_for: tuple[str, ...] = ()

    def render(self) -> str:
        """Format the composite verdict as markdown for an agent to read."""
        lines = [f"# Verification: **{self.verdict.upper()}**", ""]
        if self.ran:
            lines.append("checks that applied:")
            lines += [f"- [{g.verdict}] {g.name}: {g.detail}" for g in self.ran]
        if len(self.ran) > 1:
            lines.append(
                f"\n({len(self.ran)} checks bear on this claim; the verdict requires "
                "them to agree, so adding checks only makes a sound verdict harder)"
            )
        if self.skipped:
            # report the skips concisely so the verdict is not buried (pytest -rs style)
            names = ", ".join(name for name, _ in self.skipped)
            lines.append(
                f"\n{len(self.skipped)} checks not applicable to this data: {names}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Check:
    """A catalog entry: when it applies, how to run it, and whether it can veto."""

    name: str
    applies: Callable[[Rows, Claim], str | None]  # a reason if applicable, else None
    run: Callable[[Rows, Claim], VerificationReport]
    veto: bool = False  # a validity check whose failure voids the whole result
    certifying: bool = False  # the primary gate for its claim: if it cannot certify
    # (inconclusive), a weaker supporting gate must not rescue the whole to "sound"


def applicable(rows: Rows, **claim: Any) -> list[tuple[str, str]]:
    """The checks whose preconditions the data and claim satisfy, with the reason.

    This is the deterministic answer to "which verification does this question need":
    each check is kept exactly when the columns the caller assigned have the right
    shape for it. Usually only a few of the catalog apply to any one claim.
    """
    out = []
    for check in _CATALOG:
        reason = check.applies(rows, claim)
        if reason is not None:
            out.append((check.name, reason))
    return out


#: Variance-stabilising / linearising column transforms, applied before verification
#: when a claim asks for them. Each maps a numeric value to the transformed value, or
#: None where it is undefined (e.g. a log of a non-positive number), which drops the
#: row from that check. These are general numeric reshapings, not domain knowledge.
_TRANSFORMS: dict[str, Callable[[float], float | None]] = {
    "log": lambda v: math.log(v) if v > 0 else None,
    "log1p": lambda v: math.log1p(v) if v > -1 else None,
    "sqrt": lambda v: math.sqrt(v) if v >= 0 else None,
    "square": lambda v: v * v,
    "reciprocal": lambda v: 1.0 / v if v != 0 else None,
}


def _apply_transforms(
    rows: Rows, transforms: dict[str, str]
) -> tuple[Rows, dict[str, str]]:
    """Derive a transformed column for each ``column -> transform`` and drop the source.

    Returns the new rows and a map from each original column name to its transformed
    name (``x`` becomes ``log(x)``), so the claim can be re-pointed at it. The source
    column is removed so it is not treated as a confounder of its own transform.
    """
    fns = {c: t for c, t in transforms.items() if t in _TRANSFORMS}
    if not fns:
        return rows, {}
    renamed = {c: f"{fns[c]}({c})" for c in fns}
    out: list[dict[str, Any]] = []
    for r in rows:
        nr = {k: v for k, v in r.items() if k not in fns}
        for c, t in fns.items():
            v = _as_float(r.get(c))
            dv = _TRANSFORMS[t](v) if v is not None else None
            nr[renamed[c]] = "" if dv is None else repr(dv)
        out.append(nr)
    return out, renamed


def _retarget(value: Any, renamed: dict[str, str]) -> Any:
    """Re-point a claim's role value (a column name or list of them) at a transform."""
    if isinstance(value, str):
        return renamed.get(value, value)
    if isinstance(value, list | tuple):
        return [renamed.get(v, v) if isinstance(v, str) else v for v in value]
    return value


#: The stability pass grades a certified conclusion on whether it survives perturbation,
#: the general robustness criterion of veridical data science (Yu & Kumbier): re-run the
#: gate that certified the claim under *data* perturbation (bootstrap resamples) and
#: *specification* perturbation (dropping each auxiliary column the analyst used), and
#: report the range of the estimate and how often the conclusion holds, a PCS
#: perturbation interval. It generalises across every claim type because it re-runs
#: whatever gate certified, never assuming an effect. It never vetoes: the analyst still
#: drives the specification; this only makes how much the answer depends on it visible.
_STABILITY_ROLES = ("controls", "features", "covariates", "adjust")
_STABILITY_RESAMPLES = 15  # bootstrap replicates (data perturbation)
_STABILITY_CAP = 2500  # cap rows per resample so the pass stays cheap on large data
_STABILITY_SHARE = 0.85  # share of perturbations a robust conclusion must survive
#: Seeded-LCG constants for a deterministic bootstrap (no global RNG, reproducible).
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_MASK64 = (1 << 64) - 1


def _bootstrap(rows: Rows, seed: int) -> list[dict[str, Any]]:
    """A deterministic bootstrap resample (with replacement), capped for cost.

    Uses a seeded linear-congruential stream, not the global RNG, so the same data
    yields the same resamples: the stability range is reproducible, hence consistent.
    """
    n = min(len(rows), _STABILITY_CAP)
    total = len(rows)
    state = (seed * 2 + 1) & _MASK64
    out: list[dict[str, Any]] = []
    for _ in range(n):
        state = (state * _LCG_A + _LCG_C) & _MASK64
        out.append(rows[(state >> 33) % total])
    return out


def _spec_perturbations(claim: Claim) -> list[Claim]:
    """Claims that drop each auxiliary column in turn, and one that drops them all.

    Perturbing which covariates enter the model is the specification degree of freedom
    that generalises across claim types (controls for an effect, features for a
    prediction, covariates for an adjusted comparison), so one pass measures how much
    the conclusion turns on those choices, whatever the analysis.
    """
    out: list[Claim] = []
    for role in _STABILITY_ROLES:
        cols = claim.get(role)
        if isinstance(cols, list | tuple) and len(cols) >= 1:
            # leave each auxiliary column out in turn; for a lone column this is the
            # adjusted-vs-unadjusted contrast, the sharpest specification degree of
            # freedom (a controlled estimate against the raw one)
            out.extend({**claim, role: [c for c in cols if c != drop]} for drop in cols)
            if len(cols) > 1:
                out.append({**claim, role: []})
    return out


def _safe_run(check: _Check, rows: Rows, claim: Claim) -> VerificationReport | None:
    """Run a gate, returning None instead of raising on a degenerate perturbation."""
    try:
        return check.run(rows, claim)
    except Exception:
        return None


def _stability_result(
    rows: Rows, claim: Claim, check: _Check, base: VerificationReport
) -> GateResult | None:
    """Re-run the certifying gate under data and specification perturbations.

    Reports the range of the estimate and the share of perturbations under which the
    conclusion still certifies: a robust conclusion holds across them, a fragile one is
    an artefact of the analyst's choices. Returns None when no valid perturbation
    applies (a lone time series, where an iid bootstrap would destroy the temporal
    dependence and there are no covariates to drop), so no misleading claim is made. A
    non-vetoing annotation.
    """
    perturbations = [(rows, p) for p in _spec_perturbations(claim)]
    # An iid bootstrap is invalid where the rows are dependent: a time-indexed series or
    # a spatial surface (resampling would break the very structure being tested, and the
    # spatial gate already has its own permutation null).
    dependent = "time" in claim or _spatial_cols(claim) is not None
    if not dependent:
        perturbations += [
            (_bootstrap(rows, b), claim) for b in range(_STABILITY_RESAMPLES)
        ]
    estimates = [base.effect]
    trials = 0
    held = 0
    for data, spec in perturbations:
        rep = _safe_run(check, data, spec)
        if rep is None:
            continue
        estimates.append(rep.effect)
        trials += 1
        held += rep.verdict == "sound"
    if trials == 0:
        return None
    lo, hi = min(estimates), max(estimates)
    span = f"estimate {base.effect:.3g} (range {lo:.3g} to {hi:.3g})"
    share = held / trials
    # a stable *sign* with a magnitude spread is still robust (report the range); a sign
    # reversal across perturbations is genuine fragility; the conclusion is not settled
    flips = len({e > 0 for e in estimates if abs(e) > 1e-9}) > 1
    robust = share >= _STABILITY_SHARE and not flips
    if robust:
        detail = f"robust: holds under {share:.0%} of perturbations; {span}"
    elif flips:
        detail = f"fragile: the sign reverses across perturbations; {span}"
    else:
        detail = f"fragile: holds under only {share:.0%} of perturbations; {span}"
    return GateResult(
        "stability", "sound" if robust else "inconclusive", detail, base.effect
    )


def verify_all(rows: Rows, **claim: Any) -> CompositeReport:
    """Run every applicable check for a claim and combine them with veto precedence.

    A validity check (a sample-ratio mismatch) that fails voids the whole result
    (``invalid``) regardless of the others; otherwise the verdict is the conjunction:
    ``unsound`` if any check breaks the claim, ``sound`` if at least one supports it and
    none break it, else ``inconclusive``. Checks whose preconditions do not hold are
    reported as skipped with the reason, never silently dropped.

    A ``transforms`` mapping (``{column: "log"|"sqrt"|…}``) reshapes those columns
    before verification and re-points the claim at them, so a nonlinear relationship can
    be verified on a linearising scale (a log-log slope is an elasticity, for example).
    """
    transforms = claim.pop("transforms", None)
    # The original role->column mapping and the declared transforms, kept before the
    # retarget below rewrites the claim onto transformed columns, so the estimate can be
    # labelled in the reader's own column names and units.
    transforms_map: dict[str, str] = (
        {str(k): str(v) for k, v in transforms.items()}
        if isinstance(transforms, dict)
        else {}
    )
    orig_claim: Claim = dict(claim)
    if isinstance(transforms, dict) and transforms:
        # a transform key may be a role (``x``) or a column name directly; resolve a
        # role to the column it points at so the caller can write either
        by_column: dict[str, str] = {}
        for key, fn in transforms.items():
            role_value = claim.get(str(key))
            column = role_value if isinstance(role_value, str) else str(key)
            by_column[column] = str(fn)
        rows, renamed = _apply_transforms(rows, by_column)
        claim = {role: _retarget(value, renamed) for role, value in claim.items()}
    ran: list[GateResult] = []
    skipped: list[tuple[str, str]] = []
    invalid = False
    uncertified = False  # a certifying gate ran but could not certify the claim
    certified: tuple[_Check, VerificationReport] | None = None
    first_sound: tuple[_Check, VerificationReport] | None = None
    for check in _CATALOG:
        reason = check.applies(rows, claim)
        if reason is None:
            skipped.append((check.name, _why_not(check, rows, claim)))
            continue
        # A gate that raises on degenerate or adversarial output (a singular design, an
        # empty subgroup) must not crash the whole composite. Treat the error as an
        # inconclusive result: it cannot certify and cannot veto, and a certifying gate
        # that errors still blocks the claim from "sound" via the inconclusive branch.
        report = _safe_run(check, rows, claim) or VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("the check could not be evaluated on this data",),
        )
        # surface the finding: the pivotal issue if one broke the claim, else the
        # report's own summary caveat, not the bare applicability reason
        detail = report.pivotal or (report.caveats[0] if report.caveats else reason)
        estimate = report.effect if report.significant else None
        ran.append(GateResult(check.name, report.verdict, detail, estimate))
        if check.veto and report.verdict == "unsound":
            invalid = True
        if check.certifying and report.verdict == "inconclusive":
            uncertified = True
        if check.certifying and report.verdict == "sound":
            certified = (check, report)
        if report.verdict == "sound" and first_sound is None:
            first_sound = (check, report)

    if not ran:
        verdict = "inconclusive"
    elif invalid:
        verdict = "invalid"
    elif any(g.verdict == "unsound" for g in ran):
        verdict = "unsound"
    elif any(g.verdict == "sound" for g in ran) and not uncertified:
        verdict = "sound"
    else:
        # a supporting gate (e.g. the associational regression) may be sound, but the
        # claim's primary gate did not certify it, so the whole cannot be "sound"
        verdict = "inconclusive"
    # Grade a sound conclusion on its stability under data and specification
    # perturbations (a non-vetoing annotation), so a spec-dependent answer carries its
    # range rather than a lone point that shifts with the analyst's choices. The primary
    # gate is the one that certified, else the first that supported the claim.
    primary = certified or first_sound
    certified_estimate: float | None = None
    certified_label: str | None = None
    certified_controls: tuple[str, ...] = ()
    if verdict == "sound" and primary is not None:
        gate, base = primary
        certified_estimate = base.effect
        certified_label = _estimate_label(
            gate.name, base.effect, orig_claim, transforms_map
        )
        certified_controls = tuple(
            str(c) for c in (orig_claim.get("controls") or []) if str(c)
        )
        stability = _stability_result(rows, claim, *primary)
        if stability is not None:
            ran.append(stability)
    return CompositeReport(
        verdict,
        tuple(ran),
        tuple(skipped),
        estimate=certified_estimate,
        estimate_label=certified_label,
        adjusted_for=certified_controls,
    )


def _estimate_label(
    gate: str, effect: float, claim: Claim, transforms: dict[str, str]
) -> str:
    """A unit-aware description of the estimate the certifying gate computed.

    The magnitude is the gate's own quantity (a coefficient, a mean difference, a
    correlation, a held-out skill); its units follow the claim's roles and any declared
    transform, so a log-outcome coefficient reads as a percent change rather than a bare
    slope. Built from the original, pre-transform column names so it is readable.
    """
    x, y = claim.get("x"), claim.get("y")

    def tf(role: str, col: Any) -> str | None:
        got = transforms.get(role)
        if got is None and isinstance(col, str):
            got = transforms.get(col)
        return got

    if gate in ("effect", "regression") and isinstance(x, str) and isinstance(y, str):
        y_log, x_log = tf("y", y) in ("log", "log1p"), tf("x", x) in ("log", "log1p")
        if y_log and x_log:
            return f"elasticity {effect:+.3g}% in {y} per +1% in {x}"
        if y_log:
            return f"{(math.exp(effect) - 1) * 100.0:+.2g}% in {y} per +1 unit of {x}"
        return f"{effect:+.3g} in {y} per +1 unit of {x}"
    if gate == "correlation" and isinstance(x, str) and isinstance(y, str):
        return f"correlation r = {effect:+.2g} between {x} and {y}"
    if gate == "comparison":
        return (
            f"{effect:+.3g} difference in {claim.get('value')} between the groups "
            f"of {claim.get('group')}"
        )
    if gate == "prediction":
        return f"held-out predictive skill = {effect:.3g}"
    return f"estimate = {effect:+.3g}"


# --- column-shape predicates (the preconditions, measured from the data) -------


def _numeric(rows: Rows, col: str | None) -> bool:
    if not col:
        return False
    vals = [_as_float(r.get(col)) for r in rows if r.get(col) not in ("", None)]
    return bool(vals) and all(v is not None for v in vals)


def _distinct(rows: Rows, col: str | None) -> set[str]:
    if not col:
        return set()
    return {str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()}


def _two_level(rows: Rows, col: str | None) -> bool:
    return bool(col) and len(_distinct(rows, col)) == 2


def _categorical(rows: Rows, col: str | None) -> bool:
    return bool(col) and 2 <= len(_distinct(rows, col)) <= 10


def _multi_level(rows: Rows, col: str | None) -> bool:
    return bool(col) and 3 <= len(_distinct(rows, col)) <= 20


def _binary(rows: Rows, col: str | None) -> bool:
    if not col or not _numeric(rows, col):
        return False
    vals = {round(v) for r in rows if (v := _as_float(r.get(col))) is not None}
    return vals <= {0, 1} and len(vals) == 2


def _probability(rows: Rows, col: str | None) -> bool:
    if not col or not _numeric(rows, col):
        return False
    vals = [v for r in rows if (v := _as_float(r.get(col))) is not None]
    return (
        bool(vals)
        and all(0.0 <= v <= 1.0 for v in vals)
        and any(0 < v < 1 for v in vals)
    )


def _is_count(rows: Rows, col: str | None) -> bool:
    if not col or not _numeric(rows, col):
        return False
    vals = [v for r in rows if (v := _as_float(r.get(col))) is not None]
    return bool(vals) and all(v >= 0 and abs(v - round(v)) < 1e-9 for v in vals)


def _has(claim: Claim, *roles: str) -> bool:
    return all(claim.get(r) for r in roles)


def _has_missing(rows: Rows, col: str | None) -> bool:
    if not col:
        return False
    return any(str(r.get(col, "")).strip() == "" for r in rows)


def _spatial_cols(claim: Claim) -> tuple[str, str, str] | None:
    """Resolve (lat, long, value) for the spatial gate from the claim's geo roles.

    Accepts the coordinate role names the model uses for a map (``latitude``/``lat`` and
    ``longitude``/``lon``/``long``) plus a ``value`` to test the surface of.
    """
    lat = claim.get("latitude") or claim.get("lat")
    lon = claim.get("longitude") or claim.get("lon") or claim.get("long")
    val = claim.get("value")
    if isinstance(lat, str) and isinstance(lon, str) and isinstance(val, str):
        return lat, lon, val
    return None


def _rtm_mapping(rows: Rows, claim: Claim) -> tuple[str, str, str] | None:
    """Resolve (before, after, group) for the regression-to-the-mean gate.

    Uses explicit before/after/group roles when given; otherwise detects the trap behind
    an effect or comparison framing, a binary group (``x``, ``group`` or ``variant``)
    that is threshold-selected on some baseline column, with a numeric outcome, so the
    check runs even when the caller did not name the pre/post roles.
    """
    if (
        _has(claim, "before", "after", "group")
        and _numeric(rows, str(claim["before"]))
        and _numeric(rows, str(claim["after"]))
        and _two_level(rows, str(claim["group"]))
    ):
        return str(claim["before"]), str(claim["after"]), str(claim["group"])
    group = claim.get("group") or claim.get("x") or claim.get("variant")
    outcome = claim.get("value") or claim.get("y") or claim.get("metric")
    if not (group and outcome):
        return None
    if _two_level(rows, str(group)) and _numeric(rows, str(outcome)):
        baseline = _selecting_baseline(rows, str(group), exclude=(str(outcome),))
        if baseline is not None:
            return baseline, str(outcome), str(group)
    return None


def _rtm_applies(rows: Rows, claim: Claim) -> str | None:
    mapping = _rtm_mapping(rows, claim)
    if mapping is None:
        return None
    return f"a baseline-selected group risks regression to the mean on '{mapping[0]}'"


def _rtm_run(rows: Rows, claim: Claim) -> VerificationReport:
    mapping = _rtm_mapping(rows, claim)
    if mapping is None:  # pragma: no cover - run only follows a non-None applies
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("no baseline selection",)
        )
    return verify_rtm(rows, *mapping)


# --- the catalog: each check declares its roles and shape preconditions ---------


_CATALOG: tuple[_Check, ...] = (
    _Check(
        "effect",
        lambda rows, c: (
            "an effect needs numeric x and y"
            if _has(c, "x", "y") and _numeric(rows, c["x"]) and _numeric(rows, c["y"])
            else None
        ),
        lambda rows, c: verify_effect(
            rows,
            c["x"],
            c["y"],
            controls=[str(z) for z in (c.get("controls") or [])],
            negative_controls=[str(z) for z in (c.get("negative_controls") or [])],
        ),
        certifying=True,
    ),
    _Check(
        # a geographic surface: is the value spatially clustered (Moran's I) with real
        # hot spots (Getis-Ord Gi*)? The certified, unbiased basis for a location claim
        # and its map: a plain regression on lat/long is biased by spatial dependence.
        "spatial",
        lambda rows, c: (
            "a geographic pattern needs latitude, longitude and a numeric value"
            if (cols := _spatial_cols(c)) is not None
            and all(_numeric(rows, col) for col in cols)
            else None
        ),
        lambda rows, c: verify_spatial(rows, *(_spatial_cols(c) or ("", "", ""))),
        certifying=True,
    ),
    _Check(
        # a marginal association: only when the claim is not about a *controlled*
        # relationship (controls => effect/regression handle the partial claim)
        "correlation",
        lambda rows, c: (
            "an association needs numeric x and y (and no controls)"
            if _has(c, "x", "y")
            and not c.get("controls")
            and _numeric(rows, c["x"])
            and _numeric(rows, c["y"])
            else None
        ),
        lambda rows, c: verify_correlation(rows, c["x"], c["y"]),
    ),
    _Check(
        "regression",
        lambda rows, c: (
            "a regression coefficient needs numeric x and y"
            if _has(c, "x", "y") and _numeric(rows, c["x"]) and _numeric(rows, c["y"])
            else None
        ),
        lambda rows, c: verify_regression(
            rows, c["y"], c["x"], controls=[str(z) for z in (c.get("controls") or [])]
        ),
    ),
    _Check(
        # the E-value is a marginal-effect sensitivity, so like correlation it applies
        # only to an uncontrolled claim
        "sensitivity",
        lambda rows, c: (
            "a confounding-sensitivity check needs numeric x and y (and no controls)"
            if _has(c, "x", "y")
            and not c.get("controls")
            and _numeric(rows, c["x"])
            and _numeric(rows, c["y"])
            else None
        ),
        lambda rows, c: verify_sensitivity(rows, c["x"], c["y"]),
    ),
    _Check(
        "logistic",
        lambda rows, c: (
            "a logistic odds ratio needs numeric x and a binary y"
            if _has(c, "x", "y") and _numeric(rows, c["x"]) and _binary(rows, c["y"])
            else None
        ),
        lambda rows, c: verify_logistic(rows, c["x"], c["y"]),
    ),
    _Check(
        "comparison",
        lambda rows, c: (
            "a group comparison needs a two-level group and a numeric value"
            if _has(c, "group", "value")
            and _two_level(rows, c["group"])
            and _numeric(rows, c["value"])
            else None
        ),
        lambda rows, c: verify_comparison(rows, c["group"], c["value"]),
    ),
    _Check(
        "groups",
        lambda rows, c: (
            "a k-group comparison needs a 3+ level group and a numeric value"
            if _has(c, "group", "value")
            and _multi_level(rows, c["group"])
            and _numeric(rows, c["value"])
            else None
        ),
        lambda rows, c: verify_groups(rows, c["group"], c["value"]),
    ),
    _Check(
        "equivalence",
        lambda rows, c: (
            "an equivalence test needs a two-level group, numeric value, and a sesoi"
            if _has(c, "group", "value", "sesoi")
            and _two_level(rows, c["group"])
            and _numeric(rows, c["value"])
            else None
        ),
        lambda rows, c: verify_equivalence(
            rows, c["group"], c["value"], sesoi=float(c["sesoi"])
        ),
    ),
    _Check(
        "experiment",
        lambda rows, c: (
            "an A/B test needs a two-level variant and a numeric metric"
            if _has(c, "variant", "metric")
            and _two_level(rows, c["variant"])
            and _numeric(rows, c["metric"])
            else None
        ),
        lambda rows, c: verify_experiment(
            rows, c["variant"], c["metric"], period=c.get("time", c.get("period"))
        ),
        veto=True,
    ),
    _Check(
        # multiple-comparisons: one group screened against a family of numeric outcomes
        "screen",
        lambda rows, c: (
            "a multiple-metric screen needs a two-level group and 2+ numeric outcomes"
            if c.get("group")
            and _two_level(rows, c["group"])
            and isinstance(c.get("outcomes"), list | tuple)
            and sum(1 for o in c["outcomes"] if _numeric(rows, o)) >= 2
            else None
        ),
        lambda rows, c: verify_screen(
            rows, c["group"], [str(o) for o in c["outcomes"]]
        ),
        certifying=True,
    ),
    _Check(
        # regression to the mean: an explicit before/after/group claim, or a
        # baseline-selected group detected behind an effect/comparison framing so the
        # trap is caught however the caller named the columns (defensive routing)
        "rtm",
        _rtm_applies,
        _rtm_run,
        certifying=True,
    ),
    _Check(
        "proportions",
        lambda rows, c: (
            "a categorical association needs two categorical columns"
            if _has(c, "group", "outcome")
            and _categorical(rows, c["group"])
            and _categorical(rows, c["outcome"])
            and not _numeric(rows, c["outcome"])
            else None
        ),
        lambda rows, c: verify_proportions(rows, c["group"], c["outcome"]),
    ),
    _Check(
        "trend",
        lambda rows, c: (
            "a trend needs a numeric time index and a numeric value"
            if _has(c, "time", "value")
            and _numeric(rows, c["time"])
            and _numeric(rows, c["value"])
            else None
        ),
        lambda rows, c: verify_trend(rows, c["time"], c["value"]),
    ),
    _Check(
        "did-pretrends",
        lambda rows, c: (
            "a DiD pre-trends check needs a two-level group, time, value and start"
            if _has(c, "group", "time", "value")
            and c.get("treatment_start") is not None
            and _two_level(rows, c["group"])
            and _numeric(rows, c["time"])
            and _numeric(rows, c["value"])
            else None
        ),
        lambda rows, c: verify_did(
            rows, c["group"], c["time"], c["value"], float(c["treatment_start"])
        ),
    ),
    _Check(
        "forecast",
        lambda rows, c: (
            "a forecast needs a time index, actuals and forecasts"
            if _has(c, "time", "actual", "forecast")
            and _numeric(rows, c["time"])
            and _numeric(rows, c["actual"])
            and _numeric(rows, c["forecast"])
            else None
        ),
        lambda rows, c: verify_forecast(
            rows,
            c["time"],
            c["actual"],
            c["forecast"],
            seasonal_period=int(c.get("seasonal_period") or 1),
        ),
    ),
    _Check(
        "classification",
        lambda rows, c: (
            "a classifier evaluation needs a binary target and predictions or scores"
            if _has(c, "y_true")
            and _binary(rows, c["y_true"])
            and (c.get("y_pred") or c.get("y_score"))
            else None
        ),
        lambda rows, c: verify_classification(
            rows, c["y_true"], y_pred=c.get("y_pred"), y_score=c.get("y_score")
        ),
    ),
    _Check(
        "calibration",
        lambda rows, c: (
            "a calibration check needs a probability column and a binary outcome"
            if _has(c, "probability", "outcome")
            and _probability(rows, c["probability"])
            and _binary(rows, c["outcome"])
            else None
        ),
        lambda rows, c: verify_calibration(rows, c["probability"], c["outcome"]),
    ),
    _Check(
        "survival",
        lambda rows, c: (
            "a survival comparison needs a time, a 0/1 event, and a group"
            if _has(c, "time", "event", "group")
            and _numeric(rows, c["time"])
            and _binary(rows, c["event"])
            and _categorical(rows, c["group"])
            else None
        ),
        lambda rows, c: verify_survival(rows, c["time"], c["event"], c["group"]),
    ),
    _Check(
        "fairness",
        lambda rows, c: (
            "a fairness check needs a group, a binary y_true and y_pred"
            if _has(c, "group", "y_true", "y_pred")
            and _categorical(rows, c["group"])
            and _binary(rows, c["y_true"])
            and _binary(rows, c["y_pred"])
            else None
        ),
        lambda rows, c: verify_fairness(rows, c["group"], c["y_true"], c["y_pred"]),
    ),
    _Check(
        "leakage",
        lambda rows, c: (
            "a leakage screen needs a target column"
            if _has(c, "target") and _numeric(rows, c["target"])
            else None
        ),
        lambda rows, c: verify_leakage(rows, c["target"], features=c.get("features")),
    ),
    _Check(
        "reliability",
        lambda rows, c: (
            "a reliability check needs three or more numeric scale items"
            if len(c.get("items") or []) >= 3
            and all(_numeric(rows, it) for it in c["items"])
            else None
        ),
        lambda rows, c: verify_reliability(rows, list(c["items"])),
    ),
    _Check(
        "normality",
        lambda rows, c: (
            "a normality test needs a numeric column"
            if _has(c, "column") and _numeric(rows, c["column"])
            else None
        ),
        lambda rows, c: verify_normality(rows, c["column"]),
    ),
    _Check(
        "missingness",
        lambda rows, c: (
            "a missing-data test needs a column with missing values"
            if _has(c, "column") and _has_missing(rows, c["column"])
            else None
        ),
        lambda rows, c: verify_missingness(rows, c["column"]),
    ),
    _Check(
        "instrument",
        lambda rows, c: (
            "an IV strength check needs instrument, treatment and outcome columns"
            if _has(c, "instrument", "treatment", "outcome")
            and _numeric(rows, c["instrument"])
            and _numeric(rows, c["treatment"])
            else None
        ),
        lambda rows, c: verify_iv(rows, c["instrument"], c["treatment"], c["outcome"]),
    ),
    _Check(
        "discontinuity",
        lambda rows, c: (
            "an RDD manipulation check needs a running variable and a cutoff"
            if _has(c, "running")
            and c.get("cutoff") is not None
            and _numeric(rows, c["running"])
            else None
        ),
        lambda rows, c: verify_rdd(rows, c["running"], float(c["cutoff"])),
    ),
    _Check(
        "proportional-hazards",
        lambda rows, c: (
            "a proportional-hazards check needs time, a 0/1 event, and a group"
            if _has(c, "time", "event", "group")
            and _numeric(rows, c["time"])
            and _binary(rows, c["event"])
            and _categorical(rows, c["group"])
            else None
        ),
        lambda rows, c: verify_proportional_hazards(
            rows, c["time"], c["event"], c["group"]
        ),
    ),
    _Check(
        "counts",
        lambda rows, c: (
            "a count-model check needs a non-negative integer count column"
            if _has(c, "count") and _is_count(rows, c["count"])
            else None
        ),
        lambda rows, c: verify_counts(rows, c["count"], predictors=c.get("predictors")),
    ),
    _Check(
        "overlap",
        lambda rows, c: (
            "an overlap check needs a binary treatment and covariates"
            if _has(c, "treatment")
            and _binary(rows, c["treatment"])
            and c.get("covariates")
            else None
        ),
        lambda rows, c: verify_overlap(rows, c["treatment"], list(c["covariates"])),
    ),
    _Check(
        "prediction",
        lambda rows, c: (
            "a predictive-accuracy check needs a numeric target and feature columns"
            if _has(c, "target")
            and (c.get("features"))
            and _numeric(rows, c["target"])
            and all(_numeric(rows, f) for f in c["features"])
            else None
        ),
        lambda rows, c: verify_prediction(
            rows,
            list(c["features"]),
            c["target"],
            # honor a declared entity or time role so the held-out split is grouped by
            # entity / ordered by time, not a cross-sectional partition that would let
            # the same entity (or the future) sit on both sides and inflate the skill
            time_order=str(c["time"]) if c.get("time") else None,
            group=str(c["group"]) if c.get("group") else None,
        ),
        # The predictive test IS the claim for {target, features}: if it cannot certify
        # held-out skill, a supporting screen (leakage) must not rescue it to sound.
        certifying=True,
    ),
    _Check(
        "clusters",
        # requires no target, so a "predict target from features" claim routes to
        # prediction, not clusters (the features role is otherwise shared)
        lambda rows, c: (
            "a cluster check needs two or more numeric features and no target"
            if len(c.get("features") or []) >= 2
            and not c.get("target")
            and all(_numeric(rows, f) for f in c["features"])
            else None
        ),
        lambda rows, c: verify_clusters(rows, list(c["features"])),
    ),
)


def _why_not(check: _Check, rows: Rows, claim: Claim) -> str:
    """A short reason a check does not apply (missing roles or wrong column shape)."""
    return {
        "effect": "needs numeric x, y",
        "spatial": "needs numeric latitude, longitude and a value",
        "correlation": "needs numeric x, y",
        "regression": "needs numeric x, y",
        "sensitivity": "needs numeric x, y",
        "logistic": "needs numeric x and binary y",
        "comparison": "needs a two-level group and numeric value",
        "groups": "needs a 3+ level group and numeric value",
        "equivalence": "needs a two-level group, numeric value and a sesoi bound",
        "experiment": "needs a two-level variant and numeric metric",
        "screen": "needs a two-level group and 2+ numeric outcome columns",
        "rtm": "needs before, after and a two-level (baseline-selected) group",
        "proportions": "needs two categorical columns",
        "trend": "needs a numeric time index and value",
        "did-pretrends": "needs group, time, value and a treatment start",
        "forecast": "needs time, actual and forecast columns",
        "classification": "needs a binary target and predictions/scores",
        "calibration": "needs a probability column and a binary outcome",
        "survival": "needs time, 0/1 event and group columns",
        "fairness": "needs a group and binary y_true and y_pred",
        "leakage": "needs a numeric target column",
        "reliability": "needs 3+ numeric scale items",
        "normality": "needs a numeric column",
        "missingness": "needs a column with missing values",
        "instrument": "needs instrument, treatment, outcome columns",
        "discontinuity": "needs a running variable and a cutoff",
        "proportional-hazards": "needs time, 0/1 event and a group",
        "counts": "needs a non-negative integer count column",
        "overlap": "needs a binary treatment and covariates",
        "prediction": "needs a numeric target and feature columns",
        "clusters": "needs 2+ numeric features and no target",
    }.get(check.name, "preconditions not met")
