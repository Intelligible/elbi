"""The verifying analysis loop: an LLM uses tools to reach an answer it cannot fake.

The loop follows the standard agentic shape (the model calls a tool, the result is fed
back, repeat until the model is done) with one structural constraint: a verified answer
can only come from ``derive``, which authors the model's analysis as a derivation
(compute-as-code plus a declared conclusion), gates its output on the verification
oracle, and, when sound, caches and persists it. A sound derivation records its
attestation and the following ``answer`` interprets it; an unsound one hands the model
the issue and the loop continues. ``run_code`` is only a scratchpad for exploration,
never the answer, so a reported conclusion is always a durable, cached derivation rather
than an ephemeral probe. When the data cannot support an effect, the model calls
``conclude_inconclusive`` and the loop ends without a verified claim.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from elbi_core import (
    CodeResult,
    SessionManager,
    SubprocessExecutor,
    analyze_structure,
    profile_columns,
)

from .llm import LLMClient, Step, ToolCall, ToolSpec, Transcript, Usage

#: The model may explore freely, but a directional claim can only *leave* the loop
#: through its verification gate, so every reported effect/comparison/correlation/
#: trend has passed the checks a careful reviewer would run.
DEFAULT_SYSTEM = """\
You are a data scientist answering a question about a dataset. Identify the kind of \
analysis it needs, explore to work it out, then produce the answer as a derivation.

As you work, think out loud in brief notes the user can see: before a tool call, say \
in a sentence what you are about to do and why; after its result, what you learned \
that shapes the next step. Keep it short and genuine (a running commentary, not a \
script), so the user follows your reasoning, not just a list of tools.

- Explore first with run_code: a scratchpad only, never the answer. Datasets are a \
`data` dict (name -> list of row dicts). describe_dataset shows a dataset's schema; \
structure_map shows which columns share information (candidate confounders). Use \
run_code to prototype the computation and get a feel for the data. The sandbox starts \
bare (only the standard library is importable), so name every third-party package the \
code imports in `deps` (`import numpy, pandas` needs `deps=['numpy','pandas']`; an EBM \
needs `['interpret']`). Declare them on the first call, not after a failure. Your \
run_code calls share a live session: variables, imports, and loaded data persist from \
one call to the next, and so do files you write, so build up your analysis \
incrementally like a notebook (load and clean once, fit a model, then keep using it) \
rather than repeating setup each call. Each call has a few-minute time budget, so keep \
exploration cheap: to sanity-check a model, fit it once on a subsample or with a small \
config (few folds, few bags): do not run a full cross-validation of a heavy ensemble \
here, it will time out. Leave the full fit to `derive`, which runs with its own budget \
and is what actually gets certified. If a call does time out the session resets, so \
reload your data and continue with a lighter step.
- Produce the answer with `derive`: author your analysis as a derivation, a function \
`def <name>(ctx): ...` that reads inputs via `ctx.input('dataset')` and returns the \
analytic ROWS your conclusion is about (a list of row dicts). It runs once in the \
sandbox, its output is gated on the verification oracle, and a sound derivation is \
cached and saved as a durable answer. The derivation sandbox is bare (standard \
library only), the same as run_code, so name every third-party package the source \
imports in `deps` (e.g. deps=['numpy','pandas','scikit-learn']) or it will not import. \
Declare the conclusion as `claim` (the \
output columns mapped to verification roles), so the oracle can check it. It re-runs \
your analysis on the rows you return, so return the analytic data your conclusion is \
about, not statistics you already computed. For a predictive model, that means return \
the FEATURE rows plus the target column and claim {target, features}; do NOT fit a \
model in the derivation at all: the oracle does the fitting, so any model you fit \
there is discarded (wasted compute) and a table of predictions you made cannot be \
verified. The oracle certifies that leakage-free held-out predictive skill EXISTS with \
a simple deterministic model (a conservative lower bound); it does not reproduce your \
exploratory model's score. So report the CERTIFIED skill as the verified result, and \
present any higher number your own model reached while exploring as exploratory, not \
certified. An unsound result comes back with the \
issue to fix; fix and derive again. Roles that fit:
  - effect / association of x on y → x, y (and `controls` to hold fixed)
  - a difference between groups → group, value
  - an A/B test result → variant, metric
  - a categorical association → group, outcome
  - a trend or stationarity over time → time, value
  - a forecast's usefulness → time, actual, forecast
  - a classifier's performance → y_true, y_pred (and y_score)
  - a model's predictive accuracy → target, features
  - probabilities as risks → probability, outcome
  - a survival difference → time, event, group
  - fairness across a group → group, y_true, y_pred
  - whether segments are real → features; a count model → count; leakage → target
- For a nonlinear relationship, keep the OUTPUT columns in their natural, readable \
units and declare the reshape in the claim's `transforms` \
(e.g. {'x':'log','y':'log'}): \
the oracle logs them to check the log-log form while the rows stay readable for the \
chart. Do not output pre-logged columns. A log-log slope is an elasticity (% per \
%), a log-outcome slope a % change per unit.
- Make the finding visual where it helps: output the columns a chart needs, then \
attach a `chart` (a Vega-Lite spec) to `answer` to draw them: any chart that fits the \
finding. If the question is geographic (it names latitude and longitude, location, or \
"where"), the answer is a MAP: aggregate to grid cells, output each cell's `latitude`, \
`longitude`, and value, and encode `latitude`/`longitude` so it draws on a real map, \
never a scatter on one coordinate. The chart draws the certified rows.
- Give the user the finding with `answer`, in your own words: interpret the verified \
estimate (sign and size, the units your transform implies, and what is held fixed), \
and state the caveats. If the verification's `stability` check reports the finding is \
fragile or gives a range across perturbations, report that range, never a single \
point as if it were certain. Your `answer` is verified when a sound `derive` precedes \
it; otherwise use it directly only for a question with no soundness oracle (a \
description, ranking, count), resting on figures you computed.
- The certified magnitudes are rendered FOR you: every conclusion you certified with \
`derive` is appended to your answer as a "Certified findings" table showing the \
oracle's own estimate. So in your `summary` prose, INTERPRET the finding (what it \
means, the caveats) but do not restate those numbers, and never present a figure from \
`run_code` as if it were certified: if you cite an exploratory number, label it \
exploratory. Each certified effect is a DIRECT effect given the controls you named, \
not necessarily the total effect: say what is held fixed, and do not read several \
variables' coefficients as interchangeable causal effects. For a "how does each \
variable affect y" question, author one `derive` per variable (each with its own \
controls) so every effect is certified and listed, rather than reading them off one \
model.
- When the user wants a reusable MODEL (score new records, keep versions, serve \
predictions), use `train_model`, not `derive`. For anything beyond raw columns, \
build the PIPELINE: first author the feature engineering as a `derive` derivation \
(cleaned, certified, durable rows), then `train_model` with `derivation` naming it. \
That records the feature code as the model\'s lineage, retrains automatically when \
the features change (if a retrain policy is set), and enables `predict` with \
`raw: true`, which applies the same certified feature code to raw records before \
scoring, so training and serving cannot skew apart. About `train_model` itself: \
AutoML searches the standard learners \
under a time budget, the report's metrics come from a held-out split the search never \
saw, and the model is registered in the registry (MLflow) as a version. The first \
sound version becomes `champion`; a later version needs `promote_model`, so compare \
its held-out metrics with the current champion's and promote only when the user \
accepts it. Score records with `predict`. Report the held-out metrics; label any \
exploratory score from run_code as exploratory. Use `derive` with claim {target, \
features} instead when the question is only whether predictive signal exists.
- If the data cannot support an answer, call conclude_inconclusive with the reason.
- Distinguish association from causation; the data cannot decide causal direction on \
its own. If the user attached context (a spec / data dictionary), read it for column \
meanings and for any identification assumption that a causal reading needs: a \
randomized treatment, the sampling mechanism, an instrument, a known confounder. When \
one applies, declare it in `derive`'s `assumptions` with its source; the finding is \
then causal only *under that stated, audited premise*, never silently. Do not invent \
assumptions the context does not state.
"""

#: The retry budget. A hard question (a high-cardinality or multi-column predictor,
#: several failed framings) needs room to explore and re-derive, so this is generous;
#: reaching it degrades to a final grounded conclusion rather than a dead end.
_MAX_STEPS = 20


@dataclass(frozen=True)
class AnswerResult:
    """The outcome of a run: a verified conclusion, an inconclusive call, or neither.

    ``verified`` is true only for a sound conclusion that passed the gate; its
    ``checks``/``data_hash``/``spec`` are the material a serving layer turns into a
    portable attestation.
    """

    verified: bool
    verdict: str  # "sound" | "inconclusive" | "unverified"
    narrative: str
    checks: tuple[tuple[str, str, str], ...] = ()  # (name, verdict, detail) per gate
    #: The certified output rows, carried so a serving layer can visualize exactly the
    #: data the oracle checked (a map of the verified surface, a scatter of the fit).
    rows: tuple[dict[str, Any], ...] = ()
    #: Declared, sourced premises the conclusion is conditioned on (from the user's
    #: context): recorded for audit, so a causal reading is explicitly "sound *given*
    #: this assumption", never a silent upgrade of the oracle's row-based verdict.
    assumptions: tuple[str, ...] = ()
    spec: dict[str, Any] | None = None
    data_hash: str | None = None
    steps: int = 0
    #: The model's chart for the finding: a small grammar-of-graphics spec (a `mark` and
    #: `encoding` channels mapping certified columns to visual channels), validated
    #: against ``rows`` before it is attached. None when the model asked for no chart.
    viz: dict[str, Any] | None = None
    #: Every conclusion the oracle certified this run (one per ``derive`` that passed).
    #: The certified-findings block in ``narrative`` is rendered from these, so the
    #: magnitudes the user sees are the oracle's, not the model's prose.
    findings: tuple[Finding, ...] = ()
    #: Token counts and cost totalled over every model call this run, so a serving layer
    #: can show what a turn spent and accumulate it per conversation.
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class Finding:
    """One certified conclusion within a run: the oracle's own estimate for a claim.

    A run may certify several (a "how does each variable affect y" question authors one
    per variable). Each carries the magnitude the oracle computed and a unit-aware
    label, so the answer reports the certified number, not one the model wrote;
    ``adjusted_for`` is what the estimate holds fixed, so it is stated as a direct
    effect given those controls rather than an unconditional total effect.
    """

    name: str
    claim: dict[str, Any]
    verdict: str
    estimate: float | None
    estimate_label: str | None
    adjusted_for: tuple[str, ...]
    checks: tuple[tuple[str, str, str], ...]
    data_hash: str | None
    rows: tuple[dict[str, Any], ...]
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeriveOutcome:
    """The result of authoring a derivation: the certified output, or why it was held.

    ``certified`` with a ``sound`` verdict means the derivation ran, its declared
    conclusion passed the oracle, and it was cached and persisted; ``rendered`` is its
    served output and ``checks``/``data_hash`` the attestation. Otherwise ``detail`` (a
    held or unsound verdict) or ``error`` (a sandbox failure) says what to fix.
    """

    certified: bool
    verdict: str | None = None
    rendered: str = ""
    checks: tuple[tuple[str, str, str], ...] = ()
    data_hash: str | None = None
    detail: str = ""
    error: str | None = None
    #: The data-contract verdict and the failing clause, when the derivation declared a
    #: contract (a cleaning step). ``contract_verdict`` is None when none was declared.
    contract_verdict: str | None = None
    contract_detail: str = ""
    #: The derivation's raw output rows (the certified analytic data), so a serving
    #: layer can render a visualization of exactly what the oracle checked.
    rows: tuple[dict[str, Any], ...] = ()
    #: The magnitude the oracle computed, a unit-aware description of it, and the
    #: columns it holds fixed. Carried so the reported number is the certified one, not
    #: a figure the model wrote in exploratory code; ``estimate``/``estimate_label`` are
    #: None when the claim has no scalar estimate (or was not certified).
    estimate: float | None = None
    estimate_label: str | None = None
    adjusted_for: tuple[str, ...] = ()
    #: The derivation's content-addressed data version (a hash of code, params, and
    #: input versions). The run's input identity: an identical re-derive yields the same
    #: version, a changed input or model a new one, so a tracking layer keys on it.
    derivation_version: str | None = None


#: Injected by the app: author a derivation from (name, source, claim roles, serve
#: format, declared assumptions, third-party deps), certify its output against the
#: oracle, cache and persist it, and return the outcome. The assumptions are the model's
#: declared, sourced premises (e.g. "treatment randomized, per spec §2"): they are
#: recorded with the derivation for audit, not consumed by the oracle, which still
#: checks the rows. The deps are packages the source imports (numpy, scikit-learn),
#: provisioned into the sandbox so a certified derivation can use them. Kept as a seam
#: so the runtime carries none of the authoring/persistence stack: a run with no
#: ``derive`` capability simply cannot reach a verified answer. Positional arguments:
#: name, source, claim (conclusion roles, or None), contract (a DataContract manifest
#: the output must satisfy, or None), serve format, declared assumptions, and
#: third-party deps.
DeriveFn = Callable[
    [
        str,
        str,
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
        str,
        Sequence[str],
        Sequence[str],
    ],
    DeriveOutcome,
]

#: Launch a derivation as a durable background job instead of running it inline; returns
#: the job id at once. Same authoring contract as ``DeriveFn``, so a long model fit is
#: submitted and the run continues rather than blocking on it. ``None`` where background
#: jobs are unavailable, in which case ``derive`` always runs inline.
SubmitFn = Callable[
    [
        str,
        str,
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
        str,
        Sequence[str],
        Sequence[str],
    ],
    str,
]

#: Launch a long ``run_code`` snippet as a background job; returns the job id at once.
#: It runs in its own isolated session (the live session is single-threaded), sharing
#: the conversation's workspace directory, so files the snippet writes are there for
#: later calls. ``None`` where background jobs are unavailable.
SubmitCodeFn = Callable[[str, Sequence[str]], str]

#: Injected by the app: train an AutoML model over a dataset and register it in the
#: model registry, returning the rendered training report (or an ``error:`` message
#: the model can act on). Takes one normalized spec mapping with keys: ``name``,
#: ``dataset``, ``target``, ``features`` (tuple; empty means all others), ``task``
#: (classification/regression/ts_forecast/auto), ``time_budget`` (seconds),
#: ``metric`` (None means the task's standard one), ``ensemble`` (bool), and for
#: forecasting ``time_col``/``horizon``. Kept as a seam like ``DeriveFn`` so the
#: runtime carries none of the MLflow/AutoML stack.
TrainFn = Callable[[Mapping[str, Any]], str]

#: Render the model registry: every registered model with its versions and champion.
ListModelsFn = Callable[[], str]

#: Score rows with a registered model: (model reference, rows, raw) -> rendered
#: predictions. The reference is a name (champion or newest), ``name@alias``, or
#: ``name/version``. ``raw`` means the rows are source-shaped and the model's
#: recorded feature derivation must engineer them first (the skew-proof path).
PredictFn = Callable[[str, Sequence[Mapping[str, Any]], bool], str]

#: Point a registered model's alias at a version (the champion/challenger move):
#: (name, version, alias) -> confirmation.
PromoteFn = Callable[[str, int, str], str]

#: Score a whole source with a registered model: (model reference, source name,
#: source kind: "dataset"|"derivation") -> rendered summary; the full predictions
#: land as a run artifact server-side.
BatchScoreFn = Callable[[str, str, str], str]

#: The notebook-operating capabilities, injected by the app so the agent can drive the
#: same notebook surface a human uses. Kept as seams like ``DeriveFn`` so the runtime
#: carries none of the kernel/persistence stack. A ``None`` capability makes its tool
#: decline. Notebooks are an authoring surface: creating and running them is autonomous
#: : (sandboxed), while promoting a cell goes through the same certification gate
#: ``derive`` : does, so the gate, not the agent, is what authorizes a durable artifact.

#: Create or replace a notebook: (notebook_id or None to create, name, cells as
#: {cell_type, source} objects, deps or None) -> rendered summary with its id.
NotebookWriteFn = Callable[
    [str | None, str, Sequence[Mapping[str, Any]], Sequence[str] | None], str
]

#: Run a notebook and read its outputs back: (notebook_id, cell ids or None for all) ->
#: rendered outputs/tracebacks so the agent can self-correct.
NotebookRunFn = Callable[[str, Sequence[str] | None], str]

#: Read a notebook's cells, outputs, environment, and schedule: (notebook_id) -> render.
NotebookReadFn = Callable[[str], str]

#: List the notebooks that exist: () -> render (name, id, cell count).
NotebookListFn = Callable[[], str]

#: Promote a notebook cell's ``def <name>(ctx)`` to a certified derivation:
#: (notebook_id, cell_id) -> rendered outcome (gated server-side).
NotebookPromoteFn = Callable[[str, str], str]

#: Schedule a notebook to rerun: (notebook_id, mode "interval"|"on_data_change",
#: interval_hours or None, dataset or None) -> confirmation.
NotebookScheduleFn = Callable[[str, str, float | None, str | None], str]

#: The dashboard-operating capabilities, injected by the app so the agent can build the
#: same dashboards a human does. Seams like ``DeriveFn``; a ``None`` capability makes
#: its tool decline. Authoring is autonomous (the agent composes the declarative spec),
#: but publishing is gated: a dashboard serves a number only once every derivation it
#: binds is certified, so the gate, not the agent, is what lets a tile reach a viewer.

#: List the dashboards that exist: () -> render (title, id, status, version).
DashboardListFn = Callable[[], str]

#: List the certified derivations a widget can bind: () -> render (name, params).
DashboardSourcesFn = Callable[[], str]

#: Read a dashboard's variables, pages, and widgets: (dashboard_id) -> render.
DashboardReadFn = Callable[[str], str]

#: Create a dashboard (id None) or replace one's spec: (dashboard_id or None, spec
#: manifest) -> rendered summary with its id and URL, or validation errors to fix.
DashboardWriteFn = Callable[[str | None, Mapping[str, Any]], str]

#: Publish a dashboard, gated on certified bindings: (dashboard_id) -> outcome.
DashboardPublishFn = Callable[[str], str]

#: The feature-store capabilities, injected by the app so the agent can operate the same
#: feature store a human does. Seams like ``DeriveFn``; a ``None`` capability makes its
#: tool decline. Serving is gated: a feature view's source derivation must be certified.

#: List the feature views, their keys, and certification: () -> render.
FeatureListFn = Callable[[], str]

#: Define a feature view: (name, entities, source, features or None, timestamp field or
#: None, ttl seconds or None) -> outcome.
FeatureDefineFn = Callable[
    [str, Sequence[str], str, Sequence[str] | None, str | None, int | None], str
]

#: Materialize the online store: (feature view names or None for all) -> outcome.
FeatureMaterializeFn = Callable[[Sequence[str] | None], str]

#: Read latest online features: (feature refs, entity rows) -> rendered rows.
FeatureOnlineFn = Callable[[Sequence[str], Sequence[Mapping[str, Any]]], str]

#: Point-in-time historical features: (feature refs, entity dataframe) -> rendered rows.
FeatureHistoricalFn = Callable[[Sequence[str], Sequence[Mapping[str, Any]]], str]

#: Profile a feature view now: (feature view, set baseline) -> outcome.
FeatureStatisticsFn = Callable[[str, bool], str]

#: Check a feature view for drift against its baseline: (feature view) -> outcome.
FeatureDriftFn = Callable[[str], str]

#: Verify a feature view against its data contract: (feature view) -> outcome.
FeatureExpectationFn = Callable[[str], str]

#: Materialize a point-in-time join into a named training set:
#: (name, feature refs, entity dataframe, label or None) -> outcome.
FeatureTrainingSetFn = Callable[
    [str, Sequence[str], Sequence[Mapping[str, Any]], str | None], str
]

#: The lineage/catalog capabilities, injected by the app so the agent can search the
#: catalog and trace provenance/impact across artifacts. A ``None`` capability declines.
#: Each takes one string (a query or a node reference) and returns rendered text.
LineageQueryFn = Callable[[str], str]

#: The semantic-layer metric capabilities, injected by the app so the agent can define,
#: list, and query metrics over certified derivations. A ``None`` capability declines.
MetricListFn = Callable[[], str]
MetricDefineFn = Callable[[Mapping[str, Any]], str]
MetricQueryFn = Callable[[str, Sequence[str], str | None], str]

#: The monitoring capabilities, injected by the app so the agent can set up an anomaly
#: monitor over a certified target and report what is watched. ``None`` declines.
MonitorListFn = Callable[[], str]
MonitorCreateFn = Callable[[Mapping[str, Any]], str]

#: Report asset freshness: () -> rendered stale/materialized listing.
OrchestrationStatusFn = Callable[[], str]

#: Materialize assets in dependency order: (selection, comma-separated names) -> result.
OrchestrationMaterializeFn = Callable[[str, str], str]


@dataclass
class _RunState:
    """Run-scoped state the loop threads through the tools.

    ``executed`` records whether any code has run (what grounds a plain answer);
    ``findings`` accumulates every conclusion the oracle certified this run, keyed by
    derivation name (a re-derive of the same name replaces its entry), so a following
    ``answer`` reports all of them with the oracle's own estimates rather than
    collapsing to the last one. ``verified`` is the aggregate attestation over those
    findings (or a pre-seeded one from a finished background derivation), so a
    plain-prose answer or the budget path still returns verified when a report backs it.
    """

    executed: bool = False
    verified: AnswerResult | None = None
    findings: dict[str, Finding] = field(default_factory=dict)
    #: Token counts and cost accumulated across every model call this run.
    usage: Usage = field(default_factory=Usage)


_MISSING_MODULE_RE = re.compile(r"No module named '([\w.]+)'")


def _missing_dep_hint(error: str) -> str:
    """Turn a missing-import error into a hint to declare the package in ``deps``.

    The sandbox is bare by design; a third-party import must be provisioned on demand.
    Naming the package back to the model lets it fix the call in one step, declare the
    dependency, rather than retrying blind.
    """
    match = _MISSING_MODULE_RE.search(error)
    if not match:
        return ""
    package = match.group(1).split(".")[0]
    return (
        f"\n\nThe sandbox starts bare; provision the library by naming it in `deps`, "
        f"e.g. deps=['{package}']. If its install name differs from the import, use "
        "install name (import sklearn -> deps=['scikit-learn'])."
    )


#: The chart the model may attach to `answer` is a full Vega-Lite spec (any mark,
#: transform, layer, or facet), so any visualization is expressible, not a fixed menu.
#: The schema is deliberately open (we supply the data and validate field references);
#: constraining it to a mark list is what we are avoiding.
_CHART_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "A Vega-Lite specification (without data: we inject the certified rows). Use "
        "any mark and encoding; reference only the derivation's output columns."
    ),
}
#: The top-level keys that make a Vega-Lite spec renderable (a mark or a composition).
_CHART_ROOTS = ("mark", "layer", "facet", "hconcat", "vconcat", "concat", "repeat")


def _collect(obj: Any, key: str) -> set[str]:
    """Every string value stored under ``key`` anywhere in a nested spec."""
    found: set[str] = set()
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                found.add(v)
            else:
                found |= _collect(v, key)
    elif isinstance(obj, list | tuple):
        for item in obj:
            found |= _collect(item, key)
    return found


def _validate_chart(chart: Any, rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Check a Vega-Lite spec against the certified columns; error, or None if OK.

    Any Vega-Lite chart is allowed; the only constraints are that it is renderable and
    that every field it reads is a certified output column or one a transform derived,
    so the chart is always a faithful view of the verified rows and never invents data.
    The error names the available columns so the model fixes the spec in one step.
    """
    if not isinstance(chart, Mapping):
        return "expected a Vega-Lite spec object."
    if not any(k in chart for k in _CHART_ROOTS):
        return (
            "a chart needs a `mark` (with `encoding`) or a composition "
            "(layer/facet/concat/repeat)."
        )
    columns = set(rows[0]) if rows else set()
    derived = _collect(chart, "as")  # fields produced by transforms
    phantom = sorted(_collect(chart, "field") - columns - derived)
    if phantom:
        return (
            f"references columns not in the output: {', '.join(phantom)}. "
            f"Available columns: {', '.join(sorted(columns)) or '(none)'}."
        )
    return None


@dataclass
class Workspace:
    """The datasets a run operates over, and the tool operations against them.

    ``datasets`` maps a name to its rows (string-valued, as loaded from CSV/SQL).
    The executor is the same hardened sandbox the rest of the project uses for
    untrusted code.
    """

    datasets: dict[str, list[dict[str, str]]]
    executor: SubprocessExecutor = field(default_factory=SubprocessExecutor)
    #: The app injects this to let the run author derivations; ``None`` in a bare
    #: workspace (e.g. a test that only exercises exploration), where ``derive`` is
    #: unavailable and no verified answer can be produced.
    derive: DeriveFn | None = None
    #: Injected alongside ``derive`` to launch a derivation as a background job (a long
    #: training run) instead of blocking on it; ``None`` means background is unavailable
    #: and ``derive`` always runs inline.
    submit: SubmitFn | None = None
    #: The model-lifecycle capabilities, injected by the app when the ML stack is
    #: installed: train-and-register, list the registry, score with a registered
    #: model, and move an alias. ``None`` (a bare workspace, or the extra missing)
    #: makes the corresponding tools decline with the reason.
    train: TrainFn | None = None
    list_models: ListModelsFn | None = None
    predict_model: PredictFn | None = None
    promote_model: PromoteFn | None = None
    batch_score: BatchScoreFn | None = None
    #: The notebook-operating capabilities, injected by the app so the agent can build,
    #: run, inspect, promote from, and schedule notebooks: the same surface a human
    #: drives in the UI. ``None`` (a bare workspace) makes the notebook tools decline.
    nb_write: NotebookWriteFn | None = None
    nb_run: NotebookRunFn | None = None
    nb_read: NotebookReadFn | None = None
    nb_list: NotebookListFn | None = None
    nb_promote_cell: NotebookPromoteFn | None = None
    nb_schedule: NotebookScheduleFn | None = None
    #: The dashboard-operating capabilities, injected by the app so the agent can list,
    #: read, author, and publish dashboards: the same surface a human drives in the UI.
    #: ``None`` (a bare workspace) makes the dashboard tools decline.
    dash_list: DashboardListFn | None = None
    dash_sources: DashboardSourcesFn | None = None
    dash_read: DashboardReadFn | None = None
    dash_write: DashboardWriteFn | None = None
    dash_publish: DashboardPublishFn | None = None
    #: The feature-store capabilities, injected by the app so the agent can discover,
    #: define, materialize, and retrieve features. ``None`` makes the tools decline.
    fs_list: FeatureListFn | None = None
    fs_define: FeatureDefineFn | None = None
    fs_materialize: FeatureMaterializeFn | None = None
    fs_online: FeatureOnlineFn | None = None
    fs_historical: FeatureHistoricalFn | None = None
    fs_statistics: FeatureStatisticsFn | None = None
    fs_drift: FeatureDriftFn | None = None
    fs_expectations: FeatureExpectationFn | None = None
    fs_training_set: FeatureTrainingSetFn | None = None
    #: The lineage/catalog capabilities, injected by the app so the agent can search
    #: and trace provenance/impact. ``None`` makes the tools decline.
    lin_catalog: LineageQueryFn | None = None
    lin_lineage: LineageQueryFn | None = None
    lin_impact: LineageQueryFn | None = None
    #: The semantic-layer metric capabilities, injected by the app so the agent can
    #: define, list, and query metrics. ``None`` makes the tools decline.
    metric_list: MetricListFn | None = None
    metric_define: MetricDefineFn | None = None
    metric_query: MetricQueryFn | None = None
    #: The monitoring capabilities, injected by the app so the agent can create and list
    #: anomaly monitors. ``None`` makes the tools decline.
    monitor_list: MonitorListFn | None = None
    monitor_create: MonitorCreateFn | None = None
    #: The orchestration capabilities, injected by the app so the agent can see stale
    #: assets and materialize them. ``None`` makes the tools decline.
    orch_status: OrchestrationStatusFn | None = None
    orch_materialize: OrchestrationMaterializeFn | None = None
    #: Injected to launch a long ``run_code`` snippet as a background job (isolated
    #: session, shared workspace files); ``None`` means ``run_code`` always runs inline.
    submit_code: SubmitCodeFn | None = None
    #: A persistent scratch directory shared by this run's ``run_code`` calls, so a
    #: file one call writes is there for the next (a fitted model, an intermediate
    #: table). Scoped to one conversation by the app; ``None`` gives each call a
    #: throwaway directory. It reaches only ``run_code``: ``derive`` stays hermetic.
    scratch_dir: Path | None = None
    #: Whether ``run_code`` runs in a persistent session (a live namespace across calls,
    #: so a variable or fitted model stays in memory) rather than a fresh process each
    #: call. The app enables it; the default is off so a bare workspace stays one-shot.
    stateful: bool = False
    #: The sandbox backend for the stateful session: ``"subprocess"`` (a host process)
    #: or ``"docker"`` (an isolated container where ``run_code`` and ``bash`` share it).
    backend: str = "subprocess"
    #: Outbound network policy for the docker session: ``"full"``, ``"none"``, or a list
    #: of allowlisted hosts (ignored by the subprocess backend).
    egress: str | Sequence[str] = "full"
    #: Container image for the docker backend; ``None`` uses the slim default.
    image: str | None = None
    #: The session manager, created lazily on the first stateful call and torn down by
    #: :meth:`close`. It owns the live session, its deps, and the backend choice.
    _sessions: SessionManager | None = field(default=None, init=False, repr=False)

    def describe(self, name: str) -> str:
        """Render a dataset's shape: columns, row count, and an example row."""
        rows = self.datasets.get(name)
        if not rows:
            return f"No dataset named {name!r}. Available: {', '.join(self.datasets)}."
        columns = list(rows[0])
        return (
            f"# Dataset: {name}\n{len(rows)} rows; columns: {', '.join(columns)}\n"
            f"example row: {rows[0]}\nValues are strings; cast them (float(...))."
        )

    def profile(self, name: str) -> str:
        """Render a dataset's per-column data-quality profile, for contract authoring.

        Per column: completeness, distinct count, inferred type, numeric range, and the
        most common values. Read this before declaring a cleaning derivation's contract.
        """
        rows = self.datasets.get(name)
        if not rows:
            return f"No dataset named {name!r}. Available: {', '.join(self.datasets)}."
        profiles = json.dumps([p.to_dict() for p in profile_columns(rows)], indent=2)
        return (
            f"# Profile: {name}\n{len(rows)} rows. Per-column completeness, distinct "
            f"count, inferred type, range, and top values:\n{profiles}"
        )

    def structure(self, name: str) -> str:
        """Render how the dataset's columns relate (candidate confounders, derived)."""
        rows = self.datasets.get(name)
        if not rows:
            return f"No dataset named {name!r}. Available: {', '.join(self.datasets)}."
        return analyze_structure(rows, list(rows[0])).render()

    def run_code(self, code: str, deps: Sequence[str] = ()) -> str:
        """Run exploratory code in the sandbox; return its output or the error.

        ``deps`` names third-party packages the code imports; they are provisioned
        on demand (e.g. ``interpret`` for an EBM), so any library can be pulled.
        A missing import is turned into a hint to declare it in ``deps``, so the model
        fixes the environment in one step rather than guessing.

        In a stateful workspace the call runs in a persistent session, so a variable or
        import from a prior call is still defined. Otherwise each call is a fresh
        process and only files written under :attr:`scratch_dir` carry to the next call.
        """
        if self.stateful:
            result = self._session_manager().run_code(code, deps)
        else:
            result = self.executor.run_code(
                code, self.datasets, deps=deps, workspace=self.scratch_dir
            )
        return self._format(result)

    def run_bash(self, command: str) -> str:
        """Run a shell command in the session, sharing its filesystem and packages.

        A shell needs the isolated session, so it is only available in a stateful
        workspace; the docker backend runs it in the same container as ``run_code`` (so
        an ``apt-get`` or ``pip install`` here is usable by the next ``run_code``),
        while the host backend declines it, as a host shell would touch the user's box.
        """
        if not self.stateful:
            return "error:\nbash requires a stateful sandbox session"
        return self._format(self._session_manager().run_bash(command))

    def _session_manager(self) -> SessionManager:
        """The lazily-created session manager for this run's exploration."""
        if self._sessions is None:
            self._sessions = SessionManager(
                datasets=self.datasets,
                workspace=self.scratch_dir,
                backend=self.backend,
                egress=self.egress,
                image=self.image,
            )
        return self._sessions

    def _format(self, result: CodeResult) -> str:
        """Render a code/bash result as the model reads it (stdout/result/error)."""
        if result.error:
            return f"error:\n{result.error}{_missing_dep_hint(result.error)}"
        parts = []
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.result is not None:
            parts.append(f"result: {result.result}")
        return "\n".join(parts) or "(no output)"

    def close(self) -> None:
        """Tear down the exploration session, if one was started."""
        if self._sessions is not None:
            self._sessions.close()
            self._sessions = None


@dataclass(frozen=True)
class Event:
    """A progress event from a run, so the UI can show the model's work as it happens.

    ``reasoning`` is the model's own prose that turn (its thinking before a tool call);
    ``tool`` is a call with its ``arguments``; ``tool_result`` carries a tool's
    ``output``; ``result`` is the final answer.
    """

    kind: str  # "reasoning" | "reasoning_delta" | "tool" | "tool_result" | "result"
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    output: str | None = None
    text: str | None = None
    result: AnswerResult | None = None


def _run_step(
    client: LLMClient, transcript: Transcript
) -> Generator[Event, None, Step]:
    """Drive one model turn, returning the assembled :class:`Step`.

    When the client can stream (it has a ``stream`` method), yield a ``reasoning_delta``
    event for each text slice as the model produces it, so the UI shows the model
    thinking live; otherwise fall back to a single blocking ``step`` call.
    """
    stream_fn = getattr(client, "stream", None)
    if stream_fn is None:
        return client.step(transcript, _TOOLSPECS)
    final: Step | None = None
    for item in stream_fn(transcript, _TOOLSPECS):
        if isinstance(item, Step):
            final = item
        else:  # a StepDelta: incremental text from the model
            yield Event("reasoning_delta", text=item.text)
    # A streaming client always ends with a Step; fall back defensively if it did not.
    return final if final is not None else client.step(transcript, _TOOLSPECS)


def stream(
    question: str,
    workspace: Workspace,
    client: LLMClient,
    *,
    context: str = "",
    history: Sequence[tuple[str, str]] = (),
    memory: str = "",
    summary: str = "",
    images: Sequence[str] = (),
    verified: AnswerResult | None = None,
    max_steps: int = _MAX_STEPS,
) -> Iterator[Event]:
    """Run the loop, yielding a tool event per call and a final result event.

    The bounded step count is the retry budget that keeps a stuck model from looping
    forever; reaching it yields an unverified result rather than a guess. The final
    event always has ``kind == "result"``. ``context`` is a user-provided spec or data
    dictionary about the data; it is given to the model to read (it enriches the
    meaning it works with and can supply identification assumptions to declare), never
    as ground truth the oracle trusts.

    ``history`` is the prior turns of this conversation as ``(role, text)`` pairs (a
    caller-capped sliding window), replayed so references like "it" resolve and the
    model builds on earlier work instead of restarting; tool traces are deliberately not
    replayed. ``memory`` is a compact note of the durable results already established
    (the conversation's certified derivations), the extracted, structured memory that
    survives compaction. Both are context for the model, never trusted by the oracle,
    which still re-checks every new claim from the rows.

    ``verified`` pre-seeds the run with an already-certified attestation, so a turn that
    only narrates a result the oracle already checked (a finished background derivation)
    produces a verified ``answer`` without re-deriving. It is the machine-checked
    attestation from that prior sound derivation, not something the model can fabricate.
    """
    transcript = Transcript(system=DEFAULT_SYSTEM)
    if summary.strip():
        transcript.add_user_text(
            "Summary of earlier turns in this conversation (older messages were "
            "condensed to fit context; treat as background, not verified fact):"
            f"\n\n{summary.strip()}"
        )
    for role, text in history:
        if not text.strip():
            continue
        if role == "assistant":
            transcript.add_assistant_text(text)
        else:
            transcript.add_user_text(text)
    if memory.strip():
        transcript.add_user_text(
            "Durable results already established in this conversation (certified "
            "derivations you can build on or reference); re-derive if you change one:"
            f"\n\n{memory.strip()}"
        )
    if context.strip():
        transcript.add_user_text(
            "Context the user attached about this data (a spec / data dictionary). "
            "Use it to read column meanings and to spot any assumptions worth "
            "declaring on `derive`: it is not verified; the oracle still checks the "
            f"rows.\n\n{context.strip()}"
        )
    transcript.add_user_message(question, images)
    state = _RunState()
    if verified is not None:
        state.verified = verified
    streams = getattr(client, "stream", None) is not None
    for step_index in range(1, max_steps + 1):
        step = yield from _run_step(client, transcript)
        state.usage = state.usage + (step.usage or Usage())
        transcript.add_assistant(step)
        # Surface the model's reasoning this turn (its thinking alongside the tool
        # calls), so the trace shows *why* it took each step, not just what it ran. A
        # streaming client already emitted this text as reasoning_delta events, so only
        # emit the whole-text event when it did not stream.
        if step.text and step.tool_calls and not streams:
            yield Event("reasoning", text=step.text)
        if not step.tool_calls:
            # The model answered in prose with no tool call. If a sound report backs it,
            # that prose is the verified finding; otherwise it is not grounded.
            final = _finalize(state, step.text or "", step_index)
            yield Event("result", result=replace(final, usage=state.usage))
            return
        results: list[tuple[str, str]] = []
        for call in step.tool_calls:
            yield Event("tool", tool=call.name, arguments=call.arguments)
            output, terminal = _dispatch(call, workspace, step_index, state)
            if call.name == "run_code" and not output.startswith("error"):
                state.executed = True
            if output:
                yield Event("tool_result", tool=call.name, output=output)
            if terminal is not None:
                yield Event("result", result=replace(terminal, usage=state.usage))
                return
            results.append((call.id, output))
        transcript.add_tool_results(results)
    # Budget reached. A sound derivation still wins. Otherwise give the model one final
    # turn to conclude with what it explored, a grounded answer or an inconclusive call,
    # so a hard question ends in the best available answer, not a dead stop.
    if state.verified is not None:
        yield Event("result", result=replace(state.verified, usage=state.usage))
        return
    transcript.add_user_text(
        "You have reached the step budget; do not call exploratory tools again. Give "
        "the user your best answer now: `answer` with what you found (grounded in the "
        "figures you computed), or `conclude_inconclusive` with why the data could not "
        "settle it."
    )
    closing = client.step(transcript, _TOOLSPECS)
    state.usage = state.usage + (closing.usage or Usage())
    for call in closing.tool_calls:
        yield Event("tool", tool=call.name, arguments=call.arguments)
        _, terminal = _dispatch(call, workspace, max_steps, state)
        if terminal is not None:
            yield Event("result", result=replace(terminal, usage=state.usage))
            return
    verdict = "grounded" if state.executed else "unverified"
    yield Event(
        "result",
        result=AnswerResult(
            verified=False,
            verdict=verdict,
            narrative=closing.text
            or "Reached the step budget without a verified answer.",
            steps=max_steps,
            usage=state.usage,
        ),
    )


def _finalize(state: _RunState, text: str, step_index: int) -> AnswerResult:
    """Turn the model's free text into a result: verified if a sound report backs it."""
    if state.verified is not None:
        return replace(state.verified, narrative=text, steps=step_index)
    return AnswerResult(False, "unverified", text, steps=step_index)


def answer(
    question: str,
    workspace: Workspace,
    client: LLMClient,
    *,
    context: str = "",
    history: Sequence[tuple[str, str]] = (),
    memory: str = "",
    summary: str = "",
    images: Sequence[str] = (),
    verified: AnswerResult | None = None,
    max_steps: int = _MAX_STEPS,
) -> AnswerResult:
    """Run the loop to completion and return the final result (see :func:`stream`)."""
    final = AnswerResult(False, "unverified", "no result produced", steps=0)
    for event in stream(
        question,
        workspace,
        client,
        context=context,
        history=history,
        memory=memory,
        summary=summary,
        images=images,
        verified=verified,
        max_steps=max_steps,
    ):
        if event.kind == "result" and event.result is not None:
            final = event.result
    return final


def _dispatch(
    call: ToolCall, workspace: Workspace, step_index: int, state: _RunState
) -> tuple[str, AnswerResult | None]:
    """Run a tool call; return its output and a terminal result if the loop ends."""
    args = call.arguments
    if call.name == "describe_dataset":
        return workspace.describe(str(args.get("dataset", ""))), None
    if call.name == "profile_dataset":
        return workspace.profile(str(args.get("dataset", ""))), None
    if call.name == "structure_map":
        return workspace.structure(str(args.get("dataset", ""))), None
    if call.name == "run_code":
        deps = [str(d) for d in (args.get("deps") or [])]
        code = str(args.get("code", ""))
        if args.get("background") and workspace.submit_code is not None:
            job_id = workspace.submit_code(code, deps)
            return (
                f"Launched a background run_code job {job_id}. It runs in its own "
                "isolated session (not this live one), so have it WRITE any result to "
                "a file in the workspace; a later call here can read it. Continue, and "
                "tell the user it is running.",
                None,
            )
        return workspace.run_code(code, deps), None
    if call.name == "bash":
        return workspace.run_bash(str(args.get("command", ""))), None
    if call.name == "answer":
        # The final answer, in the model's own words. It is verified when a sound
        # report backs it (the model interprets the oracle's estimate); otherwise it is
        # grounded if it rests on executed code, else not trusted.
        summary = str(args.get("summary", ""))
        findings = list(state.findings.values())
        # When the run certified findings, the system renders their estimates into the
        # answer itself, so the magnitudes shown are the oracle's, not the model prose.
        # The model's summary is the interpretation around that block.
        if findings:
            block = _render_certified_block(findings)
            narrative = f"{summary.rstrip()}\n\n{block}" if summary.strip() else block
            base = state.verified or _aggregate(state.findings)
        else:
            narrative = summary
            base = state.verified or AnswerResult(
                False, "grounded" if state.executed else "unverified", ""
            )
        viz = None
        chart = args.get("chart")
        if chart is not None and base.rows:
            # Validate the model's chart against the certified columns; a bad spec is
            # returned for the model to fix (self-correction), not silently dropped. The
            # columns are the primary finding's, so a chart cannot mix fields from
            # different derivations (which would reference data not in one row set).
            error = _validate_chart(chart, base.rows)
            if error is not None:
                return f"The chart is invalid: {error}", None
            viz = chart
        return "", replace(base, narrative=narrative, steps=step_index, viz=viz)
    if call.name == "conclude_inconclusive":
        return "", AnswerResult(
            verified=False,
            verdict="inconclusive",
            narrative=str(args.get("reason", "")),
            steps=step_index,
        )
    if call.name == "derive":
        return _run_derive(args, workspace, step_index, state)
    if call.name == "train_model":
        return _run_train(args, workspace), None
    if call.name == "list_models":
        if workspace.list_models is None:
            return _NO_ML, None
        return workspace.list_models(), None
    if call.name == "predict":
        if workspace.predict_model is None:
            return _NO_ML, None
        rows = args.get("rows")
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            return "predict needs `rows`: a list of feature-record objects.", None
        return workspace.predict_model(
            str(args.get("model", "")), rows, bool(args.get("raw", False))
        ), None
    if call.name == "batch_score":
        if workspace.batch_score is None:
            return _NO_ML, None
        model = str(args.get("model", "")).strip()
        dataset = str(args.get("dataset", "")).strip()
        derivation = str(args.get("derivation", "")).strip()
        if dataset and derivation:
            return "batch_score takes either `dataset` or `derivation`.", None
        source = derivation or dataset
        if not model or not source:
            return (
                "batch_score needs `model` and a source (`dataset` or `derivation`).",
                None,
            )
        kind = "derivation" if derivation else "dataset"
        return workspace.batch_score(model, source, kind), None
    if call.name == "promote_model":
        if workspace.promote_model is None:
            return _NO_ML, None
        try:
            version = int(args.get("version", 0))
        except (TypeError, ValueError):
            return "promote_model needs an integer `version`.", None
        return workspace.promote_model(
            str(args.get("name", "")), version, str(args.get("alias") or "champion")
        ), None
    if call.name in _NOTEBOOK_TOOLS:
        return _dispatch_notebook(call, workspace), None
    if call.name in _DASHBOARD_TOOLS:
        return _dispatch_dashboard(call, workspace), None
    if call.name in _FEATURE_TOOLS:
        return _dispatch_features(call, workspace), None
    if call.name in _LINEAGE_TOOLS:
        return _dispatch_lineage(call, workspace), None
    if call.name in _METRIC_TOOLS:
        return _dispatch_metrics(call, workspace), None
    if call.name in _MONITOR_TOOLS:
        return _dispatch_monitors(call, workspace), None
    if call.name in _ORCHESTRATION_TOOLS:
        return _dispatch_orchestration(call, workspace), None
    return f"Unknown tool {call.name!r}.", None


#: The decline message for the model-lifecycle tools in a workspace without them.
_NO_ML = (
    "model training/serving is not available in this workspace (the server needs "
    "the ml extra: pip install 'elbi-app[ml]')."
)

#: The notebook tool names, routed to :func:`_dispatch_notebook`.
_NOTEBOOK_TOOLS = frozenset(
    {
        "write_notebook",
        "run_notebook",
        "read_notebook",
        "list_notebooks",
        "promote_notebook_cell",
        "schedule_notebook",
    }
)

_NO_NOTEBOOK = "notebooks are not available in this workspace (it needs a store)."


def _dispatch_notebook(call: ToolCall, workspace: Workspace) -> str:
    """Route a notebook tool call to its injected capability, validating arguments."""
    args = call.arguments
    name = call.name
    if name == "list_notebooks":
        return _NO_NOTEBOOK if workspace.nb_list is None else workspace.nb_list()
    if name == "read_notebook":
        if workspace.nb_read is None:
            return _NO_NOTEBOOK
        notebook_id = str(args.get("notebook_id", "")).strip()
        return (
            workspace.nb_read(notebook_id)
            if notebook_id
            else "read_notebook needs `notebook_id`."
        )
    if name == "write_notebook":
        if workspace.nb_write is None:
            return _NO_NOTEBOOK
        cells = args.get("cells")
        if not isinstance(cells, list):
            return (
                "write_notebook needs `cells`: a list of {cell_type, source} objects "
                "(cell_type is 'code' or 'markdown')."
            )
        deps = args.get("deps")
        deps = [str(d) for d in deps] if isinstance(deps, list) else None
        target_id = str(args.get("notebook_id", "")).strip() or None
        return workspace.nb_write(
            target_id, str(args.get("name") or "Untitled notebook"), cells, deps
        )
    if name == "run_notebook":
        if workspace.nb_run is None:
            return _NO_NOTEBOOK
        notebook_id = str(args.get("notebook_id", "")).strip()
        if not notebook_id:
            return "run_notebook needs `notebook_id`."
        cells = args.get("cells")
        cell_ids = [str(c) for c in cells] if isinstance(cells, list) else None
        return workspace.nb_run(notebook_id, cell_ids)
    if name == "promote_notebook_cell":
        if workspace.nb_promote_cell is None:
            return _NO_NOTEBOOK
        notebook_id = str(args.get("notebook_id", "")).strip()
        cell_id = str(args.get("cell_id", "")).strip()
        if not notebook_id or not cell_id:
            return "promote_notebook_cell needs `notebook_id` and `cell_id`."
        return workspace.nb_promote_cell(notebook_id, cell_id)
    if name == "schedule_notebook":
        if workspace.nb_schedule is None:
            return _NO_NOTEBOOK
        notebook_id = str(args.get("notebook_id", "")).strip()
        if not notebook_id:
            return "schedule_notebook needs `notebook_id`."
        raw_hours = args.get("interval_hours")
        hours = float(raw_hours) if isinstance(raw_hours, (int, float)) else None
        dataset = str(args.get("dataset") or "").strip() or None
        return workspace.nb_schedule(
            notebook_id, str(args.get("mode") or "interval"), hours, dataset
        )
    return f"Unknown notebook tool {name!r}."


#: The dashboard tool names, routed to :func:`_dispatch_dashboard`.
_DASHBOARD_TOOLS = frozenset(
    {
        "list_dashboards",
        "dashboard_sources",
        "read_dashboard",
        "write_dashboard",
        "publish_dashboard",
    }
)

_NO_DASHBOARD = "dashboards are not available in this workspace (it needs a store)."


def _dispatch_dashboard(call: ToolCall, workspace: Workspace) -> str:
    """Route a dashboard tool call to its injected capability, validating arguments."""
    args = call.arguments
    name = call.name
    if name == "list_dashboards":
        return _NO_DASHBOARD if workspace.dash_list is None else workspace.dash_list()
    if name == "dashboard_sources":
        return (
            _NO_DASHBOARD
            if workspace.dash_sources is None
            else workspace.dash_sources()
        )
    if name == "read_dashboard":
        if workspace.dash_read is None:
            return _NO_DASHBOARD
        dashboard_id = str(args.get("dashboard_id", "")).strip()
        return (
            workspace.dash_read(dashboard_id)
            if dashboard_id
            else "read_dashboard needs `dashboard_id`."
        )
    if name == "write_dashboard":
        if workspace.dash_write is None:
            return _NO_DASHBOARD
        spec = args.get("spec")
        if not isinstance(spec, Mapping):
            return (
                "write_dashboard needs `spec`: the dashboard manifest object "
                "(specVersion, kind 'Dashboard', name, pages). Call dashboard_sources "
                "first to see which certified derivations a widget can bind."
            )
        target_id = str(args.get("dashboard_id", "")).strip() or None
        return workspace.dash_write(target_id, spec)
    if name == "publish_dashboard":
        if workspace.dash_publish is None:
            return _NO_DASHBOARD
        dashboard_id = str(args.get("dashboard_id", "")).strip()
        return (
            workspace.dash_publish(dashboard_id)
            if dashboard_id
            else "publish_dashboard needs `dashboard_id`."
        )
    return f"Unknown dashboard tool {name!r}."


#: The feature-store tool names, routed to :func:`_dispatch_features`.
_FEATURE_TOOLS = frozenset(
    {
        "list_feature_views",
        "define_feature_view",
        "materialize_features",
        "get_online_features",
        "get_historical_features",
        "profile_feature_view",
        "check_feature_drift",
        "check_feature_expectations",
        "create_training_set",
    }
)

_NO_FEATURES = "the feature store is not available in this workspace (needs a store)."


def _dispatch_features(call: ToolCall, workspace: Workspace) -> str:
    """Route a feature-store tool call to its injected capability."""
    args = call.arguments
    name = call.name
    if name == "list_feature_views":
        return _NO_FEATURES if workspace.fs_list is None else workspace.fs_list()
    if name == "define_feature_view":
        if workspace.fs_define is None:
            return _NO_FEATURES
        view = str(args.get("name", "")).strip()
        entities = args.get("entities")
        source = str(args.get("source", "")).strip()
        if not view or not isinstance(entities, list) or not source:
            return "define_feature_view needs `name`, `entities` (list), and `source`."
        features = args.get("features")
        ttl = args.get("ttl_seconds")
        return workspace.fs_define(
            view,
            [str(e) for e in entities],
            source,
            [str(f) for f in features] if isinstance(features, list) else None,
            str(args["timestamp_field"]) if args.get("timestamp_field") else None,
            int(ttl) if isinstance(ttl, (int, float)) else None,
        )
    if name == "materialize_features":
        if workspace.fs_materialize is None:
            return _NO_FEATURES
        views = args.get("feature_views")
        return workspace.fs_materialize(
            [str(v) for v in views] if isinstance(views, list) else None
        )
    if name in ("get_online_features", "get_historical_features"):
        capability = (
            workspace.fs_online
            if name == "get_online_features"
            else workspace.fs_historical
        )
        if capability is None:
            return _NO_FEATURES
        features = args.get("features")
        rows_key = "entity_rows" if name == "get_online_features" else "entity_df"
        rows = args.get(rows_key)
        if not isinstance(features, list) or not isinstance(rows, list):
            return f"{name} needs `features` (list) and `{rows_key}` (list of rows)."
        return capability([str(f) for f in features], rows)
    if name == "profile_feature_view":
        if workspace.fs_statistics is None:
            return _NO_FEATURES
        view = str(args.get("feature_view", "")).strip()
        if not view:
            return "profile_feature_view needs `feature_view`."
        return workspace.fs_statistics(view, bool(args.get("set_baseline", False)))
    if name == "check_feature_drift":
        if workspace.fs_drift is None:
            return _NO_FEATURES
        view = str(args.get("feature_view", "")).strip()
        if not view:
            return "check_feature_drift needs `feature_view`."
        return workspace.fs_drift(view)
    if name == "check_feature_expectations":
        if workspace.fs_expectations is None:
            return _NO_FEATURES
        view = str(args.get("feature_view", "")).strip()
        if not view:
            return "check_feature_expectations needs `feature_view`."
        return workspace.fs_expectations(view)
    if name == "create_training_set":
        if workspace.fs_training_set is None:
            return _NO_FEATURES
        ts_name = str(args.get("name", "")).strip()
        features = args.get("features")
        entity_df = args.get("entity_df")
        if (
            not ts_name
            or not isinstance(features, list)
            or not isinstance(entity_df, list)
        ):
            return (
                "create_training_set needs `name`, `features` (list), and "
                "`entity_df` (list of rows)."
            )
        label = str(args["label"]) if args.get("label") else None
        return workspace.fs_training_set(
            ts_name, [str(f) for f in features], entity_df, label
        )
    return f"Unknown feature tool {name!r}."


#: The lineage/catalog tool names, routed to :func:`_dispatch_lineage`.
_LINEAGE_TOOLS = frozenset({"search_catalog", "trace_lineage", "impact_analysis"})

_NO_LINEAGE = "the catalog is not available in this workspace (it needs a project)."


def _dispatch_lineage(call: ToolCall, workspace: Workspace) -> str:
    """Route a lineage/catalog tool call to its injected capability."""
    args = call.arguments
    name = call.name
    if name == "search_catalog":
        if workspace.lin_catalog is None:
            return _NO_LINEAGE
        return workspace.lin_catalog(str(args.get("query", "")))
    node = str(args.get("node", "")).strip()
    if name == "trace_lineage":
        if workspace.lin_lineage is None:
            return _NO_LINEAGE
        return workspace.lin_lineage(node) if node else "trace_lineage needs `node`."
    if name == "impact_analysis":
        if workspace.lin_impact is None:
            return _NO_LINEAGE
        return workspace.lin_impact(node) if node else "impact_analysis needs `node`."
    return f"Unknown lineage tool {name!r}."


#: The metrics tool names, routed to :func:`_dispatch_metrics`.
_METRIC_TOOLS = frozenset({"list_metrics", "define_metric", "query_metric"})

_NO_METRICS = "metrics are not available in this workspace (they need a project)."


def _dispatch_metrics(call: ToolCall, workspace: Workspace) -> str:
    """Route a semantic-layer metric tool call to its injected capability."""
    args = call.arguments
    name = call.name
    if name == "list_metrics":
        if workspace.metric_list is None:
            return _NO_METRICS
        return workspace.metric_list()
    if name == "define_metric":
        if workspace.metric_define is None:
            return _NO_METRICS
        metric = args.get("metric")
        if not isinstance(metric, dict):
            return "define_metric needs `metric`: a metric-spec object."
        return workspace.metric_define(metric)
    if name == "query_metric":
        if workspace.metric_query is None:
            return _NO_METRICS
        metric_name = str(args.get("name", "")).strip()
        if not metric_name:
            return "query_metric needs `name`."
        group_by = [str(g) for g in args.get("group_by", []) if str(g).strip()]
        grain = args.get("grain")
        return workspace.metric_query(metric_name, group_by, grain)
    return f"Unknown metrics tool {name!r}."


#: The monitoring tool names, routed to :func:`_dispatch_monitors`.
_MONITOR_TOOLS = frozenset({"list_monitors", "create_monitor"})

_NO_MONITORS = "monitors are not available in this workspace (they need a project)."


def _dispatch_monitors(call: ToolCall, workspace: Workspace) -> str:
    """Route a monitoring tool call to its injected capability."""
    if call.name == "list_monitors":
        if workspace.monitor_list is None:
            return _NO_MONITORS
        return workspace.monitor_list()
    if call.name == "create_monitor":
        if workspace.monitor_create is None:
            return _NO_MONITORS
        spec = call.arguments.get("monitor")
        if not isinstance(spec, dict):
            return "create_monitor needs `monitor`: a spec object."
        return workspace.monitor_create(spec)
    return f"Unknown monitoring tool {call.name!r}."


#: The orchestration tool names, routed to :func:`_dispatch_orchestration`.
_ORCHESTRATION_TOOLS = frozenset({"asset_status", "materialize_assets"})

_NO_ORCHESTRATION = "orchestration is not available in this workspace (no project)."


def _dispatch_orchestration(call: ToolCall, workspace: Workspace) -> str:
    """Route an orchestration tool call to its injected capability."""
    args = call.arguments
    if call.name == "asset_status":
        if workspace.orch_status is None:
            return _NO_ORCHESTRATION
        return workspace.orch_status()
    if call.name == "materialize_assets":
        if workspace.orch_materialize is None:
            return _NO_ORCHESTRATION
        return workspace.orch_materialize(
            str(args.get("selection", "stale")), str(args.get("assets", ""))
        )
    return f"Unknown orchestration tool {call.name!r}."


def _run_train(args: dict[str, Any], workspace: Workspace) -> str:
    """Validate a ``train_model`` call and hand it to the injected trainer."""
    if workspace.train is None:
        return _NO_ML
    name = str(args.get("name", "")).strip()
    dataset = str(args.get("dataset", "")).strip()
    derivation = str(args.get("derivation", "")).strip()
    if dataset and derivation:
        return "train_model takes either `dataset` or `derivation`, not both."
    source = derivation or dataset
    source_kind = "derivation" if derivation else "dataset"
    target = str(args.get("target", "")).strip()
    if not name or not source or not target:
        return (
            "train_model needs `name`, a data source (`dataset` or `derivation`), "
            "and `target`."
        )
    raw_features = args.get("features")
    features = (
        tuple(str(f).strip() for f in raw_features if str(f).strip())
        if isinstance(raw_features, list)
        else ()
    )
    try:
        time_budget = float(args.get("time_budget") or 60.0)
    except (TypeError, ValueError):
        return "train_model's `time_budget` must be a number of seconds."
    task = str(args.get("task") or "auto")
    horizon: int | None = None
    if args.get("horizon") is not None:
        try:
            horizon = int(args["horizon"])
        except (TypeError, ValueError):
            return "train_model's `horizon` must be an integer of periods."
    if task == "ts_forecast" and (not args.get("time_col") or horizon is None):
        return "task='ts_forecast' needs `time_col` and `horizon`."
    return workspace.train(
        {
            "name": name,
            "dataset": source,
            "source_kind": source_kind,
            "target": target,
            "features": features,
            "task": task,
            "time_budget": time_budget,
            "metric": str(args["metric"]) if args.get("metric") else None,
            "ensemble": bool(args.get("ensemble", False)),
            "time_col": str(args["time_col"]) if args.get("time_col") else None,
            "horizon": horizon,
            "engine": str(args.get("engine") or "flaml"),
        }
    )


def _run_derive(
    args: dict[str, Any], workspace: Workspace, step_index: int, state: _RunState
) -> tuple[str, AnswerResult | None]:
    """Author the model's analysis as a derivation, then hand back its verified output.

    The model writes the compute as ``source`` and declares the conclusion's ``claim``
    (output columns mapped to verification roles). Authoring runs it once in the sandbox
    and gates the output on the oracle; a sound conclusion is cached and persisted as a
    durable derivation, and its attestation recorded so the following ``answer``, the
    model's interpretation, is returned verified. An unsound or failed derivation
    returns the issue to fix. This is the only path to a verified answer: answers come
    from derivations, not from raw-data probes.
    """
    if workspace.derive is None:
        return "deriving is not available in this workspace.", None
    name = str(args.get("name", "")).strip()
    source = str(args.get("source", ""))
    if not name or not source:
        return "derive needs a `name` and Python `source` defining that function.", None
    claim_arg = args.get("claim")
    claim = claim_arg if isinstance(claim_arg, dict) and claim_arg else None
    contract_arg = args.get("contract")
    contract = contract_arg if isinstance(contract_arg, dict) and contract_arg else None
    fmt = str(args.get("format") or "table")
    raw = args.get("assumptions")
    assumptions = (
        tuple(str(a).strip() for a in raw if str(a).strip())
        if isinstance(raw, list)
        else ()
    )
    raw_deps = args.get("deps")
    deps = (
        tuple(str(d).strip() for d in raw_deps if str(d).strip())
        if isinstance(raw_deps, list)
        else ()
    )

    if args.get("background") and workspace.submit is not None:
        # A long fit: launch it as a durable job and continue. The job authors, gates,
        # caches, and persists exactly as an inline derive would; the run does not block
        # on it. Not terminal: the model should tell the user it is training and that
        # the certified result will follow, then end the turn.
        job_id = workspace.submit(name, source, claim, contract, fmt, assumptions, deps)
        return (
            f"Launched {name!r} as background job {job_id}. It will train, verify, and "
            "certify on its own; the run does not wait. Tell the user it is training "
            "in the background (give the job id) and that the certified result will "
            "follow when it finishes, then end your turn.",
            None,
        )

    outcome = workspace.derive(name, source, claim, contract, fmt, assumptions, deps)
    if outcome.error:
        return (
            f"Authored {name!r} but it failed in the sandbox:\n{outcome.error}\n\n"
            "Fix the source and derive again." + _missing_dep_hint(outcome.error),
            None,
        )
    if not outcome.certified:
        return (_derive_not_certified(outcome), None)
    if claim is None:
        # A cleaning/validation derivation: certified against its data contract, with no
        # conclusion to interpret. Report the certified clean table directly.
        return (
            f"Authored and certified {name!r}: its output meets the data contract and "
            f"is cached and saved as a durable derivation. Verified output:\n"
            f"{outcome.rendered}\n\nTell the user the data is clean and certified "
            "(cite the contract), then end your turn.",
            None,
        )
    # Record this certified conclusion; the model supplies the narrative next. Keyed by
    # name so a re-derive replaces (not duplicates) its finding, and accumulated so a
    # multi-variable answer keeps every effect rather than collapsing to the last one.
    # Declared assumptions ride along as premises the conclusion is conditioned on.
    state.findings[name] = Finding(
        name=name,
        claim=dict(claim or {}),
        verdict="sound",
        estimate=outcome.estimate,
        estimate_label=outcome.estimate_label,
        adjusted_for=outcome.adjusted_for,
        checks=outcome.checks,
        data_hash=outcome.data_hash,
        rows=outcome.rows,
        assumptions=assumptions,
    )
    state.verified = _aggregate(state.findings)
    certified_line = ""
    if outcome.estimate_label:
        held = (
            f" (a direct effect holding {', '.join(outcome.adjusted_for)} fixed)"
            if outcome.adjusted_for
            else " (unadjusted)"
        )
        certified_line = (
            f"\n\nCertified estimate: {outcome.estimate_label}{held}. Report THIS "
            "magnitude; do not quote a number from run_code as if it were certified."
        )
    return (
        f"Authored and certified {name!r}: cached and saved as a durable derivation. "
        f"Its verified output:\n{outcome.rendered}{certified_line}\n\nNow interpret "
        "this and call `answer` with the finding for the user, in plain language.",
        None,
    )


def _derive_not_certified(outcome: DeriveOutcome) -> str:
    """The message for a derivation the gate held, naming the gate that failed."""
    if outcome.contract_verdict not in (None, "sound"):
        return (
            f"{outcome.rendered}\n\nNot certified "
            f"(contract: {outcome.contract_verdict}): {outcome.contract_detail}\n"
            "Fix the cleaning and derive again."
        )
    return (
        f"{outcome.rendered}\n\nNot certified (oracle: {outcome.verdict}): "
        f"{outcome.detail}\nFix the analysis and derive again, or "
        "conclude_inconclusive if the data cannot support it."
    )


def _aggregate(findings: dict[str, Finding], narrative: str = "") -> AnswerResult:
    """Build the run's verified result from every certified finding so far.

    The first certified finding is the primary one whose rows/claim/checks a serving
    layer draws and renders (a single chart is a view of one certified surface); every
    finding rides in ``findings`` so the certified block can list them all. Assumptions
    are the union across findings, each an audited premise on the whole.
    """
    items = list(findings.values())
    primary = items[0]
    assumptions = tuple(dict.fromkeys(a for f in items for a in f.assumptions))
    return AnswerResult(
        verified=True,
        verdict="sound",
        narrative=narrative,
        checks=primary.checks,
        rows=primary.rows,
        assumptions=assumptions,
        spec={
            "derivation": primary.name,
            "claim": primary.claim,
            "derivations": [f.name for f in items],
        },
        data_hash=primary.data_hash,
        findings=tuple(items),
    )


def _finding_subject(claim: dict[str, Any], name: str) -> str:
    """A short label for what a finding is about, from its claim roles."""
    if claim.get("x") and claim.get("y"):
        return f"{claim['x']} → {claim['y']}"
    if claim.get("variant") and claim.get("metric"):
        return f"{claim['metric']} by {claim['variant']}"
    if claim.get("group") and claim.get("value"):
        return f"{claim['value']} by {claim['group']}"
    if claim.get("target") and claim.get("features"):
        return f"predict {claim['target']}"
    if claim.get("time") and claim.get("value"):
        return f"{claim['value']} over {claim['time']}"
    return name


def _render_certified_block(findings: Sequence[Finding]) -> str:
    """The certified-findings table the system renders into a verified answer.

    The magnitudes here are the oracle's, computed and checked server-side, so they
    cannot be a figure the model wrote in exploratory code. Each is labelled with what
    it holds fixed (a direct effect given those controls, not an unconditional total
    effect), so several coefficients are not read as interchangeable causal effects.
    """
    lines = [
        "---",
        "**Certified findings**: the magnitudes below are computed and checked by "
        "the verification oracle, not written by the model:",
        "",
        "| finding | certified estimate | holding fixed | verdict |",
        "| --- | --- | --- | --- |",
    ]
    for f in findings:
        label = (f.estimate_label or "(see checks)").replace("|", "\\|")
        held = ", ".join(f.adjusted_for) if f.adjusted_for else "nothing (unadjusted)"
        subject = _finding_subject(f.claim, f.name).replace("|", "\\|")
        lines.append(f"| {subject} | {label} | {held} | {f.verdict} |")
    lines.append("")
    lines.append(
        "_Each adjusted estimate is a direct effect given the columns it holds fixed, "
        "not necessarily the total effect. Any figure stated in the prose above that "
        "is not in this table is exploratory, not certified._"
    )
    return "\n".join(lines)


def _array_schema(item_desc: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": item_desc}


#: The tools offered to the model. `derive` is the only path to a verified answer (it
#: authors and certifies a durable derivation), which is what makes verification
#: unavoidable and every reported conclusion a cached, reusable artifact.
_TOOLSPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "describe_dataset",
        "Show a dataset's row count, columns, and an example row.",
        {
            "type": "object",
            "properties": {"dataset": {"type": "string"}},
            "required": ["dataset"],
        },
    ),
    ToolSpec(
        "profile_dataset",
        "Profile a dataset for cleaning: per column, its completeness, distinct count, "
        "inferred type, numeric range, and most common values. Read this before "
        "declaring a cleaning derivation's `contract` so the constraints fit the data.",
        {
            "type": "object",
            "properties": {"dataset": {"type": "string"}},
            "required": ["dataset"],
        },
    ),
    ToolSpec(
        "structure_map",
        "Show how a dataset's columns relate: which share information with both a "
        "candidate X and Y (candidate confounders), and which are derived/redundant.",
        {
            "type": "object",
            "properties": {"dataset": {"type": "string"}},
            "required": ["dataset"],
        },
    ),
    ToolSpec(
        "run_code",
        "Run exploratory Python in a sandbox with the datasets available as a `data` "
        "dict (name -> list of row dicts). `deps` names any third-party packages the "
        "code imports; they are installed on demand (e.g. ['interpret'] for an EBM, "
        "['prophet']); only the standard library is available without deps. print() "
        "to inspect; set `result` to return. run_code calls share a live session: "
        "variables, imports, loaded data, and files persist from one call to the next, "
        "so build up state incrementally (load once, fit a model, keep using it) "
        "rather than repeating setup. For a long exploratory computation that would "
        "outrun the inline budget, set `background: true`: it runs as a job in its own "
        "isolated session, so make it self-contained and have it WRITE results to a "
        "file in the workspace, which a later inline call can then read.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "deps": {"type": "array", "items": {"type": "string"}},
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run as a background job (isolated session, shared workspace "
                        "files) for a long computation; omit to run inline in the live "
                        "session."
                    ),
                },
            },
            "required": ["code"],
        },
    ),
    ToolSpec(
        "bash",
        "Run a shell command in the same sandbox as run_code, sharing its filesystem "
        "and installed packages. Use it for shell-shaped work: install a system lib "
        "(apt-get), inspect files, or run a CLI tool. A package you install here (pip "
        "or apt) is available to the next run_code. Exploration only; the answer still "
        "comes from derive.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
    ToolSpec(
        "derive",
        "Author your analysis as a derivation and get its verified output: this is "
        "how you produce the answer (run_code is only scratch). `source` is Python "
        "defining `def <name>(ctx): ...` that reads inputs via ctx.input('dataset') "
        "and returns the analytic ROWS your conclusion is about (a list of row dicts). "
        "It runs once in the sandbox and its output is gated on the oracle; a sound "
        "derivation is cached and saved as a durable answer. The derivation sandbox "
        "starts with ONLY the standard library, so name every third-party package the "
        "source imports in `deps` (numpy/pandas/scikit-learn etc.), exactly as you do "
        "for run_code, or it fails to import. Declare the conclusion in "
        "`claim` as output columns mapped to roles, so the oracle can check it; an "
        "unsound result returns the issue to fix, then derive again. Roles: x, y (an "
        "effect/association, plus `controls`); group, value (a group difference); "
        "variant, metric (an A/B test: add `time` for a fading novelty effect); "
        "group, outcome (a categorical association); group, outcomes (one group vs "
        "MANY metrics: the oracle corrects for multiple comparisons, so include the "
        "whole family); before, after, group (a pre/post change: checks regression to "
        "the mean); time, value (a trend); time, actual, forecast (a forecast); "
        "y_true, y_pred, y_score (a classifier); target, features (predictive "
        "accuracy: return the FEATURE rows and the target and do NOT fit a model in "
        "the derivation; the oracle fits and scores held-out, certifying that "
        "leakage-free skill exists as a conservative lower bound, not your model's "
        "score); "
        "probability, outcome (calibration); time, event, group (survival); "
        "group, y_true, y_pred (fairness); features (cluster structure); count (a "
        "count model); target (data leakage); latitude, longitude, value (a geographic "
        "pattern: spatial clustering via Moran's I and Getis-Ord hot spots, the "
        "unbiased basis for a location claim). To fit a nonlinear form, keep OUTPUT "
        "columns in their natural, readable units and declare the reshape in "
        "`transforms` (e.g. {'<x>':'log','<y>':'log'}): the oracle checks the log-log "
        "elasticity while the rows stay human-readable for the chart. Do not output "
        "pre-transformed columns like `log_x`. "
        "For a geographic question (location, lat/long, 'where'), aggregate to grid "
        "cells, return each cell's `latitude`, `longitude` and value, and claim those "
        "three, not a regression on raw coordinates (spatial dependence biases it). "
        "Then attach a `map`. To CLEAN or validate data rather than draw a conclusion, "
        "return the cleaned rows and declare a `contract` (instead of a claim) the "
        "output must satisfy; it is certified only if the data meets the contract.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "snake_case derivation and function name",
                },
                "source": {
                    "type": "string",
                    "description": "Python: def <name>(ctx) -> list of row dicts",
                },
                "format": {
                    "type": "string",
                    "description": "serve format: table (default), markdown, json",
                },
                "claim": {
                    "type": "object",
                    "description": "output columns mapped to verification roles",
                    "properties": {
                        "x": {"type": "string"},
                        "y": {"type": "string"},
                        "controls": _array_schema("columns to hold fixed"),
                        "group": {"type": "string"},
                        "value": {"type": "string"},
                        "variant": {"type": "string"},
                        "metric": {"type": "string"},
                        "outcome": {"type": "string"},
                        "outcomes": _array_schema("outcome columns as a family"),
                        "before": {"type": "string"},
                        "after": {"type": "string"},
                        "time": {"type": "string"},
                        "actual": {"type": "string"},
                        "forecast": {"type": "string"},
                        "y_true": {"type": "string"},
                        "y_pred": {"type": "string"},
                        "y_score": {"type": "string"},
                        "target": {"type": "string"},
                        "features": _array_schema("feature columns"),
                        "probability": {"type": "string"},
                        "event": {"type": "string"},
                        "count": {"type": "string"},
                        "latitude": {"type": "string"},
                        "longitude": {"type": "string"},
                        "transforms": {
                            "type": "object",
                            "description": (
                                "column -> 'log'|'log1p'|'sqrt'|'square'|'reciprocal', "
                                "applied before verification. So OUTPUT columns in "
                                "their natural, readable units and declare the "
                                "here to verify a log-log or log-linear relationship: "
                                "the rows stay human-readable for the chart while the "
                                "oracle checks the right form."
                            ),
                        },
                    },
                },
                "contract": {
                    "type": "object",
                    "description": (
                        "For a CLEANING/validation step: a data contract the OUTPUT "
                        "must satisfy. Shape: {fields:[{name, type, constraints, "
                        "mostly}], table:{primaryKey, uniqueKeys, foreignKeys, "
                        "rowCount, columnsMatch}}. Field type is one of string|integer|"
                        "number|boolean|date|datetime; constraints are required, "
                        "unique, minimum, maximum, minLength, maxLength, pattern, enum "
                        "(numeric range only on numeric types, length/pattern only on "
                        "strings). `mostly` (0-1) tolerates a share of invalid values. "
                        "Certified ONLY if the output meets the contract; violations "
                        "return the rows to fix. Profile first (profile_dataset)."
                    ),
                },
                "assumptions": _array_schema(
                    "out-of-data premises the conclusion is conditioned on, each with "
                    "its source, e.g. 'treatment was randomized (spec §2)'. Declare "
                    "one only when the attached context states an identification "
                    "assumption (randomization, the sampling mechanism, an instrument, "
                    "a known confounder) that a causal reading needs: recorded as an "
                    "audited premise, not trusted by the oracle."
                ),
                "deps": _array_schema(
                    "third-party packages the source imports, provisioned into the "
                    "sandbox (like run_code's deps). The derivation sandbox starts "
                    "with only the standard library, so name every non-stdlib import "
                    "here or it fails: import numpy,pandas -> deps=['numpy','pandas']; "
                    "sklearn -> deps=['scikit-learn']. Use the install name when it "
                    "differs from the import."
                ),
                "background": {
                    "type": "boolean",
                    "description": (
                        "Set true for a long training run (a heavy fit that would "
                        "outrun the inline budget). It launches as a durable job and "
                        "certifies on its own while you continue; you tell the user it "
                        "is training and the certified result follows when done. "
                        "Omit for ordinary derivations, which run inline."
                    ),
                },
            },
            "required": ["name", "source"],
        },
    ),
    ToolSpec(
        "train_model",
        "Train a reusable predictive model with AutoML and register it in the model "
        "registry. Use this when the user wants a MODEL (to score new records, keep "
        "versions, or serve predictions), not just to know whether signal exists "
        "(that is `derive` with claim {target, features}). AutoML (FLAML) searches "
        "the standard tabular learners (LightGBM, XGBoost, random forest, linear) "
        "within `time_budget` seconds; the report's metrics are computed on a "
        "held-out split the search never saw, and the verification oracle "
        "independently checks that leakage-free signal exists. The trained model is "
        "logged to MLflow with its run and registered as a new version; the first "
        "sound version becomes the `champion` alias (later versions need "
        "promote_model). Report the held-out metrics from the report; never quote a "
        "training-time score as if it were held-out.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "snake_case registry name (reuse to add a version)",
                },
                "dataset": {"type": "string"},
                "derivation": {
                    "type": "string",
                    "description": (
                        "train on a certified derivation's output instead of a raw "
                        "dataset: the platform's feature-pipeline path. Engineer "
                        "features with `derive` first, then name that derivation "
                        "here."
                    ),
                },
                "target": {"type": "string", "description": "the column to predict"},
                "features": _array_schema(
                    "feature columns; omit to use every other column"
                ),
                "task": {
                    "type": "string",
                    "description": (
                        "classification | regression | ts_forecast | auto "
                        "(default). ts_forecast trains a forecaster and needs "
                        "time_col and horizon."
                    ),
                },
                "time_col": {
                    "type": "string",
                    "description": "timestamp column (ts_forecast only)",
                },
                "horizon": {
                    "type": "integer",
                    "description": "periods to forecast ahead (ts_forecast only)",
                },
                "time_budget": {
                    "type": "number",
                    "description": "AutoML search budget in seconds (default 60)",
                },
                "metric": {
                    "type": "string",
                    "description": (
                        "FLAML metric to optimize (accuracy, roc_auc, f1, log_loss, "
                        "r2, rmse, mae, ...); omit for the task's standard one"
                    ),
                },
                "ensemble": {
                    "type": "boolean",
                    "description": (
                        "train a stacked ensemble over the searched learners as "
                        "the final model (slower, often more accurate)"
                    ),
                },
                "engine": {
                    "type": "string",
                    "description": (
                        "flaml (default: fast, budget-aware) or autogluon "
                        "(accuracy-first stack ensembling; needs the autogluon "
                        "extra; use for long budgets where accuracy matters most)"
                    ),
                },
            },
            "required": ["name", "dataset", "target"],
        },
    ),
    ToolSpec(
        "list_models",
        "List the registered models: each with its versions, held-out metrics, and "
        "where the champion alias points.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "predict",
        "Score feature records with a registered model. `model` is a name (its "
        "champion, or newest version when none), `name@alias`, or `name/version`. "
        "`rows` are feature records shaped like the training columns (the target "
        "column omitted).",
        {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "rows": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "feature records to score",
                },
                "raw": {
                    "type": "boolean",
                    "description": (
                        "the rows are RAW source records; apply the model's "
                        "recorded feature derivation before scoring (only for "
                        "models trained on a derivation)"
                    ),
                },
            },
            "required": ["model", "rows"],
        },
    ),
    ToolSpec(
        "batch_score",
        "Score EVERY row of a dataset with a registered model (batch inference). "
        "Use it when the user wants predictions over the whole dataset rather "
        "than a few records. The full predictions are stored as a CSV artifact "
        "on a scoring run (downloadable from MLflow); you get back the summary "
        "stats and where the artifact lives. `model` is a name, name@alias, or "
        "name/version.",
        {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "dataset": {"type": "string"},
                "derivation": {
                    "type": "string",
                    "description": "score a certified derivation's output instead",
                },
            },
            "required": ["model"],
        },
    ),
    ToolSpec(
        "promote_model",
        "Point a registered model's alias at a version (default alias: champion). "
        "The champion is what a bare model name serves, so promote only when the "
        "user asks for it or accepts a new version's held-out metrics.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "integer"},
                "alias": {"type": "string"},
            },
            "required": ["name", "version"],
        },
    ),
    ToolSpec(
        "answer",
        "Give the user the final finding in your own words. When a sound `derive` "
        "precedes it, the answer is verified and carries that derivation's checks and "
        "data hash; interpret the verified estimate (sign, size, the units your "
        "transform implies, what is held fixed) and state the caveats. Otherwise use "
        "it directly only for a question with no soundness oracle (a description, "
        "ranking, count), resting on figures you computed, never from memory. When a "
        "picture helps a verified finding, attach a `chart`: a Vega-Lite spec (any "
        "mark, transform, layer, or facet: bar, line, scatter, histogram, boxplot, "
        "heatmap, whatever fits). Do NOT include data (we inject the certified rows) "
        "reference only the derivation's OUTPUT columns. Draw it so a person can read "
        "it: put the natural, readable columns on the axes (a value in its real units, "
        "not a pre-logged column), and for a skewed variable use a log SCALE on that "
        "axis (encoding `scale: {type: 'log'}`): same relationship, axis reads "
        "real units. For a geographic tile map, encode `latitude` and `longitude` "
        "(colour by a `value`); it renders on a real map. The chart draws certified "
        "rows.",
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "chart": _CHART_SCHEMA},
            "required": ["summary"],
        },
    ),
    ToolSpec(
        "conclude_inconclusive",
        "Conclude that the data cannot support a sound answer, with the reason.",
        {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    ),
    # -- notebooks: the same authoring surface a human uses --------------------------
    ToolSpec(
        "list_notebooks",
        "List the notebooks that exist (name, id, cell count). Use it to find "
        "a notebook before reading or running it.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "read_notebook",
        "Read a notebook's cells (with their ids and last outputs), its "
        "environment (packages), and its schedule. Read before editing so you "
        "edit the right cells.",
        {
            "type": "object",
            "properties": {"notebook_id": {"type": "string"}},
            "required": ["notebook_id"],
        },
    ),
    ToolSpec(
        "write_notebook",
        "Create a notebook (omit notebook_id) or replace an existing one's "
        "cells (pass notebook_id). `cells` is the full ordered list of "
        "{cell_type, source} objects (cell_type is 'code' or 'markdown'); "
        "datasets are available to code cells as a `data` dict (name -> list of "
        "row dicts). `deps` declares the kernel's third-party packages (resolved "
        "and pinned). Then call run_notebook to execute and see outputs. Use a "
        "notebook when the user wants a reusable, shareable, or scheduled "
        "analysis they can open in the UI; for a one-off verified answer, use "
        "derive.",
        {
            "type": "object",
            "properties": {
                "notebook_id": {
                    "type": "string",
                    "description": "Omit to create; pass to replace a notebook.",
                },
                "name": {"type": "string"},
                "cells": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cell_type": {"enum": ["code", "markdown"]},
                            "source": {"type": "string"},
                        },
                        "required": ["cell_type", "source"],
                    },
                },
                "deps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["cells"],
        },
    ),
    ToolSpec(
        "run_notebook",
        "Run a notebook and read its outputs back so you can self-correct: run "
        "every code cell (omit `cells`) or only the given cell ids. Returns each "
        "cell's stdout, result, and any traceback. Fix a failing cell with "
        "write_notebook, then re-run.",
        {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "cells": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cell ids to run; omit to run the whole notebook.",
                },
            },
            "required": ["notebook_id"],
        },
    ),
    ToolSpec(
        "promote_notebook_cell",
        "Promote a notebook cell that defines `def <name>(ctx): ...` to a "
        "certified, governed derivation: the same authoring loop as derive "
        "(sandboxed run, oracle verification, certification). Use it to turn a "
        "feature cell into a durable, cached derivation models can train on.",
        {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "cell_id": {"type": "string"},
            },
            "required": ["notebook_id", "cell_id"],
        },
    ),
    ToolSpec(
        "schedule_notebook",
        "Schedule a notebook to rerun automatically: on an `interval` (set "
        "`interval_hours`) or `on_data_change` of a `dataset` (set `dataset`). "
        "The user can see and change the schedule in the UI.",
        {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "mode": {"enum": ["interval", "on_data_change"]},
                "interval_hours": {"type": "number"},
                "dataset": {"type": "string"},
            },
            "required": ["notebook_id", "mode"],
        },
    ),
    ToolSpec(
        "list_dashboards",
        "List the dashboards that exist, with their id, status, and version.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "dashboard_sources",
        "List the certified derivations a dashboard widget can bind, with each one's "
        "parameters. Only certified derivations can back a widget, so check this "
        "before authoring a dashboard; author any you need with derive first.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "read_dashboard",
        "Read a dashboard's variables, pages, and widgets so you can edit it.",
        {
            "type": "object",
            "properties": {"dashboard_id": {"type": "string"}},
            "required": ["dashboard_id"],
        },
    ),
    ToolSpec(
        "write_dashboard",
        "Create a dashboard (omit dashboard_id) or replace an existing one's spec "
        "(pass dashboard_id). `spec` is the full Dashboard manifest: {specVersion: "
        "'1.0', kind: 'Dashboard', name, title, variables?, pages: [{name, widgets: "
        "[{id, type, gridPos: {x,y,w,h}, bind: {derivation, params}, viz?}]}]}. A "
        "widget of type metric/chart/map/table binds a certified derivation; a param "
        "value of '$var' resolves to a dashboard variable. Build a dashboard when the "
        "user wants a reusable, shareable view of several derivations; for a one-off "
        "answer, use derive. Then call publish_dashboard.",
        {
            "type": "object",
            "properties": {
                "dashboard_id": {
                    "type": "string",
                    "description": "Omit to create; pass to replace a dashboard.",
                },
                "spec": {
                    "type": "object",
                    "description": "The full Dashboard manifest.",
                },
            },
            "required": ["spec"],
        },
    ),
    ToolSpec(
        "publish_dashboard",
        "Publish a dashboard so viewers see it. The certification gate refuses to "
        "publish if any bound derivation is not certified, and names the offenders: "
        "author and certify them with derive, then retry.",
        {
            "type": "object",
            "properties": {"dashboard_id": {"type": "string"}},
            "required": ["dashboard_id"],
        },
    ),
    ToolSpec(
        "list_feature_views",
        "List the feature views in the feature store: each view's join keys, its "
        "source derivation, whether that source is certified, and its features.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "define_feature_view",
        "Define a feature view: a group of features keyed by entities and sourced "
        "from a certified derivation. `entities` are registered entity names; "
        "`source` is the derivation whose rows carry the join keys, the optional "
        "`timestamp_field` event time, and the feature columns; `features` restricts "
        "the exposed columns (omit for all). Set `ttl_seconds` to bound how far a "
        "point-in-time join looks back. Then materialize_features and retrieve.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}},
                "timestamp_field": {"type": "string"},
                "ttl_seconds": {"type": "integer"},
            },
            "required": ["name", "entities", "source"],
        },
    ),
    ToolSpec(
        "materialize_features",
        "Refresh the online store with the latest value per entity for each feature "
        "view (or the named `feature_views`). Required before get_online_features.",
        {
            "type": "object",
            "properties": {
                "feature_views": {"type": "array", "items": {"type": "string"}}
            },
        },
    ),
    ToolSpec(
        "get_online_features",
        "Read the latest materialized features for a batch of entity rows. "
        "`features` are 'view:feature' references; `entity_rows` each supply the join "
        "keys. Use at inference time.",
        {
            "type": "object",
            "properties": {
                "features": {"type": "array", "items": {"type": "string"}},
                "entity_rows": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["features", "entity_rows"],
        },
    ),
    ToolSpec(
        "get_historical_features",
        "Point-in-time join of features onto an entity dataframe, for building "
        "training data without leakage. `features` are 'view:feature' references; "
        "each `entity_df` row supplies the join keys and an 'event_timestamp'. Each "
        "value is the latest at or before that timestamp, never a future value.",
        {
            "type": "object",
            "properties": {
                "features": {"type": "array", "items": {"type": "string"}},
                "entity_df": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["features", "entity_df"],
        },
    ),
    ToolSpec(
        "profile_feature_view",
        "Profile a feature view's columns now (per-feature completeness, distinct "
        "count, and range) and record the snapshot. Set `set_baseline` true to also "
        "capture this as the reference distribution that check_feature_drift compares "
        "against later.",
        {
            "type": "object",
            "properties": {
                "feature_view": {"type": "string"},
                "set_baseline": {"type": "boolean"},
            },
            "required": ["feature_view"],
        },
    ),
    ToolSpec(
        "check_feature_drift",
        "Check whether a feature view's current values have drifted from its baseline "
        "distribution (set one with profile_feature_view first). Reports which "
        "features moved and whether the dataset as a whole drifted.",
        {
            "type": "object",
            "properties": {"feature_view": {"type": "string"}},
            "required": ["feature_view"],
        },
    ),
    ToolSpec(
        "check_feature_expectations",
        "Verify a feature view's current values against its attached data contract "
        "(a human sets the contract). Reports the three-valued verdict (sound / "
        "unsound / inconclusive) and which clauses failed.",
        {
            "type": "object",
            "properties": {"feature_view": {"type": "string"}},
            "required": ["feature_view"],
        },
    ),
    ToolSpec(
        "create_training_set",
        "Materialize a point-in-time join into a named, reusable training set. "
        "`features` are 'view:feature' references; each `entity_df` row supplies the "
        "join keys, an 'event_timestamp', and (typically) the `label` column to learn. "
        "Each feature is the latest value at or before that timestamp (no leakage). It "
        "persists as a training source train_model can use (kind training_set).",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}},
                "entity_df": {"type": "array", "items": {"type": "object"}},
                "label": {"type": "string"},
            },
            "required": ["name", "features", "entity_df"],
        },
    ),
    ToolSpec(
        "search_catalog",
        "Search the catalog of every artifact (datasets, derivations, models, "
        "dashboards, feature views) by name, type, or description. Use it to find "
        "what already exists before authoring something new.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    ),
    ToolSpec(
        "trace_lineage",
        "Trace an artifact's lineage: what feeds it (provenance) and what depends on "
        "it. `node` is a 'type:name' reference (e.g. 'derivation:revenue') or a bare "
        "name. Use it to answer where a number comes from.",
        {
            "type": "object",
            "properties": {"node": {"type": "string"}},
            "required": ["node"],
        },
    ),
    ToolSpec(
        "impact_analysis",
        "Show what changing or deleting an artifact would affect downstream, which "
        "derivations, models, dashboards, and feature views depend on it. `node` is a "
        "'type:name' reference or a bare name. Use it before changing a dataset.",
        {
            "type": "object",
            "properties": {"node": {"type": "string"}},
            "required": ["node"],
        },
    ),
    ToolSpec(
        "list_metrics",
        "List the defined semantic-layer metrics with their type and source. A metric "
        "is a named aggregation over a certified derivation, sliced by dimensions.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "define_metric",
        "Define (or replace) a metric. `metric` is a metric-spec object: a 'simple' "
        "metric has {name, type:'simple', source (a certified derivation), "
        "measure:{agg, column}, dimensions:[...], timeDimension:{column, grain}}; a "
        "'ratio' has {numerator, denominator}; 'derived' has {expr, inputMetrics}; "
        "'cumulative' has {inputMetric, window}. Aggregations: sum, average, count, "
        "count_distinct, min, max, median. A simple metric's source must be certified.",
        {
            "type": "object",
            "properties": {"metric": {"type": "object"}},
            "required": ["metric"],
        },
    ),
    ToolSpec(
        "query_metric",
        "Resolve a metric to verified numbers, grouped by dimensions and rolled up to "
        "a grain. `group_by` are dimensions the metric declares; `grain` (day/week/"
        "month/quarter/year) applies to its time dimension.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "group_by": {"type": "array", "items": {"type": "string"}},
                "grain": {"type": "string"},
            },
            "required": ["name"],
        },
    ),
    ToolSpec(
        "list_monitors",
        "List the anomaly monitors: what metric or derivation each watches, its latest "
        "value, and whether it is currently alerting.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "create_monitor",
        "Set up an anomaly monitor over a certified target. `monitor` is a spec: "
        "{name, target_kind: 'metric'|'derivation', target, method: 'mad'|'zscore', "
        "sensitivity (spreads allowed, default 3), min_value/max_value (optional hard "
        "bounds), window (recent points for the baseline), interval_hours}. It watches "
        "only certified targets and alerts when a value leaves its learned baseline.",
        {
            "type": "object",
            "properties": {"monitor": {"type": "object"}},
            "required": ["monitor"],
        },
    ),
    ToolSpec(
        "asset_status",
        "Show which derivation assets are stale (their inputs or code changed since "
        "they were last materialized), materialized, or never materialized.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "materialize_assets",
        "Recompute derivations in dependency order, skipping ones already fresh. "
        "`selection` is 'stale' (default) or 'all'; or pass `assets` (comma-separated "
        "names) to materialize those and their downstream. Idempotent and safe to run.",
        {
            "type": "object",
            "properties": {
                "selection": {"enum": ["stale", "all"]},
                "assets": {"type": "string"},
            },
        },
    ),
)
