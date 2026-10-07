# Agent-authored derivations

A derivation does not have to be written by a human. An agent can **propose** one
as generated Python, the framework **verifies** it runs correctly under isolation,
and a **certification policy** decides whether it is served. This is the same
skill-library loop agents use elsewhere: generate executable code, self-verify it,
keep what works.

By default, **self-verification is the gate**: a proposal that runs cleanly in the
sandbox is certified and served automatically, with no human step. A deployment
that needs human sign-off swaps in a stricter policy; see
[Certification policy](#certification-policy) below.

## Provenance and lifecycle

Every derivation carries two fields (Open Derivation Spec 1.1), defaulting to the
values a hand-written derivation already has:

| Field | Values | Default | Meaning |
| --- | --- | --- | --- |
| `origin` | `human`, `agent` | `human` | Who authored it. |
| `status` | `proposed`, `certified` | `certified` | Whether it may be served. |

The rule that makes this safe: **a `proposed` derivation is never served to an
agent over MCP.** It is discovered, listed, and runnable for review, but it is not
exposed until it is certified.

## The loop

`author()` runs the whole loop (propose, verify, and certify per the policy) in
one call:

```python
from elbi_core import Runner, SubprocessExecutor, author, serve

source = """
def revenue_by_region(ctx):
    "Total revenue grouped by region."
    rows = ctx.input("sales").rows
    by = {}
    for r in rows:
        by[r["region"]] = by.get(r["region"], 0) + float(r["amount"])
    return [{"region": k, "revenue": v} for k, v in by.items()]
"""

runner = Runner(registry, executor=SubprocessExecutor())
outcome = author(
    "revenue_by_region",
    source,
    runner=runner,
    serve=serve.table(title="Revenue"),
    predicted=[{"region": "west", "revenue": 10.0}],  # optional self-check
)
assert outcome.result.ok
assert outcome.certified  # default policy certified it on a clean verify
```

The three steps are also available individually (`propose()`, `verify()`,
`certify()`) when you want to drive them yourself. The proposed source MUST define
a top-level function named the same as the derivation; that function is the
compute. `verify` captures execution failures rather than raising, so a bad
proposal yields a result you can inspect, and a failed verification is never
certified.

## Verifying on evidence, not on "it ran"

A single clean run is a weak bar to certify generated code on. Pass `cases`, a
sequence of `GoldenCase` expected input→output examples (ideally including edge
cases like empty input, boundaries, and duplicates), and `verify()` runs each,
comparing the output to the expected value by content fingerprint (so any correct
implementation passes, however it is written):

```python
from elbi_core import GoldenCase, RequireChecks, author

outcome = author(
    "revenue_by_region",
    source,
    runner=runner,
    serve=serve.table(),
    cases=[
        GoldenCase(expected=[{"region": "west", "revenue": 10.0}], unordered=True),
        GoldenCase(expected=[], params={"region": "nowhere"}),  # edge case
    ],
    policy=RequireChecks(),  # certify only when a case actually passed
)
```

Set `unordered=True` for outputs whose row order is not meaningful (a set of rows);
leave it off for ranked outputs, where order is part of being correct. The
`RequireChecks` policy auto-certifies only a proposal that passed at least one
case: autonomy, but on execution-grounded evidence rather than a bare run.

!!! warning "Who authors the cases is the gate"
    When `cases` decide certification, author them from a source independent of the
    proposing agent: a human, the dataset's ground truth, or a separate reviewer.
    An agent that supplies both the derivation and its own checks can satisfy the
    gate with cases that only restate what its code happens to do, which certifies
    nothing. This is the same reason a coding agent must not write the tests that
    grade it. Agent-supplied cases are useful as a self-check during `propose`, but
    a certification gate should rest on cases the agent did not write.

## Certification policy

Whether a verified proposal is served is decided by a pluggable
`CertificationPolicy`, the same kind of seam as the cache store or executor:

- **`AutoCertifyOnVerify`**: the default. Certifies anything that passes
  verification. Self-verification is the gate; the agent stays autonomous, and the
  result is still auditable (provenance is kept) and reversible.
- **`ManualCertification`**: holds every proposal at `status=proposed` for an
  explicit human `certify()` (or `elbi certify <name>`). The governance
  gate, for a regulated setting where a served answer must be human-endorsed.

```python
from elbi_core import ManualCertification, author

outcome = author(..., policy=ManualCertification())  # held for review
assert not outcome.certified
```

The choice is a *deployment* decision, not a property of the derivation: the open
local default favors autonomy; the platform can require human endorsement where
the cost of a wrong autonomous answer outweighs the cost of delay.

## Execution isolation

Agent-generated source is untrusted, so it never runs in the host process. The
`Executor` is a pluggable seam, like the cache store:

- **`InProcessExecutor`**: the default. Trusted, in-process; used for hand-written
  derivations. An agent-authored derivation's compute refuses to run here.
- **`SubprocessExecutor`**: runs the source in a fresh, isolated Python subprocess
  with a **scrubbed environment** (host secrets withheld), an **ephemeral working
  directory**, **resource limits** (CPU, file size, optional memory), **denied
  network egress** by default (pass `allow_network=True` to permit it), and a
  wall-clock timeout. A bounded local default.
- **`RoutingExecutor`**: dispatches per derivation, sending anything that carries
  generated source to the sandbox and running trusted compute functions
  in-process. This is what `elbi mcp` and `serve` use, so certified
  agent-authored derivations are served correctly while human ones stay fast.

!!! warning "The subprocess sandbox is a boundary, not a fortress"
    `SubprocessExecutor` hardens a local default: environment scrub, resource
    limits, and a best-effort egress block (it disables `socket` in the child).
    It is **not** a security boundary against hostile code: a shared kernel still
    is one, and the egress block is defense in depth, not a firewall. Kernel-level
    isolation and enforced egress are a microVM executor's job.

Because `Executor` is a plain protocol (`run(derivation, context) -> value`), you
can delegate hostile-code execution to a hosted sandbox without changing any
derivation. An adapter to E2B, Modal, or Vercel Sandbox is a small class that ships
the source to the service and returns the value; a hosted sandbox provides a
microVM-class executor through the same seam.

## Guiding the agent

`elbi mcp` (and `serve`) set the MCP server's `instructions` to a default,
question-agnostic analysis rubric, so the authoring agent answers with the
discipline of a careful data scientist by default: identify the kind of analysis
and its standard pitfalls, check the data can answer the question, control for
confounders rather than report a raw two-variable relationship, quantify
uncertainty, separate association from causation, state caveats, and ground every
figure in an authored derivation's real output. It encodes discipline, not a
recipe per question type, so it scales to any question; the model supplies the
method.

A project can append its own domain context with an optional `ai_context` string
in `elbi.yaml`:

```yaml
project: acme
ai_context: |
  Revenue is in USD and excludes tax. Treat `region` as the location dimension.
```

The default guidance still applies; the project text is added after it.

## Over MCP

`elbi mcp` (and `serve`) expose cross-cutting tools in addition to the
per-derivation run tools:

- **`search_derivations`**: the retrieval entry point. An agent searches and
  selects a few derivations instead of choosing among every tool, which keeps
  selection accurate as a project's library grows to hundreds. Only served,
  certified derivations are ranked, so a proposed one can never crowd out a real
  match. See [Search](#search) for how hybrid lexical-plus-semantic ranking works.
- **`describe_dataset`**: read a bound dataset's schema (row count, columns,
  example row, inferred types) *and any author-declared meaning* before writing a
  derivation. Call with no name to list the datasets. This is how an agent learns
  the data *without* authoring throwaway probe derivations. See
  [Dataset semantic model](#dataset-semantic-model) for the declared meaning.
- **`sandbox_environment`**: report the sandbox's Python version and which compute
  libraries (numpy, pandas, …) are importable, so the agent knows what it can
  `import` before it writes code.
- **`structure_map`**: discover a dataset's relational skeleton before analyzing
  it, computed deterministically from the values: which columns are identifiers,
  which are derived or perfectly collinear, which functionally determine others,
  and which **share information** (the candidates for a confounder when one drives
  another). For a "what affects X" question this is how an agent picks the controls
  and avoids collinear regressors, rather than guessing how columns relate. It
  describes *structure*, never meaning; pair it with `describe_dataset` (meaning)
  and `run_code` (verify). See [Structure discovery](#structure-discovery).
- **`verify_analysis`**: the soundness oracle for a claimed effect of `x` on `y`.
  Before reporting an effect, it tries to *break* the claim the way a reviewer would
  by testing an observed confounder that overturns it, a collider among the controls that
  created it, fragility under resampling, reliance on outliers, and (given
  `negative_controls`) latent confounding, then returns **sound / unsound /
  inconclusive** with the pivotal issue. Deterministic; it does *not* decide causal
  direction (that needs domain knowledge, surfaced as a caveat). See
  [Verification](#verification).
- **`verify_comparison`, `verify_correlation`, `verify_trend`, `verify_regression`,
  `verify_prediction`**: the same oracle for the other claim shapes a data scientist
  reports: a group difference (significance, effect size, Simpson's-paradox
  reversal, and the rank test on small non-normal samples), a linear association
  (significance, confounding), a trend over time (autocorrelation, endpoints,
  outliers), a fitted OLS coefficient (multicollinearity, heteroskedasticity,
  influence, misspecification), and a model's predictive accuracy (re-evaluated
  leakage-free, screening target proxies and train/test contamination). Each returns
  the same **sound / unsound / inconclusive** verdict. See [Verification](#verification).
- **`run_code`**: run Python in the sandbox to explore data and prototype a
  computation *before* authoring a derivation. Datasets load as `data['<name>']`
  (row dicts); `print()` to inspect and set a `result` variable to return a value;
  a code error returns as a traceback so the agent can iterate, as in a notebook.
  This is the explore/debug loop that precedes `propose_derivation`: the agent gets
  the computation right against real output, then crystallizes the working code as
  a governed derivation. It runs in the same hardened sandbox as authored
  derivations, so it adds no execution surface beyond what they already have.
- **`delete_derivation`**: remove an agent-authored derivation it no longer needs
  (e.g. a discarded attempt), unserving its tool. Human-authored derivations are
  protected.
- **`propose_derivation`**: runs the authoring loop over MCP. Submit source, it
  runs once under isolation, and under the default policy a clean verify certifies
  it and it is **served immediately** as `run_<name>` in the same session. The call
  returns the computed result inline (so the answer needs no second call), and the
  server emits `notifications/tools/list_changed` so a connected client also
  discovers the new `run_<name>` tool. If the server runs `ManualCertification`, it
  instead stays `proposed` until a human runs `elbi certify <name>`.

## Verification

Running an analysis proves it *executed*, not that it is *right*. Data analysis
has no built-in oracle the way software has tests. `verify_analysis` supplies the
data-decidable half of one. Given a claimed effect of `x` on `y` (optionally
holding `controls` fixed), it tries to *break* the claim with the checks a careful
reviewer runs, and reports whether it survives:

- **confounding**: an observed covariate that, held fixed, overturns the effect
  (caught by sign-flip or loss of significance);
- **collider control**: a control that *created* the effect by opening a path;
- **fragility**: the effect vanishes under resampling (signal vs. noise);
- **outliers**: the effect rests on a few extreme points;
- **latent confounding**: a *negative control* (a variable `x` cannot cause,
  supplied by domain knowledge) is nonetheless associated with `x`.

The verdict is **sound**, **unsound** (a check broke it, with the pivotal issue
named), or **inconclusive** (the claim is not even significant). It is
deterministic and reproducible. It deliberately does **not** decide causal
*direction*: `x→y` vs. `y→x` needs world knowledge, so it is surfaced as a caveat
for the agent to resolve, never silently assumed. `verify_effect(rows, x, y, ...)`
exposes the same engine in the SDK.

The same discipline extends to the other claim shapes, each with its own
characteristic silent failure: `verify_comparison` (a two-group difference, covering
significance, effect size, Simpson's-paradox reversal, and, on a small non-normal
sample, the assumption-free Mann-Whitney test), `verify_correlation` (a linear
association, meaning significance and whether a covariate explains it away; never causal),
`verify_trend` (a trend over time, covering autocorrelation that overstates significance,
endpoint artifacts, outliers), `verify_regression` (a fitted OLS coefficient, covering
multicollinearity, heteroskedasticity re-tested under robust standard errors,
influential points, functional-form misspecification), and `verify_prediction`
(a predictive claim, re-evaluated under an honest split, flagging a single-feature
target proxy and rows shared across the split, and reporting the held-out skill it
measures itself). The guarantee across all of them is the same: a bad claim never
comes back **sound**.

## Structure discovery

A schema tells an agent a column's *type*; it does not tell it how the columns
*relate*, and relationships are what decide a correct analysis: which columns are
redundant, which is an identifier, and (for a "what affects X" question) which
other columns share information with both X and the outcome and so might confound
it. `structure_map` recovers that skeleton from the values, with classical
deterministic methods (no model, reproducible):

- **functional dependencies** and **derived columns** (`C = A + B` exactly), which
  expose redundancy and perfect collinearity to keep out of a regression;
- **candidate keys** (a near-unique column is an identifier, not a variable);
- a **dependency graph** by normalized mutual information, computed on
  quantile-binned values so a high-cardinality column does not spuriously appear to
  determine everything (the bias of naive entropy on continuous data).

It is the *grounding* an agent reads before forming a hypothesis: the related
columns are the candidate confounders to control for. It reports structure, never
meaning, so it composes with the semantic model (meaning) and `run_code` (verify a
hypothesis the structure suggests). `analyze_structure(rows, columns)` exposes the
same result in the SDK.

## Dataset semantic model

Schema and an example row tell an agent a column's *type*, not its *meaning*, and
meaning is what decides whether an analysis is right: that a count column is a
frequency weight, that a flag is `0/1`, what a code stands for, what units a number
is in. This is the same gap a dbt or Snowflake semantic model fills, and an agent
reasoning over unfamiliar data hits it constantly.

A project can declare a per-column semantic model in `elbi.yaml`, surfaced
through `describe_dataset` so the agent reads it before writing any code:

```yaml
datasets:
  - orders                      # a bare name still works: an empty model
  - name: survey
    description: One row per surveyed household.
    columns:
      - name: income
        description: Annual household income, pre-tax.
        type: number
        unit: USD
        role: measure
        synonyms: [earnings, salary]
      - name: n_people
        description: Number of people the row represents; a frequency weight.
        role: weight
```

Each column field is optional. `role` is one of `dimension`, `measure`, `weight`,
`identifier`, `time`: deliberately **column-intrinsic** roles that describe what a
column *is*, never how to answer a question. The model documents the *data*
(meaning, unit, encoding, alternate names, example values); it must not encode a
question or its answer. Within that line it is a pure win: authored once by whoever
knows the data, reused and audited like the rest of the project, and that
description column in `describe_dataset` is where an agent learns that `n_people` is
a weight rather than just another integer.

## Search

`search_derivations` is how an agent finds the right derivation without holding
every tool in context, so its ranking quality decides whether selection stays accurate
as a library grows. Retrieval is a pluggable seam, like the cache store and the
executor: a `Registry` ranks with whatever `Retriever` it is given.

The building block is **BM25** (`Bm25Retriever`), the standard lexical ranker. It
scores a derivation's name and description as separate fields (BM25F) with the name
weighted higher, weighs rare query terms above common ones, saturates repeated
terms, and normalizes for length. It is deterministic and needs no index, and it is
strong whenever the query shares words with a derivation, which is the common case.
Names are matched across simple plurals (`homes` finds `home`).

Lexical search has one blind spot: a query that means the same thing in different
words. "income by area" shares no tokens with `revenue_by_region`. So the default is
**hybrid search**, the retrieval-systems standard: run BM25 and a semantic
(embedding) retriever, then fuse their rankings with Reciprocal Rank Fusion. RRF
combines ranks rather than scores, so a BM25 score and a cosine similarity (on
entirely different scales) merge with no normalization or tuning; a derivation both
rank highly rises to the top, and one found by only one of them still surfaces.

The semantic half is [model2vec](https://github.com/MinishLab/model2vec) static
embeddings: numpy only (no PyTorch), a retrieval-tuned model of about 32 MB, fast
CPU inference, and deterministic output. It is a core dependency, so hybrid works
out of the box with nothing extra to install. The model is downloaded once, on the
first search; building the server loads no model.

For a purely lexical, fully offline ranker, opt down in `elbi.yaml`:

```yaml
project: acme
search: lexical     # default: hybrid
```

A reranker (a cross-encoder over the top results) is deliberately not included: it
helps most on long, complex queries over large corpora, and adds latency with
little gain on the short queries and modest libraries seen here. It would slot in
behind the same `Retriever` seam if a deployment needed it. Any embedding model
fits the `Embedder` protocol, and a platform deployment can supply a hosted or
ANN-backed retriever through the same seam without touching a derivation.

## Persistence

An accepted proposal is durable: it is written to a sidecar under
`.elbi/authored/`, not the project's `derivations/` source tree, and
reloaded the next time the server starts. This lets an autonomous loop build a
library that survives restarts without the agent writing code into your trusted,
importable source.

Reloaded derivations keep their `agent` origin, so they are reconstructed with
their generated source and run only under the sandbox executor; the source is
never imported into the host process. A human certification recorded for one (see
below) still applies on reload, and `delete_derivation` removes the persisted
record (and purges its cached results).

Moving a derivation into committed source is a separate, deliberate step:

```bash
elbi promote <name>     # write derivations/<name>.py, then review and commit
```

`promote` writes the derivation into `derivations/` as a normal, editable file
with a provenance header (noting it was AI-generated, reviewed, and is now
human-maintained), reverts its origin to human so it runs in-process, and removes
the quarantined copy. It refuses to overwrite an existing file unless `--force`,
and `--dry-run` previews the output. This is the boundary at which generated code
becomes trusted, version-controlled source: the human reviews and commits it.

## Reviewing and certifying

For human-authored derivations marked `status="proposed"`, or any server running
`ManualCertification`:

```bash
elbi derivations        # list everything with origin + status
elbi certify <name>     # approve a proposed derivation for serving
```

Certification is recorded locally in `.elbi/lifecycle.yaml`. *Who* may
certify, signatures, and multi-party review are governance concerns handled by
the platform, not the local layer.
