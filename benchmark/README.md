# Statistical-pitfall benchmark harness

A repeatable harness that drives the verification oracle
(`elbi_core.verification`) over a catalog of **trap datasets** and asserts each
gate catches the pitfall a bare reading misses. It is a regression net (a reverted gate
turns its trap red) and evidence of "verified inference." It is deliberately
**continuous with human review, not a static leaderboard**: the catalog grows as
reviewed data, and the bare-LLM baseline is refreshed and re-reviewed over time.

## Run it

```bash
just benchmark                  # run every trap, write benchmark/report/report.{md,json}, print a summary
uv run python -m benchmark.run  # the same, without just
```

The build fails only on **gate correctness**: a trap passes when its gate returns the
`expected_verdict` and (when pinned) the `expected_pivotal` reason appears in the report.
`benchmark/report/report.{md,json}` is gitignored and rebuilt each run.

> Each `just <recipe>` below is optional shorthand for the `uv run ...` command shown
> next to it. If `just` is not installed, run the `uv run ...` form directly.

### Run the tests manually

The traps are also a pytest suite (this is what CI runs on every PR):

```bash
uv run pytest benchmark/tests                 # the whole suite (traps + schema + harness)
uv run pytest benchmark/tests -v              # one line per trap and unit test
uv run pytest benchmark/tests/test_traps.py   # just the trap regression net

# a single trap by id (the folder name):
uv run pytest "benchmark/tests/test_traps.py::test_gate_catches_the_trap[simpsons_ab_reversal]"
```

- `test_traps.py` - every trap's gate must return its `expected_verdict` (+ pivotal).
- `test_schema.py` - every manifest validates, ids are unique, references resolve, and the
  reviewer-count policy holds (>=2 for real traps).
- `test_harness.py` - the loader / scoring / report / LLM-baseline plumbing (fake client).

### Generate the cached bare-LLM outputs

The "bare LLM vs gate" column reads a committed cache, `benchmark/llm_baseline.json`. It is
opt-in (needs a key) and never runs on a PR; regenerate it and review the diff before
committing:

```bash
# default model claude-sonnet-5:
ANTHROPIC_API_KEY=... just benchmark-llm-refresh
ANTHROPIC_API_KEY=... uv run python -m benchmark.run --refresh-llm   # the same, without just

# pin a different Claude model, or use another provider via LiteLLM:
uv run python -m benchmark.run --refresh-llm --model claude-opus-4-8
OPENAI_API_KEY=... uv run python -m benchmark.run --refresh-llm --provider litellm --model openai/gpt-5

git diff benchmark/llm_baseline.json          # review the captures, then commit
```

Each capture records `verdict`, `model`, `captured_at`, `prompt_version`, and a transcript
excerpt. Without a key the refresh prints a message and exits without writing, so it is
safe to invoke anywhere. See [the baseline section](#the-bare-llm-baseline-report-only) for
what the captures are used for.

## How it works

The harness is a thin driver around the real oracle; nothing here reimplements a gate.

```
traps/*/manifest.json ─(loader)→ Trap objects ─(harness)→ TrapResult ─(report)→ report.md / report.json
   manifests        validate + resolve          run gate + score        render
```

- **`loader.py`** validates each manifest against `trap.schema.json` and resolves its
  `data` block (a builder) and `gate` + `claim` (a verify call) into a runtime `Trap`.
- **`generators.py`** holds the named seeded row builders synthetic fixtures refer to.
- **`gates.py`** maps a fixture's `gate` to the actual `verify_*` function and calls
  `GATES[gate](rows, **claim)`. Every gate returns a report with `.verdict` / `.render()`.
- **`harness.py`** scores each trap: `passed` (gate correctness, the only build gate) and
  the report-only `beats_llm` / `warning` against the cached bare-LLM verdict.
- **`report.py`** renders markdown + JSON; **`run.py`** is the CLI; **`llm_baseline.py`**
  reads and refreshes the cache.

The per-trap call chain is `trap.verify(trap.data())` -> `gates.run_gate` ->
`verify_all(...)` (or a direct gate) -> the real oracle in `elbi_core`.

## Concepts

- **Pitfall** = a category (`simpsons`, `confounding`, `leakage`, `rtm`, `multiverse`,
  `selection-berkson`), each mapping to a gate. Roughly fixed.
- **Trap** = one instance: a dataset + a gate call + the ground-truth verdict it must
  return. Many traps per pitfall. Each trap is a folder `traps/<id>/` with a
  `manifest.json` and a committed `data.jsonl` (the rows the gate judged, so they are
  reviewable in the diff). The folder name is the trap id.
- **Verdict** = `sound` / `unsound` / `inconclusive` per [RUBRIC.md](./RUBRIC.md).
  `inconclusive` is a first-class, correct answer: if the rows can't decide (a confounder
  isn't measured, a sampling mechanism is undeclared), the honest verdict is inconclusive,
  never a guessed sound/unsound.

## Add a trap (no code)

A trap is a folder `traps/<id>/manifest.json` validated against `trap.schema.json`;
dropping one in makes it a new case automatically. Its `expected_verdict` + `rationale`
are the ground truth, keyed to [RUBRIC.md](./RUBRIC.md).

**Synthetic** (rows produced by a seeded generator in `generators.py`, then committed):

```json
// traps/confounding_ice_cream/manifest.json
{
  "pitfall": "confounding",
  "status": "candidate",
  "description": "Ice-cream sales and drownings correlate; temperature confounds both.",
  "provenance": "synthetic; temperature confounder (seeded generator)",
  "added_by": "you",
  "reviewed_by": ["you"],
  "data": { "file": "data.jsonl" },
  "generated_by": { "generator": "ice_cream_drownings", "seed": 0 },
  "gate": "verify_all",
  "claim": { "x": "ice_cream", "y": "drownings" },
  "expected_verdict": "unsound",
  "expected_pivotal": "temperature",
  "rationale": "RUBRIC confounding/unsound: effect vanishes conditioning on 'temperature'.",
  "naive_verdict": "sound",
  "naive_rationale": "a strong correlation reads as causal to a bare model"
}
```

Then `just benchmark-regen-data` writes the trap's `data.jsonl` from `generated_by`, and
you commit both files. The loader always reads the committed `data.jsonl`; `generated_by`
records how it was made (and marks the trap synthetic, so it needs only 1 reviewer). Only
a genuinely new synthetic *shape* needs code: add one function to `generators.py`, register
it in `GENERATORS`, and reference it from `generated_by`.

**Real-world** (a committed dataset the model could not have trained on) - a folder with
its data beside the manifest:

```
traps/<pitfall>_<context>/
  manifest.json      # "data": {"file": "data.jsonl"}, reviewed_by: [a, b] (>=2), provenance naming the source
  data.jsonl         # one JSON row object per line
```

Real traps require **>=2 reviewer sign-offs** (recorded in `reviewed_by`; enforced by
`test_schema.py`) and a `provenance` naming a **post-cutoff or private** source; famous
public datasets are contaminated (a model may recognise them from memory). `data` also
accepts `{"rows": [...]}` for tiny inline datasets.

## Generating the data

- **Every trap's rows are committed** as `traps/<id>/data.jsonl`, so the exact data the
  gate judged is reviewable in the diff. The loader reads that file.
- **Synthetic rows** are produced by a seeded generator (`generators.py`) named in the
  manifest's `generated_by`. Generators are authoring tools: after adding or changing one,
  regenerate the committed files and commit them:

  ```bash
  just benchmark-regen-data                         # rewrite every synthetic trap's data.jsonl
  uv run python -m benchmark.run --regenerate-data  # the same, without just
  ```

- **Real-world rows** are hand-committed `data.jsonl` with no `generated_by`; the source
  is recorded in that trap's manifest `provenance`.
- To eyeball any trap's rows as CSV without touching the committed files:
  `just benchmark-dump-data` (writes `benchmark/report/data/<id>.csv`, gitignored).
- **The bare-LLM baseline is the one cached *external* input** (a costly, non-deterministic
  call), committed at `benchmark/llm_baseline.json`; generate it with
  `just benchmark-llm-refresh`, described below.

## The `expected_pivotal` makes it bite

Each trap pins the *specific* reason the gate must give (`"Simpson"`, `"temperature"`,
the leaking column, `"collider"`, `"regression to the mean"`). So if a gate stops
catching a pitfall, its trap fails on the wrong or missing pivotal, not just the verdict.
To see it by hand: revert a gate's detection (for example the Simpson's scan in
`verification/experiment.py`, which `simpsons_ab_reversal` runs), run
`uv run pytest benchmark/tests`, watch the trap go red, then restore.

### The bite gate (`just benchmark-bite`)

`just benchmark-bite` (or `uv run python -m benchmark.bite`) automates that with Cosmic
Ray. For each trap it mutates only the one gate module the trap targets - the trap's
`bite.module` - running just that trap's case, and requires the trap to **kill at least
one mutant**. A trap that kills none passes regardless of its gate, so it is a vacuous
trap and the gate rejects it (non-zero exit).

This is a **non-vacuity** check, not a zero-survivor one: surviving mutants are fine -
they are gate behaviour that trap need not exercise - so it never false-alarms on
unrelated code. It stops at the first kill, so a live trap finishes in seconds. In CI the
blocking `benchmark-bite` job runs it on every PR, one runner per gate module (Cosmic Ray
mutates source in place, so a module cannot be shared across concurrent sessions). The
deeper, non-blocking whole-module survivor sweep lives in `benchmark-mutation.yml`.

`bite.module` (a filename under `verification/`) is inferred from `gate` for a
single-purpose gate; a `verify_all` trap runs the whole catalog, so it must name its
detection module explicitly (for example `effect.py` for a confounding trap).

## The bare-LLM baseline (report-only)

Each trap records a hand-written `naive_verdict` (what a surface read concludes). A
**cached real-LLM verdict** is kept alongside it in `benchmark/llm_baseline.json`
(committed, read free on every PR). The gate-vs-LLM comparison is **report-only**: it
never fails the build. When a trap's gate does not beat the cached LLM, that trap is
**saturated** (low discrimination) - the report labels it and lists it under "flagged for
pruning," per the IRT/benchmark-saturation grounding.

Regenerate the cache with `just benchmark-llm-refresh` (default `claude-sonnet-5`), or
against another model/provider - see
[Generate the cached bare-LLM outputs](#generate-the-cached-bare-llm-outputs) above.

## Human-review hook (adjudication)

Adding or editing a trap is a **PR** gated by `CODEOWNERS` on `benchmark/traps/`. The
reviewer count is enforced by `test_schema.py`:

- **Synthetic trap** (seeded generator): **1** sign-off - the answer is deterministic.
- **Real-world trap** (file-backed): **>=2** independent reviewers who label against
  [RUBRIC.md](./RUBRIC.md), then reconcile; if they can't agree, the verdict defaults to
  `inconclusive` with a note. Record sign-offs in the manifest's `reviewed_by`.

New traps land as `status: "candidate"` and are promoted to `"accepted"` once they pass
this checklist (both statuses run in CI and must pass; `status` is a vetting label, not a
correctness escape hatch):

- [ ] A real, named statistical pitfall from [RUBRIC.md](./RUBRIC.md), not a contrived edge case.
- [ ] The `expected_verdict` is defensible under the rubric (including `inconclusive` when
      the rows can't decide), and `rationale` names the rule that fired.
- [ ] A bare LLM actually misses it (check the cached capture / run a refresh); if the
      bare LLM also catches it, the trap is saturated and is a pruning candidate.
- [ ] `expected_pivotal` is specific, so the trap bites on the right reason.
- [ ] For real data: small, license-clean, from a **post-cutoff or private** source
      (contamination-safe), provenance in the manifest, `reviewed_by` has >=2 entries.
- [ ] The trap kills a mutant of its gate module, so `just benchmark-bite` accepts it
      (a trap that passes regardless of its gate is rejected in CI).

`reviewed_by` is the auditable record; the actual sign-off gate is the GitHub PR approval
via CODEOWNERS. This mirrors the SDK's `elbi certify` / `lifecycle.yaml` review
seam. A refresh where a newer model now *catches* a trap surfaces as saturation (the
frontier moved), not a failure - triage whether to retire, harden, or keep it as proof.
