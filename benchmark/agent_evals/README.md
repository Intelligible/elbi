# Agent evals: does elbi make a coding agent better at data science? (WIP)

> **Work in progress.** These are prototype harnesses with one or two rounds per result;
> they have not yet been ported onto the shipped `reconcile_effect` tool or replicated
> there. Treat the numbers as evidence for the mechanism, not as published results.

These harnesses measure an agent with elbi against the same agent without it, on public
benchmarks with deterministic grading. They are the **prototypes that produced the results
below**, kept as they were run so the numbers stay reproducible. The ideas they validated
now ship as `elbi_core.verification.second_opinion` and the `reconcile_effect` MCP tool;
porting these harnesses onto that shipped tool is the first follow-up (see
[Next steps](#next-steps)).

Everything here runs headless Claude Code (`claude -p`, model `claude-opus-5-5`) in a
fresh directory per task, with web tools and subagents disabled. The two arms get the same
prompt; the elbi arm adds one MCP server and one sentence telling the agent to write its
final answer through it.

## What was found

The gain comes from a second, independent analysis whose result is combined with the
agent's, by elbi, when it writes the final answer:

- elbi re-estimates the effect itself (three outcome models with bootstrap errors, and a
  doubly robust estimator for a two-valued treatment), given the agent's adjustment set;
- the two analyses are combined at equal weight: an interval becomes the central interval
  of the mixture of both (it widens where they disagree), a point becomes their mean;
- elbi writes the answer, so the combination cannot be skipped. An advisory tool the
  agent could ignore moved nothing.

Neither the agent alone nor elbi alone does as well as the two combined: on 117 CausalDS
interval answers, total interval score was 483.8 (agent alone) and 483.2 (elbi alone)
against 217.0 combined.

### Where each gain comes from

| Gain | Source of the improvement | Shipped as |
|---|---|---|
| CausalDS: 0.055 of the 0.060 drop in CausalDSScore (about 91%) | Interval combination on the 7 graded average-effect interval tasks: capped nIS 6.88 to 2.86 (round 1) and 9.70 to 2.78 (round 2). The 7 prediction-interval tasks elbi does not touch stayed flat (12.57 vs 12.55, 12.55 vs 12.56). | `second_opinion.combine_interval`, applied by `reconcile_effect` |
| CausalDS: the remaining 0.005 | Pass rate 97.1% to 100% in round 1 only (equal in round 2), plus 0.001 from non-interval error. Not traced to the tool: agent-side run-to-run difference. | (none) |
| QRData held-out: 2 of the 4 extra correct answers | Point combination with elbi's doubly robust estimate fixed 2 answers. It also broke 1 the agent had right, which base had missed too, so it cost nothing against base. | `second_opinion.combine_point` and `estimate_effect`'s AIPW, applied by `reconcile_effect` |
| QRData held-out: the other 2 | The agent's own answer differed from base's run; the tool did not change them. One (hospital stay) matches a pattern seen in the pilot, where agents with an elbi tool got a sign right 4/4 vs base 2/4. Not attributed. | (none) |
| No score change measured | Script re-run and the range check (v1): never changed an answer for the better. The range check would have blocked one base sign error (+1.98 against elbi's -5.6 to -16.9). | `reconcile_effect` withholds an out-of-range estimate until confirmed |
| No score change measured | Per-arm spline (the spline's penalty had shrunk a weak binary effect to about 0) and the sparse-data guard: correctness, not separately scored. | `second_opinion.estimate_effect` |

### CausalDS realistic exam (100 tasks, the benchmark's own grader)

CausalDSScore, lower is better. The elbi arm uses `submit_answer` only on tasks asking for
an average effect with a 95% interval (10 of the 100).

| Round | base | elbi | elbi, shipped tool (replay) |
|---|---|---|---|
| 1 | 0.2132 (pass 97.1%) | 0.1611 (pass 100%) | 0.1623 |
| 2 | 0.2396 (pass 97.1%) | 0.1705 (pass 97.1%) | 0.1716 |
| Mean of 2 | 0.226 | 0.166 | 0.167 |

The last column replays each recorded `submit_answer` call through the shipped
`reconcile_effect` (`causalds/replay_shipped.py`, graded by `causalds/grade_replay.py`; no
agent run), on the same rows the prototype used: the calibration file's true values when a
scene has one. Every other answer is the recorded one. Per task, the per-arm spline helps
where the prototype's joint spline had shrunk the effect (t084: interval score 0.12 to
0.01), and one task gets worse (t069: 0.49 to 0.87). There elbi's own models disagree
widely (boosted 0.10, linear 0.34, doubly robust 1.04, spline 1.11, against a true 0.42),
and the combined interval stretches to cover them.

Round 3 stopped at 16 of 100 tasks and is not reported. The published leaderboard uses
CausalDS's own agent harness and three-run means, so these numbers are not comparable to
it. CausalDS's 200-task hard exam is not public.

### QRData (Liu et al., ACL Findings 2024)

Accuracy under QRData's published rule (numbers within 3%, choices by prefix).

| Set | base | elbi v1 | elbi v2 |
|---|---|---|---|
| Pilot, 40 questions stratified (1 round) | 29/40 | 31/40 | |
| Pilot's 10 causal-number questions, 3 more rounds | 9/30 | 9/30 | |
| Held-out causal-number questions, 49 (1 round) | 27/49 | 27/49 | **31/49** |

- **v1** (`qrdata/check_server.py`) re-runs the agent's script, checks an effect answer
  against elbi's range, and blocks until the agent confirms. It never changed an answer
  for the better; the pilot's +2 did not replicate.
- **v2** (`qrdata/check_server_v2.py`, `qrdata/effect_dr.py`) adds the per-arm spline,
  the doubly robust estimate, and the equal-weight mean when the question asks for the
  effect itself. Held out: 4 better and 0 worse than base (paired 95% CI +2.0 to +16.3
  points). Of the 4, the combination itself accounts for 2 (it also broke 1 answer
  base got wrong anyway); the other 2 came from the agent. One round only.
- **Replayed through the shipped tool** (`qrdata/replay_shipped.py`, no agent run): each
  v2 agent's final submission passed to `reconcile_effect` instead of the prototype
  scores 31/49, the same as the prototype, with no question graded differently. With a
  single cross-fitting split the replay scored 30/49: on one question (t021, ATT) the
  doubly robust estimate ranged from 3.93 to 4.45 across 8 random splits, enough to move
  the combined answer across the 3% band. The shipped estimate is therefore the median
  over 5 splits; that costs about 50% more time per effect check.
- elbi's doubly robust estimate *alone* would have scored 16/37 where the agent's final
  scored 24/37: the combination helps, replacing the agent does not.

### Things that did not move the score

- Advice the agent can ignore (`estimate_effect` returning a second opinion with a rule
  for reconciling it): unhinted agents never called it; hinted agents called it but did
  not follow the rule. No measurable win on the CausalDS test set.
- Storing what an earlier session learned (as a typed claim with fitted models, or as
  notes and code) for a follow-up question on the same data: no arm differed beyond
  noise in 24 paired runs, and the reuse tool was called once.
- Structure planning before analysis (`plan_analysis`) on BLADE: no measurable win.

## Layout

```
prototype/   the CausalDS-era elbi tools
  plan_server.py    MCP server: plan_analysis, submit_answer (PLAN_SUBMIT=1), and
                    experimental arms (typed claims, reuse) behind env flags
  effect_check.py   three g-computation models, bootstrap SEs, mixture_interval
  latent_effect.py  spline g-computation, including calibration-file proxies
  effect_claim.py   typed effect claims (identification check, ETT reuse)
causalds/    cds_run.py (runner), make_exam.py, official_report.py (grader summary),
             run_official.sh (3 rounds, resumes after usage limits), prereg.md,
             replay_shipped.py + grade_replay.py (recorded answers through the
             shipped tool, then graded)
qrdata/      qr_run.py (runner; arms base, elbi = v1, elbi2 = v2), check_server.py,
             check_server_v2.py, effect_dr.py, make_exam.py, grade.py (pilot),
             grade_v2.py (held-out), replay_shipped.py (recorded answers through
             the shipped tool), prereg.md, prereg_v2.md, run_*.sh,
             exam_*_qids.json (the question ids each exam must contain)
```

## Setup

1. Claude Code on `PATH` as `claude`, logged in. Runs consume your plan's usage; the run
   scripts wait and resume when a limit is hit.
2. The elbi side runs with this workspace's Python (`uv run`); override with `ELBI_PY`.
3. CausalDS:
   ```bash
   git clone https://github.com/andleb/causalds benchmark/agent_evals/causalds/upstream
   # follow its README to create upstream/.venv (the agent's Python for both benchmarks)
   uv run python benchmark/agent_evals/causalds/make_exam.py
   ```
   `CAUSALDS_DIR` points elsewhere if the clone lives elsewhere. Results above used
   commit `3d675d3`.
4. QRData (CC BY-NC 4.0, so its data is not included here):
   ```bash
   git clone https://github.com/xxxiaol/QRData benchmark/agent_evals/qrdata/upstream
   unzip benchmark/agent_evals/qrdata/upstream/benchmark/data.zip -d benchmark/agent_evals/qrdata
   uv run python benchmark/agent_evals/qrdata/make_exam.py   # checks the ids match
   ```
   `AGENT_BIN` sets the agent's Python `bin` directory (default: CausalDS's venv).

## Run

```bash
cd benchmark/agent_evals/causalds
./run_official.sh                   # 100 tasks x base/elbi x 3 rounds
uv run python official_report.py
uv run python replay_shipped.py && upstream/.venv/bin/python grade_replay.py

cd ../qrdata
./run_pilot.sh && uv run python grade.py
RUNS=runs_heldout EXAM=exam_heldout.json ARMS="base elbi elbi2" uv run python qr_run.py
RUNS=runs_heldout EXAM=exam_heldout.json uv run python grade_v2.py
```

A single task: `uv run python qr_run.py 11` (task numbers are 1-based in the exam file).

## Caveats worth knowing

- **Pre-registration.** Each QRData step was written down before its runs (`prereg*.md`),
  including the decision rule. The CausalDS rule was amended once before its runs, to
  replace a free widening parameter with the equal-weight mixture (`causalds/prereg.md`).
- **The v2 tool file is pinned.** During the held-out run, `check_server_v2.py` was
  edited mid-run outside the experiment. Tasks 21 to 49 were rerun with the frozen file,
  whose MD5 (`60de2305ad3a38219724bbad7a9f45d0`) matches the one recorded at the freeze;
  that is the file here, and the 31/49 uses it throughout. `check_server.py` (v1) differs
  from what ran only in the path it imports `effect_claim` from.
- **Memorization.** QRData comes from public textbooks; in one task the agent said its
  answer matched "the value I recall from the textbook". Treat absolute accuracy with
  care; the paired base vs elbi difference is the result.
- **Percent answers.** The prompt asks for "the number only", so agents write `4.11`
  where the key has `4.11%`, which the grader reads as 0.0411. Two held-out questions
  are lost this way in every arm.
- **IHDP dominates the held-out set** (25 of 49). Its keys are the simulated true
  effects, where averaging two estimates naturally reduces noise.
- **The shipped module differs from the CausalDS prototype**: its spline is fitted per
  treatment arm, and it has no calibration-file handling for proxy variables
  (`prototype/latent_effect.py`). It has been checked on CausalDS only by the offline
  replay above, which hands it the calibration rows directly.

## Next steps

1. Port the runners to the shipped server (`elbi` MCP with a dataset bound per task, the
   agent told to report `reconcile_effect`'s answer) and reproduce both tables. The
   offline replay checks the tool's arithmetic on recorded submissions, not how an agent
   uses it. Decide how CausalDS's proxy variables and calibration file reach the shipped
   tool.
2. A second round of the QRData held-out set (base vs v2, about 100 runs) before any
   public claim.
3. Finish CausalDS round 3.
4. Open question from the CausalDS replay (t069): when elbi's own models disagree widely,
   each still takes an equal share of elbi's half of the mixture. Down-weighting elbi when
   its models spread far more than their standard errors is one option; one task is not
   evidence for it, so it needs a held-out check before any change.
5. Mutation-test `second_opinion` (`just mutation` with `module-path` pointed at it).
5. **Explore a persistent-memory benchmark (proposed; not yet committed).** Audit
   [DSAEval](https://github.com/AMA-CMFAI/DSAEval)'s dataset groups for multi-question
   sequences with genuine analytical dependencies. Check sequence lengths, what knowledge
   could carry over, and whether the existing scoring can detect a benefit. If the audit
   finds enough suitable sequences, pilot four conditions: a cold agent, an agent restarted
   with transcript/log memory, an agent restarted with ELBI's structured empirical memory,
   and a continuous session. Compare accuracy, tokens, wall-clock time, analytical steps,
   and memory creation/maintenance cost; the key comparison is ELBI versus transcript
   memory. Keep this distinct from the prior CausalDS reuse result: 24 paired runs showed
   no measurable accuracy difference, and the reuse tool was called once. The audit is a
   gate, not a commitment to run the full benchmark or evidence that memory helps.
6. **Check whether we can score method selection on QRData (exploratory).** Match the
   held-out questions against CauSciBench's QRData-CI items and reference method labels.
   Determine whether the saved transcripts preserve each arm's selected method well enough
   to compute MSA on a clearly defined overlap. Do not compare it directly with the published
   61.5% unless the questions, prompts, and scoring protocol are comparable; our current
   51% result is numeric-answer accuracy on a different 49-question set.
7. **Consider DABstep as a later benchmark (tentative).** Revisit it if the QRData tool
   pattern transfers and we want to test whether ELBI's targeted, explicit checks help on a
   broader data-analysis benchmark. First verify that its tasks and scoring support a fair,
   reproducible with/without-ELBI comparison.
