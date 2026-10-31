# QRData held-out test of elbi's answer check, v2 (pre-registered 2026-10-30, before any held-out run)

## Background
Pilot (prereg.md, runs_pilot): base 29/40, elbi v1 31/40; neither win traced to the tool
(one method choice made before the tool was called, one recalled textbook answer). Misses
concentrated in Causality/numerical. A 3-round replication of the pilot's 10 causal-number
tasks (runs_rep1-3, v1 tool, unchanged) runs separately and asks whether v1's edge is real.

## Change being tested (v2: check_server_v2.py + effect_dr.py)
1. elbi's effect estimates: spline outcome model fitted per treatment arm, so the treatment
   term is not shrunk by regularisation (it collapsed to ~0 on binary outcomes); probability
   models cross-validated on log-loss. Plus a cross-fitted doubly robust (AIPW) estimate,
   with an influence-function SE, for a two-valued treatment (ATE / ATT / ATC).
2. When the agent declares the question asks for the effect itself (procedure_specified =
   false), elbi writes the equal-weight mean of the agent's estimate and its doubly robust
   estimate: the point-answer form of the CausalDS rule (two independent analyses, equally
   believed, combined). For a named procedure, elbi only checks, as in v1.
   Unchanged from v1: script re-run, range check (now including the DR estimate), refusal
   until confirm=true.

Development check (dev_check_v2.py, the pilot's tasks 11-19 only, offline, with the pilot
agents' declared variables): the combination left accuracy unchanged there (3/8 on the
unprescribed tasks either way); elbi's DR estimate alone would have scored 4/8. Not tuned
on: the combination rule was written before this check, and is kept.

## Design
- Held-out set: all 49 QRData Causality/numerical questions not in the pilot
  (exam_heldout.json). 25 are IHDP (golds are the simulated true effects), 7 jobs, 17 others,
  many of which name a method.
- Arms, one round, claude-opus-5-5 headless, same prompt: base; elbi (v1 tool); elbi2 (v2
  tool). Same one-sentence hint in both tool arms.
- Grading: QRData's published rule (numbers within 3% relative), grade_v2.py.

## Outcomes
- Primary: paired accuracy difference elbi2 - base (bootstrap 95% CI, win/loss).
- Secondary: elbi2 - elbi (does v2 add over v1); elbi - base (replicates the pilot's v1 edge?).
- Mechanism: how many answers the combination changed; how many it fixed vs broke.
- Exploratory only (not a decision input): accuracy had elbi written its DR estimate alone.

## Decision
- Adopt v2's ideas into elbi (per-arm spline; DR estimate; equal-weight combination for
  unprescribed effects) if elbi2 >= base and the combination fixed more answers than it broke.
- Otherwise keep only the spline fix (a correctness fix regardless of score) and report
  that the combination does not help point answers.
- One round, n = 49: a difference of 1-3 questions is within noise; report it as such.
