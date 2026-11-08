# Pre-registration: elbi submit_answer vs plain Claude (written 2026-10-24, before any agent run)

Arms: base (plain Claude Code, Opus 5.5) vs submit_hint (elbi MCP: plan_analysis + submit_answer;
one-line hint "Write your final answer with submit_answer").

Tool frozen before agent runs: blade-mini/effect_check.py + plan_server.py submit_answer.
Tool interval = hull(spline bootstrap CI, three model point estimates) widened by 25% each side;
final = hull(agent CI, tool interval). Chosen on dev only (odd-numbered main scenes, tool-only,
true adjustment set): 2 interval rules compared, nothing tuned on agent answers.

Sets:
 A. exam_main_even.json: 50 ate_uq_95 tasks, even-numbered main scenes (never used for tuning). base and submit_hint, 1 round.
 B. exam_test_regen.json: 42 tasks on the regenerated test scenes. submit_hint added to round 2 (paired with the
    existing base r2), plus a new round 3 with both arms. Round 1 (original scene draws, now deleted) is not used.

Primary outcome: per task, interval score (width + 40 x miss) paired submit_hint vs base; better/worse counts and
sign test, by variant (clean, proxy, proxy_hard) and pooled. A missing answer counts as worse.
No-harm: abstention correctness on non-identifiable tasks must not drop.

## Amendment (2026-10-24, before any agent run of this rule)
The hull+25% rule was replaced before its runs finished (partial results kept in runs_main_even/submit_hint_hullrule,
not used): the 25% was a free parameter. New rule, no fitted parameters: each analysis is a normal with its own
sampling SE (agent: from its CI; elbi: spline, linear, boosted g-computation with bootstrap SEs); agent and elbi
weighted 1/2 each, elbi's half split equally across its models; final interval = central 95% of the mixture.
Dev check (tool only, odd main scenes): clean cover 13/15 mean IS 1.14; proxy 9/10 mean IS 1.22.
Primary outcome amended: total (equivalently mean) interval score per variant, paired bootstrap CI over tasks.
Win/loss counts with sign test reported as secondary. Reason: interval score is the proper scoring rule the task
is graded by; a sign test weighs a 0.01 width cost the same as a 20-point miss.
