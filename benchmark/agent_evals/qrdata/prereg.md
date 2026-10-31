# QRData pilot: does elbi's CausalDS gain transfer? (pre-registered 2026-10-30, before any agent run)

## Question
On CausalDS, elbi improved Claude Opus 5.5 because elbi analysed each effect independently
and wrote the final answer, so its second analysis could not be skipped. QRData (Liu et al.,
ACL Findings 2024; 411 questions, real data, deterministic grading) asks for point answers
and choices, not intervals. Does the same principle help: elbi re-derives what it can and
refuses to write an answer it cannot reproduce or that its own analysis contradicts, until
the agent has looked at the disagreement?

## Design
- Sample: 40 questions, stratified by (first keyword, type), seed 0, frozen in
  exam_pilot.json (make_exam.py): Causality MC 10, Causality numerical 10, Statistics
  numerical 10, Statistics MC 5, Social science numerical 5.
- Arms (qr_run.py), one round, Claude Code headless, claude-opus-5-5, same prompt and tools:
  - base: writes answers/answer.txt itself.
  - elbi: plus one MCP tool, submit_answer (check_server.py), and one sentence asking it to
    write the final answer with it. The tool re-runs the agent's script from scratch and
    checks it prints the answer (always); optionally re-estimates an average effect
    (ATE/ATT/ATC, three g-computation models, bootstrap; flags an answer outside
    estimate ± 2 SE of all models) or reports association tests (never blocks). A flagged
    answer is written only on resubmission with confirm=true and a reason.
  - No QRData dataset or question is named in the tool; nothing is fitted to this sample.
- Grading (grade.py): QRData's published rule (upstream eval.py, exec'd verbatim): numbers
  within 3% relative, choices by prefix match. Missing answer = wrong.

## Primary outcome
Paired accuracy difference elbi − base over the 40 tasks, with a paired bootstrap 95% CI,
plus win/loss counts. Secondary: per-stratum accuracy; cost; how often the tool was used,
refused an answer, and how often a refusal changed the final answer (the mechanism).

## Decision (pilot, so directional)
- Continue to all 411 questions (base vs elbi) if elbi's accuracy is at least base's and
  the tool refused at least one answer that was then changed for the better.
- Stop and report "the CausalDS gain does not transfer to point answers" if elbi is worse,
  or equal with no refusal having changed an answer.
- n = 40 cannot detect a few-point difference; a null is "not shown", not "no effect".

## Known limits
- One round; the model's run-to-run variance is not estimated.
- QRData is CC BY-NC 4.0; results are for internal research.
- Published QRData numbers (GPT-4 etc., 2024) use a different harness and are not comparable.
