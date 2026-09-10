# 03 · Yesterday's skill vs this morning's bar

1. `./naive.sh`
2. `./elbi.sh`

`skills/risk.md` was written after the August 6, 2026 close and tells an agent
to cut TTD 3%. `fixtures/returns.csv` already has the next morning's bar: TTD
fell 21.9% on an earnings miss. Only one of those answers is allowed to reach
a broker.

`naive.sh` is what a file-backed agent does: read the skill, act on what it
says. `elbi.sh` is the same question asked as a tool -- `run_risk_overlay`
recomputes from whatever is in `fixtures/` right now, over MCP, with no
skill file in between.

That is the forwardable artifact. If this README needs a diagram, the folder
is too clever.

## About the data

`fixtures/returns.csv` is real: daily closes for TTD, JNJ, PG, and HD from
July 2 through August 7, 2026, via Yahoo Finance's public chart endpoint. TTD's
August 7 drop is a real, reported event (a revenue and EPS miss). The
portfolio in `fixtures/positions.csv` -- the weights, and the choice of these
four names as one book -- is a hypothetical example, not anyone's real
holdings; only the prices are sourced, not the position.

```bash
cd examples/03_stale_skill
uv run pytest             # from the repo root, runs as part of the suite
```
