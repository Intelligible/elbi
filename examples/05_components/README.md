# 05 · Components

A derivation whose artifact is a list of
[OpenReasoningComponents](https://github.com/openreasoningcomponents/openreasoningcomponents)
(ORC)-shaped natural-language facts, instead of a table. Complete and runnable,
committed and CI-tested like the other examples.

```bash
cd examples/05_components
elbi validate
elbi mcp          # serve at http://localhost:7878/mcp
uv run pytest             # from the repo root, runs as part of the suite
```

It defines one derivation:

- `churn_components`: four grounded statements about the `customers` dataset --
  a column definition, a distribution, a threshold rule, and a segment --
  each with real evidence computed from the data (odds ratios, sample sizes,
  segment size), not hand-typed numbers.

`run_churn_components` returns each statement as inline text (what an agent's
reasoning actually sees) and the full component objects -- `structure`,
`evidence`, `relations`, `provenance` -- as MCP structured content.
`provenance.derivation`/`provenance.derivation_version` are stamped
automatically at serve time from this derivation's own content-hash version,
so a consumer can check whether a component is still current the same way
Elbi decides any derivation is stale, without ORC needing any refresh
machinery of its own. `search_components` finds these statements by keyword or
meaning, the same way `search_derivations` finds a derivation.

`fixtures/generate_customers.py` regenerates the committed fixture (seeded, so
it's reproducible) and can produce it at other sizes -- used by
`openreasoningcomponents`'s eval to test whether components' token advantage
over raw data widens as a dataset grows, since components scale with the
number of distinct facts rather than with row count.
