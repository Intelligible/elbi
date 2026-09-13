# 04 · Certified models

A human-vetted default-risk model is one `run_default_risk` call away. An agent
that instead tried to propose its own quick model, using a feature that only
exists after a loan has already defaulted, never gets that far: the
verification oracle catches the leak before certification, and the proposal
never becomes an MCP tool at all.

```bash
cd examples/04_certified_models
elbi validate
elbi mcp          # serve at http://localhost:7878/mcp
uv run pytest             # from the repo root, runs as part of the suite
```

This complements [`03_stale_skill`](../03_stale_skill/README.md), which is
about a different failure mode: a *stale* answer (a cached skill file that
outlives the data it described). This example is about an *unsound* one --
even freshly computed, a claim built on a leaked feature is wrong. Both point
at the same principle: an agent calls a governed tool, never its own analysis.

It defines three derivations:

- `default_model`: fits per-feature weights from four legitimate,
  available-at-application-time features (income, debt ratio, credit score,
  loan amount). Internal (`Artifact.opaque`); never served directly.
- `default_risk`: served over MCP. Scores every application in
  `fixtures/loans.csv` under the vetted model.
- `default_risk_components`: served over MCP as
  [components](../05_components/README.md) -- the same certified finding,
  expressed as durable, natural-language statements with their own evidence
  (an odds ratio, a sample size), not a table an agent re-reads and
  re-interprets on every call. This is what "certified" becoming "durable,
  searchable memory" means concretely: `search_components` finds
  "applicants with a debt ratio above 40% default substantially more often"
  by meaning, and its `provenance.derivation`/`derivation_version` are
  stamped automatically, the same staleness check every other derivation
  already gets.

The actual demonstration is in `tests/test_example.py`, mirroring
[`test_authoring_oracle.py`](../../packages/elbi-core/tests/test_authoring_oracle.py):
the identical computation is proposed twice, through `elbi_core.author()`,
with two different claims. Claiming the four legitimate features certifies
(`oracle_verdict == "sound"`). Adding `days_past_due` -- a column that is only
ever nonzero *after* a default has already happened -- gets caught by the
leakage and predictive-soundness gates (`oracle_verdict == "unsound"`), and the
proposal stays `status: "proposed"`. Two tests prove the point at the public
boundary: building the same MCP server the real deployment would, the
rejected proposal's tool name is simply absent from `list_tools()`, and it
contributes nothing to `search_components` either -- not marked untrustworthy,
not there, and not findable any other way.
