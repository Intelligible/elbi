# 01 · Getting started

A complete, runnable elbi project. This is what `elbi init`
scaffolds, committed and CI-tested so the examples never rot.

```bash
cd examples/01_getting_started
elbi validate
elbi mcp          # serve at http://localhost:7878/mcp
uv run pytest             # from the repo root, runs as part of the suite
```

It defines two derivations:

- `churn_risk`: per-customer risk scores from the `sales` dataset.
- `pricing_effects`: a Markdown summary built on top of `churn_risk`, showing
  how derivations compose.
