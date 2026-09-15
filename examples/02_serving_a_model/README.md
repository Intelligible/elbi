# 02 · Serving a model

A runnable project that trains a model and serves its predictions to an agent.
It shows the pattern behind "a derivation is any computation, not just a metric":
an opaque model artifact and structured prediction input.

```bash
cd examples/02_serving_a_model
elbi validate
elbi mcp          # serve at http://localhost:7878/mcp
uv run pytest             # from the repo root, runs as part of the suite
```

It defines three derivations:

- `churn_model`: trains a model from the `customers` dataset and returns it as an
  opaque object (`Artifact.opaque`). It has no serve contract, so it is internal:
  consumed by the predictors, never exposed to an agent.
- `predict_churn`: served over MCP. Takes one customer record as an `object`
  parameter and returns a churn probability. The model is an input, so it is
  trained once and cached, not refit on every call.
- `predict_churn_batch`: served over MCP. Takes an `array` of customer records and
  returns a probability for each.
- `predict_churn_whatif`: served over MCP. A what-if engine, not a second model --
  it reuses the same cached `churn_model` and the same scorer `predict_churn` does.
  Takes a `base` record and an `array` of `scenarios` (partial records, each merged
  onto `base`), and returns a table of each scenario's probability and its `delta`
  from the baseline. An agent (or a human) can ask "what if this customer's recency
  were half what it is?" without writing any scoring code of its own.

The model here is plain Python so the example needs no ML dependency. The shape is
the same with scikit-learn or PyTorch: fit in the training derivation, return
`Artifact.opaque(model)`, and read it back with `ctx.input("model").value`.
