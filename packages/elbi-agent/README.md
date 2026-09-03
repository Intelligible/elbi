# elbi-agent

The verifying analysis runtime for [elbi](https://github.com/Intelligible/elbi).

An LLM drives a tool loop over a dataset (describe it, map its structure, run
code, and report an effect), but the only way to *report a directional effect is
through a verification gate*. An effect that does not survive the soundness
checks (confounding, collider control, fragility, outliers, latent confounding)
is never returned as the answer; the model is handed the pivotal issue and keeps
working until the effect verifies sound or it concludes the data cannot support
one.

This is the enforced counterpart to the advisory MCP server: where a connected
agent *may* call `verify_analysis`, this runtime *cannot* answer with an
unverified effect, because verification is the answer channel itself. The same
runtime backs an interactive chat surface and unattended, scheduled runs.

The LLM is injected (`LLMClient`), so the loop runs against any provider; an
Anthropic client ships under the `anthropic` extra.
