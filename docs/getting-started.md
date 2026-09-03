# Getting started

## Install

```bash
uv tool install elbi
```

This puts the `elbi` and `elbi-app` commands on your PATH. (For
library-only use in an existing project, `uv add elbi`. For just the
CLI and MCP, with none of the chat app's heavier dependencies, `uv tool install
elbi-cli` on its own.)

Later, `elbi update` tells you whether a newer release exists and prints the command
that upgrades your particular install; see [Upgrading](upgrading.md). It only reaches
the network when you run it.

## Scaffold a project

```bash
elbi init acme-data-context
cd acme-data-context
```

`init` runs fully offline: no login, no cloud call. It writes:

```
acme-data-context/
├── elbi.yaml          # committed, logical wiring
├── elbi.dev.yaml      # local-only data bindings (gitignored)
├── derivations/churn_risk.py  # an example derivation
├── fixtures/sales.csv         # sample data for local dev + tests
├── tests/test_derivations.py  # tests of the compute functions
└── pyproject.toml
```

Use `--template minimal` for a bare project with no `derivations/`.

## Serve it

```bash
elbi serve
```

No API key needed to start: this opens your browser to a chat UI at
`http://localhost:7700` automatically once ready, and serves the same
derivations over MCP at `http://localhost:7700/mcp` -- point any MCP client at
that URL too, and you are driving the same agent loop you would have in
production. The chat UI needs a model to actually talk to, though -- add one in
Settings the first time you send a message, or skip that step by exporting a
key beforehand, e.g. for the default model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The first run takes ~15-20s longer than later ones while it warms up (loading
the embedding model, initializing the local database); that is expected, not a
hang -- the browser opens the moment it is actually ready, not before. Pass
`--no-browser` to skip opening one (e.g. on a headless machine).

Only want the MCP endpoint? Use `elbi mcp` instead -- no chat app, none
of its heavier dependencies, and no LLM key of its own to configure: whatever
MCP client you connect brings its own model and credentials.

```bash
elbi mcp
```

```
→ discovered 1 derivation(s) in derivations/
→ resolved data from elbi.dev.yaml
→ MCP server ready at  http://localhost:7878/mcp
```

## Validate and test

```bash
elbi validate   # check the project against the spec
uv sync                 # only needed for uv run; serve/mcp use their own environment
uv run pytest           # run your derivation tests
```
