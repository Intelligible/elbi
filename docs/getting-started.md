# Getting started

## Install

```bash
uv tool install elbi-cli
```

This is the CLI and the MCP server: no chat app, no model key to configure. (For
library-only use in an existing project, `uv add elbi`.)

Later, `uv tool install elbi` adds the browser app on top (`elbi serve`; see
[Browser app](#browser-app) below), and puts the `elbi-app` command on your PATH
alongside `elbi`. `elbi update` tells you whether a newer release exists and prints the
command that upgrades your particular install; see [Upgrading](upgrading.md). It only
reaches the network when you run it.

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

## Serve it over MCP

```bash
elbi mcp
```

```
→ discovered 1 derivation(s) in derivations/
→ resolved data from elbi.dev.yaml
→ MCP server ready at  http://localhost:7878/mcp
```

No API key, no chat app: whatever MCP client you connect brings its own model and
credentials. Point one at that URL:

**Cursor** -- a project file at `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "elbi": {
      "url": "http://localhost:7878/mcp"
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add --transport http elbi http://localhost:7878/mcp
claude mcp list
```

**Codex CLI**:

```bash
codex mcp add elbi --url http://localhost:7878/mcp
```

**Claude Desktop**: the same URL, in its MCP config. **Inspector**, with no client of
your own: `npx @modelcontextprotocol/inspector`.

`elbi mcp` speaks streamable HTTP only, so a client cannot spawn it over stdio. Leave
the process running in its own terminal; every client points at the same URL.

Your first win: the client lists a tool named `run_churn_risk`, and asking it who has
the highest churn risk answers from that tool, not from the model inventing SQL against
data it does not have.

## Browser app

```bash
elbi serve
```

Opens a full browser environment at `http://localhost:7700` automatically once
ready -- chat, a [SQL workbench](explore.md), reactive [notebooks](notebooks.md),
[AutoML model training](models.md) with a registry, [dashboards](dashboards.md),
[monitoring](monitoring.md) -- and serves the same derivations over MCP at
`http://localhost:7700/mcp`; point any MCP client at that URL too, and you are
driving the same agent loop you would have in production. Needs the `elbi` app
package (`uv tool install elbi`), and the chat UI needs a model to actually talk to:
add one in Settings the first time you send a message, or skip that step by
exporting a key beforehand, e.g. for the default model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Reach for this to explore, train a model, or demo without a client of your own;
it is not the default way to run a project.

The first run takes ~15-20s longer than later ones while it warms up (loading
the embedding model, initializing the local database); that is expected, not a
hang -- the browser opens the moment it is actually ready, not before. Pass
`--no-browser` to skip opening one (e.g. on a headless machine).

## Validate and test

```bash
elbi validate   # check the project against the spec
uv sync                 # only needed for uv run; serve/mcp use their own environment
uv run pytest           # run your derivation tests
```

## Next

The [SQL workbench](explore.md) and [notebooks](notebooks.md) are ungoverned surfaces
for exploring before you commit to a derivation; see [Serving over MCP](mcp.md) for
what a client sees once you do.
