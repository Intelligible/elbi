<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo-dark.svg">
    <img alt="elbi" src="./docs/assets/logo.svg" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Intelligible/elbi/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Intelligible/elbi/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/elbi/"><img alt="PyPI" src="https://img.shields.io/pypi/v/elbi.svg"></a>
  <a href="https://pypi.org/project/elbi/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/elbi.svg"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

Write a Python function. `elbi` serves it to AI agents over the
[Model Context Protocol](https://modelcontextprotocol.io) as a versioned, cached tool.

That function is a *derivation*: declared inputs, a transform, a declared output format.

```python
from elbi import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores from recent sales activity."""
    rows = ctx.input("sales").rows
    scored = [
        {"customer_id": r["customer_id"], "risk": 1 / (1 + float(r["amount"]))}
        for r in rows
    ]
    scored.sort(key=lambda r: r["risk"], reverse=True)
    return Artifact.table(scored)
```

`elbi serve` finds the function, runs it against whatever `elbi.dev.yaml` binds, and
publishes it. An agent calls `run_churn_risk` and reads back:

```
# Churn risk

| customer_id | risk |
| --- | --- |
| C-1021 | 0.5 |
| C-0820 | 0.2 |
| C-0640 | 0.1 |
```

That result is cached. It recomputes when the code changes or the `sales` data does,
and not otherwise.

## Where this fits

An agent that writes fresh SQL against raw tables on every prompt gets it wrong, and
you find out downstream. Define the computation once instead, in Python, and hand
agents the result as a governed tool.

A derivation is any computation over data: a transform, an aggregate, a trained model.
That reaches questions a metrics-only semantic layer can't phrase, because it isn't
limited to metrics. It doesn't replace a semantic layer, though. A derivation can take
one as an input, including a governed model described with
[OSI](https://open-semantic-interchange.org/).

## Installation

```bash
uv tool install elbi
```

Later, `elbi update` tells you whether a newer release exists and prints the right
command for however you installed it. It touches the network only when you run it;
nothing checks at startup or on a timer. See [Upgrading](docs/upgrading.md).

## Quick start

```bash
elbi init acme-data-context
cd acme-data-context
elbi serve
```

You don't need an API key to get going. `elbi serve` opens a chat UI at
`http://localhost:7700` once it's ready, and serves the same derivations over MCP at
`http://localhost:7700/mcp`. The chat needs a model to talk to, so add one in Settings
when you send your first message, or export a key beforehand:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The first run takes 15-20 seconds longer than the ones after it. It's loading the
embedding model and setting up the local database, and the browser opens when the app
is genuinely ready rather than before. (`uv sync` comes later, for `uv run pytest`.
`serve`, `validate` and `test` all run from their own installed environment.)

Want the MCP endpoint on its own? Install `elbi-cli` instead and run `elbi mcp`. Same
derivations at `http://localhost:7878/mcp`, no chat UI, none of the heavier
dependencies, and no model key to configure. Whichever client you point at it brings
its own.

## Connect a client

Point any MCP client at the URL that gets printed, from either `serve` or `mcp`. To
poke at it without one:

```bash
npx @modelcontextprotocol/inspector
```

Connect, then call the `run_<name>` tools. Claude Desktop, Cursor and your own agent
all take the same URL as an MCP server.

## What you get

Bind data from local files (CSV, JSON, Parquet) or from a SQL database, and write
derivations over it as plain Python. Everything caches. Any MCP client can read or run
what you publish.

An agent can also *propose* a derivation. It gets verified under isolation and only
serves once a human certifies it; see [agent-authored derivations](./docs/authoring.md).
You can gate a cleaning step on a machine-checkable
[data contract](./docs/data-contracts.md), which checks types, ranges, keys and
referential integrity before anything is served. When the data outgrows memory, push
profiling and contract checks down to [DuckDB or Polars](./docs/compute-engines.md).

Every derivation keeps its [result history](./docs/result-history.md): how the certified
answer changed, and what moved it. Each version can go to MLflow. For any certified
derivation you can export a signed, offline-verifiable
[certificate](./docs/certificates.md) recording what the oracle checked and who signed
off.

There are two places to explore before you commit to anything. The
[SQL workbench](./docs/explore.md) is deliberately ungoverned, and promotes a query to a
certified derivation in one click. The [notebook](./docs/notebooks.md) is reactive, and
promotes a cell the same way. Both are authoring surfaces; the derivation is the
artifact.

On top of certified derivations you can build:

- [dashboards](./docs/dashboards.md), where every tile binds to one, so a published
  dashboard can only show numbers the oracle verified;
- a [feature store](./docs/feature-store.md) with point-in-time-correct training joins
  and a low-latency online store, so there's no train/serve skew;
- [metrics](./docs/metrics.md) defined once, a verified semantic layer that imports from
  and exports to [OSI](https://open-semantic-interchange.org/), so every tool reads the
  same sound number.

[Monitor](./docs/monitoring.md) any metric or derivation against a learned baseline and
get alerted when a verified number moves, with the oracle's verdict attached.
[Lineage and the catalog](./docs/lineage.md) cover provenance, impact analysis and
search across every artifact. [Orchestration](./docs/orchestration.md) materializes
assets that have gone stale, on a cron schedule or a data-change sensor, and keeps
per-asset run history. And you can validate any derivation against the
[Open Derivation Spec](./spec/derivation.md).

Elbi runs on your machine. No account, no sign-in, and nothing leaves your network
unless you send it somewhere.

## Documentation

- [Getting started](./docs/getting-started.md)
- [Authoring derivations](./docs/derivations.md)
- [Data contracts](./docs/data-contracts.md)
- [Compute engines](./docs/compute-engines.md)
- [Result history](./docs/result-history.md)
- [Data sources](./docs/data-sources.md)
- [Caching](./docs/caching.md)
- [Open Derivation Spec](./spec/derivation.md)

## Community

- [GitHub Discussions](https://github.com/Intelligible/elbi/discussions) for questions, ideas, and show-and-tell.
- [Issues](https://github.com/Intelligible/elbi/issues) for bug reports and feature requests.

## Contributing

We'd be glad of the help. [CONTRIBUTING](./.github/CONTRIBUTING.md) covers the
development setup, the test commands and our PR conventions; there's also a
[Code of Conduct](./.github/CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities privately; the [Security Policy](./.github/SECURITY.md) says how.
Please don't open a public issue for one.

## Running Elbi for a team

We built Elbi for one person: your machine, your data, an install that needs no account.
If you need it to serve several people, with single sign-on, user and group management
and per-object permissions, write to sales@intelligible.ai.

## License

Apache-2.0, all of it: the specification, the SDK, the CLI, the agent runtime, the MCP
server, the web app, every verification gate. Use it for anything, commercial use
included. See [LICENSE](./LICENSE).
