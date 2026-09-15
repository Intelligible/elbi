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

**Make your data agent's memory a transparent foundation.**

Claude Code, Cursor, Codex, etc remember things about your data. But they don't let you inspect that memory, correct it, or trust that tomorrow's session is standing on the same understanding as today's. elbi fixes that.

## elbi in 60 seconds

```bash
uv tool install elbi          # install elbi
elbi init acme-data-context   # set up an example project
cd acme-data-context          
elbi mcp                      # serve elbi over mcp
```

That prints `MCP server ready at http://localhost:7878/mcp`. Point your agent (e.g. Claude Code) at that MCP server (exact steps for each are just below), and your agent has access to a transparent memory associated with that project. The agent gets acccess to tools that hold verified answers about the data (called "derivations" in elbi-speak), so the agent doesn't have to re-invent SQL every time it needs an answer.

## Connecting Clients

<details>
<summary><strong>Claude Code</strong></summary>

```bash
claude mcp add --transport http elbi http://localhost:7878/mcp
claude mcp list
```

</details>

<details>
<summary><strong>Cursor</strong></summary>

Add a project file at `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "elbi": {
      "url": "http://localhost:7878/mcp"
    }
  }
}
```

</details>

<details>
<summary><strong>Codex CLI</strong></summary>

```bash
codex mcp add elbi --url http://localhost:7878/mcp
```

</details>

<details>
<summary><strong>Browser</strong></summary>

elbi comes with a fully-fledged GUI and data science environment available, including a [SQL workbench](./docs/explore.md), reactive [notebooks](./docs/notebooks.md), [AutoML model training](./docs/models.md)
with an MLflow registry, [dashboards](./docs/dashboards.md), and [monitoring](./docs/monitoring.md).

Serve it to your browser to start working:

```bash
elbi serve
```

</details>


## Examples

Examples of how to use elbi:

- [`01_getting_started`](examples/01_getting_started): How to get started with elbi. Without elbi: you ask agent about churn risk, it invents a SQL query. With ELBI: you define a Python function (e.g. `churn_risk`), elbi serves it via MCP, and now every time you ask your agent about churn risk, it uses the correct calculation.
- [`02_serving_a_model`](examples/02_serving_a_model): Derivations can hold complex information like trained statML models. Train a model once, serve predictions to an agent, do 'what-if' analyses, without writing any code yourself.
- [`03_stale_skill`](examples/03_stale_skill): Skill files could point agents at trained statML models, but require external management to be kept fresh and organized. In this example of a financial workflow, a cached skill file tells a portfolio optimizer to cut TTD 3%; but the live tool call says to cut 9.2%. Same data streams; the difference is the refresh rate -- elbi gives freshness, while skill files require some external management.
- [`04_certified_models`](examples/04_certified_models): Since derivations are pure Python functions, they can be written by coding agents. elbi has a verification gate that catches bad derivations before they get committed to memory. In this example, an agent proposes a model that (accidentally) trains on a feature that leaks the label; elbi's verification oracle catches it before certification, and the proposal never gets cemented into the agent's memory. The one that does certify becomes durable, searchable memory, not just a table you re-read.
- [`05_components`](examples/05_components): A derivation's finding can be expressed as a natural-language, evidence-backed statement instead of a table -- searchable by meaning across every derivation on the server, not just callable by name.
- [`06_components_vs_rag`](examples/06_components_vs_rag): The same three questions against a real RAG pipeline and against Elbi's components, over one rate schedule. RAG's answer goes stale the moment a source is superseded and its vector store never marks that; Elbi's updates instantly and keeps a citable audit trail. Asked something neither side has data for, RAG guesses confidently anyway -- Elbi declines.

## FAQ
<details>
<summary>
Why not just have my agent write notes to a file or a wiki?
</summary>

You can, and some people do. Here's why elbi is a better fit for many:

1. Convenience: getting an agent to write notes to a file or a wiki means setting up agent-specific settings, a rules file for Cursor, something else for Claude Code. Elbi ships once, and every tool that speaks MCP picks it up without per-agent setup.
2. A note is right the moment you (or an agent) write it, and then it just sits there. Nothing checks whether the data behind the note has since changed, or whether the evidence was ever actually sound, confounded, biased, etc. Elbi checks both freshness and soundness for you.

</details>

## How it works
elbi serves Python functions to AI agents over MCP as versioned, cached tools. 
That function is a *derivation*, and has three key parts: declared inputs, a transform, and a declared output format. `elbi mcp` finds it, runs it against whatever `elbi.dev.yaml` binds, and
publishes. 

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

This example defines a derivation `churn_risk`. `elbi mcp` finds it, runs it against whatever `elbi.dev.yaml` binds, and publishes over MCP. Now, any agent needing to talk about churn risk knows exactly how it should be calculated, and has a fresh copy of the values for the data.


## Installation

```bash
uv tool install elbi
```

Then connect your client (see [#Connecting Clients](#connecting-clients) above), or spin up the elbi GUI with `elbi serve`.

`elbi update` tells you whether a newer release exists and prints the right command for however you installed it. See [Upgrading](docs/upgrading.md).

## What you get

With elbi, you can bind data from local files (CSV, JSON, Parquet) or from a SQL database, and write
derivations over it as plain Python. Everything caches. Any MCP client can read or run
what you publish.

Your agents can also *propose* derivations. Agent-proposed derivations get verified under isolation and only serve once a human certifies it; see [agent-authored derivations](./docs/authoring.md).
You can gate a cleaning step on a machine-checkable
[data contract](./docs/data-contracts.md), which checks types, ranges, keys and
referential integrity before anything is served. For any certified derivation you can
export a signed, offline-verifiable [certificate](./docs/certificates.md) recording
what the oracle checked and who signed off.

elbi runs on your machine. No account, no sign-in, and nothing leaves your network
unless you send it somewhere.

### More surfaces

On top of certified derivations:

- [SQL workbench](./docs/explore.md) and [notebooks](./docs/notebooks.md): ungoverned
  authoring surfaces that promote a query or a cell to a certified derivation.
- [Models](./docs/models.md): AutoML (FLAML, AutoGluon, an Optuna tuner, TabICL, an
  ensemble) with an MLflow registry and serving, driven from the app, code, or an
  agent.
- [Dashboards](./docs/dashboards.md): every tile binds to a certified derivation.
- [Feature store](./docs/feature-store.md): point-in-time-correct training joins, a
  low-latency online store.
- [Metrics](./docs/metrics.md): a verified semantic layer, importing from and
  exporting to [OSI](https://open-semantic-interchange.org/).
- [Monitoring](./docs/monitoring.md): alert when a verified number moves against a
  learned baseline.
- [Lineage and the catalog](./docs/lineage.md), [orchestration](./docs/orchestration.md),
  [result history](./docs/result-history.md), [compute engines](./docs/compute-engines.md)
  for DuckDB/Polars at scale, and the [Open Derivation Spec](./spec/derivation.md).

## Documentation

- [Getting started](./docs/getting-started.md)
- [Serving over MCP](./docs/mcp.md)
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

## The Name
elbi stands for "Empirical Layer by Intelligible". An _empirical layer_ is a place for agents to record insights about data. Intelligible is the company that built elbi.

## Running Elbi for a team

Open-source elbi runs for one person: your machine, your data, an install that needs no
account. `elbi-enterprise` turns that into a team resource: a shared data memory, with single sign-on, users and groups, per-object grants (down to one
derivation), row and column policies, directory sync, and the Kubernetes/Terraform
deployment it runs on. See [Shared team memory](docs/team-memory.md)
for more details.

Interested? Write to solutions@intelligible.ai.

## License

Apache-2.0. Use it for anything, commercial use
included. See [LICENSE](./LICENSE).
