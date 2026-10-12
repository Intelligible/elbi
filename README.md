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

**Give your agent an empirical memory.**

*Dig deeper. Build on what you find.*

elbi is an open-source data-science harness for the agent you already use. With elbi, your agent can save what it learns from your data as a *derivation*, so later sessions and any agent you connect can dig deeper and build on it. Derivations are Python functions, so you can read exactly how a finding was computed, write your own, and check, correct, or throw out any of them. elbi comes batteries-included, with a local data-science environment for notebooks, SQL, and AutoML.

## elbi in 60 seconds

It runs on your machine. There's no account to create.

```bash
uv tool install elbi          # install elbi
elbi init acme-data-context   # set up an example project
cd acme-data-context
elbi mcp                      # serve elbi over MCP
```

That prints `MCP server ready at http://localhost:7878/mcp`. Point your agent at it (steps for each client are just below) and ask who has the highest churn risk: your agent should list elbi's tools and answer from the example's `run_churn_risk` rather than inventing SQL. The example data is three rows, so it's a smoke test. Bind your own data ([Data sources](./docs/data-sources.md)) to see the agent map a dataset's structure and check its claims.

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

`elbi serve` opens a full local data-science environment in your browser: chat with elbi's own built-in agent (bring a model key) if you'd rather not use yours, a [SQL workbench](./docs/explore.md), reactive [notebooks](./docs/notebooks.md), and [AutoML model training](./docs/models.md) with an MLflow registry, alongside a view of everything your agent has recorded. The app serves the same MCP endpoint at `http://localhost:7700/mcp`, so your own agent and the browser work on the same project.

```bash
elbi serve
```

</details>

## What your agent gets

**A look before it leaps.** Before analyzing, your agent can see how a dataset fits together: which columns identify a row, which are derived from others, which carry overlapping information and could confound a comparison, and how complete each one is.

**Statistical checks it can run.** More than 30 `verify_*` tools let your agent test a claim with a method suited to it: regression, causal designs (instrumental variables, difference-in-differences, regression discontinuity), survival analysis, calibration, forecasting, and [more](./docs/mcp.md#what-gets-exposed). Most return an estimate, its caveats, and a verdict of *sound*, *unsound*, or *inconclusive* with the reason, so your agent can see when a confounder, a subgroup reversal, or a leaked feature undoes a pattern.

**A memory that shows its work.** When a finding is worth keeping, your agent records it as a *derivation*: the code, the data it ran on, and the result. Any agent you connect can find the derivation and call it again. The result is tied to the data it ran on, and the next run of elbi after that data changes recomputes it. If the agent states a claim (an effect of `x` on `y`, or a model's target and features), elbi checks the claim before recording the finding. You can revise or delete any derivation, and [require your own sign-off](./docs/authoring.md#certification-policy) if you want a stricter gate.

In SleuthBench ([arXiv:2609.34228](https://arxiv.org/abs/2609.34228)), giving six frontier models precomputed statistical artifacts alongside their Python tool raised their accuracy on questions about how features shape an outcome from 41.9% to 68.0%.

## Examples

- [`01_getting_started`](examples/01_getting_started): a complete project like the one `elbi init` scaffolds, where `pricing_effects` builds on `churn_risk`, so the agent reuses a churn calculation instead of reinventing it in SQL each time.
- [`02_serving_a_model`](examples/02_serving_a_model): give elbi a trained model and let the agent query its predictions, for one customer or a batch, without writing scoring code.
- [`03_stale_skill`](examples/03_stale_skill): a skill file written after yesterday's close tells the agent to cut a position 3%. Overnight the stock fell 21.9%, and the same question asked as an elbi tool, recomputed from this morning's prices, cuts more than twice as much. The file can't tell that its data moved.

## FAQ

<details>
<summary><strong>Why not just keep notes, or use my agent's memory?</strong></summary>

Keep using them for what you've told your agent: conventions, preferences, context. For what it found out from data, elbi is the memory built for the job. A derivation shows its work (its code, inputs, result, and lineage), so you can see what a finding rests on, rerun it, re-check its claim with `elbi test`, and see which version of the data it ran on. And elbi keeps it in one place that every MCP client can use, instead of in one agent's notes or rules file, so a finding Claude Code recorded is there for Cursor too.

The two work together rather than compete: elbi is the source of record for what your agent learned from data, and memory is one more place that knowledge can reach. Any MCP client can already read a derivation's current result as a resource.

</details>

<details>
<summary><strong>Does elbi replace my semantic layer?</strong></summary>

No, it builds on one. A semantic layer gives each business number one agreed definition, and a derivation can take that governed model as an input, including one described with [OSI](https://open-semantic-interchange.org/) (Open Semantic Interchange), so your agent's analysis starts from the definitions your team already owns. elbi also ships a semantic layer of its own ([Metrics](./docs/metrics.md)) that imports from and exports to OSI. What elbi adds is the work that goes beyond shared definitions: analyses and models your agent saves with the code, data, and result behind them, for any agent to build on.

</details>

## How it works

A derivation is a Python function with declared inputs. Your agent can write one through `propose_derivation`, or you can write it yourself:

```python
from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"customers": Dataset("customers")},
    serve=serve.table(title="Churn rate by plan"),
)
def churn_by_plan(ctx: Context) -> Artifact:
    """Share of customers who churned on each plan, with the count behind it."""
    counts: dict[str, list[int]] = {}
    for r in ctx.input("customers").rows:
        tally = counts.setdefault(r["plan"], [0, 0])
        tally[0] += int(r["churned"])
        tally[1] += 1
    return Artifact.table(
        [
            {"plan": plan, "customers": n, "churn_rate": round(churned / n, 3)}
            for plan, (churned, n) in sorted(counts.items())
        ]
    )
```

elbi runs it against whatever `elbi.dev.yaml` binds, records the result with its inputs and lineage, and exposes it to the agent as `run_churn_by_plan`. The result is cached against the code and the `customers` data: a running server keeps serving it, and the next run after either changes recomputes it.

**Serving your own computations.** You don't have to wait for an agent to discover something. Write derivations yourself (a metric, a model, a piece of business logic) and elbi serves each one to every MCP client as a cached, versioned tool, so every agent and tool gets the same answer. See [Authoring derivations](./docs/derivations.md) and [Analytics as code](./docs/analytics-as-code.md).

Data comes from local files (CSV, JSON, Parquet) or a SQL database ([Data sources](./docs/data-sources.md)). See [agent-authored derivations](./docs/authoring.md) for how an agent proposes one.

## Installation

```bash
uv tool install elbi
```

Then connect your client (see [Connecting Clients](#connecting-clients) above). `elbi update` tells you whether a newer release exists and prints the right command for however you installed it. See [Upgrading](docs/upgrading.md).

## Documentation

- [Getting started](./docs/getting-started.md)
- [Serving over MCP](./docs/mcp.md)
- [Authoring derivations](./docs/derivations.md)
- [Agent-authored derivations](./docs/authoring.md)
- [Data sources](./docs/data-sources.md)
- [Caching](./docs/caching.md)
- [Result history](./docs/result-history.md)
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

elbi stands for "Empirical Layer by Intelligible." An _empirical layer_ is where an agent keeps what it has learned from data. Intelligible is the company that builds elbi.

## Running elbi across a team

elbi is built for one person working with data already in reach: your machine, your files, no account. If you need that learning shared across people and teams, connected to your organization's data systems and kept current, that's what [Summand](https://summand.com), Intelligible's enterprise data platform, is for. Write to solutions@intelligible.ai.

## License

Apache-2.0. Use it for anything, commercial use
included. See [LICENSE](./LICENSE).
