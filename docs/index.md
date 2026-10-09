# elbi

**Give your agent an empirical memory.**

*Dig deeper. Build on what you find.*

`elbi` is an open-source data-science harness for the agent you already use
(Claude Code, Cursor, Codex, or your own). With elbi, your agent can save
what it learns from your data as a **derivation**, so later sessions and any agent you
connect can dig deeper and build on it. Derivations are Python functions, so you can
read exactly how a finding was computed, write your own, and check, correct, or throw
out any of them. elbi comes batteries-included, with a local data-science environment
for notebooks, SQL, and AutoML.

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

## Where this fits

Frontier agents can write SQL, fit models, and reason about statistics, and many keep
notes or memories between sessions. What they lack is a place to keep what they learn
from data together with the work behind it. elbi is that place: an empirical memory
where each saved finding shows its work, and any agent you connect can build on it.
Alongside it, elbi gives agents tools for the analysis itself, from mapping a dataset
to running statistical checks on a claim and training models. It works with your
semantic layer rather than replacing it: a derivation can take a governed semantic
model, such as one described with [OSI](https://open-semantic-interchange.org/), as an
input. Your agent still does the reasoning.

## It runs on your machine

Everything documented here runs locally with no account and no sign-in. Nothing leaves
your network unless you point it somewhere.

## Next

- [Getting started](getting-started.md): scaffold a project and connect your agent to it
  over the [Model Context Protocol](https://modelcontextprotocol.io) (MCP).
- [Serving over MCP](mcp.md): the tools your agent gets, and how each client connects.
- [Upgrading](upgrading.md): `elbi update`, and the command for your install.
- [Authoring derivations](derivations.md): the SDK in depth.
- [Agent-authored derivations](authoring.md): how an agent records what it found.
- [The CLI](cli.md): `init`, `dev`, `validate`.
- [The Open Derivation Spec (ODS)](spec.md): the open standard.
