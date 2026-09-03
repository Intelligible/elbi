# elbi

**Author data context as code. Serve it to agents over MCP.**

`elbi` is an open framework for defining **derivations** (versioned,
testable transformations of your data) and serving them to AI agents through the
[Model Context Protocol](https://modelcontextprotocol.io).

You write Python. You run it locally. An agent reads exactly what you serve.

```python
from elbi import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores."""
    rows = ctx.input("sales").rows
    return Artifact.table(
        [{"customer_id": r["customer_id"], "risk": 0.5} for r in rows]
    )
```

## Where this fits

The reliable way for an LLM to use structured data is to call a *governed,
pre-defined computation*, not to write SQL against raw tables every prompt.
elbi is that layer, between raw tables and text. A derivation is *any*
computation over data (a transform, an aggregate, a trained model), versioned and
content-addressed, verified and certified before it serves, and authorizable and
auditable when it does. It complements metric-only semantic layers rather than
replacing them: a derivation can consume a governed semantic model (e.g. one
described with [OSI](https://open-semantic-interchange.org/)) as an input.

## It runs on your machine

Everything documented here runs locally with no account and no sign-in. Nothing leaves
your network unless you point it somewhere.

## Next

- [Getting started](getting-started.md): scaffold a project and serve it.
- [Upgrading](upgrading.md): `elbi update`, and the command for your install.
- [Authoring derivations](derivations.md): the SDK in depth.
- [Agent-authored derivations](authoring.md): propose → verify → certify.
- [The CLI](cli.md): `init`, `dev`, `validate`.
- [The Open Derivation Spec (ODS)](spec.md): the open standard.
