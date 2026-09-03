# elbi-core

The SDK for building **derivations**: versioned, testable functions that transform
your data and serve the result to AI agents over the
[Model Context Protocol](https://modelcontextprotocol.io).

```bash
pip install elbi-core          # the SDK on its own
pip install "elbi-core[cli]"   # and the `elbi` command line
pip install "elbi-core[data]"  # and columnar formats (parquet)
```

For the chat app and web UI as well, install `elbi` instead.

```python
from elbi import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores."""
    rows = ctx.input("sales").rows
    scored = [
        {"customer_id": r["customer_id"], "risk": 1 / (1 + float(r["amount"]))}
        for r in rows
    ]
    return Artifact.table(scored)
```

See the [repository README](https://github.com/Intelligible/elbi) and
[the Open Derivation Spec](https://github.com/Intelligible/elbi/tree/main/spec)
for more.

Licensed under [Apache-2.0](https://github.com/Intelligible/elbi/blob/main/LICENSE).
