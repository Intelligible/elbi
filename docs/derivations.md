# Authoring derivations

A **derivation** pairs a compute function with the declarative metadata that makes
it servable and verifiable: its inputs, its dependencies, and its serve contract.

## The decorator

```python
from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", columns=["customer_id", "risk"], max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores derived from recent sales activity."""
    rows = ctx.input("sales").rows
    scored = [{"customer_id": r["customer_id"], "risk": _score(r)} for r in rows]
    return Artifact.table(scored)
```

- `inputs` maps a name to a `Dataset` or to another
  derivation. The compute function reads each with `ctx.input(name)`.
- `serve` declares how an agent sees the result. It is optional: a derivation
  without one is *internal* (usable as an input to other derivations, not exposed
  to agents over MCP).
- The docstring's first line becomes the derivation's description.

The derivation's manifest is validated against the spec **at definition time**,
so an invalid derivation fails on import, not when an agent calls it.

## Inputs

A dataset input resolves to a `Table`; `.rows` is a
list of dicts. A derivation input resolves to the upstream
`Artifact`; read `.value`. A semantic-model input resolves to the
`SemanticModel` itself; read `.metrics`.

```python
@derivation(
    inputs={"sales": Dataset("sales"), "risk": churn_risk},
    serve=serve.markdown(title="Pricing effects"),
)
def pricing_effects(ctx: Context) -> Artifact:
    sales = ctx.input("sales").rows
    risk = ctx.input("risk").value  # the churn_risk artifact's value
    ...
```

Dependencies on derivation inputs are inferred; `pricing_effects` automatically
`dependsOn` `churn_risk`.

### Semantic-model inputs

A derivation can consume a **governed semantic model**: metric definitions someone
else owns, so your gates run on top of them. Bind an
[Open Semantic Interchange](https://open-semantic-interchange.org/) document with
`SemanticModel.from_osi`:

```python
_MODEL = SemanticModel.from_osi(
    json.loads(Path("sales_semantics.osi.json").read_text())
)


@derivation(
    inputs={"model": _MODEL, "sales": Dataset("sales")},
    serve=serve.table(title="Metric gates"),
)
def metric_gates(ctx: Context) -> Artifact:
    model = ctx.input("model")
    available = set(ctx.input("sales").rows[0])
    ...  # check every governed metric is computable over the rows we hold
```

Three things follow from binding it as an input rather than reading a file:

- The document is **validated against the OSI schema** when the model is constructed,
  so an unusable document fails on import, not when an agent calls the tool.
- The document's **content versions the derivation**. Editing a definition changes the
  data version, so cached results and certificates covering the old definitions are
  invalidated.
- A semantic model is static content, so it implies **no `dependsOn` edge**. It still
  appears in [lineage](lineage.md) as an upstream node, so the blast radius of a
  definition change stays visible.

See [Metrics](metrics.md#interchange-with-osi) for importing and exporting OSI
documents as metric sets.

## Parameters

Parameters are values an agent supplies per call; served over MCP they become the
tool's typed input schema. Read them with `ctx.param(name)`.

```python
from elbi_core import param


@derivation(
    inputs={"houses": Dataset("kc_house")},
    params={
        "zipcode": param.string(description="Zip code to filter to"),
        "max_price": param.integer(required=False, default=0),
    },
    serve=serve.table(),
)
def houses_in_zip(ctx: Context) -> Artifact:
    rows = ctx.input("houses").rows
    return Artifact.table([r for r in rows if r["zip"] == ctx.param("zipcode")])
```

Scalars are `param.string`, `param.integer`, `param.number`, and `param.boolean`.
For structured input, `param.object` takes a record of named values and
`param.array(items=...)` a list; an array of `object` is a batch of records. Keep
these shallow: agents fill flat arguments more reliably than nested ones.

## Serve contracts

| Builder | Output to the agent |
| --- | --- |
| `serve.table(columns=..., max_rows=...)` | a Markdown table, projected and capped |
| `serve.markdown()` | a Markdown document |
| `serve.json(indent=...)` | pretty-printed JSON |
| `serve.text()` | plain text |

## Artifacts

Return an `Artifact` (`Artifact.table`, `.markdown`, `.json`, `.text`) or a raw
value, which is coerced: a list of dicts becomes a table, a string becomes text,
anything else becomes JSON.

`Artifact.opaque(value)` carries an arbitrary Python object (a trained model, a
fitted index) for a downstream derivation to consume. It has no serve rendering,
so a derivation returning one should be internal.

## Serving a model

A derivation is any computation, so "train a model and serve its predictions" is
two derivations. A training derivation returns the fitted model as an opaque
artifact; because the model is an *input* to the predictor, it is trained once and
cached, not refit on every call. The predictor takes the features as a parameter
and serves the prediction like any other result.

```python
@derivation(inputs={"customers": Dataset("customers")})  # internal: no serve
def churn_model(ctx: Context) -> Artifact:
    model = _fit(ctx.input("customers").rows)  # scikit-learn, PyTorch, or plain Python
    return Artifact.opaque(model)


@derivation(
    inputs={"model": churn_model},
    params={"customer": param.object(description="One feature record")},
    serve=serve.json(title="Churn prediction"),
)
def predict_churn(ctx: Context) -> Artifact:
    model = ctx.input("model").value
    return Artifact.json({"churn_probability": _predict(model, ctx.param("customer"))})
```

The full project is in
[`examples/02_serving_a_model`](https://github.com/Intelligible/elbi/tree/main/examples/02_serving_a_model).

## Testing

Derivations are plain functions behind a registry. Drive them with a `Runner`:

```python
from elbi_core import Registry, Runner
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

registry = Registry()
discover(Path("derivations"), registry=registry)
runner = Runner(registry, bindings=DataBindings.load(Path("elbi.dev.yaml")))
assert runner.run("churn_risk").value
```
