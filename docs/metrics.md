# Metrics: a verified semantic layer

A metric is a business number defined once ("revenue", "active users", "conversion
rate") so every dashboard, chat answer, and agent reads the *same* figure. Any semantic
layer gives you that much. Elbi adds the part no other one has: the number is **verified**, not merely consistent. A metric aggregates a
[certified derivation](authoring.md), so what it reports has already passed the oracle.

The vocabulary is the industry one (dbt MetricFlow, [OSI](https://open-semantic-interchange.org/)):
a metric names a **measure** (an aggregation) over a source, the **dimensions** it may be
sliced by, and an optional **time grain**. Because the source is a certified derivation,
already joined and verified, there are no warehouse joins or relationships to configure.

## Defining a metric

A **simple** metric aggregates one column of its source:

```json
{
  "name": "revenue",
  "type": "simple",
  "source": "orders",
  "measure": { "agg": "sum", "column": "amount" },
  "dimensions": ["region", "channel"],
  "timeDimension": { "column": "ordered_at", "grain": "month" }
}
```

`source` must be a **certified** derivation; a metric will not serve numbers from an
unverified one. Aggregations are `sum`, `average`, `count`, `count_distinct`, `min`,
`max`, and `median`. A **ratio** metric divides one metric by another:

```json
{
  "name": "conversion_rate",
  "type": "ratio",
  "numerator": "conversions",
  "denominator": "sessions",
  "dimensions": ["channel"]
}
```

The set is validated against the open
[Metric Spec](https://github.com/Intelligible/elbi/blob/main/spec/metric.schema.json): a metric
can only be sliced by a dimension it declares, so a query can never group by a column the
definition did not sanction.

## Querying a metric

Ask for a metric by the dimensions you want and, for a time dimension, the grain:

```
POST /api/metrics/revenue/query
{ "group_by": ["region"], "grain": "month",
  "filters": [{ "column": "channel", "op": "eq", "value": "web" }] }
```

Resolution compiles the definition to one `GROUP BY` and runs it through the same DuckDB
engine the [SQL workbench](explore.md) uses, over the source derivation's certified rows.
Filter values are bound as parameters, never interpolated, so a query is not an injection
surface. The result is the metric per group, verified by construction, because the rows
it aggregates were.

In the web app, the **Metrics** page lists your metrics (with each source's certification
state), defines new ones, and queries a metric by toggling its dimensions and grain, with
the result shown as a table and a chart.

## Interchange with OSI

Metrics import from and export to the [Open Semantic Interchange](https://open-semantic-interchange.org/)
standard (Apache-2.0, 2026), so a definition moves between elbi and any OSI-aware
tool. Export emits a standard OSI semantic model (datasets with dimension fields, metrics
as aggregation expressions) and also carries the exact elbi definition in a
vendor extension, so a round-trip is lossless while staying readable to other tools:

```
GET  /api/metrics/osi          # export the set as an OSI document
POST /api/metrics/osi          # import an OSI document
```

Importing a foreign OSI model maps each dataset to a source derivation and reads simple
`AGG(column)` metric expressions; a model produced by elbi restores exactly.
Imported documents are validated against the bundled OSI schema, so a malformed model is
rejected rather than half-parsed.

Importing *converts* a document into metrics. To keep it as a governed dependency
instead, bind it as a
[semantic-model input](derivations.md#semantic-model-inputs) to a derivation, so the
document's content versions the results and the model appears in lineage.

## Where this fits

A metric is not a new kind of artifact; it is a thin, governed layer *over* derivations.
That is the deliberate design: elbi does not reimplement a metrics engine (the
`GROUP BY` is trivial when the source is a pre-joined certified table), and it does not
compete with the OSI standard; it speaks it. What it contributes is the one thing a
semantic layer otherwise lacks: every number it serves has been verified.
