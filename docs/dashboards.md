# Dashboards

A dashboard is a presentation surface over your derivations. It holds no data logic
of its own: each widget *binds* to a certified derivation by name, supplies
parameters, and declares how that derivation's already-verified, already-cached result
is rendered. The derivation is the semantic layer; the dashboard is the layout.

That separation is what you trust. A published dashboard can only
bind derivations the oracle has certified, so every number a viewer sees has been
verified, not merely rendered from whatever SQL happened to sit behind a tile.

## The shape of a dashboard

A dashboard is a declarative [`DashboardSpec`](spec.md), validated against a JSON
Schema the same way a derivation manifest is. It has variables (the filter controls),
pages, and, on each page, a grid of widgets:

```json
{
  "specVersion": "1.0",
  "kind": "Dashboard",
  "name": "revenue_health",
  "title": "Revenue Health",
  "variables": [
    {
      "name": "region",
      "type": "string",
      "control": "multiselect",
      "default": ["west"],
      "options": { "derivation": "regions", "column": "region" }
    }
  ],
  "pages": [
    {
      "name": "overview",
      "widgets": [
        {
          "id": "mrr",
          "type": "metric",
          "gridPos": { "x": 0, "y": 0, "w": 6, "h": 4 },
          "bind": { "derivation": "monthly_revenue", "params": { "region": "$region" } },
          "viz": { "field": "mrr", "format": "currency" }
        },
        {
          "id": "trend",
          "type": "chart",
          "gridPos": { "x": 6, "y": 0, "w": 18, "h": 8 },
          "bind": { "derivation": "revenue_trend", "params": { "region": "$region" } },
          "viz": { "mark": "line", "encoding": { "x": {"field": "month"}, "y": {"field": "revenue"} } },
          "interactions": { "crossFilter": { "emit": { "month": "datum.month" } } }
        }
      ]
    }
  ]
}
```

## Binding widgets to derivations

Every data-bound widget names a derivation and passes it parameters:

```json
"bind": { "derivation": "monthly_revenue", "params": { "region": "$region" } }
```

A parameter value is a literal, or the string `"$name"`, which resolves to the current
value of dashboard variable `name` (a leading `$$` escapes a literal dollar sign). This
is the *only* way a widget consumes a variable; wiring is explicit, so a spec is as
reviewable as code, and there is no hidden magic matching a filter to a column.

Because a derivation is already cached and content-addressed, loading a page is a set
of cheap cached reads, and changing a filter re-runs only the widgets whose parameters
moved.

## Widget types

| Type | Renders |
| --- | --- |
| `metric` | A KPI/stat card reading one field from the bound derivation. |
| `chart` | A [grammar-of-graphics spec](result-history.md) over the derivation's rows. |
| `map` | A geographic layer over a spatially-gated derivation. |
| `table` | The derivation's rows, with paging and conditional formatting. |
| `text` | Markdown: either its own `content` (with `$name` variable interpolation) or the output of a `bind`ed derivation that returns markdown. The one freeform surface. |
| `filter` | A control bound to a variable. |

There is deliberately no arbitrary-code widget: a bespoke visual belongs in a
derivation that returns markdown, bound to a `text` widget, so it stays governed and
sandboxed like everything else:

```json
{ "id": "verdict", "type": "text", "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
  "bind": { "derivation": "cost_fixed_share" } }
```

A text widget takes `content` or `bind`, never both — otherwise which one renders would
be decided somewhere other than the spec.

## Filters, variables, and interactions

- **Variables** are the dashboard's inputs. A `filter` widget renders a control for one;
  its options are a static list or the distinct values of a derivation's column.
- **Scope** confines a variable to `global` (all pages) or a single page.
- **Cross-filter**: `interactions.crossFilter.emit` sets variables from a clicked
  datum, re-resolving every non-immune widget that references them.
- **Drill-down**: `interactions.drillDown` steps through a hierarchy of grouping columns
  by rebinding a parameter, within the same widget.
- **Drill-through**: `interactions.drillThrough` navigates to a detail page or another
  dashboard, carrying the current variable state.

## Lifecycle: draft and published

A dashboard starts as a **draft**: edits auto-save and are invisible to viewers. When
you **publish**, the current draft is snapshotted as the version viewers see. Publishing
runs the *certification gate*: it refuses if any bound derivation is missing or not yet
certified, listing the offenders. This is the guarantee: a published dashboard cannot
show a number the oracle has not verified.

Every save appends a revision, so a dashboard has the same diffable, roll-back-able
history as code.

## What a spec is not

The spec defines *what* renders, and nothing about it decides what the data may say. A
widget's numbers come from running the derivation it binds, so a dashboard can only show
what that derivation returns, so publishing is gated on every bound derivation
being certified.

## Scheduled delivery

A dashboard can carry subscriptions: a page is snapshotted on a cron cadence under a
saved filter state and delivered as an image, PDF, or CSV over email or a webhook. The
same scheduler that reruns notebooks fires them.

## Authoring from the agent

Because a dashboard is a validated declarative spec, the chat agent can compose one the
way it authors a derivation: it builds any missing derivations through the usual
propose → verify → certify loop, then lays them out as a `DashboardSpec`. An
agent-built dashboard is trustworthy by construction, because it can only publish over
certified building blocks.

See the [Dashboard Spec](spec.md) for the normative schema.
