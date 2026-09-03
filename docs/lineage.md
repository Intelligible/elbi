# Lineage & catalog

Every artifact on the platform depends on others: a derivation reads datasets and
upstream derivations; a model trains on a feature derivation; a dashboard tile and a
feature view bind to a derivation. Lineage makes that graph explicit, and the catalog
makes it searchable, with each node carrying its oracle verdict, so you see not only
where a number comes from but whether each step was certified sound.

## The graph

Nodes are typed artifacts (`dataset`, `derivation`, `model`, `dashboard`,
`feature_view`, `entity`) identified as `type:name` (e.g. `derivation:revenue`). Edges
run upstream → downstream. The graph is stitched from the references each artifact
already records, so it needs no separate instrumentation:

- `dataset → derivation` and `derivation → derivation` from a derivation's inputs.
- `derivation → model` from the model's feature-derivation lineage.
- `derivation → dashboard` from the dashboard's bound derivations.
- `derivation → feature_view` (and `entity → feature_view`) from the feature view.

This mirrors the [OpenLineage](https://openlineage.io/) model, in which a derivation is a job
that produces a dataset, with the oracle verdict as a data-quality facet on each node.

## The two questions

- **Provenance** ("where does this come from?"): a node's `ancestors`, every dataset
  and derivation that feeds it. Use it to trace a number back to its sources and to see
  the verdict at each step.
- **Impact** ("what breaks if I change this?"): a node's `descendants`, every
  derivation, model, dashboard, and feature view downstream. Run it *before* changing or
  deleting a dataset or derivation, so a schema change never silently breaks a dashboard.

## The catalog

The catalog lists every artifact as a searchable record (name, type, description,
oracle verdict, and how many inputs it has) across all types at once, which the
per-type pages can't give you. Search is a plain substring match over name, type, and
description.

## Asking more than one question

Building the graph is the expensive part: it reads the project registry, the model
registry, and six tables, plus one read per dashboard. So `LineageService` separates
building from querying. Each of its methods builds a graph and asks it one question,
which suits a single endpoint. A caller with several questions should take a
snapshot instead, so they cost one build between them and all describe the same graph:

```python
view = lineage_service.snapshot()
node = view.resolve("revenue")
view.provenance(node)
view.impact(node)
view.catalog()
```

## In the app and to the agent

The app exposes `/api/catalog` (search) and `/api/lineage/graph`,
`/api/lineage/subgraph`, `/api/lineage/impact`. The **Catalog & lineage** page renders
the graph interactively (React Flow), colored by verdict, with an impact summary for
the selected node.

The chat agent has the same surface as three tools, `search_catalog` (find what
exists before authoring something new), `trace_lineage` (provenance and dependents), and
`impact_analysis` (what a change would affect), so it can reason about the graph the
way a human reads the catalog.
