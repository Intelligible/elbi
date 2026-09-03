"""Tests for the lineage graph: construction, provenance, and impact.

The cases that matter are the two lineage questions, ``ancestors`` (what feeds a node)
and ``descendants`` (what a change to it would break), plus that the graph is built
correctly from a registry's dataset and derivation inputs, with verdicts attached.
"""

from __future__ import annotations

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    LineageGraph,
    LineageNode,
    Registry,
    SemanticModel,
    derivation,
    lineage_from_registry,
)
from elbi_core.lineage import node_id
from elbi_core.metrics.osi import OSI_VERSION
from elbi_core.registry import use_registry


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")})
        def clean_sales(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("sales").rows)

        @derivation(inputs={"rows": clean_sales})
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("rows").value)

    return registry


def test_from_registry_builds_dataset_and_derivation_edges() -> None:
    graph = lineage_from_registry(_registry(), datasets=["sales"])
    assert graph.node(node_id("dataset", "sales")) is not None
    assert graph.node(node_id("derivation", "revenue")) is not None
    # sales → clean_sales → revenue
    assert graph.downstream(node_id("dataset", "sales")) == [
        node_id("derivation", "clean_sales")
    ]
    assert graph.downstream(node_id("derivation", "clean_sales")) == [
        node_id("derivation", "revenue")
    ]


def test_from_registry_builds_semantic_model_edges() -> None:
    registry = Registry()
    model = SemanticModel.from_osi(
        {
            "version": OSI_VERSION,
            "semantic_model": [
                {
                    "name": "sales_semantics",
                    "datasets": [{"name": "sales", "source": "sales", "fields": []}],
                    "metrics": [
                        {
                            "name": "total_amount",
                            "expression": {
                                "dialects": [
                                    {
                                        "dialect": "ANSI_SQL",
                                        "expression": "SUM(amount)",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    )
    with use_registry(registry):

        @derivation(inputs={"model": model})
        def metric_gates(ctx: Context) -> Artifact:
            return Artifact.table([])

    graph = lineage_from_registry(registry)
    node = graph.node(node_id("semantic_model", "sales_semantics"))
    assert node is not None
    assert node.type == "semantic_model"
    # The model feeds the derivation, so a definition change flows downstream.
    assert graph.ancestors(node_id("derivation", "metric_gates")) == {
        node_id("semantic_model", "sales_semantics")
    }


def test_ancestors_are_full_provenance() -> None:
    graph = lineage_from_registry(_registry())
    assert graph.ancestors(node_id("derivation", "revenue")) == {
        node_id("derivation", "clean_sales"),
        node_id("dataset", "sales"),
    }


def test_descendants_are_the_impact_set() -> None:
    graph = lineage_from_registry(_registry())
    # Changing the source dataset affects both derivations downstream.
    assert graph.descendants(node_id("dataset", "sales")) == {
        node_id("derivation", "clean_sales"),
        node_id("derivation", "revenue"),
    }


def test_verdicts_are_attached() -> None:
    verdicts = {"revenue": "sound", "clean_sales": None}
    graph = lineage_from_registry(_registry(), verdict_of=verdicts.get)
    assert graph.node(node_id("derivation", "revenue")).verdict == "sound"


def test_roots_and_topological_order() -> None:
    graph = lineage_from_registry(_registry(), datasets=["sales"])
    assert graph.roots() == [node_id("dataset", "sales")]
    order = graph.topo_order()
    assert order.index(node_id("dataset", "sales")) < order.index(
        node_id("derivation", "clean_sales")
    )
    assert order.index(node_id("derivation", "clean_sales")) < order.index(
        node_id("derivation", "revenue")
    )


def test_subgraph_focuses_on_one_node_with_its_neighborhood() -> None:
    graph = lineage_from_registry(_registry())
    sub = graph.subgraph(node_id("derivation", "clean_sales"))
    ids = {n["id"] for n in sub["nodes"]}
    # clean_sales plus its ancestor (sales) and descendant (revenue).
    assert ids == {
        node_id("dataset", "sales"),
        node_id("derivation", "clean_sales"),
        node_id("derivation", "revenue"),
    }


def test_manual_graph_adds_cross_type_nodes_and_edges() -> None:
    graph = LineageGraph()
    graph.add_node(
        LineageNode(id="derivation:revenue", type="derivation", name="revenue")
    )
    graph.add_edge("derivation:revenue", "model:forecast", "trains")
    graph.add_edge("derivation:revenue", "dashboard:overview", "displays")
    assert set(graph.downstream("derivation:revenue")) == {
        "model:forecast",
        "dashboard:overview",
    }
    # Endpoints referenced by an edge are auto-created as typed nodes.
    assert graph.node("model:forecast").type == "model"
