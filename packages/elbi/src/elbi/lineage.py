"""The lineage service: the full cross-artifact graph, impact analysis, and catalog.

Stitches the core dataset→derivation graph together with the downstream artifacts the
app stores: models (via the ``elbi.feature_derivation`` tag), dashboards (via their
bound derivations), feature views (via their source derivation and entities), metrics
(via the certified derivation they aggregate), and monitors (via the metric or
derivation they watch). Every node carries its oracle verdict (a metric inherits its
source derivation's, a monitor inherits its target's), so an impact query reaches the
metrics and monitors a derivation feeds, and each carries the verified status of the
number behind it. On top of the graph it answers impact analysis (what a change to a
node would affect) and a unified, searchable catalog.

Building the graph is expensive -- the registry, the model registry, and six tables,
plus one read per dashboard -- so construction and querying are separate.
:class:`LineageService` builds, :class:`LineageView` queries. A caller with several
questions takes one :meth:`LineageService.snapshot` and asks it all of them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from elbi_core import DashboardSpec, LineageGraph, LineageNode, Registry
from elbi_core.errors import ElbiError
from elbi_core.lineage import from_registry, node_id

from .db import Store

#: One model's lineage: its name, oracle verdict, and source feature derivation.
ModelEdge = dict[str, Any]


@dataclass(frozen=True)
class LineageView:
    """One built lineage graph, queried as many times as the caller likes.

    Holds no store and no registry, so every answer describes the graph as it stood when
    the view was taken.
    """

    _graph: LineageGraph

    def graph(self) -> dict[str, Any]:
        """The whole lineage graph as ``{nodes, edges}``."""
        return self._graph.to_dict()

    def subgraph(self, node_id_: str) -> dict[str, Any]:
        """A node's neighborhood (its ancestors and descendants)."""
        return self._graph.subgraph(node_id_)

    def impact(self, node_id_: str) -> dict[str, Any]:
        """What a change to a node would affect, grouped by artifact type."""
        return self._reachable(node_id_, self._graph.descendants, "affected")

    def provenance(self, node_id_: str) -> dict[str, Any]:
        """What feeds a node, grouped by artifact type (its upstream lineage)."""
        return self._reachable(node_id_, self._graph.ancestors, "feeds_from")

    def resolve(self, reference: str) -> str | None:
        """Turn a bare name or ``type:name`` reference into a node id, if it exists."""
        if self._graph.node(reference) is not None:
            return reference
        matches = [n.id for n in self._graph.nodes() if n.name == reference]
        return matches[0] if len(matches) == 1 else None

    def catalog(self, query: str = "") -> list[dict[str, Any]]:
        """Every artifact as a searchable catalog record, newest node types first.

        A non-empty ``query`` filters by case-insensitive substring over name, type,
        and description.
        """
        needle = query.strip().lower()
        records = []
        for node in self._graph.nodes():
            haystack = f"{node.name} {node.type} {node.description or ''}".lower()
            if needle and needle not in haystack:
                continue
            record = node.to_dict()
            record["upstream"] = len(self._graph.upstream(node.id))
            records.append(record)
        return records

    def _reachable(
        self, node_id_: str, walk: Callable[[str], set[str]], key: str
    ) -> dict[str, Any]:
        """Everything ``walk`` reaches from a node, named and grouped by type."""
        if self._graph.node(node_id_) is None:
            raise LineageError(f"unknown node {node_id_!r}")
        by_type: dict[str, list[str]] = {}
        for reached in walk(node_id_):
            node = self._graph.node(reached)
            if node is not None:
                by_type.setdefault(node.type, []).append(node.name)
        return {
            "node": node_id_,
            key: {t: sorted(names) for t, names in by_type.items()},
            "count": sum(len(names) for names in by_type.values()),
        }


class LineageService:
    """Builds and queries the cross-artifact lineage graph for a project."""

    def __init__(
        self,
        store: Store,
        registry_provider: Callable[[], Registry],
        dataset_names: Callable[[], Sequence[str]] = tuple,
        models_provider: Callable[[], list[ModelEdge]] | None = None,
    ) -> None:
        self._store = store
        self._registry_provider = registry_provider
        # A callable, like ``registry_provider`` beside it: the graph is rebuilt per
        # call, so a table synced during this session belongs in it without a restart.
        self._dataset_names = dataset_names
        self._models_provider = models_provider

    # -- queries -----------------------------------------------------------------
    def snapshot(self) -> LineageView:
        """One built graph. Every method below is this call plus one question."""
        return LineageView(self._graph())

    def graph(self) -> dict[str, Any]:
        """The whole lineage graph as ``{nodes, edges}``."""
        return self.snapshot().graph()

    def subgraph(self, node_id_: str) -> dict[str, Any]:
        """A node's neighborhood (its ancestors and descendants)."""
        return self.snapshot().subgraph(node_id_)

    def impact(self, node_id_: str) -> dict[str, Any]:
        """What a change to a node would affect, grouped by artifact type."""
        return self.snapshot().impact(node_id_)

    def provenance(self, node_id_: str) -> dict[str, Any]:
        """What feeds a node, grouped by artifact type (its upstream lineage)."""
        return self.snapshot().provenance(node_id_)

    def resolve(self, reference: str) -> str | None:
        """Turn a bare name or ``type:name`` reference into a node id, if it exists."""
        return self.snapshot().resolve(reference)

    def catalog(self, query: str = "") -> list[dict[str, Any]]:
        """Every artifact as a searchable catalog record, newest node types first.

        A non-empty ``query`` filters by case-insensitive substring over name, type,
        and description.
        """
        return self.snapshot().catalog(query)

    # -- construction ------------------------------------------------------------
    def _graph(self) -> LineageGraph:
        registry = self._registry_provider()
        verdicts = self._derivation_verdicts()
        graph = from_registry(registry, self._dataset_names(), verdict_of=verdicts.get)
        self._add_dashboards(graph)
        self._add_feature_views(graph)
        self._add_training_sets(graph)
        self._add_models(graph)
        metric_sources = self._add_metrics(graph, verdicts)
        self._add_monitors(graph, verdicts, metric_sources)
        return graph

    def _derivation_verdicts(self) -> dict[str, str | None]:
        return {row.name: row.verdict for row in self._store.list_derivations()}

    def _add_models(self, graph: LineageGraph) -> None:
        if self._models_provider is None:
            return
        for model in self._models_provider():
            name = str(model["name"])
            graph.add_node(
                LineageNode(
                    id=node_id("model", name),
                    type="model",
                    name=name,
                    verdict=model.get("verdict"),
                )
            )
            upstream = _model_source_node(model)
            if upstream is not None:
                graph.add_edge(upstream, node_id("model", name), "trains")

    def _add_training_sets(self, graph: LineageGraph) -> None:
        """Add each training set downstream of the feature views it joins.

        A training set freezes a point-in-time join over ``view:feature`` references,
        so it depends on each distinct feature view named in that join. Wiring the
        edge lets an impact query on a feature view reach the models trained on its
        sets.
        """
        for row in self._store.list_training_sets():
            ts_id = node_id("training_set", row.name)
            graph.add_node(LineageNode(id=ts_id, type="training_set", name=row.name))
            views = {
                str(ref).split(":", 1)[0]
                for ref in json.loads(row.features_json)
                if ":" in str(ref)
            }
            for view in sorted(views):
                graph.add_edge(node_id("feature_view", view), ts_id, "joins")

    def _add_dashboards(self, graph: LineageGraph) -> None:
        for row in self._store.list_dashboards():
            dashboard_id = node_id("dashboard", row["name"])
            graph.add_node(
                LineageNode(
                    id=dashboard_id,
                    type="dashboard",
                    name=row["name"],
                    description=row.get("title"),
                    copied_from=row.get("copied_from"),
                )
            )
            full = self._store.get_dashboard(row["id"])
            if full is None:
                continue
            try:
                spec = DashboardSpec.from_manifest(json.loads(full.spec_json))
            except (ElbiError, json.JSONDecodeError):
                continue
            for derivation_name in spec.derivation_names():
                graph.add_edge(
                    node_id("derivation", derivation_name), dashboard_id, "displays"
                )

    def _add_feature_views(self, graph: LineageGraph) -> None:
        for row in self._store.list_feature_views():
            view_id = node_id("feature_view", row.name)
            graph.add_node(LineageNode(id=view_id, type="feature_view", name=row.name))
            graph.add_edge(node_id("derivation", row.source), view_id, "serves")
            for entity in json.loads(row.entities_json):
                graph.add_edge(node_id("entity", str(entity)), view_id, "keys")

    def _add_metrics(
        self, graph: LineageGraph, verdicts: dict[str, str | None]
    ) -> dict[str, str | None]:
        """Add each metric downstream of the certified derivation it aggregates.

        A metric inherits its source derivation's oracle verdict (the verified number it
        aggregates). Returns each metric's source derivation by name, so a monitor
        watching a metric can inherit that same verdict.
        """
        sources: dict[str, str | None] = {}
        for row in self._store.list_metrics():
            sources[row.name] = row.source
            metric_node = node_id("metric", row.name)
            graph.add_node(
                LineageNode(
                    id=metric_node,
                    type="metric",
                    name=row.name,
                    verdict=verdicts.get(row.source) if row.source else None,
                    copied_from=row.copied_from,
                )
            )
            # A simple metric aggregates one derivation; a ratio metric has no single
            # source (its operands are other metrics) and stands as a node without one.
            if row.source:
                graph.add_edge(
                    node_id("derivation", row.source), metric_node, "aggregates"
                )
        return sources

    def _add_monitors(
        self,
        graph: LineageGraph,
        verdicts: dict[str, str | None],
        metric_sources: dict[str, str | None],
    ) -> None:
        """Add each monitor downstream of the certified metric or derivation it watches.

        The monitor inherits the oracle verdict of what it watches (a derivation
        directly, or a metric via that metric's source derivation), so the monitor node
        carries the verified status of the number whose movement it alerts on.
        """
        for row in self._store.list_metric_monitors():
            if row.target_kind == "metric":
                verdict = verdicts.get(metric_sources.get(row.target) or "")
            else:
                verdict = verdicts.get(row.target)
            monitor_node = node_id("monitor", row.name)
            graph.add_node(
                LineageNode(
                    id=monitor_node, type="monitor", name=row.name, verdict=verdict
                )
            )
            graph.add_edge(
                node_id(row.target_kind, row.target), monitor_node, "watches"
            )


def _model_source_node(model: ModelEdge) -> str | None:
    """The lineage node a model trains on, from its tagged training source.

    ``source_kind`` selects the upstream node type: a derivation, a bound dataset, or a
    feature-store training set. Falls back to the legacy feature-derivation tag for
    models tagged before ``source_kind`` was recorded.
    """
    kind = model.get("source_kind")
    dataset = model.get("dataset")
    if dataset and kind in ("derivation", "dataset", "training_set"):
        return node_id(str(kind), str(dataset))
    source = model.get("source")
    return node_id("derivation", str(source)) if source else None


class LineageError(Exception):
    """A lineage query failed for a reason worth showing the caller."""
