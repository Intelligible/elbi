"""The lineage graph: typed artifact nodes and the dependency edges between them.

A node is any artifact (a dataset, a derivation, a model, a dashboard, a feature view,
an entity, a metric, or a monitor) identified by ``"{type}:{name}"``. An edge runs
upstream → downstream (a source dataset to the derivation that consumes it, a derivation
to the model that trains on it, a derivation to the metric that aggregates it, a metric
or derivation to the monitor that watches it). The graph answers the two lineage
questions: ``ancestors`` (what feeds this) and ``descendants`` (what a change to this
would affect).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..registry import Registry

#: The artifact kinds a node can be.
NODE_TYPES = (
    "dataset",
    "derivation",
    "model",
    "dashboard",
    "feature_view",
    "training_set",
    "entity",
    "metric",
    "monitor",
    "semantic_model",
)


def node_id(node_type: str, name: str) -> str:
    """The globally unique id for an artifact node."""
    return f"{node_type}:{name}"


@dataclass(frozen=True)
class LineageNode:
    """One artifact in the graph."""

    id: str
    type: str
    name: str
    verdict: str | None = None
    certified: bool | None = None
    description: str | None = None
    #: The source artifact's id or name, when this node is a duplicate of it.
    copied_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the API/UI."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "verdict": self.verdict,
            "certified": self.certified,
            "description": self.description,
            "copied_from": self.copied_from,
        }


@dataclass(frozen=True)
class LineageEdge:
    """A directed dependency: ``source`` (upstream) feeds ``target`` (downstream)."""

    source: str
    target: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        """Serialize for the API/UI."""
        return {"source": self.source, "target": self.target, "kind": self.kind}


@dataclass
class LineageGraph:
    """A directed graph of artifact nodes and dependency edges."""

    _nodes: dict[str, LineageNode] = field(default_factory=dict)
    _edges: list[LineageEdge] = field(default_factory=list)
    _out: dict[str, list[str]] = field(default_factory=dict)
    _in: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, node: LineageNode) -> None:
        """Add a node, or enrich an existing placeholder with fuller metadata."""
        existing = self._nodes.get(node.id)
        if existing is None or (existing.verdict is None and node.verdict is not None):
            self._nodes[node.id] = node
        self._out.setdefault(node.id, [])
        self._in.setdefault(node.id, [])

    def add_edge(self, source: str, target: str, kind: str) -> None:
        """Add a directed edge, registering placeholder endpoints if unseen."""
        for endpoint in (source, target):
            if endpoint not in self._nodes:
                kind_, _, name = endpoint.partition(":")
                self.add_node(LineageNode(id=endpoint, type=kind_, name=name))
        if target not in self._out[source]:
            self._out[source].append(target)
            self._in[target].append(source)
            self._edges.append(LineageEdge(source, target, kind))

    def node(self, node_id: str) -> LineageNode | None:
        """The node with ``node_id``, or ``None``."""
        return self._nodes.get(node_id)

    def nodes(self) -> list[LineageNode]:
        """All nodes."""
        return list(self._nodes.values())

    def edges(self) -> list[LineageEdge]:
        """All edges."""
        return list(self._edges)

    def upstream(self, node_id: str) -> list[str]:
        """Direct predecessors (what ``node_id`` depends on)."""
        return list(self._in.get(node_id, []))

    def downstream(self, node_id: str) -> list[str]:
        """Direct successors (what depends on ``node_id``)."""
        return list(self._out.get(node_id, []))

    def ancestors(self, node_id: str) -> set[str]:
        """Every upstream node reachable from ``node_id`` (its full provenance)."""
        return self._reach(node_id, self._in)

    def descendants(self, node_id: str) -> set[str]:
        """Every downstream node reachable from ``node_id`` (its blast radius)."""
        return self._reach(node_id, self._out)

    def roots(self) -> list[str]:
        """Nodes with no upstream (source datasets and inputs)."""
        return [nid for nid in self._nodes if not self._in.get(nid)]

    def topo_order(self) -> list[str]:
        """Node ids in dependency order (Kahn's algorithm); cycle members last."""
        indegree = {nid: len(self._in.get(nid, [])) for nid in self._nodes}
        queue = deque(sorted(nid for nid, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in sorted(self._out.get(nid, [])):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)
        order.extend(sorted(nid for nid in self._nodes if nid not in set(order)))
        return order

    def subgraph(self, node_id: str) -> dict[str, Any]:
        """The node with its ancestors and descendants, as ``{nodes, edges, focus}``."""
        keep = {node_id} | self.ancestors(node_id) | self.descendants(node_id)
        return {
            "focus": node_id,
            "nodes": [n.to_dict() for n in self._nodes.values() if n.id in keep],
            "edges": [
                e.to_dict()
                for e in self._edges
                if e.source in keep and e.target in keep
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """The whole graph as ``{nodes, edges}`` for the API/UI."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def _reach(self, start: str, adjacency: dict[str, list[str]]) -> set[str]:
        seen: set[str] = set()
        queue = deque(adjacency.get(start, []))
        while queue:
            nid = queue.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            queue.extend(adjacency.get(nid, []))
        return seen


def from_registry(
    registry: Registry,
    datasets: Sequence[str] = (),
    *,
    verdict_of: Callable[[str], str | None] | None = None,
) -> LineageGraph:
    """Build the dataset→derivation→derivation graph from a registry.

    ``datasets`` seeds declared source datasets even if unreferenced. ``verdict_of``
    supplies each derivation's oracle verdict by name.
    """
    graph = LineageGraph()
    for name in datasets:
        graph.add_node(
            LineageNode(id=node_id("dataset", name), type="dataset", name=name)
        )
    for derivation in registry:
        graph.add_node(
            LineageNode(
                id=node_id("derivation", derivation.name),
                type="derivation",
                name=derivation.name,
                verdict=verdict_of(derivation.name) if verdict_of else None,
                certified=derivation.is_certified,
                description=derivation.description,
            )
        )
    for derivation in registry:
        target = node_id("derivation", derivation.name)
        for dataset in derivation.dataset_inputs().values():
            graph.add_edge(node_id("dataset", dataset.name), target, "consumes")
        for upstream in derivation.derivation_inputs().values():
            graph.add_edge(node_id("derivation", upstream.name), target, "consumes")
        for model in derivation.semantic_model_inputs().values():
            graph.add_edge(node_id("semantic_model", model.name), target, "consumes")
    return graph


def merge(graphs: Iterable[LineageGraph]) -> LineageGraph:
    """Combine several graphs into one, unioning nodes and edges."""
    merged = LineageGraph()
    for graph in graphs:
        for node in graph.nodes():
            merged.add_node(node)
        for edge in graph.edges():
            merged.add_edge(edge.source, edge.target, edge.kind)
    return merged
