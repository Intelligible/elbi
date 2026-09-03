"""Lineage: a typed dependency graph across datasets, derivations, and consumers.

Defines the graph (:class:`LineageGraph`, :class:`LineageNode`, :class:`LineageEdge`),
its traversals (``ancestors`` for provenance, ``descendants`` for impact), and
:func:`from_registry` to build the dataset→derivation portion from a registry. The app
layer extends it with model, dashboard, and feature-view nodes.
"""

from __future__ import annotations

from .graph import (
    NODE_TYPES,
    LineageEdge,
    LineageGraph,
    LineageNode,
    from_registry,
    merge,
    node_id,
)

__all__ = [
    "NODE_TYPES",
    "LineageEdge",
    "LineageGraph",
    "LineageNode",
    "from_registry",
    "merge",
    "node_id",
]
