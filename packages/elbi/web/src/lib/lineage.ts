// API client and types for lineage + catalog: the cross-artifact dependency graph,
// impact analysis, and unified search.

export type NodeType =
  | "dataset"
  | "derivation"
  | "model"
  | "dashboard"
  | "feature_view"
  | "training_set"
  | "entity"
  | "metric"
  | "monitor"
  | "semantic_model"

export interface CatalogRecord {
  id: string
  type: NodeType
  name: string
  verdict: string | null
  certified: boolean | null
  description: string | null
  // Set when this artifact was duplicated from another. A record of where it came
  // from, not a dependency: no edge joins a copy to its source, and the source may
  // since have been deleted, so this is not safe to resolve without a lookup.
  copiedFrom: string | null
  upstream: number
}

export interface GraphNode {
  id: string
  type: NodeType
  name: string
  verdict: string | null
  certified: boolean | null
  description: string | null
  copiedFrom: string | null
}

export interface GraphEdge {
  source: string
  target: string
  kind: string
}

export interface Graph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  focus?: string
}

export interface Impact {
  node: string
  affected: Record<string, string[]>
  count: number
}

/** What feeds a node: the mirror of Impact, grouped by artifact type. */

async function json<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`)
  return (await r.json()) as T
}

const q = encodeURIComponent

export const getCatalog = (query = "") => json<CatalogRecord[]>(`/api/catalog?q=${q(query)}`)

export const getGraph = () => json<Graph>("/api/lineage/graph")

export const getSubgraph = (node: string) => json<Graph>(`/api/lineage/subgraph?node=${q(node)}`)

export const getImpact = (node: string) => json<Impact>(`/api/lineage/impact?node=${q(node)}`)
