import type { Edge, Node } from "@xyflow/react"
import { Background, Controls, ReactFlow } from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import {
  Activity,
  BookMarked,
  Boxes,
  Database,
  FileCheck2,
  Gauge,
  Layers,
  LayoutDashboard,
  Network,
  Package,
  Search,
  Table2,
} from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { Scene, SceneHeader } from "@/components/Scene"
import { Input } from "@/components/ui/input"
import { VerdictBadge } from "@/components/VerdictBadge"
import { useTheme } from "@/hooks/useTheme"
import type { CatalogRecord, Graph, Impact, NodeType } from "@/lib/lineage"
import { getCatalog, getImpact, getSubgraph } from "@/lib/lineage"

const TYPE_ICON: Record<NodeType, typeof Database> = {
  dataset: Database,
  derivation: FileCheck2,
  model: Boxes,
  dashboard: LayoutDashboard,
  feature_view: Layers,
  training_set: Package,
  entity: Table2,
  metric: Gauge,
  monitor: Activity,
  semantic_model: BookMarked,
}

// Left-to-right layers: sources on the left, consumers on the right, monitors last
// (the observability layer that watches the metrics and derivations to its left).
const COLUMN: Record<NodeType, number> = {
  dataset: 0,
  entity: 0,
  semantic_model: 0,
  derivation: 1,
  model: 3,
  dashboard: 2,
  feature_view: 2,
  training_set: 2,
  metric: 2,
  monitor: 3,
}

const VERDICT_COLOR: Record<string, string> = {
  sound: "var(--verified)",
  unsound: "var(--danger)",
  inconclusive: "var(--caution)",
}

export function CatalogPage() {
  const { resolved } = useTheme()
  const [records, setRecords] = useState<CatalogRecord[]>([])
  const [query, setQuery] = useState("")
  const [selected, setSelected] = useState<string | null>(null)
  const [sub, setSub] = useState<Graph | null>(null)
  const [impact, setImpact] = useState<Impact | null>(null)

  useEffect(() => {
    getCatalog(query).then(setRecords)
  }, [query])

  useEffect(() => {
    if (selected === null) return
    getSubgraph(selected).then(setSub)
    getImpact(selected).then(setImpact)
  }, [selected])

  const { nodes, edges } = useMemo(() => toFlow(sub, selected), [sub, selected])

  return (
    <Scene>
      <SceneHeader
        icon={<Network className="size-5" />}
        title="Catalog & lineage"
        description="Every artifact and how it connects, with its oracle verdict."
      />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Catalog list */}
        <div className="flex w-80 shrink-0 flex-col border-r border-border">
          <div className="border-b border-border p-3">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 size-4 text-text-tertiary" />
              <Input
                className="pl-8"
                placeholder="Search datasets, derivations, models…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          </div>
          <ul className="flex-1 divide-y divide-border overflow-y-auto">
            {records.map((record) => {
              const Icon = TYPE_ICON[record.type]
              return (
                <li key={record.id}>
                  <button
                    type="button"
                    className={
                      "flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm transition-colors hover:bg-muted/60" +
                      (selected === record.id ? " bg-accent" : "")
                    }
                    onClick={() => setSelected(record.id)}
                  >
                    <Icon className="size-4 shrink-0 text-text-tertiary" />
                    <span className="flex-1 truncate">{record.name}</span>
                    {record.verdict ? <VerdictBadge verdict={record.verdict} /> : null}
                    <span className="text-[10px] uppercase tracking-wide text-text-tertiary">
                      {record.type}
                    </span>
                  </button>
                </li>
              )
            })}
            {records.length === 0 ? (
              <li className="px-4 py-6 text-sm text-text-tertiary">No artifacts match.</li>
            ) : null}
          </ul>
        </div>

        {/* Lineage graph + impact */}
        <div className="flex min-w-0 flex-1 flex-col">
          {selected === null ? (
            <div className="flex flex-1 items-center justify-center text-sm text-text-tertiary">
              Select an artifact to see its lineage and impact.
            </div>
          ) : (
            <>
              <div className="min-h-0 flex-1">
                <ReactFlow
                  nodes={nodes}
                  edges={edges}
                  fitView
                  colorMode={resolved}
                  nodesDraggable={false}
                  nodesConnectable={false}
                  proOptions={{ hideAttribution: true }}
                >
                  <Background />
                  <Controls showInteractive={false} />
                </ReactFlow>
              </div>
              {impact ? (
                <div className="border-t border-border bg-card p-4 text-sm">
                  <span className="font-medium">Impact: </span>
                  {impact.count === 0 ? (
                    <span className="text-text-secondary">
                      nothing depends on this: safe to change.
                    </span>
                  ) : (
                    <span className="text-foreground">
                      changing this affects {impact.count} artifact(s):{" "}
                      {Object.entries(impact.affected)
                        .map(([type, names]) => `${names.length} ${type}`)
                        .join(", ")}
                      .
                    </span>
                  )}
                </div>
              ) : null}
            </>
          )}
        </div>
      </div>
    </Scene>
  )
}

function toFlow(graph: Graph | null, focus: string | null): { nodes: Node[]; edges: Edge[] } {
  if (graph === null) return { nodes: [], edges: [] }
  const perColumn: Record<number, number> = {}
  const nodes: Node[] = graph.nodes.map((n) => {
    const column = COLUMN[n.type]
    const row = perColumn[column] ?? 0
    perColumn[column] = row + 1
    const color = n.verdict ? VERDICT_COLOR[n.verdict] : undefined
    return {
      id: n.id,
      position: { x: column * 260, y: row * 90 },
      data: { label: `${n.name}\n(${n.type})` },
      style: {
        borderRadius: 8,
        borderWidth: n.id === focus ? 2 : 1,
        borderColor: n.id === focus ? "var(--primary)" : (color ?? "var(--border)"),
        padding: 8,
        fontSize: 12,
        whiteSpace: "pre-line",
      },
    }
  })
  const edges: Edge[] = graph.edges.map((e, i) => ({
    id: `${e.source}->${e.target}-${i}`,
    source: e.source,
    target: e.target,
    label: e.kind,
    animated: true,
  }))
  return { nodes, edges }
}
