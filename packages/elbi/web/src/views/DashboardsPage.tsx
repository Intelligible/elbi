import { Copy, LayoutDashboard, Plus, Search, Trash2 } from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { type Column, DataTable } from "@/components/ui/data-table"
import { FeedbackProvider, useFeedback } from "@/components/ui/feedback"
import { Input } from "@/components/ui/input"
import type { DashboardSummary } from "@/lib/dashboards"
import {
  createDashboard,
  deleteDashboard,
  duplicateDashboard,
  listDashboards,
  starterSpec,
} from "@/lib/dashboards"

export function DashboardsPage() {
  return (
    <FeedbackProvider>
      <DashboardsBody />
    </FeedbackProvider>
  )
}

function DashboardsBody() {
  const fb = useFeedback()
  const [dashboards, setDashboards] = useState<DashboardSummary[] | null>(null)
  const [q, setQ] = useState("")
  const navigate = useNavigate()

  const refresh = useCallback(() => listDashboards().then(setDashboards), [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const create = async () => {
    const { id } = await createDashboard(starterSpec("Untitled dashboard"))
    navigate(`/dashboards/${id}`)
  }

  const remove = async (id: string) => {
    await deleteDashboard(id)
    void refresh()
  }

  const duplicate = async (id: string) => {
    try {
      const copy = await duplicateDashboard(id)
      navigate(`/dashboards/${copy.id}`)
    } catch (e) {
      fb.toast("error", e instanceof Error ? e.message : "could not duplicate")
    }
  }

  const filtered = useMemo(() => {
    if (!dashboards) return []
    const term = q.trim().toLowerCase()
    return term
      ? dashboards.filter((d) => (d.title || d.name).toLowerCase().includes(term))
      : dashboards
  }, [dashboards, q])

  if (dashboards === null) return <SceneSkeleton />

  const columns: Column<DashboardSummary>[] = [
    {
      key: "name",
      title: "Name",
      width: "50%",
      sorter: (a, b) => (a.title || a.name).localeCompare(b.title || b.name),
      render: (d) => (
        <span className="flex items-center gap-2">
          <LayoutDashboard className="size-4 shrink-0 text-text-tertiary" />
          <span className="truncate font-medium text-foreground">{d.title || d.name}</span>
          {d.copiedFrom && (
            <Badge variant="neutral" className="shrink-0" title="Duplicated from another dashboard">
              copy
            </Badge>
          )}
        </span>
      ),
    },
    {
      key: "status",
      title: "Status",
      width: 120,
      render: (d) => (
        <Badge variant={d.status === "published" ? "success" : "neutral"}>{d.status}</Badge>
      ),
    },
    {
      key: "version",
      title: "Version",
      width: 100,
      render: (d) => (
        <span className="font-mono text-text-secondary tabular-nums">v{d.version}</span>
      ),
    },
    {
      key: "updated",
      title: "Updated",
      width: 128,
      align: "right",
      sorter: (a, b) => a.updatedAt.localeCompare(b.updatedAt),
      render: (d) => (
        <span className="text-text-tertiary tabular-nums">
          {new Date(d.updatedAt).toLocaleDateString()}
        </span>
      ),
    },
    {
      key: "actions",
      title: "",
      width: 76,
      align: "right",
      render: (d) => (
        <span className="inline-flex gap-1">
          <button
            type="button"
            onClick={() => void duplicate(d.id)}
            className="rounded-md p-1 text-text-tertiary transition-colors hover:bg-muted hover:text-foreground"
            aria-label={`Duplicate ${d.name}`}
          >
            <Copy className="size-4" />
          </button>
          <button
            type="button"
            onClick={() => void remove(d.id)}
            className="rounded-md p-1 text-text-tertiary transition-colors hover:bg-muted hover:text-danger"
            aria-label="Delete dashboard"
          >
            <Trash2 className="size-4" />
          </button>
        </span>
      ),
    },
  ]

  return (
    <Scene>
      <SceneHeader
        icon={<LayoutDashboard className="size-5" />}
        title="Dashboards"
        description="Presentation surfaces whose every tile binds to a certified derivation."
        actions={
          <Button size="sm" onClick={create}>
            <Plus className="size-4" /> New dashboard
          </Button>
        }
      />
      <SceneBody width="full" canvas>
        <DataTable
          columns={columns}
          data={filtered}
          rowKey={(d) => d.id}
          onRowClick={(d) => navigate(`/dashboards/${d.id}`)}
          empty={
            q ? (
              "No dashboards match your search."
            ) : (
              <span className="flex flex-col items-center gap-3">
                No dashboards yet. Create one and add widgets bound to your derivations.
                <Button variant="outline" size="sm" onClick={create}>
                  <Plus className="size-4" /> New dashboard
                </Button>
              </span>
            )
          }
          toolbar={
            <div className="relative max-w-xs">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search dashboards"
                className="h-8 pl-8"
              />
            </div>
          }
        />
      </SceneBody>
    </Scene>
  )
}
