import { Boxes, ExternalLink, Loader2, Plus, Search } from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { isActiveTraining, TrainingStatus, useTrainingJobs } from "@/components/TrainingJobs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { type Column, DataTable } from "@/components/ui/data-table"
import { Input } from "@/components/ui/input"
import { getRegisteredModels, type RegisteredModel, type RegistryUnavailable } from "@/lib/chat"
import { EMPTY } from "@/lib/utils"

export function ModelsPage() {
  const navigate = useNavigate()
  const [models, setModels] = useState<RegisteredModel[] | RegistryUnavailable | null>(null)
  const [q, setQ] = useState("")
  const refresh = useCallback(() => {
    void getRegisteredModels().then(setModels)
  }, [])
  useEffect(refresh, [refresh])
  const { jobs, dismiss } = useTrainingJobs(refresh)

  const list = models !== null && !("unavailable" in models) ? models : []
  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase()
    return term ? list.filter((m) => m.name.toLowerCase().includes(term)) : list
  }, [list, q])

  if (models === null) return <SceneSkeleton />
  const available = !("unavailable" in models)

  const columns: Column<RegisteredModel>[] = [
    {
      key: "name",
      title: "Name",
      width: "32%",
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (m) => (
        <span className="flex items-center gap-2">
          <Boxes className="size-4 shrink-0 text-text-tertiary" />
          <span className="truncate font-mono text-compact font-medium text-foreground">
            {m.name}
          </span>
          {jobs.some((j) => isActiveTraining(j) && j.label === `train model ${m.name}`) && (
            <Loader2 className="size-3.5 shrink-0 animate-spin text-text-tertiary" />
          )}
        </span>
      ),
    },
    {
      key: "champion",
      title: "Champion",
      width: 190,
      render: (m) =>
        m.championVersion !== null ? (
          <Badge variant="verified" className="font-mono">
            @champion → v{m.championVersion}
          </Badge>
        ) : (
          <Badge variant="neutral">no champion</Badge>
        ),
    },
    {
      key: "latest",
      title: "Latest",
      width: 100,
      render: (m) => (
        <span className="font-mono text-text-secondary tabular-nums">v{m.latestVersion}</span>
      ),
    },
    {
      key: "description",
      title: "Description",
      render: (m) => (
        <span className="line-clamp-1 text-text-secondary">{m.description || EMPTY}</span>
      ),
    },
    {
      key: "updated",
      title: "Updated",
      width: 128,
      align: "right",
      sorter: (a, b) => a.updatedAtMs - b.updatedAtMs,
      render: (m) => (
        <span className="text-text-tertiary tabular-nums">{relativeTime(m.updatedAtMs)}</span>
      ),
    },
  ]

  return (
    <Scene>
      <SceneHeader
        icon={<Boxes className="size-5" />}
        title="Models"
        description="Trained models in the registry, with a promotable champion per name."
        actions={
          available && (
            <>
              <Button variant="ghost" size="sm" asChild>
                <a href="/mlflow/" target="_blank" rel="noreferrer">
                  Open in MLflow <ExternalLink className="size-3.5" />
                </a>
              </Button>
              <Button size="sm" asChild>
                <Link to="/models/train">
                  <Plus className="size-4" /> Train model
                </Link>
              </Button>
            </>
          )
        }
      />
      <SceneBody width="full" canvas>
        {jobs.length > 0 && (
          <div className="mb-4 space-y-2">
            {jobs.map((j) => (
              <TrainingStatus key={j.id} job={j} onDismiss={() => dismiss(j.id)} />
            ))}
          </div>
        )}

        {!available ? (
          <p className="text-sm text-text-secondary">{models.unavailable}</p>
        ) : (
          <DataTable
            columns={columns}
            data={filtered}
            rowKey={(m) => m.name}
            onRowClick={(m) => navigate(`/models/${encodeURIComponent(m.name)}`)}
            empty={
              q ? (
                "No models match your search."
              ) : (
                <span className="flex flex-col items-center gap-3">
                  No models yet. Train one on a dataset, or ask the chat, and the model it trains is
                  registered here too.
                  <Button size="sm" asChild>
                    <Link to="/models/train">
                      <Plus className="size-4" /> Train model
                    </Link>
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
                  placeholder="Search models"
                  className="h-8 pl-8"
                />
              </div>
            }
          />
        )}
      </SceneBody>
    </Scene>
  )
}

// Compact relative time from an epoch-millisecond stamp (the registry's unit).
function relativeTime(ms: number): string {
  const seconds = (Date.now() - ms) / 1000
  if (seconds < 60) return "just now"
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  if (seconds < 7 * 86400) return new Date(ms).toLocaleDateString(undefined, { weekday: "short" })
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" })
}
