import { Boxes, Database, Layers, Plus, RefreshCw, Search, Trash2 } from "lucide-react"
import type { ReactNode } from "react"
import { useCallback, useEffect, useId, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Scene, SceneBody, SceneHeader, SceneSection, SceneSkeleton } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardHeader } from "@/components/ui/card"
import { type Column, DataTable } from "@/components/ui/data-table"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import type { FeatureEntity, FeatureView, TrainingSet } from "@/lib/features"
import {
  defineEntity,
  defineFeatureView,
  deleteFeatureView,
  deleteTrainingSet,
  listEntities,
  listFeatureViews,
  listTrainingSets,
  materializeFeatures,
} from "@/lib/features"
import { EMPTY, relativeTime, uuid } from "@/lib/utils"

const VALUE_TYPES = ["", "string", "integer", "float", "boolean"] as const

export function FeatureStorePage() {
  const navigate = useNavigate()
  const [views, setViews] = useState<FeatureView[] | null>(null)
  const [entities, setEntities] = useState<FeatureEntity[]>([])
  const [trainingSets, setTrainingSets] = useState<TrainingSet[]>([])
  const [query, setQuery] = useState("")
  const [busy, setBusy] = useState(false)
  const [entityOpen, setEntityOpen] = useState(false)
  const [viewOpen, setViewOpen] = useState(false)

  const refresh = useCallback(async () => {
    const [v, e, t] = await Promise.all([listFeatureViews(), listEntities(), listTrainingSets()])
    setViews(v)
    setEntities(e)
    setTrainingSets(t)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const materializeAll = async () => {
    setBusy(true)
    try {
      await materializeFeatures()
      await refresh()
    } finally {
      setBusy(false)
    }
  }

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle || views === null) return views ?? []
    return views.filter((v) =>
      `${v.name} ${v.source} ${v.description ?? ""} ${v.features.map((f) => f.name).join(" ")}`
        .toLowerCase()
        .includes(needle),
    )
  }, [views, query])

  if (views === null) return <SceneSkeleton />

  const columns: Column<FeatureView>[] = [
    {
      key: "name",
      title: "Feature view",
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (v) => (
        <div className="flex items-center gap-2">
          <span className="font-mono font-medium">{v.name}</span>
          <Badge variant={v.certified ? "verified" : "warning"}>
            {v.certified ? "certified" : "uncertified"}
          </Badge>
        </div>
      ),
    },
    {
      key: "source",
      title: "Source",
      render: (v) => <span className="font-mono text-text-secondary">{v.source}</span>,
    },
    {
      key: "keys",
      title: "Keys",
      render: (v) => (
        <span className="font-mono text-text-secondary">[{v.joinKeys.join(", ")}]</span>
      ),
    },
    {
      key: "features",
      title: "Features",
      align: "right",
      sorter: (a, b) => a.features.length - b.features.length,
      render: (v) => (
        <span className="tabular-nums text-text-secondary">{v.features.length || "all"}</span>
      ),
    },
    {
      key: "freshness",
      title: "Materialized",
      align: "right",
      sorter: (a, b) => (a.lastMaterializedAt ?? "").localeCompare(b.lastMaterializedAt ?? ""),
      render: (v) =>
        v.nOnlineKeys > 0 ? (
          <span className="tabular-nums text-text-secondary">
            {v.nOnlineKeys} keys · {relativeTime(v.lastMaterializedAt)}
          </span>
        ) : (
          <span className="text-text-tertiary">{EMPTY}</span>
        ),
    },
    {
      key: "actions",
      title: "",
      width: 44,
      render: (v) => (
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Delete feature view"
          onClick={(e) => {
            e.stopPropagation()
            void deleteFeatureView(v.name).then(refresh)
          }}
        >
          <Trash2 className="size-4" />
        </Button>
      ),
    },
  ]

  return (
    <Scene>
      <SceneHeader
        icon={<Layers className="size-5" />}
        title="Feature store"
        description="Point-in-time-correct features over certified derivations, served online and offline with no train/serve skew."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={materializeAll} disabled={busy}>
              <RefreshCw className={busy ? "size-4 animate-spin" : "size-4"} />
              Materialize all
            </Button>
            <Button size="sm" onClick={() => setViewOpen(true)}>
              <Plus className="size-4" />
              Feature view
            </Button>
          </>
        }
      />
      <SceneBody>
        <SceneSection
          title="Entities"
          action={
            <Button variant="ghost" size="sm" onClick={() => setEntityOpen(true)}>
              <Plus className="size-4" />
              New entity
            </Button>
          }
        >
          {entities.length === 0 ? (
            <p className="text-sm text-text-tertiary">
              No entities yet. An entity names a join key (e.g. <code>user_id</code>).
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {entities.map((e) => (
                <Badge key={e.name} variant="neutral" className="gap-1">
                  <Database className="size-3" />
                  {e.name} · {e.joinKey}
                </Badge>
              ))}
            </div>
          )}
        </SceneSection>

        <SceneSection title="Feature views">
          {views.length === 0 ? (
            <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border-strong py-16 text-center">
              <Boxes className="size-8 text-text-tertiary" />
              <p className="text-sm text-text-secondary">
                No feature views yet. Define one over a certified derivation.
              </p>
            </div>
          ) : (
            <DataTable
              columns={columns}
              data={filtered}
              rowKey={(v) => v.name}
              onRowClick={(v) => navigate(`/features/${encodeURIComponent(v.name)}`)}
              empty="No feature views match your search."
              toolbar={
                <div className="relative w-64">
                  <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
                  <Input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search features"
                    className="h-8 pl-8"
                  />
                </div>
              }
            />
          )}
        </SceneSection>

        {trainingSets.length > 0 && (
          <SceneSection
            title="Training sets"
            description="Materialized point-in-time joins: a leakage-free training table a model can train on directly."
          >
            <div className="space-y-2">
              {trainingSets.map((ts) => (
                <Card key={ts.name}>
                  <CardHeader className="flex-row items-center justify-between gap-2 py-3">
                    <div className="flex items-center gap-2 text-sm">
                      <Layers className="size-4 text-text-tertiary" />
                      <span className="font-mono font-medium">{ts.name}</span>
                      <span className="text-text-tertiary">
                        {ts.rowCount} rows · {ts.features.join(", ")}
                        {ts.label ? ` → ${ts.label}` : ""}
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => void deleteTrainingSet(ts.name).then(refresh)}
                      aria-label="Delete training set"
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </SceneSection>
        )}
      </SceneBody>

      <EntityDialog
        open={entityOpen}
        onOpenChange={setEntityOpen}
        onSaved={() => {
          setEntityOpen(false)
          void refresh()
        }}
      />
      <FeatureViewDialog
        open={viewOpen}
        onOpenChange={setViewOpen}
        entities={entities}
        onSaved={() => {
          setViewOpen(false)
          void refresh()
        }}
      />
    </Scene>
  )
}

function EntityDialog({
  open,
  onOpenChange,
  onSaved,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved: () => void
}) {
  const [name, setName] = useState("")
  const [joinKey, setJoinKey] = useState("")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      setName("")
      setJoinKey("")
      setError(null)
    }
  }, [open])

  const save = async () => {
    try {
      await defineEntity({ name: name.trim(), join_key: joinKey.trim() })
      onSaved()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New entity</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <Field label="Name">
            {(id) => (
              <Input
                id={id}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="user"
              />
            )}
          </Field>
          <Field label="Join key (column)">
            {(id) => (
              <Input
                id={id}
                value={joinKey}
                onChange={(e) => setJoinKey(e.target.value)}
                placeholder="user_id"
              />
            )}
          </Field>
          {error ? <p className="text-sm text-danger">{error}</p> : null}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!name.trim() || !joinKey.trim()}>
            Create
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// `rowId` is the editor's, never the view's: the submit path below maps each row to
// an explicit wire shape, so it cannot travel. Rows are keyed by it because removing
// one shifts every position after it, and React would hand the removed row's inputs to
// its successor -- caret, selection and any in-flight composition with them.
type FeatureRow = { rowId: string; name: string; dtype: string; description: string }

function FeatureViewDialog({
  open,
  onOpenChange,
  entities,
  onSaved,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  entities: FeatureEntity[]
  onSaved: () => void
}) {
  const [name, setName] = useState("")
  const [selected, setSelected] = useState<string[]>([])
  const [source, setSource] = useState("")
  const [features, setFeatures] = useState<FeatureRow[]>([])
  const [timestamp, setTimestamp] = useState("")
  const [ttlDays, setTtlDays] = useState("")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      setName("")
      setSelected([])
      setSource("")
      setFeatures([])
      setTimestamp("")
      setTtlDays("")
      setError(null)
    }
  }, [open])

  const setFeature = (rowId: string, patch: Partial<FeatureRow>) =>
    setFeatures((rows) => rows.map((r) => (r.rowId === rowId ? { ...r, ...patch } : r)))

  const save = async () => {
    const declared = features
      .filter((f) => f.name.trim())
      .map((f) => ({
        name: f.name.trim(),
        ...(f.dtype ? { dtype: f.dtype } : {}),
        ...(f.description.trim() ? { description: f.description.trim() } : {}),
      }))
    try {
      await defineFeatureView({
        name: name.trim(),
        entities: selected,
        source: source.trim(),
        features: declared.length ? declared : undefined,
        timestampField: timestamp.trim() || undefined,
        ttlSeconds: ttlDays ? Number(ttlDays) * 86400 : undefined,
      })
      onSaved()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Define feature view</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <Field label="Name">
            {(id) => (
              <Input
                id={id}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="user_stats"
              />
            )}
          </Field>
          <FieldGroup label="Entities">
            {entities.length === 0 ? (
              <p className="text-xs text-muted-foreground">Define an entity first.</p>
            ) : (
              <div className="flex flex-wrap gap-1">
                {entities.map((e) => {
                  const on = selected.includes(e.name)
                  return (
                    <Button
                      key={e.name}
                      type="button"
                      variant={on ? "default" : "outline"}
                      size="xs"
                      onClick={() =>
                        setSelected(
                          on ? selected.filter((s) => s !== e.name) : [...selected, e.name],
                        )
                      }
                    >
                      {e.name}
                    </Button>
                  )
                })}
              </div>
            )}
          </FieldGroup>
          <Field label="Source derivation">
            {(id) => (
              <Input
                id={id}
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="user_activity_features"
              />
            )}
          </Field>
          <FieldGroup label="Features (leave empty to expose all source columns)">
            <div className="space-y-2">
              {features.map((f) => (
                <div key={f.rowId} className="flex gap-2">
                  <Input
                    value={f.name}
                    onChange={(e) => setFeature(f.rowId, { name: e.target.value })}
                    placeholder="clicks_7d"
                    className="h-8 flex-1"
                  />
                  <select
                    value={f.dtype}
                    onChange={(e) => setFeature(f.rowId, { dtype: e.target.value })}
                    className="h-8 rounded-md border border-input bg-transparent px-2 text-sm"
                  >
                    {VALUE_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t || "type"}
                      </option>
                    ))}
                  </select>
                  <Input
                    value={f.description}
                    onChange={(e) => setFeature(f.rowId, { description: e.target.value })}
                    placeholder="description"
                    className="h-8 flex-1"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    aria-label="Remove feature"
                    onClick={() => setFeatures((rows) => rows.filter((r) => r.rowId !== f.rowId))}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              ))}
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() =>
                  setFeatures((rows) => [
                    ...rows,
                    { rowId: uuid(), name: "", dtype: "", description: "" },
                  ])
                }
              >
                <Plus className="size-3.5" />
                Add feature
              </Button>
            </div>
          </FieldGroup>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Event timestamp column">
              {(id) => (
                <Input
                  id={id}
                  value={timestamp}
                  onChange={(e) => setTimestamp(e.target.value)}
                  placeholder="event_timestamp"
                />
              )}
            </Field>
            <Field label="TTL (days)">
              {(id) => (
                <Input
                  id={id}
                  type="number"
                  value={ttlDays}
                  onChange={(e) => setTtlDays(e.target.value)}
                  placeholder="7"
                />
              )}
            </Field>
          </div>
          {error ? <p className="text-sm text-danger">{error}</p> : null}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!name.trim() || selected.length === 0 || !source.trim()}>
            Define
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** A labelled control. The id is handed to the child so the association is written
 *  down rather than inferred from nesting -- which is what a screen reader needs when
 *  the control is a composite rather than a bare input. */
function Field({ label, children }: { label: string; children: (id: string) => ReactNode }) {
  const id = useId()
  return (
    <div className="block space-y-1">
      <label className="text-xs font-medium text-muted-foreground" htmlFor={id}>
        {label}
      </label>
      {children(id)}
    </div>
  )
}

/** A heading over related controls. Not a label: there is no single control to name. */
function FieldGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="block space-y-1">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </div>
  )
}
