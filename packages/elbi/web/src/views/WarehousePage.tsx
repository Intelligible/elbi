import {
  Database,
  Loader2,
  Megaphone,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Warehouse,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useNavigate, useParams, useSearchParams } from "react-router-dom"
import { Scene, SceneHeader } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Phc } from "@/components/warehouse/phc"
import { SourceIcon } from "@/components/warehouse/SourceIcon"
import { type Dataset, getDatasets } from "@/lib/chat"
import { EMPTY } from "@/lib/utils"
import {
  type Catalog,
  deleteSource,
  getCatalog,
  getSource,
  listSources,
  registerInterest,
  type SchemaView,
  type SourceConfig,
  type SourceDetail,
  type SourceSummary,
  type SyncFrequency,
  type SyncOutcome,
  setSyncFrequency,
  syncSource,
  updateSchema,
} from "@/lib/warehouse"
import { NewSourceForm } from "@/views/NewSourceForm"
import { EditSourceDialog } from "./EditSourceDialog"

const FREQUENCY_LABELS: { value: SyncFrequency; label: string }[] = [
  { value: "manual", label: "Manual only" },
  { value: "30min", label: "Every 30 min" },
  { value: "1hour", label: "Hourly" },
  { value: "6hour", label: "Every 6 hours" },
  { value: "12hour", label: "Every 12 hours" },
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
]

function ErrorBanner({ error }: { error: string | null }) {
  if (!error) return null
  return (
    <div className="border-b border-danger/30 bg-danger-tint px-4 py-2 font-mono text-xs text-danger">
      {error}
    </div>
  )
}

const MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ")

function fmtRun(iso: string | null): string {
  if (!iso) return "Never"
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return "Never"
  const p = (n: number) => String(n).padStart(2, "0")
  return `${MONTHS[d.getMonth()]} ${p(d.getDate())}, ${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}`
}

function StatusTag({ source }: { source: SourceSummary }) {
  if (source.status === "error") return <span className="tag tag--danger">Error</span>
  if (source.status === "syncing") return <span className="tag tag--info">Running</span>
  if (source.syncedCount > 0)
    return <span className="tag tag--success">{source.syncedCount} synced</span>
  return <span className="tag tag--muted">Not syncing</span>
}

// GET /warehouse: the Data warehouse: bound datasets (declared in elbi.yaml) and managed
// sources (synced connectors), unified as the readable inputs the governed pipeline builds on.
// Sources are the ingestion mechanism, not a rival concept, so this is one catalog rather than
// two competing "here's your data" tabs.
export function WarehousePage() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState<Dataset[] | null>(null)
  const [sources, setSources] = useState<SourceSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState("")

  const refresh = useCallback(() => {
    listSources()
      .then(setSources)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])
  useEffect(refresh, [refresh])
  useEffect(() => {
    getDatasets()
      .then(setDatasets)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const q = search.trim().toLowerCase()
  const filtered = sources.filter(
    (s) =>
      !q ||
      s.name.toLowerCase().includes(q) ||
      s.sourceType.toLowerCase().includes(q) ||
      (s.description ?? "").toLowerCase().includes(q),
  )

  const remove = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    try {
      await deleteSource(id)
      refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="relative h-full w-full">
      <Phc>
        <div className="title-row">
          <h1>Data warehouse</h1>
          <button
            type="button"
            className="btn btn--primary btn--sm"
            onClick={() => navigate("/warehouse/new-source")}
          >
            <Plus className="size-4" /> New source
          </button>
        </div>

        {error ? <div className="err">{error}</div> : null}

        <div className="section">
          <h2>Datasets</h2>
          <p>
            Every queryable table: bound datasets from the project's <code>elbi.yaml</code> and
            synced warehouse sources: available to every analysis and derivation.
          </p>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 0 }} />
                <th>Dataset</th>
                <th className="num">Rows</th>
                <th className="num">Columns</th>
                <th>Origin</th>
              </tr>
            </thead>
            <tbody>
              {datasets === null ? (
                <tr>
                  <td colSpan={5}>
                    <div className="empty">
                      <span>Loading…</span>
                    </div>
                  </td>
                </tr>
              ) : datasets.length === 0 ? (
                <tr>
                  <td colSpan={5}>
                    <div className="empty">
                      <span>No datasets bound in this project</span>
                    </div>
                  </td>
                </tr>
              ) : (
                datasets.map((d) => (
                  <tr key={d.name}>
                    <td>
                      <Database className="size-[22px] text-[var(--phc-muted)]" />
                    </td>
                    <td>
                      <div className="link-title">{d.name}</div>
                    </td>
                    <td className="num">{d.rows.toLocaleString()}</td>
                    <td className="num">{d.columns}</td>
                    <td>
                      {d.origin === "synced" ? (
                        <span className="tag tag--info">Synced</span>
                      ) : (
                        <span className="tag tag--muted">Bound</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="section">
          <h2>Managed data warehouse sources</h2>
          <p>
            Connect to external systems and sync their data into your warehouse. Synced tables
            become first-class data: usable in Explore, notebooks, chat, models, and metrics, just
            like a bound dataset.
          </p>
          <input
            className="input search"
            type="search"
            placeholder="Search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 0 }} />
                <th>Source</th>
                <th>Table prefix</th>
                <th>Last Successful Run</th>
                <th className="num">Total Rows Synced</th>
                <th>Status</th>
                <th style={{ width: 0 }} />
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={7}>
                    <div className="empty">
                      <span>{q ? "No sources matching your search" : "No managed sources"}</span>
                      <button
                        type="button"
                        className="btn btn--secondary btn--sm"
                        onClick={() => navigate("/warehouse/new-source")}
                      >
                        <Plus className="size-4" /> New source
                      </button>
                    </div>
                  </td>
                </tr>
              ) : (
                filtered.map((s) => (
                  <tr key={s.id} onClick={() => navigate(`/warehouse/sources/${s.id}`)}>
                    <td>
                      <SourceIcon type={s.sourceType} size={28} />
                    </td>
                    <td>
                      <div className="link-title">{s.name}</div>
                      {s.description ? <div className="sub">{s.description}</div> : null}
                    </td>
                    <td>{s.prefix || "-"}</td>
                    <td>{fmtRun(s.lastSyncedAt)}</td>
                    <td className="num">{s.rows.toLocaleString()}</td>
                    <td>
                      <StatusTag source={s} />
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn--secondary btn--sm"
                        style={{ padding: "0 0.5rem" }}
                        aria-label="Delete source"
                        onClick={(e) => remove(e, s.id)}
                      >
                        <Trash2 className="size-4" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Phc>
    </div>
  )
}

// GET /warehouse/new-source[?kind=X]: the connector catalog, or the connection form for a
// chosen source. Deep-linkable: `?kind=stripe` opens that form directly.
export function NewSourcePage() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const kind = params.get("kind")
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getCatalog()
      .then(setCatalog)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const source = kind ? (catalog?.sources.find((s) => s.name === kind) ?? null) : null

  // A chosen source → the connection form as the full main content.
  if (source) {
    return (
      <div className="relative h-full w-full">
        <NewSourceForm
          source={source}
          error={error}
          onBack={() => navigate("/warehouse/new-source")}
          onError={setError}
          onCreated={(detail) => navigate(`/warehouse/sources/${detail.id}`)}
        />
      </div>
    )
  }

  return (
    <Scene>
      <SceneHeader
        icon={<Warehouse className="size-5" />}
        title="New source"
        onBack={() => navigate(kind ? "/warehouse/new-source" : "/warehouse")}
      />
      <ErrorBanner error={error} />
      <div className="min-h-0 flex-1 overflow-y-auto p-6">
        {!kind ? (
          <SourceCatalogView
            onSelect={(s) => navigate(`/warehouse/new-source?kind=${encodeURIComponent(s.name)}`)}
            onError={setError}
          />
        ) : catalog ? (
          <p className="text-sm text-text-tertiary">
            Unknown source “{kind}”.{" "}
            <button
              type="button"
              className="text-primary hover:underline"
              onClick={() => navigate("/warehouse/new-source")}
            >
              Back to the catalog
            </button>
          </p>
        ) : (
          <p className="text-sm text-text-tertiary">Loading…</p>
        )}
      </div>
    </Scene>
  )
}

// GET /warehouse/sources/:id: a source's schemas, sync, and settings.
export function SourceDetailPage() {
  const navigate = useNavigate()
  const { id = "" } = useParams()
  const [error, setError] = useState<string | null>(null)

  return (
    <Scene>
      <SceneHeader
        icon={<Warehouse className="size-5" />}
        title="Source"
        onBack={() => navigate("/warehouse")}
      />
      <ErrorBanner error={error} />
      <div className="min-h-0 flex-1 overflow-y-auto p-6">
        <SourceDetailView
          sourceId={id}
          onError={setError}
          onDeleted={() => navigate("/warehouse")}
          onChanged={() => {}}
        />
      </div>
    </Scene>
  )
}

function SourceStatusBadge({ status }: { status: string }) {
  if (status === "syncing")
    return (
      <Badge variant="info">
        <Loader2 className="size-3 animate-spin" /> syncing
      </Badge>
    )
  if (status === "error") return <Badge variant="danger">error</Badge>
  return <Badge variant="neutral">idle</Badge>
}

// -- Source catalog -----------------------------------------------------------

const RELEASE_TAG: Record<string, { label: string; variant: "warning" | "info" } | null> = {
  alpha: { label: "Alpha", variant: "warning" },
  beta: { label: "Beta", variant: "info" },
  ga: null,
}

function SourceCatalogView({
  onSelect,
  onError,
}: {
  onSelect: (source: SourceConfig) => void
  onError: (message: string) => void
}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [search, setSearch] = useState("")
  const [category, setCategory] = useState("all")
  const [requesting, setRequesting] = useState(false)
  const [notified, setNotified] = useState<Set<string>>(new Set())

  useEffect(() => {
    getCatalog()
      .then(setCatalog)
      .catch((e) => onError(e instanceof Error ? e.message : String(e)))
  }, [onError])

  const notify = (source: SourceConfig) => {
    registerInterest(source.name).catch(() => {})
    setNotified((prev) => new Set(prev).add(source.name))
  }

  const categories = useMemo(() => {
    if (!catalog) return []
    const counts = new Map<string, number>()
    for (const s of catalog.sources) counts.set(s.category, (counts.get(s.category) ?? 0) + 1)
    const used = catalog.categories.filter((c) => counts.has(c))
    return [
      { key: "all", label: "All sources", count: catalog.sources.length },
      ...used.map((c) => ({ key: c, label: c, count: counts.get(c) ?? 0 })),
    ]
  }, [catalog])

  const filtered = useMemo(() => {
    if (!catalog) return []
    const q = search.trim().toLowerCase()
    return catalog.sources.filter(
      (s) =>
        (category === "all" || s.category === category) &&
        (q === "" || s.label.toLowerCase().includes(q) || s.caption.toLowerCase().includes(q)),
    )
  }, [catalog, search, category])

  if (!catalog) return <div className="text-sm text-text-tertiary">Loading sources…</div>

  return (
    <div className="flex flex-col gap-4 sm:flex-row">
      <div className="flex shrink-0 flex-row gap-1 overflow-x-auto sm:w-56 sm:flex-col">
        {categories.map((cat) => (
          <button
            key={cat.key}
            type="button"
            onClick={() => setCategory(cat.key)}
            className={`flex items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors ${
              category === cat.key ? "bg-accent font-medium" : "hover:bg-muted/60"
            }`}
          >
            <span className="truncate">{cat.label}</span>
            <span className="text-xs text-text-tertiary tabular-nums">{cat.count}</span>
          </button>
        ))}
      </div>

      <div className="flex flex-1 flex-col gap-4">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-text-tertiary" />
          <Input
            className="pl-8"
            placeholder="Search sources…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {filtered.length === 0 ? (
          <p className="text-sm text-text-tertiary">
            No sources match.{" "}
            <button
              type="button"
              className="text-primary hover:underline"
              onClick={() => {
                setSearch("")
                setCategory("all")
              }}
            >
              Clear filters
            </button>{" "}
            or request one below.
          </p>
        ) : null}

        <div className="grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-3">
          {filtered.map((source) => (
            <SourceTile
              key={source.name}
              source={source}
              notified={notified.has(source.name)}
              onSelect={onSelect}
              onNotify={notify}
            />
          ))}
          <RequestTile onRequest={() => setRequesting(true)} />
        </div>
      </div>

      <RequestSourceDialog
        open={requesting}
        onClose={() => setRequesting(false)}
        onError={onError}
      />
    </div>
  )
}

// A horizontal tile: icon left, name + release/coming-soon tag stacked right.
// Available sources are clickable; coming-soon ones offer "Notify me".
function SourceTile({
  source,
  notified,
  onSelect,
  onNotify,
}: {
  source: SourceConfig
  notified: boolean
  onSelect: (source: SourceConfig) => void
  onNotify: (source: SourceConfig) => void
}) {
  const tag = RELEASE_TAG[source.releaseStatus]
  const base =
    "flex min-h-[8.5rem] flex-row items-center gap-4 rounded-lg border border-border bg-card p-5"
  const icon = <SourceIcon type={source.name} size={48} />
  if (source.comingSoon) {
    return (
      <div className={base}>
        {icon}
        <div className="flex min-w-0 flex-col items-start gap-2">
          <span className="text-sm font-medium leading-tight">{source.label}</span>
          <Badge variant="warning">Coming soon</Badge>
          <Button
            size="xs"
            variant="secondary"
            disabled={notified}
            onClick={() => onNotify(source)}
          >
            <Megaphone className="size-3.5" /> {notified ? "We'll let you know" : "Notify me"}
          </Button>
        </div>
      </div>
    )
  }
  return (
    <button
      type="button"
      onClick={() => onSelect(source)}
      className={`${base} text-left transition-colors hover:border-primary`}
    >
      {icon}
      <div className="flex min-w-0 flex-col items-start gap-2">
        <span className="text-sm font-medium leading-tight">{source.label}</span>
        {tag ? <Badge variant={tag.variant}>{tag.label}</Badge> : null}
      </div>
    </button>
  )
}

function RequestTile({ onRequest }: { onRequest: () => void }) {
  return (
    <button
      type="button"
      onClick={onRequest}
      className="flex min-h-[8.5rem] flex-row items-center gap-4 rounded-lg border border-dashed border-border p-5 text-left transition-colors hover:border-primary"
    >
      <Plus className="size-7 shrink-0 text-text-tertiary" />
      <div className="flex min-w-0 flex-col items-start gap-1">
        <span className="text-sm font-medium leading-tight">Request a source</span>
        <span className="text-xs text-text-tertiary">Tell us what you'd like to connect</span>
      </div>
    </button>
  )
}

function RequestSourceDialog({
  open,
  onClose,
  onError,
}: {
  open: boolean
  onClose: () => void
  onError: (message: string) => void
}) {
  const [text, setText] = useState("")
  const [sent, setSent] = useState(false)
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Request a source</DialogTitle>
          <DialogDescription>
            Tell us which source you'd like to connect and we'll take it into account.
          </DialogDescription>
        </DialogHeader>
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Acme CRM: https://acme.com/developers/api"
          rows={3}
          autoFocus
        />
        <DialogFooter>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={!text.trim() || sent}
            onClick={() => {
              registerInterest(`request: ${text.trim()}`)
                .then(() => {
                  setSent(true)
                  onClose()
                })
                .catch((e) => onError(e instanceof Error ? e.message : String(e)))
            }}
          >
            Submit request
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
        checked ? "bg-primary" : "bg-muted"
      }`}
    >
      <span
        className={`inline-block size-4 rounded-full bg-white transition-transform ${
          checked ? "translate-x-4" : "translate-x-0.5"
        }`}
      />
    </button>
  )
}

// -- Source detail: schemas, sync, delete ------------------------------------

function SourceDetailView({
  sourceId,
  onError,
  onDeleted,
  onChanged,
}: {
  sourceId: string
  onError: (message: string) => void
  onDeleted: () => void
  onChanged: () => void
}) {
  const [detail, setDetail] = useState<SourceDetail | null>(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState(false)
  const [outcomes, setOutcomes] = useState<SyncOutcome[] | null>(null)

  const load = useCallback(() => {
    getSource(sourceId)
      .then(setDetail)
      .catch((e) => onError(e instanceof Error ? e.message : String(e)))
  }, [sourceId, onError])

  useEffect(load, [load])

  const sync = useCallback(async () => {
    setBusy(true)
    onError("")
    try {
      const result = await syncSource(sourceId)
      setDetail(result.source)
      setOutcomes(result.outcomes)
      onChanged()
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [sourceId, onChanged, onError])

  const patchSchema = useCallback(
    async (schema: SchemaView, patch: Parameters<typeof updateSchema>[1]) => {
      try {
        const updated = await updateSchema(schema.id, patch)
        setDetail((d) =>
          d ? { ...d, schemas: d.schemas.map((s) => (s.id === updated.id ? updated : s)) } : d,
        )
      } catch (e) {
        onError(e instanceof Error ? e.message : String(e))
      }
    },
    [onError],
  )

  if (!detail) return <div className="text-sm text-text-tertiary">Loading…</div>

  const enabled = detail.schemas.filter((s) => s.shouldSync).length

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex items-center gap-3">
        <SourceIcon type={detail.sourceType} size={40} />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">{detail.name}</h2>
            <Badge variant="neutral" className="font-mono">
              {detail.sourceType}
            </Badge>
            <SourceStatusBadge status={detail.status} />
          </div>
          <p className="mt-0.5 font-mono text-xs text-text-tertiary">
            {enabled} of {detail.schemas.length} tables enabled
            {detail.lastSyncedAt
              ? ` · last synced ${detail.lastSyncedAt.slice(0, 16).replace("T", " ")}`
              : " · never synced"}
          </p>
        </div>
        <Select
          value={detail.syncFrequency}
          onValueChange={async (f) => {
            try {
              setDetail(await setSyncFrequency(sourceId, f as SyncFrequency))
              onChanged()
            } catch (e) {
              onError(e instanceof Error ? e.message : String(e))
            }
          }}
        >
          <SelectTrigger className="h-8 w-40 text-xs" aria-label="Sync frequency">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {FREQUENCY_LABELS.map((f) => (
              <SelectItem key={f.value} value={f.value}>
                {f.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button size="sm" onClick={sync} disabled={busy || enabled === 0}>
          {busy ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <RefreshCw className="size-3.5" />
          )}
          {busy ? "Syncing…" : "Sync now"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          aria-label="Edit source"
          onClick={() => setEditing(true)}
        >
          <Pencil className="size-3.5" /> Edit
        </Button>
        <Button
          size="icon"
          variant="ghost"
          aria-label="Delete source"
          onClick={async () => {
            await deleteSource(sourceId)
            onDeleted()
          }}
        >
          <Trash2 className="size-4" />
        </Button>
      </div>

      <EditSourceDialog
        sourceId={sourceId}
        sourceType={detail.sourceType}
        open={editing}
        onOpenChange={setEditing}
        onSaved={load}
      />

      {detail.lastError ? (
        <div className="rounded-md border border-danger/30 bg-danger-tint px-3 py-2 font-mono text-xs text-danger">
          {detail.lastError}
        </div>
      ) : null}

      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-tertiary">
              <th className="px-4 py-2 font-medium">Sync</th>
              <th className="px-4 py-2 font-medium">Table</th>
              <th className="px-4 py-2 font-medium">Method</th>
              <th className="px-4 py-2 font-medium">Rows</th>
              <th className="px-4 py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {detail.schemas.map((schema) => (
              <SchemaRow key={schema.id} schema={schema} onPatch={patchSchema} />
            ))}
          </tbody>
        </table>
      </div>

      {outcomes ? (
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-2 text-sm font-semibold">Last sync</h3>
          <ul className="space-y-1 text-sm">
            {outcomes.map((o) => (
              <li key={o.table} className="flex items-center gap-2">
                {o.ok ? (
                  <Badge variant="success">ok</Badge>
                ) : (
                  <Badge variant="danger">failed</Badge>
                )}
                <span className="font-mono text-xs">{o.table}</span>
                <span className="text-text-tertiary">{o.ok ? `${o.rows} rows` : o.error}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <p className="text-xs text-text-tertiary">
        Synced tables are available across the app: query them in{" "}
        <span className="font-medium text-text-secondary">Explore</span>, read them in notebooks and
        chat, and use them in models, metrics, and dashboards, joinable like any bound dataset.
      </p>
    </div>
  )
}

function SchemaRow({
  schema,
  onPatch,
}: {
  schema: SchemaView
  onPatch: (schema: SchemaView, patch: Parameters<typeof updateSchema>[1]) => void
}) {
  const canIncrement = schema.incrementalFields.length > 0
  return (
    <tr className={schema.shouldSync ? "" : "opacity-50"}>
      <td className="px-4 py-2.5">
        <Toggle
          checked={schema.shouldSync}
          onChange={(should_sync) => onPatch(schema, { should_sync })}
        />
      </td>
      <td className="px-4 py-2.5">
        <div className="font-medium">{schema.name}</div>
        <div className="font-mono text-xs text-text-tertiary">{schema.table}</div>
      </td>
      <td className="px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Select
            value={schema.syncType}
            onValueChange={(sync_type) =>
              onPatch(schema, {
                sync_type: sync_type as "full_refresh" | "incremental",
                ...(sync_type === "incremental" && !schema.incrementalField
                  ? { incremental_field: schema.incrementalFields[0] }
                  : {}),
              })
            }
          >
            <SelectTrigger className="h-7 w-36 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="full_refresh">Full refresh</SelectItem>
              <SelectItem value="incremental" disabled={!canIncrement}>
                Incremental
              </SelectItem>
            </SelectContent>
          </Select>
          {schema.syncType === "incremental" && canIncrement ? (
            <Select
              value={schema.incrementalField ?? ""}
              onValueChange={(incremental_field) => onPatch(schema, { incremental_field })}
            >
              <SelectTrigger className="h-7 w-32 text-xs">
                <SelectValue placeholder="cursor" />
              </SelectTrigger>
              <SelectContent>
                {schema.incrementalFields.map((f) => (
                  <SelectItem key={f} value={f}>
                    {f}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : null}
        </div>
      </td>
      <td className="px-4 py-2.5 tabular-nums text-text-secondary">{schema.rowCount ?? EMPTY}</td>
      <td className="px-4 py-2.5">
        {schema.status === "synced" ? (
          <Badge variant="success">synced</Badge>
        ) : schema.status === "error" ? (
          <Badge variant="danger" title={schema.lastError ?? undefined}>
            error
          </Badge>
        ) : schema.status === "syncing" ? (
          <Badge variant="info">
            <Loader2 className="size-3 animate-spin" /> syncing
          </Badge>
        ) : (
          <Badge variant="neutral">pending</Badge>
        )}
      </td>
    </tr>
  )
}
