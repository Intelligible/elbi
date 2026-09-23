import {
  Database,
  Loader2,
  Megaphone,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Warehouse,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useNavigate, useParams, useSearchParams } from "react-router-dom"
import { EmptyState } from "@/components/app/EmptyState"
import { IconButton } from "@/components/app/IconButton"
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
import { Switch } from "@/components/ui/switch"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
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
  if (source.status === "error") return <Badge variant="danger">Error</Badge>
  if (source.status === "syncing") return <Badge variant="info">Running</Badge>
  if (source.syncedCount > 0) return <Badge variant="success">{source.syncedCount} synced</Badge>
  return <Badge variant="neutral">Not syncing</Badge>
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
      <div className="absolute inset-0 overflow-y-auto bg-panel text-sm text-foreground">
        <div className="px-5 pt-4 pb-12">
          <div className="mb-5 flex items-center justify-between gap-3">
            <h1 className="text-title font-semibold">Data warehouse</h1>
            <Button size="sm" onClick={() => navigate("/warehouse/new-source")}>
              <Plus className="size-4" /> New source
            </Button>
          </div>

          {error ? (
            <div className="mb-4 rounded-md border border-danger/30 bg-danger-tint px-3 py-2 text-compact text-danger">
              {error}
            </div>
          ) : null}

          <WarehouseSection
            title="Datasets"
            description={
              <>
                Every queryable table: bound datasets from the project's <code>elbi.yaml</code> and
                synced warehouse sources: available to every analysis and derivation.
              </>
            }
          >
            <TableFrame>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-0" />
                    <TableHead>Dataset</TableHead>
                    <TableHead className="text-right">Rows</TableHead>
                    <TableHead className="text-right">Columns</TableHead>
                    <TableHead>Origin</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {datasets === null ? (
                    <TableRow>
                      <TableCell colSpan={5}>
                        <div className="py-8 text-center text-sm text-text-secondary">Loading…</div>
                      </TableCell>
                    </TableRow>
                  ) : datasets.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5}>
                        <EmptyState title="No datasets bound in this project" className="py-8" />
                      </TableCell>
                    </TableRow>
                  ) : (
                    datasets.map((d) => (
                      <TableRow key={d.name}>
                        <TableCell>
                          <Database className="size-5.5 text-text-tertiary" />
                        </TableCell>
                        <TableCell className="font-semibold">{d.name}</TableCell>
                        <TableCell className="text-right">{d.rows.toLocaleString()}</TableCell>
                        <TableCell className="text-right">{d.columns}</TableCell>
                        <TableCell>
                          {d.origin === "synced" ? (
                            <Badge variant="info">Synced</Badge>
                          ) : (
                            <Badge variant="neutral">Bound</Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableFrame>
          </WarehouseSection>

          <WarehouseSection
            title="Managed data warehouse sources"
            description="Connect to external systems and sync their data into your warehouse. Synced tables become first-class data: usable in Explore, notebooks, chat, models, and metrics, just like a bound dataset."
          >
            <Input
              className="mb-2 max-w-80 bg-card"
              type="search"
              placeholder="Search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <TableFrame>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-0" />
                    <TableHead>Source</TableHead>
                    <TableHead>Table prefix</TableHead>
                    <TableHead>Last Successful Run</TableHead>
                    <TableHead className="text-right">Total Rows Synced</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="w-0" />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7}>
                        <EmptyState
                          title={q ? "No sources matching your search" : "No managed sources"}
                          className="py-8"
                          action={
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => navigate("/warehouse/new-source")}
                            >
                              <Plus className="size-4" /> New source
                            </Button>
                          }
                        />
                      </TableCell>
                    </TableRow>
                  ) : (
                    filtered.map((s) => (
                      <TableRow
                        key={s.id}
                        className="group cursor-pointer"
                        onClick={() => navigate(`/warehouse/sources/${s.id}`)}
                      >
                        <TableCell>
                          <SourceIcon type={s.sourceType} size={28} />
                        </TableCell>
                        <TableCell>
                          <div className="font-semibold group-hover:text-primary">{s.name}</div>
                          {s.description ? (
                            <div className="text-xs text-text-tertiary">{s.description}</div>
                          ) : null}
                        </TableCell>
                        <TableCell>{s.prefix || "-"}</TableCell>
                        <TableCell>{fmtRun(s.lastSyncedAt)}</TableCell>
                        <TableCell className="text-right">{s.rows.toLocaleString()}</TableCell>
                        <TableCell>
                          <StatusTag source={s} />
                        </TableCell>
                        <TableCell>
                          <IconButton
                            label="Delete source"
                            variant="outline"
                            onClick={(e) => remove(e, s.id)}
                          >
                            <Trash2 className="size-4" />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableFrame>
          </WarehouseSection>
        </div>
      </div>
    </div>
  )
}

// A titled block of the warehouse page, ruled off from the one above it.
function WarehouseSection({
  title,
  description,
  children,
}: {
  title: string
  description: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="mt-6 mb-1 border-t border-border pt-4">
      <h2 className="mb-1 text-base font-semibold text-text-tertiary">{title}</h2>
      <p className="mb-3 max-w-prose text-sm leading-normal text-text-secondary">{description}</p>
      {children}
    </section>
  )
}

function TableFrame({ children }: { children: React.ReactNode }) {
  return <div className="overflow-hidden rounded-md border border-border bg-card">{children}</div>
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
            <Button
              variant="link"
              className="h-auto p-0 font-normal"
              onClick={() => navigate("/warehouse/new-source")}
            >
              Back to the catalog
            </Button>
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

// The category filter keeps its list look (no pill track): the active row takes the accent
// fill, as the rest of the app's nav lists do.
const CATEGORY_TAB =
  "h-auto flex-none justify-between gap-2 border-0 px-2.5 py-1.5 font-normal text-foreground hover:bg-muted/60 max-sm:group-data-[orientation=vertical]/tabs:w-auto group-data-[orientation=vertical]/tabs:justify-between data-[state=active]:bg-accent data-[state=active]:font-medium group-data-[variant=default]/tabs-list:data-[state=active]:shadow-none dark:text-foreground dark:data-[state=active]:border-transparent dark:data-[state=active]:bg-accent"

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
      <Tabs
        value={category}
        onValueChange={setCategory}
        orientation="vertical"
        className="shrink-0 sm:w-56"
      >
        <TabsList className="w-full items-stretch justify-start gap-1 overflow-x-auto bg-transparent p-0 max-sm:group-data-[orientation=vertical]/tabs:flex-row">
          {categories.map((cat) => (
            <TabsTrigger key={cat.key} value={cat.key} className={CATEGORY_TAB}>
              <span className="truncate">{cat.label}</span>
              <span className="text-xs text-text-tertiary tabular-nums">{cat.count}</span>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

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
            <Button
              variant="link"
              className="h-auto p-0 font-normal"
              onClick={() => {
                setSearch("")
                setCategory("all")
              }}
            >
              Clear filters
            </Button>{" "}
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
    // eslint-disable-next-line ds/no-raw-element -- a whole card is the click target; Button's fixed heights and chrome don't fit a tile
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
    // eslint-disable-next-line ds/no-raw-element -- a whole card is the click target; Button's fixed heights and chrome don't fit a tile
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

      {detail.lastError ? (
        <div className="rounded-md border border-danger/30 bg-danger-tint px-3 py-2 font-mono text-xs text-danger">
          {detail.lastError}
        </div>
      ) : null}

      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="px-4">Sync</TableHead>
              <TableHead className="px-4">Table</TableHead>
              <TableHead className="px-4">Method</TableHead>
              <TableHead className="px-4">Rows</TableHead>
              <TableHead className="px-4">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {detail.schemas.map((schema) => (
              <SchemaRow key={schema.id} schema={schema} onPatch={patchSchema} />
            ))}
          </TableBody>
        </Table>
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
    <TableRow className={schema.shouldSync ? "" : "opacity-50"}>
      <TableCell className="px-4 py-2.5">
        <Switch
          aria-label={`Sync ${schema.name}`}
          checked={schema.shouldSync}
          onCheckedChange={(should_sync) => onPatch(schema, { should_sync })}
        />
      </TableCell>
      <TableCell className="px-4 py-2.5">
        <div className="font-medium">{schema.name}</div>
        <div className="font-mono text-xs text-text-tertiary">{schema.table}</div>
      </TableCell>
      <TableCell className="px-4 py-2.5">
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
      </TableCell>
      <TableCell className="px-4 py-2.5 tabular-nums text-text-secondary">
        {schema.rowCount ?? EMPTY}
      </TableCell>
      <TableCell className="px-4 py-2.5">
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
      </TableCell>
    </TableRow>
  )
}
