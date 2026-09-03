import {
  Activity,
  BadgeCheck,
  BarChart3,
  Code2,
  Copy,
  Download,
  Gauge,
  History,
  LineChart,
  Network,
  Play,
  Plus,
  Search,
  Sparkles,
  Table2,
  Trash2,
  TrendingDown,
  TrendingUp,
  Upload,
  X,
} from "lucide-react"
import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react"
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
import { VizView } from "@/components/viz/VizView"
import { useRowKeys } from "@/hooks/useRowKeys"
import { metricExportUrl } from "@/lib/chat"
import { type Graph, type GraphNode, getSubgraph, type NodeType } from "@/lib/lineage"
import {
  type Agg,
  type AskResult,
  askMetric,
  type Cell,
  compileMetric,
  defineMetric,
  deleteMetric,
  duplicateMetric,
  exportOsi,
  type FilterClause,
  type Format,
  type FormatKind,
  type Grain,
  importOsi,
  listDerivations,
  listMetrics,
  type Metric,
  type MetricOverview,
  type MetricQueryResult,
  type MetricType,
  type MetricVersion,
  metricHistory,
  metricsOverview,
  queryMetric,
} from "@/lib/metrics"
import { createMonitor } from "@/lib/monitors"
import { EMPTY, uuid } from "@/lib/utils"

const AGGS: Agg[] = ["sum", "average", "count", "count_distinct", "min", "max", "median"]
const GRAINS: Grain[] = ["hour", "day", "week", "month", "quarter", "year"]
const OPS: FilterClause["op"][] = ["eq", "ne", "lt", "le", "gt", "ge", "in"]
type ChartType = "line" | "bar" | "table"

/** Format a metric cell, honoring the metric's display format (Intl.NumberFormat). */
function fmt(value: Cell, format?: Format): string {
  if (value === null) return "∅"
  if (typeof value !== "number") return String(value)
  if (!format || format.kind === "duration") {
    if (format?.kind === "duration") return formatDuration(value)
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, { maximumFractionDigits: 4 })
  }
  const opts: Intl.NumberFormatOptions = { notation: format.notation ?? "standard" }
  if (typeof format.precision === "number") {
    opts.minimumFractionDigits = format.precision
    opts.maximumFractionDigits = format.precision
  }
  if (format.kind === "currency") {
    opts.style = "currency"
    opts.currency = format.currency || "USD"
  } else if (format.kind === "percent") {
    opts.style = "percent" // Intl multiplies by 100; a percent metric stores the fraction
  }
  return new Intl.NumberFormat(undefined, opts).format(value)
}

/** Render seconds as a compact ``1h 23m 4s`` duration (Intl has no duration style). */
function formatDuration(seconds: number): string {
  const s = Math.floor(Math.abs(seconds))
  const parts = [
    [Math.floor(s / 3600), "h"],
    [Math.floor((s % 3600) / 60), "m"],
    [s % 60, "s"],
  ] as const
  const shown = parts.filter(([n]) => n > 0).map(([n, u]) => `${n}${u}`)
  return (seconds < 0 ? "-" : "") + (shown.join(" ") || "0s")
}

/** A metric's own formula line, for the header. */
function formula(metric: Metric): string {
  switch (metric.type) {
    case "ratio":
      return `${metric.numerator} / ${metric.denominator}`
    case "derived":
      return metric.expr ?? "(expression)"
    case "cumulative":
      return `cumulative ${metric.inputMetric}${metric.window ? ` over ${metric.window}` : " to date"}`
    default:
      return `${metric.measure?.agg}(${metric.measure?.column ?? "*"}) of ${metric.source}`
  }
}

export function MetricsPage() {
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [asking, setAsking] = useState(false)
  const [search, setSearch] = useState("")
  const [certifiedOnly, setCertifiedOnly] = useState(false)
  const [overview, setOverview] = useState<Record<string, MetricOverview>>({})

  const refresh = useCallback(() => {
    listMetrics()
      .then((m) => setMetrics(m))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
    metricsOverview()
      .then((o) => setOverview(Object.fromEntries(o.map((e) => [e.name, e]))))
      .catch(() => setOverview({}))
  }, [])

  useEffect(refresh, [refresh])

  const current = metrics.find((m) => m.name === selected) ?? null

  const shown = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return metrics
      .filter((m) => (certifiedOnly ? m.sourceCertified !== false : true))
      .filter((m) =>
        needle
          ? `${m.label ?? ""} ${m.name} ${m.description ?? ""}`.toLowerCase().includes(needle)
          : true,
      )
      .sort((a, b) => (a.label || a.name).localeCompare(b.label || b.name))
  }, [metrics, search, certifiedOnly])

  const doExport = useCallback(async () => {
    const doc = await exportOsi()
    download("metrics.osi.json", JSON.stringify(doc, null, 2))
  }, [])

  const doImport = useCallback(async () => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = "application/json,.json,.yaml,.yml"
    input.onchange = async () => {
      const file = input.files?.[0]
      if (!file) return
      try {
        await importOsi(JSON.parse(await file.text()))
        refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
    }
    input.click()
  }, [refresh])

  return (
    <Scene>
      <SceneHeader
        icon={<Gauge className="size-5" />}
        title="Metrics"
        description="A verified semantic layer: define a business number once over a certified derivation, read the same figure everywhere."
        actions={
          <>
            <Button size="sm" variant="ghost" onClick={() => setAsking(true)}>
              <Sparkles className="size-3.5" /> Ask
            </Button>
            <Button size="sm" variant="ghost" onClick={doExport}>
              <Download className="size-3.5" /> Export OSI
            </Button>
            <Button size="sm" variant="ghost" onClick={doImport}>
              <Upload className="size-3.5" /> Import
            </Button>
            <Button size="sm" onClick={() => setCreating(true)}>
              <Plus className="size-4" /> New metric
            </Button>
          </>
        }
      />
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-72 shrink-0 flex-col border-r border-border">
          <div className="space-y-2 border-b border-border p-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
              <Input
                placeholder="Search metrics"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="h-8 pl-7 text-sm"
              />
            </div>
            <label className="flex cursor-pointer items-center gap-1.5 px-1 text-xs text-text-tertiary">
              <input
                type="checkbox"
                checked={certifiedOnly}
                onChange={(e) => setCertifiedOnly(e.target.checked)}
                className="accent-primary"
              />
              Certified only
            </label>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {shown.length === 0 ? (
              <p className="px-2 py-2 text-xs text-text-tertiary">
                {metrics.length === 0
                  ? "No metrics yet. Define one over a certified derivation."
                  : "No metrics match."}
              </p>
            ) : (
              shown.map((m) => (
                <button
                  key={m.name}
                  type="button"
                  onClick={() => setSelected(m.name)}
                  className={`flex w-full flex-col gap-1 rounded-md px-2 py-1.5 text-left transition-colors ${
                    selected === m.name ? "bg-accent" : "hover:bg-muted/60"
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    <span className="flex-1 truncate text-sm">{m.label || m.name}</span>
                    {m.sourceCertified !== false ? (
                      <BadgeCheck className="size-3.5 shrink-0 text-verified" />
                    ) : null}
                    <Badge variant="neutral" className="shrink-0">
                      {m.type}
                    </Badge>
                  </span>
                  <span className="flex items-center gap-2">
                    {overview[m.name]?.value != null ? (
                      <span className="font-mono text-xs tabular-nums text-foreground">
                        {fmt(overview[m.name].value, m.format)}
                      </span>
                    ) : (
                      <span className="text-xs text-text-tertiary">{EMPTY}</span>
                    )}
                    <Sparkline points={overview[m.name]?.series ?? []} />
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        <div className="flex min-h-0 flex-1 flex-col">
          {error ? (
            <div className="flex items-center gap-2 border-b border-danger/30 bg-danger-tint px-4 py-2 font-mono text-xs text-danger">
              <span className="flex-1">{error}</span>
              <button type="button" aria-label="Dismiss" onClick={() => setError(null)}>
                <X className="size-3.5" />
              </button>
            </div>
          ) : null}
          {current ? (
            <MetricPanel
              key={current.name}
              metric={current}
              onDelete={async () => {
                await deleteMetric(current.name)
                setSelected(null)
                refresh()
              }}
              onDuplicate={async () => {
                try {
                  const copy = await duplicateMetric(current.name)
                  refresh()
                  // Select the copy: there is no per-metric route, so this is the
                  // only way the new definition becomes the one on screen.
                  setSelected(copy.name)
                } catch (e) {
                  setError(e instanceof Error ? e.message : "could not duplicate")
                }
              }}
              onError={setError}
            />
          ) : (
            <div className="grid flex-1 place-items-center text-sm text-text-tertiary">
              Select a metric, or define a new one.
            </div>
          )}
        </div>
      </div>

      {creating ? (
        <NewMetricDialog
          existing={metrics}
          onClose={() => setCreating(false)}
          onCreated={(name) => {
            setCreating(false)
            setSelected(name)
            refresh()
          }}
          onError={setError}
        />
      ) : null}

      {asking ? (
        <AskDialog
          metrics={metrics}
          onClose={() => setAsking(false)}
          onOpen={(name) => {
            setAsking(false)
            setSelected(name)
          }}
        />
      ) : null}
    </Scene>
  )
}

/** A tiny inline-SVG trend line for a catalog row (no chart dependency). */
function Sparkline({ points }: { points: { t: string; v: number }[] }) {
  const values = points.map((p) => p.v)
  if (values.length < 2) return <span className="h-4 flex-1" />
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const w = 64
  const h = 16
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w
      const y = h - ((v - min) / span) * h
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
  const rising = values[values.length - 1] >= values[0]
  return (
    <svg width={w} height={h} className="flex-1 overflow-visible" aria-hidden>
      <path
        d={path}
        fill="none"
        stroke={rising ? "var(--verified)" : "var(--danger)"}
        strokeWidth={1.25}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}

/** Trigger a client-side file download of ``text`` named ``filename``. */
function download(filename: string, text: string): void {
  const blob = new Blob([text], { type: "application/octet-stream" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function MetricPanel({
  metric,
  onDelete,
  onDuplicate,
  onError,
}: {
  metric: Metric
  onDelete: () => void
  onDuplicate: () => void
  onError: (message: string) => void
}) {
  const dims = useMemo(() => {
    const names = [...(metric.dimensions ?? [])]
    if (metric.timeDimension) names.push(metric.timeDimension.column)
    return names
  }, [metric])
  const [groupBy, setGroupBy] = useState<string[]>([])
  const [grain, setGrain] = useState<Grain | undefined>(metric.timeDimension?.grain)
  const [filters, setFilters] = useState<FilterClause[]>([])
  const [chart, setChart] = useState<ChartType>("bar")
  const [result, setResult] = useState<MetricQueryResult | null>(null)
  const [running, setRunning] = useState(false)
  const [watching, setWatching] = useState(false)

  const timeColumn = metric.timeDimension?.column

  const run = useCallback(async () => {
    setRunning(true)
    try {
      const clean = filters.filter((f) => f.column && f.value !== "")
      setResult(await queryMetric(metric.name, { groupBy, grain, filters: clean }))
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }, [metric.name, groupBy, grain, filters, onError])

  const toggle = (dim: string) => {
    setGroupBy((prev) => {
      const next = prev.includes(dim) ? prev.filter((d) => d !== dim) : [...prev, dim]
      // A time breakdown reads as a trend; a categorical one as bars.
      setChart(timeColumn && next.includes(timeColumn) ? "line" : "bar")
      return next
    })
  }

  const exportCsv = useCallback(() => {
    if (!result) return
    const head = result.columns.join(",")
    const body = result.rows
      .map((r) => result.columns.map((c) => csvCell(r[c])).join(","))
      .join("\n")
    download(`${metric.name}.csv`, `${head}\n${body}`)
  }, [result, metric.name])

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-border px-6 py-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">{metric.label || metric.name}</h2>
          {metric.sourceCertified !== false ? (
            <Badge variant="verified">
              <BadgeCheck className="size-3" /> Certified
            </Badge>
          ) : (
            <Badge variant="warning">source not certified</Badge>
          )}
          <Badge variant="neutral">{metric.type}</Badge>
          <div className="flex-1" />
          <Button size="sm" variant="ghost" onClick={() => setWatching(true)}>
            <Activity className="size-3.5" /> Watch
          </Button>
          <Button size="sm" variant="ghost" aria-label="Duplicate metric" onClick={onDuplicate}>
            <Copy className="size-3.5" />
          </Button>
          {/* Not the exportCsv helper below: that serializes rows this page already
              holds, while this is a server document (manifest + version history) the
              page never fetches. A link streams it without loading it into memory. */}
          <Button size="sm" variant="ghost" asChild>
            <a
              href={metricExportUrl(metric.name)}
              download
              aria-label={`Export the record for ${metric.name}: manifest and version history`}
            >
              <Download className="size-3.5" /> Export record
            </a>
          </Button>
          <Button size="sm" variant="ghost" aria-label="Delete metric" onClick={onDelete}>
            <Trash2 className="size-3.5" />
          </Button>
        </div>
        {metric.description ? (
          <p className="mt-1 text-sm text-text-secondary">{metric.description}</p>
        ) : null}
        <p className="mt-1 font-mono text-xs text-text-tertiary">{formula(metric)}</p>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <Overview metric={metric} onError={onError} />

        <div className="border-t border-border px-6 py-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-text-tertiary">Group by</span>
            {dims.length === 0 ? (
              <span className="text-xs text-text-tertiary">(no dimensions)</span>
            ) : (
              dims.map((dim) => (
                <button
                  key={dim}
                  type="button"
                  onClick={() => toggle(dim)}
                  className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
                    groupBy.includes(dim)
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-text-tertiary hover:text-foreground"
                  }`}
                >
                  {dim}
                </button>
              ))
            )}
            {timeColumn && groupBy.includes(timeColumn) ? (
              <Select value={grain} onValueChange={(g) => setGrain(g as Grain)}>
                <SelectTrigger className="h-7 w-28 text-xs">
                  <SelectValue placeholder="grain" />
                </SelectTrigger>
                <SelectContent>
                  {GRAINS.map((g) => (
                    <SelectItem key={g} value={g}>
                      {g}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : null}
          </div>

          <FilterBuilder columns={dims} filters={filters} setFilters={setFilters} />

          <div className="mt-3 flex items-center gap-2">
            <ChartToggle chart={chart} setChart={setChart} />
            <div className="flex-1" />
            <Button
              size="sm"
              variant="ghost"
              onClick={exportCsv}
              disabled={!result}
              aria-label="Export CSV"
            >
              <Download className="size-3.5" /> CSV
            </Button>
            <Button size="sm" onClick={run} disabled={running}>
              <Play className="size-3.5" />
              {running ? "Running…" : "Run"}
            </Button>
          </div>

          <div className="mt-4">
            <MetricResult metric={metric} result={result} groupBy={groupBy} chart={chart} />
          </div>
        </div>

        <LineageSection metric={metric} />
        <DefinitionSection metric={metric} groupBy={groupBy} grain={grain} onError={onError} />
        <HistorySection metric={metric} />
      </div>

      {watching ? (
        <WatchDialog metric={metric} onClose={() => setWatching(false)} onError={onError} />
      ) : null}
    </div>
  )
}

/** The current value, its period-over-period change, and a trend: auto-run on select. */
function Overview({ metric, onError }: { metric: Metric; onError: (message: string) => void }) {
  const timeColumn = metric.timeDimension?.column
  const [result, setResult] = useState<MetricQueryResult | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let live = true
    setLoading(true)
    queryMetric(metric.name, {
      groupBy: timeColumn ? [timeColumn] : [],
      grain: metric.timeDimension?.grain,
    })
      .then((r) => live && setResult(r))
      .catch((e) => live && onError(e instanceof Error ? e.message : String(e)))
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [metric.name, timeColumn, metric.timeDimension?.grain, onError])

  const series = useMemo(() => {
    if (!result || !timeColumn) return []
    return [...result.rows]
      .filter((r) => typeof r[metric.name] === "number")
      .sort((a, b) => String(a[timeColumn]).localeCompare(String(b[timeColumn])))
  }, [result, timeColumn, metric.name])

  if (loading) {
    return <div className="px-6 py-6 text-sm text-text-tertiary">Loading current value…</div>
  }
  if (!result || result.rows.length === 0) return null

  const values = (timeColumn ? series : result.rows).map((r) => r[metric.name])
  const latest = values[values.length - 1]
  const prior = values.length > 1 ? values[values.length - 2] : null
  const change =
    typeof latest === "number" && typeof prior === "number" && prior !== 0
      ? ((latest - prior) / Math.abs(prior)) * 100
      : null

  return (
    <div className="flex flex-wrap items-end gap-8 px-6 py-6">
      <div>
        <div className="text-3xl font-semibold tabular-nums">
          {fmt(latest ?? null, metric.format)}
        </div>
        <div className="mt-1 flex items-center gap-2 text-xs text-text-tertiary">
          <span>
            current{timeColumn ? ` (latest ${metric.timeDimension?.grain ?? "period"})` : ""}
          </span>
          {change !== null ? (
            <span
              className={`inline-flex items-center gap-0.5 font-medium ${
                change >= 0 ? "text-verified" : "text-danger"
              }`}
            >
              {change >= 0 ? (
                <TrendingUp className="size-3" />
              ) : (
                <TrendingDown className="size-3" />
              )}
              {change >= 0 ? "+" : ""}
              {change.toFixed(1)}% vs prior
            </span>
          ) : null}
        </div>
      </div>
      {timeColumn && series.length > 1 ? (
        <div className="min-w-0 flex-1">
          <VizView
            viz={{
              spec: {
                mark: { type: "line", point: true, tooltip: true },
                encoding: {
                  x: { field: timeColumn, type: "temporal" },
                  y: { field: metric.name, type: "quantitative" },
                },
              },
              rows: series as Record<string, string | number>[],
            }}
            height={140}
          />
        </div>
      ) : null}
    </div>
  )
}

/** One editable row: a clause plus the identity React needs to track its inputs.
 *
 *  The id is the editor's, not the metric's -- `FilterClause` is what goes to the API,
 *  and a key invented for rendering has no business travelling with it. Rows are keyed
 *  by it rather than by position, because removing a row shifts every position after it
 *  and React would then reuse the removed row's inputs for its successor, carrying the
 *  caret and any in-flight IME composition to the wrong filter. */
type FilterRow = { id: string; clause: FilterClause }

function FilterBuilder({
  columns,
  filters,
  setFilters,
}: {
  columns: string[]
  filters: FilterClause[]
  setFilters: (f: FilterClause[]) => void
}) {
  // Identity lives beside the clauses for as long as the editor is open. Re-keyed from
  // the incoming list only when it changes length underneath us (a reset), so typing
  // never re-mints an id and never remounts a row.
  const [rows, setRows] = useState<FilterRow[]>(() =>
    filters.map((clause) => ({ id: uuid(), clause })),
  )
  useEffect(() => {
    setRows((current) =>
      current.length === filters.length
        ? current.map((r, i) => ({ ...r, clause: filters[i] }))
        : filters.map((clause) => ({ id: uuid(), clause })),
    )
  }, [filters])

  const commit = (next: FilterRow[]) => {
    setRows(next)
    setFilters(next.map((r) => r.clause))
  }
  const add = () =>
    commit([...rows, { id: uuid(), clause: { column: columns[0] ?? "", op: "eq", value: "" } }])
  const update = (id: string, patch: Partial<FilterClause>) =>
    commit(rows.map((r) => (r.id === id ? { ...r, clause: { ...r.clause, ...patch } } : r)))
  const remove = (id: string) => commit(rows.filter((r) => r.id !== id))

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-text-tertiary">Filter</span>
        <Button size="sm" variant="ghost" className="h-6 px-1.5 text-xs" onClick={add}>
          <Plus className="size-3" /> Add
        </Button>
      </div>
      {rows.map(({ id, clause: f }) => (
        <div key={id} className="flex items-center gap-2">
          <Select value={f.column} onValueChange={(v) => update(id, { column: v })}>
            <SelectTrigger className="h-7 w-40 text-xs">
              <SelectValue placeholder="column" />
            </SelectTrigger>
            <SelectContent>
              {columns.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={f.op} onValueChange={(v) => update(id, { op: v as FilterClause["op"] })}>
            <SelectTrigger className="h-7 w-20 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {OPS.map((op) => (
                <SelectItem key={op} value={op}>
                  {op}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            placeholder={f.op === "in" ? "a, b, c" : "value"}
            value={String(f.value ?? "")}
            onChange={(e) =>
              update(id, {
                value:
                  f.op === "in" ? e.target.value.split(",").map((s) => s.trim()) : e.target.value,
              })
            }
            className="h-7 w-40 text-xs"
          />
          <button type="button" aria-label="Remove filter" onClick={() => remove(id)}>
            <X className="size-3.5 text-text-tertiary hover:text-foreground" />
          </button>
        </div>
      ))}
    </div>
  )
}

function ChartToggle({ chart, setChart }: { chart: ChartType; setChart: (c: ChartType) => void }) {
  const options: { key: ChartType; icon: typeof LineChart; label: string }[] = [
    { key: "line", icon: LineChart, label: "Line" },
    { key: "bar", icon: BarChart3, label: "Bar" },
    { key: "table", icon: Table2, label: "Table" },
  ]
  return (
    <div className="inline-flex rounded-md border border-border">
      {options.map(({ key, icon: Icon, label }) => (
        <button
          key={key}
          type="button"
          aria-label={label}
          onClick={() => setChart(key)}
          className={`flex items-center gap-1 px-2 py-1 text-xs transition-colors first:rounded-l-md last:rounded-r-md ${
            chart === key ? "bg-accent text-foreground" : "text-text-tertiary hover:text-foreground"
          }`}
        >
          <Icon className="size-3.5" />
        </button>
      ))}
    </div>
  )
}

function MetricResult({
  metric,
  result,
  groupBy,
  chart,
}: {
  metric: Metric
  result: MetricQueryResult | null
  groupBy: string[]
  chart: ChartType
}) {
  const rowKey = useRowKeys()
  if (!result) {
    return (
      <div className="grid h-full min-h-40 place-items-center text-sm text-text-tertiary">
        Group by a dimension and run to break the metric down.
      </div>
    )
  }
  const rows = result.rows as Record<string, string | number>[]
  const timeColumn = metric.timeDimension?.column
  const showChart = chart !== "table" && groupBy.length > 0 && rows.length > 0
  const xField = groupBy[0]
  const xType = timeColumn && xField === timeColumn ? "temporal" : "nominal"

  return (
    <div>
      {showChart ? (
        <div className="mb-4">
          <VizView
            viz={{
              spec: {
                mark: { type: chart, tooltip: true, ...(chart === "line" ? { point: true } : {}) },
                encoding: {
                  x: { field: xField, type: xType },
                  y: { field: metric.name, type: "quantitative" },
                  ...(groupBy.length > 1 ? { color: { field: groupBy[1], type: "nominal" } } : {}),
                },
              },
              rows,
            }}
            height={300}
          />
        </div>
      ) : null}
      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <table className="w-full border-collapse text-sm tabular-nums">
          <thead className="bg-surface-secondary text-left">
            <tr>
              {result.columns.map((c) => (
                <th
                  key={c}
                  className="border-b border-border px-3 py-2 text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-text-tertiary"
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row) => (
              <tr
                key={rowKey(row)}
                className="border-t border-border transition-colors hover:bg-muted/60"
              >
                {result.columns.map((c) => (
                  <td key={c} className="px-3 py-1.5 font-mono text-xs text-foreground">
                    {fmt(row[c], c === metric.name ? metric.format : undefined)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** Quote a CSV cell when it contains a comma, quote, or newline. */
function csvCell(value: Cell): string {
  const s = value === null ? "" : String(value)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** A collapsible panel section with an icon heading. */
function Section({
  icon: Icon,
  title,
  children,
  open = false,
}: {
  icon: typeof Network
  title: string
  children: ReactNode
  open?: boolean
}) {
  return (
    <details open={open} className="border-t border-border px-6 py-4">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-semibold uppercase tracking-[0.04em] text-text-tertiary">
        <Icon className="size-3.5" />
        {title}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  )
}

const UPSTREAM: NodeType[] = ["dataset", "derivation", "semantic_model"]

/** Upstream sources and downstream consumers of a metric, from the lineage graph. */
function LineageSection({ metric }: { metric: Metric }) {
  const [graph, setGraph] = useState<Graph | null>(null)
  useEffect(() => {
    let live = true
    getSubgraph(`metric:${metric.name}`)
      .then((g) => live && setGraph(g))
      .catch(() => live && setGraph(null))
    return () => {
      live = false
    }
  }, [metric.name])

  const others = (graph?.nodes ?? []).filter((n) => n.id !== `metric:${metric.name}`)
  const upstream = others.filter((n) => UPSTREAM.includes(n.type))
  const downstream = others.filter((n) => !UPSTREAM.includes(n.type))

  return (
    <Section icon={Network} title="Lineage">
      <div className="grid grid-cols-2 gap-6 text-sm">
        <div>
          <div className="mb-1.5 text-xs text-text-tertiary">Feeds from</div>
          <NodeList nodes={upstream} empty="no upstream sources" />
        </div>
        <div>
          <div className="mb-1.5 text-xs text-text-tertiary">Used by</div>
          <NodeList nodes={downstream} empty="not used downstream yet" />
        </div>
      </div>
    </Section>
  )
}

function NodeList({ nodes, empty }: { nodes: GraphNode[]; empty: string }) {
  if (nodes.length === 0) {
    return <p className="text-xs text-text-tertiary">{empty}</p>
  }
  return (
    <ul className="space-y-1">
      {nodes.map((n) => (
        <li key={n.id} className="flex items-center gap-1.5">
          <Badge variant="neutral" className="shrink-0">
            {n.type}
          </Badge>
          <span className="truncate">{n.name}</span>
        </li>
      ))}
    </ul>
  )
}

/** The metric's declarative definition (JSON) and its on-demand compiled SQL. */
function DefinitionSection({
  metric,
  groupBy,
  grain,
  onError,
}: {
  metric: Metric
  groupBy: string[]
  grain?: Grain
  onError: (message: string) => void
}) {
  const [sql, setSql] = useState<string | null>(null)
  const showSql = useCallback(async () => {
    try {
      const { sql: compiled } = await compileMetric(metric.name, { groupBy, grain })
      setSql(compiled)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    }
  }, [metric.name, groupBy, grain, onError])

  const definition = { ...metric }
  delete definition.sourceCertified

  return (
    <Section icon={Code2} title="Definition">
      <pre className="overflow-x-auto rounded-lg border border-border bg-surface-secondary p-3 font-mono text-xs text-foreground">
        {JSON.stringify(definition, null, 2)}
      </pre>
      <div className="mt-3">
        <Button size="sm" variant="ghost" onClick={showSql}>
          <Code2 className="size-3.5" /> {sql ? "Refresh" : "Show"} compiled SQL
        </Button>
        {sql ? (
          <pre className="mt-2 overflow-x-auto rounded-lg border border-border bg-surface-secondary p-3 font-mono text-xs text-foreground">
            {sql}
          </pre>
        ) : null}
      </div>
    </Section>
  )
}

/** The metric's definition change log (append-on-change versions). */
function HistorySection({ metric }: { metric: Metric }) {
  const [versions, setVersions] = useState<MetricVersion[]>([])
  useEffect(() => {
    let live = true
    metricHistory(metric.name)
      .then((v) => live && setVersions(v))
      .catch(() => live && setVersions([]))
    return () => {
      live = false
    }
  }, [metric.name])

  if (versions.length === 0) return null
  return (
    <Section icon={History} title={`History (${versions.length})`}>
      <ul className="space-y-1.5 text-sm">
        {versions.map((v) => (
          <li key={v.version} className="flex items-center gap-2">
            <span className="font-mono text-xs text-text-tertiary">v{v.version}</span>
            {v.verdict === "sound" ? <BadgeCheck className="size-3.5 text-verified" /> : null}
            <span className="text-text-secondary">{v.author}</span>
            <span className="flex-1" />
            <span className="text-xs text-text-tertiary">
              {new Date(v.createdAt).toLocaleString()}
            </span>
          </li>
        ))}
      </ul>
    </Section>
  )
}

function WatchDialog({
  metric,
  onClose,
  onError,
}: {
  metric: Metric
  onClose: () => void
  onError: (message: string) => void
}) {
  const [method, setMethod] = useState<"mad" | "zscore">("mad")
  const [sensitivity, setSensitivity] = useState("3")
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)

  const submit = useCallback(async () => {
    setBusy(true)
    try {
      await createMonitor({
        name: `${metric.name}_watch`,
        targetKind: "metric",
        target: metric.name,
        method,
        sensitivity: Number(sensitivity) || 3,
      })
      setDone(true)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [metric.name, method, sensitivity, onError])

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Watch {metric.label || metric.name}</DialogTitle>
          <DialogDescription>
            Create a monitor that snapshots this metric on a schedule and alerts on an anomaly
            against a learned baseline; the alert carries the oracle verdict.
          </DialogDescription>
        </DialogHeader>
        {done ? (
          <p className="text-sm text-verified">Monitor created. Find it on the Monitors tab.</p>
        ) : (
          <div className="flex items-center gap-2">
            <Select value={method} onValueChange={(v) => setMethod(v as "mad" | "zscore")}>
              <SelectTrigger className="w-40 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mad">MAD (robust)</SelectItem>
                <SelectItem value="zscore">Z-score</SelectItem>
              </SelectContent>
            </Select>
            <Input
              type="number"
              step="0.5"
              value={sensitivity}
              onChange={(e) => setSensitivity(e.target.value)}
              className="w-28"
              placeholder="sensitivity"
            />
          </div>
        )}
        <DialogFooter>
          {done ? (
            <Button onClick={onClose}>Done</Button>
          ) : (
            <Button disabled={busy} onClick={submit}>
              {busy ? "Creating…" : "Create monitor"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function AskDialog({
  metrics,
  onClose,
  onOpen,
}: {
  metrics: Metric[]
  onClose: () => void
  onOpen: (name: string) => void
}) {
  const rowKey = useRowKeys()
  const [question, setQuestion] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AskResult | null>(null)

  const ask = useCallback(async () => {
    if (!question.trim()) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await askMetric(question.trim()))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [question])

  const rq = result?.resolvedQuery
  const format = metrics.find((m) => m.name === rq?.metricName)?.format

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Ask a metric</DialogTitle>
          <DialogDescription>
            Ask in plain English. The model picks a governed metric and its dimensions/filters (it
            never writes SQL), and the verified resolver runs it.
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-2">
          <Input
            autoFocus
            placeholder="e.g. monthly revenue by region for the west"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <Button disabled={busy || !question.trim()} onClick={ask}>
            {busy ? "Asking…" : "Ask"}
          </Button>
        </div>
        {error ? <p className="font-mono text-xs text-danger">{error}</p> : null}
        {rq ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <button
                type="button"
                onClick={() => onOpen(rq.metricName)}
                className="rounded-full border border-primary bg-primary/10 px-2 py-0.5 text-primary"
              >
                {rq.metricName}
              </button>
              {rq.groupBy.map((d) => (
                <span key={d} className="rounded-full border border-border px-2 py-0.5">
                  by {d}
                </span>
              ))}
              {rq.grain ? (
                <span className="rounded-full border border-border px-2 py-0.5">{rq.grain}</span>
              ) : null}
              {rq.filters.map((f) => (
                <span
                  key={`${f.column}\u0000${f.op}\u0000${String(f.value)}`}
                  className="rounded-full border border-border px-2 py-0.5"
                >
                  {f.column} {f.op} {String(f.value)}
                </span>
              ))}
            </div>
            {result?.explanation ? (
              <p className="text-sm text-text-secondary">{result.explanation}</p>
            ) : null}
            {result ? (
              <div className="max-h-64 overflow-auto rounded-lg border border-border bg-card">
                <table className="w-full border-collapse text-sm tabular-nums">
                  <thead className="bg-surface-secondary text-left">
                    <tr>
                      {result.columns.map((c) => (
                        <th
                          key={c}
                          className="border-b border-border px-3 py-1.5 text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-text-tertiary"
                        >
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.map((row) => (
                      <tr key={rowKey(row)} className="border-t border-border">
                        {result.columns.map((c) => (
                          <td key={c} className="px-3 py-1 font-mono text-xs">
                            {fmt(row[c], c === rq.metricName ? format : undefined)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}

function NewMetricDialog({
  existing,
  onClose,
  onCreated,
  onError,
}: {
  existing: Metric[]
  onClose: () => void
  onCreated: (name: string) => void
  onError: (message: string) => void
}) {
  const [sources, setSources] = useState<string[]>([])
  const [type, setType] = useState<MetricType>("simple")
  const [name, setName] = useState("")
  const [source, setSource] = useState("")
  const [agg, setAgg] = useState<Agg>("sum")
  const [column, setColumn] = useState("")
  const [dimensions, setDimensions] = useState("")
  const [timeColumn, setTimeColumn] = useState("")
  const [numerator, setNumerator] = useState("")
  const [denominator, setDenominator] = useState("")
  const [expr, setExpr] = useState("")
  const [inputMetrics, setInputMetrics] = useState("")
  const [inputMetric, setInputMetric] = useState("")
  const [windowSize, setWindowSize] = useState("")
  const [description, setDescription] = useState("")
  const [formatKind, setFormatKind] = useState<FormatKind>("number")
  const [currency, setCurrency] = useState("USD")
  const [precision, setPrecision] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    listDerivations()
      .then((d) => setSources(d.map((x) => x.name)))
      .catch(() => setSources([]))
  }, [])

  const submit = useCallback(async () => {
    setBusy(true)
    const dims = dimensions
      .split(",")
      .map((d) => d.trim())
      .filter(Boolean)
    const inputs = inputMetrics
      .split(",")
      .map((m) => m.trim())
      .filter(Boolean)
    const desc = description.trim() || undefined
    const hasPrecision = precision.trim() !== ""
    const format: Format | undefined =
      formatKind === "number" && !hasPrecision
        ? undefined
        : {
            kind: formatKind,
            ...(formatKind === "currency" ? { currency: currency || "USD" } : {}),
            ...(hasPrecision ? { precision: Number(precision) } : {}),
          }
    let metric: Metric
    if (type === "ratio") {
      metric = { name, type, numerator, denominator, dimensions: dims, description: desc, format }
    } else if (type === "derived") {
      metric = {
        name,
        type,
        expr,
        inputMetrics: inputs,
        dimensions: dims,
        description: desc,
        format,
      }
    } else if (type === "cumulative") {
      metric = {
        name,
        type,
        inputMetric,
        window: windowSize ? Number(windowSize) : undefined,
        description: desc,
        format,
      }
    } else {
      metric = {
        name,
        type,
        source,
        measure: { agg, column: agg === "count" ? undefined : column },
        dimensions: dims,
        timeDimension: timeColumn ? { column: timeColumn } : undefined,
        description: desc,
        format,
      }
    }
    try {
      await defineMetric(metric)
      onCreated(name)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [
    type,
    name,
    source,
    agg,
    column,
    dimensions,
    timeColumn,
    numerator,
    denominator,
    expr,
    inputMetrics,
    inputMetric,
    windowSize,
    description,
    formatKind,
    currency,
    precision,
    onCreated,
    onError,
  ])

  const metricNames = existing.map((m) => m.name)

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Define a metric</DialogTitle>
          <DialogDescription>
            A metric aggregates a certified derivation. Defined once, it serves the same number
            everywhere.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <Input
            placeholder="metric_name (lowercase, underscores)"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Input
            placeholder="description (optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <div className="flex gap-2">
            <Select value={formatKind} onValueChange={(v) => setFormatKind(v as FormatKind)}>
              <SelectTrigger className="w-40 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="number">number</SelectItem>
                <SelectItem value="currency">currency</SelectItem>
                <SelectItem value="percent">percent</SelectItem>
                <SelectItem value="duration">duration</SelectItem>
              </SelectContent>
            </Select>
            {formatKind === "currency" ? (
              <Input
                placeholder="USD"
                value={currency}
                onChange={(e) => setCurrency(e.target.value.toUpperCase())}
                className="w-24"
              />
            ) : null}
            <Input
              type="number"
              placeholder="decimals"
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="w-28"
            />
          </div>
          <Select value={type} onValueChange={(v) => setType(v as MetricType)}>
            <SelectTrigger className="text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="simple">simple</SelectItem>
              <SelectItem value="ratio">ratio</SelectItem>
              <SelectItem value="derived">derived</SelectItem>
              <SelectItem value="cumulative">cumulative</SelectItem>
            </SelectContent>
          </Select>

          {type === "simple" ? (
            <>
              <Select value={source} onValueChange={setSource}>
                <SelectTrigger className="text-sm">
                  <SelectValue placeholder="Source derivation (certified)" />
                </SelectTrigger>
                <SelectContent>
                  {sources.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="flex gap-2">
                <Select value={agg} onValueChange={(v) => setAgg(v as Agg)}>
                  <SelectTrigger className="w-40 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {AGGS.map((a) => (
                      <SelectItem key={a} value={a}>
                        {a}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input
                  placeholder={agg === "count" ? "(no column)" : "column"}
                  value={column}
                  disabled={agg === "count"}
                  onChange={(e) => setColumn(e.target.value)}
                />
              </div>
              <Input
                placeholder="dimensions, comma-separated"
                value={dimensions}
                onChange={(e) => setDimensions(e.target.value)}
              />
              <Input
                placeholder="time column (optional)"
                value={timeColumn}
                onChange={(e) => setTimeColumn(e.target.value)}
              />
            </>
          ) : type === "ratio" ? (
            <>
              <Select value={numerator} onValueChange={setNumerator}>
                <SelectTrigger className="text-sm">
                  <SelectValue placeholder="Numerator metric" />
                </SelectTrigger>
                <SelectContent>
                  {metricNames.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={denominator} onValueChange={setDenominator}>
                <SelectTrigger className="text-sm">
                  <SelectValue placeholder="Denominator metric" />
                </SelectTrigger>
                <SelectContent>
                  {metricNames.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Input
                placeholder="dimensions, comma-separated"
                value={dimensions}
                onChange={(e) => setDimensions(e.target.value)}
              />
            </>
          ) : type === "derived" ? (
            <>
              <Input
                placeholder="expression, e.g. revenue / order_count"
                value={expr}
                onChange={(e) => setExpr(e.target.value)}
              />
              <Input
                placeholder="input metrics, comma-separated"
                value={inputMetrics}
                onChange={(e) => setInputMetrics(e.target.value)}
              />
              <Input
                placeholder="dimensions, comma-separated"
                value={dimensions}
                onChange={(e) => setDimensions(e.target.value)}
              />
            </>
          ) : (
            <>
              <Select value={inputMetric} onValueChange={setInputMetric}>
                <SelectTrigger className="text-sm">
                  <SelectValue placeholder="Input metric (has a time dimension)" />
                </SelectTrigger>
                <SelectContent>
                  {metricNames.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Input
                type="number"
                placeholder="window (trailing periods; blank = to date)"
                value={windowSize}
                onChange={(e) => setWindowSize(e.target.value)}
              />
            </>
          )}
        </div>
        <DialogFooter>
          <Button disabled={busy || !name} onClick={submit}>
            {busy ? "Defining…" : "Define"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
