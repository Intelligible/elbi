import type { Edge, Node } from "@xyflow/react"
import { Background, Controls, ReactFlow } from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import {
  Ban,
  LayoutList,
  Play,
  Plus,
  RefreshCw,
  Rows3,
  Search,
  ShieldCheck,
  Trash2,
  Workflow as WorkflowIcon,
  X,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { IconButton } from "@/components/app/IconButton"
import { Scene, SceneBody, SceneHeader, SceneSection } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { VerdictBadge } from "@/components/VerdictBadge"
import { useTheme } from "@/hooks/useTheme"
import type {
  AssetCheck,
  AssetGraph,
  AssetStatus,
  CheckSeverity,
  Freshness,
  HistoryRun,
  RunDetail,
  RunHistory,
  RunIf,
  RunStatus,
  RunStep,
  Schedule,
  StepState,
  Workflow,
  WorkflowStep,
} from "@/lib/orchestration"
import {
  backfill,
  cancelRun,
  createSchedule,
  deleteCheck,
  deleteSchedule,
  deleteWorkflow,
  getAssetStatus,
  getChecks,
  getGraph,
  getHistory,
  getRun,
  getSchedules,
  getSettings,
  getWorkflows,
  materialize,
  retryRun,
  runWorkflow,
  setRetryPolicy,
  upsertCheck,
  upsertWorkflow,
} from "@/lib/orchestration"
import { EMPTY, uuid } from "@/lib/utils"

const FRESHNESS_VARIANT: Record<Freshness, "success" | "warning" | "neutral"> = {
  materialized: "success",
  stale: "warning",
  never: "neutral",
}

const RUN_VARIANT: Record<RunStatus, "success" | "danger" | "warning" | "info"> = {
  succeeded: "success",
  failed: "danger",
  cancelled: "warning",
  running: "info",
}

// One vocabulary of colors for a step's outcome, shared by the matrix cells and the
// run-detail Gantt so the same state always reads the same.
const STATE_COLOR: Record<StepState, string> = {
  succeeded: "var(--success)",
  failed: "var(--danger)",
  skipped: "var(--muted-foreground)",
}

const RUN_DOT: Record<RunStatus, string> = {
  running: "var(--info)",
  succeeded: "var(--success)",
  failed: "var(--danger)",
  cancelled: "var(--warning)",
}

const TERMINAL: RunStatus[] = ["succeeded", "failed", "cancelled"]

type Tab = "overview" | "runs" | "assets" | "workflows" | "schedules"

function ago(iso: string | null): string {
  if (!iso) return "never"
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000))
  if (s < 60) return `${s}s ago`
  const m = Math.round(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.round(h / 24)}d ago`
}

function dur(ms: number | null): string {
  if (ms == null) return EMPTY
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

function runDuration(run: { startedAt: string; finishedAt: string | null }): number | null {
  if (!run.finishedAt) return null
  return new Date(run.finishedAt).getTime() - new Date(run.startedAt).getTime()
}

function median(xs: number[]): number | null {
  if (xs.length === 0) return null
  const sorted = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

export function OrchestrationPage() {
  const [tab, setTab] = useState<Tab>("overview")
  const [assets, setAssets] = useState<AssetStatus[]>([])
  const [graph, setGraph] = useState<AssetGraph | null>(null)
  const [history, setHistory] = useState<RunHistory | null>(null)
  const [schedules, setSchedules] = useState<Schedule[]>([])
  const [checks, setChecks] = useState<AssetCheck[]>([])
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [retries, setRetries] = useState(1)
  const [compute, setCompute] = useState("")
  const [selected, setSelected] = useState<RunDetail | null>(null)
  const [busy, setBusy] = useState(false)
  const selectedId = useRef<string | null>(null)
  selectedId.current = selected?.id ?? null

  const refresh = useCallback(async () => {
    const [a, g, h, s, c, cfg, w] = await Promise.all([
      getAssetStatus(),
      getGraph(),
      getHistory(),
      getSchedules(),
      getChecks(),
      getSettings(),
      getWorkflows(),
    ])
    setAssets(a)
    setGraph(g)
    setHistory(h)
    setSchedules(s)
    setChecks(c)
    setRetries(cfg.maxRetries)
    setCompute(cfg.compute)
    setWorkflows(w)
    if (selectedId.current) setSelected(await getRun(selectedId.current))
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const running = (history?.runs ?? []).some((r) => r.status === "running")
  const live = running || selected?.status === "running"
  useEffect(() => {
    if (!live) return
    const timer = setInterval(() => void refresh(), 1500)
    return () => clearInterval(timer)
  }, [live, refresh])

  const start = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
      await refresh()
    } finally {
      setBusy(false)
    }
  }

  const materializeAsset = (asset: string) =>
    start(() => materialize({ assets: [asset], includeDownstream: true }))
  const selectRun = async (id: string) => {
    setTab("runs")
    setSelected(selected?.id === id ? null : await getRun(id))
  }

  const staleCount = assets.filter((a) => a.status !== "materialized").length

  return (
    <Scene>
      <SceneHeader
        icon={<WorkflowIcon className="size-5" />}
        title="Orchestration"
        description="Your derivations as software-defined assets, materialized in dependency order. Fresh assets are skipped; a change upstream marks everything downstream stale."
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => start(() => materialize({ selection: "all" }))}
              disabled={busy || running}
            >
              <RefreshCw className={busy || running ? "size-4 animate-spin" : "size-4"} />
              Materialize all
            </Button>
            <Button
              size="sm"
              onClick={() => start(() => materialize({ selection: "stale" }))}
              disabled={busy || running || staleCount === 0}
            >
              <Play className="size-4" />
              Materialize stale{staleCount > 0 ? ` (${staleCount})` : ""}
            </Button>
          </>
        }
      >
        <TabStrip tab={tab} onTab={setTab} staleCount={staleCount} running={running} />
      </SceneHeader>

      <SceneBody width="wide">
        {tab === "overview" ? (
          <div className="space-y-6">
            <KpiHeader assets={assets} runs={history?.runs ?? []} />
            {graph && graph.nodes.length > 0 ? (
              <SceneSection
                title="Pipeline"
                description="Dependency graph: click an asset to materialize it and its downstream."
              >
                <AssetGraphView graph={graph} onMaterialize={materializeAsset} />
              </SceneSection>
            ) : null}
          </div>
        ) : null}

        {tab === "runs" ? (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_28rem]">
            <div className="min-w-0">
              <RunHistoryMatrix
                history={history}
                selectedId={selected?.id ?? null}
                onSelect={selectRun}
              />
            </div>
            <div className="min-w-0">
              {selected ? (
                <RunDetailPanel
                  detail={selected}
                  busy={busy}
                  onClose={() => setSelected(null)}
                  onRetry={() => start(() => retryRun(selected.id))}
                  onCancel={() => start(() => cancelRun(selected.id))}
                />
              ) : (
                <div className="rounded-xl border border-dashed border-border px-4 py-10 text-center text-sm text-text-tertiary">
                  Select a run to inspect its steps, timings, and logs.
                </div>
              )}
            </div>
          </div>
        ) : null}

        {tab === "assets" ? (
          <AssetsPanel
            assets={assets}
            checks={checks}
            busy={busy || running}
            onMaterialize={materializeAsset}
            onAddCheck={(body) => start(() => upsertCheck(body))}
            onDeleteCheck={(checkId) => start(() => deleteCheck(checkId))}
            onBackfill={(asset, param, values) => {
              setTab("runs")
              void start(() => backfill({ asset, param, values }))
            }}
          />
        ) : null}

        {tab === "workflows" ? (
          <WorkflowsPanel
            workflows={workflows}
            assets={assets.map((a) => a.asset)}
            busy={busy || running}
            onChange={refresh}
            onRun={(id) => {
              setTab("runs")
              void start(() => runWorkflow(id))
            }}
          />
        ) : null}

        {tab === "schedules" ? (
          <SchedulesPanel
            schedules={schedules}
            retries={retries}
            compute={compute}
            onChange={refresh}
            onSetRetries={(n) => start(() => setRetryPolicy(n))}
          />
        ) : null}
      </SceneBody>
    </Scene>
  )
}

function TabStrip({
  tab,
  onTab,
  staleCount,
  running,
}: {
  tab: Tab
  onTab: (t: Tab) => void
  staleCount: number
  running: boolean
}) {
  const tabs: { id: Tab; label: string; badge?: React.ReactNode }[] = [
    { id: "overview", label: "Overview" },
    {
      id: "assets",
      label: "Assets",
      badge:
        staleCount > 0 ? (
          <span className="tabular-nums text-warning">{staleCount}</span>
        ) : undefined,
    },
    {
      id: "runs",
      label: "Runs",
      badge: running ? <RefreshCw className="size-3 animate-spin" /> : undefined,
    },
    { id: "workflows", label: "Workflows" },
    { id: "schedules", label: "Schedules" },
  ]
  return (
    <Tabs value={tab} onValueChange={(v) => onTab(v as Tab)}>
      <TabsList className="flex w-full justify-start gap-1 rounded-none bg-transparent p-0 group-data-[orientation=horizontal]/tabs:h-auto">
        {tabs.map((t) => (
          <TabsTrigger key={t.id} value={t.id} className={TAB_TRIGGER}>
            {t.label}
            {t.badge}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  )
}

// An underlined header tab: the active one carries a primary bottom border.
const TAB_TRIGGER =
  "h-auto flex-none rounded-none border-0 border-b-2 border-transparent px-3 py-2 font-normal text-text-secondary transition-colors hover:text-foreground data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:font-medium group-data-[variant=default]/tabs-list:data-[state=active]:shadow-none dark:text-text-secondary dark:data-[state=active]:border-primary dark:data-[state=active]:bg-transparent"

function Kpi({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-text-tertiary">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums text-foreground">{value}</div>
      {hint ? <div className="text-xs text-text-tertiary">{hint}</div> : null}
    </div>
  )
}

function KpiHeader({ assets, runs }: { assets: AssetStatus[]; runs: HistoryRun[] }) {
  const fresh = assets.filter((a) => a.status === "materialized").length
  const stale = assets.length - fresh
  const last = runs[0] ?? null
  const terminal = runs.filter((r) => TERMINAL.includes(r.status))
  const succeeded = terminal.filter((r) => r.status === "succeeded").length
  const rate = terminal.length ? Math.round((succeeded / terminal.length) * 100) : null
  const med = median(terminal.map(runDuration).filter((d): d is number => d != null))

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      <Kpi label="Assets" value={String(assets.length)} hint={`${fresh} fresh`} />
      <Kpi
        label="Stale"
        value={String(stale)}
        hint={stale === 0 ? "all up to date" : "need materializing"}
      />
      <Kpi
        label="Last run"
        value={last ? last.status : EMPTY}
        hint={last ? ago(last.startedAt) : "no runs yet"}
      />
      <Kpi
        label="Success rate"
        value={rate == null ? EMPTY : `${rate}%`}
        hint={terminal.length ? `last ${terminal.length} runs` : "no runs yet"}
      />
      <Kpi label="Median duration" value={dur(med)} hint="per run" />
    </div>
  )
}

function RunHistoryMatrix({
  history,
  selectedId,
  onSelect,
}: {
  history: RunHistory | null
  selectedId: string | null
  onSelect: (runId: string) => void
}) {
  if (!history || history.runs.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-6 text-sm text-text-tertiary">
        No runs yet. Materialize to create one.
      </div>
    )
  }
  // Newest run first (left) so the latest column needs no horizontal scrolling.
  const runs = history.runs
  const gridCols = `minmax(9rem, 14rem) repeat(${runs.length}, 1.5rem)`

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-xl border border-border bg-card p-3">
        <div className="grid gap-x-1 gap-y-1" style={{ gridTemplateColumns: gridCols }}>
          <div /> {/* corner */}
          {runs.map((r) => (
            <IconButton
              key={r.id}
              label={`${r.cause} · ${r.status} · ${ago(r.startedAt)}`}
              size="icon-xs"
              onClick={() => onSelect(r.id)}
            >
              <span
                className="size-3 rounded-full transition-all"
                style={{
                  background: RUN_DOT[r.status],
                  boxShadow: r.id === selectedId ? "0 0 0 2px var(--primary)" : undefined,
                }}
              />
            </IconButton>
          ))}
          {history.assets.map((asset) => (
            <MatrixRow key={asset} asset={asset} runs={runs} onSelect={onSelect} />
          ))}
        </div>
      </div>
      <MatrixLegend />
    </div>
  )
}

function MatrixRow({
  asset,
  runs,
  onSelect,
}: {
  asset: string
  runs: HistoryRun[]
  onSelect: (runId: string) => void
}) {
  return (
    <>
      <div className="flex items-center truncate pr-2 font-mono text-xs text-text-secondary">
        {asset}
      </div>
      {runs.map((r) => {
        const state = r.cells[asset]
        return (
          <IconButton
            key={r.id}
            label={
              state ? `${asset} · ${state} · ${ago(r.startedAt)}` : `${asset} · not in this run`
            }
            size="icon-xs"
            className="h-5"
            onClick={() => onSelect(r.id)}
          >
            <span
              className="size-4 rounded-sm"
              style={{
                background: state ? STATE_COLOR[state] : "var(--muted)",
                opacity: state ? 1 : 0.4,
              }}
            />
          </IconButton>
        )
      })}
    </>
  )
}

function MatrixLegend() {
  const items: [StepState, string][] = [
    ["succeeded", "Succeeded"],
    ["failed", "Failed"],
    ["skipped", "Skipped (fresh)"],
  ]
  return (
    <div className="flex flex-wrap items-center gap-4 px-1 text-xs text-text-tertiary">
      {items.map(([state, label]) => (
        <span key={state} className="flex items-center gap-1.5">
          <span className="size-3 rounded-sm" style={{ background: STATE_COLOR[state] }} />
          {label}
        </span>
      ))}
      <span className="text-text-tertiary/70">Newest run first</span>
    </div>
  )
}

function RunDetailPanel({
  detail,
  busy,
  onClose,
  onRetry,
  onCancel,
}: {
  detail: RunDetail
  busy: boolean
  onClose: () => void
  onRetry: () => void
  onCancel: () => void
}) {
  // Assets materialize sequentially, so a waterfall from cumulative durations is an
  // accurate mini-Gantt: it shows which step dominated the run's wall-clock.
  const total = detail.steps.reduce((sum, s) => sum + s.durationMs, 0) || 1
  let offset = 0
  const bars = detail.steps.map((s) => {
    const left = (offset / total) * 100
    offset += s.durationMs
    return { step: s, left, width: (s.durationMs / total) * 100 }
  })

  return (
    <div className="rounded-xl border border-border bg-card">
      <div className="flex items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0 space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant={RUN_VARIANT[detail.status]}>
              {detail.status === "running" ? (
                <RefreshCw className="mr-1 size-3 animate-spin" />
              ) : null}
              {detail.status}
            </Badge>
            <span className="text-sm text-text-secondary">{detail.cause}</span>
            {detail.parentRunId ? (
              <span className="text-xs text-text-tertiary">(retry)</span>
            ) : null}
          </div>
          <div className="text-xs text-text-tertiary">
            started {ago(detail.startedAt)} · {dur(runDuration(detail))}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {detail.status === "running" ? (
            <Button variant="outline" size="sm" onClick={onCancel}>
              <Ban className="size-3.5" />
              Cancel
            </Button>
          ) : null}
          {detail.status === "failed" ? (
            <Button variant="outline" size="sm" onClick={onRetry} disabled={busy}>
              <RefreshCw className="size-3.5" />
              Retry
            </Button>
          ) : null}
          <IconButton
            label="Close run detail"
            size="icon-xs"
            className="text-text-tertiary"
            onClick={onClose}
          >
            <X className="size-4" />
          </IconButton>
        </div>
      </div>
      <div className="divide-y divide-border">
        {bars.map(({ step, left, width }) => (
          <StepRow key={step.asset} step={step} left={left} width={width} />
        ))}
        {detail.steps.length === 0 ? (
          <div className="px-4 py-3 text-xs text-text-tertiary">
            No assets ran (nothing was stale).
          </div>
        ) : null}
      </div>
    </div>
  )
}

function StepRow({ step, left, width }: { step: RunStep; left: number; width: number }) {
  const [showLogs, setShowLogs] = useState(false)
  return (
    <div className="px-4 py-2 text-xs">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate font-mono">{step.asset}</span>
        {step.verdict ? <VerdictBadge verdict={step.verdict} /> : null}
        <span
          className={`shrink-0 tabular-nums ${step.error ? "text-danger" : "text-text-tertiary"}`}
        >
          {step.error ? "failed" : dur(step.durationMs)}
        </span>
        {step.logs || step.error ? (
          <Button
            variant="link"
            className="h-auto p-0 text-xs font-normal text-text-tertiary underline-offset-2 hover:text-foreground"
            aria-expanded={showLogs}
            onClick={() => setShowLogs((v) => !v)}
          >
            {showLogs ? "hide" : "logs"}
          </Button>
        ) : null}
      </div>
      {/* Gantt track (full panel width, under the label line) */}
      <div className="relative mt-1 h-2 rounded bg-surface-secondary">
        <span
          className="absolute top-0 h-2 rounded"
          style={{
            left: `${left}%`,
            width: `${Math.max(width, 2)}%`,
            background: STATE_COLOR[step.state],
          }}
          title={`${step.state} · ${dur(step.durationMs)}`}
        />
      </div>
      {step.checks.length > 0 ? (
        <div className="mt-1 flex flex-wrap gap-1">
          {step.checks.map((c) => {
            const color =
              c.state === "passed"
                ? "var(--success)"
                : c.severity === "error"
                  ? "var(--danger)"
                  : "var(--warning)"
            const label =
              c.state === "passed"
                ? `${c.name} ✓`
                : c.state === "error"
                  ? `${c.name} could not run`
                  : `${c.name} ${c.failed}/${c.total}`
            return (
              <span
                key={c.name}
                className="rounded border px-1.5 py-0.5 text-3xs"
                style={{ borderColor: color, color }}
                title={c.error ?? c.expr}
              >
                {label}
              </span>
            )
          })}
        </div>
      ) : null}
      {showLogs ? (
        <pre className="mt-1 max-h-64 overflow-auto rounded-md bg-surface-secondary p-2 font-mono text-2xs leading-relaxed text-text-secondary">
          {step.error ? `${step.error}\n\n` : ""}
          {step.logs || "(no output)"}
        </pre>
      ) : null}
    </div>
  )
}

type StatusFilter = "all" | Freshness

// A 24px icon button whose negative margin keeps the 14px footprint of the icon it holds.
const ICON_14 = "-m-1.25 text-text-tertiary"
// The same for a 16px icon.
const ICON_16 = "-m-1 text-text-tertiary"

function AssetsPanel({
  assets,
  checks,
  busy,
  onMaterialize,
  onAddCheck,
  onDeleteCheck,
  onBackfill,
}: {
  assets: AssetStatus[]
  checks: AssetCheck[]
  busy: boolean
  onMaterialize: (asset: string) => void
  onAddCheck: (body: { asset: string; name: string; expr: string; severity: CheckSeverity }) => void
  onDeleteCheck: (checkId: string) => void
  onBackfill: (asset: string, param: string, values: string[]) => void
}) {
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<StatusFilter>("all")
  const [grouped, setGrouped] = useState(true)
  const [openChecks, setOpenChecks] = useState<string | null>(null)

  const checksByAsset = useMemo(() => {
    const map: Record<string, AssetCheck[]> = {}
    for (const c of checks) {
      map[c.asset] ??= []
      map[c.asset].push(c)
    }
    return map
  }, [checks])

  const counts = useMemo(() => {
    const c: Record<StatusFilter, number> = {
      all: assets.length,
      materialized: 0,
      stale: 0,
      never: 0,
    }
    for (const a of assets) c[a.status] += 1
    return c
  }, [assets])

  const shown = assets.filter(
    (a) =>
      (filter === "all" || a.status === filter) &&
      a.asset.toLowerCase().includes(query.trim().toLowerCase()),
  )

  const groups: [Freshness | "all", AssetStatus[]][] =
    grouped && filter === "all"
      ? (["stale", "never", "materialized"] as Freshness[])
          .map((s) => [s, shown.filter((a) => a.status === s)] as [Freshness, AssetStatus[]])
          .filter(([, list]) => list.length > 0)
      : [["all", shown]]

  const filters: [StatusFilter, string][] = [
    ["all", "All"],
    ["materialized", "Fresh"],
    ["stale", "Stale"],
    ["never", "Never"],
  ]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
          <Input
            className="h-8 w-52 pl-7"
            placeholder="Search assets"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-border bg-card p-0.5">
          {filters.map(([key, label]) => (
            <Button
              key={key}
              variant="ghost"
              size="xs"
              aria-pressed={filter === key}
              className={
                "gap-0.75 font-normal " +
                (filter === key
                  ? "bg-accent text-foreground hover:text-foreground dark:hover:bg-accent"
                  : "text-text-secondary")
              }
              onClick={() => setFilter(key)}
            >
              {label} <span className="tabular-nums text-text-tertiary">{counts[key]}</span>
            </Button>
          ))}
        </div>
        <Button
          variant="outline"
          size="xs"
          aria-pressed={grouped && filter === "all"}
          className={
            "ml-auto h-6.5 px-2 font-normal has-[>svg]:px-2 " +
            (grouped && filter === "all"
              ? "bg-accent text-foreground hover:text-foreground"
              : "text-text-secondary")
          }
          onClick={() => setGrouped((v) => !v)}
          disabled={filter !== "all"}
          title="Group by freshness"
        >
          {grouped ? <Rows3 className="size-3.5" /> : <LayoutList className="size-3.5" />}
          Group
        </Button>
      </div>

      <div className="space-y-3">
        {groups.map(([group, list]) => (
          <div key={group}>
            {group !== "all" ? (
              <div className="mb-1 flex items-center gap-2 px-1 text-xs uppercase tracking-wide text-text-tertiary">
                {group === "materialized" ? "fresh" : group}
                <span className="tabular-nums">{list.length}</span>
              </div>
            ) : null}
            <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card">
              {list.map((asset) => {
                const assetChecks = checksByAsset[asset.asset] ?? []
                const open = openChecks === asset.asset
                return (
                  <li key={asset.asset}>
                    <div className="flex items-center justify-between gap-2 px-4 py-2.5 text-sm">
                      <div className="min-w-0">
                        <div className="truncate font-mono text-foreground">{asset.asset}</div>
                        <div className="text-xs text-text-tertiary">
                          {asset.status === "never"
                            ? "never materialized"
                            : `materialized ${ago(asset.lastMaterializedAt)} · ${dur(asset.lastDurationMs)}`}
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        <Button
                          variant="outline"
                          size="xs"
                          aria-expanded={open}
                          className={
                            "h-5.5 font-normal " +
                            (open
                              ? "bg-accent text-foreground hover:text-foreground"
                              : "text-text-tertiary")
                          }
                          onClick={() => setOpenChecks(open ? null : asset.asset)}
                          title="Data-quality checks"
                        >
                          <ShieldCheck className="size-3.5" />
                          {assetChecks.length > 0 ? assetChecks.length : "checks"}
                        </Button>
                        {asset.verdict ? <VerdictBadge verdict={asset.verdict} /> : null}
                        <Badge variant={FRESHNESS_VARIANT[asset.status]}>{asset.status}</Badge>
                        <IconButton
                          label={`Materialize ${asset.asset}`}
                          size="icon-xs"
                          className={ICON_14}
                          onClick={() => onMaterialize(asset.asset)}
                          disabled={busy}
                        >
                          <Play className="size-3.5" />
                        </IconButton>
                      </div>
                    </div>
                    {open ? (
                      <>
                        <CheckEditor
                          asset={asset.asset}
                          checks={assetChecks}
                          onAdd={onAddCheck}
                          onDelete={onDeleteCheck}
                        />
                        <BackfillForm asset={asset.asset} onBackfill={onBackfill} />
                      </>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
        {shown.length === 0 ? (
          <div className="rounded-xl border border-border bg-card px-4 py-6 text-sm text-text-tertiary">
            No assets match.
          </div>
        ) : null}
      </div>
    </div>
  )
}

// A 28px select that sits in a row of h-7 inputs. Each gets a fixed width that fits its
// longest option, so the row does not reflow as the value changes.
const COMPACT_TRIGGER = "justify-between gap-0 bg-card pr-0.5 pl-2 text-xs data-[size=sm]:h-7"

function CheckEditor({
  asset,
  checks,
  onAdd,
  onDelete,
}: {
  asset: string
  checks: AssetCheck[]
  onAdd: (body: { asset: string; name: string; expr: string; severity: CheckSeverity }) => void
  onDelete: (checkId: string) => void
}) {
  const [name, setName] = useState("")
  const [expr, setExpr] = useState("")
  const [severity, setSeverity] = useState<CheckSeverity>("warn")

  const add = () => {
    if (!name.trim() || !expr.trim()) return
    onAdd({ asset, name: name.trim(), expr: expr.trim(), severity })
    setName("")
    setExpr("")
  }

  return (
    <div className="border-t border-border bg-surface-secondary px-4 py-3 text-xs">
      <div className="mb-2 text-text-tertiary">
        Data-quality checks: a boolean SQL expression over the asset's output columns, evaluated at
        materialization. <span className="text-warning">warn</span> records failing rows;{" "}
        <span className="text-danger">error</span> fails the run.
      </div>
      {checks.length > 0 ? (
        <ul className="mb-2 space-y-1">
          {checks.map((c) => {
            const color = c.severity === "error" ? "var(--danger)" : "var(--warning)"
            return (
              <li key={c.id} className="flex items-center gap-2">
                <span className="font-medium">{c.name}</span>
                <span
                  className="rounded border px-1 text-3xs uppercase"
                  style={{ borderColor: color, color }}
                >
                  {c.severity}
                </span>
                <code className="min-w-0 flex-1 truncate text-text-secondary">{c.expr}</code>
                <IconButton
                  label={`Delete check ${c.name}`}
                  size="icon-xs"
                  className={`${ICON_14} hover:text-danger`}
                  onClick={() => onDelete(c.id)}
                >
                  <Trash2 className="size-3.5" />
                </IconButton>
              </li>
            )
          })}
        </ul>
      ) : null}
      <div className="flex items-center gap-2">
        <Input
          className="h-7 w-32"
          placeholder="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <Input
          className="h-7 flex-1 font-mono"
          placeholder="e.g. amount >= 0"
          value={expr}
          onChange={(e) => setExpr(e.target.value)}
        />
        <Select value={severity} onValueChange={(v) => setSeverity(v as CheckSeverity)}>
          <SelectTrigger size="sm" aria-label="Severity" className={`${COMPACT_TRIGGER} w-14.5`}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="warn" className="text-xs">
              warn
            </SelectItem>
            <SelectItem value="error" className="text-xs">
              error
            </SelectItem>
          </SelectContent>
        </Select>
        <Button size="sm" variant="outline" onClick={add} disabled={!name || !expr}>
          <Plus className="size-3.5" /> Add
        </Button>
      </div>
    </div>
  )
}

function BackfillForm({
  asset,
  onBackfill,
}: {
  asset: string
  onBackfill: (asset: string, param: string, values: string[]) => void
}) {
  const [param, setParam] = useState("")
  const [values, setValues] = useState("")

  const run = () => {
    const list = values
      .split(",")
      .map((v) => v.trim())
      .filter(Boolean)
    if (!param.trim() || list.length === 0) return
    onBackfill(asset, param.trim(), list)
    setValues("")
  }

  return (
    <div className="border-t border-border bg-surface-secondary px-4 py-3 text-xs">
      <div className="mb-2 text-text-tertiary">
        Backfill: materialize this asset once per partition value (for a parameterized derivation).
        Each value runs with <code>{"{param: value}"}</code>; unchanged partitions are cache hits.
      </div>
      <div className="flex items-center gap-2">
        <Input
          className="h-7 w-32"
          placeholder="param, e.g. region"
          value={param}
          onChange={(e) => setParam(e.target.value)}
        />
        <Input
          className="h-7 flex-1 font-mono"
          placeholder="values, comma-separated: west, east, north"
          value={values}
          onChange={(e) => setValues(e.target.value)}
        />
        <Button size="sm" variant="outline" onClick={run} disabled={!param || !values}>
          <Play className="size-3.5" /> Backfill
        </Button>
      </div>
    </div>
  )
}

// Freshness → node fill; verdict (when certified) → node border, else freshness.
const FRESH_FILL: Record<Freshness, string> = {
  materialized: "var(--success-tint)",
  stale: "var(--warning-tint)",
  never: "var(--muted)",
}
const VERDICT_BORDER: Record<string, string> = {
  sound: "var(--verified)",
  unsound: "var(--danger)",
  inconclusive: "var(--caution)",
}

function toFlow(graph: AssetGraph): { nodes: Node[]; edges: Edge[] } {
  // Layered left-to-right layout: an asset's column is the longest dependency path
  // reaching it, so dependencies always sit to the left of their dependents.
  const preds: Record<string, string[]> = {}
  for (const n of graph.nodes) preds[n.asset] = []
  for (const e of graph.edges) preds[e.to]?.push(e.from)

  const depth: Record<string, number> = {}
  const depthOf = (asset: string, seen: Set<string>): number => {
    if (asset in depth) return depth[asset]
    if (seen.has(asset)) return 0 // defensive: ignore a cycle rather than recurse forever
    seen.add(asset)
    const parents = preds[asset] ?? []
    const d = parents.length === 0 ? 0 : Math.max(...parents.map((p) => depthOf(p, seen) + 1))
    depth[asset] = d
    return d
  }
  for (const n of graph.nodes) depthOf(n.asset, new Set())

  const perColumn: Record<number, number> = {}
  const nodes: Node[] = graph.nodes.map((n) => {
    const column = depth[n.asset]
    const row = perColumn[column] ?? 0
    perColumn[column] = row + 1
    const border = n.verdict ? VERDICT_BORDER[n.verdict] : undefined
    return {
      id: n.asset,
      position: { x: column * 240, y: row * 84 },
      data: { label: `${n.asset}\n${n.status}` },
      style: {
        borderRadius: 8,
        borderWidth: 1,
        borderColor: border ?? "var(--border)",
        background: FRESH_FILL[n.status],
        padding: 8,
        fontSize: 12,
        whiteSpace: "pre-line" as const,
      },
    }
  })
  const edges: Edge[] = graph.edges.map((e, i) => ({
    id: `${e.from}->${e.to}-${i}`,
    source: e.from,
    target: e.to,
    animated: true,
  }))
  return { nodes, edges }
}

function AssetGraphView({
  graph,
  onMaterialize,
}: {
  graph: AssetGraph
  onMaterialize: (asset: string) => void
}) {
  const { resolved } = useTheme()
  const { nodes, edges } = useMemo(() => toFlow(graph), [graph])

  return (
    <div className="h-80 overflow-hidden rounded-xl border border-border bg-card">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        colorMode={resolved}
        nodesDraggable={false}
        nodesConnectable={false}
        onNodeClick={(_, node) => onMaterialize(node.id)}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}

const RUN_IF_OPTIONS: { value: RunIf; label: string }[] = [
  { value: "all_success", label: "all upstream succeeded" },
  { value: "none_failed", label: "none failed" },
  { value: "at_least_one_success", label: "≥1 succeeded" },
  { value: "all_done", label: "all done (any state)" },
  { value: "at_least_one_failed", label: "≥1 failed (cleanup)" },
  { value: "all_failed", label: "all failed" },
]

type StepDraft = {
  // React's handle on the row, distinct from `id`, which the reader types and which the
  // workflow is keyed by. Keying the row on a value being edited would remount the
  // inputs on every keystroke.
  rowId: string
  id: string
  selection: "stale" | "all"
  dependsOn: string
  runIf: RunIf
}

function emptyStep(n: number): StepDraft {
  return { rowId: uuid(), id: `step${n}`, selection: "stale", dependsOn: "", runIf: "all_success" }
}

function WorkflowsPanel({
  workflows,
  busy,
  onChange,
  onRun,
}: {
  workflows: Workflow[]
  assets: string[]
  busy: boolean
  onChange: () => void
  onRun: (workflowId: string) => void
}) {
  const [editId, setEditId] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [steps, setSteps] = useState<StepDraft[]>([emptyStep(1)])

  const reset = () => {
    setEditId(null)
    setName("")
    setSteps([emptyStep(1)])
  }

  const load = (wf: Workflow) => {
    setEditId(wf.id)
    setName(wf.name)
    setSteps(
      wf.steps.map((s) => ({
        rowId: uuid(),
        id: s.id,
        selection: s.selection ?? "stale",
        dependsOn: (s.dependsOn ?? []).join(", "),
        runIf: s.runIf ?? "all_success",
      })),
    )
  }

  const save = async () => {
    if (!name.trim() || steps.length === 0) return
    const built: WorkflowStep[] = steps.map((s) => ({
      id: s.id.trim(),
      selection: s.selection,
      dependsOn: s.dependsOn
        .split(",")
        .map((d) => d.trim())
        .filter(Boolean),
      runIf: s.runIf,
    }))
    await upsertWorkflow({ id: editId ?? undefined, name: name.trim(), steps: built })
    reset()
    onChange()
  }

  const setStep = (rowId: string, patch: Partial<StepDraft>) =>
    setSteps((prev) => prev.map((s) => (s.rowId === rowId ? { ...s, ...patch } : s)))

  return (
    <div className="space-y-4">
      <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card">
        {workflows.map((wf) => (
          <li key={wf.id} className="flex items-center justify-between px-4 py-2.5 text-sm">
            <div className="min-w-0">
              <div className="truncate font-medium text-foreground">{wf.name}</div>
              <div className="text-xs text-text-tertiary">
                {wf.steps.length} step{wf.steps.length === 1 ? "" : "s"}:{" "}
                {wf.steps.map((s) => s.id).join(" → ")}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button size="sm" onClick={() => onRun(wf.id)} disabled={busy}>
                <Play className="size-3.5" /> Run
              </Button>
              <Button size="sm" variant="outline" onClick={() => load(wf)}>
                Edit
              </Button>
              <IconButton
                label={`Delete workflow ${wf.name}`}
                size="icon-xs"
                className={`${ICON_16} hover:text-danger`}
                onClick={async () => {
                  await deleteWorkflow(wf.id)
                  onChange()
                }}
              >
                <Trash2 className="size-4" />
              </IconButton>
            </div>
          </li>
        ))}
        {workflows.length === 0 ? (
          <li className="px-4 py-6 text-sm text-text-tertiary">
            No workflows yet. Build one below: a sequence of materialization steps with branch
            conditions.
          </li>
        ) : null}
      </ul>

      <div className="space-y-3 rounded-xl border border-border bg-card p-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{editId ? "Edit workflow" : "New workflow"}</span>
          <Input
            className="h-8 w-52"
            placeholder="workflow name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          {editId ? (
            <Button size="sm" variant="ghost" onClick={reset}>
              Cancel
            </Button>
          ) : null}
        </div>
        <div className="space-y-2">
          {steps.map((step) => (
            <div key={step.rowId} className="flex flex-wrap items-center gap-2 text-xs">
              <Input
                className="h-7 w-28"
                placeholder="step id"
                value={step.id}
                onChange={(e) => setStep(step.rowId, { id: e.target.value })}
              />
              <Select
                value={step.selection}
                onValueChange={(v) => setStep(step.rowId, { selection: v as "stale" | "all" })}
              >
                <SelectTrigger
                  size="sm"
                  aria-label="Selection"
                  className={`${COMPACT_TRIGGER} w-30`}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="stale" className="text-xs">
                    materialize stale
                  </SelectItem>
                  <SelectItem value="all" className="text-xs">
                    materialize all
                  </SelectItem>
                </SelectContent>
              </Select>
              <Input
                className="h-7 w-40"
                placeholder="depends on (ids)"
                value={step.dependsOn}
                onChange={(e) => setStep(step.rowId, { dependsOn: e.target.value })}
              />
              <Select
                value={step.runIf}
                onValueChange={(v) => setStep(step.rowId, { runIf: v as RunIf })}
              >
                <SelectTrigger size="sm" aria-label="Run if" className={`${COMPACT_TRIGGER} w-48`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RUN_IF_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value} className="text-xs">
                      run if {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {steps.length > 1 ? (
                <IconButton
                  label="Remove step"
                  size="icon-xs"
                  className={`${ICON_14} hover:text-danger`}
                  onClick={() => setSteps((prev) => prev.filter((s) => s.rowId !== step.rowId))}
                >
                  <Trash2 className="size-3.5" />
                </IconButton>
              ) : null}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSteps((prev) => [...prev, emptyStep(prev.length + 1)])}
          >
            <Plus className="size-3.5" /> Step
          </Button>
          <Button size="sm" onClick={save} disabled={!name.trim()}>
            {editId ? "Save changes" : "Create workflow"}
          </Button>
        </div>
      </div>
    </div>
  )
}

function SchedulesPanel({
  schedules,
  retries,
  compute,
  onChange,
  onSetRetries,
}: {
  schedules: Schedule[]
  retries: number
  compute: string
  onChange: () => void
  onSetRetries: (n: number) => void
}) {
  const [name, setName] = useState("")
  const [cron, setCron] = useState("")

  const add = async () => {
    if (!name.trim() || !cron.trim()) return
    await createSchedule({
      name: name.trim(),
      selection: "stale",
      mode: "cron",
      cron: cron.trim(),
    })
    setName("")
    setCron("")
    onChange()
  }

  return (
    <div className="space-y-2 rounded-xl border border-border bg-card p-3">
      {schedules.map((s) => (
        <div key={s.id} className="flex items-center justify-between text-sm">
          <span>
            <span className="font-medium">{s.name}</span>{" "}
            <span className="text-text-secondary">
              : materialize {s.selection}{" "}
              {s.mode === "cron" ? `on “${s.cron}”` : `when ${s.dataset} changes`}
            </span>
          </span>
          <IconButton
            label="Delete schedule"
            size="icon-xs"
            className={`${ICON_16} hover:text-danger`}
            onClick={async () => {
              await deleteSchedule(s.id)
              onChange()
            }}
          >
            <Trash2 className="size-4" />
          </IconButton>
        </div>
      ))}
      {schedules.length === 0 ? (
        <div className="px-1 py-2 text-sm text-text-tertiary">
          No schedules. Add a cron to materialize on a cadence.
        </div>
      ) : null}
      <div className="flex items-center gap-2 border-t border-border pt-3">
        <Input
          className="h-8 w-40"
          placeholder="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <Input
          className="h-8 flex-1 font-mono"
          placeholder="cron, e.g. 0 6 * * *"
          value={cron}
          onChange={(e) => setCron(e.target.value)}
        />
        <Button size="sm" variant="outline" onClick={add} disabled={!name || !cron}>
          <Plus className="size-4" />
          Add
        </Button>
      </div>
      <div className="flex items-center gap-2 border-t border-border pt-3 text-xs">
        <span className="text-text-secondary">Retry a failed asset</span>
        <Select value={String(retries)} onValueChange={(v) => onSetRetries(Number(v))}>
          <SelectTrigger
            size="sm"
            aria-label="Retry a failed asset"
            className={`${COMPACT_TRIGGER} w-20.5`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[0, 1, 2, 3, 5].map((n) => (
              <SelectItem key={n} value={String(n)} className="text-xs">
                {n === 0 ? "no retries" : `${n} ${n === 1 ? "time" : "times"}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-text-tertiary">with exponential backoff.</span>
        {compute ? (
          <span className="ml-auto text-text-tertiary">
            Compute: <span className="font-medium text-text-secondary">{compute}</span>{" "}
            (agent-authored assets; human-authored run in-process)
          </span>
        ) : null}
      </div>
      <p className="text-xs text-text-tertiary">
        Failed runs and runs that take more than 2× their recent median duration send an alert to
        the webhook configured in Settings → Webhooks (with an audit record).
      </p>
    </div>
  )
}
