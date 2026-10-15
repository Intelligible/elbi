import { Activity, AlertTriangle, Pencil, Play, Plus, Trash2 } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { FormControl, FormField } from "@/components/app/FormField"
import { Scene, SceneHeader } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Combobox } from "@/components/ui/combobox"
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
import { type FilterClause, listDerivations, listMetrics } from "@/lib/metrics"
import {
  AGGS,
  type Agg,
  buildConfig,
  describeConfig,
  type FilterRow,
  OPS,
  readFilters,
  readMeasure,
} from "@/lib/monitorConfig"
import {
  type CheckResult,
  checkMonitor,
  createMonitor,
  type DerivationColumn,
  deleteMonitor,
  getMonitor,
  listDerivationColumns,
  listMonitors,
  type Monitor,
  type MonitorHistory,
  updateMonitor,
} from "@/lib/monitors"

export function MonitorsPage() {
  const [monitors, setMonitors] = useState<Monitor[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState(false)

  const refresh = useCallback(() => {
    listMonitors()
      .then(setMonitors)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(refresh, [refresh])

  const current = monitors.find((m) => m.id === selected) ?? null

  return (
    <Scene>
      <SceneHeader
        icon={<Activity className="size-5" />}
        title="Monitors"
        description="Watch a certified metric or derivation; an anomaly against its learned baseline raises an alert that includes the source's verification verdict."
        actions={
          <Button size="sm" onClick={() => setCreating(true)}>
            <Plus className="size-4" /> New monitor
          </Button>
        }
      />
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-72 shrink-0 flex-col border-r border-border">
          <div className="flex-1 overflow-y-auto p-2">
            {monitors.length === 0 ? (
              <p className="px-2 py-2 text-xs text-text-tertiary">
                No monitors yet. Watch a certified metric or derivation.
              </p>
            ) : (
              monitors.map((m) => (
                <Button
                  key={m.id}
                  variant="ghost"
                  aria-current={selected === m.id ? "true" : undefined}
                  onClick={() => setSelected(m.id)}
                  className={`flex h-auto w-full justify-start gap-2 px-2 py-1.5 text-left font-normal hover:text-foreground has-[>svg]:px-2 ${
                    selected === m.id
                      ? "bg-accent hover:bg-accent dark:hover:bg-accent"
                      : "hover:bg-muted/60 dark:hover:bg-muted/60"
                  }`}
                >
                  <span className="flex-1 truncate">{m.name}</span>
                  {m.status === "alerting" ? (
                    <AlertTriangle className="size-3.5 shrink-0 text-danger" />
                  ) : (
                    <span className="size-2 shrink-0 rounded-full bg-verified" />
                  )}
                </Button>
              ))
            )}
          </div>
        </aside>

        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {error ? (
            <div className="border-b border-danger/30 bg-danger-tint px-4 py-2 font-mono text-xs text-danger">
              {error}
            </div>
          ) : null}
          {current ? (
            <MonitorDetail
              monitor={current}
              onError={setError}
              onChanged={refresh}
              onEdit={() => setEditing(true)}
              onDeleted={() => {
                setSelected(null)
                refresh()
              }}
            />
          ) : (
            <div className="grid flex-1 place-items-center text-sm text-text-tertiary">
              Select a monitor, or create one.
            </div>
          )}
        </div>
      </div>

      {editing && current ? (
        <MonitorDialog
          monitor={current}
          onClose={() => setEditing(false)}
          onSaved={() => {
            setEditing(false)
            refresh()
          }}
          onError={setError}
        />
      ) : null}
      {creating ? (
        <MonitorDialog
          onClose={() => setCreating(false)}
          onSaved={(id) => {
            setCreating(false)
            setSelected(id)
            refresh()
          }}
          onError={setError}
        />
      ) : null}
    </Scene>
  )
}

function MonitorDetail({
  monitor,
  onError,
  onChanged,
  onEdit,
  onDeleted,
}: {
  monitor: Monitor
  onError: (message: string) => void
  onChanged: () => void
  onEdit: () => void
  onDeleted: () => void
}) {
  const [history, setHistory] = useState<MonitorHistory | null>(null)
  const [last, setLast] = useState<CheckResult | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => {
    getMonitor(monitor.id)
      .then(setHistory)
      .catch((e) => onError(e instanceof Error ? e.message : String(e)))
  }, [monitor.id, onError])

  useEffect(() => {
    setLast(null)
    load()
  }, [load])

  const runCheck = useCallback(async () => {
    setBusy(true)
    try {
      setLast(await checkMonitor(monitor.id))
      load()
      onChanged()
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [monitor.id, load, onChanged, onError])

  const rows = (history?.snapshots ?? []).map((s) => ({
    at: s.at ?? "",
    value: s.value,
    status: s.anomalous ? "anomaly" : "normal",
  }))

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="border-b border-border px-6 py-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">{monitor.name}</h2>
          <Badge variant="neutral" className="font-mono">
            {monitor.targetKind}: {monitor.target}
          </Badge>
          {monitor.status === "alerting" ? (
            <Badge variant="danger">alerting</Badge>
          ) : (
            <Badge variant="success">ok</Badge>
          )}
          <div className="flex-1" />
          <Button size="sm" onClick={runCheck} disabled={busy}>
            <Play className="h-3.5 w-3.5" />
            {busy ? "Checking…" : "Run check"}
          </Button>
          <Button size="sm" variant="ghost" aria-label="Edit monitor" onClick={onEdit}>
            <Pencil className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            aria-label="Delete monitor"
            onClick={async () => {
              await deleteMonitor(monitor.id)
              onDeleted()
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
        <p className="mt-1 font-mono text-xs text-text-tertiary">
          {monitor.method}, sensitivity {monitor.sensitivity}
          {monitor.minValue !== null ? ` · min ${monitor.minValue}` : ""}
          {monitor.maxValue !== null ? ` · max ${monitor.maxValue}` : ""} · window {monitor.window}{" "}
          · every {monitor.intervalHours}h
        </p>
        <p className="mt-1 font-mono text-xs text-text-tertiary">
          {describeConfig(monitor.targetKind, monitor.config)}
        </p>
        {monitor.breachKeys && monitor.breachKeys.length > 0 ? (
          <p className="mt-2 text-sm break-words text-danger">
            Breaching now ({monitor.breachKeys.length}): {showKeys(monitor.breachKeys)}
          </p>
        ) : null}
        {last ? (
          <p className={`mt-2 text-sm ${last.anomalous ? "text-danger" : "text-text-secondary"}`}>
            Last check: {last.reason}
            {last.alerted ? ": alert fired" : ""}
          </p>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        {rows.length > 0 ? (
          <VizView
            viz={{
              spec: {
                mark: { type: "line", point: true, tooltip: true },
                encoding: {
                  x: { field: "at", type: "temporal", title: "time" },
                  y: { field: "value", type: "quantitative" },
                  color: { field: "status", type: "nominal" },
                },
              },
              rows,
            }}
            height={300}
          />
        ) : (
          <p className="text-sm text-text-tertiary">
            No snapshots yet. Run a check to record the first value.
          </p>
        )}

        {history && history.incidents.length > 0 ? (
          <div className="mt-6">
            <h3 className="mb-2 text-sm font-semibold">Incidents</h3>
            <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-card">
              {history.incidents.map((incident) => (
                <li key={incident.id} className="flex items-start gap-3 px-3 py-2 text-sm">
                  {incident.open ? (
                    <Badge variant="danger">open</Badge>
                  ) : (
                    <Badge variant="neutral">resolved</Badge>
                  )}
                  <div className="min-w-0 flex-1 text-text-secondary">
                    <p>{incident.reason}</p>
                    {incident.keys && incident.keys.length > 0 ? (
                      <p className="mt-0.5 text-xs break-words text-text-tertiary">
                        {showKeys(incident.keys)}
                      </p>
                    ) : null}
                  </div>
                  <span className="shrink-0 text-xs text-text-tertiary tabular-nums">
                    {incident.openedAt?.slice(0, 16).replace("T", " ")}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  )
}

const METHOD_HINT = {
  mad: "MAD compares each value to the median of recent checks, so one past spike does not widen what counts as normal and hide the next one. A good default.",
  zscore:
    "Z-score compares each value to the average of recent checks, in standard deviations. A past spike widens the normal range and can hide the next one.",
} as const

// A blank key is stored as the empty string so it cannot collide with a real key.
const showKeys = (keys: string[]) => keys.map((k) => k || "(blank)").join(", ")

function positive(text: string, what: string, whole = false): string | undefined {
  const n = Number(text)
  if (text.trim() === "" || !Number.isFinite(n) || n <= 0) return `${what} must be above 0.`
  if (whole && !Number.isInteger(n)) return `${what} must be a whole number.`
  return undefined
}

function MonitorDialog({
  monitor,
  onClose,
  onSaved,
  onError,
}: {
  monitor?: Monitor
  onClose: () => void
  onSaved: (id: string) => void
  onError: (message: string) => void
}) {
  const [targetKind, setTargetKind] = useState<"metric" | "derivation">(
    monitor?.targetKind ?? "metric",
  )
  const [targets, setTargets] = useState<string[]>([])
  const [name, setName] = useState(monitor?.name ?? "")
  const [target, setTarget] = useState(monitor?.target ?? "")
  const [method, setMethod] = useState<"mad" | "zscore">(monitor?.method ?? "mad")
  const [sensitivity, setSensitivity] = useState(String(monitor?.sensitivity ?? 3))
  const [minValue, setMinValue] = useState(monitor?.minValue?.toString() ?? "")
  const [maxValue, setMaxValue] = useState(monitor?.maxValue?.toString() ?? "")
  const [historyWindow, setHistoryWindow] = useState(String(monitor?.window ?? 30))
  const [intervalHours, setIntervalHours] = useState(String(monitor?.intervalHours ?? 1))
  const initial = readMeasure(monitor?.config ?? {})
  const [agg, setAgg] = useState<Agg>(initial.agg)
  const [column, setColumn] = useState(initial.column)
  const [key, setKey] = useState(initial.key)
  const [filters, setFilters] = useState<FilterRow[]>(readFilters(monitor?.config ?? {}))
  const [busy, setBusy] = useState(false)
  const [columns, setColumns] = useState<DerivationColumn[] | null>(null)
  const [columnsError, setColumnsError] = useState<string | null>(null)

  useEffect(() => {
    setColumns(null)
    setColumnsError(null)
    if (targetKind !== "derivation" || !target) return
    let current = true
    listDerivationColumns(target)
      .then((c) => current && setColumns(c))
      .catch((e) => current && setColumnsError(e instanceof Error ? e.message : String(e)))
    return () => {
      current = false
    }
  }, [targetKind, target])

  useEffect(() => {
    // Picking another kind clears the target; the monitor being edited keeps its own.
    if (!monitor || targetKind !== monitor.targetKind) setTarget("")
    else setTarget(monitor.target)
    const load =
      targetKind === "metric"
        ? listMetrics().then((m) => m.map((x) => x.name))
        : listDerivations().then((d) => d.map((x) => x.name))
    load.then(setTargets).catch(() => setTargets([]))
  }, [targetKind, monitor])

  // A derivation that cannot be read has no value to watch, so it cannot be chosen. A
  // monitor already watching one keeps saving, so its other settings stay editable.
  const unreadable =
    targetKind === "derivation" &&
    columnsError !== null &&
    !(monitor && monitor.targetKind === targetKind && monitor.target === target)

  const errors = {
    sensitivity: positive(sensitivity, "Sensitivity"),
    historyWindow: positive(historyWindow, "History window", true),
    intervalHours: positive(intervalHours, "Check interval"),
  }
  const invalid = Object.values(errors).some(Boolean)

  const submit = useCallback(async () => {
    setBusy(true)
    const base = monitor && monitor.targetKind === targetKind ? monitor.config : {}
    const config = buildConfig(base, targetKind, { agg, column, key, filters })
    const spec = {
      name: name || target,
      config,
      method,
      sensitivity: Number(sensitivity),
      minValue: minValue ? Number(minValue) : null,
      maxValue: maxValue ? Number(maxValue) : null,
      window: Number(historyWindow),
      intervalHours: Number(intervalHours),
    }
    try {
      // An edit names the target only when it changed: the server re-checks certification
      // for any edit that names one, which would stop a threshold change on a monitor
      // whose source has since lost certification.
      const saved = monitor
        ? await updateMonitor(monitor.id, {
            ...spec,
            targetKind: targetKind !== monitor.targetKind ? targetKind : undefined,
            target: target !== monitor.target ? target : undefined,
          })
        : await createMonitor({ ...spec, targetKind, target })
      onSaved(saved.id)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [
    monitor,
    targetKind,
    name,
    target,
    method,
    sensitivity,
    minValue,
    maxValue,
    historyWindow,
    intervalHours,
    agg,
    column,
    key,
    filters,
    onSaved,
    onError,
  ])

  const setFilter = (i: number, patch: Partial<FilterRow>) =>
    setFilters((rows) => rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{monitor ? "Edit monitor" : "New monitor"}</DialogTitle>
          <DialogDescription>
            Watch a certified metric or derivation. An anomaly against its learned baseline (or a
            static bound) raises one alert per incident.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <FormField label="Name">
            <Input
              placeholder="e.g. Workflow errors"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </FormField>
          <div className="flex gap-2">
            <FormField label="Watches" className="w-40">
              <Select
                value={targetKind}
                onValueChange={(v) => setTargetKind(v as "metric" | "derivation")}
              >
                <FormControl>
                  <SelectTrigger className="w-full text-sm">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="metric">metric</SelectItem>
                  <SelectItem value="derivation">derivation</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
            <FormField
              label={targetKind === "metric" ? "Metric" : "Derivation"}
              error={
                columnsError
                  ? `${unreadable ? "Cannot be monitored" : "Cannot be read right now"}: ${columnsError}`
                  : undefined
              }
              className="flex-1"
            >
              <Select value={target} onValueChange={setTarget}>
                <FormControl>
                  <SelectTrigger className="w-full text-sm">
                    <SelectValue placeholder={`Select a ${targetKind}`} />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {targets.map((t) => (
                    <SelectItem key={t} value={t}>
                      {t}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FormField>
          </div>
          {targetKind === "derivation" ? (
            <>
              <div className="flex gap-2">
                <FormField label="Aggregation" className="w-40">
                  <Select
                    value={agg}
                    onValueChange={(v) => setAgg(v as Agg)}
                    disabled={!column.trim()}
                  >
                    <FormControl>
                      <SelectTrigger className="w-full text-sm">
                        <SelectValue />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {AGGS.map((a) => (
                        <SelectItem key={a} value={a}>
                          {a}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </FormField>
                <FormField
                  label="Column"
                  hint="A numeric column to aggregate. Leave blank to watch the row count."
                  className="flex-1"
                >
                  {columns ? (
                    <Combobox
                      value={column}
                      onValueChange={setColumn}
                      options={columns.filter((c) => c.numeric).map((c) => ({ value: c.name }))}
                      clearLabel="(row count)"
                      searchPlaceholder="Search columns"
                    />
                  ) : (
                    <Input
                      placeholder="e.g. error_breach"
                      value={column}
                      onChange={(e) => setColumn(e.target.value)}
                    />
                  )}
                </FormField>
              </div>
              <FormField
                label="Key column (optional)"
                hint="Names each row, e.g. workflow_name, so alerts say which rows are breaching."
              >
                {columns ? (
                  <Combobox
                    value={key}
                    onValueChange={setKey}
                    options={columns.map((c) => ({ value: c.name }))}
                    clearLabel="(none)"
                    searchPlaceholder="Search columns"
                    disabled={!column.trim()}
                  />
                ) : (
                  <Input
                    placeholder="e.g. workflow_name"
                    value={key}
                    disabled={!column.trim()}
                    onChange={(e) => setKey(e.target.value)}
                  />
                )}
              </FormField>
            </>
          ) : (
            <div className="space-y-2">
              <p className="text-xs font-medium">Filters (optional)</p>
              {filters.map((f, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity beyond position
                <div key={i} className="flex gap-2">
                  <Input
                    aria-label="filter column"
                    placeholder="column"
                    value={f.column}
                    onChange={(e) => setFilter(i, { column: e.target.value })}
                  />
                  <Select
                    value={f.op}
                    onValueChange={(v) => setFilter(i, { op: v as FilterClause["op"] })}
                  >
                    <SelectTrigger className="w-24 text-sm" aria-label="filter operator">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {OPS.map((o) => (
                        <SelectItem key={o} value={o}>
                          {o}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input
                    aria-label="filter value"
                    placeholder={f.op === "in" ? "a, b, c" : "value"}
                    value={f.value}
                    onChange={(e) => setFilter(i, { value: e.target.value })}
                  />
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label="Remove filter"
                    onClick={() => setFilters((rows) => rows.filter((_, j) => j !== i))}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              ))}
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setFilters((rows) => [...rows, { column: "", op: "eq", value: "" }])}
              >
                <Plus className="size-4" /> Add filter
              </Button>
            </div>
          )}
          <div className="flex gap-2">
            <FormField label="Detection method" className="w-48">
              <Select value={method} onValueChange={(v) => setMethod(v as "mad" | "zscore")}>
                <FormControl>
                  <SelectTrigger className="w-full text-sm">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="mad">MAD (robust)</SelectItem>
                  <SelectItem value="zscore">Z-score</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
            <FormField
              label="Sensitivity (deviations)"
              error={errors.sensitivity}
              className="flex-1"
            >
              <Input
                type="number"
                min={0}
                value={sensitivity}
                onChange={(e) => setSensitivity(e.target.value)}
              />
            </FormField>
          </div>
          <p className="text-xs text-text-tertiary">{METHOD_HINT[method]}</p>
          <div className="flex gap-2">
            <FormField
              label="Min bound (optional)"
              hint="Alert below this, whatever the history."
              className="flex-1"
            >
              <Input type="number" value={minValue} onChange={(e) => setMinValue(e.target.value)} />
            </FormField>
            <FormField
              label="Max bound (optional)"
              hint="Alert above this, whatever the history."
              className="flex-1"
            >
              <Input type="number" value={maxValue} onChange={(e) => setMaxValue(e.target.value)} />
            </FormField>
          </div>
          <div className="flex gap-2">
            <FormField
              label="History window (checks)"
              hint="Past checks used to learn what is normal."
              error={errors.historyWindow}
              className="flex-1"
            >
              <Input
                type="number"
                min={1}
                step={1}
                value={historyWindow}
                onChange={(e) => setHistoryWindow(e.target.value)}
              />
            </FormField>
            <FormField label="Check every (hours)" error={errors.intervalHours} className="flex-1">
              <Input
                type="number"
                min={0}
                value={intervalHours}
                onChange={(e) => setIntervalHours(e.target.value)}
              />
            </FormField>
          </div>
        </div>
        <DialogFooter>
          <Button disabled={busy || !target || unreadable || invalid} onClick={submit}>
            {busy ? "Saving…" : monitor ? "Save monitor" : "Create monitor"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
