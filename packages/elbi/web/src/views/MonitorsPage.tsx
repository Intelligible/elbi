import { Activity, AlertTriangle, Play, Plus, Trash2 } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
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
import { listDerivations, listMetrics } from "@/lib/metrics"
import {
  type CheckResult,
  checkMonitor,
  createMonitor,
  deleteMonitor,
  getMonitor,
  listMonitors,
  type Monitor,
  type MonitorHistory,
} from "@/lib/monitors"

export function MonitorsPage() {
  const [monitors, setMonitors] = useState<Monitor[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

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
        description="Watch a certified metric or derivation; an anomaly against its learned baseline raises an alert carrying the oracle's verdict."
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

        <div className="flex min-h-0 flex-1 flex-col">
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

      {creating ? (
        <NewMonitorDialog
          onClose={() => setCreating(false)}
          onCreated={(id) => {
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
  onDeleted,
}: {
  monitor: Monitor
  onError: (message: string) => void
  onChanged: () => void
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
    <div className="flex min-h-0 flex-1 flex-col">
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
                <li key={incident.id} className="flex items-center gap-3 px-3 py-2 text-sm">
                  {incident.open ? (
                    <Badge variant="danger">open</Badge>
                  ) : (
                    <Badge variant="neutral">resolved</Badge>
                  )}
                  <span className="flex-1 truncate text-text-secondary">{incident.reason}</span>
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

function NewMonitorDialog({
  onClose,
  onCreated,
  onError,
}: {
  onClose: () => void
  onCreated: (id: string) => void
  onError: (message: string) => void
}) {
  const [targetKind, setTargetKind] = useState<"metric" | "derivation">("metric")
  const [targets, setTargets] = useState<string[]>([])
  const [name, setName] = useState("")
  const [target, setTarget] = useState("")
  const [method, setMethod] = useState<"mad" | "zscore">("mad")
  const [sensitivity, setSensitivity] = useState("3")
  const [minValue, setMinValue] = useState("")
  const [maxValue, setMaxValue] = useState("")
  const [column, setColumn] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setTarget("")
    const load =
      targetKind === "metric"
        ? listMetrics().then((m) => m.map((x) => x.name))
        : listDerivations().then((d) => d.map((x) => x.name))
    load.then(setTargets).catch(() => setTargets([]))
  }, [targetKind])

  const submit = useCallback(async () => {
    setBusy(true)
    const config: Record<string, unknown> =
      targetKind === "derivation" && column.trim()
        ? { measure: { agg: "mean", column: column.trim() } }
        : {}
    try {
      const monitor = await createMonitor({
        name: name || target,
        targetKind,
        target,
        config,
        method,
        sensitivity: Number(sensitivity) || 3,
        minValue: minValue ? Number(minValue) : null,
        maxValue: maxValue ? Number(maxValue) : null,
      })
      onCreated(monitor.id)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [
    targetKind,
    name,
    target,
    method,
    sensitivity,
    minValue,
    maxValue,
    column,
    onCreated,
    onError,
  ])

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New monitor</DialogTitle>
          <DialogDescription>
            Watch a certified metric or derivation. An anomaly against its learned baseline (or a
            static bound) raises one alert per incident.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <Input
            placeholder="monitor name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <div className="flex gap-2">
            <Select
              value={targetKind}
              onValueChange={(v) => setTargetKind(v as "metric" | "derivation")}
            >
              <SelectTrigger className="w-40 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="metric">metric</SelectItem>
                <SelectItem value="derivation">derivation</SelectItem>
              </SelectContent>
            </Select>
            <Select value={target} onValueChange={setTarget}>
              <SelectTrigger className="text-sm">
                <SelectValue placeholder={`Select a ${targetKind}`} />
              </SelectTrigger>
              <SelectContent>
                {targets.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {targetKind === "derivation" ? (
            <Input
              placeholder="numeric column to average (blank = row count)"
              value={column}
              onChange={(e) => setColumn(e.target.value)}
            />
          ) : null}
          <div className="flex gap-2">
            <Select value={method} onValueChange={(v) => setMethod(v as "mad" | "zscore")}>
              <SelectTrigger className="w-40 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mad">mad (robust)</SelectItem>
                <SelectItem value="zscore">zscore</SelectItem>
              </SelectContent>
            </Select>
            <Input
              type="number"
              placeholder="sensitivity (σ)"
              value={sensitivity}
              onChange={(e) => setSensitivity(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <Input
              type="number"
              placeholder="min bound (optional)"
              value={minValue}
              onChange={(e) => setMinValue(e.target.value)}
            />
            <Input
              type="number"
              placeholder="max bound (optional)"
              value={maxValue}
              onChange={(e) => setMaxValue(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button disabled={busy || !target} onClick={submit}>
            {busy ? "Creating…" : "Create monitor"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
