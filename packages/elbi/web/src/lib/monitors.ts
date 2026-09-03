// API client and types for anomaly monitors: watch a metric or derivation over time,
// detect anomalies against a learned baseline, and alert.

export interface Monitor {
  id: string
  name: string
  targetKind: "metric" | "derivation"
  target: string
  config: Record<string, unknown>
  method: "mad" | "zscore"
  sensitivity: number
  minValue: number | null
  maxValue: number | null
  window: number
  intervalHours: number
  enabled: boolean
  lastValue: number | null
  lastCheckedAt: string | null
  status: "ok" | "alerting"
}

export interface Snapshot {
  at: string | null
  value: number
  anomalous: boolean
  baseline: number | null
  lower: number | null
  upper: number | null
  score: number | null
  reason: string
  sourceVerdict: string | null
}

export interface Incident {
  id: string
  openedAt: string | null
  closedAt: string | null
  peakValue: number
  peakScore: number | null
  reason: string
  snapshots: number
  open: boolean
}

export interface MonitorHistory {
  snapshots: Snapshot[]
  incidents: Incident[]
}

export interface CheckResult {
  value: number
  anomalous: boolean
  baseline: number | null
  lower: number | null
  upper: number | null
  score: number | null
  reason: string
  alerted: boolean
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let detail = `${r.status}`
    try {
      const body = (await r.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // A non-JSON error body leaves the status code as the message.
    }
    throw new Error(detail)
  }
  return (await r.json()) as T
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
}

const id = encodeURIComponent

export const listMonitors = () => json<Monitor[]>("/api/monitors")

export const createMonitor = (spec: {
  name: string
  targetKind: "metric" | "derivation"
  target: string
  config?: Record<string, unknown>
  method?: "mad" | "zscore"
  sensitivity?: number
  minValue?: number | null
  maxValue?: number | null
  window?: number
  intervalHours?: number
}) => post<Monitor>("/api/monitors", spec)

export const getMonitor = (monitorId: string) =>
  json<MonitorHistory>(`/api/monitors/${id(monitorId)}`)

export const checkMonitor = (monitorId: string) =>
  post<CheckResult>(`/api/monitors/${id(monitorId)}/check`)

export const deleteMonitor = (monitorId: string) =>
  json<{ ok: boolean }>(`/api/monitors/${id(monitorId)}`, { method: "DELETE" })
