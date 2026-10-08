// API client and types for the semantic-layer metrics surface: define metrics over
// certified derivations, query them by dimension/grain/filter, and interchange via OSI.

export type Agg = "sum" | "average" | "count" | "count_distinct" | "min" | "max" | "median"

export type Grain = "hour" | "day" | "week" | "month" | "quarter" | "year"

export interface Measure {
  agg: Agg
  column?: string
}

export interface TimeDimension {
  column: string
  grain?: Grain
}

export interface FilterClause {
  column: string
  op: "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "in"
  value: unknown
}

export type MetricType = "simple" | "ratio" | "derived" | "cumulative"

export type FormatKind = "number" | "currency" | "percent" | "duration"

export interface Format {
  kind: FormatKind
  currency?: string
  precision?: number
  notation?: "standard" | "compact" | "scientific"
  pattern?: string
}

export interface Metric {
  name: string
  type: MetricType
  label?: string
  // The metric this was copied from, by name. Optional because the list view
  // omits it while a detail read and a duplicate response carry it.
  copiedFrom?: string | null
  description?: string
  source?: string
  measure?: Measure
  numerator?: string
  denominator?: string
  expr?: string
  inputMetrics?: string[]
  inputMetric?: string
  window?: number
  periodAgg?: "sum" | "average" | "min" | "max"
  dimensions?: string[]
  timeDimension?: TimeDimension
  filters?: FilterClause[]
  format?: Format
  // Attached by the server: whether the metric's source derivation is certified.
  sourceCertified?: boolean
}

export interface MetricOverview {
  name: string
  value: number | null
  series: { t: string; v: number }[]
}

export interface MetricVersion {
  version: number
  createdAt: string
  author: string
  verdict: string | null
  changeSummary: string | null
  definition: Metric
}

export interface AskResult {
  columns: string[]
  rows: Record<string, Cell>[]
  resolvedQuery: {
    metricName: string
    groupBy: string[]
    grain: Grain | null
    filters: FilterClause[]
    explanation?: string
  }
  explanation?: string
}

export type Cell = string | number | boolean | null

export interface MetricQueryResult {
  columns: string[]
  rows: Record<string, Cell>[]
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

export const listMetrics = () => json<Metric[]>("/api/metrics")

export const defineMetric = (metric: Metric) => post<Metric>("/api/metrics", metric)

export const duplicateMetric = (name: string) => post<Metric>(`/api/metrics/${id(name)}/duplicate`)

export const deleteMetric = (name: string) =>
  json<{ ok: boolean }>(`/api/metrics/${id(name)}`, { method: "DELETE" })

export const queryMetric = (
  name: string,
  body: { groupBy?: string[]; grain?: Grain; filters?: FilterClause[] },
) => post<MetricQueryResult>(`/api/metrics/${id(name)}/query`, body)

export const metricsOverview = () => json<MetricOverview[]>("/api/metrics/overview")

export const compileMetric = (
  name: string,
  body: { groupBy?: string[]; grain?: Grain; filters?: FilterClause[] },
) => post<{ sql: string }>(`/api/metrics/${id(name)}/compile`, body)

export const metricHistory = (name: string) =>
  json<MetricVersion[]>(`/api/metrics/${id(name)}/history`)

export const askMetric = (question: string) => post<AskResult>("/api/metrics/ask", { question })

export const exportOsi = () => json<Record<string, unknown>>("/api/metrics/osi")

export const importOsi = (document: Record<string, unknown>) =>
  post<{ imported: Metric[] }>("/api/metrics/osi", { document })

// Certified derivations are the only valid metric sources; the picker reads them here.
export const listDerivations = () => json<{ name: string }[]>("/api/derivations")

/** Format a metric cell, honoring the metric's display format (Intl.NumberFormat). */
export function formatMetricValue(value: Cell, format?: Format): string {
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
