// API client and types for dashboards: declarative presentation surfaces whose
// widgets bind to certified derivations. Mirrors the throwing helper style in
// notebooks.ts. The spec is the Dashboard manifest (camelCase, as stored).

export type WidgetType = "metric" | "chart" | "map" | "table" | "text" | "filter"

export interface GridPos {
  x: number
  y: number
  w: number
  h: number
}

// A tile binds either a certified derivation (with params) or a semantic-layer metric
// (with groupBy/grain/filters). Exactly one of `derivation` or `metric` is set.
export interface Bind {
  derivation?: string
  params?: Record<string, unknown>
  metric?: string
  groupBy?: string[]
  grain?: "hour" | "day" | "week" | "month" | "quarter" | "year"
  filters?: { column: string; op: string; value: unknown }[]
}

export interface Interactions {
  crossFilter?: { emit: Record<string, string> }
  drillThrough?: { target: string; carry?: string[] }
  drillDown?: { hierarchy: string[]; param: string }
}

export interface Widget {
  id: string
  type: WidgetType
  gridPos: GridPos
  title?: string
  bind?: Bind
  viz?: Record<string, unknown>
  content?: string
  variable?: string
  interactions?: Interactions
}

export interface Page {
  name: string
  title?: string
  columns?: number
  widgets: Widget[]
}

export interface VariableOptions {
  values?: unknown[]
  derivation?: string
  column?: string
  labelColumn?: string
}

export interface Variable {
  name: string
  type: "string" | "integer" | "number" | "boolean" | "date"
  control?: "dropdown" | "multiselect" | "search" | "date" | "daterange" | "range" | "toggle"
  label?: string
  default?: unknown
  options?: VariableOptions
  scope?: string
}

export interface DashboardSpec {
  specVersion: string
  kind: "Dashboard"
  name: string
  title?: string
  description?: string
  theme?: "auto" | "light" | "dark"
  variables?: Variable[]
  pages: Page[]
  refresh?: { interval: string }
}

export interface Dashboard {
  id: string
  name: string
  title: string
  status: "draft" | "published"
  version: number
  spec: DashboardSpec
  publishedSpec: DashboardSpec | null
  updatedAt: string
}

export interface DashboardSummary {
  id: string
  name: string
  title: string
  // The dashboard this was copied from, or null. A record of origin, not a
  // reference: the source may since have been deleted.
  copiedFrom: string | null
  status: string
  version: number
  updatedAt: string
}

export interface WidgetData {
  widgetId: string
  derivation: string
  kind: string | null
  value: unknown
  dataVersion: string | null
  error: string | null
}

export interface Option {
  value: unknown
  label: string
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    const detail = await r.text()
    throw new Error(`${init?.method ?? "GET"} ${path} failed: ${r.status} ${detail}`)
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

function put<T>(path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
}

const id = encodeURIComponent

export const listDashboards = () => json<DashboardSummary[]>("/api/dashboards")

export const getDashboard = (dashboardId: string) =>
  json<Dashboard>(`/api/dashboards/${id(dashboardId)}`)

export const createDashboard = (spec: DashboardSpec) =>
  post<{ id: string }>("/api/dashboards", spec)

export const duplicateDashboard = (dashboardId: string) =>
  post<Dashboard>(`/api/dashboards/${id(dashboardId)}/duplicate`)

export const saveDashboard = (dashboardId: string, spec: DashboardSpec) =>
  put<Dashboard>(`/api/dashboards/${id(dashboardId)}`, { spec })

export const deleteDashboard = (dashboardId: string) =>
  json<{ ok: boolean }>(`/api/dashboards/${id(dashboardId)}`, { method: "DELETE" })

export const publishDashboard = (dashboardId: string) =>
  post<{ ok: boolean }>(`/api/dashboards/${id(dashboardId)}/publish`)

/** The certified derivations a widget may bind, by name. */
export async function bindableDerivations(): Promise<string[]> {
  const rows = await json<{ name: string }[]>("/api/dashboards/catalog")
  return rows.map((row) => row.name)
}

/** The DashboardSpec JSON Schema, for checking a spec as it is typed. */
export const dashboardSchema = () => json<Record<string, unknown>>("/api/dashboards/schema")

/** What a derivation reads, for the trail from a tile back to its sources. */
export interface Provenance {
  derivation: string[]
  dataset: string[]
}

export async function derivationProvenance(name: string): Promise<Provenance> {
  const body = await json<{ feedsFrom?: Partial<Provenance> }>(
    `/api/lineage/provenance?node=derivation:${encodeURIComponent(name)}`,
  )
  return {
    derivation: body.feedsFrom?.derivation ?? [],
    dataset: body.feedsFrom?.dataset ?? [],
  }
}

export async function resolvePage(
  dashboardId: string,
  page: string,
  variables: Record<string, unknown>,
  published = false,
): Promise<WidgetData[]> {
  const body = await post<{ widgets: WidgetData[] }>(
    `/api/dashboards/${id(dashboardId)}/pages/${id(page)}/data`,
    { variables, published },
  )
  return body.widgets
}

export async function variableOptions(dashboardId: string, variable: string): Promise<Option[]> {
  const body = await json<{ options: Option[] }>(
    `/api/dashboards/${id(dashboardId)}/variables/${id(variable)}/options`,
  )
  return body.options
}

// A blank starter dashboard: one empty page, ready to add widgets to.
export function starterSpec(name: string): DashboardSpec {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
  return {
    specVersion: "1.0",
    kind: "Dashboard",
    name: slug || "dashboard",
    title: name,
    pages: [{ name: "main", title: "Main", widgets: [] }],
  }
}
