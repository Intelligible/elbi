// API client and types for the data warehouse: configure external sources (SQL DBs,
// files, SaaS APIs), sync their tables into the Delta Lake lakehouse, and query them.

export type FieldType = "text" | "password" | "number" | "switch" | "select" | "textarea"

export interface SourceField {
  name: string
  label: string
  type: FieldType
  required: boolean
  placeholder: string
  default: unknown
  options: { value: string; label: string }[]
  caption: string
  // A heading this field sits under. Fields sharing one render as a group; empty means
  // the connector's main body. Keeps an optional block from reading as more required
  // input than it is.
  section?: string
  // The field this one is conditional on: shown only when that field holds
  // `dependsValue`, or any truthy value when `dependsValue` is empty.
  dependsOn?: string
  dependsValue?: string
  // When true, the form offers a file upload for this field; the uploaded file's stored
  // warehouse path fills the field value (the CSV / Parquet source).
  upload?: boolean
  // Whether the field holds a credential. The server resolves this, so a PEM key or a
  // JSON key file -- a secret that renders as a textarea, not a password box -- is
  // flagged as one; never infer secrecy from `type` here.
  secret?: boolean
}

export interface SourceConfig {
  name: string
  label: string
  category: string
  fields: SourceField[]
  caption: string
  icon: string
  releaseStatus: "alpha" | "beta" | "ga"
  docsUrl: string
  comingSoon: boolean
}

export interface Catalog {
  sources: SourceConfig[]
  categories: string[]
}

export type SyncFrequency = "manual" | "30min" | "1hour" | "6hour" | "12hour" | "day" | "week"

export interface SourceSummary {
  id: string
  name: string
  sourceType: string
  description: string
  prefix: string
  syncFrequency: SyncFrequency
  status: string
  lastError: string | null
  lastSyncedAt: string | null
  createdAt: string | null
  schemaCount: number
  syncedCount: number
  rows: number
}

export interface SchemaView {
  id: string
  name: string
  table: string
  shouldSync: boolean
  syncType: "full_refresh" | "incremental"
  incrementalField: string | null
  incrementalFields: string[]
  status: string
  rowCount: number | null
  lastError: string | null
  lastSyncedAt: string | null
}

export interface SourceDetail extends SourceSummary {
  schemas: SchemaView[]
}

export interface SyncOutcome {
  table: string
  rows: number
  ok: boolean
  error: string | null
}

export interface SyncResult {
  outcomes: SyncOutcome[]
  source: SourceDetail
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

function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
}

const id = encodeURIComponent

export const getCatalog = () => json<Catalog>("/api/warehouse/catalog")

export const registerInterest = (source_type: string) =>
  send<{ ok: boolean }>("POST", "/api/warehouse/interest", { source_type })

export const listSources = () => json<SourceSummary[]>("/api/warehouse/sources")

export const getSource = (sourceId: string) =>
  json<SourceDetail>(`/api/warehouse/sources/${id(sourceId)}`)

// Upload a local CSV/Parquet file into the warehouse; returns the stored path to use as the
// source's `path`. No Content-Type header: the browser sets the multipart boundary itself,
// which a manual header would clobber.
export const uploadWarehouseFile = (file: File) => {
  const form = new FormData()
  form.append("file", file)
  return json<{ path: string; filename: string }>("/api/warehouse/uploads", {
    method: "POST",
    body: form,
  })
}

export const createSource = (spec: {
  source_type: string
  name: string
  config: Record<string, unknown>
  description?: string
  prefix?: string
}) => send<SourceDetail>("POST", "/api/warehouse/sources", spec)

export const setSyncFrequency = (sourceId: string, sync_frequency: SyncFrequency) =>
  send<SourceDetail>("PATCH", `/api/warehouse/sources/${id(sourceId)}`, {
    sync_frequency,
  })

export interface SourceConfigView {
  sourceType: string
  /**
   * Keyed by the connector's own field names (`auth_token`, not `authToken`): the API
   * treats `config` as opaque so these survive the casing boundary. Secrets come back
   * blank, and an edit sends them back blank to keep them.
   */
  config: Record<string, unknown>
  secretFields: string[]
}

export const getSourceConfig = (sourceId: string) =>
  json<SourceConfigView>(`/api/warehouse/sources/${id(sourceId)}/config`)

/**
 * Edit a source in place.
 *
 * A password field left blank keeps the stored secret, so a manifest can be corrected
 * without the operator re-entering credentials that were already working.
 */
export const updateSource = (
  sourceId: string,
  patch: {
    config?: Record<string, unknown>
    name?: string
    description?: string
  },
) => send<SourceDetail>("PATCH", `/api/warehouse/sources/${id(sourceId)}`, patch)

export const deleteSource = (sourceId: string) =>
  json<{ ok: boolean }>(`/api/warehouse/sources/${id(sourceId)}`, {
    method: "DELETE",
  })

export const syncSource = (sourceId: string) =>
  send<SyncResult>("POST", `/api/warehouse/sources/${id(sourceId)}/sync`)

export const updateSchema = (
  schemaId: string,
  patch: {
    should_sync?: boolean
    sync_type?: "full_refresh" | "incremental"
    incremental_field?: string | null
  },
) => send<SchemaView>("PATCH", `/api/warehouse/schemas/${id(schemaId)}`, patch)
