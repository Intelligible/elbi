// API client and types for the exploration surface: ad-hoc SQL over bound datasets or
// a registered data source, schema browsing, one-click profiling, saved queries, and
// promotion of a query to a certified derivation.

export interface Source {
  id: string | null // null is the bound-datasets target
  name: string
  kind: string
}

export interface CatalogColumn {
  name: string
  type: string
}

export interface CatalogTable {
  name: string
  columns: CatalogColumn[]
  rows?: number
}

export type Cell = string | number | boolean | null

export interface QueryResult {
  columns: string[]
  rows: Record<string, Cell>[]
  rowCount: number
  truncated: boolean
}

export interface ColumnProfile {
  name: string
  count: number
  present: number
  completeness: number
  distinct: number
  inferredType: string
  minimum: number | null
  maximum: number | null
  topValues: { value: Cell; count: number }[]
  isUnique: boolean
  fractionUniqueOnce: number
}

export interface ProfileResult {
  rowCount: number
  columns: ColumnProfile[]
}

export interface SavedQuery {
  id: string
  name: string
  sql: string
  sourceId: string | null
  // The query this was copied from, or null. A record of origin, not a
  // reference: the source may since have been deleted.
  copiedFrom: string | null
  createdAt: string
  updatedAt: string
}

export interface PromoteResult {
  ok: boolean
  name?: string
  certified?: boolean
  verdict?: string | null
  rendered?: string
  error?: string | null
  detail?: string | null
  flowId?: string | null
}

// The statistical claim a question asserts, as column roles the oracle checks:
// x drives y, optionally adjusted for controls.
export interface DraftClaim {
  x: string
  y: string
  controls?: string[]
}

// A verify-only draft: the verdict on a proposed derivation, certifying nothing.
// `ok` mirrors the certify gate, so Certify is enabled only when it would succeed.
export interface DraftResult {
  ok: boolean
  name?: string
  certified?: boolean
  verdict?: string | null
  contractVerdict?: string | null
  checks?: [string, string, string][]
  rendered?: string
  detail?: string | null
  error?: string | null
  sql?: string
  claim?: DraftClaim | null
  flowId?: string | null
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

export const getSources = () => json<Source[]>("/api/explore/sources")

export const getCatalog = (sourceId: string | null) =>
  json<{ tables: CatalogTable[] }>(
    `/api/explore/catalog${sourceId ? `?source_id=${id(sourceId)}` : ""}`,
  )

export const runQuery = (sql: string, sourceId: string | null, maxRows?: number) =>
  post<QueryResult>("/api/explore/query", {
    sql,
    sourceId,
    maxRows,
  })

export const profile = (opts: { dataset?: string; sql?: string; sourceId?: string | null }) =>
  post<ProfileResult>("/api/explore/profile", {
    dataset: opts.dataset,
    sql: opts.sql,
    sourceId: opts.sourceId,
  })

export const listQueries = () => json<SavedQuery[]>("/api/explore/queries")

export const saveQuery = (query: {
  id?: string
  name: string
  sql: string
  sourceId: string | null
}) =>
  post<SavedQuery>("/api/explore/queries", {
    id: query.id,
    name: query.name,
    sql: query.sql,
    sourceId: query.sourceId,
  })

export const duplicateQuery = (queryId: string) =>
  post<SavedQuery>(`/api/explore/queries/${id(queryId)}/duplicate`)

export const deleteQuery = (queryId: string) =>
  json<{ ok: boolean }>(`/api/explore/queries/${id(queryId)}`, { method: "DELETE" })

export const promoteQuery = (name: string, sql: string, sourceId: string | null) =>
  post<PromoteResult>("/api/explore/promote", { name, sql, sourceId })

export const nl2sql = (prompt: string, sourceId: string | null) =>
  post<{ sql: string }>("/api/explore/nl2sql", { prompt, sourceId })

// The one-click loop: draft -> verdict -> certify. `flowId` pairs the two calls in
// the server's loop-latency logs.
export const draftDerivation = (
  name: string,
  sql: string,
  prompt: string,
  sourceId: string | null,
  flowId: string,
) => post<DraftResult>("/api/explore/draft", { name, sql, prompt, sourceId, flowId })

export const certifyDerivation = (
  name: string,
  sql: string,
  sourceId: string | null,
  flowId: string,
  claim?: DraftClaim | null,
) => post<PromoteResult>("/api/explore/certify", { name, sql, sourceId, flowId, claim })
