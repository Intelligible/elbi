// API client and types for the feature store: entities, feature views, materialization,
// and online / point-in-time retrieval. Mirrors the throwing helper style in dashboards.ts.

export interface FeatureEntity {
  name: string
  joinKey: string
  valueType: string
  description: string | null
}

export interface FeatureColumn {
  name: string
  dtype: string | null
  description: string | null
}

export interface FeatureView {
  name: string
  entities: string[]
  joinKeys: string[]
  source: string
  certified: boolean
  timestampField: string | null
  ttlSeconds: number | null
  features: FeatureColumn[]
  description: string | null
  nOnlineKeys: number
  lastMaterializedAt: string | null
}

export interface FeatureViewInput {
  name: string
  entities: string[]
  source: string
  features?: { name: string; dtype?: string; description?: string }[]
  timestampField?: string
  ttlSeconds?: number
  description?: string
}

export type Row = Record<string, unknown>

export interface FeatureProfile {
  name: string
  count: number
  present: number
  completeness: number
  distinct: number
  inferredType: string
  minimum: number | null
  maximum: number | null
  topValues: { value: unknown; count: number }[]
}

export interface StatisticsSnapshot {
  id: string
  at: string
  rowCount: number
  isBaseline: boolean
  features: FeatureProfile[]
}

export interface ColumnDrift {
  column: string
  method: string
  score: number
  threshold: number
  drifted: boolean
}

export interface DriftCheck {
  id: string
  at: string
  nColumns: number
  nDrifted: number
  shareDrifted: number
  datasetDrift: boolean
  nCurrentRows: number
  columns: ColumnDrift[]
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

const id = encodeURIComponent

export const listEntities = () => json<FeatureEntity[]>("/api/features/entities")

export const defineEntity = (entity: {
  name: string
  join_key: string
  value_type?: string
  description?: string
}) => post<{ ok: boolean }>("/api/features/entities", entity)

export const listFeatureViews = () => json<FeatureView[]>("/api/features/views")

export interface ExpectationCheck {
  id: string
  at: string
  verdict: string
  nClauses: number
  nViolated: number
  rowCount: number
}

export interface FeatureViewDetail extends FeatureView {
  hasContract: boolean
  statistics: StatisticsSnapshot[]
  drift: DriftCheck[]
  expectations: ExpectationCheck[]
}

export const getFeatureViewDetail = (name: string) =>
  json<FeatureViewDetail>(`/api/features/views/${id(name)}/detail`)

export const defineFeatureView = (view: FeatureViewInput) =>
  post<FeatureView>("/api/features/views", view)

export const deleteFeatureView = (name: string) =>
  json<{ ok: boolean }>(`/api/features/views/${id(name)}`, { method: "DELETE" })

export const materializeFeatures = (featureViews?: string[]) =>
  post<{ written: Record<string, number> }>("/api/features/materialize", {
    feature_views: featureViews ?? null,
  })

export async function onlineFeatures(features: string[], entityRows: Row[]): Promise<Row[]> {
  const body = await post<{ rows: Row[] }>("/api/features/online", {
    features,
    entity_rows: entityRows,
  })
  return body.rows
}

export async function historicalFeatures(features: string[], entityDf: Row[]): Promise<Row[]> {
  const body = await post<{ rows: Row[] }>("/api/features/historical", {
    features,
    entity_df: entityDf,
  })
  return body.rows
}

export const snapshotStatistics = (view: string, setBaseline: boolean) =>
  post<StatisticsSnapshot>(`/api/features/views/${id(view)}/statistics`, {
    set_baseline: setBaseline,
  })

export const checkDrift = (view: string) =>
  post<DriftCheck & { featureView: string }>(`/api/features/views/${id(view)}/drift-check`)

export interface ExpectationClause {
  name: string
  verdict: string
  detail: string
}

export interface ExpectationResult {
  featureView: string
  verdict: string
  nClauses: number
  nViolated: number
  rowCount: number
  clauses: ExpectationClause[]
}

export const suggestContract = (view: string) =>
  post<{ contract: unknown }>(`/api/features/views/${id(view)}/contract/suggest`).then(
    (b) => b.contract,
  )

export const setContract = (view: string, contract: unknown | null) =>
  json<{ contract: unknown | null }>(`/api/features/views/${id(view)}/contract`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contract }),
  })

export const checkExpectations = (view: string) =>
  post<ExpectationResult>(`/api/features/views/${id(view)}/expectations-check`)

export interface TrainingSet {
  name: string
  createdAt: string
  rowCount: number
  features: string[]
  label: string | null
}

export const listTrainingSets = () => json<TrainingSet[]>("/api/features/training-sets")

export const deleteTrainingSet = (name: string) =>
  json<{ ok: boolean }>(`/api/features/training-sets/${id(name)}`, {
    method: "DELETE",
  })
