// API client and types for orchestration: asset staleness, materialization runs,
// and schedules.

export type Freshness = "materialized" | "stale" | "never"
export type StepState = "succeeded" | "failed" | "skipped"

export interface AssetStatus {
  asset: string
  status: Freshness
  verdict: string | null
  lastMaterializedAt: string | null
  lastDurationMs: number | null
}

export type RunStatus = "running" | "succeeded" | "failed" | "cancelled"

export type RunIf =
  | "all_success"
  | "at_least_one_success"
  | "none_failed"
  | "all_done"
  | "at_least_one_failed"
  | "all_failed"

// A workflow step: materialize a selection, gated by a trigger rule over its upstreams.
export interface WorkflowStep {
  id: string
  selection?: "stale" | "all"
  assets?: string[]
  includeDownstream?: boolean
  dependsOn?: string[]
  runIf?: RunIf
}

export interface Workflow {
  id: string
  name: string
  steps: WorkflowStep[]
}

export interface RunSummary {
  id: string
  cause: string
  status: RunStatus
  parentRunId: string | null
  startedAt: string
  finishedAt: string | null
  counts: Record<string, number>
}

export type CheckSeverity = "warn" | "error"

// A data-quality check definition on an asset (a boolean SQL expr over its columns).
export interface AssetCheck {
  id: string
  asset: string
  name: string
  expr: string
  severity: CheckSeverity
  enabled: boolean
}

// A check's outcome within a run. `state` distinguishes rows that violated the
// constraint from an expression that would not run at all, which is not evidence about
// the data and so carries no count.
export interface CheckResult {
  name: string
  severity: CheckSeverity
  expr: string
  state: "passed" | "failed" | "error"
  failed: number | null
  total: number
  error?: string
}

export interface RunStep {
  asset: string
  state: StepState
  verdict: string | null
  error: string | null
  attempts: number
  durationMs: number
  logs: string
  checks: CheckResult[]
}

export interface RunDetail extends Omit<RunSummary, "counts"> {
  steps: RunStep[]
}

// The asset dependency DAG: derivation nodes and their derivation→derivation edges.
export interface GraphNode {
  asset: string
  status: Freshness
  verdict: string | null
}

export interface GraphEdge {
  from: string
  to: string
}

export interface AssetGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// The async materialize/retry response: a run id to poll and its initial status.
export interface StartedRun {
  runId: string
  status: RunStatus | "done"
}

// The run-history matrix: recent runs, each with its per-asset cell state.
export interface HistoryRun extends RunSummary {
  cells: Record<string, StepState>
}

export interface RunHistory {
  assets: string[]
  runs: HistoryRun[]
}

export interface Schedule {
  id: string
  name: string
  selection: string
  mode: string
  cron: string
  dataset: string | null
  enabled: boolean
  lastRunAt: string | null
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) throw new Error(`${init?.method ?? "GET"} ${path} failed: ${r.status}`)
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

export const getAssetStatus = () => json<AssetStatus[]>("/api/orchestration/status")

export const getGraph = () => json<AssetGraph>("/api/orchestration/graph")

export const getHistory = () => json<RunHistory>("/api/orchestration/history")

export const getSettings = () =>
  json<{ maxRetries: number; compute: string }>("/api/orchestration/settings")

export const setRetryPolicy = (maxRetries: number) =>
  json<{ maxRetries: number }>("/api/orchestration/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ maxRetries }),
  })

export const getChecks = (asset?: string) =>
  json<AssetCheck[]>(`/api/orchestration/checks${asset ? `?asset=${id(asset)}` : ""}`)

export const upsertCheck = (check: {
  id?: string
  asset: string
  name: string
  expr: string
  severity?: CheckSeverity
  enabled?: boolean
}) => post<{ id: string }>("/api/orchestration/checks", check)

export const deleteCheck = (checkId: string) =>
  json<{ ok: boolean }>(`/api/orchestration/checks/${id(checkId)}`, {
    method: "DELETE",
  })

export const materialize = (
  body: { selection?: "stale" | "all"; assets?: string[]; includeDownstream?: boolean } = {},
) => post<StartedRun>("/api/orchestration/materialize", body)

export const retryRun = (runId: string) =>
  post<StartedRun>(`/api/orchestration/runs/${id(runId)}/retry`)

export const backfill = (body: { asset: string; param: string; values: string[] }) =>
  post<StartedRun>("/api/orchestration/backfill", body)

export const cancelRun = (runId: string) =>
  post<{ ok: boolean }>(`/api/orchestration/runs/${id(runId)}/cancel`)

export const getRun = (runId: string) => json<RunDetail>(`/api/orchestration/runs/${id(runId)}`)

export const getWorkflows = () => json<Workflow[]>("/api/orchestration/workflows")

export const upsertWorkflow = (wf: { id?: string; name: string; steps: WorkflowStep[] }) =>
  post<{ id: string }>("/api/orchestration/workflows", wf)

export const deleteWorkflow = (workflowId: string) =>
  json<{ ok: boolean }>(`/api/orchestration/workflows/${id(workflowId)}`, {
    method: "DELETE",
  })

export const runWorkflow = (workflowId: string) =>
  post<StartedRun>(`/api/orchestration/workflows/${id(workflowId)}/run`)

export const getSchedules = () => json<Schedule[]>("/api/orchestration/schedules")

export const createSchedule = (schedule: {
  name: string
  selection: string
  mode: string
  cron?: string
  dataset?: string
}) => post<{ id: string }>("/api/orchestration/schedules", schedule)

export const deleteSchedule = (scheduleId: string) =>
  json<{ ok: boolean }>(`/api/orchestration/schedules/${id(scheduleId)}`, {
    method: "DELETE",
  })
