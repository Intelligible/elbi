// The app's native API: models, datasets, and the authored derivations. Chat streaming
// itself is handled by the AI SDK's useChat over /api/chat, so there is no hand-rolled
// SSE parser here.

// Fetch JSON, distinguishing a real failure (network error or non-2xx) from an empty
// result: on failure it logs (so the failure is observable, not silent) and returns the
// caller's fallback, so a degraded list never masquerades as "there is nothing here".
async function getJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const r = await fetch(path)
    if (!r.ok) {
      console.error(`GET ${path} failed: ${r.status} ${r.statusText}`)
      return fallback
    }
    return (await r.json()) as T
  } catch (err) {
    console.error("GET failed", path, err)
    return fallback
  }
}

export interface Model {
  id: string
  name: string
  provider: string
  default: boolean
}

export async function getModels(): Promise<Model[]> {
  return getJson<Model[]>("/api/models", [])
}

// A named LLM config. `apiKeySet` reports whether a key is stored; the value itself is
// never returned, so the form shows a masked placeholder rather than a secret. Several
// profiles let a user register (say) OpenAI and Claude and pick which one a chat uses.
export interface LlmProfile {
  name: string
  model: string
  baseUrl: string
  apiKeySet: boolean
  reasoningEffort: string
  // Resolved server-side, because working it out needs the model registry: the provider
  // a model actually routes to (a bare `gpt-5.6-terra` is OpenAI; `bedrock/us.anthropic.…`
  // routes to Bedrock), and the reasoning budgets this model will accept. An empty
  // `reasoningEfforts` means the model has no reasoning to budget, so offer no control.
  provider: string
  // What the provider *is* -- "maker" (the lab whose model it is), "cloud" (a host
  // serving someone else's model) or "local" (a model on your own hardware). Empty for
  // an unclassified provider. LiteLLM routes to ~145 of them, so the picker draws a mark
  // per role and layers a real logo on top for the few a simple glyph can carry.
  providerLabel: string
  reasoningEfforts: string[]
}

export interface LlmProfiles {
  profiles: LlmProfile[]
  default: string
  titleProfile: string
  // Live check of whether a chat send would actually go through right now (a
  // profile, or LLM_MODEL/--model) -- not just whether any profile exists, since
  // an env-configured install has none. Defaults true on a fetch failure so a
  // transient network error never shows a false "not configured" banner.
  configured: boolean
}

export async function getLlmProfiles(): Promise<LlmProfiles> {
  return getJson<LlmProfiles>("/api/settings/llm/profiles", {
    profiles: [],
    default: "",
    titleProfile: "",
    configured: true,
  })
}

export async function setTitleProfile(name: string): Promise<boolean> {
  return mutate("/api/settings/llm/title-profile", "PUT", { name })
}

// Download a conversation as JSON via the server's attachment endpoint.
export function exportConversation(id: string): void {
  const a = document.createElement("a")
  a.href = `/api/conversations/${encodeURIComponent(id)}/export`
  a.download = ""
  document.body.appendChild(a)
  a.click()
  a.remove()
}

async function mutate(path: string, method: string, body?: unknown): Promise<boolean> {
  try {
    const r = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    })
    if (!r.ok) {
      console.error(`${method} ${path} failed: ${r.status} ${r.statusText}`)
      return false
    }
    return true
  } catch (err) {
    console.error(`${method} ${path} failed`, err)
    return false
  }
}

// Create or update a profile. Omit `api_key` to keep the stored one; send "" to clear it.
// Save a profile, throwing the server's reason (e.g. an invalid name) on failure so the
// form can show *why* rather than failing silently.
// The patch keys stay snake_case while the fields read above are camelCase, and that is
// deliberate: the API camelCases what it *sends* but leaves an already-snake_case request
// key alone (casing.py), and the handler reads `base_url`/`api_key` directly.
export async function saveLlmProfile(
  name: string,
  patch: { model?: string; api_key?: string; base_url?: string },
): Promise<void> {
  const path = `/api/settings/llm/profiles/${encodeURIComponent(name)}`
  const r = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const body = (await r.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // A non-JSON error body leaves the status line as the message.
    }
    console.error(`PUT ${path} failed: ${detail}`)
    throw new Error(detail)
  }
}

export async function deleteLlmProfile(name: string): Promise<boolean> {
  return mutate(`/api/settings/llm/profiles/${encodeURIComponent(name)}`, "DELETE")
}

export async function setDefaultProfile(name: string): Promise<boolean> {
  return mutate("/api/settings/llm/default", "PUT", { name })
}

// -- data sources -------------------------------------------------------------
export interface DataSource {
  id: string
  name: string
  kind: string
  host: string | null
  port: number | null
  database: string | null
  username: string | null
  secretSet: boolean
  secretEnv: string | null
  extra: string | null
}

export type DataSourceInput = {
  name?: string
  kind?: string
  host?: string
  port?: number | null
  database?: string
  username?: string
  secret?: string
  secret_env?: string
}

export async function getDataSources(): Promise<DataSource[]> {
  return getJson<DataSource[]>("/api/data-sources", [])
}

export async function createDataSource(input: DataSourceInput): Promise<boolean> {
  return mutate("/api/data-sources", "POST", input)
}

export async function updateDataSource(id: string, input: DataSourceInput): Promise<boolean> {
  return mutate(`/api/data-sources/${encodeURIComponent(id)}`, "PUT", input)
}

export async function deleteDataSource(id: string): Promise<boolean> {
  return mutate(`/api/data-sources/${encodeURIComponent(id)}`, "DELETE")
}

// Test a connection (saved by id, or an unsaved payload). Returns {ok, error?}.
export async function testDataSource(
  input: DataSourceInput | { id: string },
): Promise<{ ok: boolean; error?: string }> {
  const path =
    "id" in input
      ? `/api/data-sources/${encodeURIComponent(input.id)}/test`
      : "/api/data-sources/test"
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify("id" in input ? {} : input),
    })
    if (!r.ok) return { ok: false, error: `${r.status} ${r.statusText}` }
    return (await r.json()) as { ok: boolean; error?: string }
  } catch (err) {
    return { ok: false, error: String(err) }
  }
}

// -- budget -------------------------------------------------------------------
export interface Budget {
  maxBudget: number
  window: string
  spend: number
}

export async function getBudget(): Promise<Budget> {
  return getJson<Budget>("/api/settings/budget", {
    maxBudget: 0,
    window: "30d",
    spend: 0,
  })
}

export async function setBudget(max_budget: number, window: string): Promise<boolean> {
  return mutate("/api/settings/budget", "PUT", { max_budget, window })
}

// -- secrets ------------------------------------------------------------------
export interface SecretMeta {
  name: string
  description: string
}

export async function getSecrets(): Promise<SecretMeta[]> {
  return getJson<SecretMeta[]>("/api/secrets", [])
}

export async function saveSecret(name: string, value: string, description = ""): Promise<boolean> {
  return mutate("/api/secrets", "POST", { name, value, description })
}

export async function deleteSecret(name: string): Promise<boolean> {
  return mutate(`/api/secrets/${encodeURIComponent(name)}`, "DELETE")
}

// -- audit --------------------------------------------------------------------
export interface AuditEvent {
  id: string
  at: string
  action: string
  targetType: string
  targetId: string
  verdict: string | null
  dataHash: string | null
}

export async function getAudit(limit = 100): Promise<AuditEvent[]> {
  return getJson<AuditEvent[]>(`/api/audit?limit=${limit}`, [])
}

export interface Dataset {
  name: string
  rows: number
  columns: number
  /** "bound" (declared in elbi.yaml) or "synced" (from a warehouse source). */
  origin: "bound" | "synced"
}

export async function getDatasets(): Promise<Dataset[]> {
  return getJson<Dataset[]>("/api/datasets", [])
}

export interface DerivationSummary {
  name: string
  question: string
  verdict: string | null
  dataHash: string | null
  createdAt: string
  /** "repo" for a repo-authored derivation, "agent"/"human" for a chat-authored one. */
  origin?: string
}

export async function getDerivations(): Promise<DerivationSummary[]> {
  return getJson<DerivationSummary[]>("/api/derivations", [])
}

export interface Job {
  id: string
  label: string
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled"
  progress: string
  result: { certified?: boolean; verdict?: string | null } | null
  error: string | null
  createdAt: number
  startedAt: number | null
  finishedAt: number | null
}

export async function getJobs(): Promise<Job[]> {
  return getJson<Job[]>("/api/jobs", [])
}

export async function cancelJob(id: string): Promise<void> {
  try {
    const r = await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, {
      method: "POST",
    })
    if (!r.ok) console.error(`cancel job ${id} failed: ${r.status} ${r.statusText}`)
  } catch (err) {
    // best-effort; the panel reflects the state on its next poll, but log the failure
    console.error(`cancel job ${id} failed`, err)
  }
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
}

// One cursor-paginated page of conversations: pass `next_page_id` back as `pageId` to
// load the following page (null when the list is exhausted).
export interface ConversationPage {
  items: ConversationSummary[]
  next_page_id: string | null
}

export async function getConversations(
  limit = 20,
  pageId?: string,
  q?: string,
): Promise<ConversationPage> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (pageId) params.set("page_id", pageId)
  if (q) params.set("q", q)
  return getJson<ConversationPage>(`/api/conversations?${params.toString()}`, {
    items: [],
    next_page_id: null,
  })
}

// Rename a conversation; returns whether it succeeded so the caller can roll back its
// optimistic title change on failure.
export async function renameConversation(id: string, title: string): Promise<boolean> {
  try {
    const r = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    })
    if (!r.ok) {
      console.error(`rename conversation ${id} failed: ${r.status} ${r.statusText}`)
      return false
    }
    return true
  } catch (err) {
    console.error(`rename conversation ${id} failed`, err)
    return false
  }
}

export interface StoredMessage {
  id?: string
  role: string
  content: string
  result: Record<string, unknown> | null
  feedback?: "up" | "down" | null
}

// Rate an assistant turn (or clear it with null). Keyed on the server's message id.
export async function setMessageFeedback(
  messageId: string,
  feedback: "up" | "down" | null,
): Promise<boolean> {
  return mutate(`/api/messages/${encodeURIComponent(messageId)}/feedback`, "PUT", { feedback })
}

export async function getConversationMessages(id: string): Promise<StoredMessage[]> {
  return getJson<StoredMessage[]>(`/api/conversations/${encodeURIComponent(id)}/messages`, [])
}

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  cost: number
}

export async function getConversationUsage(id: string): Promise<Usage> {
  return getJson<Usage>(`/api/conversations/${encodeURIComponent(id)}/usage`, {
    prompt_tokens: 0,
    completion_tokens: 0,
    cost: 0,
  })
}

// Delete a conversation and its messages. Returns whether it succeeded, so the caller
// can restore its optimistically-removed row if the request failed (a 404 for an
// already-gone conversation, or a network error), rather than hiding the failure.
export async function deleteConversation(id: string): Promise<boolean> {
  try {
    const r = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    })
    if (!r.ok) {
      console.error(`delete conversation ${id} failed: ${r.status} ${r.statusText}`)
      return false
    }
    return true
  } catch (err) {
    console.error(`delete conversation ${id} failed`, err)
    return false
  }
}

export interface DerivationDetail extends DerivationSummary {
  conversationId: string | null
  source: string
  claim: Record<string, unknown> | null
  serve: Record<string, unknown> | null
  narrative: string
  rendered: string | null
  attestation: {
    checks?: { name: string; verdict: string; detail: string }[]
  } | null
  assumptions: string[]
}

export async function getDerivation(name: string): Promise<DerivationDetail | null> {
  return getJson<DerivationDetail | null>(`/api/derivations/${encodeURIComponent(name)}`, null)
}

// The download URLs for a derivation's signed, tamper-evident verification certificate:
// the canonical JSON (checked offline with `elbi certificate verify`) and a
// one-page PDF rendering of the same envelope.
export function certificateUrl(name: string): string {
  return `/api/certificates/${encodeURIComponent(name)}`
}

export function certificatePdfUrl(name: string): string {
  return `/api/certificates/${encodeURIComponent(name)}/pdf`
}

// Download URLs for the data-portability exports (IP-31). Each is one object's record:
// what it is, plus the evidence behind it, as a single JSON document.
export function derivationExportUrl(name: string): string {
  return `/api/exports/derivations/${encodeURIComponent(name)}`
}

// The slow one: a dashboard has no run history, so its export resolves every page to
// capture what it actually shows. Expect seconds, not milliseconds.
export function dashboardExportUrl(id: string): string {
  return `/api/exports/dashboards/${encodeURIComponent(id)}`
}

// The same resolution as the record, rendered as one self-contained, read-only page:
// what `elbi snapshot` writes, for sharing with someone who has no access to the app.
export function dashboardSnapshotUrl(id: string): string {
  return `/api/exports/dashboards/${encodeURIComponent(id)}/snapshot`
}

export function metricExportUrl(name: string): string {
  return `/api/exports/metrics/${encodeURIComponent(name)}`
}

// -- certified version history -------------------------------------------------

// One certified version of a derivation, keyed on its content-addressed `derivation_version`.
// Every scalar here, the `estimate` above all, is the verification oracle's certified value,
// not a self-reported number. `changed` names the input dimensions ("data", "code", "controls",
// "params", "claim") that moved the estimate from the next-older version; it is empty for the
// oldest version.
export interface CertifiedRun {
  name: string
  derivationVersion: string
  verdict: string
  createdAt: string
  dataHash: string | null
  estimate: number | null
  estimateLabel: string | null
  adjustedFor: string[]
  claim: Record<string, string>
  params: Record<string, unknown>
  deps: string[]
  checks: [string, string, string][]
  codeVersion: string | null
  inputVersions: Record<string, string>
  question: string
  conversationId: string | null
  shortVersion: string
  metrics: Record<string, number>
  changed: string[]
}

// A derivation's certified versions, its result history, newest first.
export async function getDerivationHistory(name: string): Promise<CertifiedRun[]> {
  return getJson<CertifiedRun[]>(`/api/derivations/${encodeURIComponent(name)}/history`, [])
}

// -- MLflow export -------------------------------------------------------------
// Where certified runs export to. `source` reports whether the env var, the stored
// setting, or nothing is in effect: the MLFLOW_TRACKING_URI env var overrides the setting.
export interface MlflowSettings {
  trackingUri: string
  source: string
}

export async function getMlflowSettings(): Promise<MlflowSettings> {
  return getJson<MlflowSettings>("/api/settings/mlflow", {
    trackingUri: "",
    source: "none",
  })
}

// What is running, and the command that upgrades it. The server reports only what is already on
// the machine: it never asks whether a newer release exists, because that would be an outbound
// request nobody asked for. `elbi update` is where that is asked.
export interface AppVersion {
  version: string
  package: string
  install: { kind: string; command: string; note: string }
}

export async function getAppVersion(): Promise<AppVersion> {
  return getJson<AppVersion>("/api/version", {
    version: "",
    package: "elbi",
    install: { kind: "", command: "", note: "" },
  })
}

export async function setMlflowTrackingUri(tracking_uri: string): Promise<boolean> {
  return mutate("/api/settings/mlflow", "PUT", { tracking_uri })
}

// -- MLflow model registry -------------------------------------------------
export interface RegisteredModel {
  name: string
  description: string
  latestVersion: number
  championVersion: number | null
  createdAtMs: number
  updatedAtMs: number
  tags: Record<string, string>
}

export interface ModelVersion {
  version: number
  runId: string
  // The MLflow experiment holding the source run; "" when the run is unknown.
  experimentId: string
  createdAtMs: number
  aliases: string[]
  description: string
  metrics: Record<string, number>
  params: Record<string, string>
  tags: Record<string, string>
  // The oracle's judgement of this version and why, which is what answers "so why
  // is this one not the champion?". Null on a version trained before it was recorded.
  verdict: string | null
  verdictDetail: string | null
}

export interface RegisteredModelDetail {
  name: string
  championVersion: number | null
  versions: ModelVersion[]
}

// The registry answers 503 with a `detail` naming the missing extra when model serving is
// not installed; that message becomes the page's empty state rather than a logged-away
// generic failure.
export type RegistryUnavailable = { unavailable: string }

async function getRegistryJson<T>(path: string, fallback: T): Promise<T | RegistryUnavailable> {
  try {
    const r = await fetch(path)
    if (r.status === 503) {
      const body = (await r.json().catch(() => null)) as { detail?: string } | null
      return { unavailable: body?.detail ?? "Model registry is unavailable." }
    }
    if (!r.ok) {
      console.error(`GET ${path} failed: ${r.status} ${r.statusText}`)
      return fallback
    }
    return (await r.json()) as T
  } catch (err) {
    console.error("GET failed", path, err)
    return fallback
  }
}

export async function getRegisteredModels(): Promise<RegisteredModel[] | RegistryUnavailable> {
  return getRegistryJson<RegisteredModel[]>("/api/registry/models", [])
}

export async function getRegisteredModel(
  name: string,
): Promise<RegisteredModelDetail | RegistryUnavailable | null> {
  return getRegistryJson<RegisteredModelDetail | null>(
    `/api/registry/models/${encodeURIComponent(name)}`,
    null,
  )
}

// Point an alias (default "champion") at a version, so serving picks it up.
export async function promoteModelVersion(
  name: string,
  version: number,
  alias?: string,
): Promise<boolean> {
  return mutate(
    `/api/registry/models/${encodeURIComponent(name)}/promote`,
    "POST",
    alias ? { version, alias } : { version },
  )
}

export async function deleteModel(name: string): Promise<boolean> {
  return mutate(`/api/registry/models/${encodeURIComponent(name)}`, "DELETE")
}

export async function deleteModelVersion(name: string, version: number): Promise<boolean> {
  return mutate(`/api/registry/models/${encodeURIComponent(name)}/versions/${version}`, "DELETE")
}

// -- training ----------------------------------------------------------------
export interface DatasetColumn {
  name: string
  numeric: boolean
}

export async function getDatasetColumns(name: string): Promise<DatasetColumn[]> {
  return getJson<DatasetColumn[]>(`/api/datasets/${encodeURIComponent(name)}/columns`, [])
}

export interface TrainResult {
  name: string
  version: number
  task: string
  target: string
  features: string[]
  bestEstimator: string
  metrics: Record<string, number>
  oracleVerdict: string | null
  champion: boolean
  rendered: string
}

// A training job, as returned by POST /api/registry/train and GET /api/jobs/{id}.
// `id` is null when training finished inline, in which case `state` is already terminal
// (and `progress` is a list rather than the job store's latest-line string).
export interface TrainJob {
  id: string | null
  label: string
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled"
  progress: string | string[]
  result: TrainResult | null
  error: string | null
}

// A training job as stored durably in /api/jobs: the source of truth the models pages
// poll, so tracking survives navigation and reloads.
export interface TrainingJob {
  id: string
  label: string
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled"
  progress: string
  result: TrainResult | null
  error: string | null
  createdAt: number
  finishedAt: number | null
}

// The durable training jobs, newest first. Returns null (not []) on a failed fetch so a
// poller can tell a transient error apart from "no jobs" and keep its last good state.
export async function getTrainingJobs(): Promise<TrainingJob[] | null> {
  try {
    const r = await fetch("/api/jobs")
    if (!r.ok) {
      console.error(`GET /api/jobs failed: ${r.status} ${r.statusText}`)
      return null
    }
    const all = (await r.json()) as TrainingJob[]
    return all.filter((j) => j.label.startsWith("train model "))
  } catch (err) {
    console.error("GET /api/jobs failed", err)
    return null
  }
}

// Everything a model can train on: bound datasets and certified derivations
// (feature pipelines), for the grouped data-source selects.
export interface FeatureSource {
  name: string
  kind: "dataset" | "derivation" | "training_set"
}

export async function getFeatureSources(): Promise<FeatureSource[]> {
  return getJson<FeatureSource[]>("/api/registry/feature-sources", [])
}

// The AutoML engines the training surface offers. `flaml` (default) and `autogluon`
// search; `optuna` is an explicit Bayesian tuner; `tabicl` is an open pretrained tabular
// foundation model (single forward pass, no ts_forecast); `ensemble` blends a diverse
// base set. Only `flaml` supports ts_forecast.
export type TrainingEngine = "flaml" | "autogluon" | "optuna" | "tabicl" | "ensemble"

export interface TrainRequest {
  name: string
  // The data source: exactly one of `dataset` (a bound dataset), `derivation`
  // (a certified feature derivation whose output rows are the training table), or
  // `training_set` (a materialized point-in-time join from the feature store).
  dataset?: string
  derivation?: string
  training_set?: string
  target: string
  features?: string[]
  task?: "auto" | "classification" | "regression" | "ts_forecast"
  time_budget?: number
  metric?: string
  // Stacked ensemble of the best models found: slower, often more accurate.
  ensemble?: boolean
  // The AutoML engine; the backend defaults to flaml. Only flaml supports ts_forecast.
  engine?: TrainingEngine
  // Required for ts_forecast: the timestamp column and how many periods ahead.
  time_col?: string
  horizon?: number
}

// Kick off training. A 400's `detail` (bad name/dataset/target) comes back as `error`
// so the form can show it inline instead of logging it away.
export async function trainModel(
  req: TrainRequest,
): Promise<{ job: TrainJob } | { error: string }> {
  try {
    const r = await fetch("/api/registry/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    })
    if (!r.ok) {
      const body = (await r.json().catch(() => null)) as { detail?: string } | null
      return { error: body?.detail ?? `${r.status} ${r.statusText}` }
    }
    return { job: (await r.json()) as TrainJob }
  } catch (err) {
    return { error: String(err) }
  }
}

export async function getTrainJob(id: string): Promise<TrainJob | null> {
  return getJson<TrainJob | null>(`/api/jobs/${encodeURIComponent(id)}`, null)
}

// -- serving -------------------------------------------------------------------
export interface ModelSchemaField {
  name: string
  type: string
  required: boolean
}

export interface ModelSchema {
  version: number
  inputs: ModelSchemaField[]
  outputs: ModelSchemaField[]
  inputExample: Array<Record<string, unknown>> | null
}

export async function getModelSchema(name: string, version: number): Promise<ModelSchema | null> {
  return getJson<ModelSchema | null>(
    `/api/registry/models/${encodeURIComponent(name)}/versions/${version}/schema`,
    null,
  )
}

// -- batch scoring ---------------------------------------------------------------

// The result of a finished batch-scoring job: summary stats over the predictions, a
// small sample of scored rows, and the run holding the full predictions.csv artifact.
export interface BatchScoreResult {
  name: string
  version: string
  dataset: string
  nRows: number
  runId: string
  artifact: string
  stats: Record<string, number>
  sample: Array<Record<string, unknown>>
  rendered: string
}

// A batch-scoring job, following the training-job contract: `id` is null when scoring
// finished inline (the state is already terminal); otherwise poll /api/jobs/{id}.
export interface BatchScoreJob {
  id: string | null
  label: string
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled"
  progress: string | string[]
  result: BatchScoreResult | null
  error: string | null
}

// The downloadable full-predictions artifact of a batch-scoring run.
export function batchScoreArtifactUrl(runId: string): string {
  return `/mlflow/get-artifact?path=predictions.csv&run_uuid=${encodeURIComponent(runId)}`
}

// Kick off batch scoring of a dataset against a model. A 4xx `detail` (bad dataset,
// missing version) comes back as `error` for inline display.
export async function batchScoreModel(
  name: string,
  // Exactly one of `dataset` or `derivation` names the rows to score.
  req: { dataset?: string; derivation?: string; version?: string; limit?: number },
): Promise<{ job: BatchScoreJob } | { error: string }> {
  try {
    const r = await fetch(`/api/registry/models/${encodeURIComponent(name)}/batch-score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    })
    if (!r.ok) {
      const body = (await r.json().catch(() => null)) as { detail?: string } | null
      return { error: body?.detail ?? `${r.status} ${r.statusText}` }
    }
    return { job: (await r.json()) as BatchScoreJob }
  } catch (err) {
    return { error: String(err) }
  }
}

export async function getBatchScoreJob(id: string): Promise<BatchScoreJob | null> {
  return getJson<BatchScoreJob | null>(`/api/jobs/${encodeURIComponent(id)}`, null)
}

// -- inference log ---------------------------------------------------------------

// One serving request against a model, newest first in the log.
export interface InferenceLogEntry {
  id: string
  at: string
  version: string
  nRows: number
  latencyMs: number
  status: "ok" | "error"
  error: string | null
  predictions: unknown
}

export async function getInferenceLog(
  name: string,
  limit: number,
  offset: number,
): Promise<InferenceLogEntry[]> {
  return getJson<InferenceLogEntry[]>(
    `/api/registry/models/${encodeURIComponent(name)}/inference?limit=${limit}&offset=${offset}`,
    [],
  )
}

// -- drift ----------------------------------------------------------------------

export interface DriftColumn {
  column: string
  method: string
  score: number
  threshold: number
  drifted: boolean
}

// One drift check: recent serving traffic compared against the training data,
// column by column. The full HTML report lives on the run as drift_report.html.
export interface DriftResult {
  name: string
  version: string
  nCurrentRows: number
  runId: string
  nColumns: number
  nDrifted: number
  shareDrifted: number
  datasetDrift: boolean
  columns: DriftColumn[]
}

// A past drift check, as listed in the history. `at` is epoch milliseconds.
export interface DriftHistoryEntry {
  runId: string
  at: number
  version: string
  nDrifted: number
  shareDrifted: number
  datasetDrift: boolean
  nCurrentRows: number
}

// The full HTML drift report logged by a drift-check run.
export function driftReportUrl(runId: string): string {
  return `/mlflow/get-artifact?path=drift_report.html&run_uuid=${encodeURIComponent(runId)}`
}

// Run a drift check now. A 400 `detail` (e.g. not enough serving traffic yet) comes
// back as `error` so the page can show it as an inline notice.
export async function runDriftCheck(name: string): Promise<DriftResult | { error: string }> {
  try {
    const r = await fetch(`/api/registry/models/${encodeURIComponent(name)}/drift-check`, {
      method: "POST",
    })
    if (!r.ok) {
      const body = (await r.json().catch(() => null)) as { detail?: string } | null
      return { error: body?.detail ?? `${r.status} ${r.statusText}` }
    }
    return (await r.json()) as DriftResult
  } catch (err) {
    return { error: String(err) }
  }
}

export async function getDriftHistory(name: string, limit = 20): Promise<DriftHistoryEntry[]> {
  return getJson<DriftHistoryEntry[]>(
    `/api/registry/models/${encodeURIComponent(name)}/drift?limit=${limit}`,
    [],
  )
}

// -- retraining policy ------------------------------------------------------------

// A model's automatic-retraining policy; `configured: false` when none is set.
export type RetrainPolicy =
  | { configured: false }
  | {
      configured: true
      // The source name; `sourceKind` says whether it is a bound dataset or a
      // certified feature derivation.
      dataset: string
      sourceKind: "dataset" | "derivation"
      target: string
      features: string[] | null
      task: string
      timeBudget: number
      metric: string | null
      ensemble: boolean
      engine?: TrainingEngine
      mode: "on_data_change" | "interval"
      intervalHours: number
      enabled: boolean
      lastRunAt: string | null
    }

export interface RetrainPolicyInput {
  // Exactly one of `dataset` or `derivation` names the training source.
  dataset?: string
  derivation?: string
  target: string
  features?: string[]
  task?: string
  time_budget?: number
  metric?: string
  ensemble?: boolean
  engine?: TrainingEngine
  mode?: "on_data_change" | "interval"
  interval_hours?: number
  enabled?: boolean
  time_col?: string
  horizon?: number
}

// Returns null on a failed fetch, so the panel can tell "unavailable" from
// "not configured".
export async function getRetrainPolicy(name: string): Promise<RetrainPolicy | null> {
  return getJson<RetrainPolicy | null>(
    `/api/registry/models/${encodeURIComponent(name)}/retrain`,
    null,
  )
}

export async function putRetrainPolicy(name: string, input: RetrainPolicyInput): Promise<boolean> {
  return mutate(`/api/registry/models/${encodeURIComponent(name)}/retrain`, "PUT", input)
}

export async function deleteRetrainPolicy(name: string): Promise<boolean> {
  return mutate(`/api/registry/models/${encodeURIComponent(name)}/retrain`, "DELETE")
}

// -- model webhooks ----------------------------------------------------------------

// Where model-registry events are POSTed. As with LLM keys, the secret itself is never
// returned: `secret_set` reports whether one is stored.
export interface WebhookSettings {
  url: string
  secretSet: boolean
}

export async function getWebhookSettings(): Promise<WebhookSettings> {
  return getJson<WebhookSettings>("/api/settings/webhooks", {
    url: "",
    secretSet: false,
  })
}

// Save the webhook URL; omit `secret` to keep the stored one.
export async function setWebhookSettings(url: string, secret?: string): Promise<boolean> {
  return mutate("/api/settings/webhooks", "PUT", secret === undefined ? { url } : { url, secret })
}

// Score records against a served version. Non-2xx `detail` comes back as `error`.
export async function invokeModel(
  name: string,
  version: number,
  body: unknown,
): Promise<{ predictions: unknown[] } | { error: string }> {
  try {
    const r = await fetch(
      `/api/serving/${encodeURIComponent(name)}/invocations?version=${version}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    )
    if (!r.ok) {
      const b = (await r.json().catch(() => null)) as { detail?: string } | null
      return { error: b?.detail ?? `${r.status} ${r.statusText}` }
    }
    return (await r.json()) as { predictions: unknown[] }
  } catch (err) {
    return { error: String(err) }
  }
}

// Raw scoring against the served version: the model's recorded feature derivation
// engineers the rows, then they score. `engineered` is a sample of the rows the
// model actually saw.
export interface RawScoreResult {
  predictions: unknown[]
  featuresApplied: string
  nRawRows: number
  nScoredRows: number
  engineered: Array<Record<string, unknown>>
}

// Score RAW source rows through the model's feature derivation: the skew-proof path for
// derivation-trained models. A 400 `detail` (e.g. the model was not trained on a feature
// derivation) comes back as `error`.
export async function rawInvokeModel(
  name: string,
  body: unknown,
): Promise<RawScoreResult | { error: string }> {
  try {
    const r = await fetch(`/api/serving/${encodeURIComponent(name)}/raw-invocations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    if (!r.ok) {
      const b = (await r.json().catch(() => null)) as { detail?: string } | null
      return { error: b?.detail ?? `${r.status} ${r.statusText}` }
    }
    return (await r.json()) as RawScoreResult
  } catch (err) {
    return { error: String(err) }
  }
}
