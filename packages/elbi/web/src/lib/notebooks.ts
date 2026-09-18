// The notebooks API client: typed fetch helpers plus a streaming run.
//
// A run is a POST that returns Server-Sent Events (outputs as they stream from the
// kernel), so it is read with fetch + a stream reader rather than EventSource (which is
// GET-only). Everything else is a plain JSON call, matching the app's other clients.

export type CellType = "code" | "markdown" | "raw"

// An nbformat output object (stream / execute_result / display_data / error). Kept loose
// because the `data` MIME bundle is heterogeneous; the renderer narrows by output_type.
export type Output = {
  output_type: "stream" | "execute_result" | "display_data" | "error"
  name?: string
  text?: string
  data?: Record<string, unknown>
  metadata?: Record<string, unknown>
  execution_count?: number | null
  ename?: string
  evalue?: string
  traceback?: string[]
}

// What a cell pushed to the warehouse. Recorded per run, so the claim that work happens
// where the data is can be checked rather than trusted.
export type CellQuery = {
  sql: string
  duration_ms: number
  rows: number
  truncated: boolean
  error?: string | null
}

export type Cell = {
  id: string
  cell_type: CellType
  source: string
  metadata: Record<string, unknown>
  outputs: Output[]
  execution_count: number | null
}

/** Cells the editor does not draw: environment a derivation needs bound, not the thing
 * being edited. They still run, and `Run all` still runs them. */
export function isSetupCell(cell: Cell): boolean {
  const elbi = cell.metadata?.elbi
  return typeof elbi === "object" && elbi !== null && (elbi as { role?: unknown }).role === "setup"
}

export type CellDeps = {
  defs: string[]
  refs: string[]
  upstream: string[]
  downstream: string[]
  syntax_error: string | null
}

export type NotebookGraph = {
  cells: Record<string, CellDeps>
  conflicts: Record<string, string[]>
  cycle: string[]
}

export type NotebookEnvironment = {
  base_env: string | null
  base_environments: string[]
  lock: string[] | null
  locked: boolean
}

export type NotebookView = {
  id: string
  name: string
  deps: string[]
  metadata: Record<string, unknown>
  schedule: Record<string, unknown> | null
  kernel_status: "idle" | "running"
  cells: Cell[]
  graph: NotebookGraph
  environment: NotebookEnvironment
}

export type EnvironmentResult = {
  ok: boolean
  deps?: string[]
  lock?: string[] | null
  error?: string | null
}

export type BaseEnvironment = { name: string; deps: string[] }

export type NotebookSummary = {
  id: string
  name: string
  folder_id: string | null
  // The notebook this was copied from, or null. A record of origin, not a
  // reference: the source may since have been deleted.
  copied_from: string | null
  created_at: string
  updated_at: string
  cell_count: number
}

// A folder in the workspace tree. `parent_id` is null at the caller's root; the client
// rebuilds the tree and breadcrumb from the flat list this shape comes in.
export type NotebookFolder = {
  id: string
  name: string
  parent_id: string | null
  created_at: string
  updated_at: string
}

// A streamed run event. The union mirrors the backend's event shapes.
export type RunEvent =
  | { event: "cell_start"; cell: string; execution_count: number }
  | { event: "output"; cell: string; output: Output }
  | { event: "status"; cell: string; status: string; execution_count: number }
  | { event: "stale"; cells: string[] }
  | { event: "input_request"; cell: string; prompt: string; password: boolean }
  | { event: "done" }
  | { event: "error"; message: string }

// A Jupyter-shaped completion reply: the matches and the token span they replace.
export type CompleteReply = {
  matches: string[]
  cursor_start: number
  cursor_end: number
}

// A Jupyter-shaped inspect reply: whether a name was found and its help MIME bundle.
export type InspectReply = { found: boolean; data: Record<string, unknown> }

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

function put<T>(path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
}

export const listNotebooks = () => json<NotebookSummary[]>("/api/notebooks")

export const createNotebook = (name: string, folderId?: string | null) =>
  post<{ id: string }>("/api/notebooks", { name, folder_id: folderId ?? null })

export const duplicateNotebook = (notebookId: string) =>
  post<{ id: string }>(`/api/notebooks/${encodeURIComponent(notebookId)}/duplicate`)

export const listFolders = () => json<NotebookFolder[]>("/api/notebooks/folders")

export const createFolder = (name: string, parentId?: string | null) =>
  post<{ id: string }>("/api/notebooks/folders", {
    name,
    parent_id: parentId ?? null,
  })

export const renameFolder = (id: string, name: string) =>
  put<{ ok: boolean }>(`/api/notebooks/folders/${id}`, { name })

export const moveFolder = (id: string, parentId: string | null) =>
  put<{ ok: boolean }>(`/api/notebooks/folders/${id}`, { parent_id: parentId })

export const deleteFolder = (id: string, recursive = false) =>
  json(`/api/notebooks/folders/${id}?recursive=${recursive}`, {
    method: "DELETE",
  })

export const moveNotebook = (id: string, folderId: string | null) =>
  post<{ ok: boolean }>(`/api/notebooks/${id}/move`, { folder_id: folderId })

export const getNotebook = (id: string) => json<NotebookView>(`/api/notebooks/${id}`)

// A live-kernel variable: its name, Python type, and a one-line summary (a DataFrame's
// shape, a container's length, a scalar's value).
export interface NotebookVariable {
  name: string
  type: string
  summary: string
}

export const getNotebookVariables = (id: string) =>
  json<NotebookVariable[]>(`/api/notebooks/${id}/variables`)

export const deleteNotebook = (id: string) => json(`/api/notebooks/${id}`, { method: "DELETE" })

export type Schedule = {
  enabled: boolean
  mode: "interval" | "on_data_change"
  interval_hours?: number
  dataset?: string
  params?: Record<string, unknown>
}

export const updateNotebook = (
  id: string,
  fields: {
    name?: string
    deps?: string[]
    base_env?: string | null
    metadata?: Record<string, unknown>
    schedule?: Schedule | null
    compute_profile?: string | null
  },
) => put<EnvironmentResult & { ok: boolean }>(`/api/notebooks/${id}`, fields)

// Compute. The menu is infrastructure -- there is no writer here on purpose, because a
// control the UI offered but could not save would be worse than its absence.
export type ComputeProfile = {
  name: string
  cpu: string
  memory: string
  gpu: number
  gpuType: string | null
  image: string | null
  idleTimeout: number
  maxRuntime: number
  egress: string | string[]
  allowedGroups: string[]
  spot: boolean
  warmPoolSize: number
  runtimeClass: string | null
  pids: number
  version: string
  costPerHour: number
}

export type ComputeMenu = {
  default: string
  maxCostPerHour: number | null
  profiles: ComputeProfile[]
}

// `drift` is set when the profile's definition changed after this kernel started, which
// is the thing an admin who tightened a limit needs told rather than assumed.
export type NotebookCompute = {
  profile: ComputeProfile
  profileVersion: string
  status: "idle" | "running"
  drift: string | null
  // Set when the notebook asked for a profile that no longer exists and got the default
  // instead. Reported rather than absorbed: running something other than what was asked
  // for is how a resized limit goes unnoticed.
  unavailable: string | null
  startedWith?: ComputeProfile
  uptimeSeconds?: number
}

export const getComputeProfiles = () => json<ComputeMenu>("/api/compute/profiles")

export type ComputeSession = {
  endedAt: string
  kind: string
  notebookId: string | null
  profile: string
  profileVersion: string
  seconds: number
  cost: number
}

export type ComputeUsageReport = {
  since: string
  spend: number
  // Interactive versus batch. A scheduled rerun and an abandoned session cost the same
  // per second and mean entirely different things.
  byKind: Record<string, number>
  limit: number | null
  // Set when the window's spend passed the limit. An alert and not a cap: sessions keep
  // running, because ending one to recover the overage destroys work to save cents.
  alert: string | null
  sessions: ComputeSession[]
}

export const getComputeUsage = (days = 30) =>
  json<ComputeUsageReport>(`/api/compute/usage?days=${days}`)

export const getNotebookCompute = (id: string) =>
  json<NotebookCompute>(`/api/compute/notebooks/${id}`)

export const relockNotebook = (id: string) => post<EnvironmentResult>(`/api/notebooks/${id}/relock`)

export const getBaseEnvironments = () =>
  json<BaseEnvironment[]>("/api/settings/notebook-environments")

export const setBaseEnvironments = (environments: BaseEnvironment[]) =>
  put<{ ok: boolean }>("/api/settings/notebook-environments", { environments })

export const addCell = (
  id: string,
  opts: { cell_type?: CellType; after?: string; source?: string },
) => post<{ id: string }>(`/api/notebooks/${id}/cells`, opts)

export const updateCell = (
  notebookId: string,
  cellId: string,
  fields: {
    source?: string
    cell_type?: CellType
    metadata?: Record<string, unknown>
  },
) => put<{ ok: boolean }>(`/api/notebooks/${notebookId}/cells/${cellId}`, fields)

export const deleteCell = (notebookId: string, cellId: string) =>
  json(`/api/notebooks/${notebookId}/cells/${cellId}`, { method: "DELETE" })

export const reorderCells = (id: string, order: string[]) =>
  post<{ ok: boolean }>(`/api/notebooks/${id}/cells/reorder`, { order })

export const interruptNotebook = (id: string) =>
  post<{ ok: boolean }>(`/api/notebooks/${id}/interrupt`)

export const completeCell = (id: string, code: string, cursorPos: number) =>
  post<CompleteReply>(`/api/notebooks/${id}/complete`, {
    code,
    cursor_pos: cursorPos,
  })

export const inspectCell = (id: string, code: string, cursorPos: number, detailLevel = 0) =>
  post<InspectReply>(`/api/notebooks/${id}/inspect`, {
    code,
    cursor_pos: cursorPos,
    detail_level: detailLevel,
  })

export const sendInput = (id: string, value: string) =>
  post<{ ok: boolean }>(`/api/notebooks/${id}/input`, { value })

export const restartNotebook = (id: string) => post<{ ok: boolean }>(`/api/notebooks/${id}/restart`)

export const importNotebook = (name: string, ipynb: unknown) =>
  post<{ id: string }>("/api/notebooks/import", { name, ipynb })

export type PromoteResult = {
  ok: boolean
  name?: string
  certified?: boolean
  verdict?: string | null
  rendered?: string | null
  error?: string | null
  detail?: string | null
}

export const promoteCell = (notebookId: string, cellId: string) =>
  post<PromoteResult>(`/api/notebooks/${notebookId}/cells/${cellId}/promote`)

export const notebookFromDerivation = (name: string) =>
  post<{ id: string }>(`/api/notebooks/from-derivation/${encodeURIComponent(name)}`)

export const notebookFromModel = (name: string, version?: string) =>
  post<{ id: string }>(
    `/api/notebooks/from-model/${encodeURIComponent(name)}` +
      (version ? `?version=${encodeURIComponent(version)}` : ""),
  )

export function exportNotebookUrl(id: string): string {
  return `/api/notebooks/${id}/export`
}

// Run cells (or everything) and invoke `onEvent` for each streamed event. Returns a
// promise that resolves when the stream ends and an `abort` to cancel the run's stream.
export function runNotebook(
  id: string,
  body: { cells?: string[]; run_all?: boolean; fresh?: boolean },
  onEvent: (event: RunEvent) => void,
): { done: Promise<void>; abort: () => void } {
  const controller = new AbortController()
  const done = (async () => {
    const resp = await fetch(`/api/notebooks/${id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!resp.body) return
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""
    while (true) {
      const { value, done: finished } = await reader.read()
      if (finished) break
      buffer += decoder.decode(value, { stream: true })
      // SSE frames are separated by a blank line and each `data:` line by a newline. The server
      // writes CRLF (`\r\n`), so split on either ending: matching only `\n\n` never separates
      // the CRLF frames and no event is ever parsed.
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() ?? ""
      for (const frame of frames) {
        for (const line of frame.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue
          const payload = line.slice(5).trim()
          if (payload === "[DONE]") return
          let event: RunEvent
          try {
            event = JSON.parse(payload) as RunEvent
          } catch {
            continue // skip a malformed frame rather than aborting the run
          }
          onEvent(event)
        }
      }
    }
  })()
  return { done, abort: () => controller.abort() }
}
