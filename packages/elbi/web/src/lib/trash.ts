// API client and types for the trash surface: soft-deleted artifacts a caller may
// restore or erase permanently. Mirrors the throwing helper style in dashboards.ts.

export type TrashKind =
  | "notebook"
  | "folder"
  | "dashboard"
  | "saved_query"
  | "metric"
  | "feature_view"
  | "derivation"

export interface TrashItem {
  type: TrashKind
  id: string
  name: string
  deletedAt: string
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    const detail = await r.text()
    throw new Error(`${init?.method ?? "GET"} ${path} failed: ${r.status} ${detail}`)
  }
  return (await r.json()) as T
}

const id = encodeURIComponent

export const listTrash = () => json<TrashItem[]>("/api/trash")

export const restoreTrash = (kind: TrashKind, itemId: string) =>
  json<{ ok: boolean }>(`/api/trash/${id(kind)}/${id(itemId)}/restore`, { method: "POST" })

export const deleteTrashForever = (kind: TrashKind, itemId: string) =>
  json<{ ok: boolean }>(`/api/trash/${id(kind)}/${id(itemId)}`, { method: "DELETE" })
