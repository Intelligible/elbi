// What a warehouse source's last sync actually produced, derived from the fields the API
// already returns. The stored `status` is the job state (`idle` only means nothing is
// running), so it cannot be shown as health on its own: a source whose every sync lands
// zero rows, or whose tables failed, would read as fine.

import type { SchemaView, SourceDetail, SourceSummary, SyncOutcome } from "@/lib/warehouse"

export type SyncHealth = "syncing" | "failed" | "empty" | "synced" | "never" | "off"

export type HealthVariant = "info" | "danger" | "warning" | "success" | "neutral"

export interface SyncStatus {
  health: SyncHealth
  label: string
  variant: HealthVariant
  // One or two plain sentences on what this state means for this source or table.
  explanation: string
}

const VARIANT: Record<SyncHealth, HealthVariant> = {
  syncing: "info",
  failed: "danger",
  empty: "warning",
  synced: "success",
  never: "neutral",
  off: "neutral",
}

const status = (health: SyncHealth, label: string, explanation: string): SyncStatus => ({
  health,
  label,
  variant: VARIANT[health],
  explanation,
})

const plural = (n: number, one: string, many = `${one}s`) =>
  `${n.toLocaleString()} ${n === 1 ? one : many}`

const PARTIAL_WRITE =
  "A failed sync can leave the table partly rewritten, so its row count may not match what is in it."

const NO_ROWS =
  "A full refresh that gets no rows leaves any earlier rows in place. Check that the source has data and that the connector can read its response."

/** One table's state, from its own last run. */
export function tableSyncStatus(schema: SchemaView): SyncStatus {
  if (!schema.shouldSync)
    return status(
      "off",
      "Off",
      "This table is turned off, so a sync skips it. Its warehouse table keeps what the last sync left.",
    )
  if (schema.status === "syncing") return status("syncing", "Syncing", "This table is syncing now.")
  if (schema.status === "error")
    return status(
      "failed",
      "Failed",
      `The last sync of this table failed${schema.lastError ? `: ${schema.lastError}` : "."} ${PARTIAL_WRITE}`,
    )
  if (schema.status === "synced") {
    if (schema.rowCount === 0)
      return status(
        "empty",
        "No rows",
        `The last sync of this table succeeded but got no rows. ${NO_ROWS}`,
      )
    return status("synced", "Synced", "The last sync of this table succeeded.")
  }
  return status("never", "Never synced", "This table has not been synced yet.")
}

// What a source's headline is derived from: the job state plus its enabled tables'
// last-sync results, whether counted from the schemas (detail) or by the API (list).
interface SourceFacts {
  status: string
  lastError: string | null
  enabled: number
  synced: number
  failed: number
  tableError: string | null
  emptyTables: string[]
  rows: number
}

function deriveSourceStatus(f: SourceFacts): SyncStatus {
  if (f.status === "syncing")
    return status("syncing", "Syncing…", "A sync of this source is running now.")

  if (f.status === "error" || f.failed > 0) {
    const error = f.lastError ?? f.tableError
    const which =
      f.failed > 0
        ? `The last sync failed for ${f.failed} of ${plural(f.enabled, "enabled table")}`
        : "The last sync failed"
    return status("failed", "Failed", `${which}${error ? `: ${error}` : "."}`)
  }

  if (f.enabled === 0)
    return status(
      "off",
      "No tables enabled",
      "Every table is turned off, so a sync has nothing to read. Turn one on with its Sync switch.",
    )

  if (f.synced === 0)
    return status(
      "never",
      "Never synced",
      "None of the enabled tables has been synced yet. Click Sync now, or wait for the schedule.",
    )

  if (f.emptyTables.length > 0)
    return status(
      "empty",
      "Synced, but no rows",
      `The last sync succeeded, but ${plural(f.emptyTables.length, "table")} got no rows: ${f.emptyTables.join(", ")}. ${NO_ROWS}`,
    )

  return status(
    "synced",
    `Synced · ${plural(f.rows, "row")}`,
    `The last sync of every enabled table succeeded. They hold ${plural(f.rows, "row")} as of their last syncs.`,
  )
}

/** The headline for a whole source: the worst state among its enabled tables. */
export function sourceSyncStatus(source: SourceDetail): SyncStatus {
  const enabled = source.schemas.filter((s) => s.shouldSync)
  const failed = enabled.filter((s) => s.status === "error")
  const synced = enabled.filter((s) => s.status === "synced")
  return deriveSourceStatus({
    status: source.status,
    lastError: source.lastError,
    enabled: enabled.length,
    synced: synced.length,
    failed: failed.length,
    tableError: failed.find((s) => s.lastError)?.lastError ?? null,
    emptyTables: synced.filter((s) => s.rowCount === 0).map((s) => s.table),
    rows: synced.reduce((n, s) => n + (s.rowCount ?? 0), 0),
  })
}

/**
 * The same headline for a row of the sources list, from the API's per-source summary.
 * A healthy row reads just "Synced": the list has its own rows column beside it.
 */
export function summarySyncStatus(source: SourceSummary): SyncStatus {
  const st = deriveSourceStatus({
    status: source.status,
    lastError: source.lastError,
    enabled: source.enabledCount,
    synced: source.enabledSyncedCount,
    failed: source.failedCount,
    tableError: source.tableError,
    emptyTables: source.emptyTables,
    rows: source.enabledRows,
  })
  return st.health === "synced" ? { ...st, label: "Synced" } : st
}

/** One line of a just-finished sync's result. */
export function outcomeStatus(outcome: SyncOutcome): SyncStatus {
  if (!outcome.ok) return status("failed", "Failed", outcome.error ?? "This table failed to sync.")
  if (outcome.rows === 0)
    return status("empty", "No rows", `This table synced but got no rows. ${NO_ROWS}`)
  return status("synced", "Synced", `This table synced ${plural(outcome.rows, "row")}.`)
}
