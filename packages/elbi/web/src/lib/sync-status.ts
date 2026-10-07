// What a warehouse source's last sync actually produced, derived from the fields the API
// already returns. The stored `status` is the job state (`idle` only means nothing is
// running), so it cannot be shown as health on its own: a source whose every sync lands
// zero rows, or whose tables failed, would read as fine.

import type { SchemaView, SourceDetail, SyncOutcome } from "@/lib/warehouse"

export type SyncHealth = "syncing" | "failed" | "partial" | "empty" | "synced" | "never" | "off"

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
  partial: "neutral",
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

/** The headline for a whole source: the worst state among its enabled tables. */
export function sourceSyncStatus(source: SourceDetail): SyncStatus {
  if (source.status === "syncing")
    return status("syncing", "Syncing…", "A sync of this source is running now.")

  const enabled = source.schemas.filter((s) => s.shouldSync)
  const failed = enabled.filter((s) => s.status === "error")
  if (source.status === "error" || failed.length > 0) {
    const error = source.lastError ?? failed[0]?.lastError ?? null
    const which =
      failed.length > 0
        ? `The last sync failed for ${failed.length} of ${plural(enabled.length, "enabled table")}`
        : "The last sync failed"
    return status("failed", "Failed", `${which}${error ? `: ${error}` : "."}`)
  }

  if (enabled.length === 0)
    return status(
      "off",
      "No tables enabled",
      "Every table is turned off, so a sync has nothing to read. Turn one on with its Sync switch.",
    )

  const synced = enabled.filter((s) => s.status === "synced")
  if (synced.length === 0)
    return status(
      "never",
      "Never synced",
      "None of the enabled tables has been synced yet. Click Sync now, or wait for the schedule.",
    )

  // A table just switched on is enabled but pending, so "every enabled table" below would lie.
  const pending = enabled.length - synced.length
  if (pending > 0)
    return status(
      "partial",
      `Synced · ${synced.length} of ${plural(enabled.length, "table")}`,
      `${plural(pending, "enabled table has", "enabled tables have")} not been synced yet. Click Sync now, or wait for the schedule.`,
    )

  const empty = synced.filter((s) => s.rowCount === 0)
  if (empty.length > 0)
    return status(
      "empty",
      "Synced, but no rows",
      `The last sync succeeded, but ${plural(empty.length, "table")} got no rows: ${empty
        .map((s) => s.table)
        .join(", ")}. ${NO_ROWS}`,
    )

  const rows = synced.reduce((n, s) => n + (s.rowCount ?? 0), 0)
  return status(
    "synced",
    `Synced · ${plural(rows, "row")}`,
    `The last sync of every enabled table succeeded. They hold ${plural(rows, "row")} as of their last syncs.`,
  )
}

/** One line of a just-finished sync's result. */
export function outcomeStatus(outcome: SyncOutcome, syncType?: SchemaView["syncType"]): SyncStatus {
  if (!outcome.ok) return status("failed", "Failed", outcome.error ?? "This table failed to sync.")
  // `rows` is what this run wrote, so on an incremental table 0 means nothing new, not empty.
  if (outcome.rows === 0 && syncType === "incremental")
    return status(
      "synced",
      "No new rows",
      "This table synced and found no rows newer than the last sync, so it is unchanged.",
    )
  if (outcome.rows === 0)
    return status("empty", "No rows", `This table synced but got no rows. ${NO_ROWS}`)
  return status("synced", "Synced", `This table synced ${plural(outcome.rows, "row")}.`)
}
