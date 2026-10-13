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

// "<prefix>: <error>." with exactly one closing period; the service stores raw exception text.
const withError = (prefix: string, error: string | null | undefined) => {
  const trimmed = error?.replace(/[.\s]+$/, "") ?? ""
  return trimmed ? `${prefix}: ${trimmed}.` : `${prefix}.`
}

const CHECK_SOURCE = "Check that the source has data and that the connector can read its response."

const NO_ROWS = `A full refresh that gets no rows leaves any earlier rows in place. ${CHECK_SOURCE}`

// Whether a sync appends: the sync service only does so with a cursor column set; an
// "incremental" table without one is overwritten like a full refresh (warehouse/sync.py).
const appends = (s?: Pick<SchemaView, "syncType" | "incrementalField">) =>
  s?.syncType === "incremental" && s.incrementalField !== null && s.incrementalField !== undefined

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
      withError("The last sync of this table failed", schema.lastError),
    )
  if (schema.status === "synced") {
    if (schema.rowCount === 0)
      return status(
        "empty",
        "No rows",
        `The last sync of this table succeeded but got no rows. ${appends(schema) ? CHECK_SOURCE : NO_ROWS}`,
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
    return status("failed", "Failed", withError(which, error))
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

  // Checked before pending tables: a warning outranks "not synced yet".
  const empty = synced.filter((s) => s.rowCount === 0)
  if (empty.length > 0)
    return status(
      "empty",
      "Synced, but no rows",
      `The last sync succeeded, but ${plural(empty.length, "table")} got no rows: ${empty
        .map((s) => s.table)
        .join(", ")}. ${empty.every(appends) ? CHECK_SOURCE : NO_ROWS}`,
    )

  // A table just switched on is enabled but pending, so "every enabled table" below would lie.
  const pending = enabled.length - synced.length
  if (pending > 0)
    return status(
      "partial",
      `Synced · ${synced.length} of ${plural(enabled.length, "table")}`,
      `${plural(pending, "enabled table has", "enabled tables have")} not been synced yet. Click Sync now, or wait for the schedule.`,
    )

  const rows = synced.reduce((n, s) => n + (s.rowCount ?? 0), 0)
  return status(
    "synced",
    `Synced · ${plural(rows, "row")}`,
    `The last sync of every enabled table succeeded. They hold ${plural(rows, "row")} as of their last syncs.`,
  )
}

/** One line of a just-finished sync's result. */
export function outcomeStatus(
  outcome: SyncOutcome,
  schema?: Pick<SchemaView, "syncType" | "incrementalField">,
): SyncStatus {
  if (!outcome.ok) return status("failed", "Failed", outcome.error ?? "This table failed to sync.")
  // `rows` is what this run wrote, so on an appending table 0 means nothing new, not empty.
  if (outcome.rows === 0 && appends(schema))
    return status(
      "synced",
      "No new rows",
      "This table synced and found no rows newer than the last sync, so it is unchanged.",
    )
  if (outcome.rows === 0)
    return status("empty", "No rows", `This table synced but got no rows. ${NO_ROWS}`)
  return status("synced", "Synced", `This table synced ${plural(outcome.rows, "row")}.`)
}
