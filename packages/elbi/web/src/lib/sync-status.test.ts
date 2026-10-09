import { describe, expect, it } from "vitest"

import { outcomeStatus, sourceSyncStatus, summarySyncStatus, tableSyncStatus } from "./sync-status"
import type { SchemaView, SourceDetail } from "./warehouse"

const table = (s: Partial<SchemaView> = {}): SchemaView => ({
  id: "t",
  name: "invoices",
  table: "stripe__invoices",
  shouldSync: true,
  syncType: "full_refresh",
  incrementalField: null,
  incrementalFields: [],
  status: "synced",
  rowCount: 233,
  lastError: null,
  lastSyncedAt: "2026-10-01T09:00:00",
  ...s,
})

const source = (schemas: SchemaView[], s: Partial<SourceDetail> = {}): SourceDetail => ({
  id: "src",
  name: "stripe",
  sourceType: "stripe",
  description: "",
  prefix: "stripe",
  syncFrequency: "day",
  status: "idle",
  lastError: null,
  lastSyncedAt: "2026-10-01T09:00:00",
  createdAt: null,
  schemaCount: schemas.length,
  syncedCount: schemas.filter((x) => x.status === "synced").length,
  rows: 0,
  enabledCount: 0,
  enabledSyncedCount: 0,
  enabledRows: 0,
  failedCount: 0,
  tableError: null,
  emptyTables: [],
  emptyTablesAppend: false,
  schemas,
  ...s,
})

describe("sourceSyncStatus", () => {
  it("is syncing while a run is in progress, whatever the tables last did", () => {
    const st = sourceSyncStatus(source([table({ status: "error" })], { status: "syncing" }))
    expect(st.health).toBe("syncing")
    expect(st.variant).toBe("info")
  })

  it("is failed with the error when the last run failed", () => {
    const st = sourceSyncStatus(
      source([table({ status: "error", lastError: "cannot merge" }), table({ id: "u" })], {
        status: "error",
        lastError: "cannot merge",
      }),
    )
    expect(st).toMatchObject({ health: "failed", label: "Failed", variant: "danger" })
    expect(st.explanation).toMatch(
      /^The last sync failed for 1 of 2 enabled tables: cannot merge\. /,
    )
  })

  it("is failed when an enabled table failed even if the job reads idle", () => {
    const st = sourceSyncStatus(source([table({ status: "error", lastError: "boom" })]))
    expect(st.health).toBe("failed")
    expect(st.explanation).toContain("boom")
  })

  it("ignores a failure on a table that is turned off", () => {
    const st = sourceSyncStatus(
      source([table({ status: "error", shouldSync: false }), table({ id: "u" })]),
    )
    expect(st.health).toBe("synced")
  })

  it("is synced with the row count when every enabled table landed rows", () => {
    const st = sourceSyncStatus(
      source([table({ rowCount: 1000 }), table({ id: "u", rowCount: 234 })]),
    )
    expect(st).toMatchObject({ health: "synced", variant: "success" })
    expect(st.label).toBe(`Synced · ${(1234).toLocaleString()} rows`)
  })

  it("does not claim every table synced while a newly enabled one is still pending", () => {
    const st = sourceSyncStatus(
      source([table({ rowCount: 1000 }), table({ id: "u", status: "pending", rowCount: null })]),
    )
    expect(st).toMatchObject({
      health: "partial",
      label: "Synced · 1 of 2 tables",
      variant: "neutral",
    })
    expect(st.explanation).toBe(
      "1 enabled table has not been synced yet. Click Sync now, or wait for the schedule.",
    )
  })

  // The headline is the worst state among enabled tables: a warning outranks "not synced yet".
  it("still warns on an empty table while another enabled table is pending", () => {
    const st = sourceSyncStatus(
      source([table({ rowCount: 0 }), table({ id: "u", status: "pending", rowCount: null })]),
    )
    expect(st).toMatchObject({ health: "empty", variant: "warning" })
  })

  it("warns, not errors, when a successful sync landed no rows for a table", () => {
    const st = sourceSyncStatus(
      source([
        table({ table: "custom__activation_funnel", rowCount: 0 }),
        table({ id: "u", rowCount: 10 }),
      ]),
    )
    expect(st).toMatchObject({
      health: "empty",
      label: "Synced, but no rows",
      variant: "warning",
    })
    expect(st.explanation).toContain("1 table got no rows: custom__activation_funnel")
  })

  it("is never synced when no enabled table has run", () => {
    const st = sourceSyncStatus(
      source([table({ status: "pending", rowCount: null })], { lastSyncedAt: null }),
    )
    expect(st).toMatchObject({ health: "never", label: "Never synced", variant: "neutral" })
  })

  it("says so when every table is turned off", () => {
    expect(sourceSyncStatus(source([table({ shouldSync: false })])).health).toBe("off")
  })
})

describe("tableSyncStatus", () => {
  it("marks a failed table with its own error and the partial-write caveat", () => {
    const st = tableSyncStatus(table({ status: "error", lastError: "cannot merge" }))
    expect(st).toMatchObject({ health: "failed", variant: "danger" })
    expect(st.explanation).toContain("failed: cannot merge")
    expect(st.explanation).toContain("partly rewritten")
  })

  // The service stores the raw exception text, which usually has no closing period.
  it("ends the error with one period", () => {
    for (const lastError of ["connection refused", "connection refused."])
      expect(tableSyncStatus(table({ status: "error", lastError })).explanation).toMatch(
        /^The last sync of this table failed: connection refused\. A failed/,
      )
    expect(
      sourceSyncStatus(source([table({ status: "error", lastError: "connection refused." })]))
        .explanation,
    ).toMatch(/^The last sync failed for 1 of 1 enabled table: connection refused\. A failed/)
    expect(tableSyncStatus(table({ status: "error", lastError: ". " })).explanation).toMatch(
      /^The last sync of this table failed\. A failed/,
    )
  })

  // An incremental rowCount is a running total, so 0 is not a full refresh landing nothing.
  it("gives an empty incremental table no full-refresh explanation", () => {
    const st = tableSyncStatus(
      table({ rowCount: 0, syncType: "incremental", incrementalField: "updated_at" }),
    )
    expect(st.health).toBe("empty")
    expect(st.explanation).not.toContain("full refresh")
    expect(tableSyncStatus(table({ rowCount: 0 })).explanation).toContain("A full refresh")
  })

  it("warns on a synced table with no rows", () => {
    expect(tableSyncStatus(table({ rowCount: 0 }))).toMatchObject({
      health: "empty",
      label: "No rows",
      variant: "warning",
    })
  })

  it("is synced with rows, never synced while pending, off when disabled", () => {
    expect(tableSyncStatus(table()).health).toBe("synced")
    expect(tableSyncStatus(table({ status: "pending", rowCount: null })).health).toBe("never")
    expect(tableSyncStatus(table({ status: "error", shouldSync: false })).health).toBe("off")
    expect(tableSyncStatus(table({ status: "syncing" })).health).toBe("syncing")
  })
})

describe("outcomeStatus", () => {
  it("separates a failed table, an empty one and a populated one", () => {
    const base = { table: "custom__activation_funnel", error: null }
    expect(outcomeStatus({ ...base, rows: 0, ok: false, error: "x" }).health).toBe("failed")
    expect(outcomeStatus({ ...base, rows: 0, ok: true }).health).toBe("empty")
    expect(outcomeStatus({ ...base, rows: 5, ok: true }).health).toBe("synced")
  })

  // The API's outcome rows are what this run wrote (warehouse service: an incremental
  // table's rowCount is the running total, its outcome only the rows appended).
  it("reads 0 rows on an incremental table as nothing new, not as empty", () => {
    const zero = { table: "custom__events", rows: 0, ok: true, error: null }
    expect(outcomeStatus(zero, { syncType: "incremental", incrementalField: "ts" })).toMatchObject({
      health: "synced",
      label: "No new rows",
      variant: "success",
    })
    expect(outcomeStatus(zero, { syncType: "full_refresh", incrementalField: null }).health).toBe(
      "empty",
    )
    expect(outcomeStatus(zero, undefined).health).toBe("empty")
  })

  // The sync service appends only with a cursor column; without one it overwrites (sync.py).
  it("warns on 0 rows from an incremental table with no cursor, which ran as a refresh", () => {
    const zero = { table: "custom__events", rows: 0, ok: true, error: null }
    expect(outcomeStatus(zero, { syncType: "incremental", incrementalField: null }).health).toBe(
      "empty",
    )
  })
})

describe("summarySyncStatus", () => {
  // The list only has the API's counts, but has to say what the source page says.
  it.each([
    ["a full refresh", table({ rowCount: 0 })],
    ["an appending table", table({ rowCount: 0, syncType: "incremental", incrementalField: "id" })],
  ])("explains an empty table in %s like the source page", (_, t) => {
    const detail = source([t])
    const row = {
      ...detail,
      enabledCount: 1,
      enabledSyncedCount: 1,
      emptyTables: [t.table],
      emptyTablesAppend: t.syncType === "incremental",
    }
    expect(summarySyncStatus(row).explanation).toBe(sourceSyncStatus(detail).explanation)
  })
})
