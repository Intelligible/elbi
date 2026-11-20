// Pins how the Monitors form reads and rewrites a monitor's config, which can come from
// sync or the API and so is not always what the form itself would have written.

import { describe, expect, it } from "vitest"

import { buildConfig, describeConfig, filterValue, readFilters } from "./monitorConfig"

const edit = (over: Partial<Parameters<typeof buildConfig>[2]> = {}) => ({
  agg: "sum" as const,
  column: "failed",
  key: "stage",
  filters: [],
  ...over,
})

describe("readFilters", () => {
  it("survives a clause with no column or an unknown operator", () => {
    const rows = readFilters({
      filters: [{ value: 1 }, { column: "a", op: "like", value: 2 }, null],
    })
    expect(rows.map((r) => [r.column, r.op])).toEqual([
      ["", "eq"],
      ["a", "eq"],
      ["", "eq"],
    ])
  })
})

describe("filterValue", () => {
  it("keeps a value the user did not touch as it was stored", () => {
    const [zip, ids] = readFilters({
      filters: [
        { column: "zip", op: "eq", value: "02139" },
        { column: "name", op: "in", value: ["a, inc", "b"] },
      ],
    })
    expect(filterValue(zip)).toBe("02139")
    expect(filterValue(ids)).toEqual(["a, inc", "b"])
  })

  it("parses a value the user changed", () => {
    const [row] = readFilters({ filters: [{ column: "zip", op: "eq", value: "02139" }] })
    expect(filterValue({ ...row, value: "10" })).toBe(10)
    expect(filterValue({ ...row, op: "in", value: "a, 2" })).toEqual(["a", 2])
  })
})

describe("buildConfig", () => {
  it("keeps settings the form does not edit", () => {
    const base = { group_by: ["region"], grain: "day", filters: [] }
    const out = buildConfig(base, "metric", edit())
    expect(out).toEqual({ group_by: ["region"], grain: "day" })
  })

  it("keeps the stored measure as written when nothing about it changed", () => {
    const stored = { key: "stage", column: "failed" }
    const out = buildConfig({ measure: stored }, "derivation", edit())
    expect(out.measure).toBe(stored)
  })

  it("replaces the measure when the key changes", () => {
    const out = buildConfig(
      { measure: { column: "failed", key: "stage" } },
      "derivation",
      edit({ key: "day" }),
    )
    expect(out).toEqual({ measure: { agg: "sum", column: "failed", key: "day" } })
  })
})

describe("describeConfig", () => {
  it("names a metric's grouping and grain", () => {
    expect(describeConfig("metric", { group_by: ["region", "plan"], grain: "month" })).toBe(
      "the metric's total, group by region, plan, grain month",
    )
  })
})
