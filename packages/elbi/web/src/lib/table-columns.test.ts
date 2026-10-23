// Pins what a table tile's `viz.columns` may say and how a width change is written back.
// The contract: a plain name still works, an object adds a label, a width and wrapping,
// and a cell holding a web address becomes a link without any other scheme ever doing so.

import { describe, expect, it } from "vitest"

import {
  cellSegments,
  columnFields,
  MIN_COLUMN_WIDTH,
  parseColumns,
  withColumnWidth,
} from "./table-columns"

const ROWS = [
  { unit: "chat", cost: 1, reason: "why" },
  { unit: "sql", extra: 2 },
]

describe("parseColumns", () => {
  it("shows every key any row has when nothing is configured", () => {
    expect(parseColumns(undefined, ROWS).map((c) => c.field)).toEqual([
      "unit",
      "cost",
      "reason",
      "extra",
    ])
  })

  it("accepts plain names exactly as before", () => {
    expect(parseColumns(["cost", "unit"], ROWS)).toEqual([{ field: "cost" }, { field: "unit" }])
  })

  it("accepts an object with a label, a width and wrapping", () => {
    const [col] = parseColumns([{ field: "reason", label: "Why", width: 420, wrap: true }], ROWS)
    expect(col).toEqual({ field: "reason", label: "Why", width: 420, wrap: true })
  })

  it("accepts fit and fill as widths", () => {
    const cols = parseColumns(
      [
        { field: "unit", width: "fit" },
        { field: "reason", width: "fill" },
      ],
      ROWS,
    )
    expect(cols.map((c) => c.width)).toEqual(["fit", "fill"])
  })

  it("mixes names and objects in one list", () => {
    const cols = parseColumns(["unit", { field: "reason", width: 300 }], ROWS)
    expect(cols).toEqual([{ field: "unit" }, { field: "reason", width: 300 }])
  })

  it("drops a width that is not a positive number, fit or fill", () => {
    const cols = parseColumns(
      [
        { field: "a", width: -5 },
        { field: "b", width: 0 },
        { field: "c", width: "wide" },
        { field: "d", width: Number.NaN },
      ],
      ROWS,
    )
    expect(cols.every((c) => c.width === undefined)).toBe(true)
  })

  it("skips an entry that names no field", () => {
    expect(parseColumns([{ width: 100 }, 7, null, "unit"], ROWS)).toEqual([{ field: "unit" }])
  })
})

describe("columnFields", () => {
  it("is the field names, in order, whatever the configured shape", () => {
    expect(columnFields(parseColumns(["unit", { field: "cost", width: 90 }], ROWS))).toEqual([
      "unit",
      "cost",
    ])
  })
})

describe("withColumnWidth", () => {
  it("sets one column's width and leaves the others as they were", () => {
    const viz = withColumnWidth({ columns: ["unit", "reason"], pageSize: 10 }, "reason", 420, ROWS)
    expect(viz).toEqual({ columns: ["unit", { field: "reason", width: 420 }], pageSize: 10 })
  })

  it("writes out the derived column list when none was configured", () => {
    const viz = withColumnWidth(undefined, "cost", 90, ROWS)
    expect(viz.columns).toEqual(["unit", { field: "cost", width: 90 }, "reason", "extra"])
  })

  it("keeps a column's label and wrapping when only its width changes", () => {
    const viz = withColumnWidth(
      { columns: [{ field: "reason", label: "Why", wrap: true, width: 200 }] },
      "reason",
      360,
      ROWS,
    )
    expect(viz.columns).toEqual([{ field: "reason", label: "Why", wrap: true, width: 360 }])
  })

  it("rounds a dragged width and never goes below the minimum", () => {
    expect(withColumnWidth({ columns: ["unit"] }, "unit", 200.6, ROWS).columns).toEqual([
      { field: "unit", width: 201 },
    ])
    expect(withColumnWidth({ columns: ["unit"] }, "unit", 3, ROWS).columns).toEqual([
      { field: "unit", width: MIN_COLUMN_WIDTH },
    ])
  })

  it("sets fit, which a drag back to content size writes", () => {
    expect(
      withColumnWidth({ columns: [{ field: "unit", width: 300 }] }, "unit", "fit", ROWS).columns,
    ).toEqual([{ field: "unit", width: "fit" }])
  })

  it("does not change the viz it was given", () => {
    const viz = { columns: ["unit"] }
    withColumnWidth(viz, "unit", 100, ROWS)
    expect(viz).toEqual({ columns: ["unit"] })
  })
})

describe("cellSegments", () => {
  it("is one plain segment when there is no link", () => {
    expect(cellSegments("Above Modal's raw compute")).toEqual([
      { text: "Above Modal's raw compute" },
    ])
  })

  it("turns a markdown link into a labelled link", () => {
    expect(cellSegments("see [Modal pricing](https://modal.com/pricing) for it")).toEqual([
      { text: "see " },
      { text: "Modal pricing", href: "https://modal.com/pricing" },
      { text: " for it" },
    ])
  })

  it("turns a bare web address into a link that shows the address", () => {
    expect(cellSegments("https://modal.com/pricing")).toEqual([
      { text: "https://modal.com/pricing", href: "https://modal.com/pricing" },
    ])
  })

  it("leaves trailing punctuation out of a bare address", () => {
    expect(cellSegments("at https://modal.com/pricing, then")).toEqual([
      { text: "at " },
      { text: "https://modal.com/pricing", href: "https://modal.com/pricing" },
      { text: ", then" },
    ])
  })

  it("links two addresses separated by a space", () => {
    const links = cellSegments("https://a.example/x https://b.example/y").filter((s) => s.href)
    expect(links.map((s) => s.href)).toEqual(["https://a.example/x", "https://b.example/y"])
  })

  it("never links a scheme other than http or https", () => {
    for (const bad of [
      "[x](javascript:alert(1))",
      "[x](data:text/html,hi)",
      "javascript:alert(1)",
      "[x](ftp://host/file)",
    ]) {
      expect(cellSegments(bad).some((s) => s.href)).toBe(false)
    }
  })

  it("keeps the text of a refused markdown link visible", () => {
    expect(
      cellSegments("[x](javascript:alert(1))")
        .map((s) => s.text)
        .join(""),
    ).toContain("x")
  })

  it("is empty for an empty cell", () => {
    expect(cellSegments("")).toEqual([])
  })
})
