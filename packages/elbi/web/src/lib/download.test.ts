import { describe, expect, it } from "vitest"

import { toCsv } from "./download"

describe("toCsv", () => {
  it("quotes a field holding a carriage return, as RFC 4180 requires", () => {
    expect(toCsv(["note"], [{ note: "a\rb" }])).toBe('note\n"a\rb"')
  })

  it("opens a text cell a spreadsheet would run as a formula as text instead", () => {
    const cells = ['=HYPERLINK("http://x","y")', "+cmd", "-2+3", "@SUM(A1)", "\t=1"]
    const rows = cells.map((name) => ({ name }))
    expect(toCsv(["name"], rows).split("\n")).toEqual([
      "name",
      `"'=HYPERLINK(""http://x"",""y"")"`,
      "'+cmd",
      "'-2+3",
      "'@SUM(A1)",
      "'\t=1",
    ])
  })

  it("leaves numbers alone, negative ones and numeric text included", () => {
    const rows = [
      { n: -5, s: "-5" },
      { n: 1.5, s: "+1.5" },
      { n: 2e-7, s: "-1e3" },
    ]
    expect(toCsv(["n", "s"], rows)).toBe("n,s\n-5,-5\n1.5,+1.5\n2e-7,-1e3")
  })

  it("treats a column name like any other text cell", () => {
    expect(toCsv(["=cmd", "ok"], [{ "=cmd": 1, ok: 2 }])).toBe("'=cmd,ok\n1,2")
  })
})
