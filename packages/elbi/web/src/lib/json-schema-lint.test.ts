import { json } from "@codemirror/lang-json"
import { EditorState } from "@codemirror/state"
import { describe, expect, it } from "vitest"

import { compileValidator, schemaDiagnostics } from "./json-schema-lint"

// A cut-down stand-in for the widget definition the server serves.
const SCHEMA = {
  $defs: {
    widget: {
      type: "object",
      additionalProperties: false,
      required: ["id", "type", "gridPos"],
      properties: {
        id: { type: "string" },
        type: { enum: ["metric", "chart", "table", "text", "filter"] },
        gridPos: {
          type: "object",
          required: ["x", "y", "w", "h"],
          properties: {
            x: { type: "integer" },
            y: { type: "integer" },
            w: { type: "integer", maximum: 24 },
            h: { type: "integer" },
          },
        },
      },
    },
  },
}

const state = (text: string) => EditorState.create({ doc: text, extensions: [json()] })

const widget = (extra: Record<string, unknown> = {}) =>
  JSON.stringify(
    { id: "mrr", type: "metric", gridPos: { x: 0, y: 0, w: 6, h: 3 }, ...extra },
    null,
    2,
  )

describe("schemaDiagnostics", () => {
  const validate = compileValidator(SCHEMA, "widget")
  if (!validate) throw new Error("the fixture schema should compile")

  it("passes a valid widget", () => {
    expect(schemaDiagnostics(state(widget()), validate)).toEqual([])
  })

  it("says nothing about text that is not JSON yet", () => {
    // The parse linter owns that; two errors for one typo is noise.
    expect(schemaDiagnostics(state("{ not json"), validate)).toEqual([])
  })

  it("reports a type outside the allowed set, at the value", () => {
    const text = widget({ type: "bogus" }).replace('"metric"', '"bogus"')
    const [error] = schemaDiagnostics(state(text), validate)

    expect(error.message).toContain("must be one of")
    expect(text.slice(error.from, error.to)).toBe('"bogus"')
  })

  it("reports a missing required key against the object it is missing from", () => {
    const text = JSON.stringify({ id: "mrr", type: "metric" }, null, 2)
    const [error] = schemaDiagnostics(state(text), validate)

    expect(error.message).toContain("gridPos")
    expect(error.from).toBe(0)
  })

  it("reports a bad value nested inside an object, at that value", () => {
    const text = widget().replace('"w": 6', '"w": 99')
    const [error] = schemaDiagnostics(state(text), validate)

    expect(error.message).toMatch(/<= 24|maximum/)
    expect(text.slice(error.from, error.to)).toBe("99")
  })

  it("names an unknown key rather than only pointing at the object", () => {
    const text = widget({ colour: "red" })
    const [error] = schemaDiagnostics(state(text), validate)

    expect(error.message).toContain("colour")
  })
})

describe("compileValidator", () => {
  it("returns null for a definition the schema does not have", () => {
    expect(compileValidator(SCHEMA, "nope")).toBeNull()
  })

  it("compiles the whole schema when no definition is named", () => {
    expect(compileValidator({ type: "object" })).not.toBeNull()
  })
})
