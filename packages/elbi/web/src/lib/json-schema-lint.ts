import { syntaxTree } from "@codemirror/language"
import type { Diagnostic } from "@codemirror/lint"
import type { EditorState } from "@codemirror/state"
// The 2020-12 build: the spec's schema declares that draft, and ajv's default
// export is draft-07 and throws on it.
import type { ErrorObject, ValidateFunction } from "ajv"
import Ajv from "ajv/dist/2020"

/**
 * Validate a JSON document against a schema and mark each violation where it is.
 *
 * The parse error is the other linter's job; this one only runs on text that parses,
 * and answers the next question: is this a *valid* widget, or a well-formed object the
 * save will refuse.
 */
export function schemaDiagnostics(state: EditorState, validate: ValidateFunction): Diagnostic[] {
  const text = state.doc.toString()
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return [] // not JSON yet; jsonParseLinter is already saying so
  }
  if (validate(parsed)) return []
  return (validate.errors ?? []).map((error) => {
    const { from, to } = locate(state, error)
    return { from, to, severity: "error" as const, message: describe(error) }
  })
}

/** Compile a validator for a schema, or a sub-schema of it by `$defs` name. */
export function compileValidator(
  schema: Record<string, unknown>,
  definition?: string,
): ValidateFunction | null {
  // The spec's schema is drafted with formats this editor does not need to check;
  // an unknown keyword should not stop the whole document being validated.
  const ajv = new Ajv({ allErrors: true, strict: false })
  try {
    if (!definition) return ajv.compile(schema)
    const defs = (schema.$defs ?? {}) as Record<string, unknown>
    if (!defs[definition]) return null
    // Compiled against the whole schema so `$ref`s into sibling definitions resolve.
    return ajv.compile({ ...schema, ...(defs[definition] as object) })
  } catch {
    return null
  }
}

/** Where in the text an error's `instancePath` points, falling back to the whole doc. */
function locate(state: EditorState, error: ErrorObject): { from: number; to: number } {
  const path = error.instancePath.split("/").filter(Boolean).map(decodePointer)
  // A missing property is reported against its parent, which is the right place to
  // draw it: the gap is in the object, not at some key that isn't there.
  let node = syntaxTree(state).topNode.firstChild
  for (const step of path) {
    const child = descend(state, node, step)
    if (!child) break
    node = child
  }
  const whole = { from: 0, to: Math.min(state.doc.length, 1) }
  if (!node) return whole
  return { from: node.from, to: Math.max(node.to, node.from + 1) }
}

function descend(
  state: EditorState,
  node: ReturnType<typeof syntaxTree>["topNode"]["firstChild"],
  step: string,
) {
  if (!node) return null
  const index = Number(step)
  if (Number.isInteger(index) && node.name === "Array") {
    let seen = 0
    for (let child = node.firstChild; child; child = child.nextSibling) {
      if (child.name === "⚠") continue
      if (seen === index) return child
      seen += 1
    }
    return null
  }
  for (let child = node.firstChild; child; child = child.nextSibling) {
    if (child.name !== "Property") continue
    const key = child.firstChild
    if (!key) continue
    if (state.doc.sliceString(key.from + 1, key.to - 1) === step) {
      // The value is the property's last child; `key.nextSibling` is the ':' between.
      const value = child.lastChild
      return value && value !== key ? value : child
    }
  }
  return null
}

function decodePointer(step: string): string {
  return step.replace(/~1/g, "/").replace(/~0/g, "~")
}

function describe(error: ErrorObject): string {
  const where = error.instancePath || "this object"
  if (error.keyword === "required") {
    return `${where} is missing "${(error.params as { missingProperty: string }).missingProperty}"`
  }
  if (error.keyword === "enum") {
    const allowed = (error.params as { allowedValues: unknown[] }).allowedValues
    return `${where} must be one of: ${allowed.join(", ")}`
  }
  if (error.keyword === "additionalProperties") {
    return `${where} has an unknown key "${(error.params as { additionalProperty: string }).additionalProperty}"`
  }
  return `${where} ${error.message ?? "is invalid"}`
}
