import { closeBrackets } from "@codemirror/autocomplete"
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands"
import { json, jsonParseLinter } from "@codemirror/lang-json"
import {
  bracketMatching,
  defaultHighlightStyle,
  indentOnInput,
  syntaxHighlighting,
} from "@codemirror/language"
import { linter, lintGutter } from "@codemirror/lint"
import { Compartment, EditorState } from "@codemirror/state"
import { oneDark } from "@codemirror/theme-one-dark"
import { EditorView, keymap, lineNumbers } from "@codemirror/view"
import { useEffect, useMemo, useRef } from "react"

import { compileValidator, schemaDiagnostics } from "@/lib/json-schema-lint"

// Smaller than the notebook's cells: this is a config panel inside a dialog, not a
// document being written.
const baseTheme = EditorView.theme({
  "&": { fontSize: "12px", backgroundColor: "transparent" },
  ".cm-content": {
    fontFamily: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace",
    padding: "8px 0",
  },
  ".cm-gutters": { backgroundColor: "transparent", border: "none" },
  "&.cm-focused": { outline: "none" },
  ".cm-line": { padding: "0 8px" },
})

/**
 * A small JSON editor: highlighting, bracket matching, and the parse error marked where
 * it is rather than only reported on save.
 *
 * The value is controlled from outside, but the document is only replaced when it
 * actually differs from what is on screen — writing every keystroke back through
 * `value` would move the cursor to the end of the document on each one.
 */
export function JsonEditor({
  value,
  label,
  dark,
  schema,
  definition,
  onChange,
}: {
  value: string
  label: string
  dark: boolean
  /** The JSON Schema to check against, once it has loaded. */
  schema?: Record<string, unknown> | null
  /** A `$defs` name to check against instead of the whole schema (a single widget). */
  definition?: string
  onChange: (next: string) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const theme = useRef(new Compartment())
  const latest = useRef(onChange)
  latest.current = onChange
  // What the editor is built with. Read through a ref so the build effect depends on
  // nothing and never tears the view down mid-edit when a prop changes.
  const initial = useRef({ value, label, dark })
  const validator = useMemo(
    () => (schema ? compileValidator(schema, definition) : null),
    [schema, definition],
  )
  // Read through a ref so a schema arriving later needs no rebuild of the view: the
  // linter simply starts finding things on its next run.
  const check = useRef(validator)
  check.current = validator

  useEffect(() => {
    if (!host.current || view.current) return
    const editor = new EditorView({
      state: EditorState.create({
        doc: initial.current.value,
        extensions: [
          lineNumbers(),
          history(),
          indentOnInput(),
          bracketMatching(),
          closeBrackets(),
          json(),
          // oneDark carries its own colours; without this, the light theme has none
          // and the JSON renders as one undifferentiated block.
          syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
          linter(jsonParseLinter()),
          // Two linters, two questions: is this JSON at all, and is it a valid widget.
          linter((view) => (check.current ? schemaDiagnostics(view.state, check.current) : [])),
          lintGutter(),
          keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
          EditorView.lineWrapping,
          baseTheme,
          theme.current.of(initial.current.dark ? oneDark : []),
          EditorView.contentAttributes.of({
            "aria-label": initial.current.label,
            role: "textbox",
          }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) latest.current(update.state.doc.toString())
          }),
        ],
      }),
      parent: host.current,
    })
    view.current = editor
    return () => {
      editor.destroy()
      view.current = null
    }
    // Built once: later edits arrive through the effects below.
  }, [])

  useEffect(() => {
    const editor = view.current
    if (!editor) return
    const current = editor.state.doc.toString()
    if (current === value) return
    editor.dispatch({ changes: { from: 0, to: current.length, insert: value } })
  }, [value])

  useEffect(() => {
    view.current?.dispatch({ effects: theme.current.reconfigure(dark ? oneDark : []) })
  }, [dark])

  // The schema is fetched, so it usually arrives after the editor is built. Nudge the
  // document so the linters run again against it.
  useEffect(() => {
    const editor = view.current
    if (!editor || !validator) return
    editor.dispatch({ changes: { from: 0, to: 0, insert: "" } })
  }, [validator])

  return <div ref={host} className="overflow-auto rounded-md border border-border" />
}
