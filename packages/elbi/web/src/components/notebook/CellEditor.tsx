import {
  autocompletion,
  type CompletionContext,
  type CompletionResult,
  closeBrackets,
} from "@codemirror/autocomplete"
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands"
import { python } from "@codemirror/lang-python"
import { type SQLNamespace, sql } from "@codemirror/lang-sql"
import {
  bracketMatching,
  defaultHighlightStyle,
  indentOnInput,
  syntaxHighlighting,
} from "@codemirror/language"
import type { Extension } from "@codemirror/state"
import { Compartment, EditorState, Prec } from "@codemirror/state"
import { oneDark } from "@codemirror/theme-one-dark"
import { EditorView, hoverTooltip, keymap, lineNumbers } from "@codemirror/view"
import { useEffect, useRef } from "react"

import type { CompleteReply, InspectReply } from "@/lib/notebooks"

// A single code (or markdown) cell editor built on CodeMirror 6. The view is created once
// and kept across renders; the parent's latest callbacks live in a ref so the run/change
// handlers stay current without rebuilding the editor (rebuilding would drop cursor and
// undo history). External value changes (a cell loaded from the server) reconcile through
// a transaction only when they actually differ, so local typing is never clobbered.

type Props = {
  value: string
  language: "python" | "markdown" | "sql"
  // For SQL, a {table: [column, ...]} map that drives schema-aware autocomplete. It can
  // change as the selected source's catalog loads; the editor reconfigures in place.
  sqlSchema?: SQLNamespace
  editable?: boolean
  dark: boolean
  onChange: (value: string) => void
  onRun: () => void
  onRunNoAdvance: () => void
  onFocus?: () => void
  // SQL-editor mode: when provided, ⌘↵ runs the selection (or the statement under the
  // cursor when nothing is selected) and ⌘⇧↵ runs the whole document, passing the exact
  // text to execute. Notebooks leave this unset and use onRun/onRunNoAdvance.
  onRunText?: (text: string) => void
  // Kernel-backed help. When provided (notebook Python cells), completions come from the live
  // namespace and hovering a name shows its signature and docstring: the editor's
  // tab-completion and Shift-Tab help. Left unset elsewhere (SQL/markdown).
  complete?: (code: string, cursorPos: number) => Promise<CompleteReply>
  inspect?: (code: string, cursorPos: number) => Promise<InspectReply>
}

// The text ⌘↵ should execute in SQL mode: the selection if any, else the ;-delimited
// statement containing the cursor, else the whole document.
function queryToRun(view: EditorView): string {
  const { state } = view
  const sel = state.selection.main
  if (!sel.empty) return state.sliceDoc(sel.from, sel.to)
  const doc = state.doc.toString()
  const head = sel.head
  let start = 0
  for (let i = 0; i < doc.length; i++) {
    if (doc[i] === ";") {
      if (head <= i) return doc.slice(start, i).trim() || doc.trim()
      start = i + 1
    }
  }
  return doc.slice(start).trim() || doc.trim()
}

// The language extension for a cell: Python, SQL (optionally schema-aware), or none.
function languageExtension(language: Props["language"], sqlSchema?: SQLNamespace): Extension {
  if (language === "python") return python()
  if (language === "sql") return sql(sqlSchema ? { schema: sqlSchema } : undefined)
  return []
}

// A compact base theme so cells read like a notebook, not a full IDE pane.
const baseTheme = EditorView.theme({
  "&": { fontSize: "13px", backgroundColor: "transparent" },
  ".cm-content": {
    fontFamily: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace",
    padding: "8px 0",
  },
  ".cm-gutters": { backgroundColor: "transparent", border: "none" },
  "&.cm-focused": { outline: "none" },
  ".cm-line": { padding: "0 8px" },
  ".cm-inspect-tooltip pre": {
    margin: "0",
    padding: "8px 10px",
    maxWidth: "480px",
    maxHeight: "320px",
    overflow: "auto",
    whiteSpace: "pre-wrap",
    fontSize: "12px",
    lineHeight: "1.45",
  },
})

export function CellEditor({
  value,
  language,
  sqlSchema,
  editable = true,
  dark,
  onChange,
  onRun,
  onRunNoAdvance,
  onFocus,
  onRunText,
  complete,
  inspect,
}: Props) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const themeComp = useRef(new Compartment())
  const editableComp = useRef(new Compartment())
  const langComp = useRef(new Compartment())
  // Everything the editor needs from props goes through a ref, and the ref is refreshed
  // every render. One rule, so nothing is read two ways: the handlers stay live without
  // rebuilding the view, and the values below are read once, at mount, because each has
  // a Compartment effect further down that reconfigures the running editor when it
  // changes. Listing them as effect dependencies instead would tear the editor down on
  // every theme flip or keystroke, taking the cursor, selection and undo history with
  // it -- which is the whole reason CodeMirror has compartments.
  const cb = useRef({ onChange, onRun, onRunNoAdvance, onFocus, onRunText, complete, inspect })
  cb.current = { onChange, onRun, onRunNoAdvance, onFocus, onRunText, complete, inspect }
  const initial = useRef({ value, language, sqlSchema, dark, editable })
  initial.current = { value, language, sqlSchema, dark, editable }

  useEffect(() => {
    const runKeys = Prec.highest(
      keymap.of([
        {
          key: "Mod-Shift-Enter",
          run: (v) => {
            if (!cb.current.onRunText) return false
            cb.current.onRunText(v.state.doc.toString())
            return true
          },
        },
        {
          key: "Shift-Enter",
          run: (v) => {
            if (cb.current.onRunText) cb.current.onRunText(queryToRun(v))
            else cb.current.onRun()
            return true // handled, so CodeMirror does not also insert a newline
          },
        },
        {
          key: "Mod-Enter",
          run: (v) => {
            if (cb.current.onRunText) cb.current.onRunText(queryToRun(v))
            else cb.current.onRunNoAdvance()
            return true // handled, so CodeMirror does not also insert a newline
          },
        },
      ]),
    )
    // Kernel-backed completion: ask the live namespace for matches at the cursor and map
    // the reply's token span to a CodeMirror completion result. Falls back to null when
    // no kernel help is wired, leaving the default (word/SQL-schema) completion in place.
    const kernelCompletion = async (
      context: CompletionContext,
    ): Promise<CompletionResult | null> => {
      const fn = cb.current.complete
      if (!fn) return null
      const word = context.matchBefore(/[\w.]+/)
      if (!context.explicit && (!word || word.from === word.to)) return null
      const reply = await fn(context.state.doc.toString(), context.pos)
      if (!reply.matches.length) return null
      return {
        from: reply.cursor_start,
        to: reply.cursor_end,
        options: reply.matches.map((label) => ({ label })),
        validFor: /^[\w.]*$/,
      }
    }
    const completion = cb.current.complete
      ? autocompletion({ override: [kernelCompletion] })
      : autocompletion()
    // Kernel-backed hover help: the object's signature and docstring under the pointer.
    const hover = hoverTooltip(async (_view, pos) => {
      const fn = cb.current.inspect
      if (!fn) return null
      const reply = await fn(_view.state.doc.toString(), pos)
      const text = reply.found ? String(reply.data["text/plain"] ?? "") : ""
      if (!text) return null
      return {
        pos,
        create: () => {
          const dom = document.createElement("div")
          dom.className = "cm-inspect-tooltip"
          const pre = document.createElement("pre")
          pre.textContent = text
          dom.appendChild(pre)
          return { dom }
        },
      }
    })
    const state = EditorState.create({
      doc: initial.current.value,
      extensions: [
        lineNumbers(),
        history(),
        indentOnInput(),
        bracketMatching(),
        closeBrackets(),
        completion,
        ...(cb.current.inspect ? [hover] : []),
        syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
        keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
        runKeys,
        langComp.current.of(languageExtension(initial.current.language, initial.current.sqlSchema)),
        baseTheme,
        themeComp.current.of(initial.current.dark ? oneDark : []),
        editableComp.current.of(EditorView.editable.of(initial.current.editable)),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) cb.current.onChange(u.state.doc.toString())
          if (u.focusChanged && u.view.hasFocus) cb.current.onFocus?.()
        }),
      ],
    })
    if (!host.current) return
    view.current = new EditorView({ state, parent: host.current })
    return () => {
      view.current?.destroy()
      view.current = null
    }
  }, [])

  // Reconcile an external value change without disturbing in-progress local edits.
  useEffect(() => {
    const v = view.current
    if (!v) return
    const current = v.state.doc.toString()
    if (value !== current) {
      v.dispatch({ changes: { from: 0, to: current.length, insert: value } })
    }
  }, [value])

  useEffect(() => {
    view.current?.dispatch({
      effects: themeComp.current.reconfigure(dark ? oneDark : []),
    })
  }, [dark])

  useEffect(() => {
    view.current?.dispatch({
      effects: editableComp.current.reconfigure(EditorView.editable.of(editable)),
    })
  }, [editable])

  // Reconfigure the language when the SQL schema arrives or changes, so autocomplete
  // knows the real tables and columns without rebuilding the editor.
  useEffect(() => {
    view.current?.dispatch({
      effects: langComp.current.reconfigure(languageExtension(language, sqlSchema)),
    })
  }, [language, sqlSchema])

  return <div ref={host} />
}
