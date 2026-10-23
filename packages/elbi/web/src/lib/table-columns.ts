// What a table tile's `viz.columns` may say, and how a width change is written back.
//
// An entry is either a plain field name, as it always was, or an object that adds a
// label, a width and wrapping. Both live in the dashboard spec, so a width set by hand in
// YAML and one dragged in the UI are the same value and survive a pull and a sync alike.

export const MIN_COLUMN_WIDTH = 48

// Pixels, `fit` to size to the content, or `fill` to take what the other columns leave.
export type ColumnWidth = number | "fit" | "fill"

export interface ColumnSpec {
  field: string
  label?: string
  width?: ColumnWidth
  wrap?: boolean
}

type Row = Record<string, unknown>

function validWidth(value: unknown): ColumnWidth | undefined {
  if (value === "fit" || value === "fill") return value
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined
}

function parseEntry(entry: unknown): ColumnSpec | null {
  if (typeof entry === "string") return { field: entry }
  if (entry === null || typeof entry !== "object") return null
  const { field, label, width, wrap } = entry as Record<string, unknown>
  if (typeof field !== "string" || field === "") return null
  const spec: ColumnSpec = { field }
  if (typeof label === "string") spec.label = label
  const w = validWidth(width)
  if (w !== undefined) spec.width = w
  if (typeof wrap === "boolean") spec.wrap = wrap
  return spec
}

// The columns a table shows: the configured ones, else every key any row has, so a key
// the first row lacks is kept.
export function parseColumns(configured: unknown, rows: Row[]): ColumnSpec[] {
  if (Array.isArray(configured)) {
    return configured.map(parseEntry).filter((c): c is ColumnSpec => c !== null)
  }
  return [...new Set(rows.flatMap((r) => Object.keys(r)))].map((field) => ({ field }))
}

export function columnFields(columns: ColumnSpec[]): string[] {
  return columns.map((c) => c.field)
}

function entryField(entry: unknown): string | undefined {
  if (typeof entry === "string") return entry
  const field = (entry as Row | null)?.field
  return typeof field === "string" ? field : undefined
}

// A copy of `viz` with one column's width set. Other entries are left exactly as they
// were, so a hand-written label or a plain name is not rewritten by a drag.
export function withColumnWidth(
  viz: Row | undefined,
  field: string,
  width: ColumnWidth,
  rows: Row[],
): Row {
  const base = Array.isArray(viz?.columns)
    ? (viz.columns as unknown[])
    : parseColumns(undefined, rows).map((c) => c.field)
  const next = typeof width === "number" ? Math.max(MIN_COLUMN_WIDTH, Math.round(width)) : width
  const columns = base.map((entry) => {
    if (entryField(entry) !== field) return entry
    return typeof entry === "string" ? { field, width: next } : { ...(entry as Row), width: next }
  })
  return { ...(viz ?? {}), columns }
}

export interface CellSegment {
  text: string
  href?: string
}

// A markdown link, or a bare web address up to the next space.
const LINK = /\[([^\]\n]+)\]\(([^)\s]+)\)|https?:\/\/[^\s<>"]+/gi
const WEB = /^https?:\/\//i
// Punctuation that ends a sentence rather than an address.
const TRAILING = /[.,;:!?)\]'"]+$/

// Splits a cell into plain text and links. Only http and https ever become a link; a
// markdown link to anything else stays visible as the text it was written as.
export function cellSegments(text: string): CellSegment[] {
  const out: CellSegment[] = []
  const plain = (s: string) => {
    if (!s) return
    const last = out[out.length - 1]
    if (last && last.href === undefined) last.text += s
    else out.push({ text: s })
  }
  let at = 0
  for (const m of text.matchAll(LINK)) {
    const start = m.index ?? 0
    plain(text.slice(at, start))
    if (m[2] !== undefined) {
      if (WEB.test(m[2])) out.push({ text: m[1], href: m[2] })
      else plain(m[0])
      at = start + m[0].length
    } else {
      const url = m[0].replace(TRAILING, "")
      out.push({ text: url, href: url })
      plain(m[0].slice(url.length))
      at = start + m[0].length
    }
  }
  plain(text.slice(at))
  return out
}
