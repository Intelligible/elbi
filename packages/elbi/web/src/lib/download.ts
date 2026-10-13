// Client-side exports of rows the page already holds: no round trip, no re-run.

export function cellText(v: unknown): string {
  if (v === null || v === undefined) return ""
  return typeof v === "object" ? JSON.stringify(v) : String(v)
}

// RFC 4180: a field holding a quote, comma, CR or LF is quoted, its quotes doubled.
function csvField(s: string): string {
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

// A spreadsheet runs a text cell that starts with = + - @ tab or CR as a formula (CSV
// injection), so such a cell gets a leading ' and opens as the text it is. A number,
// or text that is one ("-5"), is left alone: it opens as that number either way.
function csvText(s: string): string {
  const formula = /^[=+\-@\t\r]/.test(s) && !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s)
  return csvField(formula ? `'${s}` : s)
}

export function toCsv(columns: string[], rows: Record<string, unknown>[]): string {
  const cell = (v: unknown) => (typeof v === "string" ? csvText(v) : csvField(cellText(v)))
  const head = columns.map(csvText).join(",")
  const body = rows.map((r) => columns.map((c) => cell(r[c])).join(",")).join("\n")
  return `${head}\n${body}`
}

export function downloadText(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  // In the document, and the URL kept while the browser reads it: some browsers
  // otherwise skip the download or save an empty file. 40 s is FileSaver.js's delay.
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 40_000)
}
