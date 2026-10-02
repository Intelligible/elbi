// Client-side exports of rows the page already holds: no round trip, no re-run.

export function cellText(v: unknown): string {
  if (v === null || v === undefined) return ""
  return typeof v === "object" ? JSON.stringify(v) : String(v)
}

export function toCsv(columns: string[], rows: Record<string, unknown>[]): string {
  const esc = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s)
  const head = columns.map(esc).join(",")
  const body = rows.map((r) => columns.map((c) => esc(cellText(r[c]))).join(",")).join("\n")
  return `${head}\n${body}`
}

export function downloadText(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
