// What a cell pushed to the warehouse, and what it cost. For a product whose scale story *is*
// pushdown, showing the SQL and its cost is the difference between "this is fast" and a claim
// about it. The fields mirror what Databricks' query history records (statement, duration, rows
// produced) minus the ones a single-node engine cannot honestly report, like bytes scanned.

import { ChevronDown, ChevronRight, Database } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { useRowKeys } from "@/hooks/useRowKeys"
import type { CellQuery } from "@/lib/notebooks"

const ms = (value: number) =>
  value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`

export function CellQueries({ queries }: { queries: CellQuery[] }) {
  const rowKey = useRowKeys()
  const [open, setOpen] = useState(false)
  if (queries.length === 0) return null

  const total = queries.reduce((sum, q) => sum + q.duration_ms, 0)
  const failed = queries.filter((q) => q.error).length

  return (
    <div className="border-t border-border/60 px-3 py-1.5 text-xs">
      <Button
        variant="link"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="h-auto gap-1.5 p-0 text-xs font-normal text-text-tertiary hover:text-foreground has-[>svg]:px-0"
      >
        {open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        <Database className="size-3" />
        <span>
          {queries.length} {queries.length === 1 ? "query" : "queries"} in the warehouse ·{" "}
          {ms(total)}
          {failed > 0 ? ` · ${failed} failed` : ""}
        </span>
      </Button>
      {open ? (
        <div className="mt-1.5 space-y-1.5">
          {queries.map((query) => (
            <div key={rowKey(query)} className="rounded-md border border-border/60 bg-muted/30 p-2">
              <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-2xs leading-snug">
                {query.sql}
              </pre>
              <div className="mt-1 flex flex-wrap gap-x-3 text-3xs text-text-tertiary">
                <span>{ms(query.duration_ms)}</span>
                <span>
                  {query.rows.toLocaleString()} {query.rows === 1 ? "row" : "rows"}
                  {query.truncated ? " (truncated)" : ""}
                </span>
                {query.error ? <span className="text-destructive">{query.error}</span> : null}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
