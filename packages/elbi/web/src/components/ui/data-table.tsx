import * as React from "react"
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react"

import { cn } from "@/lib/utils"

// A dense, full-width data table for list scenes: sortable column headers, tight
// bordered rows, whole-row click, an
// optional trailing actions column, and client-side pagination. List pages own
// their search/filter row (rendered above via the `toolbar`) and pass the
// already-filtered rows in.

export type Column<T> = {
  key: string
  title: React.ReactNode
  /* Fixed column width, e.g. "34%" or 140. */
  width?: string | number
  align?: "left" | "right" | "center"
  /* Provide to make the header sortable. */
  sorter?: (a: T, b: T) => number
  render: (row: T) => React.ReactNode
  className?: string
  headerClassName?: string
}

type SortState = { key: string; dir: "asc" | "desc" } | null

export function DataTable<T>({
  columns,
  data,
  rowKey,
  onRowClick,
  pageSize = 25,
  toolbar,
  empty = "Nothing here yet.",
  className,
}: {
  columns: Column<T>[]
  data: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  pageSize?: number
  /* A search / filter row rendered above the table. */
  toolbar?: React.ReactNode
  empty?: React.ReactNode
  className?: string
}) {
  const [sort, setSort] = React.useState<SortState>(null)
  const [page, setPage] = React.useState(0)

  const sorted = React.useMemo(() => {
    if (!sort) return data
    const col = columns.find((c) => c.key === sort.key)
    if (!col?.sorter) return data
    const s = col.sorter
    const out = [...data].sort(s)
    if (sort.dir === "desc") out.reverse()
    return out
  }, [data, sort, columns])

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize))
  const clampedPage = Math.min(page, pageCount - 1)
  const start = clampedPage * pageSize
  const rows = sorted.slice(start, start + pageSize)

  const toggleSort = (key: string) =>
    setSort((prev) =>
      prev?.key !== key
        ? { key, dir: "asc" }
        : prev.dir === "asc"
          ? { key, dir: "desc" }
          : null
    )

  return (
    <div className={cn("space-y-3", className)}>
      {(toolbar || sorted.length > pageSize) && (
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 flex-1">{toolbar}</div>
          {sorted.length > pageSize && (
            <div className="flex shrink-0 items-center gap-1 text-xs text-text-tertiary tabular-nums">
              <span>
                {start + 1}–{Math.min(start + pageSize, sorted.length)} of{" "}
                {sorted.length}
              </span>
              <button
                type="button"
                className="rounded-md px-1.5 py-1 transition-colors hover:bg-muted disabled:opacity-40"
                disabled={clampedPage === 0}
                onClick={() => setPage(clampedPage - 1)}
                aria-label="Previous page"
              >
                <ChevronUp className="size-3.5 -rotate-90" />
              </button>
              <button
                type="button"
                className="rounded-md px-1.5 py-1 transition-colors hover:bg-muted disabled:opacity-40"
                disabled={clampedPage >= pageCount - 1}
                onClick={() => setPage(clampedPage + 1)}
                aria-label="Next page"
              >
                <ChevronDown className="size-3.5 -rotate-90" />
              </button>
            </div>
          )}
        </div>
      )}

      <div className="w-full overflow-x-auto rounded-lg border border-border bg-card shadow-sm">
        <table className="w-full border-collapse text-[0.8125rem]">
          <thead>
            <tr className="border-b border-border bg-surface-secondary">
              {columns.map((col) => {
                const active = sort?.key === col.key
                return (
                  <th
                    key={col.key}
                    style={{ width: col.width }}
                    className={cn(
                      "px-3 py-2 text-left align-middle text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-text-tertiary whitespace-nowrap select-none",
                      col.align === "right" && "text-right",
                      col.align === "center" && "text-center",
                      col.sorter && "cursor-pointer hover:text-foreground",
                      col.headerClassName
                    )}
                    onClick={col.sorter ? () => toggleSort(col.key) : undefined}
                  >
                    <span
                      className={cn(
                        "inline-flex items-center gap-1",
                        col.align === "right" && "flex-row-reverse"
                      )}
                    >
                      {col.title}
                      {col.sorter &&
                        (active ? (
                          sort!.dir === "asc" ? (
                            <ChevronUp className="size-3" />
                          ) : (
                            <ChevronDown className="size-3" />
                          )
                        ) : (
                          <ChevronsUpDown className="size-3 opacity-40" />
                        ))}
                    </span>
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-12 text-center text-sm text-text-tertiary"
                >
                  {empty}
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr
                  key={rowKey(row)}
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  className={cn(
                    "border-t border-border transition-colors",
                    onRowClick && "cursor-pointer hover:bg-muted/60"
                  )}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={cn(
                        "px-3 py-1.5 align-middle text-foreground",
                        col.align === "right" && "text-right",
                        col.align === "center" && "text-center",
                        col.className
                      )}
                    >
                      {col.render(row)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
