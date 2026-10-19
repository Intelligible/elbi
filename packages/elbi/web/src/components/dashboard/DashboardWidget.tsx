// Renders a single dashboard widget from its spec and resolved data. Chart and map
// widgets reuse the shared VizView (which routes a lat/long spec to the deck.gl map);
// metric (a shared metric's value), table, and text are light renderers. A widget whose derivation failed shows
// its error in place rather than blanking.

import { AlertCircle, GripVertical, MoreHorizontal } from "lucide-react"
import {
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react"

import { NotebookMarkdown } from "@/components/notebook/NotebookMarkdown"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { VizView } from "@/components/viz/VizView"
import { useRowKeys } from "@/hooks/useRowKeys"
import type { Widget, WidgetData } from "@/lib/dashboards"
import { downloadText, toCsv } from "@/lib/download"
import { formatMetricValue } from "@/lib/metrics"
import {
  type CellSegment,
  type ColumnSpec,
  type ColumnWidth,
  cellSegments,
  columnFields,
  MIN_COLUMN_WIDTH,
  parseColumns,
} from "@/lib/table-columns"
import { EMPTY } from "@/lib/utils"

type Row = Record<string, string | number>

function asRows(value: unknown): Row[] {
  return Array.isArray(value) ? (value as Row[]) : []
}

// The fields a tile downloads: the same columns it shows, so the CSV matches the screen.
function columnsOf(widget: Widget, rows: Row[]): string[] {
  return columnFields(parseColumns(widget.viz?.columns, rows))
}

function CenterNote({ children }: { children: ReactNode }) {
  return <div className="grid h-full place-items-center text-xs text-text-tertiary">{children}</div>
}

function MetricBody({ widget, data }: { widget: Widget; data?: WidgetData }) {
  // The tile shows its shared metric's single value (the one row, keyed by the metric's
  // name), formatted as the metric defines, so the number and its display live in one
  // place for every surface that shows it.
  const name = widget.bind?.metric
  const raw = name ? asRows(data?.value)[0]?.[name] : undefined
  if (raw === null || raw === undefined) {
    const hint = name ? `no value for metric “${name}”` : "bind a metric to this tile"
    return (
      <div className="flex h-full flex-col justify-center">
        <div className="text-3xl font-semibold text-text-tertiary/40">{EMPTY}</div>
        <div className="mt-1 text-xs text-text-tertiary">{hint}</div>
      </div>
    )
  }
  return (
    <div className="flex h-full flex-col justify-center">
      <div className="text-3xl font-semibold tabular-nums">
        {formatMetricValue(raw, data?.format ?? undefined)}
      </div>
    </div>
  )
}

// How far one keypress moves a column edge.
const KEY_STEP = 10

function widthStyle(width: ColumnWidth | undefined) {
  if (typeof width === "number") return { width, minWidth: width, maxWidth: width }
  return width === "fit" ? { width: "1%" } : undefined
}

function Cell({ segments }: { segments: CellSegment[] }) {
  return segments.map((seg, i) =>
    seg.href ? (
      <a
        // A segment's position is its identity: the text is not unique and never reorders.
        // biome-ignore lint/suspicious/noArrayIndexKey: see above
        key={i}
        href={seg.href}
        target="_blank"
        rel="noopener noreferrer"
        className="underline underline-offset-2 hover:text-foreground"
        // A link belongs to the page it opens, not to the row that holds it.
        onClick={(e) => e.stopPropagation()}
      >
        {seg.text}
      </a>
    ) : (
      // biome-ignore lint/suspicious/noArrayIndexKey: see above
      <span key={i}>{seg.text}</span>
    ),
  )
}

// The draggable edge of a column header. Dragging previews the width live and saves it on
// release; a double click returns the column to its content width; the arrow keys nudge it.
function ResizeHandle({
  column,
  onPreview,
  onResize,
}: {
  column: ColumnSpec
  onPreview: (width: number | null) => void
  onResize: (field: string, width: ColumnWidth) => void
}) {
  const stop = useRef<(() => void) | null>(null)
  useEffect(() => () => stop.current?.(), [])

  const current = (el: HTMLElement) =>
    typeof column.width === "number"
      ? column.width
      : (el.parentElement?.getBoundingClientRect().width ?? 0)
  const clamp = (w: number) => Math.max(MIN_COLUMN_WIDTH, w)

  const begin = (e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const x0 = e.clientX
    const w0 = current(e.currentTarget)
    const move = (ev: globalThis.MouseEvent) => onPreview(clamp(w0 + ev.clientX - x0))
    const end = (ev: globalThis.MouseEvent) => {
      stop.current?.()
      onPreview(null)
      if (ev.clientX !== x0) onResize(column.field, clamp(w0 + ev.clientX - x0))
    }
    stop.current = () => {
      window.removeEventListener("mousemove", move)
      window.removeEventListener("mouseup", end)
      stop.current = null
    }
    window.addEventListener("mousemove", move)
    window.addEventListener("mouseup", end)
  }

  const nudge = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.key === "ArrowRight" ? KEY_STEP : e.key === "ArrowLeft" ? -KEY_STEP : 0
    if (!step) return
    e.preventDefault()
    onResize(column.field, clamp(current(e.currentTarget) + step))
  }

  return (
    // The ARIA window-splitter pattern: a focusable separator that moves an edge. An <hr>
    // cannot take focus or a drag, so there is no native element for this.
    // biome-ignore lint/a11y/useSemanticElements: see above
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={`Resize column ${column.field}`}
      aria-valuenow={typeof column.width === "number" ? column.width : undefined}
      aria-valuemin={MIN_COLUMN_WIDTH}
      tabIndex={0}
      className="dash-no-drag absolute top-0 right-0 h-full w-1.5 cursor-col-resize touch-none select-none hover:bg-border focus-visible:bg-ring"
      onMouseDown={begin}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={() => onResize(column.field, "fit")}
      onKeyDown={nudge}
    />
  )
}

function RowsTable({
  widget,
  data,
  onRowClick,
  onColumnWidth,
}: {
  widget: Widget
  data?: WidgetData
  onRowClick?: (row: Row) => void
  onColumnWidth?: (field: string, width: ColumnWidth) => void
}) {
  const rowKey = useRowKeys()
  // The width a column is being dragged to, shown before it is saved.
  const [dragging, setDragging] = useState<{ field: string; width: number } | null>(null)
  const rows = asRows(data?.value)
  if (rows.length === 0) return <div className="text-sm text-text-tertiary">No rows.</div>
  const viz = widget.viz ?? {}
  const columns = parseColumns(viz.columns, rows)
  const pageSize = typeof viz.pageSize === "number" ? viz.pageSize : 50
  const widthOf = (c: ColumnSpec) => (dragging?.field === c.field ? dragging.width : c.width)
  return (
    // Fills the tile rather than sizing to the rows: a taller tile shows more of them,
    // and a shorter one scrolls, instead of leaving the extra height as padding.
    <div className="h-full overflow-auto">
      <Table className="text-sm">
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            {columns.map((c) => (
              <TableHead
                key={c.field}
                className={onColumnWidth ? "relative px-2 py-1 pr-3" : "px-2 py-1"}
                style={widthStyle(widthOf(c))}
              >
                {c.label ?? c.field}
                {onColumnWidth ? (
                  <ResizeHandle
                    column={{ ...c, width: widthOf(c) }}
                    onPreview={(width) =>
                      setDragging(width === null ? null : { field: c.field, width })
                    }
                    onResize={onColumnWidth}
                  />
                ) : null}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody className="[&_tr]:border-border/50 [&_tr:last-child]:border-b">
          {rows.slice(0, pageSize).map((row) => (
            <TableRow
              key={rowKey(row)}
              className={onRowClick ? "cursor-pointer hover:bg-accent" : undefined}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              {columns.map((c) => {
                const width = widthOf(c)
                // A fit column never wraps; a fixed one wraps unless it says not to, and
                // then clips rather than pushing its neighbours.
                const nowrap = width === "fit" || c.wrap === false
                return (
                  <TableCell
                    key={c.field}
                    className={
                      nowrap
                        ? `px-2 py-1 whitespace-nowrap${typeof width === "number" ? " overflow-hidden text-ellipsis" : ""}`
                        : "px-2 py-1"
                    }
                    style={widthStyle(width)}
                  >
                    <Cell segments={cellSegments(String(row[c.field] ?? ""))} />
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function TextBody({ widget, variables }: { widget: Widget; variables: Record<string, unknown> }) {
  // Interpolate $name references with the current variable values.
  const content = (widget.content ?? "").replace(/\$([a-z][a-z0-9_]*)/g, (m, name) =>
    name in variables ? String(variables[name]) : m,
  )
  return <NotebookMarkdown>{content}</NotebookMarkdown>
}

export function DashboardWidget({
  widget,
  data,
  variables,
  onCrossFilter,
  onDrillThrough,
  onEdit,
  onDelete,
  onColumnWidth,
}: {
  widget: Widget
  data?: WidgetData
  variables: Record<string, unknown>
  onCrossFilter?: (emit: Record<string, unknown>) => void
  onDrillThrough?: () => void
  onEdit?: () => void
  onDelete?: () => void
  /** Saves a column's new width. Without it a table is read-only and shows no handles. */
  onColumnWidth?: (field: string, width: ColumnWidth) => void
}) {
  const crossFilter = widget.interactions?.crossFilter
  // The rows a data tile already holds, offered as a CSV from the tile menu.
  const csvRows = widget.type !== "text" && !data?.error ? asRows(data?.value) : []
  const drillThrough = widget.interactions?.drillThrough

  // A clicked table row emits its fields into the mapped variables (cross-filter),
  // per interactions.crossFilter.emit: { variable -> "datum.field" }.
  const onRowClick =
    crossFilter && onCrossFilter
      ? (row: Row) => {
          const emit: Record<string, unknown> = {}
          for (const [name, expr] of Object.entries(crossFilter.emit)) {
            emit[name] = row[expr.replace(/^datum\./, "")]
          }
          onCrossFilter(emit)
        }
      : undefined

  const body = () => {
    // Text usually carries its own content and never waits on data. Naming a
    // derivation instead makes it a tile like any other, resolved below.
    if (widget.type === "text" && !widget.bind)
      return <TextBody widget={widget} variables={variables} />
    if (data?.error)
      return (
        <div className="flex items-start gap-2 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{data.error}</span>
        </div>
      )
    // Until the derivation resolves, show a loading note rather than rendering a
    // chart against empty data (which makes Vega warn about an infinite extent /
    // empty log-scale domain in the console).
    if (!data) return <CenterNote>Loading…</CenterNote>
    const rows = asRows(data.value)
    switch (widget.type) {
      // A derivation returning markdown: the governed way to put a bespoke visual on
      // a dashboard, so the prose carries live figures instead of copied ones.
      case "text":
        return (
          <NotebookMarkdown>{typeof data.value === "string" ? data.value : ""}</NotebookMarkdown>
        )
      case "metric":
        return <MetricBody widget={widget} data={data} />
      case "table":
        return rows.length ? (
          <RowsTable
            widget={widget}
            data={data}
            onRowClick={onRowClick}
            onColumnWidth={onColumnWidth}
          />
        ) : (
          <CenterNote>No rows.</CenterNote>
        )
      case "chart":
      case "map":
        return rows.length ? (
          <VizView viz={{ spec: widget.viz ?? {}, rows }} height="container" bare />
        ) : (
          <CenterNote>No data for this selection.</CenterNote>
        )
      default:
        return null
    }
  }

  return (
    <div className="group flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card">
      {/* Header doubles as the drag handle (react-grid-layout draggableHandle). */}
      <div className="dash-drag-handle flex cursor-grab items-center justify-between gap-2 px-3 pt-2.5 pb-1 active:cursor-grabbing">
        <div className="truncate text-sm font-medium" title={widget.title}>
          {widget.title ?? ""}
        </div>
        <div className="flex items-center gap-1.5">
          {drillThrough && onDrillThrough ? (
            <Button
              variant="link"
              className="dash-no-drag h-auto p-0 text-xs font-normal text-text-tertiary underline-offset-2 hover:text-foreground"
              onClick={onDrillThrough}
            >
              Details →
            </Button>
          ) : null}
          {csvRows.length > 0 || onEdit || onDelete ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Tile actions: ${widget.title ?? widget.id}`}
                  className="dash-no-drag text-text-tertiary opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <MoreHorizontal className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="dash-no-drag">
                {csvRows.length > 0 ? (
                  <DropdownMenuItem
                    onSelect={() =>
                      downloadText(
                        // Named as ExplorePage names its downloads, after what the tile shows.
                        `${(widget.title ?? widget.id).trim().replace(/\s+/g, "_").toLowerCase() || widget.id}.csv`,
                        toCsv(columnsOf(widget, csvRows), csvRows),
                        "text/csv",
                      )
                    }
                  >
                    Download CSV
                  </DropdownMenuItem>
                ) : null}
                {onEdit ? <DropdownMenuItem onSelect={onEdit}>Edit…</DropdownMenuItem> : null}
                {onDelete ? (
                  <DropdownMenuItem variant="destructive" onSelect={onDelete}>
                    Delete
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
          <GripVertical className="h-3.5 w-3.5 shrink-0 text-text-tertiary/30 opacity-0 transition group-hover:opacity-100" />
        </div>
      </div>
      <div className="min-h-0 flex-1 px-3 pb-3">{body()}</div>
    </div>
  )
}
