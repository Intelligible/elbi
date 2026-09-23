// Renders a single dashboard widget from its spec and resolved data. Chart and map
// widgets reuse the shared VizView (which routes a lat/long spec to the deck.gl map);
// metric, table, and text are light renderers. A widget whose derivation failed shows
// its error in place rather than blanking.

import { AlertCircle, Download, GripVertical } from "lucide-react"
import type { ReactNode } from "react"

import { VizView } from "@/components/viz/VizView"
import { useRowKeys } from "@/hooks/useRowKeys"
import type { Widget, WidgetData } from "@/lib/dashboards"
import { downloadText, toCsv } from "@/lib/download"
import { EMPTY } from "@/lib/utils"

type Row = Record<string, string | number>

function asRows(value: unknown): Row[] {
  return Array.isArray(value) ? (value as Row[]) : []
}

function CenterNote({ children }: { children: ReactNode }) {
  return (
    <div className="grid h-full place-items-center text-xs text-muted-foreground">{children}</div>
  )
}

function formatValue(value: unknown, format?: string): string {
  if (value === null || value === undefined) return EMPTY
  const num = typeof value === "number" ? value : Number(value)
  if (Number.isNaN(num)) return String(value)
  if (format === "currency")
    return num.toLocaleString(undefined, {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    })
  if (format === "percent")
    return num.toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 })
  // Integers read cleanest with no decimals; a fractional KPI (a mean) keeps one.
  return num.toLocaleString(undefined, {
    maximumFractionDigits: Number.isInteger(num) ? 0 : 1,
  })
}

function aggregate(rows: Row[], field: string, agg: string): number | undefined {
  const nums = rows.map((r) => Number(r[field])).filter((n) => Number.isFinite(n))
  if (nums.length === 0) return undefined
  switch (agg) {
    case "sum":
      return nums.reduce((a, b) => a + b, 0)
    case "mean":
      return nums.reduce((a, b) => a + b, 0) / nums.length
    case "min":
      return Math.min(...nums)
    case "max":
      return Math.max(...nums)
    default:
      return undefined
  }
}

function MetricBody({ widget, data }: { widget: Widget; data?: WidgetData }) {
  const viz = widget.viz ?? {}
  const field = typeof viz.field === "string" ? viz.field : undefined
  const agg = typeof viz.agg === "string" ? viz.agg : undefined
  const rows = asRows(data?.value)
  // A KPI is usually an aggregate over the bound derivation's rows: a count of rows,
  // or a mean/sum/min/max of a column. Falling back to the first row's value (or a
  // scalar result) covers a derivation that already returns a single figure.
  let raw: unknown
  if (agg === "count") {
    raw = rows.length
  } else if (agg && field) {
    raw = aggregate(rows, field, agg)
  } else if (field && rows.length > 0) {
    raw = rows[0][field]
  } else if (typeof data?.value === "number" || typeof data?.value === "string") {
    raw = data?.value
  }
  if (raw === null || raw === undefined) {
    // Say why it's blank: a metric needs a `viz.field` naming a column the bound derivation
    // returns, or a scalar result, so a wrong field is obvious, not a silent dash.
    const hint = field
      ? `no column “${field}” in the result`
      : "set viz.field to a column of the bound derivation"
    return (
      <div className="flex h-full flex-col justify-center">
        <div className="text-3xl font-semibold text-muted-foreground/40">{EMPTY}</div>
        <div className="mt-1 text-xs text-muted-foreground">{hint}</div>
      </div>
    )
  }
  return (
    <div className="flex h-full flex-col justify-center">
      <div className="text-3xl font-semibold tabular-nums">
        {formatValue(raw, typeof viz.format === "string" ? viz.format : undefined)}
      </div>
    </div>
  )
}

function TableBody({
  widget,
  data,
  onRowClick,
}: {
  widget: Widget
  data?: WidgetData
  onRowClick?: (row: Row) => void
}) {
  const rowKey = useRowKeys()
  const rows = asRows(data?.value)
  if (rows.length === 0) return <div className="text-sm text-muted-foreground">No rows.</div>
  const viz = widget.viz ?? {}
  const columns = Array.isArray(viz.columns) ? (viz.columns as string[]) : Object.keys(rows[0])
  const pageSize = typeof viz.pageSize === "number" ? viz.pageSize : 50
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-background">
          <tr className="border-b border-border text-left text-muted-foreground">
            {columns.map((c) => (
              <th key={c} className="px-2 py-1 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, pageSize).map((row) => (
            <tr
              key={rowKey(row)}
              className={
                onRowClick
                  ? "cursor-pointer border-b border-border/50 hover:bg-accent"
                  : "border-b border-border/50"
              }
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              {columns.map((c) => (
                <td key={c} className="px-2 py-1 tabular-nums">
                  {String(row[c] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TextBody({ widget, variables }: { widget: Widget; variables: Record<string, unknown> }) {
  // Interpolate $name references with the current variable values.
  const content = (widget.content ?? "").replace(/\$([a-z][a-z0-9_]*)/g, (m, name) =>
    name in variables ? String(variables[name]) : m,
  )
  return <div className="whitespace-pre-wrap text-sm leading-relaxed">{content}</div>
}

export function DashboardWidget({
  widget,
  data,
  variables,
  onCrossFilter,
  onDrillThrough,
}: {
  widget: Widget
  data?: WidgetData
  variables: Record<string, unknown>
  onCrossFilter?: (emit: Record<string, unknown>) => void
  onDrillThrough?: () => void
}) {
  const crossFilter = widget.interactions?.crossFilter
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
    // Text carries its own content and never waits on data.
    if (widget.type === "text") return <TextBody widget={widget} variables={variables} />
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
      case "metric":
        return <MetricBody widget={widget} data={data} />
      case "table":
        return rows.length ? (
          <TableBody widget={widget} data={data} onRowClick={onRowClick} />
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
    <div className="group flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      {/* Header doubles as the drag handle (react-grid-layout draggableHandle). */}
      <div className="dash-drag-handle flex cursor-grab items-center justify-between gap-2 px-3 pt-2.5 pb-1 active:cursor-grabbing">
        <div className="truncate text-sm font-medium" title={widget.title}>
          {widget.title ?? ""}
        </div>
        <div className="flex items-center gap-1.5">
          {widget.type !== "text" && !data?.error && asRows(data?.value).length > 0 ? (
            <button
              type="button"
              className="dash-no-drag text-muted-foreground opacity-0 transition group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100"
              title="Download this widget's data as CSV"
              aria-label={`Download ${widget.title ?? widget.id} data as CSV`}
              onClick={() => {
                const rows = asRows(data?.value)
                downloadText(`${widget.id}.csv`, toCsv(Object.keys(rows[0]), rows), "text/csv")
              }}
            >
              <Download className="h-3.5 w-3.5" />
            </button>
          ) : null}
          {drillThrough && onDrillThrough ? (
            <button
              type="button"
              className="dash-no-drag text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
              onClick={onDrillThrough}
            >
              Details →
            </button>
          ) : null}
          <GripVertical className="h-3.5 w-3.5 shrink-0 text-muted-foreground/30 opacity-0 transition group-hover:opacity-100" />
        </div>
      </div>
      <div className="min-h-0 flex-1 px-3 pb-3">{body()}</div>
    </div>
  )
}
