import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import type { GridPos, Widget } from "@/lib/dashboards"
import { listMetrics, type Metric } from "@/lib/metrics"
import { JsonEditor } from "./JsonEditor"
import { ProvenanceTab } from "./TileProvenance"

/** What the editor can change. A grammar-of-graphics `viz` stays with the spec editor. */
export interface TilePatch {
  title?: string
  content?: string
  derivation?: string
  /** The shared metric a metric tile shows. */
  metric?: string
  gridPos?: GridPos
  /** The whole widget, from the JSON tab: everything the fields cannot reach. */
  widget?: Record<string, unknown>
}

/**
 * Edit one tile: what it says, what it reads, and how big it is.
 *
 * The fields here are the ones someone reaches for while laying a page out. A
 * chart's `viz` is a grammar-of-graphics spec, so it lives on the JSON tab rather than
 * as a dialog of one input per key.
 */
export function TileEditor({
  widget,
  catalog,
  columns,
  dark,
  schema,
  onCancel,
  onSave,
}: {
  widget: Widget | null
  catalog: string[]
  columns: number
  dark: boolean
  schema?: Record<string, unknown> | null
  onCancel: () => void
  /** Rejects with the server's reason when the save is refused. */
  onSave: (id: string, patch: TilePatch) => Promise<void>
}) {
  const [title, setTitle] = useState("")
  const [content, setContent] = useState("")
  const [derivation, setDerivation] = useState("")
  const [metric, setMetric] = useState("")
  const [width, setWidth] = useState(1)
  const [height, setHeight] = useState(1)
  const [tab, setTab] = useState<"fields" | "json" | "lineage">("fields")
  const [draft, setDraft] = useState("")
  const [error, setError] = useState<string | null>(null)

  // Reset every field when a different tile is opened, so the dialog never shows the
  // previous tile's values.
  useEffect(() => {
    setTitle(widget?.title ?? "")
    setContent(widget?.content ?? "")
    setDerivation(widget?.bind?.derivation ?? "")
    setMetric(widget?.bind?.metric ?? "")
    setWidth(widget?.gridPos.w ?? 1)
    setHeight(widget?.gridPos.h ?? 1)
    setTab("fields")
    setDraft("")
    setError(null)
  }, [widget])

  const [metrics, setMetrics] = useState<Metric[]>([])
  // Kept apart from an empty list: a picker with nothing to offer should say why.
  const [metricsState, setMetricsState] = useState<"loading" | "ready" | "failed">("loading")
  const editsMetricTile = widget?.type === "metric"
  useEffect(() => {
    let live = true
    if (!editsMetricTile) return
    setMetricsState("loading")
    void listMetrics()
      .then((all) => {
        if (!live) return
        setMetrics(all)
        setMetricsState("ready")
      })
      .catch(() => live && setMetricsState("failed"))
    return () => {
      live = false
    }
  }, [editsMetricTile])

  if (widget === null) return null

  const bound = widget.bind !== undefined && widget.bind !== null
  const editsContent = widget.type === "text" && !bound
  // A metric tile shows a shared metric, so it picks a metric rather than a derivation.
  const editsMetric = widget.type === "metric"
  // A chart or table may bind a metric too; that binding is edited on the JSON tab.
  const editsDerivation = bound && !editsMetric && !widget.bind?.metric
  const unknown = editsDerivation && derivation.length > 0 && !catalog.includes(derivation)
  const metricNames = metrics.map((m) => m.name)
  const boundMetric = metrics.find((m) => m.name === metric)
  const unknownMetric = metric.length > 0 && metricsState === "ready" && boundMetric === undefined
  // A tile from before metric tiles bound shared metrics aggregates a derivation itself.
  const privateFigure = editsMetric && !metric && Boolean(widget.bind?.derivation)
  // The trail runs through the metric once one is bound. Only a simple metric names a
  // derivation to trace; every other case gets a note rather than a derivation's trail.
  const lineageOf = editsMetric && metric ? (boundMetric?.source ?? "") : derivation
  const metricTrailNote = (): string => {
    if (metricsState === "loading") return "Loading the metric…"
    if (metricsState === "failed") return "Could not load the metric, so its trail is unknown."
    if (boundMetric === undefined)
      return `No metric named “${metric}” exists, so there is nothing to trace.`
    return `${metric} is a ${boundMetric.type} metric built from other metrics rather than one derivation; the Metrics page lists them.`
  }

  // The widget the fields currently describe. Switching to JSON shows this rather than
  // what the tile was opened with, so an edit made in the fields is not silently lost.
  const asWidget = (): Record<string, unknown> => {
    const next: Record<string, unknown> = { ...(widget as unknown as Record<string, unknown>) }
    if (title.trim()) next.title = title.trim()
    else delete next.title
    next.gridPos = { ...widget.gridPos, w: clamp(width, 1, columns), h: clamp(height, 1, 80) }
    if (editsContent) next.content = content
    if (editsDerivation) next.bind = { ...(widget.bind ?? {}), derivation }
    if (editsMetric && metric) next.bind = metricBind(widget, metric)
    return next
  }

  const showJson = () => {
    setDraft(JSON.stringify(asWidget(), null, 2))
    setError(null)
    setTab("json")
  }

  const showFields = () => {
    // The draft is only live on the JSON tab. From anywhere else the fields are already
    // current, and parsing a draft written earlier, or for the previous tile, would
    // overwrite them.
    if (tab !== "json") {
      setError(null)
      setTab("fields")
      return
    }
    // Parse first: dropping back to the fields would otherwise discard a JSON edit.
    try {
      const parsed = JSON.parse(draft) as Widget
      setTitle(parsed.title ?? "")
      setContent(parsed.content ?? "")
      setDerivation(parsed.bind?.derivation ?? "")
      setMetric(parsed.bind?.metric ?? "")
      setWidth(parsed.gridPos?.w ?? width)
      setHeight(parsed.gridPos?.h ?? height)
      setError(null)
      setTab("fields")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  // A refused save keeps the dialog open with the server's reason, rather than closing
  // on an edit that never landed.
  const submit = async (patch: TilePatch) => {
    try {
      await onSave(widget.id, patch)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const saveJson = () => {
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(draft) as Record<string, unknown>
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return
    }
    // The id addresses the tile in the layout and in this dialog; changing it here
    // would orphan both, so it is pinned rather than trusted from the text.
    void submit({ widget: { ...parsed, id: widget.id } })
  }

  const save = () => {
    const patch: TilePatch = {
      title,
      gridPos: {
        ...widget.gridPos,
        w: clamp(width, 1, columns),
        h: clamp(height, 1, 40),
      },
    }
    if (editsContent) patch.content = content
    if (editsDerivation) patch.derivation = derivation
    if (editsMetric && metric) patch.metric = metric
    void submit(patch)
  }

  return (
    <Dialog open onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[85vh] max-w-xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit tile</DialogTitle>
          <p className="text-xs text-text-tertiary">
            <code>{widget.id}</code> · {widget.type}
          </p>
        </DialogHeader>

        <div className="flex gap-1 rounded-md bg-surface-secondary p-1">
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={tab === "fields"}
            className={tabClass(tab === "fields")}
            onClick={showFields}
          >
            Fields
          </Button>
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={tab === "json"}
            className={tabClass(tab === "json")}
            onClick={showJson}
          >
            JSON
          </Button>
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={tab === "lineage"}
            className={tabClass(tab === "lineage")}
            onClick={() => setTab("lineage")}
          >
            Lineage
          </Button>
        </div>

        {tab === "lineage" ? (
          <>
            {editsMetric && metric && !lineageOf ? (
              <p className="text-sm text-text-tertiary">{metricTrailNote()}</p>
            ) : (
              <ProvenanceTab derivation={lineageOf} />
            )}
            <DialogFooter>
              <Button variant="ghost" onClick={onCancel}>
                Close
              </Button>
            </DialogFooter>
          </>
        ) : tab === "json" ? (
          <>
            <Field
              label="This tile's config"
              hint="The widget exactly as the dashboard spec stores it. Everything the fields do not reach — a chart's viz, params, interactions — is editable here."
            >
              <JsonEditor
                value={draft}
                label="This tile's config"
                dark={dark}
                schema={schema}
                definition="widget"
                onChange={(next) => {
                  setDraft(next)
                  setError(null)
                }}
              />
            </Field>
            {error ? (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            ) : null}
            <DialogFooter>
              <Button variant="ghost" onClick={onCancel}>
                Cancel
              </Button>
              <Button onClick={saveJson}>Save</Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <Field label="Title" htmlFor="tile-title">
              <Input
                id="tile-title"
                value={title}
                placeholder="Untitled tile"
                onChange={(e) => setTitle(e.target.value)}
              />
            </Field>

            {editsContent ? (
              <Field
                label="Content"
                htmlFor="tile-content"
                hint="Markdown. A figure typed here is a copy that goes stale — bind a derivation that returns markdown to keep it live."
              >
                <Textarea
                  id="tile-content"
                  className="h-48 font-mono text-xs"
                  value={content}
                  spellCheck={false}
                  onChange={(e) => setContent(e.target.value)}
                />
              </Field>
            ) : null}

            {editsDerivation ? (
              <Field label="Derivation" htmlFor="tile-derivation">
                <Picker
                  id="tile-derivation"
                  value={derivation}
                  options={named(withCurrent(catalog, derivation))}
                  placeholder="Pick a derivation"
                  onChange={setDerivation}
                />
                {unknown ? (
                  <p className="text-xs text-destructive">
                    Nothing named “{derivation}” is available to bind. It is kept so saving does not
                    silently drop it, but the tile renders an error until it exists.
                  </p>
                ) : null}
              </Field>
            ) : null}

            {editsMetric ? (
              <Field
                label="Metric"
                htmlFor="tile-metric"
                hint="The tile shows this metric's value, in the metric's own format. Metrics are defined on the Metrics page, so the same figure reads the same everywhere."
              >
                <Picker
                  id="tile-metric"
                  value={metric}
                  options={named(withCurrent(metricNames, metric))}
                  placeholder="Pick a metric"
                  onChange={setMetric}
                />
                {privateFigure ? (
                  <p className="text-xs text-destructive">
                    This tile computes its own figure from {widget.bind?.derivation}. Pick the
                    shared metric it should show; define one on the Metrics page if none fits.
                  </p>
                ) : null}
                {unknownMetric ? (
                  <p className="text-xs text-destructive">
                    No metric named “{metric}” exists. It is kept so saving does not silently drop
                    it, but the tile renders an error until it exists.
                  </p>
                ) : null}
                {metricsState === "failed" ? (
                  <p className="text-xs text-destructive">
                    Could not load the metrics, so only this tile's current one is offered.
                  </p>
                ) : null}
              </Field>
            ) : null}

            {widget.type !== "metric" && widget.type !== "text" ? (
              <p className="text-xs text-text-tertiary">
                This tile's <code>viz</code> is a grammar-of-graphics spec. The JSON tab has it.
              </p>
            ) : null}

            <Field
              label="Size"
              hint={`Width is in grid columns, out of ${columns}; height is in rows. Dragging the tile’s bottom-right corner does the same thing.`}
            >
              <div className="flex items-center gap-2">
                <Input
                  aria-label="Width"
                  type="number"
                  min={1}
                  max={columns}
                  value={width}
                  className="w-24"
                  onChange={(e) => setWidth(Number(e.target.value))}
                />
                <span className="text-sm text-text-tertiary">×</span>
                <Input
                  aria-label="Height"
                  type="number"
                  min={1}
                  max={40}
                  value={height}
                  className="w-24"
                  onChange={(e) => setHeight(Number(e.target.value))}
                />
              </div>
            </Field>

            {error ? (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            ) : null}
            <DialogFooter>
              <Button variant="ghost" onClick={onCancel}>
                Cancel
              </Button>
              <Button onClick={save}>Save</Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string
  htmlFor?: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      {htmlFor ? (
        <label className="block text-sm font-medium" htmlFor={htmlFor}>
          {label}
        </label>
      ) : (
        <p className="block text-sm font-medium">{label}</p>
      )}
      {children}
      {hint ? <p className="text-xs text-text-tertiary">{hint}</p> : null}
    </div>
  )
}

/**
 * The options, with the current value included even when it is not among them.
 *
 * A Select cannot show a value it has no option for: it would render blank, and saving
 * would write that blank over something nobody meant to clear.
 */
function withCurrent(options: string[], current: string): string[] {
  if (!current || options.includes(current)) return options
  return [current, ...options]
}

/** A metric tile's binding: the metric, keeping any filters, and nothing a derivation used. */
export function metricBind(widget: Widget, metric: string): NonNullable<Widget["bind"]> {
  const filters = widget.bind?.filters
  return filters?.length ? { metric, filters } : { metric }
}

/**
 * A select that behaves like a dropdown rather than a native picker.
 *
 * Radix defaults to `item-aligned`, which positions the list so the selected item
 * covers the trigger — the list jumps over the fields above it, and where it lands
 * depends on which item is selected. `popper` drops it below, left-aligned, at the
 * trigger's width, which is what the surrounding inputs look like.
 */
function Picker({
  id,
  value,
  options,
  placeholder,
  onChange,
}: {
  id: string
  value: string
  options: { value: string; label: string }[]
  placeholder: string
  onChange: (next: string) => void
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger id={id} className="w-full">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent position="popper" align="start" sideOffset={4} className="max-h-72">
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            <span className="truncate">{option.label}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** Plain names as options: the value is the label. */
function named(values: string[]): { value: string; label: string }[] {
  return values.map((value) => ({ value, label: value }))
}

function tabClass(active: boolean): string {
  return active
    ? "flex-1 bg-card shadow-sm hover:bg-card"
    : "flex-1 font-normal text-text-tertiary hover:text-foreground"
}

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low
  return Math.min(high, Math.max(low, Math.round(value)))
}
