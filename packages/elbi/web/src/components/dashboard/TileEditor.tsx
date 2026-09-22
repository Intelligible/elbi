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
import { JsonEditor } from "./JsonEditor"
import { TileProvenance } from "./TileProvenance"

/** What the editor can change. A grammar-of-graphics `viz` stays with the spec editor. */
export interface TilePatch {
  title?: string
  content?: string
  derivation?: string
  viz?: Record<string, unknown>
  gridPos?: GridPos
  /** The whole widget, from the JSON tab: everything the fields cannot reach. */
  widget?: Record<string, unknown>
}

/** A metric reads one column; how it reduces the rows to a single figure. */
const AGGREGATES = [
  { value: "first", label: "First row's value" },
  { value: "sum", label: "Sum" },
  { value: "mean", label: "Mean" },
  { value: "min", label: "Minimum" },
  { value: "max", label: "Maximum" },
  { value: "count", label: "Count of rows" },
]

const FORMATS = [
  { value: "plain", label: "Plain number" },
  { value: "currency", label: "Currency (USD)" },
  { value: "percent", label: "Percent" },
]

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
  onCancel,
  onSave,
}: {
  widget: Widget | null
  catalog: string[]
  columns: number
  dark: boolean
  onCancel: () => void
  onSave: (id: string, patch: TilePatch) => void
}) {
  const [title, setTitle] = useState("")
  const [content, setContent] = useState("")
  const [derivation, setDerivation] = useState("")
  const [field, setField] = useState("")
  const [agg, setAgg] = useState("first")
  const [format, setFormat] = useState("plain")
  const [width, setWidth] = useState(1)
  const [height, setHeight] = useState(1)
  const [tab, setTab] = useState<"fields" | "json">("fields")
  const [draft, setDraft] = useState("")
  const [jsonError, setJsonError] = useState<string | null>(null)

  // Reset every field when a different tile is opened, so the dialog never shows the
  // previous tile's values.
  useEffect(() => {
    const viz = (widget?.viz ?? {}) as Record<string, unknown>
    setTitle(widget?.title ?? "")
    setContent(widget?.content ?? "")
    setDerivation(widget?.bind?.derivation ?? "")
    setField(typeof viz.field === "string" ? viz.field : "")
    setAgg(typeof viz.agg === "string" ? viz.agg : "first")
    setFormat(typeof viz.format === "string" ? viz.format : "plain")
    setWidth(widget?.gridPos.w ?? 1)
    setHeight(widget?.gridPos.h ?? 1)
    setTab("fields")
    setJsonError(null)
  }, [widget])

  if (widget === null) return null

  const bound = widget.bind !== undefined && widget.bind !== null
  const editsContent = widget.type === "text" && !bound
  const editsMetric = widget.type === "metric"
  // A metric bound to a column the derivation does not return renders a dash, so an
  // unknown name is worth saying before it is saved rather than after the tile blanks.
  const unknown = bound && derivation.length > 0 && !catalog.includes(derivation)

  // The widget the fields currently describe. Switching to JSON shows this rather than
  // what the tile was opened with, so an edit made in the fields is not silently lost.
  const asWidget = (): Record<string, unknown> => {
    const next: Record<string, unknown> = { ...(widget as unknown as Record<string, unknown>) }
    if (title.trim()) next.title = title.trim()
    else delete next.title
    next.gridPos = { ...widget.gridPos, w: clamp(width, 1, columns), h: clamp(height, 1, 80) }
    if (editsContent) next.content = content
    if (bound) next.bind = { ...(widget.bind ?? {}), derivation }
    if (editsMetric) next.viz = vizFromFields()
    return next
  }

  const vizFromFields = (): Record<string, unknown> => {
    const viz: Record<string, unknown> = { ...(widget.viz ?? {}), field }
    if (agg === "first") delete viz.agg
    else viz.agg = agg
    if (format === "plain") delete viz.format
    else viz.format = format
    return viz
  }

  const showJson = () => {
    setDraft(JSON.stringify(asWidget(), null, 2))
    setJsonError(null)
    setTab("json")
  }

  const showFields = () => {
    // Parse first: dropping back to the fields would otherwise discard a JSON edit.
    try {
      const parsed = JSON.parse(draft) as Widget
      const viz = (parsed.viz ?? {}) as Record<string, unknown>
      setTitle(parsed.title ?? "")
      setContent(parsed.content ?? "")
      setDerivation(parsed.bind?.derivation ?? "")
      setField(typeof viz.field === "string" ? viz.field : "")
      setAgg(typeof viz.agg === "string" ? viz.agg : "first")
      setFormat(typeof viz.format === "string" ? viz.format : "plain")
      setWidth(parsed.gridPos?.w ?? width)
      setHeight(parsed.gridPos?.h ?? height)
      setJsonError(null)
      setTab("fields")
    } catch (err) {
      setJsonError(err instanceof Error ? err.message : String(err))
    }
  }

  const saveJson = () => {
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(draft) as Record<string, unknown>
    } catch (err) {
      setJsonError(err instanceof Error ? err.message : String(err))
      return
    }
    // The id addresses the tile in the layout and in this dialog; changing it here
    // would orphan both, so it is pinned rather than trusted from the text.
    onSave(widget.id, { widget: { ...parsed, id: widget.id } })
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
    if (bound) patch.derivation = derivation
    // "first" and "plain" are the absence of an aggregate and of a format, so they are
    // dropped rather than written as values the renderer would have to know.
    if (editsMetric) patch.viz = vizFromFields()
    onSave(widget.id, patch)
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
          <button
            type="button"
            aria-pressed={tab === "fields"}
            className={tabClass(tab === "fields")}
            onClick={showFields}
          >
            Fields
          </button>
          <button
            type="button"
            aria-pressed={tab === "json"}
            className={tabClass(tab === "json")}
            onClick={showJson}
          >
            JSON
          </button>
        </div>

        {bound ? <TileProvenance derivation={derivation} /> : null}

        {tab === "json" ? (
          <>
            <Field
              label="This tile's config"
              hint="The widget exactly as the dashboard spec stores it. Everything the fields do not reach — a chart's viz, params, interactions — is editable here."
            >
              <JsonEditor
                value={draft}
                label="This tile's config"
                dark={dark}
                onChange={(next) => {
                  setDraft(next)
                  setJsonError(null)
                }}
              />
            </Field>
            {jsonError ? (
              <p role="alert" className="text-xs text-destructive">
                {jsonError}
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

            {bound ? (
              <Field label="Derivation" htmlFor="tile-derivation">
                <Input
                  id="tile-derivation"
                  list="tile-derivation-options"
                  value={derivation}
                  onChange={(e) => setDerivation(e.target.value)}
                />
                <datalist id="tile-derivation-options">
                  {catalog.map((name) => (
                    <option key={name} value={name} />
                  ))}
                </datalist>
                {unknown ? (
                  <p className="text-xs text-destructive">
                    Nothing named “{derivation}” is available to bind. The tile will render an error
                    until it exists.
                  </p>
                ) : null}
              </Field>
            ) : null}

            {editsMetric ? (
              <>
                <Field
                  label="Column"
                  htmlFor="tile-field"
                  hint="A column of the bound derivation, spelled the way the derivation spells it."
                >
                  <Input
                    id="tile-field"
                    value={field}
                    placeholder="mrr_usd"
                    onChange={(e) => setField(e.target.value)}
                  />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Aggregate" htmlFor="tile-agg">
                    <Select value={agg} onValueChange={setAgg}>
                      <SelectTrigger id="tile-agg">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {AGGREGATES.map((a) => (
                          <SelectItem key={a.value} value={a.value}>
                            {a.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <Field label="Format" htmlFor="tile-format">
                    <Select value={format} onValueChange={setFormat}>
                      <SelectTrigger id="tile-format">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {FORMATS.map((f) => (
                          <SelectItem key={f.value} value={f.value}>
                            {f.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                </div>
                {format === "percent" ? (
                  <p className="text-xs text-text-tertiary">
                    Percent multiplies by 100: a column already holding 69.5 renders as 6,950%. Use
                    a plain number for a column that is already a percentage.
                  </p>
                ) : null}
              </>
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

function tabClass(active: boolean): string {
  return active
    ? "flex-1 rounded px-3 py-1 text-sm font-medium bg-card shadow-sm"
    : "flex-1 rounded px-3 py-1 text-sm text-text-tertiary hover:text-foreground"
}

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low
  return Math.min(high, Math.max(low, Math.round(value)))
}
