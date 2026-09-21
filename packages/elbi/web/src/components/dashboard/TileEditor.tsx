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

/** What the editor can change. A grammar-of-graphics `viz` stays with the spec editor. */
export interface TilePatch {
  title?: string
  content?: string
  derivation?: string
  viz?: Record<string, unknown>
  gridPos?: GridPos
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
 * chart's `viz` is a grammar-of-graphics spec and stays with the whole-spec editor,
 * which is a better surface for it than a dialog of one input per key.
 */
export function TileEditor({
  widget,
  catalog,
  columns,
  onCancel,
  onSave,
}: {
  widget: Widget | null
  catalog: string[]
  columns: number
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
  }, [widget])

  if (widget === null) return null

  const bound = widget.bind !== undefined && widget.bind !== null
  const editsContent = widget.type === "text" && !bound
  const editsMetric = widget.type === "metric"
  // A metric bound to a column the derivation does not return renders a dash, so an
  // unknown name is worth saying before it is saved rather than after the tile blanks.
  const unknown = bound && derivation.length > 0 && !catalog.includes(derivation)

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
    if (editsMetric) {
      // "first" and "plain" are the absence of an aggregate and of a format, so they
      // are dropped rather than written as values the renderer would have to know.
      patch.viz = {
        ...(widget.viz ?? {}),
        field,
        ...(agg === "first" ? { agg: undefined } : { agg }),
        ...(format === "plain" ? { format: undefined } : { format }),
      }
      for (const key of Object.keys(patch.viz)) {
        if (patch.viz[key] === undefined) delete patch.viz[key]
      }
    }
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
                Percent multiplies by 100: a column already holding 69.5 renders as 6,950%. Use a
                plain number for a column that is already a percentage.
              </p>
            ) : null}
          </>
        ) : null}

        {widget.type !== "metric" && widget.type !== "text" ? (
          <p className="text-xs text-text-tertiary">
            This tile’s <code>viz</code> is a grammar-of-graphics spec. Use “Edit spec” to change
            how it draws.
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

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low
  return Math.min(high, Math.max(low, Math.round(value)))
}
