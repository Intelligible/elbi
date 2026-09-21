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
import { Textarea } from "@/components/ui/textarea"
import type { Widget } from "@/lib/dashboards"

/** What the editor can change. Anything deeper stays with the whole-spec JSON editor. */
export interface TilePatch {
  title?: string
  content?: string
  derivation?: string
}

/**
 * Edit one tile's title and body.
 *
 * A text tile's body is its own markdown or a derivation that returns markdown; a
 * data tile's body is whichever derivation it binds. Those are the fields someone
 * reaches for while laying a dashboard out, so they are here rather than behind the
 * spec JSON: `viz`, params and interactions still live there.
 */
export function TileEditor({
  widget,
  catalog,
  onCancel,
  onSave,
}: {
  widget: Widget | null
  catalog: string[]
  onCancel: () => void
  onSave: (id: string, patch: TilePatch) => void
}) {
  const [title, setTitle] = useState("")
  const [content, setContent] = useState("")
  const [derivation, setDerivation] = useState("")

  // Reset the fields each time a different tile is opened, so the dialog never shows
  // the previous tile's text.
  useEffect(() => {
    setTitle(widget?.title ?? "")
    setContent(widget?.content ?? "")
    setDerivation(widget?.bind?.derivation ?? "")
  }, [widget])

  if (widget === null) return null

  const bound = widget.bind !== undefined && widget.bind !== null
  const editsContent = widget.type === "text" && !bound
  // A metric bound to nothing renders a dash, so an unknown name is worth saying
  // before it is saved rather than after the tile goes blank.
  const unknown = bound && derivation.length > 0 && !catalog.includes(derivation)

  return (
    <Dialog open onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Edit tile</DialogTitle>
          <p className="text-xs text-text-tertiary">
            <code>{widget.id}</code> · {widget.type}
          </p>
        </DialogHeader>

        <label className="block text-sm font-medium" htmlFor="tile-title">
          Title
        </label>
        <Input
          id="tile-title"
          value={title}
          placeholder="Untitled tile"
          onChange={(e) => setTitle(e.target.value)}
        />

        {editsContent ? (
          <>
            <label className="block text-sm font-medium" htmlFor="tile-content">
              Content
            </label>
            <Textarea
              id="tile-content"
              className="h-56 font-mono text-xs"
              value={content}
              spellCheck={false}
              onChange={(e) => setContent(e.target.value)}
            />
            <p className="text-xs text-text-tertiary">
              Markdown. A figure typed here is a copy that goes stale — bind a derivation that
              returns markdown to keep it live.
            </p>
          </>
        ) : null}

        {bound ? (
          <>
            <label className="block text-sm font-medium" htmlFor="tile-derivation">
              Derivation
            </label>
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
          </>
        ) : null}

        <DialogFooter>
          <Button variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            onClick={() =>
              onSave(widget.id, {
                title,
                ...(editsContent ? { content } : {}),
                ...(bound ? { derivation } : {}),
              })
            }
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
