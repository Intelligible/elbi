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
import { getCatalog, getSourceConfig, type SourceField, updateSource } from "@/lib/warehouse"

/**
 * Edit an existing source's connection settings.
 *
 * Without this a correction costs the whole source: delete, re-create, re-enter every
 * secret, and lose the tables it produced. Secrets arrive blank from the server and are
 * sent back blank unless retyped, which the API reads as "unchanged" — so fixing a
 * manifest never costs credentials that were already working.
 */
export function EditSourceDialog({
  sourceId,
  sourceType,
  initialName,
  initialDescription,
  open,
  onOpenChange,
  onSaved,
}: {
  sourceId: string
  sourceType: string
  initialName: string
  initialDescription: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved: () => void
}) {
  const [fields, setFields] = useState<SourceField[]>([])
  const [inUse, setInUse] = useState<string[]>([])
  const [config, setConfig] = useState<Record<string, unknown>>({})
  const [name, setName] = useState(initialName)
  const [description, setDescription] = useState(initialDescription)
  // What the server last gave us, so a save can tell whether the connection actually
  // changed. It matters: a config change is re-tested against the live API, and a
  // lapsed key should not stand between an operator and a typo in a description.
  const [loaded, setLoaded] = useState("")
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setLoading(true)
    setError(null)
    Promise.all([getSourceConfig(sourceId), getCatalog()])
      .then(([view, catalog]) => {
        setConfig(view.config)
        setLoaded(JSON.stringify(view.config))
        setInUse(view.secretFields)
        setName(initialName)
        setDescription(initialDescription)
        const entry = catalog.sources.find((s) => s.name === sourceType)
        setFields(entry?.fields ?? [])
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [open, sourceId, sourceType, initialName, initialDescription])

  // A field with `dependsOn` appears only once the field it names holds the right
  // value, and only if that field is itself showing -- the same rule the create form
  // applies, so the two render a connector identically.
  const visible = (field: SourceField, seen: Set<string> = new Set()): boolean => {
    if (!field.dependsOn || seen.has(field.name)) return true
    const parent = fields.find((f) => f.name === field.dependsOn)
    if (!parent) return true
    if (!visible(parent, new Set([...seen, field.name]))) return false
    const held = config[parent.name] ?? parent.default
    return field.dependsValue ? held === field.dependsValue : Boolean(held)
  }

  // A connector can declare several credentials and use one -- Custom REST offers five,
  // picked by the manifest's auth type. Showing all of them on an edit is noise around
  // the one field that matters, so only the credentials the source actually holds are
  // offered. A source with none yet shows them all, since there is nothing to narrow to.
  const anyHeld = inUse.length > 0
  const shown = fields.filter(
    (f) => visible(f) && (f.type !== "password" || !anyHeld || inUse.includes(f.name)),
  )

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const connectionChanged = JSON.stringify(config) !== loaded
      await updateSource(sourceId, {
        name,
        description,
        // Only when it actually changed: sending it re-tests the connection, and
        // renaming a source should not fail because a credential has since lapsed.
        ...(connectionChanged ? { config } : {}),
      })
      onOpenChange(false)
      onSaved()
    } catch (err) {
      // The connection is tested before anything is stored, so a failure here is the
      // reason the edit was refused. Keep the dialog open with the message on it.
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Edit source</DialogTitle>
        </DialogHeader>

        {loading ? (
          <div className="text-sm text-text-tertiary">Loading…</div>
        ) : (
          <div className="flex max-h-[60vh] flex-col gap-4 overflow-y-auto">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="edit-source-name" className="font-medium text-sm">
                Name
              </label>
              <Input id="edit-source-name" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="flex flex-col gap-1.5">
              <label htmlFor="edit-source-description" className="font-medium text-sm">
                Description
              </label>
              <Input
                id="edit-source-description"
                placeholder="What this source is for"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
            {shown.map((field) => {
              const value = String(config[field.name] ?? "")
              const set = (v: string) => setConfig((c) => ({ ...c, [field.name]: v }))
              return (
                <div key={field.name} className="flex flex-col gap-1.5">
                  <label htmlFor={`edit-${field.name}`} className="font-medium text-sm">
                    {field.label}
                  </label>
                  {field.type === "textarea" ? (
                    // A manifest is a document, not a line: monospace, and tall enough
                    // to read its structure. Capped against the viewport rather than
                    // fixed, so on a short screen it yields instead of pushing the
                    // credential field below the fold.
                    <Textarea
                      id={`edit-${field.name}`}
                      rows={12}
                      spellCheck={false}
                      className="max-h-[38vh] min-h-[10rem] resize-y font-mono text-xs leading-relaxed"
                      value={value}
                      onChange={(e) => set(e.target.value)}
                    />
                  ) : (
                    <Input
                      id={`edit-${field.name}`}
                      type={field.type === "password" ? "password" : "text"}
                      placeholder={
                        field.type === "password"
                          ? "Leave blank to keep the stored value"
                          : field.placeholder
                      }
                      value={value}
                      onChange={(e) => set(e.target.value)}
                    />
                  )}
                  {field.caption ? (
                    <p className="text-text-tertiary text-xs">{field.caption}</p>
                  ) : null}
                </div>
              )
            })}
          </div>
        )}

        {error ? (
          <div
            role="alert"
            className="rounded-md border border-danger/30 bg-danger-tint px-3 py-2 font-mono text-xs text-danger"
          >
            {error}
          </div>
        ) : null}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={saving || loading}>
            {saving ? "Testing connection…" : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
