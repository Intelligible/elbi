import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { getCatalog, getSourceConfig, type SourceField, updateSource } from "@/lib/warehouse"
import { Field } from "./NewSourceForm"

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
  open,
  onOpenChange,
  onSaved,
}: {
  sourceId: string
  sourceType: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved: () => void
}) {
  const [fields, setFields] = useState<SourceField[]>([])
  const [config, setConfig] = useState<Record<string, unknown>>({})
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
        const entry = catalog.sources.find((s) => s.name === sourceType)
        setFields(entry?.fields ?? [])
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [open, sourceId, sourceType])

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      await updateSource(sourceId, { config })
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
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit source</DialogTitle>
        </DialogHeader>

        {loading ? (
          <div className="text-sm text-text-tertiary">Loading…</div>
        ) : (
          <div className="flex max-h-[60vh] flex-col gap-3 overflow-y-auto">
            {fields.map((field) => (
              <Field
                key={field.name}
                field={field}
                value={config[field.name]}
                onChange={(v) => setConfig((c) => ({ ...c, [field.name]: v }))}
              />
            ))}
            {fields.some((f) => f.type === "password") ? (
              <p className="text-xs text-text-tertiary">
                Leave a password field blank to keep the stored value.
              </p>
            ) : null}
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
