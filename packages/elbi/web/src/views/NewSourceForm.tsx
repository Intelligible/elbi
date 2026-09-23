// The connection form for any warehouse source. Data-driven from the connector's field
// config and wired to the real create flow, so a new connector needs no new UI.

import { Fragment, useCallback, useId, useState } from "react"

import { FormControl, FormField } from "@/components/app/FormField"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"
import { SourceIcon } from "@/components/warehouse/SourceIcon"
import {
  createSource,
  type SourceConfig,
  type SourceDetail,
  type SourceField,
  uploadWarehouseFile,
} from "@/lib/warehouse"

// The upload control paired with a text field: pick a local CSV/Parquet file, send it
// to the warehouse, and fill the field with the stored path it returns. The plain text
// input stays available so a path or object-store URL can still be typed instead.
function UploadControl({ onUploaded }: { onUploaded: (path: string) => void }) {
  const [busy, setBusy] = useState(false)
  const [name, setName] = useState("")
  const [err, setErr] = useState("")

  const pick = useCallback(
    async (file: File | undefined) => {
      if (!file) return
      setBusy(true)
      setErr("")
      try {
        const { path } = await uploadWarehouseFile(file)
        setName(file.name)
        onUploaded(path)
      } catch (e) {
        setName("")
        setErr(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(false)
      }
    },
    [onUploaded],
  )

  return (
    <div className="mt-0.5 flex items-center gap-2.5">
      <Button variant="outline" size="sm" asChild>
        <label className="cursor-pointer">
          {busy ? "Uploading…" : "Upload file"}
          <Input
            type="file"
            accept=".csv,.parquet"
            className="hidden"
            disabled={busy}
            onChange={(e) => void pick(e.target.files?.[0])}
          />
        </label>
      </Button>
      {name && !err ? <span className="text-xs text-text-tertiary">Uploaded {name}</span> : null}
      {err ? <span className="text-xs text-danger">{err}</span> : null}
    </div>
  )
}

function Field({
  field,
  value,
  onChange,
}: {
  field: SourceField
  value: unknown
  onChange: (v: unknown) => void
}) {
  const fieldId = useId()
  // A switch reads as a control with its label beside it, not under a heading of its
  // own, so it is built here rather than dropped into FormField.
  if (field.type === "switch") {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Checkbox
            id={fieldId}
            checked={Boolean(value)}
            onCheckedChange={(c) => onChange(c === true)}
          />
          <Label htmlFor={fieldId}>{field.label}</Label>
        </div>
        {field.caption ? <p className="text-xs text-text-tertiary">{field.caption}</p> : null}
      </div>
    )
  }

  const text = String(value ?? "")
  const control =
    field.type === "select" ? (
      <Select value={text || undefined} onValueChange={onChange}>
        <FormControl>
          <SelectTrigger className="w-full bg-card">
            <SelectValue />
          </SelectTrigger>
        </FormControl>
        <SelectContent>
          {field.options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    ) : field.type === "textarea" ? (
      <Textarea
        className="bg-card"
        placeholder={field.placeholder}
        value={text}
        onChange={(e) => onChange(e.target.value)}
      />
    ) : (
      <Input
        className="bg-card"
        type={field.type === "password" ? "password" : field.type === "number" ? "number" : "text"}
        placeholder={field.placeholder}
        value={text}
        onChange={(e) => onChange(e.target.value)}
      />
    )
  return (
    <div className="space-y-1">
      {/* The upload row sits between the input and its caption, so an upload field
          renders its caption itself rather than through FormField's hint. */}
      <FormField label={field.label} hint={field.upload ? undefined : field.caption || undefined}>
        {control}
      </FormField>
      {field.upload ? <UploadControl onUploaded={onChange} /> : null}
      {field.upload && field.caption ? (
        <p className="text-xs text-text-tertiary">{field.caption}</p>
      ) : null}
    </div>
  )
}

export function NewSourceForm({
  source,
  error,
  onBack,
  onCreated,
  onError,
}: {
  source: SourceConfig
  error: string | null
  onBack: () => void
  onCreated: (detail: SourceDetail) => void
  onError: (message: string) => void
}) {
  const nameId = useId()
  const descriptionId = useId()
  const prefixId = useId()
  const [name, setName] = useState(source.name)
  const [description, setDescription] = useState("")
  const [prefix, setPrefix] = useState("")
  const [config, setConfig] = useState<Record<string, unknown>>(() => {
    const initial: Record<string, unknown> = {}
    for (const f of source.fields) {
      if (f.default !== null && f.default !== undefined) initial[f.name] = f.default
      // A single-option select is auto-selected (and hidden below).
      else if (f.type === "select" && f.options.length === 1) initial[f.name] = f.options[0].value
    }
    return initial
  })
  const [busy, setBusy] = useState(false)

  // Hide selects that have only one option: there's nothing to choose.
  const fields = source.fields.filter((f) => !(f.type === "select" && f.options.length <= 1))

  // A field with `dependsOn` appears only once the field it names holds the right
  // value -- and only if that field is itself showing, so a chain of conditions
  // collapses together rather than leaving an orphan visible under a hidden switch.
  const visible = useCallback(
    (field: SourceField, seen: Set<string> = new Set()): boolean => {
      if (!field.dependsOn || seen.has(field.name)) return true
      const parent = fields.find((f) => f.name === field.dependsOn)
      if (!parent) return true
      if (!visible(parent, new Set([...seen, field.name]))) return false
      const held = config[parent.name] ?? parent.default
      return field.dependsValue ? held === field.dependsValue : Boolean(held)
    },
    [fields, config],
  )
  const shown = fields.filter((f) => visible(f))

  const submit = useCallback(async () => {
    setBusy(true)
    onError("")
    try {
      const detail = await createSource({
        source_type: source.name,
        name: name.trim(),
        config,
        description,
        prefix,
      })
      onCreated(detail)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [source.name, name, description, prefix, config, onCreated, onError])

  const optional = <span className="font-normal text-text-tertiary">(optional)</span>

  return (
    <div className="absolute inset-0 overflow-y-auto bg-panel text-sm text-foreground">
      <div className="px-5 pt-4 pb-12">
        <Button
          variant="link"
          className="mb-1 h-auto px-0 py-1 text-compact font-semibold text-text-tertiary hover:text-foreground"
          onClick={onBack}
        >
          ‹ Sources
        </Button>

        <div className="mb-5 flex items-center justify-between gap-3">
          <h1 className="text-title font-semibold">New data warehouse source</h1>
          <Button variant="outline" size="sm" onClick={onBack}>
            Cancel
          </Button>
        </div>

        {error ? (
          <div className="mb-4 rounded-md border border-danger/30 bg-danger-tint px-3 py-2 text-compact text-danger">
            {error}
          </div>
        ) : null}

        <div className="mb-4 flex items-center gap-3">
          <SourceIcon type={source.name} size={40} />
          <div>
            <h4 className="text-lg font-semibold">Link your data source</h4>
            <p className="mt-0.5 text-sm text-text-tertiary">
              Sync data from {source.label} into your data warehouse.
            </p>
          </div>
        </div>

        {source.caption ? (
          <p className="mt-2 mb-3 text-sm leading-normal text-text-secondary">{source.caption}</p>
        ) : null}
        {source.docsUrl ? (
          <div className="mb-4 flex items-center gap-2 text-compact">
            <a
              href={source.docsUrl}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
            >
              View docs ↗
            </a>
          </div>
        ) : null}

        <div className="space-y-4">
          <FormField label="Source name" hint="A unique name for this connection.">
            <Input
              className="bg-card"
              id={nameId}
              placeholder={source.name}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </FormField>

          <FormField
            label={<>Description {optional}</>}
            hint="A note to help you identify this source, e.g. 'Billing Stripe account'."
          >
            <Input
              className="bg-card"
              id={descriptionId}
              placeholder="e.g. Production database"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </FormField>

          {shown.map((field, index) => (
            <Fragment key={field.name}>
              {/* A heading appears the first time a section is seen, so a run of related
                  fields reads as one block rather than as more of the same list. */}
              {field.section && field.section !== shown[index - 1]?.section ? (
                <h3 className="mt-6 mb-1 border-t border-border pt-4 text-compact font-semibold text-text-tertiary">
                  {field.section}
                </h3>
              ) : null}
              <Field
                field={field}
                value={config[field.name]}
                onChange={(v) => setConfig((c) => ({ ...c, [field.name]: v }))}
              />
            </Fragment>
          ))}

          <FormField
            label={<>Table prefix {optional}</>}
            hint={
              <>
                Tables land as {prefix.trim() || source.name}.table_name. Use only letters, numbers,
                and underscores; must start with a letter or underscore.
              </>
            }
          >
            <Input
              className="bg-card"
              id={prefixId}
              placeholder={source.name}
              value={prefix}
              onChange={(e) => setPrefix(e.target.value)}
            />
          </FormField>
        </div>

        <Separator className="my-6" />
        <div className="my-4 flex justify-end gap-2">
          <Button variant="outline" onClick={onBack} disabled={busy}>
            Back
          </Button>
          <Button onClick={submit} disabled={busy || !name.trim()}>
            {busy ? "Connecting…" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  )
}
