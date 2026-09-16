// The connection form for any warehouse source. Data-driven from the connector's field
// config and wired to the real create flow, so a new connector needs no new UI. Its
// spacing and sizing come from the same derived stylesheet as the rest of this surface
// (see `phc.tsx`, and NOTICE for attribution), and both themes are defined. Styles are
// scoped under `.phc` so they don't leak into the rest of the app.

import { Fragment, useCallback, useId, useState } from "react"

import { Phc } from "@/components/warehouse/phc"
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
    <div className="upload-row">
      <label className="btn btn--secondary btn--sm upload-btn">
        {busy ? "Uploading…" : "Upload file"}
        <input
          type="file"
          accept=".csv,.parquet"
          hidden
          disabled={busy}
          onChange={(e) => void pick(e.target.files?.[0])}
        />
      </label>
      {name && !err ? <span className="help">Uploaded {name}</span> : null}
      {err ? <span className="err-inline">{err}</span> : null}
    </div>
  )
}

export function Field({
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
  // own, so it is built here rather than dropped into the shared wrapper below.
  if (field.type === "switch") {
    return (
      <div className="field">
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span>{field.label}</span>
        </label>
        {field.caption ? <div className="help">{field.caption}</div> : null}
      </div>
    )
  }

  const control =
    field.type === "select" ? (
      <select
        className="select"
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
      >
        {field.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    ) : field.type === "textarea" ? (
      <textarea
        className="input"
        placeholder={field.placeholder}
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
      />
    ) : (
      <input
        id={fieldId}
        className="input"
        type={field.type === "password" ? "password" : field.type === "number" ? "number" : "text"}
        placeholder={field.placeholder}
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
      />
    )
  return (
    <div className="field">
      <label htmlFor={fieldId}>{field.label}</label>
      {control}
      {field.upload ? <UploadControl onUploaded={onChange} /> : null}
      {field.caption ? <div className="help">{field.caption}</div> : null}
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

  return (
    <Phc>
      <button type="button" className="crumb" onClick={onBack}>
        ‹ Sources
      </button>

      <div className="title-row">
        <h1>New data warehouse source</h1>
        <button type="button" className="btn btn--secondary btn--sm" onClick={onBack}>
          Cancel
        </button>
      </div>

      {error ? <div className="err">{error}</div> : null}

      <div className="src-head">
        <SourceIcon type={source.name} size={40} />
        <div>
          <h4>Link your data source</h4>
          <p>Sync data from {source.label} into your data warehouse.</p>
        </div>
      </div>

      {source.caption ? <p className="caption">{source.caption}</p> : null}
      {source.docsUrl ? (
        <div className="docs-row">
          <a href={source.docsUrl} target="_blank" rel="noreferrer">
            View docs ↗
          </a>
        </div>
      ) : null}

      <div className="field">
        <label htmlFor={nameId}>Source name</label>
        <input
          id={nameId}
          className="input"
          placeholder={source.name}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <div className="help">A unique name for this connection.</div>
      </div>

      <div className="field">
        <label htmlFor={descriptionId}>
          Description <span className="opt">(optional)</span>
        </label>
        <input
          id={descriptionId}
          className="input"
          placeholder="e.g. Production database"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <div className="help">
          A note to help you identify this source, e.g. 'Billing Stripe account'.
        </div>
      </div>

      {shown.map((field, index) => (
        <Fragment key={field.name}>
          {/* A heading appears the first time a section is seen, so a run of related
              fields reads as one block rather than as more of the same list. */}
          {field.section && field.section !== shown[index - 1]?.section ? (
            <h3 className="section">{field.section}</h3>
          ) : null}
          <Field
            field={field}
            value={config[field.name]}
            onChange={(v) => setConfig((c) => ({ ...c, [field.name]: v }))}
          />
        </Fragment>
      ))}

      <div className="field">
        <label htmlFor={prefixId}>
          Table prefix <span className="opt">(optional)</span>
        </label>
        <input
          id={prefixId}
          className="input"
          placeholder={source.name}
          value={prefix}
          onChange={(e) => setPrefix(e.target.value)}
        />
        <div className="help">
          Tables land as {prefix.trim() || source.name}.table_name. Use only letters, numbers, and
          underscores; must start with a letter or underscore.
        </div>
      </div>

      <hr className="divider" />
      <div className="footer">
        <button type="button" className="btn btn--secondary" onClick={onBack} disabled={busy}>
          Back
        </button>
        <button
          type="button"
          className="btn btn--primary"
          onClick={submit}
          disabled={busy || !name.trim()}
        >
          {busy ? "Connecting…" : "Next"}
        </button>
      </div>
    </Phc>
  )
}
