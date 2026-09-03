import {
  Boxes,
  Check,
  Copy,
  Crown,
  Download,
  ExternalLink,
  Loader2,
  NotebookPen,
  Plus,
  Trash2,
} from "lucide-react"
import { useCallback, useEffect, useId, useRef, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { Streamdown } from "streamdown"
import { Scene, SceneBody, SceneHeader, SceneSection, SceneSkeleton } from "@/components/Scene"
import { TrainingStatus, useTrainingJobs } from "@/components/TrainingJobs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { FeedbackProvider, useFeedback } from "@/components/ui/feedback"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { VerdictBadge } from "@/components/VerdictBadge"
import { useRowKeys } from "@/hooks/useRowKeys"
import {
  type BatchScoreResult,
  batchScoreArtifactUrl,
  batchScoreModel,
  type DatasetColumn,
  type DriftHistoryEntry,
  type DriftResult,
  deleteModel,
  deleteModelVersion,
  deleteRetrainPolicy,
  driftReportUrl,
  type FeatureSource,
  getBatchScoreJob,
  getDatasetColumns,
  getDriftHistory,
  getFeatureSources,
  getInferenceLog,
  getModelSchema,
  getRegisteredModel,
  getRetrainPolicy,
  type InferenceLogEntry,
  invokeModel,
  type ModelSchema,
  type ModelVersion,
  promoteModelVersion,
  putRetrainPolicy,
  type RawScoreResult,
  type RegisteredModelDetail,
  type RegistryUnavailable,
  type RetrainPolicy,
  type RetrainPolicyInput,
  rawInvokeModel,
  runDriftCheck,
} from "@/lib/chat"
import { notebookFromModel } from "@/lib/notebooks"
import { copyText, EMPTY } from "@/lib/utils"

export function ModelDetailPage() {
  return (
    <FeedbackProvider>
      <ModelDetailBody />
    </FeedbackProvider>
  )
}

function ModelDetailBody() {
  const { name = "" } = useParams()
  const navigate = useNavigate()
  const fb = useFeedback()
  const [d, setD] = useState<RegisteredModelDetail | RegistryUnavailable | "missing" | null>(null)
  const load = useCallback(() => {
    getRegisteredModel(name).then((r) => setD(r ?? "missing"))
  }, [name])
  useEffect(load, [load])
  const { jobs, dismiss } = useTrainingJobs(load, name)

  const detail = d !== null && d !== "missing" && !("unavailable" in d) ? d : null

  const removeModel = async () => {
    if (!detail) return
    if (
      !(await fb.confirm({
        title: `Delete model "${detail.name}"?`,
        body: `All ${detail.versions.length} version${
          detail.versions.length === 1 ? "" : "s"
        } and the @champion alias are removed from the registry, and serving stops. This cannot be undone.`,
        danger: true,
      }))
    )
      return
    if (await deleteModel(detail.name)) {
      navigate("/models")
    } else {
      fb.toast("error", `Could not delete "${detail.name}"`)
    }
  }

  if (d === null) return <SceneSkeleton />

  if (d === "missing" || "unavailable" in d) {
    return (
      <Scene>
        <SceneHeader
          icon={<Boxes className="size-5" />}
          title={name}
          mono
          backTo="/models"
          backLabel="Models"
        />
        <SceneBody width="wide">
          <p className="text-sm text-text-secondary">
            {d === "missing" ? `No model named ${name}.` : d.unavailable}
          </p>
        </SceneBody>
      </Scene>
    )
  }

  return (
    <Scene>
      <SceneHeader
        icon={<Boxes className="size-5" />}
        title={d.name}
        mono
        backTo="/models"
        backLabel="Models"
        badges={
          d.championVersion !== null ? (
            <Badge variant="verified">
              <Crown className="size-3.5" /> @champion → v{d.championVersion}
            </Badge>
          ) : (
            <Badge variant="neutral">no champion</Badge>
          )
        }
        actions={
          <>
            <Button variant="outline" size="sm" asChild>
              <a
                href={`/mlflow/#/models/${encodeURIComponent(d.name)}`}
                target="_blank"
                rel="noreferrer"
              >
                <ExternalLink className="size-4" /> View in MLflow
              </a>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => notebookFromModel(d.name).then((r) => navigate(`/notebooks/${r.id}`))}
              title="Open the training script in a notebook"
            >
              <NotebookPen className="size-4" /> Open in notebook
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link to={`/models/train?name=${encodeURIComponent(d.name)}`}>
                <Plus className="size-4" /> Train new version
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-destructive"
              onClick={() => void removeModel()}
            >
              <Trash2 className="size-3.5" /> Delete model
            </Button>
          </>
        }
      />
      <SceneBody width="wide">
        <div className="space-y-6">
          {(d.versions.length > 0 || servedFeatureDerivation(d) !== null) && (
            <div className="space-y-1">
              {d.versions.length > 0 && (
                <p className="text-xs text-text-tertiary">
                  Created{" "}
                  {new Date(Math.min(...d.versions.map((v) => v.createdAtMs))).toLocaleDateString()}{" "}
                  · Updated{" "}
                  {new Date(Math.max(...d.versions.map((v) => v.createdAtMs))).toLocaleDateString()}
                </p>
              )}
              <LineageLine detail={d} />
            </div>
          )}

          {jobs.length > 0 && (
            <div className="space-y-2">
              {jobs.map((j) => (
                <TrainingStatus key={j.id} job={j} onDismiss={() => dismiss(j.id)} />
              ))}
            </div>
          )}

          <ModelCardSection detail={d} />
          <VersionsSection detail={d} onChanged={load} />
          <QueryPane detail={d} />
          <BatchScoreSection detail={d} />
          <MonitoringSection name={d.name} />
          <AutoRetrainSection detail={d} />
        </div>
      </SceneBody>
    </Scene>
  )
}

// The registered versions, newest first. Held-out metrics get a column per `holdout_*`
// key seen across versions, so mixed tasks (accuracy vs r2) still line up.
function VersionsSection({
  detail,
  onChanged,
}: {
  detail: RegisteredModelDetail
  onChanged: () => void
}) {
  const fb = useFeedback()
  const [promoting, setPromoting] = useState<ModelVersion | null>(null)
  const metricKeys = [
    ...new Set(
      detail.versions.flatMap((v) =>
        Object.keys(v.metrics).filter((k) => k.startsWith("holdout_")),
      ),
    ),
  ].sort()

  const promote = async (version: number) => {
    setPromoting(null)
    if (await promoteModelVersion(detail.name, version)) onChanged()
  }

  const removeVersion = async (v: ModelVersion) => {
    if (
      !(await fb.confirm({
        title: `Delete v${v.version} of "${detail.name}"?`,
        body: `The version and its artifacts are removed from the registry${
          v.version === detail.championVersion ? ", and @champion stops resolving" : ""
        }. This cannot be undone.`,
        danger: true,
      }))
    )
      return
    if (await deleteModelVersion(detail.name, v.version)) {
      fb.toast("ok", `Deleted v${v.version}`)
    } else {
      fb.toast("error", `Could not delete v${v.version}`)
    }
    onChanged()
  }

  return (
    <SceneSection title="Versions">
      {detail.versions.length === 0 ? (
        <p className="text-sm text-text-secondary">No versions registered yet.</p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border bg-card">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface-secondary text-left text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-text-tertiary">
                <th className="px-4 py-2">Version</th>
                <th className="px-4 py-2">Created</th>
                <th className="px-4 py-2">Aliases</th>
                {metricKeys.map((k) => (
                  <th key={k} className="px-4 py-2">
                    {k.slice("holdout_".length)}
                  </th>
                ))}
                <th className="px-4 py-2">Estimator</th>
                <th className="px-4 py-2">Task</th>
                <th className="px-4 py-2">Run</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {detail.versions.map((v) => {
                const isChampion = v.version === detail.championVersion
                return (
                  <tr key={v.version}>
                    <td className="px-4 py-2.5 font-mono">
                      <a
                        href={`/mlflow/#/models/${encodeURIComponent(detail.name)}/versions/${v.version}`}
                        target="_blank"
                        rel="noreferrer"
                        title="Open this version in MLflow"
                        className="text-foreground/90 underline decoration-border underline-offset-2 transition hover:text-foreground hover:decoration-foreground/40"
                      >
                        v{v.version}
                      </a>
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">
                      {new Date(v.createdAtMs).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-2.5">
                      {v.aliases.length === 0 ? (
                        // No alias is a question ("why is this not deployed?"), and the
                        // oracle's verdict is the answer, with its reason on hover.
                        v.verdict ? (
                          <span title={v.verdictDetail ?? undefined}>
                            <VerdictBadge verdict={v.verdict} />
                          </span>
                        ) : (
                          <span className="text-text-tertiary">{EMPTY}</span>
                        )
                      ) : (
                        <span className="flex flex-wrap gap-1">
                          {v.aliases.map((a) => (
                            <Badge
                              key={a}
                              variant={a === "champion" ? "verified" : "neutral"}
                              className="font-mono"
                            >
                              @{a}
                            </Badge>
                          ))}
                        </span>
                      )}
                    </td>
                    {metricKeys.map((k) => (
                      <td key={k} className="px-4 py-2.5 font-mono text-xs">
                        {k in v.metrics ? v.metrics[k].toFixed(4) : EMPTY}
                      </td>
                    ))}
                    <td className="px-4 py-2.5 font-mono text-xs">
                      {v.params.bestEstimator ?? EMPTY}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-muted-foreground">
                      {v.params.task ?? EMPTY}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 font-mono text-xs">
                      {v.experimentId && v.runId ? (
                        <a
                          href={`/mlflow/#/experiments/${encodeURIComponent(v.experimentId)}/runs/${encodeURIComponent(v.runId)}`}
                          target="_blank"
                          rel="noreferrer"
                          title="Open the source run in MLflow"
                          className="inline-flex items-center gap-1 text-muted-foreground transition hover:text-foreground"
                        >
                          {v.runId.slice(0, 8)} <ExternalLink className="h-3 w-3" />
                        </a>
                      ) : (
                        <span className="text-muted-foreground">{EMPTY}</span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-right">
                      {!isChampion && (
                        <Button variant="outline" size="xs" onClick={() => setPromoting(v)}>
                          Promote to champion
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Delete v${v.version}`}
                        className="ml-1 text-destructive"
                        onClick={() => void removeVersion(v)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <ConfirmPromoteDialog
        version={promoting}
        currentChampion={detail.championVersion}
        onOpenChange={(open) => !open && setPromoting(null)}
        onConfirm={(version) => void promote(version)}
      />
    </SceneSection>
  )
}

function ConfirmPromoteDialog({
  version,
  currentChampion,
  onOpenChange,
  onConfirm,
}: {
  version: ModelVersion | null
  currentChampion: number | null
  onOpenChange: (open: boolean) => void
  onConfirm: (version: number) => void
}) {
  return (
    <Dialog open={version !== null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Promote v{version?.version} to champion?</DialogTitle>
          <DialogDescription>
            The <span className="font-mono text-foreground">@champion</span> alias moves to v
            {version?.version}
            {currentChampion !== null && ` (currently v${currentChampion})`}, and serving picks it
            up immediately.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => version && onConfirm(version.version)}>Promote</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// Query the serving endpoint from the page: pick a version, fill (or auto-fill) a
// dataframe_records body, send it, and read the predictions, with the signature and a copyable
// curl alongside.
function QueryPane({ detail }: { detail: RegisteredModelDetail }) {
  const rowKey = useRowKeys()
  // Derivation-trained models also take RAW source rows via /raw-invocations, where
  // the recorded feature derivation engineers the features before scoring.
  const featureDerivation = servedFeatureDerivation(detail)
  const [mode, setMode] = useState<"engineered" | "raw">("engineered")
  const [version, setVersion] = useState<number | null>(
    detail.championVersion ?? detail.versions[0]?.version ?? null,
  )
  const [schema, setSchema] = useState<ModelSchema | null>(null)
  const [body, setBody] = useState("")
  const [sending, setSending] = useState(false)
  const [response, setResponse] = useState<{ ok: boolean; text: string } | null>(null)
  const [rawResult, setRawResult] = useState<RawScoreResult | null>(null)
  const [copied, setCopied] = useState(false)

  // Keep the selected version valid across refetches (a promote or a deleted version).
  useEffect(() => {
    if (version === null || !detail.versions.some((v) => v.version === version)) {
      setVersion(detail.championVersion ?? detail.versions[0]?.version ?? null)
    }
  }, [detail, version])

  useEffect(() => {
    setSchema(null)
    setResponse(null)
    if (version === null) return
    void getModelSchema(detail.name, version).then(setSchema)
  }, [detail.name, version])

  if (detail.versions.length === 0 || version === null) return null

  const exampleRecords = (s: ModelSchema | null): Array<Record<string, unknown>> =>
    s?.inputExample ??
    (s ? [Object.fromEntries(s.inputs.map((i) => [i.name, exampleValue(i.type)]))] : [{}])

  // Prefer the version's logged input example; fall back to a skeleton built from the
  // signature. Refetches when the mount-time schema load failed or has not landed yet.
  const fillExample = async () => {
    let s = schema
    if (s === null) {
      s = await getModelSchema(detail.name, version)
      if (s) setSchema(s)
    }
    setBody(JSON.stringify({ dataframe_records: exampleRecords(s) }, null, 2))
  }

  const switchMode = (next: "engineered" | "raw") => {
    if (next === mode) return
    setMode(next)
    setResponse(null)
    setRawResult(null)
  }

  const send = async () => {
    setResponse(null)
    setRawResult(null)
    let parsed: unknown
    try {
      parsed = JSON.parse(body) as unknown
    } catch {
      setResponse({ ok: false, text: "The request body is not valid JSON." })
      return
    }
    setSending(true)
    if (mode === "raw") {
      const r = await rawInvokeModel(detail.name, parsed)
      setSending(false)
      if ("error" in r) setResponse({ ok: false, text: r.error })
      else setRawResult(r)
      return
    }
    const r = await invokeModel(detail.name, version, parsed)
    setSending(false)
    setResponse(
      "error" in r ? { ok: false, text: r.error } : { ok: true, text: JSON.stringify(r, null, 2) },
    )
  }

  const curl =
    mode === "raw"
      ? [
          `curl -X POST '${window.location.origin}/api/serving/${encodeURIComponent(detail.name)}/raw-invocations' \\`,
          `  -H 'Content-Type: application/json' \\`,
          `  -d '${JSON.stringify({ dataframe_records: [{}] })}'`,
        ].join("\n")
      : [
          `curl -X POST '${window.location.origin}/api/serving/${encodeURIComponent(detail.name)}/invocations?version=${version}' \\`,
          `  -H 'Content-Type: application/json' \\`,
          `  -d '${JSON.stringify({ dataframe_records: exampleRecords(schema) })}'`,
        ].join("\n")

  const engineeredColumns = Object.keys(rawResult?.engineered[0] ?? {})

  return (
    <SceneSection title="Query endpoint">
      <div className="space-y-3 rounded-xl border border-border bg-card p-4">
        {featureDerivation !== null && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium">Input</span>
            <Button
              variant={mode === "engineered" ? "outline" : "ghost"}
              size="xs"
              onClick={() => switchMode("engineered")}
            >
              Engineered features
            </Button>
            <Button
              variant={mode === "raw" ? "outline" : "ghost"}
              size="xs"
              onClick={() => switchMode("raw")}
            >
              Raw records
            </Button>
          </div>
        )}

        {mode === "raw" ? (
          <p className="text-xs text-muted-foreground">
            Raw rows score against the served version: the certified derivation{" "}
            <span className="font-mono text-foreground">{featureDerivation}</span> engineers the
            features first, so send source-shaped records: the same columns the derivation reads.
          </p>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium">Version</span>
            <select
              className="rounded-md border border-border bg-background px-2 py-1 font-mono text-xs"
              value={version}
              onChange={(e) => setVersion(Number(e.target.value))}
            >
              {detail.versions.map((v) => (
                <option key={v.version} value={v.version}>
                  v{v.version}
                  {v.version === detail.championVersion ? " (champion)" : ""}
                </option>
              ))}
            </select>
          </div>
        )}

        {mode === "raw" ? null : schema === null ? (
          <p className="text-xs text-muted-foreground">
            No signature is available for this version.
          </p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border bg-card">
            <table className="w-full text-xs">
              <thead>
                <tr className={headRowClass}>
                  <th className={thClass}>Input</th>
                  <th className={thClass}>Type</th>
                  <th className={thClass}>Required</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {schema.inputs.map((i) => (
                  <tr key={i.name}>
                    <td className="px-3 py-1.5 font-mono">{i.name}</td>
                    <td className="px-3 py-1.5 font-mono text-muted-foreground">{i.type}</td>
                    <td className="px-3 py-1.5 text-muted-foreground">
                      {i.required ? "yes" : "no"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <Textarea
          className="min-h-28 font-mono text-xs md:text-xs"
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder={
            mode === "raw"
              ? '{"dataframe_records": [{"source_column": "value"}]}'
              : '{"dataframe_records": [{"feature": 0}]}'
          }
          spellCheck={false}
        />
        <div className="flex items-center gap-2">
          {mode === "engineered" && (
            <Button variant="outline" size="sm" onClick={() => void fillExample()}>
              Show example
            </Button>
          )}
          <Button size="sm" disabled={sending || body.trim() === ""} onClick={() => void send()}>
            {sending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {sending ? "Sending…" : "Send request"}
          </Button>
        </div>

        {rawResult && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              Derivation{" "}
              <span className="font-mono text-foreground">{rawResult.featuresApplied}</span>{" "}
              engineered {rawResult.nRawRows} raw row
              {rawResult.nRawRows === 1 ? "" : "s"} into {rawResult.nScoredRows} scored row
              {rawResult.nScoredRows === 1 ? "" : "s"}. Engineered sample:
            </p>
            {rawResult.engineered.length > 0 && (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={headRowClass}>
                      {engineeredColumns.map((k) => (
                        <th key={k} className={thClass}>
                          {k}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {rawResult.engineered.map((row) => (
                      <tr key={rowKey(row)}>
                        {engineeredColumns.map((k) => (
                          <td key={k} className="whitespace-nowrap px-3 py-1.5 font-mono">
                            {cellText(row[k])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <pre className="overflow-x-auto rounded-lg border border-border bg-muted px-3 py-2.5 font-mono text-xs">
              {JSON.stringify({ predictions: rawResult.predictions }, null, 2)}
            </pre>
          </div>
        )}

        {response && (
          <pre
            className={`overflow-x-auto rounded-lg border px-3 py-2.5 font-mono text-xs ${
              response.ok
                ? "border-border bg-muted"
                : "border-destructive/30 bg-destructive/10 text-destructive"
            }`}
          >
            {response.text}
          </pre>
        )}

        <div className="flex items-start gap-2 rounded-lg border border-border bg-muted px-3 py-2.5">
          <pre className="min-w-0 flex-1 overflow-x-auto font-mono text-xs">{curl}</pre>
          <button
            type="button"
            onClick={() => {
              void copyText(curl)
              setCopied(true)
            }}
            className="shrink-0 text-muted-foreground hover:text-foreground"
            aria-label="Copy"
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </SceneSection>
  )
}

// A placeholder value matching an MLflow signature type, for the skeleton request body.
function exampleValue(type: string): unknown {
  const t = type.toLowerCase()
  if (t.includes("int") || t.includes("long")) return 0
  if (t.includes("float") || t.includes("double")) return 0.0
  if (t.includes("bool")) return false
  return ""
}

const fieldClass =
  "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring"

const thClass = "px-3 py-1.5"
const headRowClass =
  "border-b border-border bg-surface-secondary text-left text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-text-tertiary"

// The version serving resolves to: the champion, or the newest one without a champion.
function servedVersion(detail: RegisteredModelDetail): ModelVersion | undefined {
  return detail.versions.find((v) => v.version === detail.championVersion) ?? detail.versions[0]
}

// The feature derivation the served version was trained on, from its lineage tag;
// null for models trained straight on a dataset.
function servedFeatureDerivation(detail: RegisteredModelDetail): string | null {
  return servedVersion(detail)?.tags["elbi.feature_derivation"] ?? null
}

// Feature lineage in the header: a derivation-trained model links to the certified
// pipeline that engineered its training features.
function LineageLine({ detail }: { detail: RegisteredModelDetail }) {
  const derivation = servedFeatureDerivation(detail)
  if (derivation === null) return null
  return (
    <p className="text-xs text-text-tertiary">
      Features from derivation{" "}
      <Link
        to={`/derivations/${encodeURIComponent(derivation)}`}
        className="font-mono text-foreground/90 underline decoration-border underline-offset-2 transition hover:text-foreground hover:decoration-foreground/40"
      >
        {derivation}
      </Link>
    </p>
  )
}

// A data source (dataset or derivation) encoded into one select value; names never
// contain ":" but split on the first one regardless.
function encodeSource(kind: FeatureSource["kind"], name: string): string {
  return `${kind}:${name}`
}

function decodeSource(value: string): { kind: FeatureSource["kind"]; name: string } {
  const i = value.indexOf(":")
  return {
    kind: value.slice(0, i) === "derivation" ? "derivation" : "dataset",
    name: value.slice(i + 1),
  }
}

// The grouped data-source select shared by batch scoring and the retrain policy:
// bound datasets, then certified feature derivations. `extra` keeps a stored value
// selectable when it is missing from the fetched list.
function SourceSelect({
  sources,
  value,
  onChange,
  extra,
}: {
  sources: FeatureSource[]
  value: string
  onChange: (value: string) => void
  extra?: string
}) {
  const groups: Array<[string, FeatureSource[]]> = [
    ["Datasets", sources.filter((s) => s.kind === "dataset")],
    ["Feature derivations", sources.filter((s) => s.kind === "derivation")],
  ]
  const listed = (v: string) => sources.some((s) => encodeSource(s.kind, s.name) === v)
  return (
    <select className={fieldClass} value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="" disabled>
        Choose a data source…
      </option>
      {extra !== undefined && extra !== "" && !listed(extra) && (
        <option value={extra}>{decodeSource(extra).name}</option>
      )}
      {groups.map(([label, items]) =>
        items.length === 0 ? null : (
          <optgroup key={label} label={label}>
            {items.map((s) => (
              <option key={s.name} value={encodeSource(s.kind, s.name)}>
                {s.name}
              </option>
            ))}
          </optgroup>
        ),
      )}
    </select>
  )
}

function formatNumber(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(4)
}

// A compact cell rendering for arbitrary prediction values in the sample table.
function cellText(v: unknown): string {
  if (v === null || v === undefined) return EMPTY
  if (typeof v === "number") return formatNumber(v)
  if (typeof v === "object") return JSON.stringify(v)
  return String(v)
}

// The served version's model card, its markdown `description`, collapsed by default.
function ModelCardSection({ detail }: { detail: RegisteredModelDetail }) {
  const served = servedVersion(detail)
  if (!served || served.description.trim() === "") return null
  return (
    <SceneSection title="Model card">
      <details className="rounded-xl border border-border bg-card">
        <summary className="cursor-pointer px-4 py-3 text-sm text-text-secondary transition hover:text-foreground">
          Model card for v{served.version}
          {served.version === detail.championVersion && " (champion)"}
        </summary>
        <div className="border-t border-border px-4 py-3 text-sm">
          <Streamdown>{served.description}</Streamdown>
        </div>
      </details>
    </SceneSection>
  )
}

// Score a whole dataset against the model. The endpoint follows the training-job
// contract, but the job's label ("batch score <name>") is outside the training hook's
// view, so this pane polls /api/jobs/{id} itself until the job is terminal.
function BatchScoreSection({ detail }: { detail: RegisteredModelDetail }) {
  const rowKey = useRowKeys()
  const [sources, setSources] = useState<FeatureSource[]>([])
  // The rows to score, as an encoded `kind:name` source value.
  const [source, setSource] = useState("")
  const [version, setVersion] = useState("")
  const [running, setRunning] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState<BatchScoreResult | null>(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    void getFeatureSources().then((s) => {
      if (alive.current) setSources(s)
    })
    return () => {
      alive.current = false
    }
  }, [])

  const run = async () => {
    setRunning(true)
    setError("")
    setResult(null)
    const src = decodeSource(source)
    const r = await batchScoreModel(detail.name, {
      ...(src.kind === "derivation" ? { derivation: src.name } : { dataset: src.name }),
      ...(version !== "" ? { version } : {}),
    })
    if ("error" in r) {
      if (alive.current) {
        setError(r.error)
        setRunning(false)
      }
      return
    }
    // Inline execution (id null) is already terminal; otherwise poll to the end. A
    // transient poll failure keeps the last state and simply tries again.
    let job = r.job
    while (job.id !== null && (job.state === "queued" || job.state === "running")) {
      await new Promise((resolve) => setTimeout(resolve, 2000))
      if (!alive.current) return
      job = (await getBatchScoreJob(job.id)) ?? job
    }
    if (!alive.current) return
    setRunning(false)
    if (job.state === "succeeded" && job.result) setResult(job.result)
    else setError(job.error ?? `Batch scoring ${job.state}.`)
  }

  const sampleColumns = Object.keys(result?.sample[0] ?? {})

  return (
    <SceneSection title="Batch score">
      <div className="space-y-3 rounded-xl border border-border bg-card p-4">
        <p className="text-xs text-text-tertiary">
          Score a whole dataset or a certified derivation&apos;s output with this model; the
          predictions land as a CSV artifact on an MLflow run.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1">
            <span className="text-xs font-medium">Data source</span>
            <SourceSelect sources={sources} value={source} onChange={setSource} />
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium">Version</span>
            <select
              className={fieldClass}
              value={version}
              onChange={(e) => setVersion(e.target.value)}
            >
              <option value="">
                served
                {detail.championVersion !== null ? ` (v${detail.championVersion})` : ""}
              </option>
              {detail.versions.map((v) => (
                <option key={v.version} value={String(v.version)}>
                  v{v.version}
                </option>
              ))}
            </select>
          </label>
        </div>
        <Button size="sm" disabled={running || source === ""} onClick={() => void run()}>
          {running && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {running ? "Scoring…" : "Run batch scoring"}
        </Button>
        {error && <p className="text-xs text-destructive">{error}</p>}
        {result && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <span>
                Scored <span className="font-medium">{result.nRows}</span> rows of{" "}
                <span className="font-mono">{result.dataset}</span> with v{result.version}.
              </span>
              <a
                href={batchScoreArtifactUrl(result.runId)}
                className="inline-flex items-center gap-1 text-xs text-muted-foreground transition hover:text-foreground"
              >
                <Download className="h-3.5 w-3.5" /> Download predictions.csv
              </a>
            </div>
            {Object.keys(result.stats).length > 0 && (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={headRowClass}>
                      {Object.keys(result.stats).map((k) => (
                        <th key={k} className={thClass}>
                          {k}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      {Object.entries(result.stats).map(([k, v]) => (
                        <td key={k} className="px-3 py-1.5 font-mono">
                          {formatNumber(v)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
            {result.sample.length > 0 && (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={headRowClass}>
                      {sampleColumns.map((k) => (
                        <th key={k} className={thClass}>
                          {k}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {result.sample.map((row) => (
                      <tr key={rowKey(row)}>
                        {sampleColumns.map((k) => (
                          <td key={k} className="whitespace-nowrap px-3 py-1.5 font-mono">
                            {cellText(row[k])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </SceneSection>
  )
}

// Drift checks plus the inference log: how the model behaves in production.
function MonitoringSection({ name }: { name: string }) {
  return (
    <SceneSection title="Monitoring">
      <div className="space-y-4">
        <DriftPanel name={name} />
        <InferenceLogPanel name={name} />
      </div>
    </SceneSection>
  )
}

function DriftBadge({ drifted }: { drifted: boolean }) {
  return drifted ? (
    <Badge variant="danger">drift detected</Badge>
  ) : (
    <Badge variant="success">no drift</Badge>
  )
}

function DriftPanel({ name }: { name: string }) {
  const [history, setHistory] = useState<DriftHistoryEntry[]>([])
  const [latest, setLatest] = useState<DriftResult | null>(null)
  const [notice, setNotice] = useState("")
  const [checking, setChecking] = useState(false)

  const loadHistory = useCallback(() => {
    void getDriftHistory(name).then(setHistory)
  }, [name])
  useEffect(loadHistory, [loadHistory])

  const check = async () => {
    setChecking(true)
    setNotice("")
    const r = await runDriftCheck(name)
    setChecking(false)
    // A 400 (e.g. not enough serving traffic yet) is a notice, not a page failure.
    if ("error" in r) {
      setNotice(r.error)
      return
    }
    setLatest(r)
    loadHistory()
  }

  return (
    <div className="space-y-3 rounded-xl border border-border bg-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium">Drift</div>
          <p className="text-xs text-text-tertiary">
            Compare recent serving traffic against the training data, column by column.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="shrink-0"
          disabled={checking}
          onClick={() => void check()}
        >
          {checking && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {checking ? "Checking…" : "Check drift now"}
        </Button>
      </div>
      {notice && (
        <p className="rounded-lg border border-warning/30 bg-warning-tint px-3 py-2 text-xs text-warning">
          {notice}
        </p>
      )}
      {latest && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <DriftBadge drifted={latest.datasetDrift} />
            <span className="text-xs text-muted-foreground">
              {latest.nDrifted} of {latest.nColumns} columns drifted (
              {Math.round(latest.shareDrifted * 100)}%) over {latest.nCurrentRows} recent rows,
              against v{latest.version}.
            </span>
            <a
              href={driftReportUrl(latest.runId)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs text-muted-foreground transition hover:text-foreground"
            >
              Full report <ExternalLink className="h-3 w-3" />
            </a>
          </div>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-xs">
              <thead>
                <tr className={headRowClass}>
                  <th className={thClass}>Column</th>
                  <th className={thClass}>Method</th>
                  <th className={thClass}>Score</th>
                  <th className={thClass}>Threshold</th>
                  <th className={thClass}>Drifted</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {latest.columns.map((c) => (
                  <tr key={c.column} className={c.drifted ? "bg-danger-tint/50" : ""}>
                    <td className="px-3 py-1.5 font-mono">{c.column}</td>
                    <td className="px-3 py-1.5 text-text-tertiary">{c.method}</td>
                    <td className="px-3 py-1.5 font-mono">{c.score.toFixed(4)}</td>
                    <td className="px-3 py-1.5 font-mono text-text-tertiary">
                      {c.threshold.toFixed(4)}
                    </td>
                    <td
                      className={`px-3 py-1.5 ${
                        c.drifted ? "font-medium text-danger" : "text-text-tertiary"
                      }`}
                    >
                      {c.drifted ? "yes" : "no"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {history.length > 0 ? (
        <div className="space-y-1">
          <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            History
          </div>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-xs">
              <thead>
                <tr className={headRowClass}>
                  <th className={thClass}>When</th>
                  <th className={thClass}>Version</th>
                  <th className={thClass}>Rows</th>
                  <th className={thClass}>Drifted columns</th>
                  <th className={thClass}>Verdict</th>
                  <th className={thClass}>Report</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {history.map((h) => (
                  <tr key={h.runId}>
                    <td className="whitespace-nowrap px-3 py-1.5 text-muted-foreground">
                      {new Date(h.at).toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 font-mono">v{h.version}</td>
                    <td className="px-3 py-1.5 font-mono">{h.nCurrentRows}</td>
                    <td className="px-3 py-1.5 font-mono">
                      {h.nDrifted} ({Math.round(h.shareDrifted * 100)}%)
                    </td>
                    <td className="px-3 py-1.5">
                      <DriftBadge drifted={h.datasetDrift} />
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5">
                      <a
                        href={driftReportUrl(h.runId)}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 text-muted-foreground transition hover:text-foreground"
                      >
                        Report <ExternalLink className="h-3 w-3" />
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        !latest && !notice && <p className="text-xs text-muted-foreground">No drift checks yet.</p>
      )}
    </div>
  )
}

const LOG_PAGE = 25

function InferenceLogPanel({ name }: { name: string }) {
  const [entries, setEntries] = useState<InferenceLogEntry[]>([])
  const [exhausted, setExhausted] = useState(false)
  const [loading, setLoading] = useState(true)

  const loadPage = useCallback(
    async (offset: number) => {
      setLoading(true)
      const page = await getInferenceLog(name, LOG_PAGE, offset)
      setEntries((cur) => (offset === 0 ? page : [...cur, ...page]))
      setExhausted(page.length < LOG_PAGE)
      setLoading(false)
    },
    [name],
  )
  useEffect(() => {
    setEntries([])
    void loadPage(0)
  }, [loadPage])

  return (
    <div className="space-y-3 rounded-xl border border-border bg-card p-4">
      <div>
        <div className="text-sm font-medium">Inference log</div>
        <p className="text-xs text-text-tertiary">
          Serving requests against this model, newest first.
        </p>
      </div>
      {entries.length === 0 ? (
        loading ? (
          <Skeleton className="h-4 w-48" />
        ) : (
          <p className="text-xs text-text-tertiary">No serving requests logged yet.</p>
        )
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-xs">
            <thead>
              <tr className={headRowClass}>
                <th className={thClass}>When</th>
                <th className={thClass}>Version</th>
                <th className={thClass}>Rows</th>
                <th className={thClass}>Latency</th>
                <th className={thClass}>Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {entries.map((e) => (
                <tr key={e.id}>
                  <td className="whitespace-nowrap px-3 py-1.5 text-muted-foreground">
                    {new Date(e.at).toLocaleString()}
                  </td>
                  <td className="px-3 py-1.5 font-mono">v{e.version}</td>
                  <td className="px-3 py-1.5 font-mono">{e.nRows}</td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono">
                    {Math.round(e.latencyMs)} ms
                  </td>
                  <td
                    className={`px-3 py-1.5 ${
                      e.status === "error" ? "text-danger" : "text-success"
                    }`}
                    title={e.error ?? undefined}
                  >
                    {e.status}
                    {e.error && (
                      <span className="block max-w-64 truncate text-[11px] text-danger/80">
                        {e.error}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!exhausted && entries.length > 0 && (
        <Button
          variant="outline"
          size="sm"
          disabled={loading}
          onClick={() => void loadPage(entries.length)}
        >
          {loading ? "Loading…" : "Load more"}
        </Button>
      )}
    </div>
  )
}

const BUDGET_UNITS = { seconds: 1, minutes: 60, hours: 3600 } as const
type BudgetUnit = keyof typeof BUDGET_UNITS
const MIN_BUDGET_SECONDS = 5
const MAX_BUDGET_SECONDS = 86400 // 24 hours

// Split a seconds budget into the largest unit that divides it evenly, for the form.
function splitBudget(seconds: number): { amount: string; unit: BudgetUnit } {
  if (seconds > 0 && seconds % 3600 === 0) return { amount: String(seconds / 3600), unit: "hours" }
  if (seconds > 0 && seconds % 60 === 0) return { amount: String(seconds / 60), unit: "minutes" }
  return { amount: String(seconds), unit: "seconds" }
}

// The automatic-retraining policy: when to retrain (on data change, or on an interval)
// and with what spec. The form prefills from the served version's training params.
function AutoRetrainSection({ detail }: { detail: RegisteredModelDetail }) {
  const targetId = useId()
  const fb = useFeedback()
  const [policy, setPolicy] = useState<RetrainPolicy | null | "loading">("loading")
  const [sources, setSources] = useState<FeatureSource[]>([])
  const [columns, setColumns] = useState<DatasetColumn[]>([])
  // The training source, as an encoded `kind:name` value ("" until chosen).
  const [source, setSource] = useState("")
  const [target, setTarget] = useState("")
  const [mode, setMode] = useState<"on_data_change" | "interval">("on_data_change")
  const [intervalHours, setIntervalHours] = useState("24")
  const [budget, setBudget] = useState("5")
  const [budgetUnit, setBudgetUnit] = useState<BudgetUnit>("minutes")
  const [enabled, setEnabled] = useState(true)
  const [saving, setSaving] = useState(false)

  const served = servedVersion(detail)

  const load = useCallback(() => {
    void getRetrainPolicy(detail.name).then(setPolicy)
  }, [detail.name])
  useEffect(load, [load])

  useEffect(() => {
    void getFeatureSources().then(setSources)
  }, [])

  const sourceKind = source === "" ? null : decodeSource(source).kind

  useEffect(() => {
    setColumns([])
    if (source === "") return
    const src = decodeSource(source)
    // Only bound datasets have a column listing; a derivation's target is free text.
    if (src.kind !== "dataset") return
    void getDatasetColumns(src.name).then(setColumns)
  }, [source])

  // Prefill from the stored policy, or from the served version's training lineage for
  // a first-time configuration.
  useEffect(() => {
    if (policy === "loading" || policy === null) return
    if (policy.configured) {
      setSource(encodeSource(policy.sourceKind, policy.dataset))
      setTarget(policy.target)
      setMode(policy.mode)
      setIntervalHours(String(policy.intervalHours || 24))
      const b = splitBudget(policy.timeBudget)
      setBudget(b.amount)
      setBudgetUnit(b.unit)
      setEnabled(policy.enabled)
    } else {
      const servedDerivation = served?.tags["elbi.feature_derivation"]
      const seeded = servedDerivation
        ? encodeSource("derivation", servedDerivation)
        : served?.params.dataset
          ? encodeSource("dataset", served.params.dataset)
          : ""
      setSource((cur) => cur || seeded)
      setTarget((cur) => cur || (served?.params.target ?? ""))
    }
  }, [policy, served])

  const budgetSeconds = Number(budget) * BUDGET_UNITS[budgetUnit]
  const budgetValid =
    Number.isFinite(budgetSeconds) &&
    budgetSeconds >= MIN_BUDGET_SECONDS &&
    budgetSeconds <= MAX_BUDGET_SECONDS
  const hoursValid =
    mode !== "interval" || (Number.isFinite(Number(intervalHours)) && Number(intervalHours) >= 1)
  const ready = source !== "" && target.trim() !== "" && budgetValid && hoursValid

  const save = async () => {
    setSaving(true)
    const src = decodeSource(source)
    const body: RetrainPolicyInput = {
      ...(src.kind === "derivation" ? { derivation: src.name } : { dataset: src.name }),
      target: target.trim(),
      time_budget: budgetSeconds,
      mode,
      enabled,
    }
    if (mode === "interval") body.interval_hours = Number(intervalHours)
    // Keep the parts of a stored policy this form does not edit.
    if (policy !== "loading" && policy !== null && policy.configured) {
      body.task = policy.task
      if (policy.metric) body.metric = policy.metric
      body.ensemble = policy.ensemble
      if (policy.engine) body.engine = policy.engine
      if (policy.features) body.features = policy.features
    }
    const ok = await putRetrainPolicy(detail.name, body)
    setSaving(false)
    fb.toast(
      ok ? "ok" : "error",
      ok ? "Retraining policy saved" : "Could not save the retraining policy",
    )
    if (ok) load()
  }

  const remove = async () => {
    if (
      !(await fb.confirm({
        title: "Remove the retraining policy?",
        body: "Automatic retraining stops for this model. Existing versions are untouched.",
        danger: true,
      }))
    )
      return
    const ok = await deleteRetrainPolicy(detail.name)
    fb.toast(ok ? "ok" : "error", ok ? "Retraining policy removed" : "Could not remove the policy")
    if (ok) setPolicy({ configured: false })
  }

  return (
    <SceneSection title="Auto-retrain">
      <div className="space-y-3 rounded-xl border border-border bg-card p-4">
        {policy === "loading" ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-64" />
            <Skeleton className="h-9 w-full" />
          </div>
        ) : policy === null ? (
          <p className="text-xs text-text-tertiary">Could not load the retraining policy.</p>
        ) : (
          <>
            <div className="flex items-start justify-between gap-3">
              <p className="text-xs text-text-tertiary">
                Retrain this model automatically and register the result as a new version.
              </p>
              {policy.configured ? (
                <Badge variant={policy.enabled ? "success" : "neutral"}>
                  {policy.enabled ? "enabled" : "configured, disabled"}
                </Badge>
              ) : (
                <Badge variant="neutral">not configured</Badge>
              )}
            </div>
            {policy.configured && (
              <p className="text-xs text-text-tertiary">
                Retrains from{" "}
                {policy.sourceKind === "derivation" ? "feature derivation" : "dataset"}{" "}
                <span className="font-mono text-foreground">{policy.dataset}</span> · Last run:{" "}
                {policy.lastRunAt ? new Date(policy.lastRunAt).toLocaleString() : "never"}
              </p>
            )}
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="text-xs font-medium">Data source</span>
                <SourceSelect
                  sources={sources}
                  value={source}
                  extra={source}
                  onChange={(value) => {
                    setSource(value)
                    setTarget("")
                  }}
                />
              </label>
              <label className="space-y-1" htmlFor={targetId}>
                <span className="text-xs font-medium">Target column</span>
                {sourceKind === "derivation" ? (
                  <input
                    id={targetId}
                    className={fieldClass}
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder="output column to predict"
                  />
                ) : (
                  <select
                    id={targetId}
                    className={fieldClass}
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    <option value="" disabled>
                      {source !== "" ? "Choose the target…" : "Pick a data source first"}
                    </option>
                    {target !== "" && !columns.some((c) => c.name === target) && (
                      <option value={target}>{target}</option>
                    )}
                    {columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                )}
              </label>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="text-xs font-medium">Trigger</span>
                <select
                  className={fieldClass}
                  value={mode}
                  onChange={(e) => setMode(e.target.value as "on_data_change" | "interval")}
                >
                  <option value="on_data_change">On data change</option>
                  <option value="interval">On an interval</option>
                </select>
              </label>
              {mode === "interval" && (
                <label className="space-y-1">
                  <span className="text-xs font-medium">Every (hours)</span>
                  <input
                    className={fieldClass}
                    type="number"
                    min={1}
                    value={intervalHours}
                    onChange={(e) => setIntervalHours(e.target.value)}
                  />
                </label>
              )}
            </div>
            <div className="space-y-1">
              <span className="text-xs font-medium">Search budget</span>
              <div className="grid grid-cols-2 gap-3">
                <input
                  className={fieldClass}
                  type="number"
                  min={1}
                  value={budget}
                  onChange={(e) => setBudget(e.target.value)}
                  aria-label="Budget amount"
                />
                <select
                  className={fieldClass}
                  value={budgetUnit}
                  onChange={(e) => setBudgetUnit(e.target.value as BudgetUnit)}
                  aria-label="Budget unit"
                >
                  {(Object.keys(BUDGET_UNITS) as BudgetUnit[]).map((u) => (
                    <option key={u} value={u}>
                      {u}
                    </option>
                  ))}
                </select>
              </div>
              {budget !== "" && !budgetValid && (
                <p className="text-xs text-destructive">
                  The training budget must be between 5 seconds and 24 hours.
                </p>
              )}
            </div>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
              />
              Enabled
            </label>
            <div className="flex items-center justify-end gap-2">
              {policy.configured && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive"
                  onClick={() => void remove()}
                >
                  <Trash2 className="h-3.5 w-3.5" /> Remove policy
                </Button>
              )}
              <Button size="sm" disabled={!ready || saving} onClick={() => void save()}>
                {saving ? "Saving…" : policy.configured ? "Save policy" : "Configure"}
              </Button>
            </div>
          </>
        )}
      </div>
    </SceneSection>
  )
}
