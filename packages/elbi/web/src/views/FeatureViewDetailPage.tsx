import { Activity, Clock, Layers, RefreshCw } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { FormField } from "@/components/app/FormField"
import {
  Scene,
  SceneBody,
  SceneHeader,
  ScenePanelLabel,
  SceneSection,
  SceneSkeleton,
} from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { VerdictBadge } from "@/components/VerdictBadge"
import {
  checkDrift,
  checkExpectations,
  type DriftCheck,
  type FeatureViewDetail,
  getFeatureViewDetail,
  historicalFeatures,
  materializeFeatures,
  onlineFeatures,
  type Row,
  type StatisticsSnapshot,
  setContract,
  snapshotStatistics,
  suggestContract,
} from "@/lib/features"
import { getImpact } from "@/lib/lineage"
import { EMPTY, relativeTime } from "@/lib/utils"

type Consumers = { models: string[]; trainingSets: string[] }

export function FeatureViewDetailPage() {
  const { name = "" } = useParams()
  const navigate = useNavigate()
  const [d, setD] = useState<FeatureViewDetail | null | "missing">(null)
  const [consumers, setConsumers] = useState<Consumers>({
    models: [],
    trainingSets: [],
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    let detail: FeatureViewDetail
    try {
      detail = await getFeatureViewDetail(name)
    } catch {
      setD("missing")
      return
    }
    setD(detail)
    try {
      const impact = await getImpact(`feature_view:${name}`)
      setConsumers({
        models: impact.affected.model ?? [],
        trainingSets: impact.affected.training_set ?? [],
      })
    } catch {
      setConsumers({ models: [], trainingSets: [] })
    }
  }, [name])

  useEffect(() => {
    void load()
  }, [load])

  if (d === null) return <SceneSkeleton />
  if (d === "missing")
    return (
      <Scene>
        <SceneHeader backTo="/features" backLabel="Feature store" title="Not found" />
        <SceneBody>
          <p className="text-sm text-text-secondary">
            No feature view named <span className="font-mono">{name}</span>.
          </p>
        </SceneBody>
      </Scene>
    )

  const run = (fn: () => Promise<unknown>) => async () => {
    setBusy(true)
    setError(null)
    try {
      await fn()
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const stats = d.statistics[0] ?? null
  const drift = d.drift[0] ?? null
  const expectation = d.expectations[0] ?? null
  const hasBaseline = d.statistics.some((s) => s.isBaseline)

  return (
    <Scene>
      <SceneHeader
        icon={<Layers className="size-5" />}
        mono
        title={d.name}
        backTo="/features"
        backLabel="Feature store"
        badges={
          <Badge variant={d.certified ? "verified" : "warning"}>
            {d.certified ? "certified source" : "uncertified source"}
          </Badge>
        }
        actions={
          <Button size="sm" onClick={run(() => materializeFeatures([d.name]))} disabled={busy}>
            <RefreshCw className={busy ? "size-4 animate-spin" : "size-4"} />
            Materialize
          </Button>
        }
      />
      <SceneBody
        aside={
          <>
            <ScenePanelLabel label="Source derivation">
              <span className="font-mono text-compact">{d.source}</span>
            </ScenePanelLabel>
            <ScenePanelLabel label="Entities / keys">
              <span className="font-mono text-compact">[{d.joinKeys.join(", ")}]</span>
            </ScenePanelLabel>
            {d.timestampField ? (
              <ScenePanelLabel label="Point-in-time">
                <span className="flex items-center gap-1.5">
                  <Clock className="size-3.5 text-text-tertiary" />
                  {d.timestampField}
                  {d.ttlSeconds ? ` · TTL ${Math.round(d.ttlSeconds / 86400)}d` : ""}
                </span>
              </ScenePanelLabel>
            ) : null}
            <ScenePanelLabel label="Online store">
              {d.nOnlineKeys > 0 ? (
                <span>
                  {d.nOnlineKeys} keys · {relativeTime(d.lastMaterializedAt)}
                </span>
              ) : (
                <span className="text-text-tertiary">not materialized</span>
              )}
            </ScenePanelLabel>
          </>
        }
      >
        {error ? <p className="text-sm text-danger">{error}</p> : null}

        <SceneSection
          title="Schema"
          description={
            d.features.length
              ? undefined
              : "No features declared: every non-key, non-timestamp source column is exposed."
          }
        >
          {d.features.length ? (
            <div className="overflow-hidden rounded-lg border border-border bg-card">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="py-1.5">Feature</TableHead>
                    <TableHead className="py-1.5">Type</TableHead>
                    <TableHead className="py-1.5">Description</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {d.features.map((f) => (
                    <TableRow key={f.name}>
                      <TableCell className="py-1.5 font-mono font-medium">{f.name}</TableCell>
                      <TableCell className="py-1.5 text-text-secondary">
                        {f.dtype ?? EMPTY}
                      </TableCell>
                      <TableCell className="py-1.5 text-text-secondary">
                        {f.description ?? EMPTY}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : null}
        </SceneSection>

        <LookupSection view={d} />

        <SceneSection
          title="Statistics"
          action={
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={run(() => snapshotStatistics(d.name, false))}
              >
                Snapshot
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={run(() => snapshotStatistics(d.name, true))}
              >
                Set baseline
              </Button>
            </div>
          }
        >
          {stats ? (
            <StatsTable snapshot={stats} />
          ) : (
            <p className="text-sm text-text-tertiary">
              No snapshot yet. Take one to profile this view's features.
            </p>
          )}
        </SceneSection>

        <SceneSection
          title="Drift"
          action={
            <Button
              size="sm"
              disabled={busy || !hasBaseline}
              title={hasBaseline ? undefined : "Set a baseline snapshot first"}
              onClick={run(() => checkDrift(d.name))}
            >
              <Activity className="size-4" />
              Check drift
            </Button>
          }
        >
          {drift ? (
            <DriftView check={drift} />
          ) : (
            <p className="text-sm text-text-tertiary">
              {hasBaseline
                ? "No drift check yet."
                : "Set a baseline snapshot to enable drift checks."}
            </p>
          )}
        </SceneSection>

        <SceneSection
          title="Expectations"
          description="A data contract on the view's feature values, checked with the oracle's verdict."
          action={
            d.hasContract ? (
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={run(() => checkExpectations(d.name))}
              >
                Check contract
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={run(async () => {
                  await setContract(d.name, await suggestContract(d.name))
                  await checkExpectations(d.name)
                })}
              >
                Suggest &amp; attach contract
              </Button>
            )
          }
        >
          {expectation ? (
            <div className="flex items-center gap-2 text-sm text-text-secondary">
              <VerdictBadge verdict={expectation.verdict} />
              <span>
                {expectation.nViolated}/{expectation.nClauses} clauses failing ·{" "}
                {expectation.rowCount} rows
              </span>
            </div>
          ) : (
            <p className="text-sm text-text-tertiary">
              {d.hasContract
                ? "A contract is attached. Check it against the current values."
                : "No contract yet. Suggest one from a profile of the current values."}
            </p>
          )}
        </SceneSection>

        <SceneSection
          title="Consuming models"
          description="Models trained on this view's features, via its training sets."
        >
          {consumers.models.length || consumers.trainingSets.length ? (
            <div className="space-y-2 text-sm">
              {consumers.trainingSets.length ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-text-tertiary">Training sets:</span>
                  {consumers.trainingSets.map((t) => (
                    <Badge key={t} variant="neutral">
                      {t}
                    </Badge>
                  ))}
                </div>
              ) : null}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-text-tertiary">Models:</span>
                {consumers.models.length ? (
                  consumers.models.map((m) => (
                    <Button
                      key={m}
                      variant="ghost"
                      className="h-auto p-0 transition-opacity hover:bg-transparent hover:opacity-80 dark:hover:bg-transparent"
                      onClick={() => navigate(`/models/${encodeURIComponent(m)}`)}
                    >
                      <Badge variant="info">{m}</Badge>
                    </Button>
                  ))
                ) : (
                  <span className="text-text-tertiary">none yet</span>
                )}
              </div>
            </div>
          ) : (
            <p className="text-sm text-text-tertiary">
              No models consume this view yet. Create a training set from it, then train a model on
              that set.
            </p>
          )}
        </SceneSection>
      </SceneBody>
    </Scene>
  )
}

function StatsTable({ snapshot }: { snapshot: StatisticsSnapshot }) {
  return (
    <div>
      <div className="mb-1 text-xs text-text-tertiary">
        {snapshot.rowCount} rows
        {snapshot.isBaseline ? " · baseline" : ""} · {new Date(snapshot.at).toLocaleString()}
      </div>
      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <Table className="text-xs">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="py-1.5">Feature</TableHead>
              <TableHead className="py-1.5 text-right">Complete</TableHead>
              <TableHead className="py-1.5 text-right">Distinct</TableHead>
              <TableHead className="py-1.5 text-right">Range</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {snapshot.features.map((f) => (
              <TableRow key={f.name}>
                <TableCell className="py-1 font-mono font-medium">{f.name}</TableCell>
                <TableCell className="py-1 text-right">
                  {Math.round(f.completeness * 100)}%
                </TableCell>
                <TableCell className="py-1 text-right">{f.distinct}</TableCell>
                <TableCell className="py-1 text-right text-text-tertiary">
                  {f.minimum === null ? EMPTY : `${f.minimum} … ${f.maximum}`}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

function DriftView({ check }: { check: DriftCheck }) {
  return (
    <div>
      <div className="mb-1.5 flex flex-wrap items-center gap-2 text-xs text-text-tertiary">
        <Badge variant={check.datasetDrift ? "danger" : "success"}>
          {check.datasetDrift ? "dataset drift" : "no drift"}
        </Badge>
        <span>
          {check.nDrifted}/{check.nColumns} features · {check.nCurrentRows} rows ·{" "}
          {new Date(check.at).toLocaleString()}
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {check.columns.map((c) => (
          <Badge key={c.column} variant={c.drifted ? "danger" : "neutral"}>
            {c.column}
          </Badge>
        ))}
      </div>
    </div>
  )
}

function LookupSection({ view }: { view: FeatureViewDetail }) {
  const [key, setKey] = useState("")
  const [mode, setMode] = useState<"online" | "historical">("online")
  const [timestamp, setTimestamp] = useState("")
  const [result, setResult] = useState<Row | null>(null)
  const [error, setError] = useState<string | null>(null)

  const joinKey = view.joinKeys[0] ?? "id"
  const refs = view.features.map((f) => `${view.name}:${f.name}`)

  const run = async () => {
    setError(null)
    try {
      const entity: Row = { [joinKey]: key }
      const rows =
        mode === "online"
          ? await onlineFeatures(refs, [entity])
          : await historicalFeatures(refs, [{ ...entity, event_timestamp: timestamp }])
      setResult(rows[0] ?? null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <SceneSection title="Lookup" description="Read this view's features for one entity key.">
      <div className="space-y-3 rounded-lg border border-border bg-card p-4">
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex gap-1">
            <Button
              variant={mode === "online" ? "secondary" : "ghost"}
              size="sm"
              aria-pressed={mode === "online"}
              onClick={() => setMode("online")}
            >
              Online (latest)
            </Button>
            {view.timestampField ? (
              <Button
                variant={mode === "historical" ? "secondary" : "ghost"}
                size="sm"
                aria-pressed={mode === "historical"}
                onClick={() => setMode("historical")}
              >
                Point-in-time
              </Button>
            ) : null}
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <FormField label={joinKey}>
            <Input value={key} onChange={(e) => setKey(e.target.value)} className="h-8 w-40" />
          </FormField>
          {mode === "historical" ? (
            <FormField label="As of (event_timestamp)">
              <Input
                value={timestamp}
                onChange={(e) => setTimestamp(e.target.value)}
                placeholder="2024-03-15"
                className="h-8 w-40"
              />
            </FormField>
          ) : null}
          <Button size="sm" onClick={() => void run()} disabled={!key.trim()}>
            Fetch
          </Button>
        </div>
        {result ? (
          <pre className="overflow-auto rounded-lg border border-border bg-surface-secondary p-3 font-mono text-xs">
            {JSON.stringify(result, null, 2)}
          </pre>
        ) : null}
        {error ? <p className="text-sm text-danger">{error}</p> : null}
      </div>
    </SceneSection>
  )
}
