// The full-page training form (/models/train): pick a model name, a data source (a
// bound dataset or a certified feature derivation), a target and features, and a search
// budget, then hand the run to the job runner and return to /models, where the jobs
// polling shows its progress. `?name=<model>` prefills the form for training a new
// version, seeded with the served version's source and target.

import { Boxes } from "lucide-react"
import { useEffect, useId, useState } from "react"
import { Link, useNavigate, useSearchParams } from "react-router-dom"
import { FormControl, FormField } from "@/components/app/FormField"
import { Scene, SceneBody, SceneHeader, SceneSection } from "@/components/Scene"
import { rememberInlineJob } from "@/components/TrainingJobs"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  type DatasetColumn,
  type FeatureSource,
  getDatasetColumns,
  getFeatureSources,
  getRegisteredModel,
  type TrainRequest,
  trainModel,
} from "@/lib/chat"

// Radix forbids an empty item value; "auto" stands in for the unset metric ("" = let
// the backend choose) so the item list can include it as a real option.
const AUTO_METRIC = "auto"

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const TASKS: { value: NonNullable<TrainRequest["task"]>; label: string }[] = [
  { value: "auto", label: "auto" },
  { value: "classification", label: "classification" },
  { value: "regression", label: "regression" },
  { value: "ts_forecast", label: "Time-series forecast" },
]
const METRICS = ["accuracy", "roc_auc", "f1", "log_loss", "r2", "rmse", "mae"]

// One line explaining what each engine does with the budget/ensemble controls, shown
// under the picker so the choice is legible without leaving the form.
const ENGINE_HINTS: Record<NonNullable<TrainRequest["engine"]>, string> = {
  flaml: "Budget-aware search over the standard tabular learners.",
  autogluon: "Stacks a portfolio of models: best accuracy, longer budgets.",
  optuna: "Tunes gradient boosting with a TPE study across the search budget.",
  ensemble: "Fits several model families and blends them by greedy weighted selection.",
  tabicl:
    "Open pretrained foundation model: predicts in one pass, no per-dataset training; the budget and ensemble options don't apply. Best on small-to-mid data.",
}
const BUDGET_UNITS = { seconds: 1, minutes: 60, hours: 3600 } as const
type BudgetUnit = keyof typeof BUDGET_UNITS
const MAX_BUDGET_SECONDS = 86400 // 24 hours
const MIN_BUDGET_SECONDS = 5

type SourceKind = FeatureSource["kind"]

// The grouped select encodes both fields of a source in the option value; names never
// contain ":" but split on the first one regardless.
function encodeSource(kind: SourceKind, name: string): string {
  return `${kind}:${name}`
}

function decodeSource(value: string): { kind: SourceKind; name: string } {
  const i = value.indexOf(":")
  const kind = value.slice(0, i)
  return {
    kind: kind === "derivation" || kind === "training_set" ? (kind as SourceKind) : "dataset",
    name: value.slice(i + 1),
  }
}

// Parse a comma-separated feature list from the derivation-source free-text input.
function parseFeatureList(text: string): string[] {
  return text
    .split(",")
    .map((f) => f.trim())
    .filter((f) => f !== "")
}

export function TrainModelPage() {
  const ensembleId = useId()
  const featureIdPrefix = useId()
  const navigate = useNavigate()
  const [search] = useSearchParams()
  const presetName = search.get("name") ?? ""

  const [sources, setSources] = useState<FeatureSource[]>([])
  const [columns, setColumns] = useState<DatasetColumn[]>([])
  const [name, setName] = useState(presetName)
  // The training source: a bound dataset or a certified feature derivation.
  const [source, setSource] = useState<{ kind: SourceKind; name: string } | null>(null)
  const [target, setTarget] = useState("")
  // null means "all columns except the target": the default; a Set appears only once the user
  // customizes the selection. Dataset sources only.
  const [features, setFeatures] = useState<Set<string> | null>(null)
  // Derivation sources have no column listing; features arrive as free text instead.
  const [featuresText, setFeaturesText] = useState("")
  const [task, setTask] = useState<NonNullable<TrainRequest["task"]>>("auto")
  const [engine, setEngine] = useState<NonNullable<TrainRequest["engine"]>>("flaml")
  const [budget, setBudget] = useState("5")
  const [budgetUnit, setBudgetUnit] = useState<BudgetUnit>("minutes")
  const [metric, setMetric] = useState("")
  const [ensemble, setEnsemble] = useState(false)
  const [timeCol, setTimeCol] = useState("")
  const [horizon, setHorizon] = useState("")
  const [error, setError] = useState("")
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    void getFeatureSources().then(setSources)
  }, [])

  // Training a new version defaults to the served version's source and target, so a
  // retrain reproduces the same spec unless the user changes it.
  useEffect(() => {
    if (!presetName) return
    void getRegisteredModel(presetName).then((d) => {
      if (d === null || "unavailable" in d) return
      const served = d.versions.find((v) => v.version === d.championVersion) ?? d.versions[0]
      if (!served) return
      const derivation = served.tags["elbi.feature_derivation"]
      const kind = served.tags["elbi.source_kind"]
      const seeded: { kind: SourceKind; name: string } | null = derivation
        ? { kind: "derivation", name: derivation }
        : served.params.dataset
          ? {
              kind: kind === "training_set" ? "training_set" : "dataset",
              name: served.params.dataset,
            }
          : null
      setSource((current) => current ?? seeded)
      setTarget((current) => current || (served.params.target ?? ""))
    })
  }, [presetName])

  const sourceKind: SourceKind = source?.kind ?? "dataset"
  const sourceName = source?.name ?? ""

  useEffect(() => {
    setColumns([])
    if (sourceKind !== "dataset" || !sourceName) return
    void getDatasetColumns(sourceName).then(setColumns)
  }, [sourceKind, sourceName])

  const datasets = sources.filter((s) => s.kind === "dataset")
  const derivations = sources.filter((s) => s.kind === "derivation")
  const trainingSets = sources.filter((s) => s.kind === "training_set")
  // Datasets list their columns for pick-list target/features; derivations and training
  // sets do not, so those take free-text target and a comma-separated feature list.
  const freeText = sourceKind !== "dataset"

  const candidates = columns.filter((c) => c.name !== target)
  const selected = candidates.map((c) => c.name).filter((n) => features === null || features.has(n))
  const nameValid = NAME_RE.test(name.trim())
  const budgetSeconds = Number(budget) * BUDGET_UNITS[budgetUnit]
  const budgetValid =
    Number.isFinite(budgetSeconds) &&
    budgetSeconds >= MIN_BUDGET_SECONDS &&
    budgetSeconds <= MAX_BUDGET_SECONDS
  const horizonValid = Number.isInteger(Number(horizon)) && Number(horizon) >= 1
  const forecastReady = task !== "ts_forecast" || (timeCol.trim() !== "" && horizonValid)
  const featuresReady = freeText || selected.length > 0
  const ready =
    nameValid &&
    sourceName !== "" &&
    target.trim() !== "" &&
    budgetValid &&
    featuresReady &&
    forecastReady

  const pickSource = (value: string) => {
    setSource(decodeSource(value))
    setTarget("")
    setFeatures(null)
    setFeaturesText("")
    setTimeCol("")
  }

  const toggleFeature = (col: string) => {
    const next = new Set(features ?? candidates.map((c) => c.name))
    if (next.has(col)) next.delete(col)
    else next.add(col)
    setFeatures(next)
  }

  const submit = async () => {
    setSubmitting(true)
    setError("")
    const req: TrainRequest = {
      name: name.trim(),
      target: target.trim(),
      task,
      time_budget: budgetSeconds,
    }
    if (sourceKind === "derivation") req.derivation = sourceName
    else if (sourceKind === "training_set") req.training_set = sourceName
    else req.dataset = sourceName
    if (metric) req.metric = metric
    if (freeText) {
      const listed = parseFeatureList(featuresText)
      if (listed.length > 0) req.features = listed
    } else if (selected.length < candidates.length) {
      req.features = selected
    }
    if (ensemble) req.ensemble = true
    if (engine === "autogluon") req.engine = engine
    if (task === "ts_forecast") {
      req.time_col = timeCol.trim()
      req.horizon = Number(horizon)
    }
    const r = await trainModel(req)
    setSubmitting(false)
    if ("error" in r) {
      setError(r.error)
      return
    }
    // A backend without a job store trains inline and answers with a terminal job that
    // never reaches /api/jobs; park it so the models page can still show the outcome.
    if (r.job.id === null) {
      const now = Date.now() / 1000
      rememberInlineJob({
        id: `inline-${Date.now()}`,
        label: r.job.label,
        state: r.job.state,
        progress: "",
        result: r.job.result,
        error: r.job.error,
        createdAt: now,
        finishedAt: now,
      })
    }
    navigate("/models")
  }

  return (
    <Scene>
      <SceneHeader
        icon={<Boxes className="size-5" />}
        title={presetName ? "Train new version" : "Train model"}
        description={
          <>
            Fit a model on a dataset or a certified feature derivation with an AutoML search and
            register the result
            {presetName ? " as a new version." : " in the model registry."} Training runs in the
            background; progress shows on the models page.
          </>
        }
        backTo={presetName ? `/models/${encodeURIComponent(presetName)}` : "/models"}
        backLabel={presetName ? presetName : "Models"}
        actions={
          <>
            <Button size="sm" variant="outline" asChild>
              <Link to="/models">Cancel</Link>
            </Button>
            <Button size="sm" disabled={!ready || submitting} onClick={() => void submit()}>
              {submitting ? "Starting…" : "Start training"}
            </Button>
          </>
        }
      />
      <SceneBody width="default" className="space-y-8">
        <SceneSection title="Model">
          <FormField
            label="Model name"
            error={
              name.trim() !== "" && !nameValid ? (
                <>
                  1–64 characters, starting with a letter or digit; letters, digits and{" "}
                  <code className="font-mono">. _ -</code> only.
                </>
              ) : undefined
            }
          >
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="churn_predictor"
            />
          </FormField>
        </SceneSection>

        <SceneSection title="Data">
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Data source">
              <Select
                value={sourceName === "" ? undefined : encodeSource(sourceKind, sourceName)}
                onValueChange={pickSource}
              >
                <FormControl>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose a data source…" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {datasets.length > 0 && (
                    <SelectGroup>
                      <SelectLabel>Datasets</SelectLabel>
                      {datasets.map((s) => (
                        <SelectItem
                          key={encodeSource(s.kind, s.name)}
                          value={encodeSource(s.kind, s.name)}
                        >
                          {s.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                  {derivations.length > 0 && (
                    <SelectGroup>
                      <SelectLabel>Feature derivations</SelectLabel>
                      {derivations.map((s) => (
                        <SelectItem
                          key={encodeSource(s.kind, s.name)}
                          value={encodeSource(s.kind, s.name)}
                        >
                          {s.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                  {trainingSets.length > 0 && (
                    <SelectGroup>
                      <SelectLabel>Training sets</SelectLabel>
                      {trainingSets.map((s) => (
                        <SelectItem
                          key={encodeSource(s.kind, s.name)}
                          value={encodeSource(s.kind, s.name)}
                        >
                          {s.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                </SelectContent>
              </Select>
            </FormField>
            {freeText ? (
              <FormField label="Target column">
                <Input
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder="output column to predict"
                />
              </FormField>
            ) : (
              <FormField label="Target column">
                <Select
                  value={target || undefined}
                  disabled={columns.length === 0}
                  onValueChange={(v) => {
                    setTarget(v)
                    setFeatures(null)
                  }}
                >
                  <FormControl>
                    <SelectTrigger className="w-full">
                      <SelectValue
                        placeholder={sourceName ? "Choose the target…" : "Pick a data source first"}
                      />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {columns.map((c) => (
                      <SelectItem key={c.name} value={c.name}>
                        {c.name}
                        {c.numeric ? " (numeric)" : " (categorical)"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FormField>
            )}
          </div>
          {freeText ? (
            <FormField
              label="Features (optional)"
              hint="Target and features name columns of the source's output. Leave features empty to use every output column except the target; otherwise list them comma-separated."
            >
              <Input
                value={featuresText}
                onChange={(e) => setFeaturesText(e.target.value)}
                placeholder="recency, frequency, monetary"
              />
            </FormField>
          ) : (
            <div className="space-y-1">
              <span className="text-xs font-medium">Features</span>
              {target === "" ? (
                <p className="text-xs text-text-tertiary">
                  Pick a target first: every other column is a feature by default.
                </p>
              ) : (
                <>
                  <p className="text-xs text-text-tertiary">
                    {selected.length === candidates.length
                      ? `All ${candidates.length} columns except the target.`
                      : `${selected.length} of ${candidates.length} columns.`}
                  </p>
                  <div className="grid max-h-56 grid-cols-2 gap-x-3 gap-y-1 overflow-y-auto rounded-lg border border-border p-2">
                    {candidates.map((c) => (
                      <div key={c.name} className="flex min-w-0 items-center gap-2 text-xs">
                        <Checkbox
                          id={`${featureIdPrefix}-${c.name}`}
                          checked={features === null || features.has(c.name)}
                          onCheckedChange={() => toggleFeature(c.name)}
                        />
                        <Label
                          htmlFor={`${featureIdPrefix}-${c.name}`}
                          className="min-w-0 gap-2 text-xs font-normal cursor-pointer"
                        >
                          <span className="min-w-0 truncate font-mono">{c.name}</span>
                          {!c.numeric && <span className="shrink-0 text-text-tertiary">(cat)</span>}
                        </Label>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </SceneSection>

        <SceneSection title="Training">
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Task">
              <Select
                value={task}
                onValueChange={(v) => {
                  const next = v as NonNullable<TrainRequest["task"]>
                  setTask(next)
                  // AutoGluon does not support ts_forecast; fall back to FLAML.
                  if (next === "ts_forecast") setEngine("flaml")
                }}
              >
                <FormControl>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {TASKS.map((t) => (
                    <SelectItem key={t.value} value={t.value}>
                      {t.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FormField>
            <FormField label="Metric">
              <Select
                value={metric || AUTO_METRIC}
                onValueChange={(v) => setMetric(v === AUTO_METRIC ? "" : v)}
              >
                <FormControl>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value={AUTO_METRIC}>auto</SelectItem>
                  {METRICS.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FormField>
          </div>
          <FormField
            label="Engine"
            hint={
              task === "ts_forecast"
                ? "Time-series forecasting uses FLAML."
                : ENGINE_HINTS[engine ?? "flaml"]
            }
          >
            <Select
              value={engine}
              onValueChange={(v) => setEngine(v as NonNullable<TrainRequest["engine"]>)}
            >
              <FormControl>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
              </FormControl>
              <SelectContent>
                {/* Only FLAML forecasts; every other engine is disabled for ts_forecast. */}
                <SelectItem value="flaml">FLAML: fast, budget-aware (default)</SelectItem>
                <SelectItem value="autogluon" disabled={task === "ts_forecast"}>
                  AutoGluon: maximum accuracy, longer budgets
                </SelectItem>
                <SelectItem value="optuna" disabled={task === "ts_forecast"}>
                  Optuna: Bayesian tuning of gradient boosting
                </SelectItem>
                <SelectItem value="ensemble" disabled={task === "ts_forecast"}>
                  Ensemble: diverse models blended (often most accurate)
                </SelectItem>
                <SelectItem value="tabicl" disabled={task === "ts_forecast"}>
                  TabICL: open pretrained foundation model (small–mid data)
                </SelectItem>
              </SelectContent>
            </Select>
          </FormField>
          {task === "ts_forecast" && (
            <div className="grid grid-cols-2 gap-3">
              <FormField label="Time column">
                {sourceKind === "derivation" ? (
                  <Input
                    value={timeCol}
                    onChange={(e) => setTimeCol(e.target.value)}
                    placeholder="timestamp output column"
                  />
                ) : (
                  <Select
                    value={timeCol || undefined}
                    disabled={columns.length === 0}
                    onValueChange={setTimeCol}
                  >
                    <FormControl>
                      <SelectTrigger className="w-full">
                        <SelectValue
                          placeholder={
                            sourceName ? "Choose the time column…" : "Pick a data source first"
                          }
                        />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {columns
                        .filter((c) => c.name !== target)
                        .map((c) => (
                          <SelectItem key={c.name} value={c.name}>
                            {c.name}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                )}
              </FormField>
              <FormField
                label="Horizon"
                error={
                  horizon !== "" && !horizonValid
                    ? "The horizon must be a whole number of at least 1."
                    : undefined
                }
              >
                <Input
                  type="number"
                  min={1}
                  step={1}
                  value={horizon}
                  onChange={(e) => setHorizon(e.target.value)}
                  placeholder="periods to forecast"
                />
              </FormField>
            </div>
          )}
          <div className="flex items-start gap-2">
            <Checkbox
              id={ensembleId}
              checked={ensemble}
              onCheckedChange={(c) => setEnsemble(c === true)}
              className="mt-0.5"
            />
            <Label htmlFor={ensembleId} className="block gap-0 text-xs font-normal cursor-pointer">
              <span className="font-medium">Ensemble</span>
              <span className="block text-text-tertiary">
                Stack the best models found into an ensemble: slower to train, often more accurate.
              </span>
            </Label>
          </div>
          <div className="space-y-1">
            <span className="text-xs font-medium">Search budget</span>
            <div className="grid grid-cols-2 gap-3">
              <Input
                type="number"
                min={1}
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                aria-label="Budget amount"
              />
              <Select value={budgetUnit} onValueChange={(v) => setBudgetUnit(v as BudgetUnit)}>
                <SelectTrigger className="w-full" aria-label="Budget unit">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(Object.keys(BUDGET_UNITS) as BudgetUnit[]).map((u) => (
                    <SelectItem key={u} value={u}>
                      {u}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {budget !== "" && !budgetValid ? (
              <p className="text-xs text-danger">
                The training budget must be between 5 seconds and 24 hours.
              </p>
            ) : (
              <p className="text-xs text-text-tertiary">
                How long the AutoML search may run. Longer budgets try more models.
              </p>
            )}
          </div>
        </SceneSection>

        {error && <p className="text-sm text-danger">{error}</p>}
      </SceneBody>
    </Scene>
  )
}
