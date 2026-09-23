// Training-job tracking for the models pages. The jobs live in /api/jobs (the durable store),
// so the hook polls that and derives what to show: a remount (navigating between the list and a
// model page, or away and back mid-training) resumes tracking instead of orphaning a
// component-local snapshot, which is how a finished or failed job used to go unnoticed until a
// page refresh.

import { Check, Loader2, X } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

import { IconButton } from "@/components/app/IconButton"
import { getTrainingJobs, type TrainingJob } from "@/lib/chat"

const ACTIVE = new Set<TrainingJob["state"]>(["queued", "running"])
// Terminal jobs stay visible this long after finishing, so a result that landed while the user
// was elsewhere is still shown (with its error, for a failure) on return.
const RECENT_SECONDS = 300
// Session-wide dismissals, so a dismissed banner stays dismissed across remounts.
const dismissed = new Set<string>()
// A backend without a job store trains inline; its terminal result never appears in
// /api/jobs, so the train page parks it here for the next mounted hook to show.
let inlineJob: TrainingJob | null = null

export function rememberInlineJob(job: TrainingJob): void {
  inlineJob = job
}

export function isActiveTraining(job: TrainingJob): boolean {
  return ACTIVE.has(job.state)
}

// Poll the durable training jobs (~2s) and report the ones worth showing: active jobs
// plus recent terminal ones, minus dismissals. `onFinished` fires once per job this
// instance watched reach a terminal state, so the caller can refetch its model data.
// `model` narrows the view to one model's jobs (for its detail page).
export function useTrainingJobs(
  onFinished: () => void,
  model?: string,
): { jobs: TrainingJob[]; dismiss: (id: string) => void } {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  // Dismissals live module-wide; this counter just re-renders the hook's owner.
  const [, setDismissals] = useState(0)
  const finished = useRef(onFinished)
  useEffect(() => {
    finished.current = onFinished
  })
  // The states this instance has seen, so a completion notifies exactly once.
  const seen = useRef(new Map<string, TrainingJob["state"]>())

  useEffect(() => {
    let alive = true
    const poll = async () => {
      const all = await getTrainingJobs()
      // A transient fetch failure (null) skips this tick and keeps the last good
      // state; the interval itself always survives to try again.
      if (!alive || all === null) return
      setJobs(all)
      let completed = false
      for (const job of all) {
        const prior = seen.current.get(job.id)
        seen.current.set(job.id, job.state)
        if (prior !== undefined && ACTIVE.has(prior) && !ACTIVE.has(job.state)) {
          completed = true
        }
      }
      if (completed) finished.current()
    }
    void poll()
    const timer = setInterval(() => void poll(), 2000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  const dismiss = useCallback((id: string) => {
    dismissed.add(id)
    if (inlineJob?.id === id) inlineJob = null
    setDismissals((n) => n + 1)
  }, [])

  const now = Date.now() / 1000
  const shown = (inlineJob ? [inlineJob, ...jobs] : jobs).filter(
    (j) =>
      (model === undefined || j.label === `train model ${model}`) &&
      !dismissed.has(j.id) &&
      (ACTIVE.has(j.state) || (j.finishedAt !== null && now - j.finishedAt < RECENT_SECONDS)),
  )
  return { jobs: shown, dismiss }
}

// The banner for one training job: a spinner while it runs, the registered version and
// its held-out metrics on success, the job's error prominently on failure.
export function TrainingStatus({ job, onDismiss }: { job: TrainingJob; onDismiss: () => void }) {
  const active = ACTIVE.has(job.state)
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-border px-4 py-3 text-sm">
      {active ? (
        <>
          <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-text-tertiary" />
          <div className="min-w-0 flex-1">
            <span className="font-medium">{job.label}</span>{" "}
            <span className="text-text-tertiary">training…</span>
            {job.progress && (
              <div className="truncate text-xs text-text-tertiary">{job.progress}</div>
            )}
          </div>
        </>
      ) : job.state === "succeeded" && job.result ? (
        <>
          <Check className="mt-0.5 size-4 shrink-0 text-verified" />
          <div className="min-w-0 flex-1">
            <span>
              Registered <span className="font-mono text-foreground/90">{job.result.name}</span> v
              {job.result.version}, {job.result.bestEstimator}
              {job.result.champion && " (champion)"}
            </span>
            {Object.keys(job.result.metrics).length > 0 && (
              <div className="truncate font-mono text-xs text-text-tertiary">
                {Object.entries(job.result.metrics)
                  .map(([k, v]) => `${k.replace(/^holdout_/, "")} ${v.toFixed(4)}`)
                  .join(" · ")}
              </div>
            )}
          </div>
        </>
      ) : (
        <>
          <X className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
          <div className="min-w-0 flex-1">
            <span className="font-medium text-destructive">
              {job.label} {job.state}
            </span>
            {job.error && <div className="text-xs text-destructive/90">{job.error}</div>}
          </div>
        </>
      )}
      {!active && (
        // Same footprint as the old padding-less button: icon-xs sized down to the
        // 16px icon itself (size-4 overrides icon-xs's default 24px).
        <IconButton
          label="Dismiss"
          size="icon-xs"
          className="size-4 text-text-tertiary hover:text-foreground"
          onClick={onDismiss}
        >
          <X className="h-4 w-4" />
        </IconButton>
      )}
    </div>
  )
}
