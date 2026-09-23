// A compact strip of background training jobs. Polls /api/jobs and shows anything
// active plus anything that finished recently, so the user can watch a long derivation
// train, certify, or be cancelled without holding the chat turn open. Renders nothing
// when there are no jobs to show, so it stays out of the way. When a certified job
// newly finishes, it fires `onCertifiedComplete` once, so the chat can pull in the
// agent's verified follow-up answer.
import { useEffect, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import { cancelJob, getJobs, type Job } from "@/lib/chat"

const ACTIVE = new Set(["queued", "running"])
const RECENT_SECONDS = 300

function stateLabel(job: Job): string {
  if (job.state === "succeeded") return job.result?.certified ? "certified" : "done"
  return job.state
}

function stateClass(job: Job): string {
  if (job.state === "failed") return "text-danger"
  if (job.state === "cancelled") return "text-text-tertiary"
  if (job.state === "succeeded")
    return job.result?.certified ? "text-verified" : "text-text-tertiary"
  return "text-info"
}

export function JobsBar({ onCertifiedComplete }: { onCertifiedComplete?: () => void }) {
  const [jobs, setJobs] = useState<Job[]>([])
  // Certified jobs already announced, so the follow-up is pulled in exactly once.
  const announced = useRef<Set<string>>(new Set())
  // Read the callback through a ref so the poll effect can run with an empty dependency
  // array: the interval is set up once on mount and never torn down and recreated when
  // the parent re-renders (which it does on every streaming token), so /api/jobs is
  // polled on the interval, not on nearly every render.
  const onCertified = useRef(onCertifiedComplete)
  useEffect(() => {
    onCertified.current = onCertifiedComplete
  })

  useEffect(() => {
    let alive = true
    const poll = async () => {
      const all = await getJobs()
      if (!alive) return
      setJobs(all)
      for (const job of all) {
        if (job.state === "succeeded" && job.result?.certified && !announced.current.has(job.id)) {
          announced.current.add(job.id)
          onCertified.current?.()
        }
      }
    }
    void poll()
    const id = setInterval(() => void poll(), 3000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const now = Date.now() / 1000
  const shown = jobs.filter(
    (j) => ACTIVE.has(j.state) || (j.finishedAt != null && now - j.finishedAt < RECENT_SECONDS),
  )
  if (shown.length === 0) return null

  return (
    <div className="flex flex-col gap-1 border-b border-border bg-surface-secondary px-4 py-2">
      {shown.map((job) => (
        <div key={job.id} className="flex items-center gap-2 text-sm">
          <span className="font-medium">{job.label}</span>
          <span className="text-text-tertiary">training in the background</span>
          <span className={`font-medium ${stateClass(job)}`}>{stateLabel(job)}</span>
          {job.progress && ACTIVE.has(job.state) && (
            <span className="truncate text-xs text-text-tertiary">{job.progress}</span>
          )}
          {ACTIVE.has(job.state) && (
            <Button
              variant="link"
              onClick={() => void cancelJob(job.id)}
              aria-label={`Cancel ${job.label}`}
              className="ml-auto h-auto p-0 text-xs font-normal text-text-tertiary underline underline-offset-auto hover:text-foreground"
            >
              Cancel
            </Button>
          )}
        </div>
      ))}
    </div>
  )
}
