// What compute cost, and who used it.
//
// Reported rather than enforced, and the copy says so: Databricks is candid that its own
// compute spend limits are notification-only and that they do "not proactively terminate
// resources to maintain the limit". Ending a session to recover an overage destroys work
// to save cents, so a limit here raises an alert and nothing more.

import { AlertTriangle } from "lucide-react"
import { useEffect, useState } from "react"

import { type ComputeUsageReport, getComputeUsage } from "@/lib/notebooks"

const WINDOWS = [
  { days: 7, label: "7 days" },
  { days: 30, label: "30 days" },
  { days: 90, label: "90 days" },
] as const

const money = (value: number) => `$${value.toFixed(2)}`

const duration = (seconds: number) => {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

export function ComputeUsageSection() {
  const [days, setDays] = useState<number>(30)
  const [report, setReport] = useState<ComputeUsageReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setError(null)
    void getComputeUsage(days)
      .then((next) => {
        if (!cancelled) setReport(next)
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause))
      })
    return () => {
      cancelled = true
    }
  }, [days])

  // Grouped by (profile, kind): the same size costs the same per second whether a person
  // or a schedule asked for it, and that is exactly why they have to be told apart.
  const grouped = new Map<
    string,
    { profile: string; kind: string; seconds: number; cost: number; runs: number }
  >()
  for (const session of report?.sessions ?? []) {
    const kind = session.kind || "interactive"
    const key = `${session.profile}\u0000${kind}`
    const row = grouped.get(key) ?? { profile: session.profile, kind, seconds: 0, cost: 0, runs: 0 }
    row.seconds += session.seconds
    row.cost += session.cost
    row.runs += 1
    grouped.set(key, row)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold">Compute usage</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Attributed when a kernel ends. An estimate for comparing sizes and spotting growth, not
            a bill &mdash; the rates come from your deployment&rsquo;s configuration.
          </p>
        </div>
        <div className="flex shrink-0 gap-1">
          {WINDOWS.map((window) => (
            <button
              key={window.days}
              type="button"
              onClick={() => setDays(window.days)}
              className={`rounded-md px-2 py-1 text-xs ${
                days === window.days
                  ? "bg-primary/10 font-medium text-primary"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {window.label}
            </button>
          ))}
        </div>
      </div>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      {report?.alert ? (
        <p className="flex gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs text-amber-700 dark:text-amber-400">
          <AlertTriangle className="mt-px size-4 shrink-0" />
          <span>{report.alert}</span>
        </p>
      ) : null}

      {report ? (
        <div className="flex flex-wrap gap-6 rounded-lg border border-border p-3">
          <Figure label="Spend" value={money(report.spend)} />
          <Figure label="Sessions" value={String(report.sessions.length)} />
          <Figure label="Limit" value={report.limit === null ? "none set" : money(report.limit)} />
          {/* Split out because they mean different things: batch is work somebody
              scheduled, interactive is work somebody is doing, and only the second is
              worth chasing when it sits idle. */}
          <Figure label="Interactive" value={money(report.byKind?.interactive ?? 0)} />
          <Figure label="Scheduled" value={money(report.byKind?.batch ?? 0)} />
        </div>
      ) : null}

      {grouped.size > 0 ? (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-xs">
            <thead className="bg-muted/50 text-left text-muted-foreground">
              <tr>
                <th className="px-3 py-2 font-medium">Profile</th>
                <th className="px-3 py-2 font-medium">Kind</th>
                <th className="px-3 py-2 font-medium">Sessions</th>
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="px-3 py-2 text-right font-medium">Cost</th>
              </tr>
            </thead>
            <tbody>
              {[...grouped.entries()]
                .sort((a, b) => b[1].cost - a[1].cost)
                .map(([key, row]) => (
                  <tr key={key} className="border-t border-border">
                    <td className="px-3 py-2 font-medium">{row.profile}</td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {row.kind === "batch" ? "scheduled" : "interactive"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{row.runs}</td>
                    <td className="px-3 py-2 text-muted-foreground">{duration(row.seconds)}</td>
                    <td className="px-3 py-2 text-right">{money(row.cost)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : report ? (
        <p className="text-sm text-muted-foreground">No compute sessions in this window.</p>
      ) : null}
    </div>
  )
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  )
}
