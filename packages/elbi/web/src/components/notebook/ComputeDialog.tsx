// What a notebook runs on. The menu is defined by whoever runs the deployment, so this
// is a picker over it and never an editor of it -- a field offered here that could not
// be saved would be worse than not showing it at all.

import { AlertTriangle, Check, Cpu, Zap } from "lucide-react"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  type ComputeMenu,
  type ComputeProfile,
  getComputeProfiles,
  getNotebookCompute,
  type NotebookCompute,
  updateNotebook,
} from "@/lib/notebooks"

const money = (value: number) =>
  value >= 1 ? `$${value.toFixed(2)}/hr` : `${(value * 100).toFixed(1)}¢/hr`

const duration = (seconds: number) =>
  seconds >= 3600 ? `${Math.round(seconds / 3600)}h` : `${Math.round(seconds / 60)}m`

export function ComputeDialog({
  notebookId,
  open,
  onOpenChange,
  onChanged,
}: {
  notebookId: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onChanged?: () => void
}) {
  const [menu, setMenu] = useState<ComputeMenu | null>(null)
  const [state, setState] = useState<NotebookCompute | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setError(null)
    void Promise.all([getComputeProfiles(), getNotebookCompute(notebookId)])
      .then(([profiles, current]) => {
        if (cancelled) return
        setMenu(profiles)
        setState(current)
        setSelected(current.profile.name)
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause))
      })
    return () => {
      cancelled = true
    }
  }, [open, notebookId])

  const save = async () => {
    if (!selected || selected === state?.profile.name) return onOpenChange(false)
    setSaving(true)
    setError(null)
    try {
      await updateNotebook(notebookId, { compute_profile: selected })
      onChanged?.()
      onOpenChange(false)
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Compute</DialogTitle>
          <DialogDescription>
            The machine this notebook's kernel runs on. Sizes are defined by your deployment. A
            change applies the next time the kernel starts.
          </DialogDescription>
        </DialogHeader>

        {state?.unavailable ? (
          <p className="flex gap-2 rounded-md border border-warning/40 bg-warning-tint p-2.5 text-xs text-warning">
            <AlertTriangle className="mt-px size-4 shrink-0" />

            <span>
              This notebook asked for a size that is no longer offered, so it is using the default.{" "}
              {state.unavailable}
            </span>
          </p>
        ) : null}

        {state?.drift ? (
          <p className="flex gap-2 rounded-md border border-warning/40 bg-warning-tint p-2.5 text-xs text-warning">
            <AlertTriangle className="mt-px size-4 shrink-0" />
            <span>{state.drift}</span>
          </p>
        ) : null}

        <div className="max-h-80 space-y-1.5 overflow-y-auto">
          {menu?.profiles.map((profile) => (
            <ProfileRow
              key={profile.name}
              profile={profile}
              selected={profile.name === selected}
              isDefault={profile.name === menu.default}
              running={state?.status === "running" && profile.name === state.profile.name}
              onSelect={() => setSelected(profile.name)}
            />
          ))}
          {menu && menu.profiles.length === 0 ? (
            <p className="text-sm text-text-tertiary">No compute profiles are available to you.</p>
          ) : null}
        </div>

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => void save()} disabled={saving || !selected}>
            {saving ? "Saving…" : "Use this size"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ProfileRow({
  profile,
  selected,
  isDefault,
  running,
  onSelect,
}: {
  profile: ComputeProfile
  selected: boolean
  isDefault: boolean
  running: boolean
  onSelect: () => void
}) {
  return (
    // eslint-disable-next-line ds/no-raw-element -- a whole profile card is the option; Button's nowrap, centring and fixed heights don't fit a multi-line row
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={`flex w-full items-start gap-3 rounded-md border p-3 text-left transition-colors ${
        selected ? "border-primary bg-primary/5" : "border-border hover:bg-muted/50"
      }`}
    >
      {profile.gpu > 0 ? (
        <Zap className="mt-0.5 size-4 shrink-0 text-primary" />
      ) : (
        <Cpu className="mt-0.5 size-4 shrink-0 text-text-tertiary" />
      )}
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2 text-sm font-medium">
          {profile.name}
          {isDefault ? (
            <span className="text-xs font-normal text-text-tertiary">default</span>
          ) : null}
          {running ? <span className="text-xs font-normal text-primary">running</span> : null}
        </span>
        <span className="mt-0.5 block text-xs text-text-tertiary">
          {profile.cpu} CPU · {profile.memory}
          {profile.gpu > 0 ? ` · ${profile.gpu}× ${profile.gpuType ?? "GPU"}` : ""} · idle{" "}
          {duration(profile.idleTimeout)}
          {profile.spot ? " · spot" : ""}
          {profile.egress === "none" ? " · offline" : ""}
        </span>
      </span>
      <span className="flex shrink-0 items-center gap-2 text-xs text-text-tertiary">
        {/* An estimate for comparing sizes, not a bill -- said plainly in the docs. */}
        <span title="Estimated, for comparing sizes">~{money(profile.costPerHour)}</span>
        {selected ? <Check className="size-4 text-primary" /> : null}
      </span>
    </button>
  )
}
