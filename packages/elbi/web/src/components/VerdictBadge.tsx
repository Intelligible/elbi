import type { LucideIcon } from "lucide-react"
import { Circle, ShieldAlert, ShieldCheck, ShieldQuestion, Sigma } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import type { CertifiedRun } from "@/lib/chat"
import { EMPTY } from "@/lib/utils"

type VerdictStyle = {
  label: string
  variant: "verified" | "success" | "warning" | "danger" | "info" | "neutral"
  icon: LucideIcon
}

// The oracle's verdict, rendered from one vocabulary so the same word means the same thing (and
// wears the same color) everywhere it appears. Only a certified- sound run gets the shield and
// the verified tint: the UI never dresses an unproven run as verified.
const VERDICTS: Record<string, VerdictStyle> = {
  sound: { label: "Verified", variant: "verified", icon: ShieldCheck },
  verified: { label: "Verified", variant: "verified", icon: ShieldCheck },
  computed: { label: "Computed", variant: "info", icon: Sigma },
  inconclusive: { label: "Inconclusive", variant: "warning", icon: ShieldQuestion },
  unsound: { label: "Not sound", variant: "danger", icon: ShieldAlert },
  invalid: { label: "Invalid", variant: "danger", icon: ShieldAlert },
  unverified: { label: "Unverified", variant: "neutral", icon: Circle },
}

function styleFor(verdict: string): VerdictStyle {
  return (
    VERDICTS[verdict.toLowerCase()] ?? {
      label: verdict,
      variant: "neutral",
      icon: Circle,
    }
  )
}

export function VerdictBadge({ verdict }: { verdict: string }) {
  const { label, variant, icon: Icon } = styleFor(verdict)
  return (
    <Badge variant={variant} className="gap-1 font-medium">
      <Icon className="size-3.5" />
      {label}
    </Badge>
  )
}

// The run's certified estimate for display, with its label when present, or an em dash for
// a run that carries no scalar estimate (e.g. a data-contract check).
export function estimateText(run: Pick<CertifiedRun, "estimate" | "estimateLabel">): string {
  if (run.estimate === null) return EMPTY
  const value = run.estimate.toLocaleString(undefined, { maximumFractionDigits: 4 })
  return run.estimateLabel ? `${value} ${run.estimateLabel}` : value
}
