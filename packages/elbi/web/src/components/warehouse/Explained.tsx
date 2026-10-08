import { CircleHelp } from "lucide-react"
import type { ReactNode } from "react"

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { docsUrl } from "@/lib/docs"

// The data-sources docs section each explanation on the source page links to.
export const SYNC_DOCS = {
  status: docsUrl("data-sources", "sync-status"),
  method: docsUrl("data-sources", "incremental-sync"),
  schedule: docsUrl("data-sources", "sync-schedule"),
} as const

// Plain-language help for the source page's jargon. Each line is checked against the sync
// service: what a field holds and when it moves.
export const SYNC_HELP = {
  enabled: "Enabled tables are the ones a sync reads; turn each on or off with its Sync switch.",
  lastSynced:
    "When a sync last finished with every enabled table succeeding. A run where any table fails does not move it.",
  schedule:
    "How often the app syncs this source on its own. Manual only means it syncs only when you click Sync now.",
  method:
    "There are two variants of sync. Full refresh reads every row and replaces the table on each sync. Incremental appends only rows past the cursor column's value from the last sync.",
  rows: "Rows in this table as of its last successful sync: everything a full refresh wrote, or the running total of incremental syncs.",
} as const

/**
 * Jargon followed by a help icon: hovering or focusing the icon shows the tip, and the icon
 * links to the docs. The tip stays text-only so it works the same by mouse, keyboard and touch.
 */
export function Explained({
  tip,
  docs,
  about,
  children,
}: {
  tip: string
  docs: string
  // What the help is about, for the icon's accessible name.
  about: string
  children: ReactNode
}) {
  return (
    <span className="inline-flex items-center gap-1">
      {children}
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href={docs}
            target="_blank"
            rel="noreferrer"
            aria-label={`What “${about}” means`}
            className="text-text-tertiary hover:text-foreground"
          >
            <CircleHelp className="size-3.5" />
          </a>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-left leading-normal">{tip}</TooltipContent>
      </Tooltip>
    </span>
  )
}
