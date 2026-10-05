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
  rows: "Number of rows recorded by this table's last successful sync.",
} as const

/**
 * Hover help on a piece of jargon: the trigger keeps its own look and gains a dotted
 * underline, so it reads as explained without turning into a link.
 */
export function Explained({
  tip,
  docs,
  children,
  asChild = false,
}: {
  tip: string
  docs?: string
  children: ReactNode
  // Wrap an element that is already focusable (a select trigger) instead of a text label.
  asChild?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger
        asChild={asChild}
        type={asChild ? undefined : "button"}
        className={
          asChild
            ? undefined
            : "cursor-help underline decoration-dotted decoration-text-tertiary underline-offset-2"
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-left leading-normal">
        {tip}
        {docs ? (
          <>
            {" "}
            <a href={docs} target="_blank" rel="noreferrer" className="font-medium underline">
              Learn more in the docs
            </a>
          </>
        ) : null}
      </TooltipContent>
    </Tooltip>
  )
}
