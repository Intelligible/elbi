import { GripVertical } from "lucide-react"
import type * as React from "react"
import * as ResizablePrimitive from "react-resizable-panels"

import { cn } from "@/lib/utils"

/**
 * A row or column of panels the reader can resize.
 *
 * Pair it with {@link useResizableLayout} to remember where they left the split.
 */
function ResizablePanelGroup({
  className,
  orientation = "horizontal",
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Group>) {
  return (
    <ResizablePrimitive.Group
      data-slot="resizable-panel-group"
      orientation={orientation}
      className={cn("flex h-full w-full min-w-0", className)}
      {...props}
    />
  )
}

function ResizablePanel({ ...props }: React.ComponentProps<typeof ResizablePrimitive.Panel>) {
  return <ResizablePrimitive.Panel data-slot="resizable-panel" {...props} />
}

/**
 * The drag target between two panels.
 *
 * A real separator, not a styled div: it carries the window-splitter role and its
 * value/min/max, takes focus, and resizes on the arrow keys, so the split is adjustable
 * without a pointer. `withHandle` draws the visible grip; without it the target is an
 * invisible hairline that lights up on hover.
 */
function ResizableHandle({
  withHandle,
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Separator> & {
  withHandle?: boolean
}) {
  return (
    <ResizablePrimitive.Separator
      data-slot="resizable-handle"
      className={cn(
        "relative flex w-px items-center justify-center bg-border transition-colors",
        "hover:bg-primary/50 data-[state=drag]:bg-primary/50",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        // The hit area is wider than the hairline, so the grip is easy to catch.
        "after:absolute after:inset-y-0 after:left-1/2 after:w-1.5 after:-translate-x-1/2",
        className
      )}
      {...props}
    >
      {withHandle ? (
        <div className="z-10 flex h-4 w-3 items-center justify-center rounded-xs border border-border bg-card">
          <GripVertical className="size-2.5 text-muted-foreground" />
        </div>
      ) : null}
    </ResizablePrimitive.Separator>
  )
}

/** Remembers a group's split across reloads. Spread the result onto the group. */
const useResizableLayout = ResizablePrimitive.useDefaultLayout

export { ResizablePanelGroup, ResizablePanel, ResizableHandle, useResizableLayout }
