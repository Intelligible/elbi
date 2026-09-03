import { cn } from "@/lib/utils"

// A shimmering placeholder for content that hasn't loaded yet. Prefer this over a "Loading…"
// string: it holds the page's shape so nothing jumps on arrival.
function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="skeleton"
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }
