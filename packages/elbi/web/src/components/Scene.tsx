import { ArrowLeft } from "lucide-react"
import type * as React from "react"
import { useMemo } from "react"
import { Link } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { cn, uuid } from "@/lib/utils"

/* The standard page scaffold every browse/detail view is built from, so a page
   is a consistent object, a fixed header band (title, description, status,
   actions) over a scrolling body, rather than a bespoke document. The chrome
   comes from the system, not from each page. */

// Shell: fills the panel height as a column so the header stays put and only the
// body scrolls beneath it.
export function Scene({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="scene"
      className={cn("flex h-full min-h-0 flex-1 flex-col", className)}
      {...props}
    />
  )
}

type SceneHeaderProps = {
  title: React.ReactNode
  description?: React.ReactNode
  /* Status/verdict chips shown inline after the title. */
  badges?: React.ReactNode
  /* Right-aligned action controls (buttons, links). */
  actions?: React.ReactNode
  /* An icon rendered before the title. */
  icon?: React.ReactNode
  /* A back link rendered above the title (detail pages). */
  backTo?: string
  /* A back control rendered above the title when navigation is in-page state, not a
     route (e.g. a wizard step). Takes precedence over backTo. */
  onBack?: () => void
  backLabel?: string
  /* Render the title in the mono face: for code-like names (derivations, models). */
  mono?: boolean
  /* A tab strip or filter row rendered flush at the header's bottom edge. */
  children?: React.ReactNode
}

export function SceneHeader({
  title,
  description,
  badges,
  actions,
  icon,
  backTo,
  onBack,
  backLabel = "Back",
  mono,
  children,
}: SceneHeaderProps) {
  const backClass =
    "mb-3 -ml-1 inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 text-sm text-text-tertiary transition-colors hover:text-foreground"
  return (
    <header className="shrink-0 border-b border-border bg-background px-5 pt-4 pb-3.5">
      {onBack ? (
        <Button
          variant="ghost"
          onClick={onBack}
          className={cn(
            backClass,
            "h-auto font-normal has-[>svg]:px-1 hover:bg-transparent dark:hover:bg-transparent",
          )}
        >
          <ArrowLeft className="size-4" /> {backLabel}
        </Button>
      ) : backTo ? (
        <Link to={backTo} className={backClass}>
          <ArrowLeft className="size-4" /> {backLabel}
        </Link>
      ) : null}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2.5">
            {icon && <span className="shrink-0 text-text-tertiary">{icon}</span>}
            <h1
              className={cn(
                "min-w-0 truncate text-title font-semibold leading-tight tracking-tight text-foreground",
                mono && "font-mono text-lg tracking-normal",
              )}
            >
              {title}
            </h1>
            {badges}
          </div>
          {description && <p className="max-w-2xl text-sm text-text-secondary">{description}</p>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      {children && <div className="mt-4 -mb-4">{children}</div>}
    </header>
  )
}

type Width = "narrow" | "default" | "wide" | "full"
const WIDTH: Record<Width, string> = {
  narrow: "max-w-3xl",
  default: "max-w-5xl",
  wide: "max-w-7xl",
  full: "max-w-none",
}

// Scrolling content region. Pass `aside` to get a right-hand metadata panel
// (borders off on small screens, where it stacks below). Pass `canvas` to sit the
// content on a subtly greyer field, so card-shaped content (tables, panels) reads as
// distinct raised sections against it.
export function SceneBody({
  children,
  aside,
  width = "default",
  canvas,
  className,
}: {
  children: React.ReactNode
  aside?: React.ReactNode
  width?: Width
  canvas?: boolean
  className?: string
}) {
  return (
    <div className={cn("min-h-0 flex-1 overflow-y-auto", canvas && "bg-panel")}>
      {aside ? (
        <div className="mx-auto grid w-full max-w-6xl grid-cols-1 gap-8 px-5 py-5 lg:grid-cols-[minmax(0,1fr)_15rem]">
          <div className={cn("min-w-0 space-y-6", className)}>{children}</div>
          <aside className="space-y-5 lg:border-l lg:border-border lg:pl-6">{aside}</aside>
        </div>
      ) : (
        <div className={cn("mx-auto w-full px-5 py-5", WIDTH[width], className)}>{children}</div>
      )}
    </div>
  )
}

// A titled block within the body. Confident sm/semibold heading with an optional right-aligned
// action, not a faint uppercase whisper.
export function SceneSection({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: React.ReactNode
  description?: React.ReactNode
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={cn("space-y-3", className)}>
      {(title || action) && (
        <div className="flex items-center justify-between gap-3">
          <div className="space-y-0.5">
            {title && <h2 className="text-sm font-semibold text-foreground">{title}</h2>}
            {description && <p className="text-xs text-text-tertiary">{description}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  )
}

// A labeled metadata row for the right-hand panel: small-caps label over value.
export function ScenePanelLabel({
  label,
  children,
}: {
  label: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1">
      <div className="text-2xs font-semibold uppercase tracking-[0.05em] text-text-tertiary">
        {label}
      </div>
      <div className="text-sm text-foreground">{children}</div>
    </div>
  )
}

// Standard loading state: a header stub over a few content bars, so the page
// holds its shape instead of flashing a "Loading…" line.
export function SceneSkeleton({ rows = 4 }: { rows?: number }) {
  // The bars are indistinguishable, so nothing about one identifies it. Minting the
  // ids once per row count gives each a name that outlives a re-render, which is all a
  // key is for here.
  const bars = useMemo(() => Array.from({ length: rows }, () => uuid()), [rows])
  return (
    <Scene>
      <div className="shrink-0 border-b border-border px-6 pt-5 pb-4">
        <Skeleton className="h-7 w-56" />
        <Skeleton className="mt-2 h-4 w-80" />
      </div>
      <div className="mx-auto w-full max-w-5xl space-y-3 px-6 py-6">
        {bars.map((id) => (
          <Skeleton key={id} className="h-14 w-full" />
        ))}
      </div>
    </Scene>
  )
}
