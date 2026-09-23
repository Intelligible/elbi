import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

// Spacing/typography matches NotificationsPage's empty card, the fullest call site (icon +
// title + description): 8px under the icon, 4px to the description, 16px to an action, and
// a description as wide as the card. Compact or actioned call sites override padding/border
// via className instead of this default gaining variants.
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: {
  icon?: LucideIcon
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn("flex flex-col items-center px-4 py-10 text-center", className)}>
      {Icon && <Icon className="mb-2 size-6 text-text-tertiary" aria-hidden />}
      <p className="text-sm font-medium text-foreground">{title}</p>
      {description && <p className="mt-1 text-sm text-text-secondary">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
