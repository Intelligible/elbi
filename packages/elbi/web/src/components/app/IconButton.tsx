import type { ComponentProps, ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

type IconButtonProps = Omit<ComponentProps<typeof Button>, "size" | "children"> & {
  label: string
  size?: "icon" | "icon-xs" | "icon-sm" | "icon-lg"
  children: ReactNode
}

// An icon-only button always carries a name: it is the aria-label and the tooltip.
export function IconButton({
  label,
  variant = "ghost",
  size = "icon-sm",
  children,
  ...props
}: IconButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button type="button" variant={variant} size={size} aria-label={label} {...props}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}
