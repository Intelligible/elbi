import type { ComponentProps, ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

type IconButtonProps = Omit<ComponentProps<typeof Button>, "size" | "children" | "aria-label"> & {
  label: string
  size?: "icon" | "icon-xs" | "icon-sm" | "icon-lg"
  children: ReactNode
}

// An icon-only button always carries a name: it is the aria-label and the tooltip.
export function IconButton({
  label,
  variant = "ghost",
  size = "icon-sm",
  asChild,
  children,
  ...props
}: IconButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          // A slotted child (such as an <a>) is not a button, so it takes no type.
          type={asChild ? undefined : "button"}
          variant={variant}
          size={size}
          asChild={asChild}
          aria-label={label}
          {...props}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}
