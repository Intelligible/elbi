import * as React from "react"
import { type VariantProps } from "class-variance-authority"
import { ChevronDown } from "lucide-react"

import { Button, buttonVariants } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"

type SplitButtonProps = {
  /** The primary action fired by the left (main) half. */
  onClick: () => void
  /** Main-half label and icon. */
  children: React.ReactNode
  /** The dropdown body: a set of `DropdownMenuItem`s revealed by the chevron half. */
  menu: React.ReactNode
  variant?: VariantProps<typeof buttonVariants>["variant"]
  size?: VariantProps<typeof buttonVariants>["size"]
  disabled?: boolean
  /** Which edge the dropdown aligns to (defaults to the button's right edge). */
  align?: "start" | "center" | "end"
  /** Accessible name for the chevron half. */
  menuLabel?: string
  className?: string
}

// A split button: a main action joined to a dropdown of related actions: the main button, a
// divider, then a chevron half. The two halves are separate `Button`s sharing the chosen
// variant, so the app's own colors and framing carry through; only the shared edge is squared
// and a `--btn-frame` divider marks the seam.
export function SplitButton({
  onClick,
  children,
  menu,
  variant = "default",
  size = "default",
  disabled = false,
  align = "end",
  menuLabel = "More actions",
  className,
}: SplitButtonProps) {
  return (
    <div className={cn("inline-flex", className)}>
      <Button
        variant={variant}
        size={size}
        disabled={disabled}
        onClick={onClick}
        className="rounded-r-none"
      >
        {children}
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant={variant}
            size={size}
            disabled={disabled}
            aria-label={menuLabel}
            className="rounded-l-none border-l border-[color:var(--btn-frame)] px-2"
          >
            <ChevronDown className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align={align} className="min-w-48">
          {menu}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
