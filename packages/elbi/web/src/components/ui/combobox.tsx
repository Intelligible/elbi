"use client"

import { CheckIcon, ChevronDownIcon } from "lucide-react"
import * as React from "react"

import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"

export interface ComboboxOption {
  value: string
  // Shown beside the value, e.g. a column's type.
  detail?: string
}

// A select whose options can be searched. For choosing among names that already exist,
// where a long list makes scrolling slow and a typo would be a silent mistake.
function Combobox({
  value,
  onValueChange,
  options,
  placeholder = "Select…",
  searchPlaceholder = "Search…",
  emptyText = "No matches.",
  // Lets the user pick "nothing", shown as this label, which sets the value to "".
  clearLabel,
  disabled,
  className,
  ...triggerProps
}: Omit<React.ComponentProps<"button">, "value" | "onChange"> & {
  value: string
  onValueChange: (value: string) => void
  options: ComboboxOption[]
  placeholder?: string
  searchPlaceholder?: string
  emptyText?: string
  clearLabel?: string
}) {
  const [open, setOpen] = React.useState(false)
  const choose = (next: string) => {
    onValueChange(next)
    setOpen(false)
  }
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          data-slot="combobox-trigger"
          className={cn(
            "flex h-[2.3125rem] w-full items-center justify-between gap-2 rounded-md border border-border bg-background px-2.5 text-sm whitespace-nowrap transition-colors outline-none hover:border-border-strong focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive",
            !value && "text-muted-foreground",
            className
          )}
          {...triggerProps}
        >
          <span className="truncate">{value || clearLabel || placeholder}</span>
          <ChevronDownIcon className="size-4 shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent>
        <Command>
          <CommandInput placeholder={searchPlaceholder} />
          <CommandList>
            <CommandEmpty>{emptyText}</CommandEmpty>
            <CommandGroup>
              {clearLabel ? (
                <CommandItem value="" onSelect={() => choose("")}>
                  <span className="text-muted-foreground">{clearLabel}</span>
                  {value === "" ? <CheckIcon className="ml-auto size-4" /> : null}
                </CommandItem>
              ) : null}
              {options.map((o) => (
                <CommandItem key={o.value} value={o.value} onSelect={() => choose(o.value)}>
                  <span className="truncate">{o.value}</span>
                  {o.detail ? (
                    <span className="text-xs text-muted-foreground">{o.detail}</span>
                  ) : null}
                  {value === o.value ? <CheckIcon className="ml-auto size-4" /> : null}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

export { Combobox }
