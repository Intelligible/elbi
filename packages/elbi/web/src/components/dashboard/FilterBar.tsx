// The row of filter controls for a dashboard's variables. Each control writes into the
// shared variable state; a change re-resolves the page's widgets. Dynamic options are
// loaded from the backing derivation on mount.

import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { Option, Variable } from "@/lib/dashboards"
import { variableOptions } from "@/lib/dashboards"

function label(variable: Variable): string {
  return variable.label ?? variable.name.replace(/_/g, " ")
}

function Control({
  dashboardId,
  variable,
  value,
  onChange,
}: {
  dashboardId: string
  variable: Variable
  value: unknown
  onChange: (value: unknown) => void
}) {
  const [options, setOptions] = useState<Option[]>([])

  useEffect(() => {
    if (variable.options?.derivation) {
      variableOptions(dashboardId, variable.name)
        .then(setOptions)
        .catch(() => setOptions([]))
    } else if (variable.options?.values) {
      setOptions(
        variable.options.values.map((v) =>
          typeof v === "object" && v !== null ? (v as Option) : { value: v, label: String(v) },
        ),
      )
    }
  }, [dashboardId, variable])

  const control = variable.control ?? "dropdown"

  if (control === "search") {
    return (
      <Input
        className="h-8 w-40"
        placeholder={label(variable)}
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value)}
      />
    )
  }

  if (control === "toggle") {
    return (
      <Button variant={value ? "default" : "outline"} size="sm" onClick={() => onChange(!value)}>
        {label(variable)}
      </Button>
    )
  }

  if (control === "multiselect") {
    const selected = Array.isArray(value) ? (value as unknown[]) : []
    return (
      <div className="flex flex-wrap items-center gap-1">
        {options.map((o) => {
          const on = selected.some((s) => s === o.value)
          return (
            <Button
              key={String(o.value)}
              variant={on ? "default" : "outline"}
              size="xs"
              onClick={() =>
                onChange(on ? selected.filter((s) => s !== o.value) : [...selected, o.value])
              }
            >
              {o.label}
            </Button>
          )
        })}
      </div>
    )
  }

  // dropdown (and date/range fall back to a plain select of known options)
  return (
    <Select
      value={value === undefined || value === null ? "" : String(value)}
      onValueChange={(v) => onChange(v)}
    >
      <SelectTrigger size="sm" className="w-40">
        <SelectValue placeholder={label(variable)} />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={String(o.value)} value={String(o.value)}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export function FilterBar({
  dashboardId,
  variables,
  state,
  onChange,
}: {
  dashboardId: string
  variables: Variable[]
  state: Record<string, unknown>
  onChange: (name: string, value: unknown) => void
}) {
  if (variables.length === 0) return null
  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-border px-6 py-3">
      {variables.map((variable) => (
        <div key={variable.name} className="flex items-center gap-2">
          <span className="text-xs font-medium text-text-tertiary capitalize">
            {label(variable)}
          </span>
          <Control
            dashboardId={dashboardId}
            variable={variable}
            value={state[variable.name]}
            onChange={(v) => onChange(variable.name, v)}
          />
        </div>
      ))}
    </div>
  )
}
