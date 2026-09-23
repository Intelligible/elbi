import {
  cloneElement,
  createContext,
  type ReactElement,
  type ReactNode,
  useContext,
  useId,
} from "react"

import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

type ControlProps = { id?: string; "aria-describedby"?: string; "aria-invalid"?: boolean }
type FieldState = { id: string; describedBy?: string; invalid: boolean }

const FieldContext = createContext<FieldState | null>(null)

// Adds the field's id/describedBy/invalid without dropping what the control already
// carries: a control that already names its own describer, or is independently invalid,
// keeps that alongside the hint/error rather than having it overwritten.
function mergeControlProps(props: ControlProps, field: FieldState): ControlProps {
  const describedBy =
    [props["aria-describedby"], field.describedBy].filter(Boolean).join(" ") || undefined
  return {
    id: field.id,
    "aria-describedby": describedBy,
    "aria-invalid": field.invalid ? true : props["aria-invalid"],
  }
}

export function FormField({
  label,
  hint,
  error,
  className,
  children,
}: {
  label: ReactNode
  hint?: ReactNode
  error?: ReactNode
  className?: string
  children: ReactElement<ControlProps>
}) {
  const auto = useId()
  const id = children.props.id ?? auto
  const hintId = hint ? `${id}-hint` : undefined
  const errorId = error ? `${id}-error` : undefined
  const describedBy = [hintId, errorId].filter(Boolean).join(" ") || undefined
  const field: FieldState = { id, describedBy, invalid: Boolean(error) }

  return (
    <FieldContext.Provider value={field}>
      <div className={cn("space-y-1", className)}>
        <Label htmlFor={id} className="text-xs font-medium">
          {label}
        </Label>
        {cloneElement(children, mergeControlProps(children.props, field))}
        {hint && (
          <p id={hintId} className="text-xs text-text-tertiary">
            {hint}
          </p>
        )}
        {error && (
          <p id={errorId} className="text-xs text-danger">
            {error}
          </p>
        )}
      </div>
    </FieldContext.Provider>
  )
}

// Wires a control FormField's own clone can't reach, such as a Radix Select's trigger
// nested inside the Root it clones (the Root renders no DOM node, so a cloned id there
// names nothing). No Slot: the radix-ui import ban applies to this file same as any
// other under components/app, so this clones its single child directly, same as FormField.
export function FormControl({ children }: { children: ReactElement<ControlProps> }) {
  const field = useContext(FieldContext)
  if (!field) return children
  return cloneElement(children, mergeControlProps(children.props, field))
}
