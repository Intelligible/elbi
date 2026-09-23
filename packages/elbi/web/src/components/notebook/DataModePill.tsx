// Which side of the data boundary a cell landed on.
//
// Hex puts this on the cell, and the reason to copy that is the reason the split exists
// at all: the difference between "the aggregation ran in the warehouse and a preview came
// back" and "every row is now in this kernel" is what decides whether a notebook scales,
// and a crossing nobody can see just moves the memory failure later.
import { Database, Download } from "lucide-react"

const MODES = {
  pushed_down: {
    label: "in warehouse",
    title:
      "This cell's query ran in the warehouse and only a bounded result came back. " +
      "Aggregations here scale with the data, not with the kernel's memory.",
    icon: Database,
    className: "bg-success-tint text-success",
  },
  materialised: {
    label: "in kernel",
    title:
      "This cell pulled rows into the kernel, so its memory limit is the bound. " +
      "Use sql(...) with an aggregate to do the work where the data is.",
    icon: Download,
    className: "bg-warning-tint text-warning",
  },
} as const

export function DataModePill({ mode }: { mode: string }) {
  const shape = MODES[mode as keyof typeof MODES]
  // An unknown mode from a newer kernel is not worth rendering a mystery badge for.
  if (!shape) return null
  const Icon = shape.icon
  return (
    <span
      title={shape.title}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-3xs font-medium ${shape.className}`}
    >
      <Icon className="size-2.5" />
      {shape.label}
    </span>
  )
}
