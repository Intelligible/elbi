// Deleted artifacts, and the two things you can do with one: put it back, or end it.
//
// A settings section rather than a top-level page: trash is somewhere you go after a
// mistake, not somewhere you work, and it sits beside the other account-scoped views of
// your own things. `FeedbackProvider` is not mounted here because `SettingsPage` already
// supplies one -- "delete forever" needs the confirm dialog it provides.

import {
  FileCheck2,
  Folder,
  Gauge,
  Layers,
  LayoutDashboard,
  NotebookPen,
  RotateCcw,
  Search,
  Trash2,
} from "lucide-react"
import type { ComponentType } from "react"
import { useCallback, useEffect, useMemo, useState } from "react"

import { IconButton } from "@/components/app/IconButton"
import { SectionHeader } from "@/components/settings/SectionHeader"
import { type Column, DataTable } from "@/components/ui/data-table"
import { useFeedback } from "@/components/ui/feedback"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import {
  deleteTrashForever,
  listTrash,
  restoreTrash,
  type TrashItem,
  type TrashKind,
} from "@/lib/trash"

// One icon and one readable label per trash-scoped kind, so a mixed list of trashed
// artifacts reads clearly without a separate lookup table at the call site.
const KIND_META: Record<TrashKind, { label: string; icon: ComponentType<{ className?: string }> }> =
  {
    notebook: { label: "Notebook", icon: NotebookPen },
    folder: { label: "Folder", icon: Folder },
    dashboard: { label: "Dashboard", icon: LayoutDashboard },
    saved_query: { label: "Saved query", icon: Search },
    metric: { label: "Metric", icon: Gauge },
    feature_view: { label: "Feature view", icon: Layers },
    derivation: { label: "Derivation", icon: FileCheck2 },
  }

export function TrashSection() {
  const [items, setItems] = useState<TrashItem[] | null>(null)
  const [q, setQ] = useState("")
  const feedback = useFeedback()

  const refresh = useCallback(() => listTrash().then(setItems), [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const restore = async (item: TrashItem) => {
    try {
      await restoreTrash(item.type, item.id)
      feedback.toast("ok", `Restored "${item.name}".`)
      void refresh()
    } catch (err) {
      feedback.toast("error", err instanceof Error ? err.message : "Restore failed.")
    }
  }

  const eraseForever = async (item: TrashItem) => {
    const ok = await feedback.confirm({
      title: `Delete "${item.name}" forever?`,
      body: "This permanently removes it, its history, and any cached results. It cannot be restored.",
      danger: true,
    })
    if (!ok) return
    try {
      await deleteTrashForever(item.type, item.id)
      feedback.toast("ok", `Deleted "${item.name}" forever.`)
      void refresh()
    } catch (err) {
      feedback.toast("error", err instanceof Error ? err.message : "Delete failed.")
    }
  }

  const filtered = useMemo(() => {
    if (!items) return []
    const term = q.trim().toLowerCase()
    return term ? items.filter((i) => i.name.toLowerCase().includes(term)) : items
  }, [items, q])

  const header = (
    <SectionHeader
      title="Trash"
      hint="Deleted artifacts stay here, recoverable, until the retention window purges them."
    />
  )

  if (items === null) {
    return (
      <>
        {header}
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-11 w-full rounded-lg" />
          ))}
        </div>
      </>
    )
  }

  const columns: Column<TrashItem>[] = [
    {
      key: "name",
      title: "Name",
      width: "40%",
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (item) => {
        const { icon: Icon } = KIND_META[item.type]
        return (
          <span className="flex items-center gap-2">
            <Icon className="size-4 shrink-0 text-text-tertiary" />
            <span className="truncate font-medium text-foreground">{item.name}</span>
          </span>
        )
      },
    },
    {
      key: "type",
      title: "Type",
      width: 140,
      render: (item) => <span className="text-text-secondary">{KIND_META[item.type].label}</span>,
    },
    {
      key: "deletedAt",
      title: "Deleted",
      width: 160,
      sorter: (a, b) => a.deletedAt.localeCompare(b.deletedAt),
      render: (item) => (
        <span className="text-text-tertiary tabular-nums">
          {new Date(item.deletedAt).toLocaleString()}
        </span>
      ),
    },
    {
      key: "actions",
      title: "",
      width: 96,
      align: "right",
      render: (item) => (
        <span className="flex items-center justify-end gap-1">
          <IconButton
            label={`Restore ${item.name}`}
            className="text-text-tertiary hover:text-foreground"
            onClick={(e) => {
              e.stopPropagation()
              void restore(item)
            }}
          >
            <RotateCcw className="size-4" />
          </IconButton>
          <IconButton
            label={`Delete ${item.name} forever`}
            className="text-text-tertiary hover:text-danger"
            onClick={(e) => {
              e.stopPropagation()
              void eraseForever(item)
            }}
          >
            <Trash2 className="size-4" />
          </IconButton>
        </span>
      ),
    },
  ]

  return (
    <>
      {header}
      <DataTable
        columns={columns}
        data={filtered}
        rowKey={(item) => `${item.type}:${item.id}`}
        empty={q ? "No trashed items match your search." : "Trash is empty."}
        toolbar={
          <div className="relative max-w-xs">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
            <Input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search trash"
              className="h-8 pl-8"
            />
          </div>
        }
      />
    </>
  )
}
