import {
  ChevronRight,
  Copy,
  Folder,
  FolderInput,
  FolderPlus,
  MoreHorizontal,
  NotebookPen,
  Pencil,
  Plus,
  Search,
  Trash2,
  Upload,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { EmptyState } from "@/components/app/EmptyState"
import { IconButton } from "@/components/app/IconButton"
import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { type Column, DataTable } from "@/components/ui/data-table"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { FeedbackProvider, useFeedback } from "@/components/ui/feedback"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { SplitButton } from "@/components/ui/split-button"
import {
  createFolder,
  createNotebook,
  deleteFolder,
  deleteNotebook,
  duplicateNotebook,
  importNotebook,
  listFolders,
  listNotebooks,
  moveFolder,
  moveNotebook,
  type NotebookFolder,
  type NotebookSummary,
  renameFolder,
} from "@/lib/notebooks"
import { cn } from "@/lib/utils"

// The sentinel a `Select` uses for "the root", since Radix items cannot hold "".
const ROOT = "__root__"

// A browser row is either a folder or a notebook; folders are listed first.
type Row =
  | { kind: "folder"; id: string; folder: NotebookFolder; count: number }
  | { kind: "notebook"; id: string; notebook: NotebookSummary }

// The ancestor chain from the root down to (and including) `folderId`.
function breadcrumbOf(
  byId: Map<string, NotebookFolder>,
  folderId: string | null,
): NotebookFolder[] {
  const chain: NotebookFolder[] = []
  const seen = new Set<string>()
  let cur = folderId ? byId.get(folderId) : undefined
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id)
    chain.unshift(cur)
    cur = cur.parent_id ? byId.get(cur.parent_id) : undefined
  }
  return chain
}

export function NotebooksPage() {
  return (
    <FeedbackProvider>
      <NotebooksBody />
    </FeedbackProvider>
  )
}

function NotebooksBody() {
  const fb = useFeedback()
  const [folders, setFolders] = useState<NotebookFolder[] | null>(null)
  const [notebooks, setNotebooks] = useState<NotebookSummary[] | null>(null)
  const [q, setQ] = useState("")
  const [params, setParams] = useSearchParams()
  const folderId = params.get("folder")
  const navigate = useNavigate()
  const fileInput = useRef<HTMLInputElement>(null)

  // Dialog state: a name prompt (new folder / rename) and a move-to-folder picker.
  const [nameDialog, setNameDialog] = useState<{
    mode: "new-folder" | "rename"
    id?: string
    value: string
  } | null>(null)
  const [moveDialog, setMoveDialog] = useState<{
    item: Row
    target: string
  } | null>(null)

  const refresh = useCallback(
    () =>
      Promise.all([listFolders(), listNotebooks()]).then(([f, n]) => {
        setFolders(f)
        setNotebooks(n)
      }),
    [],
  )
  useEffect(() => {
    void refresh()
  }, [refresh])

  const goTo = (id: string | null) => setParams(id ? { folder: id } : {})

  const byId = useMemo(() => new Map((folders ?? []).map((f) => [f.id, f])), [folders])
  const breadcrumb = useMemo(() => breadcrumbOf(byId, folderId), [byId, folderId])

  // Descendants of a folder, used to keep item counts honest and to stop a move into a
  // folder's own subtree.
  const descendantsOf = useMemo(() => {
    const children = new Map<string | null, NotebookFolder[]>()
    for (const f of folders ?? []) {
      const list = children.get(f.parent_id) ?? []
      list.push(f)
      children.set(f.parent_id, list)
    }
    return (id: string): Set<string> => {
      const out = new Set<string>()
      const stack = [...(children.get(id) ?? [])]
      while (stack.length) {
        const f = stack.pop()
        if (!f || out.has(f.id)) continue
        out.add(f.id)
        stack.push(...(children.get(f.id) ?? []))
      }
      return out
    }
  }, [folders])

  // The rows in the current folder: child folders (with a direct item count), then the
  // notebooks that live here, both filtered by the search box.
  const rows = useMemo<Row[]>(() => {
    if (!folders || !notebooks) return []
    const term = q.trim().toLowerCase()
    const match = (name: string) => !term || name.toLowerCase().includes(term)
    const here = folderId ?? null
    const folderRows: Row[] = folders
      .filter((f) => (f.parent_id ?? null) === here && match(f.name))
      .map((f) => {
        const childFolders = folders.filter((c) => c.parent_id === f.id).length
        const childNotebooks = notebooks.filter((n) => n.folder_id === f.id).length
        return {
          kind: "folder",
          id: f.id,
          folder: f,
          count: childFolders + childNotebooks,
        }
      })
    const notebookRows: Row[] = notebooks
      .filter((n) => (n.folder_id ?? null) === here && match(n.name))
      .map((n) => ({ kind: "notebook", id: n.id, notebook: n }))
    return [...folderRows, ...notebookRows]
  }, [folders, notebooks, folderId, q])

  const createHere = async () => {
    const { id } = await createNotebook("Untitled notebook", folderId)
    navigate(`/notebooks/${id}`)
  }

  const onImport = async (file: File) => {
    const text = await file.text()
    const { id } = await importNotebook(file.name.replace(/\.ipynb$/, ""), JSON.parse(text))
    if (folderId) await moveNotebook(id, folderId)
    navigate(`/notebooks/${id}`)
  }

  const submitName = async () => {
    if (!nameDialog) return
    const value = nameDialog.value.trim()
    if (!value) return
    if (nameDialog.mode === "new-folder") await createFolder(value, folderId)
    else if (nameDialog.id) await renameFolder(nameDialog.id, value)
    setNameDialog(null)
    void refresh()
  }

  const submitMove = async () => {
    if (!moveDialog) return
    const target = moveDialog.target === ROOT ? null : moveDialog.target
    if (moveDialog.item.kind === "folder") {
      await moveFolder(moveDialog.item.id, target)
    } else {
      await moveNotebook(moveDialog.item.id, target)
    }
    setMoveDialog(null)
    void refresh()
  }

  const removeFolder = async (row: Extract<Row, { kind: "folder" }>) => {
    const nested = descendantsOf(row.id)
    const notebookCount = (notebooks ?? []).filter(
      (n) => n.folder_id === row.id || (n.folder_id && nested.has(n.folder_id)),
    ).length
    const total = nested.size + notebookCount
    if (total > 0) {
      const ok = await fb.confirm({
        title: "Delete folder",
        body: `Delete "${row.folder.name}" and its ${total} item${total === 1 ? "" : "s"}?`,
        danger: true,
      })
      if (!ok) return
      await deleteFolder(row.id, true)
    } else {
      await deleteFolder(row.id, false)
    }
    void refresh()
  }

  const removeNotebook = async (id: string) => {
    await deleteNotebook(id)
    void refresh()
  }

  const duplicate = async (id: string) => {
    try {
      const copy = await duplicateNotebook(id)
      navigate(`/notebooks/${copy.id}`)
    } catch (e) {
      fb.toast("error", e instanceof Error ? e.message : "could not duplicate")
    }
  }

  // Folders a move can target: every folder except the one being moved and its own
  // descendants (which would form a cycle), each labelled by depth for readability.
  const moveTargets = useMemo(() => {
    if (!moveDialog || !folders) return []
    const banned =
      moveDialog.item.kind === "folder"
        ? new Set([moveDialog.item.id, ...descendantsOf(moveDialog.item.id)])
        : new Set<string>()
    const depth = (f: NotebookFolder): number => breadcrumbOf(byId, f.id).length - 1
    return folders
      .filter((f) => !banned.has(f.id))
      .map((f) => ({ id: f.id, label: `${"  ".repeat(depth(f))}${f.name}` }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [moveDialog, folders, byId, descendantsOf])

  if (folders === null || notebooks === null) return <SceneSkeleton />

  // Keep folders above notebooks regardless of the active sort.
  const foldersFirst = (a: Row, b: Row) => (a.kind === b.kind ? 0 : a.kind === "folder" ? -1 : 1)
  const nameOf = (r: Row) => (r.kind === "folder" ? r.folder.name : r.notebook.name)
  const editedOf = (r: Row) => (r.kind === "folder" ? r.folder.updated_at : r.notebook.updated_at)

  const columns: Column<Row>[] = [
    {
      key: "name",
      title: "Name",
      width: "50%",
      sorter: (a, b) => foldersFirst(a, b) || nameOf(a).localeCompare(nameOf(b)),
      render: (r) =>
        r.kind === "folder" ? (
          <span className="flex items-center gap-2">
            <Folder className="size-4 shrink-0 text-text-tertiary" />
            <span className="truncate font-medium text-foreground">{r.folder.name}</span>
          </span>
        ) : (
          <span className="flex items-center gap-2">
            <NotebookPen className="size-4 shrink-0 text-text-tertiary" />
            <span className="truncate font-medium text-foreground">{r.notebook.name}</span>
            {r.notebook.copied_from && (
              <Badge
                variant="neutral"
                className="shrink-0"
                title="Duplicated from another notebook"
              >
                copy
              </Badge>
            )}
          </span>
        ),
    },
    {
      key: "meta",
      title: "Contents",
      width: 120,
      align: "right",
      sorter: (a, b) => foldersFirst(a, b),
      render: (r) => (
        <span className="text-text-tertiary tabular-nums">
          {r.kind === "folder"
            ? `${r.count} item${r.count === 1 ? "" : "s"}`
            : `${r.notebook.cell_count} cells`}
        </span>
      ),
    },
    {
      key: "edited",
      title: "Edited",
      width: 180,
      align: "right",
      sorter: (a, b) => foldersFirst(a, b) || editedOf(a).localeCompare(editedOf(b)),
      render: (r) => (
        <span className="text-text-tertiary tabular-nums">
          {new Date(editedOf(r)).toLocaleString()}
        </span>
      ),
    },
    {
      key: "actions",
      title: "",
      width: 48,
      align: "right",
      render: (r) => (
        // h-7.5 holds the rows at 42px: this cell is the tallest in each row.
        <span className="flex h-7.5 justify-end">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <IconButton label="Row actions" size="icon-xs" className="text-text-tertiary">
                <MoreHorizontal className="size-4" />
              </IconButton>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="min-w-40">
              {r.kind === "folder" && (
                <DropdownMenuItem
                  onSelect={() =>
                    setNameDialog({
                      mode: "rename",
                      id: r.id,
                      value: r.folder.name,
                    })
                  }
                >
                  <Pencil className="size-4" /> Rename
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                onSelect={() =>
                  setMoveDialog({
                    item: r,
                    target:
                      (r.kind === "folder" ? r.folder.parent_id : r.notebook.folder_id) ?? ROOT,
                  })
                }
              >
                <FolderInput className="size-4" /> Move to…
              </DropdownMenuItem>
              {r.kind === "notebook" && (
                <DropdownMenuItem onSelect={() => void duplicate(r.id)}>
                  <Copy className="size-4" /> Duplicate
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                onSelect={() =>
                  r.kind === "folder" ? void removeFolder(r) : void removeNotebook(r.id)
                }
              >
                <Trash2 className="size-4" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </span>
      ),
    },
  ]

  const openRow = (r: Row) => (r.kind === "folder" ? goTo(r.id) : navigate(`/notebooks/${r.id}`))

  return (
    <Scene>
      <SceneHeader
        icon={<NotebookPen className="size-5" />}
        title="Notebooks"
        description="Explore your data in a reactive notebook, then promote a cell to a certified derivation or a registered model."
        actions={
          <>
            <input
              ref={fileInput}
              type="file"
              accept=".ipynb"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void onImport(file)
                e.target.value = ""
              }}
            />
            <SplitButton
              size="sm"
              onClick={createHere}
              menuLabel="More create options"
              menu={
                <>
                  <DropdownMenuItem
                    onSelect={() => setNameDialog({ mode: "new-folder", value: "New folder" })}
                  >
                    <FolderPlus className="size-4" /> New folder
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => fileInput.current?.click()}>
                    <Upload className="size-4" /> Import .ipynb
                  </DropdownMenuItem>
                </>
              }
            >
              <Plus className="size-4" /> New notebook
            </SplitButton>
          </>
        }
      />
      <SceneBody width="full" canvas>
        <nav
          className="mb-3 flex items-center gap-1 text-sm text-text-tertiary"
          aria-label="Breadcrumb"
        >
          <Button
            variant="ghost"
            size="xs"
            className="px-1.5 text-sm"
            aria-current={breadcrumb.length === 0 ? "page" : undefined}
            onClick={() => goTo(null)}
          >
            Home
          </Button>
          {breadcrumb.map((f, i) => (
            <span key={f.id} className="flex items-center gap-1">
              <ChevronRight className="size-3.5 shrink-0" />
              <Button
                variant="ghost"
                size="xs"
                className={cn(
                  "px-1.5 text-sm font-normal",
                  i === breadcrumb.length - 1 && "font-medium text-foreground",
                )}
                aria-current={i === breadcrumb.length - 1 ? "page" : undefined}
                onClick={() => goTo(f.id)}
              >
                {f.name}
              </Button>
            </span>
          ))}
        </nav>
        <DataTable
          columns={columns}
          data={rows}
          rowKey={(r) => `${r.kind}:${r.id}`}
          onRowClick={openRow}
          empty={
            q ? (
              "Nothing here matches your search."
            ) : (
              <EmptyState
                title="This folder is empty."
                className="p-0"
                action={
                  <Button size="sm" onClick={createHere}>
                    <Plus className="size-4" /> New notebook
                  </Button>
                }
              />
            )
          }
          toolbar={
            <div className="relative max-w-xs">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search this folder"
                className="h-8 pl-8"
              />
            </div>
          }
        />
      </SceneBody>

      <Dialog open={nameDialog !== null} onOpenChange={(o) => !o && setNameDialog(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>
              {nameDialog?.mode === "rename" ? "Rename folder" : "New folder"}
            </DialogTitle>
          </DialogHeader>
          <Input
            autoFocus
            value={nameDialog?.value ?? ""}
            onChange={(e) => setNameDialog((d) => (d ? { ...d, value: e.target.value } : d))}
            onKeyDown={(e) => {
              if (e.key === "Enter") void submitName()
            }}
            placeholder="Folder name"
          />
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setNameDialog(null)}>
              Cancel
            </Button>
            <Button size="sm" onClick={() => void submitName()}>
              {nameDialog?.mode === "rename" ? "Rename" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={moveDialog !== null} onOpenChange={(o) => !o && setMoveDialog(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>
              Move{" "}
              {moveDialog?.item.kind === "folder"
                ? `"${moveDialog.item.folder.name}"`
                : moveDialog?.item.kind === "notebook"
                  ? `"${moveDialog.item.notebook.name}"`
                  : ""}
            </DialogTitle>
          </DialogHeader>
          <Select
            value={moveDialog?.target ?? ROOT}
            onValueChange={(v) => setMoveDialog((d) => (d ? { ...d, target: v } : d))}
          >
            <SelectTrigger>
              <SelectValue placeholder="Destination folder" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ROOT}>Home</SelectItem>
              {moveTargets.map((t) => (
                <SelectItem key={t.id} value={t.id}>
                  {t.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setMoveDialog(null)}>
              Cancel
            </Button>
            <Button size="sm" onClick={() => void submitMove()}>
              Move
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Scene>
  )
}
