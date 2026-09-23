import {
  Boxes,
  CalendarClock,
  ChevronDown,
  ChevronUp,
  CirclePlay,
  Cpu,
  Download,
  Loader2,
  Lock,
  MoreHorizontal,
  NotebookPen,
  Play,
  Plus,
  Power,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Variable,
  X,
  Zap,
} from "lucide-react"
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { useAssistantContext } from "@/components/AssistantContext"
import type { PageContext } from "@/components/AssistantPanel"
import { FormControl, FormField } from "@/components/app/FormField"
import { IconButton } from "@/components/app/IconButton"
import { CellEditor } from "@/components/notebook/CellEditor"
import { CellOutputs } from "@/components/notebook/CellOutput"
import { CellQueries } from "@/components/notebook/CellQueries"
import { ComputeDialog } from "@/components/notebook/ComputeDialog"
import { DataModePill } from "@/components/notebook/DataModePill"
import { NotebookMarkdown } from "@/components/notebook/NotebookMarkdown"
import { WidgetManagerContext } from "@/components/notebook/WidgetView"
import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
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
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { SplitButton } from "@/components/ui/split-button"
import { Textarea } from "@/components/ui/textarea"
import {
  type DatasetColumn,
  type FeatureSource,
  getDatasetColumns,
  getFeatureSources,
  type TrainRequest,
  trainModel,
} from "@/lib/chat"
import {
  addCell,
  type Cell,
  type CellQuery,
  type CellType,
  completeCell,
  deleteCell,
  type EnvironmentResult,
  exportNotebookUrl,
  getNotebook,
  getNotebookVariables,
  inspectCell,
  interruptNotebook,
  type NotebookVariable,
  type NotebookView,
  type Output,
  type PromoteResult,
  promoteCell,
  relockNotebook,
  reorderCells,
  restartNotebook,
  runNotebook,
  type Schedule,
  sendInput,
  updateCell,
  updateNotebook,
} from "@/lib/notebooks"
import { cn } from "@/lib/utils"
import type { NotebookWidgetManager } from "@/lib/widgets"

// The base-environment Select's value for "None", since Radix items cannot hold "".
const NO_BASE_ENV = "__none__"

// Detect the app's dark theme (a `.dark` ancestor) so CodeMirror matches it, and keep it
// in sync if the theme toggles.
function useDarkTheme(): boolean {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"))
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setDark(document.documentElement.classList.contains("dark")),
    )
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })
    return () => observer.disconnect()
  }, [])
  return dark
}

export function NotebookPage() {
  const { name: notebookId = "" } = useParams()
  const navigate = useNavigate()
  const dark = useDarkTheme()
  const [view, setView] = useState<NotebookView | null>(null)
  const [missing, setMissing] = useState(false)
  const [running, setRunning] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [stale, setStale] = useState<Set<string>>(new Set())
  const [focused, setFocused] = useState<string | null>(null)
  // A cell's pending input() prompt (from the run stream), and the text being typed.
  const [inputPrompt, setInputPrompt] = useState<{
    cell: string
    text: string
    password: boolean
  } | null>(null)
  const [inputValue, setInputValue] = useState("")
  // The notebook's live ipywidgets manager, connected to the comm WebSocket for the run.
  const [widgetManager, setWidgetManager] = useState<NotebookWidgetManager | null>(null)
  const abortRef = useRef<(() => void) | null>(null)
  // Debounced source saves, keyed by cell id, so typing does not fire a request per key.
  const saveTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  // The live kernel's variables and whether the inspector panel is open.
  const [variables, setVariables] = useState<NotebookVariable[]>([])
  const [showVariables, setShowVariables] = useState(false)

  const load = useCallback(async () => {
    try {
      setView(await getNotebook(notebookId))
      setMissing(false)
    } catch {
      setMissing(true)
    }
  }, [notebookId])

  const refreshVariables = useCallback(async () => {
    try {
      setVariables(await getNotebookVariables(notebookId))
    } catch {
      setVariables([]) // no live kernel yet, or it went away
    }
  }, [notebookId])

  useEffect(() => {
    void load()
    void refreshVariables()
  }, [load, refreshVariables])

  // The assistant panel can edit this notebook (add a cell, run it); when it does it fires
  // this event so the page reloads and the change appears live.
  useEffect(() => {
    const onRefetch = () => {
      void load()
      void refreshVariables()
    }
    window.addEventListener("elbi:refetch", onRefetch)
    return () => window.removeEventListener("elbi:refetch", onRefetch)
  }, [load, refreshVariables])

  // Publish this notebook's live state to the assistant panel: its name, id, and a cell
  // outline. The model then knows exactly what the user is looking at and edits THIS
  // notebook (via its write_notebook/run_notebook tools) rather than guessing.
  const assistantContext = useMemo<PageContext>(() => {
    if (view === null) return null
    const outline = view.cells
      .map((c, i) => {
        const firstLine = c.source.split("\n").find((l) => l.trim()) ?? "(empty)"
        return `[${i + 1}] ${c.cell_type}: ${firstLine.slice(0, 60)}`
      })
      .join("; ")
    return {
      label: view.name,
      text:
        `The user is viewing the notebook "${view.name}" (id "${notebookId}") with ` +
        `${view.cells.length} cells: ${outline}. When they ask you to add, change, ` +
        `visualize, or run something, act on THIS notebook: read_notebook (id ` +
        `"${notebookId}") for full cell sources, write_notebook to add or edit cells, ` +
        `and run_notebook to execute it. Prefer editing this notebook over a new one.`,
    }
  }, [view, notebookId])
  useAssistantContext(assistantContext)

  // Connect the notebook's ipywidgets comm channel for the session, tearing it down on
  // navigation away or notebook change. The manager (heavy @jupyter-widgets code) is
  // dynamically imported so it loads only on the notebook route.
  useEffect(() => {
    let dispose: (() => void) | null = null
    let cancelled = false
    void import("@/lib/widgets")
      .then(({ connectWidgets }) => {
        if (cancelled) return
        const { manager, close } = connectWidgets(notebookId)
        setWidgetManager(manager)
        dispose = close
      })
      .catch(() => {
        // ipywidgets support is optional and its manager bundle can fail to initialize
        // in some builds; a notebook without widgets is unaffected, so degrade quietly
        // instead of surfacing an unhandled rejection.
      })
    return () => {
      cancelled = true
      setWidgetManager(null)
      dispose?.()
    }
  }, [notebookId])

  const reactive = (view?.metadata?.reactive ?? true) as boolean

  // Streamed run events are buffered and flushed on an animation frame, not applied one at a
  // time. The `await reader.read()` loop that delivers them chains microtasks, and the browser
  // does not paint between microtasks, so applying each event inline (even with flushSync)
  // never renders until the run ends. Draining the buffer inside requestAnimationFrame runs in
  // the paint phase, so output appears live as it streams.
  const pendingEvents = useRef<Parameters<Parameters<typeof runNotebook>[2]>[0][]>([])
  const flushHandle = useRef<number | null>(null)

  const flushEvents = useCallback(() => {
    flushHandle.current = null
    const events = pendingEvents.current
    pendingEvents.current = []
    if (events.length === 0) return
    setView((cur) => {
      let next = cur
      for (const event of events) {
        if (next === null) break
        if (event.event === "cell_start") {
          next = patchCell(next, event.cell, {
            outputs: [],
            execution_count: null,
          })
        } else if (event.event === "output") {
          const cell = next.cells.find((c) => c.id === event.cell)
          next = patchCell(next, event.cell, {
            outputs: coalesce([...(cell?.outputs ?? []), event.output]),
          })
        } else if (event.event === "status") {
          next = patchCell(next, event.cell, {
            execution_count: event.execution_count,
          })
        }
      }
      return next
    })
    for (const event of events) {
      if (event.event === "cell_start") setRunning(event.cell)
      if (event.event === "status" || event.event === "done") setRunning(null)
      if (event.event === "stale") setStale(new Set(event.cells))
      if (event.event === "input_request") {
        setInputPrompt({
          cell: event.cell,
          text: event.prompt,
          password: event.password,
        })
      }
      // Any cell boundary clears a pending prompt: the request belongs to one cell's run.
      if (["cell_start", "status", "done", "error"].includes(event.event)) {
        setInputPrompt(null)
      }
    }
  }, [])

  const applyEvent = useCallback(
    (event: Parameters<Parameters<typeof runNotebook>[2]>[0]) => {
      pendingEvents.current.push(event)
      if (flushHandle.current === null) {
        flushHandle.current = requestAnimationFrame(flushEvents)
      }
    },
    [flushEvents],
  )

  const run = useCallback(
    async (body: { cells?: string[]; run_all?: boolean; fresh?: boolean }) => {
      if (busy) return
      setBusy(true)
      const { done, abort } = runNotebook(notebookId, body, applyEvent)
      abortRef.current = abort
      try {
        await done
      } finally {
        // Drop any unflushed frame so a late rAF can't repaint stale events over the
        // authoritative state that load() reconciles from the server.
        if (flushHandle.current !== null) {
          cancelAnimationFrame(flushHandle.current)
          flushHandle.current = null
        }
        pendingEvents.current = []
        setBusy(false)
        setRunning(null)
        abortRef.current = null
        void load() // reconcile persisted outputs, counts, and the fresh dependency graph
        void refreshVariables() // the run changed the kernel namespace
      }
    },
    [busy, notebookId, applyEvent, load, refreshVariables],
  )

  const onSource = useCallback(
    (cellId: string, source: string) => {
      setView((current) => (current ? patchCell(current, cellId, { source }) : current))
      clearTimeout(saveTimers.current[cellId])
      saveTimers.current[cellId] = setTimeout(() => {
        void updateCell(notebookId, cellId, { source })
      }, 400)
    },
    [notebookId],
  )

  const flushSave = useCallback(
    async (cellId: string, source: string) => {
      clearTimeout(saveTimers.current[cellId])
      await updateCell(notebookId, cellId, { source })
    },
    [notebookId],
  )

  const structural = useCallback(
    async (op: () => Promise<unknown>) => {
      await op()
      await load()
    },
    [load],
  )

  const toggleReactive = useCallback(async () => {
    if (!view) return
    const metadata = { ...view.metadata, reactive: !reactive }
    setView({ ...view, metadata })
    await updateNotebook(notebookId, { metadata })
  }, [view, reactive, notebookId])

  const rename = useCallback(
    async (name: string) => {
      await updateNotebook(notebookId, { name })
    },
    [notebookId],
  )

  const saveSchedule = useCallback(
    async (schedule: Schedule | null) => {
      await updateNotebook(notebookId, { schedule })
      await load()
    },
    [notebookId, load],
  )

  const saveEnvironment = useCallback(
    async (deps: string[], baseEnv: string | null): Promise<EnvironmentResult> => {
      const result = await updateNotebook(notebookId, {
        deps,
        base_env: baseEnv,
      })
      await load()
      return result
    },
    [notebookId, load],
  )

  const relock = useCallback(async (): Promise<EnvironmentResult> => {
    const result = await relockNotebook(notebookId)
    await load()
    return result
  }, [notebookId, load])

  const toggleParams = useCallback(
    (cell: Cell) => {
      const tags = ((cell.metadata.tags as string[]) ?? []).filter((t) => t !== "parameters")
      const next = (cell.metadata.tags as string[])?.includes("parameters")
        ? tags
        : [...tags, "parameters"]
      void structural(() =>
        updateCell(notebookId, cell.id, {
          metadata: { ...cell.metadata, tags: next },
        }),
      )
    },
    [notebookId, structural],
  )

  if (view === null) {
    if (missing) {
      return (
        <Scene>
          <SceneHeader backTo="/notebooks" backLabel="Notebooks" title="Not found" />
          <SceneBody>
            <p className="text-sm text-text-secondary">
              This notebook was deleted, or the link is out of date.
            </p>
          </SceneBody>
        </Scene>
      )
    }
    return <SceneSkeleton />
  }

  return (
    <WidgetManagerContext.Provider value={widgetManager}>
      <Scene>
        <Toolbar
          view={view}
          busy={busy}
          reactive={reactive}
          onRunAll={() => run({ run_all: true })}
          onRunFresh={() => run({ run_all: true, fresh: true })}
          onInterrupt={() => void interruptNotebook(notebookId)}
          onRestart={() => void restartNotebook(notebookId).then(load)}
          onToggleReactive={toggleReactive}
          onRename={rename}
          onSaveSchedule={saveSchedule}
          onSaveEnvironment={saveEnvironment}
          onRelock={relock}
          onModelTrained={() => navigate("/models")}
          showVariables={showVariables}
          variableCount={variables.length}
          onToggleVariables={() => {
            setShowVariables((open) => !open)
            void refreshVariables()
          }}
        />
        <div className="flex min-h-0 flex-1">
          <div className="min-w-0 flex-1 overflow-y-auto">
            <div className="mx-auto max-w-4xl px-6 py-8">
              {view.cells.map((cell, index) => (
                <CellRow
                  key={cell.id}
                  cell={cell}
                  notebookId={notebookId}
                  dark={dark}
                  deps={view.graph.cells[cell.id]}
                  stale={stale.has(cell.id)}
                  running={running === cell.id}
                  queued={busy && running !== cell.id && stale.has(cell.id)}
                  focused={focused === cell.id}
                  onFocus={() => setFocused(cell.id)}
                  onSource={(source) => onSource(cell.id, source)}
                  onRun={async () => {
                    await flushSave(cell.id, cell.source)
                    void run({ cells: [cell.id] })
                  }}
                  onAddBelow={(cellType) =>
                    void structural(() =>
                      addCell(notebookId, {
                        after: cell.id,
                        cell_type: cellType,
                      }),
                    )
                  }
                  onDelete={() => void structural(() => deleteCell(notebookId, cell.id))}
                  onMove={(dir) => {
                    const order = view.cells.map((c) => c.id)
                    const target = index + dir
                    if (target < 0 || target >= order.length) return
                    ;[order[index], order[target]] = [order[target], order[index]]
                    void structural(() => reorderCells(notebookId, order))
                  }}
                  onSetType={(cellType) =>
                    void structural(() => updateCell(notebookId, cell.id, { cell_type: cellType }))
                  }
                  onToggleParams={() => toggleParams(cell)}
                />
              ))}
              {inputPrompt ? (
                <form
                  className="mt-3 flex items-center gap-2 rounded-md border border-info/50 bg-info-tint px-3 py-2"
                  onSubmit={(e) => {
                    e.preventDefault()
                    void sendInput(notebookId, inputValue)
                    setInputPrompt(null)
                    setInputValue("")
                  }}
                >
                  <span className="font-mono text-sm text-info">
                    {inputPrompt.text || "Input:"}
                  </span>
                  <Input
                    autoFocus
                    type={inputPrompt.password ? "password" : "text"}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    className="h-8 flex-1"
                    aria-label="Cell input"
                  />
                  <Button type="submit" size="sm">
                    Submit
                  </Button>
                </form>
              ) : null}
              <div className="mt-3 flex justify-center">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void structural(() => addCell(notebookId, { cell_type: "code" }))}
                >
                  <Plus className="mr-1.5 size-4" /> Add cell
                </Button>
              </div>
            </div>
          </div>
          {showVariables ? (
            <VariablesPanel
              variables={variables}
              onRefresh={refreshVariables}
              onClose={() => setShowVariables(false)}
            />
          ) : null}
        </div>
      </Scene>
    </WidgetManagerContext.Provider>
  )
}

function VariablesPanel({
  variables,
  onRefresh,
  onClose,
}: {
  variables: NotebookVariable[]
  onRefresh: () => void
  onClose: () => void
}) {
  return (
    <aside className="flex w-72 shrink-0 flex-col border-l border-border bg-card/40">
      <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
        <span className="flex items-center gap-1.5 text-sm font-medium">
          <Variable className="size-4 text-text-tertiary" /> Variables
          <span className="text-xs text-text-tertiary">{variables.length}</span>
        </span>
        <div className="flex items-center gap-0.5">
          <Button variant="ghost" size="icon-xs" title="Refresh" onClick={onRefresh}>
            <RefreshCw className="size-3.5" />
          </Button>
          <Button variant="ghost" size="icon-xs" title="Close" onClick={onClose}>
            <X className="size-3.5" />
          </Button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {variables.length === 0 ? (
          <p className="px-3 py-4 text-xs text-text-tertiary">
            No variables yet. Run a cell and what it defines shows up here.
          </p>
        ) : (
          <ul className="divide-y divide-border/60">
            {variables.map((v) => (
              <li key={v.name} className="px-3 py-2">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-compact font-medium text-foreground">
                    {v.name}
                  </span>
                  <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-3xs text-text-tertiary">
                    {v.type}
                  </span>
                </div>
                {v.summary ? (
                  <div className="mt-0.5 truncate font-mono text-2xs text-text-secondary">
                    {v.summary}
                  </div>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  )
}

function Toolbar({
  view,
  busy,
  reactive,
  onRunAll,
  onRunFresh,
  onInterrupt,
  onRestart,
  onToggleReactive,
  onRename,
  onSaveSchedule,
  onSaveEnvironment,
  onRelock,
  onModelTrained,
  showVariables,
  variableCount,
  onToggleVariables,
}: {
  view: NotebookView
  busy: boolean
  reactive: boolean
  showVariables: boolean
  variableCount: number
  onToggleVariables: () => void
  onRunAll: () => void
  onRunFresh: () => void
  onInterrupt: () => void
  onRestart: () => void
  onToggleReactive: () => void
  onRename: (name: string) => void
  onSaveSchedule: (schedule: Schedule | null) => void
  onSaveEnvironment: (deps: string[], baseEnv: string | null) => Promise<EnvironmentResult>
  onRelock: () => Promise<EnvironmentResult>
  onModelTrained: () => void
}) {
  const [name, setName] = useState(view.name)
  const [scheduleOpen, setScheduleOpen] = useState(false)
  const [envOpen, setEnvOpen] = useState(false)
  const [computeOpen, setComputeOpen] = useState(false)
  const [trainOpen, setTrainOpen] = useState(false)
  useEffect(() => setName(view.name), [view.name])
  return (
    <>
      <SceneHeader
        backTo={
          view.folder_id ? `/notebooks?folder=${encodeURIComponent(view.folder_id)}` : "/notebooks"
        }
        backLabel={view.folder_name ?? "Notebooks"}
        icon={<NotebookPen className="size-5" />}
        mono
        title={
          <Input
            aria-label="Notebook name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onBlur={() => name !== view.name && onRename(name)}
            className="h-auto rounded-none border-0 bg-transparent p-0 text-lg font-semibold tracking-tight focus-visible:ring-0"
          />
        }
        actions={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <span
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${
                busy
                  ? "bg-info-tint text-info"
                  : view.kernel_status === "running"
                    ? "bg-success-tint text-success"
                    : "bg-muted text-text-tertiary"
              }`}
            >
              <span
                className={`size-1.5 rounded-full ${
                  busy
                    ? "bg-info"
                    : view.kernel_status === "running"
                      ? "bg-success"
                      : "bg-muted-foreground/50"
                }`}
              />
              {busy ? "running" : view.kernel_status === "running" ? "kernel ready" : "idle"}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={onToggleReactive}
              title={
                reactive
                  ? "Reactive: a cell reruns its dependents automatically"
                  : "Manual: cells run only when you run them"
              }
            >
              <Zap
                className={`mr-1.5 size-4 ${reactive ? "text-primary" : "text-text-tertiary"}`}
              />
              {reactive ? "Reactive" : "Manual"}
            </Button>
            <Button
              variant={showVariables ? "secondary" : "ghost"}
              size="sm"
              onClick={onToggleVariables}
              title="Show the live kernel's variables"
            >
              <Variable className="mr-1.5 size-4" /> Variables
              {variableCount ? (
                <span className="ml-1.5 text-xs text-text-tertiary">{variableCount}</span>
              ) : null}
            </Button>
            {busy ? (
              <Button variant="outline" size="sm" onClick={onInterrupt}>
                <Square className="mr-1.5 size-3.5" /> Interrupt
              </Button>
            ) : (
              <SplitButton
                size="sm"
                onClick={onRunAll}
                menuLabel="More run options"
                menu={
                  <>
                    <DropdownMenuItem onClick={onRunFresh}>
                      <RotateCcw className="size-4" /> Restart &amp; run all
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={onRestart}>
                      <Power className="size-4" /> Restart kernel
                    </DropdownMenuItem>
                  </>
                }
              >
                <CirclePlay className="mr-1.5 size-4" /> Run all
              </SplitButton>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon-sm" title="More actions">
                  <MoreHorizontal className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                <DropdownMenuItem onClick={() => setEnvOpen(true)}>
                  <Boxes className="size-4" /> Environment
                  <span className="ml-auto text-xs text-text-tertiary">
                    {view.deps.length + (view.environment.base_env ? 1 : 0) || 0} deps
                  </span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setComputeOpen(true)}>
                  <Cpu className="size-4" /> Compute
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTrainOpen(true)}>
                  <Sparkles className="size-4" /> Train model
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setScheduleOpen(true)}>
                  <CalendarClock
                    className={`size-4 ${view.schedule?.enabled ? "text-primary" : ""}`}
                  />
                  Schedule reruns
                  {view.schedule?.enabled ? (
                    <span className="ml-auto text-xs text-primary">on</span>
                  ) : null}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <a href={exportNotebookUrl(view.id)} download>
                    <Download className="size-4" /> Download .ipynb
                  </a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        }
      />
      <ScheduleDialog
        // Remount on each open, and whenever the saved schedule changes underneath.
        key={`${scheduleOpen}:${JSON.stringify(view.schedule ?? null)}`}
        open={scheduleOpen}
        onOpenChange={setScheduleOpen}
        current={(view.schedule as Schedule | null) ?? null}
        onSave={(schedule) => {
          onSaveSchedule(schedule)
          setScheduleOpen(false)
        }}
      />
      <EnvironmentDialog
        open={envOpen}
        onOpenChange={setEnvOpen}
        view={view}
        onSave={onSaveEnvironment}
        onRelock={onRelock}
      />
      <ComputeDialog notebookId={view.id} open={computeOpen} onOpenChange={setComputeOpen} />
      <TrainModelDialog
        open={trainOpen}
        onOpenChange={setTrainOpen}
        onStarted={() => {
          setTrainOpen(false)
          onModelTrained()
        }}
      />
    </>
  )
}

function TrainModelDialog({
  open,
  onOpenChange,
  onStarted,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onStarted: () => void
}) {
  const [sources, setSources] = useState<FeatureSource[]>([])
  const [source, setSource] = useState("")
  const [columns, setColumns] = useState<DatasetColumn[]>([])
  const [name, setName] = useState("")
  const [target, setTarget] = useState("")
  const [task, setTask] = useState<TrainRequest["task"]>("auto")
  const [engine, setEngine] = useState<"flaml" | "autogluon">("flaml")
  const [minutes, setMinutes] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      void getFeatureSources().then(setSources)
      setError(null)
    }
  }, [open])

  const colon = source.indexOf(":")
  const kind = colon >= 0 ? source.slice(0, colon) : ""
  const sourceName = colon >= 0 ? source.slice(colon + 1) : ""
  useEffect(() => {
    if (kind === "dataset" && sourceName) void getDatasetColumns(sourceName).then(setColumns)
    else setColumns([])
    setTarget("")
  }, [kind, sourceName])

  const submit = async () => {
    setBusy(true)
    setError(null)
    const req: TrainRequest = {
      name: name.trim(),
      target: target.trim(),
      task,
      engine,
      time_budget: Math.max(5, Math.round(minutes * 60)),
      ...(kind === "dataset" ? { dataset: sourceName } : { derivation: sourceName }),
    }
    const result = await trainModel(req)
    setBusy(false)
    if ("error" in result) setError(result.error)
    else onStarted()
  }

  const ready = name.trim() && sourceName && target.trim()
  const datasets = sources.filter((s) => s.kind === "dataset")
  const derivations = sources.filter((s) => s.kind === "derivation")
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Train a model</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 text-sm">
          <p className="text-xs text-text-tertiary">
            Trains through the platform's AutoML on the server, so the model passes the prediction
            gate and lands in the Models page: governed, with its evidence and lineage. Train on a
            dataset, or on a feature derivation you promoted here.
          </p>
          <FormField label="Model name">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="churn"
              className="h-7.5"
            />
          </FormField>
          <FormField label="Training source">
            <Select value={source || undefined} onValueChange={setSource}>
              <FormControl>
                <SelectTrigger size="sm" className="w-full px-2 data-[size=sm]:h-7.5">
                  <SelectValue placeholder="Select a dataset or derivation…" />
                </SelectTrigger>
              </FormControl>
              <SelectContent>
                {datasets.length > 0 && (
                  <SelectGroup>
                    <SelectLabel>Datasets</SelectLabel>
                    {datasets.map((s) => (
                      <SelectItem key={`dataset:${s.name}`} value={`dataset:${s.name}`}>
                        {s.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
                {derivations.length > 0 && (
                  <SelectGroup>
                    <SelectLabel>Feature derivations</SelectLabel>
                    {derivations.map((s) => (
                      <SelectItem key={`derivation:${s.name}`} value={`derivation:${s.name}`}>
                        {s.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
              </SelectContent>
            </Select>
          </FormField>
          <FormField label="Target column">
            {columns.length > 0 ? (
              <Select value={target || undefined} onValueChange={setTarget}>
                <FormControl>
                  <SelectTrigger size="sm" className="w-full px-2 data-[size=sm]:h-7.5">
                    <SelectValue placeholder="Select the column to predict…" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {columns.map((c) => (
                    <SelectItem key={c.name} value={c.name}>
                      {c.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="the column to predict"
                className="h-7.5"
              />
            )}
          </FormField>
          <div className="flex gap-2">
            <FormField label="Task" className="flex-1">
              <Select value={task} onValueChange={(v) => setTask(v as TrainRequest["task"])}>
                <FormControl>
                  <SelectTrigger size="sm" className="w-full px-2 data-[size=sm]:h-7.5">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="classification">Classification</SelectItem>
                  <SelectItem value="regression">Regression</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
            <FormField label="Engine" className="flex-1">
              <Select value={engine} onValueChange={(v) => setEngine(v as "flaml" | "autogluon")}>
                <FormControl>
                  <SelectTrigger size="sm" className="w-full px-2 data-[size=sm]:h-7.5">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="flaml">FLAML (fast)</SelectItem>
                  <SelectItem value="autogluon">AutoGluon (accuracy)</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
            <FormField label="Budget (min)" className="w-24">
              <Input
                type="number"
                min={1}
                value={minutes}
                onChange={(e) => setMinutes(Number(e.target.value))}
                className="h-7.5"
              />
            </FormField>
          </div>
          {error ? <p className="text-xs text-danger">{error}</p> : null}
        </div>
        <DialogFooter>
          <Button size="sm" disabled={busy || !ready} onClick={submit}>
            {busy ? "Starting…" : "Train & register"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function EnvironmentDialog({
  open,
  onOpenChange,
  view,
  onSave,
  onRelock,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  view: NotebookView
  onSave: (deps: string[], baseEnv: string | null) => Promise<EnvironmentResult>
  onRelock: () => Promise<EnvironmentResult>
}) {
  const baseEnvId = useId()
  const [depsText, setDepsText] = useState(view.deps.join("\n"))
  const [baseEnv, setBaseEnv] = useState(view.environment.base_env ?? "")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (open) {
      setDepsText(view.deps.join("\n"))
      setBaseEnv(view.environment.base_env ?? "")
      setError(null)
    }
  }, [open, view])

  const parseDeps = () =>
    depsText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)

  const run = async (fn: () => Promise<EnvironmentResult>) => {
    setBusy(true)
    setError(null)
    try {
      const result = await fn()
      if (!result.ok) setError(result.error ?? "Could not resolve the environment")
      else onOpenChange(false)
    } finally {
      setBusy(false)
    }
  }

  const lock = view.environment.lock
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Environment &amp; packages</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 text-sm">
          <p className="text-xs text-text-tertiary">
            Declare the packages this notebook's kernel needs: one per line, e.g.
            <code className="mx-1">scikit-learn==1.5.2</code>. Saving resolves and pins them, then
            restarts the kernel. You can also install from a cell with
            <code className="mx-1">%pip install &lt;pkg&gt;</code>.
          </p>
          {view.environment.base_environments.length > 0 ? (
            <div className="flex items-center gap-2">
              <Label htmlFor={baseEnvId}>Base environment</Label>
              <Select
                value={baseEnv || NO_BASE_ENV}
                onValueChange={(v) => setBaseEnv(v === NO_BASE_ENV ? "" : v)}
              >
                <SelectTrigger
                  id={baseEnvId}
                  size="sm"
                  className="flex-1 px-2 data-[size=sm]:h-7.5"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_BASE_ENV}>None</SelectItem>
                  {view.environment.base_environments.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}
          <Textarea
            aria-label="Packages"
            value={depsText}
            onChange={(e) => setDepsText(e.target.value)}
            rows={6}
            spellCheck={false}
            placeholder="pandas&#10;scikit-learn==1.5.2"
            // A fixed six-row box rather than Textarea's grow-to-fit.
            className="field-sizing-fixed p-2 font-mono text-compact"
          />
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 text-text-tertiary">
              {view.environment.locked ? (
                <>
                  <Lock className="size-3.5 text-success" />
                  Locked · {lock?.length ?? 0} pinned packages
                </>
              ) : (
                "Not locked yet"
              )}
            </span>
            <Button
              variant="link"
              onClick={() => run(onRelock)}
              disabled={busy}
              className="h-auto gap-1 p-0 text-xs font-normal text-text-tertiary hover:text-foreground has-[>svg]:px-0"
            >
              <RefreshCw className={cn("size-3.5", busy && "animate-spin")} />
              Re-lock
            </Button>
          </div>
          {error ? <p className="text-xs text-danger">{error}</p> : null}
        </div>
        <DialogFooter>
          <Button
            size="sm"
            disabled={busy}
            onClick={() => run(() => onSave(parseDeps(), baseEnv || null))}
          >
            {busy ? "Resolving…" : "Save & lock"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ScheduleDialog({
  open,
  onOpenChange,
  current,
  onSave,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  current: Schedule | null
  onSave: (schedule: Schedule | null) => void
}) {
  // Initial values only: the caller keys this dialog on the schedule it is opened
  // with, so a reopen -- or a different schedule -- is a fresh mount rather than a
  // mount followed by an effect that resets what the mount just set.
  const [enabled, setEnabled] = useState(current?.enabled ?? false)
  const [mode, setMode] = useState<Schedule["mode"]>(current?.mode ?? "interval")
  const [hours, setHours] = useState(current?.interval_hours ?? 24)
  const [dataset, setDataset] = useState(current?.dataset ?? "")
  const enabledId = useId()
  const modes: { value: Schedule["mode"]; label: string }[] = [
    { value: "interval", label: "On an interval" },
    { value: "on_data_change", label: "When a dataset changes" },
  ]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Schedule reruns</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 text-sm">
          <div className="flex items-center gap-2">
            <Checkbox
              id={enabledId}
              checked={enabled}
              onCheckedChange={(c) => setEnabled(c === true)}
            />
            <Label htmlFor={enabledId} className="font-normal">
              Run this notebook automatically
            </Label>
          </div>
          <div className="flex gap-2">
            {modes.map((m) => (
              <Button
                key={m.value}
                variant="outline"
                aria-pressed={mode === m.value}
                onClick={() => setMode(m.value)}
                className={cn(
                  "h-auto flex-1 justify-start px-3 py-2 text-xs font-normal",
                  mode === m.value && "border-foreground/40 bg-muted hover:bg-muted",
                )}
              >
                {m.label}
              </Button>
            ))}
          </div>
          {mode === "interval" ? (
            <label className="flex items-center gap-2">
              Every
              <Input
                type="number"
                min={1}
                value={hours}
                onChange={(e) => setHours(Number(e.target.value))}
                className="h-7.5 w-20"
              />
              hours
            </label>
          ) : (
            <label className="flex items-center gap-2">
              Dataset
              <Input
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                placeholder="dataset name"
                className="h-7.5 flex-1"
              />
            </label>
          )}
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => onSave(null)}>
            Clear
          </Button>
          <Button
            size="sm"
            onClick={() =>
              onSave({
                enabled,
                mode,
                interval_hours: hours,
                dataset: dataset || undefined,
              })
            }
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function CellRow({
  cell,
  notebookId,
  dark,
  deps,
  stale,
  running,
  queued,
  focused,
  onFocus,
  onSource,
  onRun,
  onAddBelow,
  onDelete,
  onMove,
  onSetType,
  onToggleParams,
}: {
  cell: Cell
  notebookId: string
  dark: boolean
  deps?: { defs: string[]; refs: string[]; syntax_error: string | null }
  stale: boolean
  running: boolean
  queued: boolean
  focused: boolean
  onFocus: () => void
  onSource: (source: string) => void
  onRun: () => void
  onAddBelow: (cellType: CellType) => void
  onDelete: () => void
  onMove: (dir: -1 | 1) => void
  onSetType: (cellType: CellType) => void
  onToggleParams: () => void
}) {
  const isCode = cell.cell_type === "code"
  const isParams = ((cell.metadata.tags as string[]) ?? []).includes("parameters")
  // How the last run reached its data. Shown because the crossing between "the work ran
  // in the warehouse" and "the rows are in this kernel" is what decides whether a
  // notebook scales, and a crossing nobody can see only moves the failure later.
  const ours = cell.metadata.elbi as { data_mode?: string; queries?: CellQuery[] } | undefined
  const dataMode = ours?.data_mode
  const cellQueries = ours?.queries ?? []
  const [editingMd, setEditingMd] = useState(cell.source.trim() === "")
  const [promoting, setPromoting] = useState(false)
  const [promoted, setPromoted] = useState<PromoteResult | null>(null)
  const label = running ? "*" : queued ? "…" : (cell.execution_count ?? " ")

  // A cell can be promoted to a derivation only when it defines `def <name>(ctx): ...`.
  const promotable = isCode && /def\s+\w+\s*\(\s*ctx\b/.test(cell.source)
  const onPromote = async () => {
    setPromoting(true)
    try {
      setPromoted(await promoteCell(notebookId, cell.id))
    } finally {
      setPromoting(false)
    }
  }

  return (
    <div
      className={`group relative mb-3 overflow-hidden rounded-xl border transition ${
        running
          ? "border-info/60 bg-info-tint/30 ring-1 ring-info/30"
          : isCode
            ? focused
              ? "border-border bg-card"
              : "border-border/70 bg-card hover:border-border"
            : focused
              ? "border-border/60 bg-transparent"
              : "border-transparent bg-transparent hover:border-border/50"
      } ${stale && !running ? "ring-1 ring-warning/40" : ""}`}
    >
      <div className="flex">
        {/* Run gutter: click to run this cell; shows the execution count or a spinner. */}
        <div className="flex w-12 shrink-0 flex-col items-center pt-2.5">
          {isCode ? (
            <IconButton
              label="Run cell (Shift+Enter)"
              size="icon-xs"
              onClick={onRun}
              className="size-7 text-text-tertiary"
            >
              {running ? (
                <Loader2 className="size-4 animate-spin text-info" />
              ) : (
                <Play className="size-4 opacity-70 group-hover:opacity-100" />
              )}
            </IconButton>
          ) : null}
          {isCode ? (
            <span
              className={`mt-1 font-mono text-3xs ${
                running ? "text-info" : "text-text-tertiary/60"
              }`}
            >
              [{label}]
            </span>
          ) : null}
        </div>

        <div className="min-w-0 flex-1 py-1">
          {isParams || dataMode ? (
            <div className="flex gap-1.5 px-3 pb-1">
              {isParams ? (
                <span className="rounded-full bg-info-tint px-2 py-0.5 text-3xs font-medium text-info">
                  parameters
                </span>
              ) : null}
              {dataMode ? <DataModePill mode={dataMode} /> : null}
            </div>
          ) : null}
          {isCode || editingMd ? (
            <CellEditor
              value={cell.source}
              language={isCode ? "python" : "markdown"}
              dark={dark}
              onChange={onSource}
              // A code cell runs on the kernel; a markdown cell "runs" by rendering,
              // the way Shift-Enter commits a markdown cell in Jupyter.
              onRun={isCode ? onRun : () => setEditingMd(false)}
              onRunNoAdvance={isCode ? onRun : () => setEditingMd(false)}
              onFocus={onFocus}
              complete={isCode ? (code, pos) => completeCell(notebookId, code, pos) : undefined}
              inspect={isCode ? (code, pos) => inspectCell(notebookId, code, pos) : undefined}
            />
          ) : (
            // eslint-disable-next-line ds/no-raw-element -- the rendered markdown is the click target; Button's nowrap, fixed height and svg sizing would restyle the cell's content
            <button
              type="button"
              aria-label="Edit markdown"
              className="w-full cursor-text px-3 py-2 text-left"
              onClick={() => setEditingMd(true)}
            >
              {cell.source ? (
                <NotebookMarkdown>{cell.source}</NotebookMarkdown>
              ) : (
                <span className="text-text-tertiary">Empty markdown cell: click to edit</span>
              )}
            </button>
          )}
          {deps?.syntax_error && isCode ? (
            <div className="px-3 pb-1 text-2xs text-danger">{deps.syntax_error}</div>
          ) : null}
          {isCode && (deps?.refs.length || deps?.defs.length) ? (
            <div className="flex flex-wrap gap-x-3 px-3 pb-1 text-3xs text-text-tertiary/60">
              {deps?.defs.length ? <span>defines {deps.defs.join(", ")}</span> : null}
              {deps?.refs.length ? <span>uses {deps.refs.join(", ")}</span> : null}
              {stale ? <span className="text-warning">stale</span> : null}
            </div>
          ) : null}
        </div>

        {/* Per-cell actions, revealed on hover. */}
        <div className="flex shrink-0 items-start gap-0.5 p-1 opacity-0 transition group-hover:opacity-100">
          {promotable ? (
            <CellAction label="Promote to a certified derivation" onClick={onPromote}>
              {promoting ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <ShieldCheck className="size-3.5" />
              )}
            </CellAction>
          ) : null}
          {isCode ? (
            <CellAction
              label={isParams ? "Unmark parameters cell" : "Mark as parameters cell"}
              onClick={onToggleParams}
            >
              <SlidersHorizontal className={`size-3.5 ${isParams ? "text-primary" : ""}`} />
            </CellAction>
          ) : null}
          <CellAction label="Move up" onClick={() => onMove(-1)}>
            <ChevronUp className="size-3.5" />
          </CellAction>
          <CellAction label="Move down" onClick={() => onMove(1)}>
            <ChevronDown className="size-3.5" />
          </CellAction>
          <CellAction
            label={isCode ? "Make markdown" : "Make code"}
            onClick={() => onSetType(isCode ? "markdown" : "code")}
          >
            <span className="text-3xs font-medium">{isCode ? "M↓" : "{ }"}</span>
          </CellAction>
          <CellAction label="Delete cell" onClick={onDelete}>
            <Trash2 className="size-3.5" />
          </CellAction>
        </div>
      </div>

      {isCode ? <CellOutputs outputs={cell.outputs} /> : null}
      {isCode ? <CellQueries queries={cellQueries} /> : null}

      {promoted ? (
        <div
          className={`mx-3 mb-2 rounded-md border px-3 py-2 text-xs ${
            promoted.certified
              ? "border-verified/30 bg-verified-tint text-verified"
              : "border-warning/30 bg-warning-tint text-warning"
          }`}
        >
          {promoted.certified ? (
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="size-3.5" />
              Certified as derivation{" "}
              <Link
                to={`/derivations/${encodeURIComponent(promoted.name ?? "")}`}
                className="font-mono underline"
              >
                {promoted.name}
              </Link>
              {promoted.verdict ? ` · ${promoted.verdict}` : ""}
            </span>
          ) : (
            <span>Not certified: {promoted.error || promoted.detail || promoted.verdict}</span>
          )}
        </div>
      ) : null}

      {/* Insert affordance between cells. */}
      <div className="flex justify-center gap-1 pb-1 opacity-0 transition group-hover:opacity-100">
        <Button
          variant="ghost"
          size="xs"
          onClick={() => onAddBelow("code")}
          className="h-auto px-2 py-0.5 text-3xs font-normal text-text-tertiary"
        >
          + code
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={() => onAddBelow("markdown")}
          className="h-auto px-2 py-0.5 text-3xs font-normal text-text-tertiary"
        >
          + text
        </Button>
      </div>
    </div>
  )
}

// A cell's hover action: a 24px icon button in the muted cell chrome.
function CellAction({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <IconButton label={label} size="icon-xs" onClick={onClick} className="text-text-tertiary">
      {children}
    </IconButton>
  )
}

// -- local helpers -------------------------------------------------------------------

function patchCell(view: NotebookView, cellId: string, fields: Partial<Cell>): NotebookView {
  return {
    ...view,
    cells: view.cells.map((c) => (c.id === cellId ? { ...c, ...fields } : c)),
  }
}

// Merge adjacent same-stream outputs so live streaming shows one growing block, not many.
function coalesce(outputs: Output[]): Output[] {
  const merged: Output[] = []
  for (const output of outputs) {
    const last = merged[merged.length - 1]
    if (
      output.output_type === "stream" &&
      last?.output_type === "stream" &&
      last.name === output.name
    ) {
      merged[merged.length - 1] = {
        ...last,
        text: (last.text ?? "") + (output.text ?? ""),
      }
    } else {
      merged.push(output)
    }
  }
  return merged
}
