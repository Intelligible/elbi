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
import { CellEditor } from "@/components/notebook/CellEditor"
import { CellOutputs } from "@/components/notebook/CellOutput"
import { CellQueries } from "@/components/notebook/CellQueries"
import { ComputeDialog } from "@/components/notebook/ComputeDialog"
import { DataModePill } from "@/components/notebook/DataModePill"
import { NotebookMarkdown } from "@/components/notebook/NotebookMarkdown"
import { WidgetManagerContext } from "@/components/notebook/WidgetView"
import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { Button } from "@/components/ui/button"
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
import { SplitButton } from "@/components/ui/split-button"
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
  isSetupCell,
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
import type { NotebookWidgetManager } from "@/lib/widgets"

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
    prompt: string
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
  // The environment a seeded derivation needs is bound by a cell that runs but is not
  // drawn: imports and a contract are not what anyone opened the notebook to edit.
  const visibleCells = useMemo(
    () => (view?.cells ?? []).filter((cell) => !isSetupCell(cell)),
    [view],
  )

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
          prompt: event.prompt,
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
              {visibleCells.map((cell, index) => (
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
                    // Swap with the neighbour that is drawn, in the full order: a
                    // hidden setup cell is not a position anyone can move through.
                    const neighbour = visibleCells[index + dir]
                    if (!neighbour) return
                    const order = view.cells.map((c) => c.id)
                    const from = order.indexOf(cell.id)
                    const target = order.indexOf(neighbour.id)
                    ;[order[from], order[target]] = [order[target], order[from]]
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
                    {inputPrompt.prompt || "Input:"}
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
          <Variable className="size-4 text-muted-foreground" /> Variables
          <span className="text-xs text-muted-foreground">{variables.length}</span>
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
          <p className="px-3 py-4 text-xs text-muted-foreground">
            No variables yet. Run a cell and what it defines shows up here.
          </p>
        ) : (
          <ul className="divide-y divide-border/60">
            {variables.map((v) => (
              <li key={v.name} className="px-3 py-2">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-[0.8125rem] font-medium text-foreground">
                    {v.name}
                  </span>
                  <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                    {v.type}
                  </span>
                </div>
                {v.summary ? (
                  <div className="mt-0.5 truncate font-mono text-[11px] text-text-secondary">
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
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onBlur={() => name !== view.name && onRename(name)}
            className="w-full min-w-0 bg-transparent font-semibold tracking-tight outline-none"
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
                    : "bg-muted text-muted-foreground"
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
                className={`mr-1.5 size-4 ${reactive ? "text-primary" : "text-muted-foreground"}`}
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
                <span className="ml-1.5 text-xs text-muted-foreground">{variableCount}</span>
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
                  <span className="ml-auto text-xs text-muted-foreground">
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
  const targetId = useId()
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
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Train a model</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 text-sm">
          <p className="text-xs text-muted-foreground">
            Trains through the platform's AutoML on the server, so the model passes the prediction
            gate and lands in the Models page: governed, with its evidence and lineage. Train on a
            dataset, or on a feature derivation you promoted here.
          </p>
          <label className="block space-y-1">
            <span className="text-xs font-medium">Model name</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="churn"
              className="w-full rounded-md border border-border bg-transparent px-2 py-1"
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs font-medium">Training source</span>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="w-full rounded-md border border-border bg-transparent px-2 py-1"
            >
              <option value="">Select a dataset or derivation…</option>
              {sources.filter((s) => s.kind === "dataset").length > 0 && (
                <optgroup label="Datasets">
                  {sources
                    .filter((s) => s.kind === "dataset")
                    .map((s) => (
                      <option key={`dataset:${s.name}`} value={`dataset:${s.name}`}>
                        {s.name}
                      </option>
                    ))}
                </optgroup>
              )}
              {sources.filter((s) => s.kind === "derivation").length > 0 && (
                <optgroup label="Feature derivations">
                  {sources
                    .filter((s) => s.kind === "derivation")
                    .map((s) => (
                      <option key={`derivation:${s.name}`} value={`derivation:${s.name}`}>
                        {s.name}
                      </option>
                    ))}
                </optgroup>
              )}
            </select>
          </label>
          <label className="block space-y-1" htmlFor={targetId}>
            <span className="text-xs font-medium">Target column</span>
            {columns.length > 0 ? (
              <select
                id={targetId}
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="w-full rounded-md border border-border bg-transparent px-2 py-1"
              >
                <option value="">Select the column to predict…</option>
                {columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id={targetId}
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="the column to predict"
                className="w-full rounded-md border border-border bg-transparent px-2 py-1"
              />
            )}
          </label>
          <div className="flex gap-2">
            <label className="flex-1 space-y-1">
              <span className="text-xs font-medium">Task</span>
              <select
                value={task}
                onChange={(e) => setTask(e.target.value as TrainRequest["task"])}
                className="w-full rounded-md border border-border bg-transparent px-2 py-1"
              >
                <option value="auto">Auto</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </label>
            <label className="flex-1 space-y-1">
              <span className="text-xs font-medium">Engine</span>
              <select
                value={engine}
                onChange={(e) => setEngine(e.target.value as "flaml" | "autogluon")}
                className="w-full rounded-md border border-border bg-transparent px-2 py-1"
              >
                <option value="flaml">FLAML (fast)</option>
                <option value="autogluon">AutoGluon (accuracy)</option>
              </select>
            </label>
            <label className="w-24 space-y-1">
              <span className="text-xs font-medium">Budget (min)</span>
              <input
                type="number"
                min={1}
                value={minutes}
                onChange={(e) => setMinutes(Number(e.target.value))}
                className="w-full rounded-md border border-border bg-transparent px-2 py-1"
              />
            </label>
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
          <p className="text-xs text-muted-foreground">
            Declare the packages this notebook's kernel needs: one per line, e.g.
            <code className="mx-1">scikit-learn==1.5.2</code>. Saving resolves and pins them, then
            restarts the kernel. You can also install from a cell with
            <code className="mx-1">%pip install &lt;pkg&gt;</code>.
          </p>
          {view.environment.base_environments.length > 0 ? (
            <label className="flex items-center gap-2">
              Base environment
              <select
                value={baseEnv}
                onChange={(e) => setBaseEnv(e.target.value)}
                className="flex-1 rounded-md border border-border bg-transparent px-2 py-1"
              >
                <option value="">None</option>
                {view.environment.base_environments.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <textarea
            value={depsText}
            onChange={(e) => setDepsText(e.target.value)}
            rows={6}
            spellCheck={false}
            placeholder="pandas&#10;scikit-learn==1.5.2"
            className="w-full rounded-md border border-border bg-transparent p-2 font-mono text-[13px]"
          />
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              {view.environment.locked ? (
                <>
                  <Lock className="size-3.5 text-success" />
                  Locked · {lock?.length ?? 0} pinned packages
                </>
              ) : (
                "Not locked yet"
              )}
            </span>
            <button
              type="button"
              onClick={() => run(onRelock)}
              disabled={busy}
              className="flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              <RefreshCw className={`size-3.5 ${busy ? "animate-spin" : ""}`} />
              Re-lock
            </button>
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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Schedule reruns</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 text-sm">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            Run this notebook automatically
          </label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setMode("interval")}
              className={`flex-1 rounded-md border px-3 py-2 text-left text-xs ${
                mode === "interval" ? "border-foreground/40 bg-muted" : "border-border"
              }`}
            >
              On an interval
            </button>
            <button
              type="button"
              onClick={() => setMode("on_data_change")}
              className={`flex-1 rounded-md border px-3 py-2 text-left text-xs ${
                mode === "on_data_change" ? "border-foreground/40 bg-muted" : "border-border"
              }`}
            >
              When a dataset changes
            </button>
          </div>
          {mode === "interval" ? (
            <label className="flex items-center gap-2">
              Every
              <input
                type="number"
                min={1}
                value={hours}
                onChange={(e) => setHours(Number(e.target.value))}
                className="w-20 rounded-md border border-border bg-transparent px-2 py-1"
              />
              hours
            </label>
          ) : (
            <label className="flex items-center gap-2">
              Dataset
              <input
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                placeholder="dataset name"
                className="flex-1 rounded-md border border-border bg-transparent px-2 py-1"
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
          ? "border-info/60 bg-info-tint/30 shadow-sm ring-1 ring-info/30"
          : isCode
            ? focused
              ? "border-border bg-card shadow-sm"
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
            <button
              type="button"
              onClick={onRun}
              className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground"
              title="Run cell (Shift+Enter)"
            >
              {running ? (
                <Loader2 className="size-4 animate-spin text-info" />
              ) : (
                <Play className="size-4 opacity-70 group-hover:opacity-100" />
              )}
            </button>
          ) : null}
          {isCode ? (
            <span
              className={`mt-1 font-mono text-[10px] ${
                running ? "text-info" : "text-muted-foreground/60"
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
                <span className="rounded-full bg-info-tint px-2 py-0.5 text-[10px] font-medium text-info">
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
            <button
              type="button"
              aria-label="Edit markdown"
              className="w-full cursor-text px-3 py-2 text-left"
              onClick={() => setEditingMd(true)}
            >
              {cell.source ? (
                <NotebookMarkdown>{cell.source}</NotebookMarkdown>
              ) : (
                <span className="text-muted-foreground">Empty markdown cell: click to edit</span>
              )}
            </button>
          )}
          {deps?.syntax_error && isCode ? (
            <div className="px-3 pb-1 text-[11px] text-danger">{deps.syntax_error}</div>
          ) : null}
          {isCode && (deps?.refs.length || deps?.defs.length) ? (
            <div className="flex flex-wrap gap-x-3 px-3 pb-1 text-[10px] text-muted-foreground/60">
              {deps?.defs.length ? <span>defines {deps.defs.join(", ")}</span> : null}
              {deps?.refs.length ? <span>uses {deps.refs.join(", ")}</span> : null}
              {stale ? <span className="text-warning">stale</span> : null}
            </div>
          ) : null}
        </div>

        {/* Per-cell actions, revealed on hover. */}
        <div className="flex shrink-0 items-start gap-0.5 p-1 opacity-0 transition group-hover:opacity-100">
          {promotable ? (
            <IconButton title="Promote to a certified derivation" onClick={onPromote}>
              {promoting ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <ShieldCheck className="size-3.5" />
              )}
            </IconButton>
          ) : null}
          {isCode ? (
            <IconButton
              title={isParams ? "Unmark parameters cell" : "Mark as parameters cell"}
              onClick={onToggleParams}
            >
              <SlidersHorizontal className={`size-3.5 ${isParams ? "text-primary" : ""}`} />
            </IconButton>
          ) : null}
          <IconButton title="Move up" onClick={() => onMove(-1)}>
            <ChevronUp className="size-3.5" />
          </IconButton>
          <IconButton title="Move down" onClick={() => onMove(1)}>
            <ChevronDown className="size-3.5" />
          </IconButton>
          <IconButton
            title={isCode ? "Make markdown" : "Make code"}
            onClick={() => onSetType(isCode ? "markdown" : "code")}
          >
            <span className="text-[10px] font-medium">{isCode ? "M↓" : "{ }"}</span>
          </IconButton>
          <IconButton title="Delete cell" onClick={onDelete}>
            <Trash2 className="size-3.5" />
          </IconButton>
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
        <button
          type="button"
          onClick={() => onAddBelow("code")}
          className="rounded px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted"
        >
          + code
        </button>
        <button
          type="button"
          onClick={() => onAddBelow("markdown")}
          className="rounded px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted"
        >
          + text
        </button>
      </div>
    </div>
  )
}

function IconButton({
  title,
  onClick,
  children,
}: {
  title: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition hover:bg-muted hover:text-foreground"
    >
      {children}
    </button>
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
