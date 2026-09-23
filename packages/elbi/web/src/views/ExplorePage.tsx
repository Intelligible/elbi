import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table"
import {
  BarChart3,
  Braces,
  ChevronDown,
  ChevronRight,
  Clock,
  Columns3,
  Copy,
  Download,
  Info,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Save,
  Search,
  Sparkles,
  Table2,
  TerminalSquare,
  Trash2,
  Wand2,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { IconButton } from "@/components/app/IconButton"
import { CellEditor } from "@/components/notebook/CellEditor"
import { Scene } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { VerdictBadge } from "@/components/VerdictBadge"
import { VizView } from "@/components/viz/VizView"
import {
  type CatalogTable,
  type Cell,
  type ColumnProfile,
  certifyDerivation,
  type DraftResult,
  deleteQuery,
  draftDerivation,
  duplicateQuery,
  getCatalog,
  getSources,
  listQueries,
  nl2sql,
  profile as profileQuery,
  promoteQuery,
  type QueryResult,
  runQuery,
  type SavedQuery,
  type Source,
  saveQuery,
} from "@/lib/explore"
import { copyText, EMPTY, uuid } from "@/lib/utils"

// The bound-datasets target has a null id; Radix Select needs a non-empty string value,
// so it is represented by this sentinel in the picker and mapped back to null.
const BOUND = "__bound__"

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

type Tab = "results" | "chart" | "profile" | "info"

// One open query in the editor: each tab carries its own SQL, source, and last result, the way
// a real SQL IDE keeps several queries side by side.
type QueryTab = {
  id: string
  name: string
  sql: string
  sourceId: string | null
  savedId: string | null
  result: QueryResult | null
  profileRows: ColumnProfile[] | null
  error: string | null
  elapsedMs: number | null
  view: Tab
  running: boolean
}

// Bounds for the editor pane.
const MIN_EDITOR_H = 140
const MAX_EDITOR_H = 640

// Bounds for a result column, declared on the table and read back by the separator.
const MIN_COL_W = 80
const MAX_COL_W = 800

type HistoryEntry = { sql: string; sourceId: string | null; at: number }
const HISTORY_KEY = "explore.history"

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : []
  } catch {
    return []
  }
}
function pushHistory(entry: HistoryEntry): HistoryEntry[] {
  const next = [entry, ...loadHistory().filter((h) => h.sql !== entry.sql)].slice(0, 50)
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
  } catch {
    /* storage may be unavailable; history is best-effort */
  }
  return next
}

// `{{name}}` template variables: detected in the SQL, filled from a values menu, and
// substituted just before the query runs (never interpolated into the stored text).
const VAR_RE = /\{\{\s*([a-zA-Z_]\w*)\s*\}\}/g
function findVars(sql: string): string[] {
  return [...new Set(Array.from(sql.matchAll(VAR_RE), (m) => m[1]))]
}
function applyVars(sql: string, values: Record<string, string>): string {
  return sql.replace(VAR_RE, (_, name: string) => values[name] ?? "")
}

function newQueryTab(n: number, sourceId: string | null): QueryTab {
  return {
    id: uuid(),
    name: `Query ${n}`,
    sql: "SELECT 1 AS example",
    sourceId,
    savedId: null,
    result: null,
    profileRows: null,
    error: null,
    elapsedMs: null,
    view: "results",
    running: false,
  }
}

export function ExplorePage() {
  const dark = useDarkTheme()
  const [sources, setSources] = useState<Source[]>([])
  const [saved, setSaved] = useState<SavedQuery[]>([])
  const [tables, setTables] = useState<CatalogTable[]>([])
  const first = useMemo(() => newQueryTab(1, null), [])
  const [tabs, setTabs] = useState<QueryTab[]>([first])
  const [activeId] = useState(first.id)
  const [schemaOpen, setSchemaOpen] = useState(true)
  const [nlPrompt, setNlPrompt] = useState("")
  const [nlBusy, setNlBusy] = useState(false)
  const [vars, setVars] = useState<Record<string, string>>({})
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [editorH, setEditorH] = useState(224)

  const active = tabs.find((t) => t.id === activeId) ?? tabs[0]
  const patch = useCallback(
    (id: string, partial: Partial<QueryTab>) =>
      setTabs((ts) => ts.map((t) => (t.id === id ? { ...t, ...partial } : t))),
    [],
  )

  const refreshSaved = useCallback(() => {
    listQueries()
      .then(setSaved)
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    getSources()
      .then(setSources)
      .catch(() => undefined)
    refreshSaved()
    setHistory(loadHistory())
  }, [refreshSaved])

  // Reload the schema browser for whichever source the active tab targets.
  useEffect(() => {
    setTables([])
    getCatalog(active.sourceId)
      .then((c) => setTables(c.tables))
      .catch(() => setTables([]))
  }, [active.sourceId])

  const run = useCallback(
    async (override?: string) => {
      const raw = override ?? active.sql
      const q = applyVars(raw, vars)
      patch(active.id, { running: true, error: null })
      const started = performance.now()
      try {
        const res = await runQuery(q, active.sourceId)
        patch(active.id, {
          result: res,
          elapsedMs: performance.now() - started,
          profileRows: null,
          view: "results",
          running: false,
        })
        setHistory(pushHistory({ sql: raw, sourceId: active.sourceId, at: Date.now() }))
      } catch (e) {
        patch(active.id, {
          result: null,
          elapsedMs: null,
          error: e instanceof Error ? e.message : String(e),
          running: false,
        })
      }
    },
    [active.id, active.sql, active.sourceId, vars, patch],
  )

  const showProfile = useCallback(async () => {
    patch(active.id, { error: null, view: "profile" })
    try {
      const res = await profileQuery({
        sql: applyVars(active.sql, vars),
        sourceId: active.sourceId,
      })
      patch(active.id, { profileRows: res.columns })
    } catch (e) {
      patch(active.id, { profileRows: null, error: e instanceof Error ? e.message : String(e) })
    }
  }, [active.id, active.sql, active.sourceId, vars, patch])

  const insert = useCallback(
    (text: string) => patch(active.id, { sql: active.sql.trim() ? `${active.sql} ${text}` : text }),
    [active.id, active.sql, patch],
  )

  const askNl = useCallback(async () => {
    const prompt = nlPrompt.trim()
    if (!prompt) return
    setNlBusy(true)
    try {
      const res = await nl2sql(prompt, active.sourceId)
      if (res.sql) patch(active.id, { sql: res.sql })
    } catch (e) {
      patch(active.id, { error: e instanceof Error ? e.message : String(e) })
    } finally {
      setNlBusy(false)
    }
  }, [nlPrompt, active.id, active.sourceId, patch])

  const loadQuery = useCallback(
    (sql: string, sourceId: string | null, savedId: string | null, name?: string) =>
      patch(active.id, {
        sql,
        sourceId,
        savedId,
        ...(name ? { name } : {}),
      }),
    [active.id, patch],
  )

  const varNames = useMemo(() => findVars(active.sql), [active.sql])

  const sqlSchema = useMemo(
    () => Object.fromEntries(tables.map((t) => [t.name, t.columns.map((c) => c.name)])),
    [tables],
  )

  return (
    <Scene>
      {/* Thin top bar: query context on the left, right-aligned actions. No tall
          header band, because this is a tool page rather than a document. */}
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-border px-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <TerminalSquare className="size-4 text-text-tertiary" />
          Explore
          <span className="text-text-tertiary">/</span>
          <span className="font-normal text-text-secondary">{active.name}</span>
        </div>
        <div className="flex items-center gap-1">
          {varNames.length > 0 ? (
            <VariablesMenu names={varNames} values={vars} onChange={setVars} />
          ) : null}
          <Button size="sm" variant="ghost" onClick={() => setHistoryOpen(true)}>
            <Clock className="size-3.5" /> History
          </Button>
          <SavedQueries
            saved={saved}
            onLoad={(q) => loadQuery(q.sql, q.sourceId, q.id, q.name)}
            onDuplicate={async (queryId) => {
              try {
                await duplicateQuery(queryId)
              } catch {
                // The popover has no room for an error line, and the caller drops
                // this promise. The refresh below shows the truth either way:
                // the copy appears, or the list is unchanged.
              } finally {
                refreshSaved()
              }
            }}
            onDelete={async (queryId) => {
              await deleteQuery(queryId)
              if (queryId === active.savedId) patch(active.id, { savedId: null })
              refreshSaved()
            }}
          />
          <SaveButton
            sql={active.sql}
            sourceId={active.sourceId}
            currentId={active.savedId}
            onSaved={(q) => {
              patch(active.id, { savedId: q.id, name: q.name })
              refreshSaved()
            }}
          />
          <DraftButton sql={active.sql} sourceId={active.sourceId} onDone={refreshSaved} />
          <PromoteButton sql={active.sql} sourceId={active.sourceId} onDone={refreshSaved} />
        </div>
      </div>

      {/* The sources panel shares the top bar's surface (bg-background) so they flow as
          one frame; the query/results area is the distinct surface (bg-card): its own
          section. The sources panel collapses to give the editor the width. */}
      <div className="flex min-h-0 flex-1">
        {schemaOpen ? (
          <SchemaBrowser
            tables={tables}
            onInsert={insert}
            sources={sources}
            sourceId={active.sourceId}
            onSource={(v) => patch(active.id, { sourceId: v })}
            onCollapse={() => setSchemaOpen(false)}
          />
        ) : null}

        <div className="flex min-h-0 flex-1 flex-col border-l border-border bg-card">
          {/* Editor toolbar on top: Run is the prominent accent action. */}
          <div className="flex w-full shrink-0 items-center gap-2 border-b border-border px-2 py-1">
            {!schemaOpen ? (
              <Button
                size="icon-sm"
                variant="outline"
                onClick={() => setSchemaOpen(true)}
                aria-label="Show sources"
              >
                <PanelLeftOpen className="size-4" />
              </Button>
            ) : null}
            <Button size="sm" onClick={() => run()} disabled={active.running}>
              <Play className="size-3.5" />
              {active.running ? "Running…" : "Run"}
              <span className="ml-1 text-3xs opacity-70">⌘↵</span>
            </Button>
            <div className="mx-1 h-5 w-px shrink-0 bg-border" />
            <Wand2 className="size-4 shrink-0 text-text-tertiary" />
            <Input
              value={nlPrompt}
              onChange={(e) => setNlPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") askNl()
              }}
              placeholder="Ask in plain English: e.g. average amount by region"
              className="h-8 flex-1 text-sm"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={askNl}
              disabled={nlBusy || !nlPrompt.trim()}
            >
              {nlBusy ? "Thinking…" : "Draft SQL"}
            </Button>
          </div>
          <div
            style={{ height: editorH }}
            className="min-h-0 shrink-0 overflow-auto bg-background px-3 py-2"
          >
            <CellEditor
              key={active.id}
              value={active.sql}
              language="sql"
              sqlSchema={sqlSchema}
              dark={dark}
              onChange={(v) => patch(active.id, { sql: v })}
              onRun={() => run()}
              onRunNoAdvance={() => run()}
              onRunText={(t) => run(t)}
            />
          </div>

          <EditorResizer
            height={editorH}
            onDelta={(dy) =>
              setEditorH((h) => Math.min(MAX_EDITOR_H, Math.max(MIN_EDITOR_H, h + dy)))
            }
          />

          {active.error ? (
            <div className="shrink-0 border-y border-danger/30 bg-danger-tint px-4 py-2 font-mono text-xs text-danger">
              {active.error}
            </div>
          ) : null}

          <div className="flex shrink-0 items-center gap-1 border-b border-border px-3 py-1.5">
            {/* Profile sends a request, so it activates on click (and profiles again when
                clicked while active) rather than through the tab value. */}
            <Tabs
              value={active.view}
              activationMode="manual"
              onValueChange={(v) => {
                if (v !== "profile") patch(active.id, { view: v as Tab })
              }}
            >
              <TabsList className="h-auto gap-1 bg-transparent p-0 group-data-[orientation=horizontal]/tabs:h-auto">
                <TabsTrigger value="results" className={TAB_TRIGGER}>
                  <Table2 className="size-3.5" /> Results
                </TabsTrigger>
                <TabsTrigger value="chart" className={TAB_TRIGGER}>
                  <BarChart3 className="size-3.5" /> Chart
                </TabsTrigger>
                <TabsTrigger value="profile" className={TAB_TRIGGER} onClick={showProfile}>
                  <Columns3 className="size-3.5" /> Profile
                </TabsTrigger>
                <TabsTrigger value="info" className={TAB_TRIGGER}>
                  <Info className="size-3.5" /> Info
                </TabsTrigger>
              </TabsList>
            </Tabs>
            {active.result && active.result.rows.length > 0 ? (
              <div className="ml-auto flex items-center gap-1">
                <ResultActions result={active.result} name={active.name} />
              </div>
            ) : null}
          </div>

          <div className="min-h-0 flex-1 overflow-auto">
            {active.view === "results" ? <ResultsGrid result={active.result} /> : null}
            {active.view === "chart" ? <ChartPanel result={active.result} /> : null}
            {active.view === "profile" ? <ProfilePanel rows={active.profileRows} /> : null}
            {active.view === "info" ? (
              <QueryInfoPanel
                result={active.result}
                sql={active.sql}
                elapsedMs={active.elapsedMs}
              />
            ) : null}
          </div>

          {active.view === "results" && active.result && active.result.rows.length > 0 ? (
            <div className="flex shrink-0 items-center gap-2 border-t border-border bg-surface-secondary px-4 py-1.5 text-xs text-text-tertiary tabular-nums">
              <span>
                {active.result.rows.length.toLocaleString()} row
                {active.result.rows.length === 1 ? "" : "s"}
                {active.result.truncated ? " (truncated)" : ""}
              </span>
              {active.elapsedMs !== null ? (
                <span>
                  ·{" "}
                  {active.elapsedMs < 1000
                    ? `${Math.round(active.elapsedMs)} ms`
                    : `${(active.elapsedMs / 1000).toFixed(2)} s`}
                </span>
              ) : null}
              <span className="ml-auto">
                {active.result.columns.length} column
                {active.result.columns.length === 1 ? "" : "s"}
              </span>
            </div>
          ) : null}
        </div>
      </div>

      <HistoryDialog
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        entries={history}
        sources={sources}
        onLoad={(e) => {
          loadQuery(e.sql, e.sourceId, null)
          setHistoryOpen(false)
        }}
      />
    </Scene>
  )
}

// A draggable divider that resizes the editor pane above the results.
function EditorResizer({ height, onDelta }: { height: number; onDelta: (dy: number) => void }) {
  const dragging = useRef(false)
  const lastY = useRef(0)
  const cb = useRef(onDelta)
  cb.current = onDelta
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragging.current) return
      cb.current(e.clientY - lastY.current)
      lastY.current = e.clientY
    }
    const up = () => {
      dragging.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
    window.addEventListener("mousemove", move)
    window.addEventListener("mouseup", up)
    return () => {
      window.removeEventListener("mousemove", move)
      window.removeEventListener("mouseup", up)
    }
  }, [])
  return (
    <hr
      aria-orientation="horizontal"
      aria-label="Resize the editor"
      aria-valuenow={height}
      aria-valuemin={MIN_EDITOR_H}
      aria-valuemax={MAX_EDITOR_H}
      tabIndex={0}
      onMouseDown={(e) => {
        dragging.current = true
        lastY.current = e.clientY
        document.body.style.cursor = "row-resize"
        document.body.style.userSelect = "none"
      }}
      onKeyDown={(e) => {
        if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return
        e.preventDefault()
        cb.current(e.key === "ArrowUp" ? -16 : 16)
      }}
      // `hr` is void, so the grip is a pseudo-element.
      className="group m-0 flex h-2 shrink-0 cursor-row-resize items-center justify-center border-0 border-b border-border bg-surface-secondary before:block before:h-0.5 before:w-8 before:rounded-full before:bg-border-strong before:transition-colors hover:before:bg-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring before:content-['']"
    />
  )
}

// Values for the `{{name}}` variables found in the current query.
function VariablesMenu({
  names,
  values,
  onChange,
}: {
  names: string[]
  values: Record<string, string>
  onChange: (v: Record<string, string>) => void
}) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <Button size="sm" variant="ghost" onClick={() => setOpen(true)}>
        <Braces className="size-3.5" /> Variables ({names.length})
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Query variables</DialogTitle>
            <DialogDescription>
              Fill the <code>{"{{name}}"}</code> placeholders in your query. Values are substituted
              at run time, never stored in the query text.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            {names.map((n) => (
              <label key={n} className="flex items-center gap-3 text-sm">
                <span className="w-32 shrink-0 truncate font-mono text-text-secondary">{n}</span>
                <Input
                  value={values[n] ?? ""}
                  onChange={(e) => onChange({ ...values, [n]: e.target.value })}
                  placeholder="value"
                  className="h-8"
                />
              </label>
            ))}
          </div>
          <DialogFooter>
            <Button size="sm" onClick={() => setOpen(false)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

// Recently-run queries (kept per browser), loadable back into the active tab.
function HistoryDialog({
  open,
  onOpenChange,
  entries,
  sources,
  onLoad,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  entries: HistoryEntry[]
  sources: Source[]
  onLoad: (e: HistoryEntry) => void
}) {
  const sourceName = (id: string | null) =>
    id === null ? "bound datasets" : (sources.find((s) => s.id === id)?.name ?? id)
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Query history</DialogTitle>
          <DialogDescription>The queries you've run, most recent first.</DialogDescription>
        </DialogHeader>
        <div className="max-h-[60vh] space-y-1 overflow-y-auto">
          {entries.length === 0 ? (
            <p className="py-8 text-center text-sm text-text-tertiary">No history yet.</p>
          ) : (
            entries.map((e) => (
              <Button
                variant="ghost"
                key={e.sql}
                onClick={() => onLoad(e)}
                className="block h-auto w-full rounded-md border border-border bg-card px-3 py-2 text-left font-normal whitespace-normal hover:border-border-strong hover:bg-muted/60 dark:hover:bg-muted/60"
              >
                <pre className="truncate font-mono text-xs text-foreground">{e.sql}</pre>
                <div className="mt-1 text-2xs text-text-tertiary">
                  {sourceName(e.sourceId)} · {relativeTime(e.at)}
                </div>
              </Button>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Compact relative time for the history list.
function relativeTime(ms: number): string {
  const s = (Date.now() - ms) / 1000
  if (s < 60) return "just now"
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

// A compact result-view tab: the active one sits on a muted fill.
const TAB_TRIGGER =
  "h-auto flex-none gap-1.5 border-0 px-2.5 py-1 text-xs text-text-tertiary hover:text-foreground data-[state=active]:bg-muted data-[state=active]:text-foreground group-data-[variant=default]/tabs-list:data-[state=active]:shadow-none dark:text-text-tertiary dark:hover:text-foreground dark:data-[state=active]:bg-muted"

// A full-width schema-browser row with a flat muted hover.
const SCHEMA_ROW =
  "flex h-auto w-full justify-start gap-1 rounded-none px-3 py-1.5 text-left font-normal hover:bg-muted hover:text-foreground dark:hover:bg-muted"

function SchemaBrowser({
  tables,
  onInsert,
  sources,
  sourceId,
  onSource,
  onCollapse,
}: {
  tables: CatalogTable[]
  onInsert: (text: string) => void
  sources: Source[]
  sourceId: string | null
  onSource: (v: string | null) => void
  onCollapse: () => void
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [q, setQ] = useState("")
  const term = q.trim().toLowerCase()
  // Filter to tables whose name or any column matches; expand matches automatically.
  const shown = term
    ? tables.filter(
        (t) =>
          t.name.toLowerCase().includes(term) ||
          t.columns.some((c) => c.name.toLowerCase().includes(term)),
      )
    : tables
  return (
    <aside className="flex w-72 shrink-0 flex-col overflow-hidden border-r border-border bg-background">
      {/* Connection selector at the top of the sources rail. */}
      <div className="flex shrink-0 items-center gap-2 border-b border-border p-2">
        <Select value={sourceId ?? BOUND} onValueChange={(v) => onSource(v === BOUND ? null : v)}>
          <SelectTrigger className="h-8 flex-1 text-sm">
            <SelectValue placeholder="Source" />
          </SelectTrigger>
          <SelectContent>
            {sources.map((s) => (
              <SelectItem key={s.id ?? BOUND} value={s.id ?? BOUND}>
                {s.name}
                {s.kind !== "project" ? ` (${s.kind})` : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button size="icon-sm" variant="ghost" onClick={onCollapse} aria-label="Collapse sources">
          <PanelLeftClose className="size-4" />
        </Button>
      </div>
      <div className="shrink-0 border-b border-border p-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search tables & columns"
            className="h-8 pl-8"
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {shown.length === 0 ? (
          <p className="px-3 py-2 text-xs text-text-tertiary">
            {term ? "No matches." : "No tables."}
          </p>
        ) : (
          <ul className="pb-4">
            {shown.map((t) => (
              <li key={t.name}>
                <Button
                  variant="ghost"
                  aria-expanded={!!open[t.name]}
                  onClick={() => setOpen((o) => ({ ...o, [t.name]: !o[t.name] }))}
                  onDoubleClick={() => onInsert(t.name)}
                  className={SCHEMA_ROW}
                >
                  <ChevronRight
                    className={`size-3 shrink-0 text-text-tertiary transition-transform ${
                      open[t.name] ? "rotate-90" : ""
                    }`}
                  />
                  <Table2 className="size-3.5 shrink-0 text-text-tertiary" />
                  <span className="flex-1 truncate font-medium">{t.name}</span>
                  {t.rows !== undefined ? (
                    <span className="text-3xs text-text-tertiary">{t.rows}</span>
                  ) : null}
                </Button>
                {open[t.name] ? (
                  <ul className="pb-1">
                    {t.columns.map((c) => (
                      <li key={c.name}>
                        <Button
                          variant="ghost"
                          onClick={() => onInsert(c.name)}
                          className={`${SCHEMA_ROW} gap-2 py-0.5 pl-9 text-xs text-text-tertiary`}
                        >
                          <span className="flex-1 truncate">{c.name}</span>
                          {c.type ? <span className="text-3xs opacity-70">{c.type}</span> : null}
                        </Button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  )
}

// Infer a column's type from its first non-null value, so the grid header can label
// each column the way a SQL editor does (name + type) even though the query result
// carries only values, not a declared schema.
function inferType(rows: Record<string, Cell>[], name: string): string {
  // Numeric columns need a scan (not just the first value) to tell integer from double: e.g. a
  // `bathrooms` column whose first row is 1 but later rows are 2.25.
  let numeric = false
  let allInteger = true
  for (const row of rows.slice(0, 200)) {
    const v = row[name]
    if (v === null || v === undefined) continue
    if (typeof v === "number") {
      numeric = true
      if (!Number.isInteger(v)) allInteger = false
    } else if (typeof v === "boolean") return "boolean"
    else if (typeof v === "string")
      return /^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?/.test(v) ? "date" : "text"
    else return "text"
  }
  if (numeric) return allInteger ? "integer" : "double"
  return "null"
}

function ResultsGrid({ result }: { result: QueryResult | null }) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [detail, setDetail] = useState<Record<string, Cell> | null>(null)
  const columns = useMemo<ColumnDef<Record<string, Cell>>[]>(
    () =>
      (result?.columns ?? []).map((name) => {
        const type = inferType(result?.rows ?? [], name)
        return {
          accessorKey: name,
          header: () => (
            <span className="flex flex-col leading-tight">
              <span className="font-mono text-xs font-medium tracking-normal text-foreground normal-case">
                {name}
              </span>
              <span className="text-3xs leading-tight font-normal italic tracking-normal text-text-tertiary normal-case">
                {type}
              </span>
            </span>
          ),
          cell: (info) => formatCell(info.getValue() as Cell),
        }
      }),
    [result],
  )
  const table = useReactTable({
    data: result?.rows ?? [],
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    columnResizeMode: "onChange",
    enableColumnResizing: true,
    defaultColumn: { minSize: MIN_COL_W, maxSize: MAX_COL_W, size: 180 },
  })

  if (!result) {
    return (
      <Empty>
        Query results will appear here. Press{" "}
        <kbd className="rounded border border-border bg-surface-secondary px-1 py-0.5 font-mono text-3xs">
          ⌘↵
        </kbd>{" "}
        to run the query.
      </Empty>
    )
  }
  if (result.rows.length === 0) {
    return <Empty>The query ran, but returned no rows.</Empty>
  }
  return (
    // The table's own scroll wrapper fills the results pane, so the sticky header
    // sticks to the pane's scroll rather than to a wrapper as tall as the table.
    <div className="h-full [&>div]:h-full">
      <Table className="text-sm" style={{ width: table.getTotalSize(), minWidth: "100%" }}>
        <TableHeader>
          {table.getHeaderGroups().map((group) => (
            <TableRow key={group.id} className="hover:bg-transparent">
              {group.headers.map((header) => (
                <TableHead
                  key={header.id}
                  style={{ width: header.getSize() }}
                  className="group relative select-none py-1.5 align-bottom"
                >
                  <Button
                    variant="ghost"
                    onClick={header.column.getToggleSortingHandler()}
                    aria-label={`Sort by ${header.column.id}`}
                    className="flex h-auto w-full items-end justify-between gap-2 rounded-none p-0 text-left font-normal whitespace-normal hover:bg-transparent hover:opacity-80 dark:hover:bg-transparent"
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    <span className="pb-0.5 text-xs font-bold text-primary">
                      {{ asc: "↑", desc: "↓" }[header.column.getIsSorted() as string] ?? ""}
                    </span>
                  </Button>
                  <hr
                    aria-orientation="vertical"
                    aria-label={`Resize the ${header.column.id} column`}
                    aria-valuenow={header.getSize()}
                    aria-valuemin={MIN_COL_W}
                    aria-valuemax={MAX_COL_W}
                    tabIndex={0}
                    onMouseDown={header.getResizeHandler()}
                    onTouchStart={header.getResizeHandler()}
                    onKeyDown={(e) => {
                      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return
                      e.preventDefault()
                      const step = e.key === "ArrowLeft" ? -16 : 16
                      table.setColumnSizing((sizes) => ({
                        ...sizes,
                        [header.column.id]: Math.min(
                          MAX_COL_W,
                          Math.max(MIN_COL_W, header.getSize() + step),
                        ),
                      }))
                    }}
                    className={`absolute top-0 right-0 m-0 h-full w-1 cursor-col-resize touch-none select-none border-0 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                      header.column.getIsResizing() ? "bg-primary opacity-100" : "bg-border-strong"
                    }`}
                  />
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.map((row) => (
            <TableRow
              key={row.id}
              onClick={() => setDetail(row.original)}
              className="cursor-pointer"
            >
              {row.getVisibleCells().map((cell) => (
                <TableCell
                  key={cell.id}
                  style={{ width: cell.column.getSize() }}
                  className="truncate py-1.5 font-mono text-xs"
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <RowDetailDialog row={detail} columns={result.columns} onClose={() => setDetail(null)} />
    </div>
  )
}

// Click a result row to inspect the full record: every column, with copy-per-value.
function RowDetailDialog({
  row,
  columns,
  onClose,
}: {
  row: Record<string, Cell> | null
  columns: string[]
  onClose: () => void
}) {
  return (
    <Dialog open={row !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="sm:max-w-xl"
        // Focus the dialog itself: landing on the first copy button would open its tooltip.
        onOpenAutoFocus={(e) => {
          const content = e.currentTarget as HTMLElement
          e.preventDefault()
          content.focus()
        }}
      >
        <DialogHeader>
          <DialogTitle>Row details</DialogTitle>
        </DialogHeader>
        {row ? (
          <div className="max-h-[60vh] overflow-y-auto rounded-lg border border-border">
            <Table className="text-sm">
              <TableBody>
                {columns.map((c) => (
                  <TableRow key={c} className="group">
                    <TableCell className="w-1/3 py-1.5 align-top font-mono text-xs text-text-tertiary">
                      {c}
                    </TableCell>
                    <TableCell className="py-1.5 font-mono text-xs break-all">
                      {formatCell(row[c])}
                    </TableCell>
                    <TableCell className="w-8 px-1 py-1.5 text-right">
                      <IconButton
                        label={`Copy ${c}`}
                        size="icon-xs"
                        onClick={() => void copyText(cellText(row[c]))}
                        className="size-5 rounded-sm text-text-tertiary opacity-0 transition group-hover:opacity-100 hover:bg-muted hover:text-foreground focus-visible:opacity-100 dark:hover:bg-muted"
                      >
                        <Copy className="size-3" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}

function ChartPanel({ result }: { result: QueryResult | null }) {
  const columns = result?.columns ?? []
  const [mark, setMark] = useState("bar")
  const [x, setX] = useState<string>("")
  const [y, setY] = useState<string>("")

  useEffect(() => {
    if (columns.length && !columns.includes(x)) setX(columns[0])
    if (columns.length && !columns.includes(y)) setY(columns[columns.length - 1])
  }, [columns, x, y])

  if (!result || result.rows.length === 0) {
    return <Empty>Run a query to chart its result.</Empty>
  }

  const rows = result.rows as Record<string, string | number>[]
  const spec = {
    mark: { type: mark, tooltip: true },
    encoding: {
      x: { field: x, type: fieldType(rows, x) },
      y: { field: y, type: fieldType(rows, y) },
    },
  }
  return (
    <div className="p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <AxisPicker
          label="Mark"
          value={mark}
          options={["bar", "line", "point", "area"]}
          onChange={setMark}
        />
        <AxisPicker label="X" value={x} options={columns} onChange={setX} />
        <AxisPicker label="Y" value={y} options={columns} onChange={setY} />
      </div>
      <VizView viz={{ spec, rows }} height={360} />
    </div>
  )
}

function AxisPicker({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-text-tertiary">{label}</span>
      <Select value={value || undefined} onValueChange={onChange}>
        <SelectTrigger className="h-7 w-36 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o} value={o}>
              {o}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function ProfilePanel({ rows }: { rows: ColumnProfile[] | null }) {
  if (rows === null) {
    return <Empty>Profile a query result to see column statistics.</Empty>
  }
  return (
    // As in the results grid: the scroll wrapper fills the pane so the header sticks.
    <div className="h-full [&>div]:h-full">
      <Table className="text-sm">
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            {["Column", "Type", "Complete", "Distinct", "Min", "Max", "Top values"].map((h) => (
              <TableHead key={h}>{h}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((c) => (
            <TableRow key={c.name} className="[&>td]:align-top">
              <TableCell className="font-medium">
                {c.name}
                {c.isUnique ? (
                  <Badge variant="secondary" className="ml-2">
                    unique
                  </Badge>
                ) : null}
              </TableCell>
              <TableCell className="text-text-tertiary">{c.inferredType}</TableCell>
              <TableCell>{(c.completeness * 100).toFixed(0)}%</TableCell>
              <TableCell>{c.distinct}</TableCell>
              <TableCell className="font-mono text-xs">{c.minimum ?? EMPTY}</TableCell>
              <TableCell className="font-mono text-xs">{c.maximum ?? EMPTY}</TableCell>
              <TableCell className="text-xs text-text-tertiary">
                {c.topValues
                  .slice(0, 3)
                  .map((t) => `${formatCell(t.value)} (${t.count})`)
                  .join(", ")}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

// A 14px-wide, 24px-tall icon button: the vertical negative margin keeps the row height, and
// the icon-wide box keeps side-by-side buttons from overlapping.
const ICON_14 = "-my-1.25 w-3.5 text-text-tertiary"

function SavedQueries({
  saved,
  onLoad,
  onDelete,
  onDuplicate,
}: {
  saved: SavedQuery[]
  onLoad: (q: SavedQuery) => void
  onDelete: (id: string) => void
  onDuplicate: (id: string) => void
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <Button size="sm" variant="ghost" onClick={() => setOpen((o) => !o)}>
        Saved ({saved.length})
      </Button>
      {open ? (
        <div className="absolute right-0 top-9 z-20 w-72 rounded-lg border border-border bg-popover p-1 shadow-elevation">
          {saved.length === 0 ? (
            <p className="px-2 py-2 text-xs text-text-tertiary">No saved queries.</p>
          ) : (
            saved.map((q) => (
              <div
                key={q.id}
                className="flex items-center gap-1 rounded-md px-2 py-1.5 hover:bg-muted"
              >
                <Button
                  variant="ghost"
                  onClick={() => {
                    onLoad(q)
                    setOpen(false)
                  }}
                  className="h-auto min-w-0 flex-1 justify-start p-0 text-left font-normal hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"
                >
                  <span className="truncate">{q.name}</span>
                </Button>
                <IconButton
                  label={`Duplicate ${q.name}`}
                  size="icon-xs"
                  onClick={() => onDuplicate(q.id)}
                  className={`${ICON_14} hover:text-foreground`}
                >
                  <Copy className="size-3.5" />
                </IconButton>
                <IconButton
                  label={`Delete query ${q.name}`}
                  size="icon-xs"
                  onClick={() => onDelete(q.id)}
                  className={`${ICON_14} hover:text-danger`}
                >
                  <Trash2 className="size-3.5" />
                </IconButton>
              </div>
            ))
          )}
        </div>
      ) : null}
    </div>
  )
}

function SaveButton({
  sql,
  sourceId,
  currentId,
  onSaved,
}: {
  sql: string
  sourceId: string | null
  currentId: string | null
  onSaved: (q: SavedQuery) => void
}) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState("")
  return (
    <>
      <Button size="sm" variant="outline" onClick={() => setOpen(true)}>
        <Save className="size-3.5" />
        Save
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save query</DialogTitle>
            <DialogDescription>
              A saved query is a personal exploration artifact, not a certified derivation.
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            placeholder="Query name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <DialogFooter>
            <Button
              onClick={async () => {
                const q = await saveQuery({
                  id: currentId ?? undefined,
                  name: name || "Untitled query",
                  sql,
                  sourceId,
                })
                onSaved(q)
                setOpen(false)
                setName("")
              }}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

// The one-click loop in one dialog: draft from a question (or the editor SQL), read
// the oracle's verdict, certify. Certify reuses the promote path and stays disabled
// unless the draft verified, so a rejected draft can never serve.
function DraftButton({
  sql,
  sourceId,
  onDone,
}: {
  sql: string
  sourceId: string | null
  onDone: () => void
}) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState("")
  const [prompt, setPrompt] = useState("")
  // Which action is in flight, so each button reports only its own progress.
  const [busy, setBusy] = useState<"verify" | "certify" | null>(null)
  const [draft, setDraft] = useState<DraftResult | null>(null)
  const [flowId, setFlowId] = useState("")
  const [message, setMessage] = useState<string | null>(null)
  const clearDraft = () => {
    setDraft(null)
    setMessage(null)
  }
  return (
    <>
      <Button size="sm" variant="outline" onClick={() => setOpen(true)}>
        <Wand2 className="size-3.5" />
        Draft &amp; verify
      </Button>
      <Dialog
        open={open}
        onOpenChange={(o) => {
          setOpen(o)
          if (!o) clearDraft()
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Draft, verify, then certify</DialogTitle>
            <DialogDescription>
              Draft a derivation from a question (or the current query), see the oracle&apos;s
              verdict, and certify it with one click. Nothing is served until you certify.
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            placeholder="derivation_name (lowercase, underscores)"
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              clearDraft()
            }}
          />
          <Input
            placeholder="Ask in natural language (optional: uses the editor SQL if blank)"
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value)
              clearDraft()
            }}
          />
          {draft ? (
            <div className="min-w-0 space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-text-tertiary">Verdict</span>
                <VerdictBadge verdict={draft.verdict || (draft.ok ? "computed" : "unverified")} />
              </div>
              {draft.claim ? (
                <p className="break-words text-xs text-text-tertiary">
                  Claim: {draft.claim.x} → {draft.claim.y}
                  {draft.claim.controls?.length
                    ? ` (controls: ${draft.claim.controls.join(", ")})`
                    : ""}
                </p>
              ) : null}
              {draft.sql ? (
                <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs">
                  {draft.sql}
                </pre>
              ) : null}
              {draft.checks?.length ? (
                <ul className="space-y-0.5 text-xs text-text-tertiary">
                  {draft.checks.map((c) => (
                    <li key={c[0]}>
                      {c[0]}: {c[1]}
                    </li>
                  ))}
                </ul>
              ) : null}
              {draft.detail ? (
                <p className="break-words text-xs text-text-tertiary">{draft.detail}</p>
              ) : null}
            </div>
          ) : null}
          {message ? <p className="text-sm text-text-tertiary">{message}</p> : null}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={busy !== null || !name || (!prompt && !sql)}
              onClick={async () => {
                setBusy("verify")
                setMessage(null)
                setDraft(null)
                const fid = crypto.randomUUID()
                setFlowId(fid)
                try {
                  const res = await draftDerivation(name, sql, prompt, sourceId, fid)
                  setDraft(res)
                  if (!res.ok) setMessage(res.error || res.detail || "Not verified.")
                } catch (e) {
                  setMessage(e instanceof Error ? e.message : String(e))
                } finally {
                  setBusy(null)
                }
              }}
            >
              {busy === "verify" ? "Verifying…" : draft ? "Re-verify" : "Draft & verify"}
            </Button>
            <Button
              disabled={busy !== null || !draft?.ok}
              onClick={async () => {
                setBusy("certify")
                setMessage(null)
                try {
                  const res = await certifyDerivation(
                    name,
                    draft?.sql || sql,
                    sourceId,
                    flowId,
                    draft?.claim,
                  )
                  if (res.certified) {
                    setMessage(`Certified as ${res.name} (verdict: ${res.verdict ?? "sound"}).`)
                    setDraft(null) // done: a second click cannot re-certify
                    onDone()
                  } else {
                    setMessage(res.error || res.detail || "Not certified.")
                  }
                } catch (e) {
                  setMessage(e instanceof Error ? e.message : String(e))
                } finally {
                  setBusy(null)
                }
              }}
            >
              <Sparkles className="size-3.5" />
              {busy === "certify" ? "Certifying…" : "Certify"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function PromoteButton({
  sql,
  sourceId,
  onDone,
}: {
  sql: string
  sourceId: string | null
  onDone: () => void
}) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState("")
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  return (
    <>
      <Button size="sm" onClick={() => setOpen(true)}>
        <Sparkles className="size-3.5" />
        Promote
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Promote to a certified derivation</DialogTitle>
            <DialogDescription>
              Authors this query as a governed, cached derivation through the verification loop.
              Exploration itself is never gated; this is the opt-in bridge into the certified world.
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            placeholder="derivation_name (lowercase, underscores)"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          {message ? <p className="text-sm text-text-tertiary">{message}</p> : null}
          <DialogFooter>
            <Button
              disabled={busy || !name}
              onClick={async () => {
                setBusy(true)
                setMessage(null)
                try {
                  const res = await promoteQuery(name, sql, sourceId)
                  if (res.certified) {
                    setMessage(`Certified as ${res.name} (verdict: ${res.verdict}).`)
                    onDone()
                  } else {
                    setMessage(res.error || res.detail || "Not certified.")
                  }
                } catch (e) {
                  setMessage(e instanceof Error ? e.message : String(e))
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? "Promoting…" : "Promote"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-40 items-center justify-center px-4">
      <p className="max-w-sm text-center text-sm leading-relaxed text-text-tertiary">{children}</p>
    </div>
  )
}

function formatCell(value: Cell): string {
  if (value === null) return "∅"
  if (typeof value === "boolean") return value ? "true" : "false"
  return String(value)
}

// ---- Result export / copy (CSV · JSON · Markdown) ---------------------------
function cellText(v: Cell): string {
  return v === null || v === undefined ? "" : String(v)
}
function toCsv(columns: string[], rows: Record<string, Cell>[]): string {
  const esc = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s)
  const head = columns.map(esc).join(",")
  const body = rows.map((r) => columns.map((c) => esc(cellText(r[c]))).join(",")).join("\n")
  return `${head}\n${body}`
}
function toMarkdown(columns: string[], rows: Record<string, Cell>[]): string {
  const head = `| ${columns.join(" | ")} |`
  const sep = `| ${columns.map(() => "---").join(" | ")} |`
  const body = rows.map((r) => `| ${columns.map((c) => cellText(r[c])).join(" | ")} |`).join("\n")
  return `${head}\n${sep}\n${body}`
}
function downloadText(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function ResultActions({ result, name }: { result: QueryResult; name: string }) {
  const base = name.replace(/\s+/g, "_").toLowerCase() || "query"
  const copy = (text: string) => void copyText(text)
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="ghost">
            <Copy className="size-3.5" /> Copy <ChevronDown className="size-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => copy(toCsv(result.columns, result.rows))}>
            CSV
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => copy(JSON.stringify(result.rows, null, 2))}>
            JSON
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => copy(toMarkdown(result.columns, result.rows))}>
            Markdown
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => downloadText(`${base}.csv`, toCsv(result.columns, result.rows), "text/csv")}
      >
        <Download className="size-3.5" /> Export
      </Button>
    </>
  )
}

// The executed query, its columns and inferred types, and how long it took.
function QueryInfoPanel({
  result,
  sql,
  elapsedMs,
}: {
  result: QueryResult | null
  sql: string
  elapsedMs: number | null
}) {
  if (!result) return <Empty>Run a query to see its details.</Empty>
  return (
    <div className="space-y-5 p-4">
      <div>
        <div className="mb-1 text-2xs font-semibold uppercase tracking-[0.05em] text-text-tertiary">
          Query
        </div>
        <pre className="overflow-x-auto rounded-lg border border-border bg-surface-secondary p-3 font-mono text-xs text-foreground">
          {sql}
        </pre>
      </div>
      <div className="flex flex-wrap gap-6 text-sm">
        <div>
          <span className="text-text-tertiary">Rows </span>
          <span className="font-mono tabular-nums">
            {result.rows.length.toLocaleString()}
            {result.truncated ? " (truncated)" : ""}
          </span>
        </div>
        <div>
          <span className="text-text-tertiary">Columns </span>
          <span className="font-mono tabular-nums">{result.columns.length}</span>
        </div>
        {elapsedMs !== null ? (
          <div>
            <span className="text-text-tertiary">Time </span>
            <span className="font-mono tabular-nums">
              {elapsedMs < 1000
                ? `${Math.round(elapsedMs)} ms`
                : `${(elapsedMs / 1000).toFixed(2)} s`}
            </span>
          </div>
        ) : null}
      </div>
      <div>
        <div className="mb-1 text-2xs font-semibold uppercase tracking-[0.05em] text-text-tertiary">
          Columns
        </div>
        <div className="overflow-hidden rounded-lg border border-border bg-card">
          <Table className="text-sm">
            <TableBody>
              {result.columns.map((c) => (
                <TableRow key={c}>
                  <TableCell className="py-1.5 font-mono text-xs">{c}</TableCell>
                  <TableCell className="py-1.5 text-right font-mono text-xs italic text-text-tertiary">
                    {inferType(result.rows, c)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

function fieldType(
  rows: Record<string, string | number>[],
  field: string,
): "quantitative" | "nominal" | "temporal" {
  const sample = rows.find((r) => r[field] !== null && r[field] !== undefined)
  const value = sample?.[field]
  if (typeof value === "number") return "quantitative"
  if (typeof value === "string" && !Number.isNaN(Date.parse(value))) {
    // Only treat clearly date-like strings as temporal; bare numbers-as-strings stay
    // nominal so a category code is not charted as a timeline.
    if (/\d{4}-\d{2}-\d{2}/.test(value)) return "temporal"
  }
  return "nominal"
}
