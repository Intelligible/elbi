import { Code2, Download, LayoutDashboard, RefreshCw, Rocket } from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import GridLayout, { type Layout, type LayoutItem, WidthProvider } from "react-grid-layout/legacy"
import { useNavigate, useParams } from "react-router-dom"

import "react-grid-layout/css/styles.css"
import "react-resizable/css/styles.css"

import { DashboardWidget } from "@/components/dashboard/DashboardWidget"
import { FilterBar } from "@/components/dashboard/FilterBar"
import { TileEditor, type TilePatch } from "@/components/dashboard/TileEditor"
import { Scene, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { dashboardExportUrl } from "@/lib/chat"
import type { Dashboard, DashboardSpec, Variable, Widget, WidgetData } from "@/lib/dashboards"
import {
  bindableDerivations,
  getDashboard,
  publishDashboard,
  resolvePage,
  saveDashboard,
} from "@/lib/dashboards"

const Grid = WidthProvider(GridLayout)
const ROW_HEIGHT = 44
//: The spec's default page width; a page omitting `columns` is 24 wide, not 12.
const DEFAULT_COLUMNS = 24

function initialState(spec: DashboardSpec): Record<string, unknown> {
  const state: Record<string, unknown> = {}
  for (const v of spec.variables ?? []) {
    if (v.default !== undefined && v.default !== null) state[v.name] = v.default
  }
  return state
}

function pageVariables(spec: DashboardSpec, page: string): Variable[] {
  return (spec.variables ?? []).filter(
    (v) => (v.scope ?? "global") === "global" || v.scope === page,
  )
}

export function DashboardPage() {
  const { id = "" } = useParams()
  const navigate = useNavigate()
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [pageName, setPageName] = useState<string>("")
  const [variables, setVariables] = useState<Record<string, unknown>>({})
  const [widgets, setWidgets] = useState<Record<string, WidgetData>>({})
  const [loading, setLoading] = useState(false)
  const [publishError, setPublishError] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [catalog, setCatalog] = useState<string[]>([])
  const [tileUnderEdit, setTileUnderEdit] = useState<Widget | null>(null)
  const [tileToDelete, setTileToDelete] = useState<Widget | null>(null)

  const load = useCallback(async () => {
    const view = await getDashboard(id)
    setDashboard(view)
    setPageName((current) => current || view.spec.pages[0]?.name || "")
    setVariables((current) => (Object.keys(current).length > 0 ? current : initialState(view.spec)))
  }, [id])

  useEffect(() => {
    void load()
  }, [load])

  const spec = dashboard?.spec

  const resolve = useCallback(async () => {
    if (!spec || !pageName) return
    setLoading(true)
    try {
      const data = await resolvePage(id, pageName, variables)
      setWidgets(Object.fromEntries(data.map((d) => [d.widgetId, d])))
    } finally {
      setLoading(false)
    }
  }, [id, pageName, spec, variables])

  useEffect(() => {
    void resolve()
  }, [resolve])

  // The names a tile may bind, for the editor's derivation field.
  useEffect(() => {
    void bindableDerivations()
      .then(setCatalog)
      .catch(() => setCatalog([]))
  }, [])

  const page = useMemo(() => spec?.pages.find((p) => p.name === pageName), [spec, pageName])
  // 24, matching the spec's default and what `Page.to_manifest` omits when unchanged.
  // Defaulting to 12 here silently halved the grid of every page that took the default.
  const columns = page?.columns ?? DEFAULT_COLUMNS
  const visibleWidgets = useMemo(
    () => (page?.widgets ?? []).filter((w) => w.type !== "filter"),
    [page],
  )

  const layout = useMemo<LayoutItem[]>(
    () =>
      visibleWidgets.map((w) => ({
        i: w.id,
        x: w.gridPos.x,
        y: w.gridPos.y,
        w: w.gridPos.w,
        h: w.gridPos.h,
        minW: 2,
        minH: 3,
      })),
    [visibleWidgets],
  )

  // Persist a drag/resize back into the spec (debounced), so the layout a user arranges is
  // durable: the dashboard-as-code stays the source of truth.
  const persistLayout = useCallback(
    (next: Layout) => {
      if (!dashboard || !page) return
      const byId = new Map(next.map((l) => [l.i, l]))
      const nextSpec: DashboardSpec = {
        ...dashboard.spec,
        pages: dashboard.spec.pages.map((p) =>
          p.name !== page.name
            ? p
            : {
                ...p,
                widgets: p.widgets.map((w) => {
                  const l = byId.get(w.id)
                  return l ? { ...w, gridPos: { x: l.x, y: l.y, w: l.w, h: l.h } } : w
                }),
              },
        ),
      }
      setDashboard({ ...dashboard, spec: nextSpec })
      if (saveTimer.current) clearTimeout(saveTimer.current)
      saveTimer.current = setTimeout(() => {
        void saveDashboard(id, nextSpec)
      }, 600)
    },
    [dashboard, page, id],
  )

  // One place every spec edit goes through: rebuild the spec, show it immediately, and
  // write it. Unlike a drag there is nothing to debounce — a dialog's Save and a delete
  // are single deliberate acts, so they persist at once.
  const writeSpec = useCallback(
    async (nextSpec: DashboardSpec) => {
      setDashboard((current) => (current ? { ...current, spec: nextSpec } : current))
      await saveDashboard(id, nextSpec)
    },
    [id],
  )

  const mapWidgets = useCallback(
    (change: (widgets: Widget[]) => Widget[]): DashboardSpec | null => {
      if (!dashboard || !page) return null
      return {
        ...dashboard.spec,
        pages: dashboard.spec.pages.map((p) =>
          p.name === page.name ? { ...p, widgets: change(p.widgets) } : p,
        ),
      }
    },
    [dashboard, page],
  )

  const saveTile = useCallback(
    async (widgetId: string, patch: TilePatch) => {
      const nextSpec = mapWidgets((widgets) =>
        widgets.map((w) => {
          if (w.id !== widgetId) return w
          const next: Widget = { ...w }
          // An emptied title is no title, rather than an empty line above the body.
          if (patch.title !== undefined) {
            if (patch.title.trim()) next.title = patch.title.trim()
            else delete next.title
          }
          if (patch.content !== undefined) next.content = patch.content
          if (patch.derivation !== undefined && w.bind) {
            next.bind = { ...w.bind, derivation: patch.derivation }
          }
          if (patch.viz !== undefined) next.viz = patch.viz
          if (patch.gridPos !== undefined) next.gridPos = patch.gridPos
          return next
        }),
      )
      if (!nextSpec) return
      setTileUnderEdit(null)
      await writeSpec(nextSpec)
      void resolve()
    },
    [mapWidgets, writeSpec, resolve],
  )

  const deleteTile = useCallback(
    async (widgetId: string) => {
      const nextSpec = mapWidgets((widgets) => widgets.filter((w) => w.id !== widgetId))
      if (!nextSpec) return
      setTileToDelete(null)
      await writeSpec(nextSpec)
    },
    [mapWidgets, writeSpec],
  )

  const publish = async () => {
    setPublishError(null)
    try {
      await publishDashboard(id)
      await load()
    } catch (err) {
      setPublishError(err instanceof Error ? err.message : String(err))
    }
  }

  const openEditor = () => {
    if (spec) setDraft(JSON.stringify(spec, null, 2))
    setEditing(true)
  }

  const saveEditor = async () => {
    const parsed = JSON.parse(draft) as DashboardSpec
    const updated = await saveDashboard(id, parsed)
    setDashboard(updated)
    setEditing(false)
    void resolve()
  }

  if (dashboard === null || !spec) {
    return <SceneSkeleton />
  }

  return (
    <Scene>
      <SceneHeader
        icon={<LayoutDashboard className="size-5" />}
        title={dashboard.title || dashboard.name}
        backTo="/dashboards"
        backLabel="Dashboards"
        badges={
          <Badge variant={dashboard.status === "published" ? "success" : "neutral"}>
            {dashboard.status}
          </Badge>
        }
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => void resolve()}>
              <RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} />
            </Button>
            {/* Resolves every page to capture what the dashboard actually shows, so
                this one is genuinely slow -- and a plain <a download> shows no
                progress. The title says so before the click, not after. */}
            <Button variant="outline" size="sm" asChild>
              <a
                href={dashboardExportUrl(id)}
                download
                title="Runs every page's widgets to capture current values; this can take a few seconds"
                aria-label="Export this dashboard's record: spec, saved versions, and current values"
              >
                <Download className="size-4" />
                Export record
              </a>
            </Button>
            <Button variant="outline" size="sm" onClick={openEditor}>
              <Code2 className="size-4" />
              Edit spec
            </Button>
            <Button size="sm" onClick={publish}>
              <Rocket className="size-4" />
              Publish
            </Button>
          </>
        }
      >
        {spec.pages.length > 1 ? (
          <div className="flex gap-1">
            {spec.pages.map((p) => (
              <Button
                key={p.name}
                variant={p.name === pageName ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setPageName(p.name)}
              >
                {p.title || p.name}
              </Button>
            ))}
          </div>
        ) : null}
      </SceneHeader>

      {publishError ? (
        <div className="shrink-0 border-b border-danger/30 bg-danger-tint px-6 py-2 text-sm text-danger">
          {publishError}
        </div>
      ) : null}

      <FilterBar
        dashboardId={id}
        variables={pageVariables(spec, pageName)}
        state={variables}
        onChange={(name, value) =>
          // Same value, same object: re-picking what is already selected must not
          // count as a change, or the page re-resolves and every widget runs again.
          setVariables((current) =>
            Object.is(current[name], value) ? current : { ...current, [name]: value },
          )
        }
      />

      {/* Canvas: a recessed neutral so cards read as raised, per BI design guidance. */}
      <div className="min-h-0 flex-1 overflow-y-auto bg-surface-secondary p-6">
        {visibleWidgets.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-text-tertiary">
            This page has no widgets yet. Use “Edit spec” to add widgets bound to a certified
            derivation or a metric.
          </div>
        ) : (
          <Grid
            className="layout"
            layout={layout}
            cols={columns}
            rowHeight={ROW_HEIGHT}
            margin={[16, 16]}
            containerPadding={[0, 0]}
            draggableHandle=".dash-drag-handle"
            draggableCancel=".dash-no-drag"
            onDragStop={persistLayout}
            onResizeStop={persistLayout}
            isBounded
          >
            {visibleWidgets.map((widget: Widget) => (
              <div key={widget.id} className="h-full">
                <DashboardWidget
                  widget={widget}
                  data={widgets[widget.id]}
                  variables={variables}
                  onCrossFilter={(emit) => setVariables((current) => ({ ...current, ...emit }))}
                  onEdit={() => setTileUnderEdit(widget)}
                  onDelete={() => setTileToDelete(widget)}
                  onDrillThrough={() => {
                    const target = widget.interactions?.drillThrough?.target ?? ""
                    const [kind, ref] = target.split(":")
                    if (kind === "page" && spec.pages.some((p) => p.name === ref)) {
                      setPageName(ref)
                    } else if (kind === "dashboard") {
                      navigate(`/dashboards/${ref}`)
                    }
                  }}
                />
              </div>
            ))}
          </Grid>
        )}
      </div>

      <TileEditor
        widget={tileUnderEdit}
        catalog={catalog}
        columns={columns}
        onCancel={() => setTileUnderEdit(null)}
        onSave={(widgetId, patch) => void saveTile(widgetId, patch)}
      />

      <Dialog open={tileToDelete !== null} onOpenChange={(open) => !open && setTileToDelete(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete this tile?</DialogTitle>
            <p className="text-sm text-text-tertiary">
              “{tileToDelete?.title ?? tileToDelete?.id}” is removed from this page. The derivation
              behind it is untouched, and an earlier version of the dashboard is still in its
              history.
            </p>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setTileToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => tileToDelete && void deleteTile(tileToDelete.id)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={editing} onOpenChange={setEditing}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Edit dashboard spec</DialogTitle>
            <p className="text-xs text-text-tertiary">
              A widget’s <code>bind</code> is a certified derivation (
              <code>{'{"derivation": "name", "params": {…}}'}</code>) or a metric (
              <code>{'{"metric": "name", "groupBy": ["…"], "grain": "month"}'}</code>).
            </p>
          </DialogHeader>
          <Textarea
            className="h-[60vh] font-mono text-xs"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            spellCheck={false}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(false)}>
              Cancel
            </Button>
            <Button onClick={saveEditor}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Scene>
  )
}
