// Pins what a refused save does: the board goes back to the last spec the server
// accepted, and the editor that made the edit stays open with the server's reason.

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { Dashboard, WidgetData } from "@/lib/dashboards"

vi.mock("@/components/dashboard/JsonEditor", () => ({ JsonEditor: () => null }))

vi.mock("@/lib/dashboards", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/dashboards")>()),
  getDashboard: vi.fn(),
  resolvePage: vi.fn(async () => []),
  bindableDerivations: vi.fn(async () => []),
  dashboardSchema: vi.fn(async () => null),
  saveDashboard: vi.fn(),
}))

import { getDashboard, resolvePage, saveDashboard } from "@/lib/dashboards"
import { DashboardPage } from "./DashboardPage"

const DASHBOARD: Dashboard = {
  id: "d1",
  name: "ops",
  title: "Ops",
  status: "draft",
  version: 1,
  spec: {
    specVersion: "1",
    kind: "Dashboard",
    name: "ops",
    pages: [
      {
        name: "main",
        widgets: [
          {
            id: "note",
            type: "text",
            title: "Caveat",
            content: "hi",
            gridPos: { x: 0, y: 0, w: 12, h: 6 },
          },
        ],
      },
    ],
  },
  publishedSpec: null,
  updatedAt: "2026-09-20T00:00:00Z",
}

afterEach(() => {
  vi.clearAllMocks()
  vi.mocked(resolvePage).mockResolvedValue([])
})

describe("DashboardPage", () => {
  it("rolls a refused tile edit back and keeps its editor open", async () => {
    const user = userEvent.setup()
    vi.mocked(getDashboard).mockResolvedValue(DASHBOARD)
    vi.mocked(saveDashboard).mockRejectedValue(new Error("widget 'note': bad title"))
    render(
      <TooltipProvider>
        <MemoryRouter initialEntries={["/dashboards/d1"]}>
          <Routes>
            <Route path="/dashboards/:id" element={<DashboardPage />} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>,
    )

    const trigger = await screen.findByLabelText("Tile actions: Caveat")
    trigger.focus()
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Edit…" }))
    const dialog = await screen.findByRole("dialog")
    const title = within(dialog).getByLabelText("Title")
    await user.clear(title)
    await user.type(title, "Renamed")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(await within(dialog).findByRole("alert")).toHaveTextContent("bad title")
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    expect(screen.getByLabelText("Tile actions: Caveat")).toBeInTheDocument()
    expect(screen.queryByText("Renamed")).toBeNull()
  })

  it("does not undo one dashboard's edits onto the next one drilled into", async () => {
    // A drill-through stays on this route, so React reuses the page for the new id.
    const user = userEvent.setup()
    const [note] = DASHBOARD.spec.pages[0].widgets
    const first: Dashboard = {
      ...DASHBOARD,
      spec: {
        ...DASHBOARD.spec,
        pages: [
          {
            name: "main",
            widgets: [{ ...note, interactions: { drillThrough: { target: "dashboard:d2" } } }],
          },
        ],
      },
    }
    const second: Dashboard = {
      ...DASHBOARD,
      id: "d2",
      spec: {
        ...DASHBOARD.spec,
        pages: [{ name: "main", widgets: [{ ...note, id: "other", title: "Elsewhere" }] }],
      },
    }
    vi.mocked(getDashboard).mockImplementation(async (id) => (id === "d2" ? second : first))
    vi.mocked(saveDashboard).mockResolvedValue(first)
    render(
      <TooltipProvider>
        <MemoryRouter initialEntries={["/dashboards/d1"]}>
          <Routes>
            <Route path="/dashboards/:id" element={<DashboardPage />} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>,
    )

    const trigger = await screen.findByLabelText("Tile actions: Caveat")
    trigger.focus()
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Edit…" }))
    const title = within(await screen.findByRole("dialog")).getByLabelText("Title")
    await user.clear(title)
    await user.type(title, "Renamed")
    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    await user.click(screen.getByRole("button", { name: "Details →" }))
    await screen.findByLabelText("Tile actions: Elsewhere")
    vi.mocked(saveDashboard).mockClear()
    await user.keyboard("{Control>}z{/Control}")

    expect(saveDashboard).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("Tile actions: Caveat")).toBeNull()
  })

  it("says why the tiles did not load when the server refuses the page", async () => {
    vi.mocked(getDashboard).mockResolvedValue(DASHBOARD)
    // What post() throws for a stored dashboard the current spec refuses.
    vi.mocked(resolvePage).mockRejectedValue(
      new Error(
        'POST /api/dashboards/d1/pages/main/data failed: 404 {"detail": "manifest failed spec validation:\\n' +
          "  - pages/main/widgets/kpi: a 'metric' widget binds a shared metric, not a derivation; " +
          "define the metric, then bind it as {'metric': name}\"}",
      ),
    )
    render(
      <TooltipProvider>
        <MemoryRouter initialEntries={["/dashboards/d1"]}>
          <Routes>
            <Route path="/dashboards/:id" element={<DashboardPage />} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>,
    )

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Tiles not loaded:")
    expect(alert).toHaveTextContent("binds a shared metric, not a derivation")
  })

  describe("when the tiles are fetched again", () => {
    const BOUND: Dashboard = {
      ...DASHBOARD,
      spec: {
        ...DASHBOARD.spec,
        pages: [
          {
            ...DASHBOARD.spec.pages[0],
            widgets: [
              {
                id: "note",
                type: "text",
                title: "Caveat",
                bind: { derivation: "notes" },
                gridPos: { x: 0, y: 0, w: 12, h: 6 },
              },
            ],
          },
        ],
      },
    }
    const figure = (value: string) => [
      {
        widgetId: "note",
        derivation: "notes",
        kind: "text",
        value,
        dataVersion: null,
        error: null,
      },
    ]

    function renderBoard() {
      const { container } = render(
        <TooltipProvider>
          <MemoryRouter initialEntries={["/dashboards/d1"]}>
            <Routes>
              <Route path="/dashboards/:id" element={<DashboardPage />} />
            </Routes>
          </MemoryRouter>
        </TooltipProvider>,
      )
      const refresh = () => {
        const button = container.querySelector(".lucide-refresh-cw")?.closest("button")
        if (!button) throw new Error("no refresh button")
        return button
      }
      return { refresh }
    }

    it("drops the old figures when the new ones are refused", async () => {
      const user = userEvent.setup()
      vi.mocked(getDashboard).mockResolvedValue(BOUND)
      vi.mocked(resolvePage).mockResolvedValueOnce(figure("old figure"))
      const { refresh } = renderBoard()
      expect(await screen.findByText("old figure")).toBeInTheDocument()

      vi.mocked(resolvePage).mockRejectedValueOnce(new Error("503"))
      await user.click(refresh())

      expect(await screen.findByRole("alert")).toHaveTextContent("Tiles not loaded: 503")
      expect(screen.queryByText("old figure")).toBeNull()
    })

    it("ignores an older fetch that finishes after a newer one", async () => {
      const user = userEvent.setup()
      vi.mocked(getDashboard).mockResolvedValue(BOUND)
      let failFirst: (err: Error) => void = () => {}
      vi.mocked(resolvePage)
        .mockReturnValueOnce(new Promise((_, reject) => (failFirst = reject)))
        .mockResolvedValueOnce(figure("new figure"))
      const { refresh } = renderBoard()
      await waitFor(() => expect(resolvePage).toHaveBeenCalledTimes(1))
      await user.click(refresh())
      expect(await screen.findByText("new figure")).toBeInTheDocument()

      failFirst(new Error("stale"))
      await new Promise((resolve) => setTimeout(resolve, 0))

      expect(screen.queryByRole("alert")).toBeNull()
      expect(screen.getByText("new figure")).toBeInTheDocument()
    })
  })
})

describe("resizing a table column on the board", () => {
  const tableDashboard = (viz?: Record<string, unknown>): Dashboard => ({
    ...DASHBOARD,
    spec: {
      ...DASHBOARD.spec,
      pages: [
        {
          name: "main",
          widgets: [
            {
              id: "t",
              type: "table",
              title: "Evidence",
              bind: { derivation: "evidence" },
              ...(viz ? { viz } : {}),
              gridPos: { x: 0, y: 0, w: 24, h: 8 },
            },
          ],
        },
      ],
    },
  })
  const evidence: WidgetData = {
    widgetId: "t",
    derivation: "evidence",
    kind: "table",
    value: [{ unit: "chat", reason: "why" }],
    dataVersion: "v1",
    error: null,
  }
  const open = async (dashboard: Dashboard) => {
    vi.mocked(getDashboard).mockResolvedValue(dashboard)
    vi.mocked(resolvePage).mockResolvedValue([evidence])
    vi.mocked(saveDashboard).mockResolvedValue(undefined as never)
    render(
      <TooltipProvider>
        <MemoryRouter initialEntries={["/dashboards/d1"]}>
          <Routes>
            <Route path="/dashboards/:id" element={<DashboardPage />} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>,
    )
    return screen.findByRole("separator", { name: "Resize column reason" })
  }
  const drag = (handle: HTMLElement, by: number) => {
    fireEvent.mouseDown(handle, { clientX: 100 })
    fireEvent.mouseUp(window, { clientX: 100 + by })
  }
  const savedColumns = () => {
    const spec = vi.mocked(saveDashboard).mock.calls.at(-1)?.[1]
    return spec?.pages[0].widgets[0].viz
  }

  it("saves a dragged width into the tile's viz and keeps the rest of it", async () => {
    const handle = await open(
      tableDashboard({ columns: ["unit", { field: "reason", width: 300 }], pageSize: 5 }),
    )
    drag(handle, 60)
    await waitFor(() => expect(saveDashboard).toHaveBeenCalled())
    expect(savedColumns()).toEqual({
      columns: ["unit", { field: "reason", width: 360 }],
      pageSize: 5,
    })
  })

  it("writes out the derived column list when the tile had none", async () => {
    const handle = await open(tableDashboard())
    // jsdom has no layout, so the starting width is 0 and the drag is the whole width.
    drag(handle, 120)
    await waitFor(() => expect(saveDashboard).toHaveBeenCalled())
    expect(savedColumns()).toEqual({ columns: ["unit", { field: "reason", width: 120 }] })
  })

  it("says why when the server refuses the width", async () => {
    const handle = await open(tableDashboard({ columns: [{ field: "reason", width: 300 }] }))
    vi.mocked(saveDashboard).mockRejectedValue(new Error("viz.columns: bad width"))
    drag(handle, 40)
    expect(await screen.findByText(/Not saved: viz\.columns: bad width/)).toBeInTheDocument()
  })

  it("undoes a width with the same keystroke as any other edit", async () => {
    const user = userEvent.setup()
    const handle = await open(tableDashboard({ columns: [{ field: "reason", width: 300 }] }))
    drag(handle, 60)
    await waitFor(() => expect(saveDashboard).toHaveBeenCalledTimes(1))
    await user.keyboard("{Control>}z{/Control}")
    await waitFor(() => expect(saveDashboard).toHaveBeenCalledTimes(2))
    expect(savedColumns()).toEqual({ columns: [{ field: "reason", width: 300 }] })
  })
})
