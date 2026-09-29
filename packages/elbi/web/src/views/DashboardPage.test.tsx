// Pins what a refused save does: the board goes back to the last spec the server
// accepted, and the editor that made the edit stays open with the server's reason.

import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { Dashboard } from "@/lib/dashboards"

vi.mock("@/components/dashboard/JsonEditor", () => ({ JsonEditor: () => null }))

vi.mock("@/lib/dashboards", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/dashboards")>()),
  getDashboard: vi.fn(),
  resolvePage: vi.fn(async () => []),
  bindableDerivations: vi.fn(async () => []),
  dashboardSchema: vi.fn(async () => null),
  saveDashboard: vi.fn(),
}))

import { getDashboard, saveDashboard } from "@/lib/dashboards"
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
})
