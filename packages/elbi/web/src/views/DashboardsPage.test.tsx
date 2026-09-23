// Pins the dashboards list's row actions: Duplicate copies and opens the copy, Delete
// removes the dashboard it sits on.

import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { DashboardSummary } from "@/lib/dashboards"

vi.mock("@/lib/dashboards", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/dashboards")>()),
  listDashboards: vi.fn(),
  deleteDashboard: vi.fn(async () => ({ ok: true })),
  duplicateDashboard: vi.fn(async () => ({ id: "d9" })),
}))

import { deleteDashboard, duplicateDashboard, listDashboards } from "@/lib/dashboards"
import { DashboardsPage } from "./DashboardsPage"

const ROWS: DashboardSummary[] = [
  {
    id: "d1",
    name: "sales",
    title: "Sales overview",
    copiedFrom: null,
    status: "published",
    version: 3,
    updatedAt: "2026-09-20T00:00:00Z",
  },
  {
    id: "d2",
    name: "ops",
    title: "",
    copiedFrom: null,
    status: "draft",
    version: 1,
    updatedAt: "2026-09-21T00:00:00Z",
  },
]

afterEach(() => {
  vi.clearAllMocks()
})

function renderPage() {
  vi.mocked(listDashboards).mockResolvedValue(ROWS)
  return render(
    <TooltipProvider>
      <MemoryRouter initialEntries={["/dashboards"]}>
        <Routes>
          <Route path="/dashboards" element={<DashboardsPage />} />
          <Route path="/dashboards/:id" element={<div>opened dashboard</div>} />
        </Routes>
      </MemoryRouter>
    </TooltipProvider>,
  )
}

describe("DashboardsPage row actions", () => {
  it("duplicates a dashboard and opens the copy", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole("button", { name: "Duplicate ops" }))
    expect(duplicateDashboard).toHaveBeenCalledWith("d2")
    expect(await screen.findByText("opened dashboard")).toBeInTheDocument()
  })

  it("deletes the dashboard on its row", async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Sales overview")
    const deletes = screen.getAllByRole("button", { name: /^Delete/ })
    expect(deletes).toHaveLength(2)
    await user.click(deletes[0])
    await waitFor(() => expect(deleteDashboard).toHaveBeenCalledWith("d1"))
  })
})
