import { render, screen } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import type { NotebookView } from "@/lib/notebooks"

vi.mock("@/lib/notebooks", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/notebooks")>()),
  getNotebook: vi.fn(),
  getNotebookVariables: vi.fn(async () => []),
}))

import { getNotebook } from "@/lib/notebooks"
import { NotebookPage } from "./NotebookPage"

const VIEW: NotebookView = {
  id: "nb1",
  name: "Q3 report",
  folder_id: "fold1",
  folder_name: "Research",
  deps: [],
  metadata: {},
  schedule: null,
  kernel_status: "idle",
  cells: [],
  graph: { cells: {}, conflicts: {}, cycle: [] },
  environment: {
    base_env: null,
    base_environments: [],
    lock: null,
    locked: false,
  },
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/notebooks/nb1"]}>
      <Routes>
        <Route path="/notebooks/:name" element={<NotebookPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe("NotebookPage back link", () => {
  it("returns to the folder the notebook lives in", async () => {
    vi.mocked(getNotebook).mockResolvedValue(VIEW)
    renderPage()
    const back = await screen.findByRole("link", { name: "Research" })
    expect(back.getAttribute("href")).toBe("/notebooks?folder=fold1")
  })

  it("returns to the notebooks root for a notebook outside any folder", async () => {
    vi.mocked(getNotebook).mockResolvedValue({
      ...VIEW,
      folder_id: null,
      folder_name: null,
    })
    renderPage()
    const back = await screen.findByRole("link", { name: "Notebooks" })
    expect(back.getAttribute("href")).toBe("/notebooks")
  })
})
