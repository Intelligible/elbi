import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import type { ModelVersion, RegisteredModelDetail } from "@/lib/chat"

vi.mock("@/lib/notebooks", () => ({ notebookFromModel: vi.fn() }))

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getRegisteredModel: vi.fn(),
  getServingSchema: vi.fn(async () => null),
  listDriftChecks: vi.fn(async () => []),
}))

import { getRegisteredModel } from "@/lib/chat"
import { notebookFromModel } from "@/lib/notebooks"
import { ModelDetailPage } from "./ModelDetailPage"

const version = (n: number): ModelVersion => ({
  version: n,
  runId: `run${n}`,
  experimentId: "1",
  createdAtMs: 1_700_000_000_000 + n,
  aliases: [],
  description: "",
  metrics: {},
  params: {},
  tags: {},
  verdict: "inconclusive",
  verdictDetail: "no signal",
})

const detail = (extra: Partial<RegisteredModelDetail> = {}): RegisteredModelDetail => ({
  name: "cost_model",
  championVersion: null,
  versions: [version(1)],
  ...extra,
})

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/models/cost_model"]}>
      <Routes>
        <Route path="/models/:name" element={<ModelDetailPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe("ModelDetailPage: open in notebook", () => {
  it("opens the newest version when no version is the champion", async () => {
    // Without a version the server resolves @champion, which an all-inconclusive model
    // has none of: the request 404s and the button appears to do nothing at all.
    const user = userEvent.setup()
    vi.mocked(getRegisteredModel).mockResolvedValue(detail({ versions: [version(1), version(2)] }))
    vi.mocked(notebookFromModel).mockResolvedValue({ id: "nb1" })
    renderPage()

    await user.click(await screen.findByRole("button", { name: /Open in notebook/ }))

    expect(notebookFromModel).toHaveBeenCalledWith("cost_model", "2")
  })

  it("opens the champion when there is one", async () => {
    const user = userEvent.setup()
    vi.mocked(getRegisteredModel).mockResolvedValue(
      detail({ championVersion: 1, versions: [version(1), version(2)] }),
    )
    vi.mocked(notebookFromModel).mockResolvedValue({ id: "nb1" })
    renderPage()

    await user.click(await screen.findByRole("button", { name: /Open in notebook/ }))

    expect(notebookFromModel).toHaveBeenCalledWith("cost_model", "1")
  })

  it("says why nothing opened rather than failing silently", async () => {
    const user = userEvent.setup()
    vi.mocked(getRegisteredModel).mockResolvedValue(detail())
    vi.mocked(notebookFromModel).mockRejectedValue(
      new Error("that model version has no training script"),
    )
    renderPage()

    await user.click(await screen.findByRole("button", { name: /Open in notebook/ }))

    expect(await screen.findByText(/no training script/)).toBeInTheDocument()
  })

  it("offers nothing to open when the model has no versions", async () => {
    vi.mocked(getRegisteredModel).mockResolvedValue(detail({ versions: [] }))
    renderPage()

    await screen.findByText("cost_model")
    expect(screen.queryByRole("button", { name: /Open in notebook/ })).toBeNull()
  })
})
