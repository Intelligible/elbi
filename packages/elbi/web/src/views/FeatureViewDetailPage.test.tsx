// Pins the feature-view detail's controls: the lookup mode decides which retrieval runs,
// and a consuming model opens its detail page.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { FeatureViewDetail } from "@/lib/features"

vi.mock("@/lib/features", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/features")>()),
  getFeatureViewDetail: vi.fn(),
  onlineFeatures: vi.fn(async () => [{ user_id: "u1", clicks_7d: 4 }]),
  historicalFeatures: vi.fn(async () => [{ user_id: "u1", clicks_7d: 2 }]),
}))
vi.mock("@/lib/lineage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/lineage")>()),
  getImpact: vi.fn(async () => ({ node: "", affected: { model: ["churn"] }, count: 1 })),
}))

import { getFeatureViewDetail, historicalFeatures, onlineFeatures } from "@/lib/features"
import { FeatureViewDetailPage } from "./FeatureViewDetailPage"

const DETAIL: FeatureViewDetail = {
  name: "user_stats",
  entities: ["user"],
  joinKeys: ["user_id"],
  source: "user_activity",
  certified: true,
  timestampField: "event_timestamp",
  ttlSeconds: null,
  features: [{ name: "clicks_7d", dtype: "integer", description: null }],
  description: null,
  nOnlineKeys: 0,
  lastMaterializedAt: null,
  hasContract: false,
  statistics: [],
  drift: [],
  expectations: [],
}

afterEach(() => {
  vi.clearAllMocks()
})

function renderPage() {
  vi.mocked(getFeatureViewDetail).mockResolvedValue(DETAIL)
  return render(
    <MemoryRouter initialEntries={["/features/user_stats"]}>
      <Routes>
        <Route path="/features/:name" element={<FeatureViewDetailPage />} />
        <Route path="/models/:name" element={<div>model page</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe("FeatureViewDetailPage", () => {
  it("looks a key up online by default", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.type(await screen.findByLabelText("user_id"), "u1")
    await user.click(screen.getByRole("button", { name: "Fetch" }))
    expect(onlineFeatures).toHaveBeenCalledWith(["user_stats:clicks_7d"], [{ user_id: "u1" }])
    expect(historicalFeatures).not.toHaveBeenCalled()
  })

  it("looks a key up as of a timestamp in point-in-time mode", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole("button", { name: "Point-in-time" }))
    await user.type(screen.getByLabelText("user_id"), "u1")
    await user.type(screen.getByLabelText("As of (event_timestamp)"), "2024-03-15")
    await user.click(screen.getByRole("button", { name: "Fetch" }))
    expect(historicalFeatures).toHaveBeenCalledWith(
      ["user_stats:clicks_7d"],
      [{ user_id: "u1", event_timestamp: "2024-03-15" }],
    )
    expect(onlineFeatures).not.toHaveBeenCalled()
  })

  it("opens a consuming model", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole("button", { name: "churn" }))
    expect(await screen.findByText("model page")).toBeInTheDocument()
  })
})
