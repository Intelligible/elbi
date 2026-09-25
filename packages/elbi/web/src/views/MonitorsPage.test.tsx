// Pins the monitors list: choosing a monitor opens its detail and loads its history.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { Monitor } from "@/lib/monitors"

vi.mock("@/components/viz/VizView", () => ({ VizView: () => <div data-testid="viz" /> }))
vi.mock("@/lib/monitors", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/monitors")>()),
  listMonitors: vi.fn(),
  getMonitor: vi.fn(async () => ({ snapshots: [], incidents: [] })),
}))

import { getMonitor, listMonitors } from "@/lib/monitors"
import { MonitorsPage } from "./MonitorsPage"

const monitor = (id: string, name: string, status: Monitor["status"]): Monitor => ({
  id,
  name,
  targetKind: "metric",
  target: "revenue",
  config: {},
  method: "mad",
  sensitivity: 3,
  minValue: null,
  maxValue: null,
  window: 30,
  intervalHours: 24,
  enabled: true,
  lastValue: null,
  lastCheckedAt: null,
  status,
})

afterEach(() => {
  vi.clearAllMocks()
})

describe("MonitorsPage", () => {
  it("opens the chosen monitor", async () => {
    const user = userEvent.setup()
    vi.mocked(listMonitors).mockResolvedValue([
      monitor("m1", "Revenue watch", "ok"),
      monitor("m2", "Churn spike", "alerting"),
    ])
    render(<MonitorsPage />)
    expect(await screen.findByText("Select a monitor, or create one.")).toBeInTheDocument()
    await user.click(await screen.findByRole("button", { name: /Churn spike/ }))
    expect(getMonitor).toHaveBeenCalledWith("m2")
    expect(screen.queryByText("Select a monitor, or create one.")).toBeNull()
  })
})
