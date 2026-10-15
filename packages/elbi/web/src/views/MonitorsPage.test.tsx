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
  updateMonitor: vi.fn(async (id: string) => ({ id })),
}))
vi.mock("@/lib/metrics", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/metrics")>()),
  listMetrics: vi.fn(async () => []),
  listDerivations: vi.fn(async () => [{ name: "workflow_health" }]),
}))

import { getMonitor, listMonitors, updateMonitor } from "@/lib/monitors"
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
  breachKeys: null,
})

const keyed = (): Monitor => ({
  ...monitor("m3", "Workflow errors", "alerting"),
  targetKind: "derivation",
  target: "workflow_health",
  config: { measure: { agg: "sum", column: "error_breach", key: "workflow" } },
  maxValue: 0,
  breachKeys: ["Brain Compactor", ""],
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

  it("names the breaching rows and the keys of each incident", async () => {
    const user = userEvent.setup()
    vi.mocked(listMonitors).mockResolvedValue([keyed()])
    vi.mocked(getMonitor).mockResolvedValue({
      snapshots: [],
      incidents: [
        {
          id: "i1",
          openedAt: "2026-10-13T10:00:00",
          closedAt: null,
          cause: null,
          peakValue: 2,
          peakScore: null,
          reason: "2 is above the ceiling 0",
          snapshots: 2,
          open: true,
          keys: ["Brain Compactor", "Lead capture"],
        },
      ],
    })
    render(<MonitorsPage />)
    await user.click(await screen.findByRole("button", { name: /Workflow errors/ }))
    expect(await screen.findByText(/sum of error_breach, naming rows by workflow/)).toBeVisible()
    expect(screen.getByText("Breaching now (2): Brain Compactor, (blank)")).toBeVisible()
    expect(await screen.findByText(/Brain Compactor, Lead capture/)).toBeVisible()
  })

  it("edits a keyed monitor without losing its aggregation or key", async () => {
    const user = userEvent.setup()
    vi.mocked(listMonitors).mockResolvedValue([keyed()])
    render(<MonitorsPage />)
    await user.click(await screen.findByRole("button", { name: /Workflow errors/ }))
    await user.click(await screen.findByRole("button", { name: "Edit monitor" }))
    expect(screen.getByPlaceholderText("key column naming each row (optional)")).toHaveValue(
      "workflow",
    )
    await user.click(screen.getByRole("button", { name: "Save monitor" }))
    expect(updateMonitor).toHaveBeenCalledWith(
      "m3",
      expect.objectContaining({
        target: "workflow_health",
        maxValue: 0,
        config: { measure: { agg: "sum", column: "error_breach", key: "workflow" } },
      }),
    )
  })
})
