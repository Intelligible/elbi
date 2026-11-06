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
  listDerivationColumns: vi.fn(async () => [
    { name: "workflow", numeric: false },
    { name: "error_breach", numeric: true },
    { name: "owner", numeric: false },
  ]),
}))
vi.mock("@/lib/metrics", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/metrics")>()),
  listMetrics: vi.fn(async () => []),
  listDerivations: vi.fn(async () => [{ name: "workflow_health" }]),
}))

import { listDerivations } from "@/lib/metrics"
import { getMonitor, listDerivationColumns, listMonitors, updateMonitor } from "@/lib/monitors"
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
    expect(
      await screen.findByRole("combobox", { name: "Key column (optional)" }),
    ).toHaveTextContent("workflow")
    await user.click(screen.getByRole("button", { name: "Save monitor" }))
    expect(updateMonitor).toHaveBeenCalledWith(
      "m3",
      expect.objectContaining({
        target: undefined,
        maxValue: 0,
        config: { measure: { agg: "sum", column: "error_breach", key: "workflow" } },
      }),
    )
  })

  it("picks the key from the derivation's own columns", async () => {
    const user = userEvent.setup()
    vi.mocked(listMonitors).mockResolvedValue([keyed()])
    render(<MonitorsPage />)
    await user.click(await screen.findByRole("button", { name: /Workflow errors/ }))
    await user.click(await screen.findByRole("button", { name: "Edit monitor" }))
    await user.click(await screen.findByRole("combobox", { name: "Key column (optional)" }))
    await user.type(screen.getByPlaceholderText("Search columns"), "own")
    expect(screen.queryByRole("option", { name: /workflow/ })).toBeNull()
    await user.click(screen.getByRole("option", { name: /owner/ }))
    await user.click(screen.getByRole("button", { name: "Save monitor" }))
    expect(updateMonitor).toHaveBeenCalledWith(
      "m3",
      expect.objectContaining({
        config: { measure: { agg: "sum", column: "error_breach", key: "owner" } },
      }),
    )
  })

  it("will not create a monitor on a derivation whose rows cannot be read", async () => {
    const user = userEvent.setup()
    vi.mocked(listMonitors).mockResolvedValue([])
    vi.mocked(listDerivations).mockResolvedValue([{ name: "pricing_effects" }] as Awaited<
      ReturnType<typeof listDerivations>
    >)
    vi.mocked(listDerivationColumns).mockRejectedValue(
      new Error("derivation 'pricing_effects' does not produce a table of rows"),
    )
    render(<MonitorsPage />)
    await user.click(await screen.findByRole("button", { name: /New monitor/ }))
    await user.click(screen.getAllByRole("combobox")[0])
    await user.click(await screen.findByRole("option", { name: "derivation" }))
    await user.click(screen.getAllByRole("combobox")[1])
    await user.click(await screen.findByRole("option", { name: "pricing_effects" }))
    expect(await screen.findByText(/Cannot be monitored: .*does not produce a table/)).toBeVisible()
    expect(screen.getByRole("button", { name: "Create monitor" })).toBeDisabled()
  })

  it("saves a rename without naming the target or dropping settings it does not edit", async () => {
    const user = userEvent.setup()
    const grouped: Monitor = {
      ...monitor("m4", "Revenue by region", "ok"),
      config: { group_by: ["region"], grain: "day" },
    }
    vi.mocked(listMonitors).mockResolvedValue([grouped])
    render(<MonitorsPage />)
    await user.click(await screen.findByRole("button", { name: /Revenue by region/ }))
    await user.click(await screen.findByRole("button", { name: "Edit monitor" }))
    await user.click(screen.getByRole("button", { name: "Save monitor" }))
    const spec = vi.mocked(updateMonitor).mock.calls.at(-1)?.[1]
    expect(spec?.config).toEqual({ group_by: ["region"], grain: "day" })
    expect(spec?.target).toBeUndefined()
    expect(spec?.targetKind).toBeUndefined()
  })

  it("will not save a history window below one or a fractional one", async () => {
    const user = userEvent.setup()
    vi.mocked(listMonitors).mockResolvedValue([monitor("m5", "Revenue watch", "ok")])
    render(<MonitorsPage />)
    await user.click(await screen.findByRole("button", { name: /Revenue watch/ }))
    await user.click(await screen.findByRole("button", { name: "Edit monitor" }))
    const window = screen.getByLabelText("History window (checks)")
    await user.clear(window)
    await user.type(window, "-5")
    expect(screen.getByText("History window must be above 0.")).toBeVisible()
    expect(screen.getByRole("button", { name: "Save monitor" })).toBeDisabled()
    await user.clear(window)
    await user.type(window, "2.5")
    expect(screen.getByText("History window must be a whole number.")).toBeVisible()
  })
})
