// Pins the orchestration page's controls: each test drives a control that switches the
// panel, selects a run, or changes a request (checks, workflows, the retry policy), so a
// mapping mistake shows up as a wrong API call rather than only as an eslint pass.

import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { AssetStatus, RunDetail, RunHistory } from "@/lib/orchestration"

vi.mock("@/lib/orchestration", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/orchestration")>()),
  getAssetStatus: vi.fn(),
  getGraph: vi.fn(),
  getHistory: vi.fn(),
  getSchedules: vi.fn(),
  getChecks: vi.fn(),
  getSettings: vi.fn(),
  getWorkflows: vi.fn(),
  getRun: vi.fn(),
  materialize: vi.fn(),
  upsertCheck: vi.fn(),
  deleteCheck: vi.fn(),
  upsertWorkflow: vi.fn(),
  deleteWorkflow: vi.fn(),
  deleteSchedule: vi.fn(),
  setRetryPolicy: vi.fn(),
}))

import {
  deleteCheck,
  deleteSchedule,
  deleteWorkflow,
  getAssetStatus,
  getChecks,
  getGraph,
  getHistory,
  getRun,
  getSchedules,
  getSettings,
  getWorkflows,
  materialize,
  setRetryPolicy,
  upsertCheck,
  upsertWorkflow,
} from "@/lib/orchestration"
import { OrchestrationPage } from "./OrchestrationPage"

const ASSETS: AssetStatus[] = [
  {
    asset: "orders_clean",
    status: "materialized",
    verdict: null,
    lastMaterializedAt: new Date().toISOString(),
    lastDurationMs: 1200,
  },
  {
    asset: "revenue",
    status: "stale",
    verdict: null,
    lastMaterializedAt: null,
    lastDurationMs: null,
  },
]

const HISTORY: RunHistory = {
  assets: ["orders_clean", "revenue"],
  runs: [
    {
      id: "r2",
      cause: "manual",
      status: "failed",
      parentRunId: null,
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      counts: {},
      cells: { orders_clean: "succeeded", revenue: "failed" },
    },
    {
      id: "r1",
      cause: "schedule",
      status: "succeeded",
      parentRunId: null,
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      counts: {},
      cells: { orders_clean: "succeeded" },
    },
  ],
}

const DETAIL: RunDetail = {
  id: "r2",
  cause: "manual",
  status: "failed",
  parentRunId: null,
  startedAt: new Date().toISOString(),
  finishedAt: new Date().toISOString(),
  steps: [
    {
      asset: "revenue",
      state: "failed",
      verdict: null,
      error: "division by zero",
      attempts: 1,
      durationMs: 40,
      logs: "step output",
      checks: [],
    },
  ],
}

function renderPage() {
  return render(
    <TooltipProvider>
      <OrchestrationPage />
    </TooltipProvider>,
  )
}

// jsdom fires a window blur on pointerdown once an earlier test's cleanup leaves nothing
// focused, which closes a just-opened Radix Select; focusing the trigger first keeps it
// open. A jsdom artefact, not a product bug.
async function pick(
  user: ReturnType<typeof userEvent.setup>,
  combobox: HTMLElement,
  option: string,
) {
  if (combobox instanceof HTMLSelectElement) {
    await user.selectOptions(combobox, within(combobox).getByRole("option", { name: option }))
    return
  }
  combobox.focus()
  await user.click(combobox)
  await user.click(screen.getByRole("option", { name: option }))
}

const tab = (name: string | RegExp) => screen.getByRole("tab", { name })

describe("OrchestrationPage", () => {
  beforeEach(() => {
    vi.mocked(getAssetStatus).mockResolvedValue(ASSETS)
    vi.mocked(getGraph).mockResolvedValue({ nodes: [], edges: [] })
    vi.mocked(getHistory).mockResolvedValue(HISTORY)
    vi.mocked(getSchedules).mockResolvedValue([
      {
        id: "s1",
        name: "nightly",
        selection: "stale",
        mode: "cron",
        cron: "0 6 * * *",
        dataset: null,
        enabled: true,
        lastRunAt: null,
      },
    ])
    vi.mocked(getChecks).mockResolvedValue([
      {
        id: "c1",
        asset: "orders_clean",
        name: "positive",
        expr: "amount >= 0",
        severity: "warn",
        enabled: true,
      },
    ])
    vi.mocked(getSettings).mockResolvedValue({ maxRetries: 1, compute: "" })
    vi.mocked(getWorkflows).mockResolvedValue([
      { id: "w1", name: "daily", steps: [{ id: "step1", selection: "stale" }] },
    ])
    vi.mocked(getRun).mockResolvedValue(DETAIL)
    for (const fn of [
      materialize,
      upsertCheck,
      deleteCheck,
      upsertWorkflow,
      deleteWorkflow,
      deleteSchedule,
      setRetryPolicy,
    ]) {
      vi.mocked(fn).mockResolvedValue(undefined as never)
    }
  })
  afterEach(() => vi.clearAllMocks())

  it("switches panels from the tab strip and shows the stale count on Assets", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(tab(/Assets/)).toHaveTextContent("Assets1"))
    expect(screen.getByText("Success rate")).toBeInTheDocument()

    await user.click(tab("Runs"))
    expect(tab("Runs")).toHaveAttribute("aria-selected", "true")
    expect(screen.getByText("Select a run to inspect its steps, timings, and logs.")).toBeVisible()
    await user.click(tab("Schedules"))
    expect(screen.getByText("nightly")).toBeInTheDocument()
  })

  it("selects a run from a run-history cell, toggles it off, and closes the detail", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(getHistory).toHaveBeenCalled())
    await user.click(tab("Runs"))

    const run = await screen.findByRole("button", { name: /^manual · failed/ })
    expect(run).toHaveAttribute("aria-pressed", "false")
    await user.click(run)
    await waitFor(() => expect(getRun).toHaveBeenCalledWith("r2"))
    await waitFor(() => expect(run).toHaveAttribute("aria-pressed", "true"))
    expect(await screen.findByRole("button", { name: "Close run detail" })).toBeInTheDocument()

    // A cell of the selected run deselects it.
    await user.click(screen.getByRole("button", { name: /^revenue · failed/ }))
    expect(await screen.findByText(/Select a run to inspect/)).toBeInTheDocument()

    await user.click(screen.getAllByRole("button", { name: /^orders_clean · succeeded/ })[0])
    await screen.findByRole("button", { name: "Close run detail" })
    await user.click(screen.getByRole("button", { name: "logs" }))
    expect(screen.getByText(/division by zero/)).toBeInTheDocument()
    expect(screen.getByText(/step output/)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Close run detail" }))
    expect(screen.getByText(/Select a run to inspect/)).toBeInTheDocument()
  })

  it("adds a check with the picked severity and deletes an existing one", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(getAssetStatus).toHaveBeenCalled())
    await user.click(tab(/Assets/))

    await user.click(await screen.findByRole("button", { name: "1" }))
    await user.type(screen.getByPlaceholderText("name"), "nonneg")
    await user.type(screen.getByPlaceholderText("e.g. amount >= 0"), "x > 0")
    await pick(user, screen.getByRole("combobox", { name: "Severity" }), "error")
    await user.click(screen.getByRole("button", { name: "Add" }))
    await waitFor(() =>
      expect(upsertCheck).toHaveBeenCalledWith({
        asset: "orders_clean",
        name: "nonneg",
        expr: "x > 0",
        severity: "error",
      }),
    )

    await user.click(screen.getByRole("button", { name: "Delete check positive" }))
    await waitFor(() => expect(deleteCheck).toHaveBeenCalledWith("c1"))
  })

  it("filters assets by freshness and materializes one", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(getAssetStatus).toHaveBeenCalled())
    await user.click(tab(/Assets/))

    const filter = await screen.findByRole("button", { name: "Stale 1" })
    await user.click(filter)
    expect(filter).toHaveAttribute("aria-pressed", "true")
    expect(screen.queryByText("orders_clean")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Group/ })).toBeDisabled()

    await user.click(screen.getByRole("button", { name: "Materialize revenue and its downstream" }))
    await waitFor(() =>
      expect(materialize).toHaveBeenCalledWith({ assets: ["revenue"], includeDownstream: true }),
    )
  })

  it("builds a workflow from the step selects and deletes one", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(getWorkflows).toHaveBeenCalled())
    await user.click(tab("Workflows"))

    await user.type(screen.getByPlaceholderText("workflow name"), "nightly")
    await user.click(screen.getByRole("button", { name: "Step" }))
    await user.type(screen.getAllByPlaceholderText("depends on (ids)")[1], "step1")
    expect(screen.getAllByRole("combobox", { name: "Selection" })).toHaveLength(2)
    expect(screen.getAllByRole("combobox", { name: "Run if" })).toHaveLength(2)
    await pick(user, screen.getAllByRole("combobox", { name: "Selection" })[1], "materialize all")
    await pick(user, screen.getAllByRole("combobox", { name: "Run if" })[1], "run if none failed")
    await user.click(screen.getByRole("button", { name: "Create workflow" }))
    await waitFor(() =>
      expect(upsertWorkflow).toHaveBeenCalledWith({
        id: undefined,
        name: "nightly",
        steps: [
          { id: "step1", selection: "stale", dependsOn: [], runIf: "all_success" },
          { id: "step2", selection: "all", dependsOn: ["step1"], runIf: "none_failed" },
        ],
      }),
    )

    await user.click(screen.getByRole("button", { name: "Delete workflow daily" }))
    await waitFor(() => expect(deleteWorkflow).toHaveBeenCalledWith("w1"))
  })

  it("removes a draft step", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(getWorkflows).toHaveBeenCalled())
    await user.click(tab("Workflows"))
    await user.click(screen.getByRole("button", { name: "Step" }))
    expect(screen.getAllByPlaceholderText("step id")).toHaveLength(2)
    await user.click(screen.getAllByRole("button", { name: "Remove step" })[0])
    expect(screen.getAllByPlaceholderText("step id")).toHaveLength(1)
    expect(screen.getByPlaceholderText("step id")).toHaveValue("step2")
  })

  it("sets the retry policy and deletes a schedule", async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(getSettings).toHaveBeenCalled())
    await user.click(tab("Schedules"))

    const retries = screen.getByRole("combobox", { name: "Retry a failed asset" })
    await pick(user, retries, "3 times")
    await waitFor(() => expect(setRetryPolicy).toHaveBeenCalledWith(3))

    await user.click(
      within(screen.getByText("nightly").closest("div") as HTMLElement).getByRole("button", {
        name: "Delete schedule",
      }),
    )
    await waitFor(() => expect(deleteSchedule).toHaveBeenCalledWith("s1"))
  })
})
