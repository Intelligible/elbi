import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { ComputeMenu, ComputeProfile, NotebookCompute } from "@/lib/notebooks"

vi.mock("@/lib/notebooks", () => ({
  getComputeProfiles: vi.fn(),
  getNotebookCompute: vi.fn(),
  updateNotebook: vi.fn(async () => ({ ok: true })),
}))

import { getComputeProfiles, getNotebookCompute, updateNotebook } from "@/lib/notebooks"
import { ComputeDialog } from "./ComputeDialog"

const profile = (over: Partial<ComputeProfile> = {}): ComputeProfile => ({
  name: "small",
  cpu: "2",
  memory: "4Gi",
  gpu: 0,
  gpuType: null,
  image: null,
  idleTimeout: 1800,
  maxRuntime: 120,
  egress: "full",
  allowedGroups: [],
  spot: false,
  warmPoolSize: 0,
  runtimeClass: null,
  pids: 512,
  version: "aaa111",
  costPerHour: 0.1,
  ...over,
})

const MENU: ComputeMenu = {
  default: "small",
  maxCostPerHour: null,
  profiles: [
    profile(),
    profile({ name: "gpu", gpu: 1, gpuType: "nvidia-l4", costPerHour: 1.4, version: "bbb" }),
  ],
}

function open(state: Partial<NotebookCompute> = {}) {
  vi.mocked(getComputeProfiles).mockResolvedValue(MENU)
  vi.mocked(getNotebookCompute).mockResolvedValue({
    profile: profile(),
    profileVersion: "aaa111",
    status: "idle",
    drift: null,
    unavailable: null,
    ...state,
  })
  return render(<ComputeDialog notebookId="nb1" open onOpenChange={() => {}} />)
}

describe("ComputeDialog", () => {
  it("shows what each size is, so a choice can be made without documentation", async () => {
    open()
    await screen.findByText("small")
    // Shape and price, because those are what a person is choosing between.
    // Both profiles are 2 CPU / 4Gi here, so the shape line is matched per row.
    expect(screen.getAllByText(/2 CPU · 4Gi/)).toHaveLength(2)
    expect(screen.getByText(/1× nvidia-l4/)).toBeTruthy()
    expect(screen.getByText("~10.0¢/hr")).toBeTruthy()
    expect(screen.getByText("~$1.40/hr")).toBeTruthy()
    expect(screen.getByText("default")).toBeTruthy()
  })

  it("saves the picked profile to the notebook", async () => {
    open()
    await screen.findByText("gpu")
    await userEvent.click(screen.getByText("gpu"))
    await userEvent.click(screen.getByRole("button", { name: /use this size/i }))
    await waitFor(() =>
      expect(updateNotebook).toHaveBeenCalledWith("nb1", { compute_profile: "gpu" }),
    )
  })

  it("says when a running kernel no longer matches its profile", async () => {
    // The whole point of tracking a profile version: an admin who tightened a limit
    // should not be left believing it applied to sessions already running.
    open({
      status: "running",
      drift: "running on small as it was defined at version aaa111; restart to pick it up.",
    })
    expect(await screen.findByText(/restart to pick it up/i)).toBeTruthy()
  })

  it("says so when the caller may use nothing", async () => {
    vi.mocked(getComputeProfiles).mockResolvedValue({ ...MENU, profiles: [] })
    vi.mocked(getNotebookCompute).mockResolvedValue({
      profile: profile(),
      profileVersion: "aaa111",
      status: "idle",
      drift: null,
      unavailable: null,
    })
    render(<ComputeDialog notebookId="nb1" open onOpenChange={() => {}} />)
    expect(await screen.findByText(/no compute profiles are available/i)).toBeTruthy()
  })

  it("says when the size a notebook asked for is gone", async () => {
    // Falling back keeps the notebook openable, but silently running something other
    // than what was asked for is how a resized limit goes unnoticed.
    open({ unavailable: "unknown compute profile 'huge'" })
    expect(await screen.findByText(/no longer offered, so it is using the default/i)).toBeTruthy()
  })
})
