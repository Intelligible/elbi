// Pins the jobs strip: an active job can be cancelled, and a finished certified job
// announces itself once so the chat can pull the follow-up.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { Job } from "@/lib/chat"

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getJobs: vi.fn(),
  cancelJob: vi.fn(async () => {}),
}))

import { cancelJob, getJobs } from "@/lib/chat"
import { JobsBar } from "./JobsBar"

function job(over: Partial<Job>): Job {
  return {
    id: "j1",
    label: "effect_of_x",
    state: "running",
    progress: "epoch 3/10",
    result: null,
    error: null,
    createdAt: 0,
    startedAt: 0,
    finishedAt: null,
    ...over,
  }
}

afterEach(() => {
  vi.clearAllMocks()
})

describe("JobsBar", () => {
  it("cancels an active job", async () => {
    const user = userEvent.setup()
    vi.mocked(getJobs).mockResolvedValue([job({})])
    render(<JobsBar />)
    await user.click(await screen.findByRole("button", { name: "Cancel effect_of_x" }))
    expect(cancelJob).toHaveBeenCalledWith("j1")
  })

  it("offers no cancel for a finished job and announces certification once", async () => {
    const onCertifiedComplete = vi.fn()
    vi.mocked(getJobs).mockResolvedValue([
      job({ state: "succeeded", result: { certified: true }, finishedAt: Date.now() / 1000 }),
    ])
    render(<JobsBar onCertifiedComplete={onCertifiedComplete} />)
    expect(await screen.findByText("certified")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Cancel/ })).toBeNull()
    expect(onCertifiedComplete).toHaveBeenCalledOnce()
  })
})
