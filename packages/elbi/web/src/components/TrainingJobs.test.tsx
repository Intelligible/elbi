// Pins TrainingStatus's dismiss control before its raw <button> moves to IconButton
// (task-9-brief.md R2): dismissing drives useTrainingJobs' dismissed set, which is real
// filtering logic, not just styling.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { TrainingJob } from "@/lib/chat"
import { TrainingStatus } from "./TrainingJobs"

const JOB: TrainingJob = {
  id: "j1",
  label: "train model churn",
  state: "succeeded",
  progress: "",
  result: {
    name: "churn",
    version: 3,
    task: "classification",
    target: "churned",
    features: [],
    bestEstimator: "lgbm",
    metrics: {},
    oracleVerdict: "sound",
    champion: true,
    rendered: "",
  },
  error: null,
  createdAt: 0,
  finishedAt: 1,
}

function mount(ui: React.ReactElement) {
  return render(<TooltipProvider>{ui}</TooltipProvider>)
}

describe("TrainingStatus", () => {
  it("an active job has no dismiss control", () => {
    mount(<TrainingStatus job={{ ...JOB, state: "running" }} onDismiss={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "Dismiss" })).toBeNull()
  })

  it("clicking dismiss on a terminal job calls onDismiss", async () => {
    const onDismiss = vi.fn()
    const user = userEvent.setup()
    mount(<TrainingStatus job={JOB} onDismiss={onDismiss} />)
    await user.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(onDismiss).toHaveBeenCalledTimes(1)
  })
})
