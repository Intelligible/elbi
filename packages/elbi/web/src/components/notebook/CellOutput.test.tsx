import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  copyText: vi.fn(async () => true),
}))

import { copyText } from "@/lib/utils"
import { CellOutputs } from "./CellOutput"

const OUTPUTS = [
  { output_type: "stream", name: "stdout", text: "hello\n" },
  { output_type: "stream", name: "stdout", text: "world\n" },
] as never

const mount = () =>
  render(
    <TooltipProvider>
      <CellOutputs outputs={OUTPUTS} />
    </TooltipProvider>,
  )

describe("CellOutputs controls", () => {
  afterEach(() => vi.clearAllMocks())

  it("copies every output's text", async () => {
    mount()
    await userEvent.click(screen.getByRole("button", { name: "Copy output" }))
    expect(copyText).toHaveBeenCalledWith("hello\nworld")
  })

  it("collapses from the Out rail and expands from the hidden notice", async () => {
    mount()
    await userEvent.click(screen.getByRole("button", { name: /^out$|hide output/i }))
    expect(screen.queryByText(/hello/)).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: /2 outputs hidden/ }))
    expect(screen.getByText(/hello/)).toBeInTheDocument()
  })
})
