import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { X } from "lucide-react"
import { describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"

import { IconButton } from "./IconButton"

const renderIn = (ui: React.ReactElement) =>
  render(<TooltipProvider delayDuration={0}>{ui}</TooltipProvider>)

describe("IconButton", () => {
  it("is a button named by its label", async () => {
    const onClick = vi.fn()
    renderIn(
      <IconButton label="Close" onClick={onClick}>
        <X />
      </IconButton>,
    )
    await userEvent.click(screen.getByRole("button", { name: "Close" }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("shows the label as a tooltip on hover", async () => {
    renderIn(
      <IconButton label="Close">
        <X />
      </IconButton>,
    )
    await userEvent.hover(screen.getByRole("button", { name: "Close" }))
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Close")
  })

  it("defaults to a ghost icon button", () => {
    renderIn(
      <IconButton label="Close">
        <X />
      </IconButton>,
    )
    const button = screen.getByRole("button", { name: "Close" })
    expect(button).toHaveAttribute("data-variant", "ghost")
    expect(button).toHaveAttribute("data-size", "icon-sm")
  })
})
