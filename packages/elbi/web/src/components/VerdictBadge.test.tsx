import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { EMPTY } from "@/lib/utils"
import { estimateText, VerdictBadge } from "./VerdictBadge"

describe("VerdictBadge", () => {
  it("maps a sound verdict to the 'Verified' label", () => {
    render(<VerdictBadge verdict="sound" />)
    expect(screen.getByText("Verified")).toBeTruthy()
  })

  it("labels an unsound verdict as 'Not sound' (never dressed as verified)", () => {
    render(<VerdictBadge verdict="unsound" />)
    expect(screen.getByText("Not sound")).toBeTruthy()
  })

  it("passes an unknown verdict through verbatim", () => {
    render(<VerdictBadge verdict="mystery" />)
    expect(screen.getByText("mystery")).toBeTruthy()
  })
})

describe("estimateText", () => {
  it("shows an em dash when there is no scalar estimate", () => {
    expect(estimateText({ estimate: null, estimateLabel: null })).toBe(EMPTY)
  })

  it("formats the estimate with its label", () => {
    expect(estimateText({ estimate: 4.08, estimateLabel: "per unit" })).toBe("4.08 per unit")
  })
})
