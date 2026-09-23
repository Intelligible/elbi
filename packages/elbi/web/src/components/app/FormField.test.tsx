import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { Input } from "@/components/ui/input"

import { FormField } from "./FormField"

describe("FormField", () => {
  it("names its control by the label", () => {
    render(
      <FormField label="Dataset">
        <Input />
      </FormField>,
    )
    expect(screen.getByRole("textbox", { name: "Dataset" })).toBeInTheDocument()
  })

  it("describes the control with its hint and error, and marks it invalid", () => {
    render(
      <FormField label="Rows" hint="At most 10k" error="Must be a number">
        <Input />
      </FormField>,
    )
    const input = screen.getByRole("textbox", { name: "Rows" })
    expect(input).toHaveAccessibleDescription("At most 10k Must be a number")
    expect(input).toHaveAttribute("aria-invalid", "true")
  })

  it("keeps an id the caller already set", () => {
    render(
      <FormField label="Name">
        <Input id="given" />
      </FormField>,
    )
    expect(screen.getByRole("textbox", { name: "Name" })).toHaveAttribute("id", "given")
  })
})
