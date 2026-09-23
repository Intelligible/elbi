import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import { FormControl, FormField } from "./FormField"

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

  it("merges the child's own aria-describedby with the hint, instead of overwriting it", () => {
    render(
      <FormField label="Rows" hint="At most 10k">
        <Input aria-describedby="other" />
      </FormField>,
    )
    const input = screen.getByRole("textbox", { name: "Rows" })
    const hintEl = screen.getByText("At most 10k")
    const ids = input.getAttribute("aria-describedby")?.split(" ")
    expect(ids).toContain("other")
    expect(ids).toContain(hintEl.id)
  })

  it("keeps the child's own aria-invalid when there is no error", () => {
    render(
      <FormField label="Rows">
        <Input aria-invalid />
      </FormField>,
    )
    expect(screen.getByRole("textbox", { name: "Rows" })).toHaveAttribute("aria-invalid", "true")
  })

  it("renders no aria-describedby when there is nothing to describe it with", () => {
    render(
      <FormField label="Rows">
        <Input />
      </FormField>,
    )
    expect(screen.getByRole("textbox", { name: "Rows" })).not.toHaveAttribute("aria-describedby")
  })

  it("names a Select trigger via FormControl, and describes it with the hint", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {})
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})

    render(
      <FormField label="Model" hint="Pick one">
        <Select value="a" onValueChange={() => {}}>
          <FormControl>
            <SelectTrigger>
              <SelectValue placeholder="Choose…" />
            </SelectTrigger>
          </FormControl>
          <SelectContent>
            <SelectItem value="a">A</SelectItem>
          </SelectContent>
        </Select>
      </FormField>,
    )

    const trigger = screen.getByRole("combobox", { name: "Model" })
    expect(trigger).toHaveAccessibleDescription("Pick one")
    expect(error).not.toHaveBeenCalled()
    expect(warn).not.toHaveBeenCalled()

    error.mockRestore()
    warn.mockRestore()
  })
})
