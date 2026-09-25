// Pins the page header's back control: an in-page onBack runs its handler and takes
// precedence over a route.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import { SceneHeader } from "./Scene"

describe("SceneHeader back control", () => {
  it("runs onBack instead of following backTo", async () => {
    const user = userEvent.setup()
    const onBack = vi.fn()
    render(
      <MemoryRouter>
        <SceneHeader title="New source" onBack={onBack} backTo="/warehouse" backLabel="Catalog" />
      </MemoryRouter>,
    )
    expect(screen.queryByRole("link", { name: "Catalog" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Catalog" }))
    expect(onBack).toHaveBeenCalledOnce()
  })

  it("renders a back link for backTo alone", () => {
    render(
      <MemoryRouter>
        <SceneHeader title="Model" backTo="/models" />
      </MemoryRouter>,
    )
    expect(screen.getByRole("link", { name: "Back" })).toHaveAttribute("href", "/models")
  })
})
