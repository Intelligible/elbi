import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { DataModePill } from "./DataModePill"

describe("DataModePill", () => {
  it("distinguishes work done in the warehouse from rows pulled into the kernel", () => {
    const { unmount } = render(<DataModePill mode="pushed_down" />)
    expect(screen.getByText("in warehouse")).toBeTruthy()
    // The explanation is what makes the pill act on: a label alone says nothing about
    // why one of these scales and the other does not.
    expect(screen.getByTitle(/scale with the data/i)).toBeTruthy()
    unmount()

    render(<DataModePill mode="materialised" />)
    expect(screen.getByText("in kernel")).toBeTruthy()
    expect(screen.getByTitle(/memory limit is the bound/i)).toBeTruthy()
  })

  it("renders nothing for a mode it does not know", () => {
    // A newer kernel reporting something else should not draw a mystery badge.
    const { container } = render(<DataModePill mode="teleported" />)
    expect(container.firstChild).toBeNull()
  })
})
