import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { CellOutputs } from "./CellOutput"

const ESC = String.fromCharCode(27)

/**
 * A traceback carries more than colour. A cell that printed a progress bar leaves
 * cursor-move and erase-line sequences in the stream, and rich/pytest emit OSC
 * hyperlinks -- none of which an SGR-only matcher removes, so the raw bytes reached
 * the page.
 */
describe("CellOutputs tracebacks", () => {
  const traceback = (lines: string[]) =>
    render(<CellOutputs outputs={[{ output_type: "error", traceback: lines } as never]} />)

  it("strips every escape sequence, not only the colour codes", () => {
    const { container } = traceback([
      `${ESC}[31mValueError${ESC}[0m: bad`,
      `progress${ESC}[2A${ESC}[Kdone`,
      `${ESC}]8;;http://docs${ESC}${String.fromCharCode(92)}see docs${ESC}]8;;${ESC}${String.fromCharCode(92)}`,
    ])
    expect(container.textContent).not.toContain(ESC)
    expect(screen.getByText(/ValueError: bad/)).toBeInTheDocument()
    expect(container.textContent).toContain("progressdone")
    expect(container.textContent).toContain("see docs")
  })
})
