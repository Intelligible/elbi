import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { NotebookMarkdown } from "./NotebookMarkdown"

describe("NotebookMarkdown", () => {
  it("leaves currency alone rather than reading it as math", () => {
    render(<NotebookMarkdown>{"$2,692.89 of $4,254.31 over 90 days."}</NotebookMarkdown>)

    expect(screen.getByText(/\$2,692\.89 of \$4,254\.31 over 90 days\./)).toBeInTheDocument()
  })

  it("still renders display math", () => {
    const { container } = render(<NotebookMarkdown>{"$$x^2$$"}</NotebookMarkdown>)

    expect(container.querySelector(".katex")).not.toBeNull()
  })

  it("still renders GitHub-flavored markdown", () => {
    render(<NotebookMarkdown>{"**Run rate:** $47.27/day"}</NotebookMarkdown>)

    expect(screen.getByText("Run rate:").tagName).toBe("STRONG")
  })
})
