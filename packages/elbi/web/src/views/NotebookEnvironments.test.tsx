import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { FeedbackProvider } from "@/components/ui/feedback"
import { NotebookEnvironmentsSection } from "./SettingsPage"

/**
 * Rows keyed by identity rather than position.
 *
 * With a positional key, removing a row makes React hand each surviving row the DOM of
 * the one that used to sit at its index. The values still read correctly, because they
 * are controlled -- what moves is everything the DOM owns and React does not: the
 * caret, the selection, an in-flight IME composition. Node identity is what the test
 * can see, and it is the same reuse that carries all of them.
 */
vi.mock("@/lib/notebooks", () => ({
  getBaseEnvironments: async () => [
    { name: "first", deps: ["a"] },
    { name: "second", deps: ["b"] },
    { name: "third", deps: ["c"] },
  ],
  setBaseEnvironments: async () => ({ ok: true }),
}))

describe("NotebookEnvironmentsSection", () => {
  afterEach(() => vi.clearAllMocks())

  it("keeps each row's own DOM state when one before it is removed", async () => {
    render(
      <FeedbackProvider>
        <NotebookEnvironmentsSection />
      </FeedbackProvider>,
    )
    await waitFor(() => expect(screen.getByDisplayValue("first")).toBeInTheDocument())

    const third = screen.getByDisplayValue("third")

    await userEvent.click(screen.getAllByTitle("Remove")[0])

    await waitFor(() => expect(screen.queryByDisplayValue("first")).not.toBeInTheDocument())
    expect(screen.getByDisplayValue("second")).toBeInTheDocument()
    // The very same input still carries "third". Keyed by position, "third" would have
    // moved from index 2 to index 1 and been rendered into the node that held "second".
    expect(screen.getByDisplayValue("third")).toBe(third)
  })
})
