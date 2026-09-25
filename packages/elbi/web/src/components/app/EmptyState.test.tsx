import { render, screen } from "@testing-library/react"
import { Inbox } from "lucide-react"
import { describe, expect, it } from "vitest"

import { Button } from "@/components/ui/button"

import { EmptyState } from "./EmptyState"

describe("EmptyState", () => {
  it("shows title, description and action", () => {
    render(
      <EmptyState
        icon={Inbox}
        title="No notifications yet"
        description="You'll see run results here."
        action={<Button>Refresh</Button>}
      />,
    )
    expect(screen.getByText("No notifications yet")).toBeInTheDocument()
    expect(screen.getByText("You'll see run results here.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()
  })

  it("renders with only a title", () => {
    render(<EmptyState title="Nothing here" />)
    expect(screen.getByText("Nothing here")).toBeInTheDocument()
  })
})
