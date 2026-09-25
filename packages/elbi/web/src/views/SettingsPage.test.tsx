import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { FeedbackProvider } from "@/components/ui/feedback"

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getDataSources: vi.fn(),
  createDataSource: vi.fn(),
  getBudget: vi.fn(),
  setBudget: vi.fn(),
}))

vi.mock("@/lib/notifications", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/notifications")>()),
  getNotificationPreferences: vi.fn(),
  setNotificationPreferences: vi.fn(),
}))

import { createDataSource, getBudget, getDataSources, setBudget } from "@/lib/chat"
import { getNotificationPreferences, setNotificationPreferences } from "@/lib/notifications"
import { BudgetSection, ConnectionsSection, NotificationsSection } from "./SettingsPage"

const mount = (ui: React.ReactElement) => render(<FeedbackProvider>{ui}</FeedbackProvider>)

// jsdom fires a window blur on pointerdown once an earlier test's cleanup leaves nothing
// focused, which closes a just-opened Radix Select; focusing the trigger first keeps it
// open. A jsdom artefact, not a product bug.
const openSelect = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  const trigger = screen.getByRole("combobox", { name })
  trigger.focus()
  await user.click(trigger)
}

describe("ConnectionsSection", () => {
  afterEach(() => vi.clearAllMocks())

  it("switching the connection kind to sqlite swaps host/port for a file path", async () => {
    vi.mocked(getDataSources).mockResolvedValue([])
    const user = userEvent.setup()
    mount(<ConnectionsSection />)
    await user.click(await screen.findByRole("button", { name: "Add connection" }))
    expect(screen.getByLabelText("Host")).toBeInTheDocument()

    await openSelect(user, "Kind")
    await user.click(screen.getByRole("option", { name: "sqlite" }))

    expect(screen.getByLabelText("Database file path")).toBeInTheDocument()
    expect(screen.queryByLabelText("Host")).toBeNull()
  })

  it("saves the chosen kind when creating a connection", async () => {
    vi.mocked(getDataSources).mockResolvedValue([])
    vi.mocked(createDataSource).mockResolvedValue(true)
    const user = userEvent.setup()
    mount(<ConnectionsSection />)
    await user.click(await screen.findByRole("button", { name: "Add connection" }))
    await user.type(screen.getByLabelText("Name"), "warehouse")

    await openSelect(user, "Kind")
    await user.click(screen.getByRole("option", { name: "mysql" }))
    await user.type(screen.getByLabelText("Host"), "db.local")

    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(createDataSource).toHaveBeenCalledWith(expect.objectContaining({ kind: "mysql" })),
    )
  })
})

describe("BudgetSection", () => {
  afterEach(() => vi.clearAllMocks())

  it("saves the selected window", async () => {
    vi.mocked(getBudget).mockResolvedValue({ maxBudget: 0, window: "30d", spend: 0 })
    vi.mocked(setBudget).mockResolvedValue(true)
    const user = userEvent.setup()
    mount(<BudgetSection />)
    await screen.findByRole("button", { name: "Save" })

    await openSelect(user, "Window")
    await user.click(screen.getByRole("option", { name: "7d" }))
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(setBudget).toHaveBeenCalledWith(0, "7d"))
  })
})

describe("NotificationsSection", () => {
  afterEach(() => vi.clearAllMocks())

  it("toggles and saves an in-app preference", async () => {
    vi.mocked(getNotificationPreferences).mockResolvedValue({
      emailAvailable: true,
      prefs: [{ eventType: "run.failed", inApp: false, email: false }],
    })
    vi.mocked(setNotificationPreferences).mockResolvedValue({
      emailAvailable: true,
      prefs: [{ eventType: "run.failed", inApp: true, email: false }],
    })
    const user = userEvent.setup()
    mount(<NotificationsSection />)

    const checkbox = await screen.findByRole("checkbox", {
      name: "In-app notifications for Run failed",
    })
    await user.click(checkbox)
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(setNotificationPreferences).toHaveBeenCalledWith([
        { eventType: "run.failed", inApp: true, email: false },
      ]),
    )
  })
})
