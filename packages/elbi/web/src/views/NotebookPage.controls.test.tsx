// Pins the notebook page's controls: each test drives one that changes a request (a cell
// action, the rename, the environment, the schedule, the train dialog, a kernel input),
// so a mapping mistake shows up as a wrong API call rather than only as an eslint pass.

import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { Cell, NotebookView, RunEvent } from "@/lib/notebooks"

vi.mock("@/lib/notebooks", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/notebooks")>()),
  getNotebook: vi.fn(),
  getNotebookVariables: vi.fn(async () => []),
  updateNotebook: vi.fn(async () => ({ ok: true })),
  relockNotebook: vi.fn(async () => ({ ok: true })),
  updateCell: vi.fn(async () => ({ ok: true })),
  addCell: vi.fn(async () => ({ ok: true })),
  deleteCell: vi.fn(async () => ({ ok: true })),
  reorderCells: vi.fn(async () => ({ ok: true })),
  runNotebook: vi.fn(),
  sendInput: vi.fn(async () => ({ ok: true })),
}))

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getFeatureSources: vi.fn(),
  getDatasetColumns: vi.fn(),
  trainModel: vi.fn(),
}))

import { getDatasetColumns, getFeatureSources, trainModel } from "@/lib/chat"
import {
  addCell,
  deleteCell,
  getNotebook,
  relockNotebook,
  reorderCells,
  runNotebook,
  sendInput,
  updateCell,
  updateNotebook,
} from "@/lib/notebooks"
import { NotebookPage } from "./NotebookPage"

const cell = (id: string, extra: Partial<Cell> = {}): Cell => ({
  id,
  cell_type: "code",
  source: `x_${id} = 1`,
  metadata: {},
  outputs: [],
  execution_count: null,
  ...extra,
})

const VIEW: NotebookView = {
  id: "nb1",
  name: "Q3 report",
  folder_id: null,
  folder_name: null,
  deps: ["pandas"],
  metadata: {},
  schedule: null,
  kernel_status: "idle",
  cells: [cell("c1"), cell("c2")],
  graph: { cells: {}, conflicts: {}, cycle: [] },
  environment: {
    base_env: "ml",
    base_environments: ["ml", "geo"],
    lock: null,
    locked: false,
  },
}

function renderPage() {
  return render(
    <TooltipProvider>
      <MemoryRouter initialEntries={["/notebooks/nb1"]}>
        <Routes>
          <Route path="/notebooks/:name" element={<NotebookPage />} />
          <Route path="/models" element={<div>models page</div>} />
        </Routes>
      </MemoryRouter>
    </TooltipProvider>,
  )
}

// Radix menus and selects close on a window blur, which jsdom fires at pointerdown once an
// earlier test's cleanup leaves nothing focused; focusing the trigger first keeps them
// open. A jsdom artefact, not a product bug.
async function press(user: ReturnType<typeof userEvent.setup>, el: HTMLElement) {
  el.focus()
  await user.click(el)
}

async function openMenuItem(user: ReturnType<typeof userEvent.setup>, item: RegExp) {
  await press(user, screen.getByRole("button", { name: "More actions" }))
  await user.click(await screen.findByRole("menuitem", { name: item }))
  return screen.findByRole("dialog")
}

// A native select takes selectOptions; a Radix Select is opened and its option clicked.
async function choose(
  user: ReturnType<typeof userEvent.setup>,
  combobox: HTMLElement,
  option: string,
) {
  if (combobox instanceof HTMLSelectElement) {
    await user.selectOptions(combobox, option)
    return
  }
  await press(user, combobox)
  await user.click(await screen.findByRole("option", { name: option }))
}

describe("NotebookPage controls", () => {
  beforeEach(() => {
    vi.mocked(getNotebook).mockResolvedValue(VIEW)
    vi.mocked(runNotebook).mockReturnValue({ done: Promise.resolve(), abort: () => {} })
  })
  afterEach(() => vi.clearAllMocks())

  it("renames the notebook from its title on blur", async () => {
    const user = userEvent.setup()
    renderPage()
    const title = await screen.findByDisplayValue("Q3 report")
    await user.clear(title)
    await user.type(title, "Q4 report")
    await user.tab()
    await waitFor(() => expect(updateNotebook).toHaveBeenCalledWith("nb1", { name: "Q4 report" }))
  })

  it("runs a cell from its gutter", async () => {
    const user = userEvent.setup()
    renderPage()
    const [run] = await screen.findAllByRole("button", { name: "Run cell (Shift+Enter)" })
    await user.click(run)
    await waitFor(() =>
      expect(runNotebook).toHaveBeenCalledWith("nb1", { cells: ["c1"] }, expect.any(Function)),
    )
  })

  it("adds, moves, retypes and deletes cells from a cell's actions", async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findAllByRole("button", { name: "Move down" })

    await user.click(screen.getAllByRole("button", { name: "+ code" })[0])
    await waitFor(() =>
      expect(addCell).toHaveBeenCalledWith("nb1", { after: "c1", cell_type: "code" }),
    )
    await user.click(screen.getAllByRole("button", { name: "+ text" })[1])
    await waitFor(() =>
      expect(addCell).toHaveBeenCalledWith("nb1", { after: "c2", cell_type: "markdown" }),
    )
    await user.click(screen.getAllByRole("button", { name: "Move down" })[0])
    await waitFor(() => expect(reorderCells).toHaveBeenCalledWith("nb1", ["c2", "c1"]))
    await user.click(screen.getAllByRole("button", { name: "Make markdown" })[1])
    await waitFor(() =>
      expect(updateCell).toHaveBeenCalledWith("nb1", "c2", { cell_type: "markdown" }),
    )
    await user.click(screen.getAllByRole("button", { name: "Mark as parameters cell" })[0])
    await waitFor(() =>
      expect(updateCell).toHaveBeenCalledWith("nb1", "c1", {
        metadata: { tags: ["parameters"] },
      }),
    )
    await user.click(screen.getAllByRole("button", { name: "Delete cell" })[1])
    await waitFor(() => expect(deleteCell).toHaveBeenCalledWith("nb1", "c2"))
  })

  it("answers a cell's input() prompt", async () => {
    const user = userEvent.setup()
    vi.mocked(runNotebook).mockImplementation((_id, _body, onEvent) => {
      onEvent({ event: "input_request", cell: "c1", prompt: "Name?", password: false } as RunEvent)
      return { done: new Promise(() => {}), abort: () => {} }
    })
    renderPage()
    const [run] = await screen.findAllByRole("button", { name: "Run cell (Shift+Enter)" })
    await user.click(run)
    const input = await screen.findByRole("textbox", { name: "Cell input" })
    expect(screen.getByText("Name?")).toBeInTheDocument()
    await user.type(input, "Ada")
    await user.click(screen.getByRole("button", { name: "Submit" }))
    await waitFor(() => expect(sendInput).toHaveBeenCalledWith("nb1", "Ada"))
  })

  it("masks a password input() prompt", async () => {
    const user = userEvent.setup()
    vi.mocked(runNotebook).mockImplementation((_id, _body, onEvent) => {
      onEvent({ event: "input_request", cell: "c1", prompt: "Token:", password: true } as RunEvent)
      return { done: new Promise(() => {}), abort: () => {} }
    })
    const { container } = renderPage()
    const [run] = await screen.findAllByRole("button", { name: "Run cell (Shift+Enter)" })
    await user.click(run)
    await screen.findByText("Token:")
    // A password field has no textbox role, so it is found by its label attribute.
    const field = container.querySelector('input[aria-label="Cell input"]')
    expect(field).toHaveAttribute("type", "password")
    await user.type(field as HTMLElement, "s3cret")
    await user.click(screen.getByRole("button", { name: "Submit" }))
    await waitFor(() => expect(sendInput).toHaveBeenCalledWith("nb1", "s3cret"))
  })

  it("saves the environment's packages and base environment", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    renderPage()
    await screen.findByDisplayValue("Q3 report")
    let dialog = await openMenuItem(user, /environment/i)

    await choose(user, within(dialog).getByRole("combobox", { name: /base environment/i }), "geo")
    const deps = within(dialog).getByRole("textbox")
    await user.clear(deps)
    await user.type(deps, "polars{Enter}{Enter} numpy==2.0 ")
    await user.click(within(dialog).getByRole("button", { name: "Save & lock" }))
    await waitFor(() =>
      expect(updateNotebook).toHaveBeenCalledWith("nb1", {
        deps: ["polars", "numpy==2.0"],
        base_env: "geo",
      }),
    )

    // "None" is sent as null: the choice that clears the base environment.
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    dialog = await openMenuItem(user, /environment/i)
    await choose(user, within(dialog).getByRole("combobox", { name: /base environment/i }), "None")
    await user.click(within(dialog).getByRole("button", { name: "Save & lock" }))
    await waitFor(() =>
      expect(updateNotebook).toHaveBeenLastCalledWith("nb1", { deps: ["pandas"], base_env: null }),
    )
  })

  it("re-locks the environment", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    renderPage()
    await screen.findByDisplayValue("Q3 report")
    const dialog = await openMenuItem(user, /environment/i)
    await user.click(within(dialog).getByRole("button", { name: /re-lock/i }))
    await waitFor(() => expect(relockNotebook).toHaveBeenCalledWith("nb1"))
  })

  it("saves an interval schedule", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    renderPage()
    await screen.findByDisplayValue("Q3 report")
    const dialog = await openMenuItem(user, /schedule reruns/i)

    await user.click(within(dialog).getByRole("checkbox", { name: /run this notebook/i }))
    const hours = within(dialog).getByRole("spinbutton")
    await user.clear(hours)
    await user.type(hours, "6")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(updateNotebook).toHaveBeenCalledWith("nb1", {
        schedule: { enabled: true, mode: "interval", interval_hours: 6, dataset: undefined },
      }),
    )
  })

  it("saves a dataset-change schedule", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    renderPage()
    await screen.findByDisplayValue("Q3 report")
    const dialog = await openMenuItem(user, /schedule reruns/i)

    await user.click(within(dialog).getByRole("button", { name: "When a dataset changes" }))
    await user.type(within(dialog).getByPlaceholderText("dataset name"), "orders")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(updateNotebook).toHaveBeenCalledWith("nb1", {
        schedule: { enabled: false, mode: "on_data_change", interval_hours: 24, dataset: "orders" },
      }),
    )
  })

  it("trains a model on a dataset column picked in the train dialog", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    vi.mocked(getFeatureSources).mockResolvedValue([
      { name: "orders", kind: "dataset" },
      { name: "churn_features", kind: "derivation" },
    ])
    vi.mocked(getDatasetColumns).mockResolvedValue([
      { name: "amount", numeric: true },
      { name: "country", numeric: false },
    ])
    vi.mocked(trainModel).mockResolvedValue({ id: "job1" } as never)
    renderPage()
    await screen.findByDisplayValue("Q3 report")
    const dialog = await openMenuItem(user, /train model/i)

    await user.type(within(dialog).getByRole("textbox", { name: "Model name" }), "churn")
    await waitFor(() => expect(getFeatureSources).toHaveBeenCalled())
    await choose(user, within(dialog).getByRole("combobox", { name: "Training source" }), "orders")
    await waitFor(() => expect(getDatasetColumns).toHaveBeenCalledWith("orders"))
    await choose(
      user,
      await within(dialog).findByRole("combobox", { name: "Target column" }),
      "amount",
    )
    await choose(user, within(dialog).getByRole("combobox", { name: "Task" }), "Regression")
    await choose(
      user,
      within(dialog).getByRole("combobox", { name: "Engine" }),
      "AutoGluon (accuracy)",
    )
    const budget = within(dialog).getByRole("spinbutton", { name: "Budget (min)" })
    await user.clear(budget)
    await user.type(budget, "3")
    await user.click(within(dialog).getByRole("button", { name: "Train & register" }))

    await waitFor(() =>
      expect(trainModel).toHaveBeenCalledWith({
        name: "churn",
        target: "amount",
        task: "regression",
        engine: "autogluon",
        time_budget: 180,
        dataset: "orders",
      }),
    )
    expect(await screen.findByText("models page")).toBeInTheDocument()
  })

  it("trains on a derivation with a typed target", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    vi.mocked(getFeatureSources).mockResolvedValue([{ name: "churn_features", kind: "derivation" }])
    vi.mocked(trainModel).mockResolvedValue({ error: "no rows" })
    renderPage()
    await screen.findByDisplayValue("Q3 report")
    const dialog = await openMenuItem(user, /train model/i)

    await user.type(within(dialog).getByRole("textbox", { name: "Model name" }), "churn")
    await waitFor(() => expect(getFeatureSources).toHaveBeenCalled())
    await choose(
      user,
      within(dialog).getByRole("combobox", { name: "Training source" }),
      "churn_features",
    )
    await user.type(within(dialog).getByRole("textbox", { name: "Target column" }), "label")
    await user.click(within(dialog).getByRole("button", { name: "Train & register" }))

    await waitFor(() =>
      expect(trainModel).toHaveBeenCalledWith({
        name: "churn",
        target: "label",
        task: "auto",
        engine: "flaml",
        time_budget: 60,
        derivation: "churn_features",
      }),
    )
    expect(await within(dialog).findByText("no rows")).toBeInTheDocument()
  })
})
