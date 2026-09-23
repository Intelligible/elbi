// Pins the train-model form's behaviour: each test drives a control that changes the
// submitted request, so a mapping mistake shows up as a wrong `trainModel` call rather
// than only as an eslint pass.

import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DatasetColumn, FeatureSource } from "@/lib/chat"

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getFeatureSources: vi.fn(),
  getDatasetColumns: vi.fn(),
  getRegisteredModel: vi.fn(),
  trainModel: vi.fn(),
}))

import { getDatasetColumns, getFeatureSources, trainModel } from "@/lib/chat"
import { TrainModelPage } from "./TrainModelPage"

const SOURCES: FeatureSource[] = [
  { name: "orders", kind: "dataset" },
  { name: "churn_features", kind: "derivation" },
]

const COLUMNS: DatasetColumn[] = [
  { name: "id", numeric: true },
  { name: "amount", numeric: true },
  { name: "country", numeric: false },
]

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/models/train"]}>
      <Routes>
        <Route path="/models/train" element={<TrainModelPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

const lastReq = () => vi.mocked(trainModel).mock.calls.at(-1)?.[0]

// jsdom fires a window blur on pointerdown once an earlier test's cleanup leaves nothing
// focused, which closes a just-opened Radix Select; focusing the trigger first keeps it
// open. A jsdom artefact, not a product bug.
async function pick(
  user: ReturnType<typeof userEvent.setup>,
  label: string | RegExp,
  option: string | RegExp,
) {
  const trigger = await screen.findByRole("combobox", { name: label })
  trigger.focus()
  await user.click(trigger)
  await user.click(screen.getByRole("option", { name: option }))
}

describe("TrainModelPage", () => {
  afterEach(() => vi.clearAllMocks())

  it("picks a grouped dataset source and target, then submits all columns but the target", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "churn_model")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() =>
      expect(lastReq()).toMatchObject({
        name: "churn_model",
        dataset: "orders",
        target: "amount",
        time_budget: 300,
      }),
    )
    expect(lastReq()?.features).toBeUndefined()
  })

  it("unchecking a feature removes it from the submitted features", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)

    await user.click(screen.getByRole("checkbox", { name: /country/ }))
    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() => expect(lastReq()?.features).toEqual(["id"]))
  })

  it("choosing a feature-derivation source switches target/features to free text", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "churn_features")

    await user.type(screen.getByLabelText("Target column", { exact: false }), "churned")
    await user.type(
      screen.getByLabelText("Features (optional)", { exact: false }),
      "recency, frequency",
    )

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() =>
      expect(lastReq()).toMatchObject({
        derivation: "churn_features",
        target: "churned",
        features: ["recency", "frequency"],
      }),
    )
  })

  it("changing the budget unit recomputes the submitted time_budget", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)

    await user.clear(screen.getByLabelText("Budget amount", { exact: false }))
    await user.type(screen.getByLabelText("Budget amount", { exact: false }), "2")
    await pick(user, "Budget unit", "hours")

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() => expect(lastReq()?.time_budget).toBe(7200))
  })

  it("checking ensemble includes it in the request", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)
    await user.click(screen.getByRole("checkbox", { name: /Ensemble/ }))

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() => expect(lastReq()?.ensemble).toBe(true))
  })

  it("selecting a metric sends it in the request", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)
    await pick(user, "Metric", "f1")

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() => expect(lastReq()?.metric).toBe("f1"))
  })

  it("switching the metric back to auto sends no metric at all, never the sentinel", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)
    await pick(user, "Metric", "f1")
    await pick(user, "Metric", "auto")

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() => expect(lastReq()).toBeDefined())
    expect(lastReq()).not.toHaveProperty("metric")
    expect(lastReq()?.metric).not.toBe("auto")
  })

  it("ts_forecast resets the engine to flaml, and submits the time column and horizon", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    vi.mocked(trainModel).mockResolvedValue({
      job: { id: "j1", label: "x", state: "succeeded", progress: "", result: null, error: null },
    })
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Model name", { exact: false }), "m")
    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)

    await pick(user, "Engine", /AutoGluon/)
    await pick(user, "Task", "Time-series forecast")
    // AutoGluon does not support ts_forecast: the page falls back to FLAML, so the
    // request never carries an `engine` override.
    expect(screen.getByRole("combobox", { name: "Engine" })).toHaveTextContent(/FLAML/)

    await pick(user, "Time column", "id")
    await user.type(screen.getByLabelText("Horizon", { exact: false }), "7")

    await user.click(screen.getByRole("button", { name: "Start training" }))

    await waitFor(() =>
      expect(lastReq()).toMatchObject({ task: "ts_forecast", time_col: "id", horizon: 7 }),
    )
    expect(lastReq()?.engine).toBeUndefined()
  })
  it("switching the source clears the target back to its placeholder, and the time column too", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue([
      ...SOURCES,
      { name: "customers", kind: "dataset" },
    ])
    vi.mocked(getDatasetColumns).mockResolvedValue(COLUMNS)
    const errors = vi.spyOn(console, "error")
    const warns = vi.spyOn(console, "warn")
    const user = userEvent.setup()
    renderPage()

    await pick(user, "Data source", "orders")
    await pick(user, "Target column", /amount/)
    await pick(user, "Task", "Time-series forecast")
    await pick(user, "Time column", "id")
    await pick(user, "Data source", "customers")

    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "Target column" })).toHaveTextContent(
        "Choose the target…",
      ),
    )
    expect(screen.getByRole("combobox", { name: "Time column" })).not.toHaveTextContent("id")
    const controlled = (spy: typeof errors) =>
      spy.mock.calls.filter((c) => c.join(" ").includes("controlled"))
    expect(controlled(errors)).toEqual([])
    expect(controlled(warns)).toEqual([])
    errors.mockRestore()
    warns.mockRestore()
  })
})
