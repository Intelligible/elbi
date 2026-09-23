// Pins the model page's form behaviour: each test drives a control that changes a request
// (serving, batch scoring, the retrain policy), so a mapping mistake shows up as a wrong
// API call rather than only as an eslint pass.

import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { FeatureSource, ModelVersion, RegisteredModelDetail, RetrainPolicy } from "@/lib/chat"

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getRegisteredModel: vi.fn(),
  getTrainingJobs: vi.fn(),
  getModelSchema: vi.fn(),
  getFeatureSources: vi.fn(),
  getDatasetColumns: vi.fn(),
  getDriftHistory: vi.fn(),
  getInferenceLog: vi.fn(),
  getRetrainPolicy: vi.fn(),
  putRetrainPolicy: vi.fn(),
  invokeModel: vi.fn(),
  batchScoreModel: vi.fn(),
}))

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  copyText: vi.fn(),
}))

import {
  batchScoreModel,
  getDatasetColumns,
  getDriftHistory,
  getFeatureSources,
  getInferenceLog,
  getModelSchema,
  getRegisteredModel,
  getRetrainPolicy,
  getTrainingJobs,
  invokeModel,
  putRetrainPolicy,
} from "@/lib/chat"
import { copyText } from "@/lib/utils"
import { ModelDetailPage } from "./ModelDetailPage"

const version = (v: number, extra: Partial<ModelVersion> = {}): ModelVersion => ({
  version: v,
  runId: `run${v}`,
  experimentId: "1",
  createdAtMs: 1_700_000_000_000 + v,
  aliases: [],
  description: "",
  metrics: {},
  params: {},
  tags: {},
  verdict: null,
  verdictDetail: null,
  ...extra,
})

const DETAIL: RegisteredModelDetail = {
  name: "churn",
  championVersion: 2,
  versions: [
    version(2, { aliases: ["champion"], params: { dataset: "orders", target: "amount" } }),
    version(1),
  ],
}

const SOURCES: FeatureSource[] = [
  { name: "orders", kind: "dataset" },
  { name: "churn_features", kind: "derivation" },
]

function renderPage() {
  return render(
    <TooltipProvider>
      <MemoryRouter initialEntries={["/models/churn"]}>
        <Routes>
          <Route path="/models/:name" element={<ModelDetailPage />} />
        </Routes>
      </MemoryRouter>
    </TooltipProvider>,
  )
}

const section = async (title: string) =>
  (await screen.findByRole("heading", { name: title })).closest("section") as HTMLElement

// jsdom fires a window blur on pointerdown once an earlier test's cleanup leaves nothing
// focused, which closes a just-opened Radix Select; focusing the trigger first keeps it
// open. A jsdom artefact, not a product bug.
async function choose(
  user: ReturnType<typeof userEvent.setup>,
  combobox: HTMLElement,
  option: string | RegExp,
) {
  combobox.focus()
  await user.click(combobox)
  await user.click(screen.getByRole("option", { name: option }))
}

// The text a select shows for its current value.
const shown = (combobox: HTMLElement) =>
  combobox instanceof HTMLSelectElement
    ? (combobox.selectedOptions[0]?.textContent ?? "")
    : (combobox.textContent ?? "")

describe("ModelDetailPage", () => {
  beforeEach(() => {
    vi.mocked(getRegisteredModel).mockResolvedValue(DETAIL)
    vi.mocked(getTrainingJobs).mockResolvedValue([])
    vi.mocked(getModelSchema).mockResolvedValue(null)
    vi.mocked(getFeatureSources).mockResolvedValue(SOURCES)
    vi.mocked(getDatasetColumns).mockResolvedValue([
      { name: "amount", numeric: true },
      { name: "country", numeric: false },
    ])
    vi.mocked(getDriftHistory).mockResolvedValue([])
    vi.mocked(getInferenceLog).mockResolvedValue([])
    vi.mocked(getRetrainPolicy).mockResolvedValue({ configured: false })
    vi.mocked(putRetrainPolicy).mockResolvedValue(true)
    vi.mocked(invokeModel).mockResolvedValue({ predictions: [1] })
    vi.mocked(batchScoreModel).mockResolvedValue({ error: "stop" })
  })
  afterEach(() => vi.clearAllMocks())

  it("sends the query to the version picked in the query pane", async () => {
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Query endpoint")

    await choose(user, within(pane).getByRole("combobox"), "v1")
    await waitFor(() => expect(getModelSchema).toHaveBeenLastCalledWith("churn", 1))
    await user.click(within(pane).getByRole("textbox"))
    await user.paste('{"dataframe_records": [{"x": 1}]}')
    await user.click(within(pane).getByRole("button", { name: "Send request" }))

    await waitFor(() =>
      expect(invokeModel).toHaveBeenCalledWith("churn", 1, { dataframe_records: [{ x: 1 }] }),
    )
  })

  it("copies the curl command", async () => {
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Query endpoint")

    await user.click(within(pane).getByRole("button", { name: "Copy" }))

    expect(copyText).toHaveBeenCalledWith(expect.stringContaining("/api/serving/churn/invocations"))
  })

  it("batch-scores a picked derivation against a picked version", async () => {
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Batch score")

    await choose(
      user,
      within(pane).getByRole("combobox", { name: /^Data source/ }),
      "churn_features",
    )
    await choose(user, within(pane).getByRole("combobox", { name: /^Version/ }), "v1")
    await user.click(within(pane).getByRole("button", { name: "Run batch scoring" }))

    await waitFor(() =>
      expect(batchScoreModel).toHaveBeenCalledWith("churn", {
        derivation: "churn_features",
        version: "1",
      }),
    )
  })

  it("batch-scores the served version by sending no version at all", async () => {
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Batch score")

    await choose(user, within(pane).getByRole("combobox", { name: /^Data source/ }), "orders")
    await choose(user, within(pane).getByRole("combobox", { name: /^Version/ }), "v1")
    await choose(user, within(pane).getByRole("combobox", { name: /^Version/ }), "served (v2)")
    await user.click(within(pane).getByRole("button", { name: "Run batch scoring" }))

    await waitFor(() => expect(batchScoreModel).toHaveBeenCalledTimes(1))
    expect(vi.mocked(batchScoreModel).mock.calls[0]).toEqual(["churn", { dataset: "orders" }])
  })

  it("configures a retrain policy from the edited form", async () => {
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Auto-retrain")

    // Prefilled from the served version's lineage: dataset "orders", target "amount".
    await waitFor(() =>
      expect(shown(within(pane).getByRole("combobox", { name: /^Target column/ }))).toBe("amount"),
    )
    await choose(user, within(pane).getByRole("combobox", { name: /^Target column/ }), "country")
    await choose(user, within(pane).getByRole("combobox", { name: /^Trigger/ }), "On an interval")
    const hours = within(pane).getByRole("spinbutton", { name: /^Every \(hours\)/ })
    await user.clear(hours)
    await user.type(hours, "12")
    const amount = within(pane).getByRole("spinbutton", { name: "Budget amount" })
    await user.clear(amount)
    await user.type(amount, "2")
    await choose(user, within(pane).getByRole("combobox", { name: "Budget unit" }), "hours")
    await user.click(within(pane).getByRole("checkbox", { name: "Enabled" }))
    await user.click(within(pane).getByRole("button", { name: "Configure" }))

    await waitFor(() =>
      expect(putRetrainPolicy).toHaveBeenCalledWith("churn", {
        dataset: "orders",
        target: "country",
        time_budget: 7200,
        mode: "interval",
        interval_hours: 12,
        enabled: false,
      }),
    )
  })

  it("takes a free-text target for a derivation source", async () => {
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Auto-retrain")

    await choose(
      user,
      within(pane).getByRole("combobox", { name: /^Data source/ }),
      "churn_features",
    )
    await user.type(within(pane).getByRole("textbox", { name: "Target column" }), "label")
    await user.click(within(pane).getByRole("button", { name: "Configure" }))

    await waitFor(() =>
      expect(putRetrainPolicy).toHaveBeenCalledWith(
        "churn",
        expect.objectContaining({ derivation: "churn_features", target: "label" }),
      ),
    )
  })

  it("keeps a stored source that is missing from the fetched list", async () => {
    const stored: RetrainPolicy = {
      configured: true,
      dataset: "archived",
      sourceKind: "dataset",
      target: "amount",
      features: null,
      task: "classification",
      timeBudget: 300,
      metric: null,
      ensemble: false,
      mode: "on_data_change",
      intervalHours: 0,
      enabled: true,
      lastRunAt: null,
    }
    vi.mocked(getRetrainPolicy).mockResolvedValue(stored)
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Auto-retrain")

    await waitFor(() =>
      expect(shown(within(pane).getByRole("combobox", { name: /^Data source/ }))).toBe("archived"),
    )
    await user.click(within(pane).getByRole("button", { name: "Save policy" }))

    await waitFor(() =>
      expect(putRetrainPolicy).toHaveBeenCalledWith(
        "churn",
        expect.objectContaining({ dataset: "archived", target: "amount", time_budget: 300 }),
      ),
    )
  })
  it("switching the retrain source clears the target back to its placeholder", async () => {
    vi.mocked(getFeatureSources).mockResolvedValue([
      ...SOURCES,
      { name: "customers", kind: "dataset" },
    ])
    const errors = vi.spyOn(console, "error")
    const warns = vi.spyOn(console, "warn")
    const user = userEvent.setup()
    renderPage()
    const pane = await section("Auto-retrain")
    const target = () => within(pane).getByRole("combobox", { name: /^Target column/ })

    await waitFor(() => expect(shown(target())).toBe("amount"))
    await choose(user, target(), "country")
    await choose(user, within(pane).getByRole("combobox", { name: /^Data source/ }), "customers")

    await waitFor(() => expect(shown(target())).toBe("Choose the target…"))
    const controlled = (spy: typeof errors) =>
      spy.mock.calls.filter((c) => c.join(" ").includes("controlled"))
    expect(controlled(errors)).toEqual([])
    expect(controlled(warns)).toEqual([])
    errors.mockRestore()
    warns.mockRestore()
  })
})
