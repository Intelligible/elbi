// Pins the metrics page's controls: the catalog filter and selection, the group-by chips,
// the filter rows, the chart toggle and the ask flow each change a request or what is on
// screen, so a mapping mistake shows up as a wrong API call rather than only as an eslint pass.

import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { Metric } from "@/lib/metrics"

vi.mock("@/lib/metrics", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/metrics")>()),
  listMetrics: vi.fn(),
  metricsOverview: vi.fn(),
  queryMetric: vi.fn(),
  metricHistory: vi.fn(),
  askMetric: vi.fn(),
}))
vi.mock("@/lib/lineage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/lineage")>()),
  getSubgraph: vi.fn(),
}))
// Vega needs a real layout engine; the page only hands it a spec.
vi.mock("@/components/viz/VizView", () => ({ VizView: () => <div data-testid="viz" /> }))

import { getSubgraph } from "@/lib/lineage"
import { askMetric, listMetrics, metricHistory, metricsOverview, queryMetric } from "@/lib/metrics"
import { MetricsPage } from "./MetricsPage"

const METRICS: Metric[] = [
  {
    name: "revenue",
    label: "Revenue",
    type: "simple",
    source: "orders_clean",
    measure: { agg: "sum", column: "amount" },
    dimensions: ["region"],
    timeDimension: { column: "day", grain: "month" },
    sourceCertified: true,
  },
  {
    name: "churn",
    label: "Churn",
    type: "simple",
    source: "accounts",
    measure: { agg: "count" },
    sourceCertified: false,
  },
]

function renderPage() {
  return render(
    <TooltipProvider>
      <MetricsPage />
    </TooltipProvider>,
  )
}

const catalogItem = (name: RegExp) => screen.getByRole("button", { name })

async function openRevenue(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: /^Revenue/ }))
  await screen.findByRole("heading", { name: "Revenue" })
}

describe("MetricsPage", () => {
  beforeEach(() => {
    vi.mocked(listMetrics).mockResolvedValue(METRICS)
    vi.mocked(metricsOverview).mockResolvedValue([])
    vi.mocked(queryMetric).mockResolvedValue({
      columns: ["region", "revenue"],
      rows: [
        { region: "west", revenue: 10 },
        { region: "east", revenue: 7 },
      ],
    })
    vi.mocked(metricHistory).mockResolvedValue([])
    vi.mocked(getSubgraph).mockResolvedValue({ nodes: [], edges: [] })
  })
  afterEach(() => vi.clearAllMocks())

  it("filters the catalog to certified metrics", async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("button", { name: /^Churn/ })

    await user.click(screen.getByRole("checkbox", { name: "Certified only" }))
    expect(screen.queryByRole("button", { name: /^Churn/ })).not.toBeInTheDocument()
    expect(catalogItem(/^Revenue/)).toBeInTheDocument()

    await user.click(screen.getByRole("checkbox", { name: "Certified only" }))
    expect(catalogItem(/^Churn/)).toBeInTheDocument()
  })

  it("groups by a dimension chip and sends it with the run", async () => {
    const user = userEvent.setup()
    renderPage()
    await openRevenue(user)

    expect(catalogItem(/^Revenue/)).toHaveAttribute("aria-current", "true")
    expect(screen.getByRole("button", { name: "region" })).toHaveAttribute("aria-pressed", "false")
    await user.click(screen.getByRole("button", { name: "region" }))
    expect(screen.getByRole("button", { name: "region" })).toHaveAttribute("aria-pressed", "true")
    await user.click(screen.getByRole("button", { name: /^Run/ }))
    await waitFor(() =>
      expect(queryMetric).toHaveBeenLastCalledWith("revenue", {
        groupBy: ["region"],
        grain: "month",
        filters: [],
      }),
    )

    await user.click(screen.getByRole("button", { name: "region" }))
    await user.click(screen.getByRole("button", { name: /^Run/ }))
    await waitFor(() =>
      expect(queryMetric).toHaveBeenLastCalledWith("revenue", {
        groupBy: [],
        grain: "month",
        filters: [],
      }),
    )
  })

  it("adds and removes a filter row", async () => {
    const user = userEvent.setup()
    renderPage()
    await openRevenue(user)

    await user.click(screen.getByRole("button", { name: /^Add/ }))
    await user.type(screen.getByPlaceholderText("value"), "west")
    await user.click(screen.getByRole("button", { name: /^Run/ }))
    await waitFor(() =>
      expect(queryMetric).toHaveBeenLastCalledWith("revenue", {
        groupBy: [],
        grain: "month",
        filters: [{ column: "region", op: "eq", value: "west" }],
      }),
    )

    await user.click(screen.getByRole("button", { name: /^Remove filter/ }))
    expect(screen.queryByPlaceholderText("value")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /^Run/ }))
    await waitFor(() =>
      expect(queryMetric).toHaveBeenLastCalledWith("revenue", {
        groupBy: [],
        grain: "month",
        filters: [],
      }),
    )
  })

  it("switches the breakdown between a chart and a table", async () => {
    const user = userEvent.setup()
    renderPage()
    await openRevenue(user)
    await user.click(screen.getByRole("button", { name: "region" }))
    await user.click(screen.getByRole("button", { name: /^Run/ }))
    await screen.findByText("west")
    // The overview's trend is a chart too; the breakdown adds one.
    const charts = () => screen.queryAllByTestId("viz").length
    const withChart = charts()

    await user.click(screen.getByRole("button", { name: /^Table/ }))
    expect(charts()).toBe(withChart - 1)
    expect(screen.getByRole("button", { name: /^Table/ })).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByRole("button", { name: /^Bar/ })).toHaveAttribute("aria-pressed", "false")
    expect(screen.getByText("west")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /^Line/ }))
    expect(charts()).toBe(withChart)
  })

  it("shows a request error and dismisses it", async () => {
    vi.mocked(queryMetric).mockRejectedValue(new Error("source is gone"))
    const user = userEvent.setup()
    renderPage()
    await openRevenue(user)

    expect(await screen.findByText("source is gone")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /^Dismiss/ }))
    expect(screen.queryByText("source is gone")).not.toBeInTheDocument()
  })

  it("opens the metric an ask resolves to", async () => {
    vi.mocked(askMetric).mockResolvedValue({
      columns: ["revenue"],
      rows: [{ revenue: 17 }],
      resolvedQuery: { metricName: "revenue", groupBy: [], grain: null, filters: [] },
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("button", { name: /^Revenue/ })

    await user.click(screen.getByRole("button", { name: /Ask/ }))
    const dialog = await screen.findByRole("dialog", { name: "Ask a metric" })
    await user.type(within(dialog).getByRole("textbox"), "total revenue{Enter}")
    await waitFor(() => expect(askMetric).toHaveBeenCalledWith("total revenue"))

    await user.click(await within(dialog).findByRole("button", { name: "revenue" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(screen.getByRole("heading", { name: "Revenue" })).toBeInTheDocument()
  })
})
