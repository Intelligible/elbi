import { describe, expect, it } from "vitest"

import { vegaThemeConfig } from "./chart-theme"

describe("vegaThemeConfig", () => {
  it("themes chrome transparently and draws marks from the 15-color brand palette", () => {
    const cfg = vegaThemeConfig("light") as {
      background: string
      view: { stroke: string }
      range: { category: string[]; ordinal: string[]; ramp: string[] }
      axis: Record<string, unknown>
      legend: Record<string, unknown>
    }
    expect(cfg.background).toBe("transparent")
    expect(cfg.view.stroke).toBe("transparent")
    expect(cfg.range.category).toHaveLength(15)
    expect(cfg.range.ordinal).toHaveLength(15)
    expect(cfg.axis).toHaveProperty("gridColor")
    expect(cfg.legend).toHaveProperty("labelColor")
  })
})
