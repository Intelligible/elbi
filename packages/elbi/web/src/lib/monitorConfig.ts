// What a monitor's `config` holds and how the Monitors form reads and writes it. A
// config can also arrive from `elbi sync` or the API, so reading it trusts nothing and
// writing it keeps every key the form does not edit.

import type { FilterClause } from "@/lib/metrics"

export const AGGS = ["mean", "sum", "min", "max", "count"] as const
export type Agg = (typeof AGGS)[number]
export const OPS: FilterClause["op"][] = ["eq", "ne", "lt", "le", "gt", "ge", "in"]

export interface Measure {
  agg: Agg
  column: string
  key: string
}

export interface FilterRow {
  column: string
  op: FilterClause["op"]
  value: string
  // The stored clause this row was read from, so a value the user did not touch keeps
  // its type ("007" stays a string, a list keeps its commas).
  source?: FilterClause
}

type Config = Record<string, unknown>

export function readMeasure(config: Config): Measure {
  const m = config.measure
  if (m && typeof m === "object") {
    const o = m as Record<string, unknown>
    const agg = AGGS.includes(o.agg as Agg) ? (o.agg as Agg) : "sum"
    return { agg, column: String(o.column ?? ""), key: String(o.key ?? "") }
  }
  return { agg: "mean", column: "", key: "" }
}

const show = (value: unknown) => (Array.isArray(value) ? value.join(", ") : String(value ?? ""))

export function readFilters(config: Config): FilterRow[] {
  const f = config.filters
  if (!Array.isArray(f)) return []
  return f.map((raw) => {
    const c = (raw && typeof raw === "object" ? raw : {}) as Partial<FilterClause>
    const op = OPS.includes(c.op as FilterClause["op"]) ? (c.op as FilterClause["op"]) : "eq"
    const valid = typeof c.column === "string" && c.column !== "" && op === c.op
    return {
      column: String(c.column ?? ""),
      op,
      value: show(c.value),
      source: valid ? (c as FilterClause) : undefined,
    }
  })
}

export function filterValue(row: FilterRow): unknown {
  const { source } = row
  if (source && row.op === source.op && row.value === show(source.value)) return source.value
  const one = (v: string) => (v.trim() !== "" && !Number.isNaN(Number(v)) ? Number(v) : v.trim())
  return row.op === "in" ? row.value.split(",").map(one) : one(row.value)
}

export interface ConfigEdit {
  agg: Agg
  column: string
  key: string
  filters: FilterRow[]
}

// The config to save: the stored one with only the measure (derivation) or filters
// (metric) replaced, so settings the form does not model, such as a metric monitor's
// `group_by` and `grain`, are not dropped. Where an edit changes nothing, the stored
// value is kept as it was written, so the backend does not see a different config.
export function buildConfig(base: Config, kind: "metric" | "derivation", edit: ConfigEdit): Config {
  if (kind === "derivation") {
    const { measure: stored, ...rest } = base
    const column = edit.column.trim()
    if (!column) return rest
    const key = edit.key.trim()
    const next = { agg: edit.agg, column, ...(key ? { key } : {}) }
    const old = readMeasure(base)
    const same =
      stored &&
      typeof stored === "object" &&
      old.agg === edit.agg &&
      old.column === column &&
      old.key === key
    return { ...rest, measure: same ? stored : next }
  }
  const { filters: stored, ...rest } = base
  const next = edit.filters
    .filter((f) => f.column.trim())
    .map((f) => ({ column: f.column.trim(), op: f.op, value: filterValue(f) }))
  if (next.length === 0) return rest
  return { ...rest, filters: JSON.stringify(next) === JSON.stringify(stored) ? stored : next }
}

export function describeConfig(kind: "metric" | "derivation", config: Config): string {
  const extra = ["group_by", "grain"]
    .filter((k) => config[k] != null && String(config[k]) !== "")
    .map((k) => `${k.replace("_", " ")} ${show(config[k])}`)
  if (kind === "metric") {
    const filters = readFilters(config)
    const where = filters.length
      ? `where ${filters.map((f) => `${f.column} ${f.op} ${f.value}`).join(" and ")}`
      : "the metric's total"
    return [where, ...extra].join(", ")
  }
  const m = readMeasure(config)
  if (!m.column) return "row count"
  return `${m.agg} of ${m.column}${m.key ? `, naming rows by ${m.key}` : ""}`
}
