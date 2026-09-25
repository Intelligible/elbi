// @vitest-environment node
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs"
import path from "node:path"

import { ESLint } from "eslint"
import { describe, expect, it } from "vitest"

import { classifyColorUtility } from "../../eslint/design-system/color-utility.js"
import { readTheme } from "../../eslint/design-system/theme.js"

// The design-system contract, enforced: a violation fails `npm test`, not only `npm run lint`.
const WEB = path.resolve(import.meta.dirname, "../..")
const SRC = path.join(WEB, "src")
const SELF = path.join(SRC, "test/design-system.test.ts")

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = path.join(dir, name)
    return statSync(p).isDirectory() ? walk(p) : [p]
  })
}

describe("design system", () => {
  it("app code passes the design-system lint", async () => {
    const eslint = new ESLint({ cwd: WEB })
    const results = await eslint.lintFiles(["src"])
    const errors = results.flatMap((r) =>
      r.messages
        .filter((m) => m.severity === 2)
        .map(
          (m) => `${path.relative(WEB, r.filePath)}:${m.line} ${m.ruleId ?? "parse"} ${m.message}`,
        ),
    )
    expect(errors, errors.join("\n")).toEqual([])
  }, 60_000)

  it("the retired warehouse stylesheet is gone everywhere, vendored code included", () => {
    const legacy = /\bbtn--|\btitle-row\b|\.phc\b|PHC_CSS|--ph-/
    const hits = walk(SRC)
      .filter((f) => /\.(tsx?|css)$/.test(f) && f !== SELF)
      .flatMap((f) =>
        readFileSync(f, "utf8")
          .split("\n")
          .flatMap((line, i) => (legacy.test(line) ? [`${path.relative(WEB, f)}:${i + 1}`] : [])),
      )
    expect(hits, hits.join("\n")).toEqual([])
  })

  it("no file imports cn from the raw npm package, vendored code included", () => {
    // The shadcn CLI emits `import { cn } from "cn"`, which skips the type-scale merge
    // config in src/lib/utils.ts (see AGENTS.md, "Vendored: do not hand-edit").
    const fromRawCn = /from\s+["']cn["']/
    const hits = walk(SRC)
      .filter((f) => /\.tsx?$/.test(f) && f !== SELF)
      .flatMap((f) =>
        readFileSync(f, "utf8")
          .split("\n")
          .flatMap((line, i) =>
            fromRawCn.test(line) ? [`${path.relative(WEB, f)}:${i + 1}`] : [],
          ),
      )
    expect(hits, hits.join("\n")).toEqual([])
  })

  it("every colour utility the docs name is a real, allowed token", () => {
    const theme = readTheme(path.join(SRC, "index.css"))
    const docs = [path.join(WEB, ".interface-design/system.md"), path.join(WEB, "AGENTS.md")]
    const problems = docs.flatMap((doc) =>
      readFileSync(doc, "utf8")
        .split("\n")
        // A line that starts "Banned:" (after any list marker) names what not to use.
        .filter((line) => !/^\s*(?:[-*]\s+)?Banned:/.test(line))
        .flatMap((line) => [...line.matchAll(/`([a-z][a-z0-9-]*(?:\/\d+)?)`/g)].map((m) => m[1]))
        .flatMap((cls) => {
          const message = classifyColorUtility(cls, theme)
          return message ? [`${path.relative(WEB, doc)}: ${message}`] : []
        }),
    )
    expect(problems, problems.join("\n")).toEqual([])
  })

  it("every app-layer component has a colocated test", () => {
    const dir = path.join(SRC, "components/app")
    const components = existsSync(dir)
      ? readdirSync(dir).filter((f) => f.endsWith(".tsx") && !/\.(test|stories)\.tsx$/.test(f))
      : []
    const untested = components.filter(
      (f) => !existsSync(path.join(dir, f.replace(/\.tsx$/, ".test.tsx"))),
    )
    expect(untested).toEqual([])
  })
})
