// The logos are vendored files, so the failure mode is a broken reference rather than a
// wrong render: an alias pointing at a logo we do not ship, or a file edited away from what
// upstream published. Both show up as a blank icon in the picker and nothing in the console,
// which is exactly the kind of fault a test has to catch instead of a person.

import { createHash } from "node:crypto"
import { readdirSync, readFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { COLOURLESS_MODEL_ICONS, VENDORED_MODEL_ICONS } from "@/lib/modelIcons.generated"
import { ICON_ALIASES, iconSlugFor, ProviderGlyph } from "./ProviderIcon"

const iconDir = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "model-icons")

type Manifest = {
  commit: string
  icons: Record<string, { upstream: string; sha256: string; colourless: boolean }>
}
const manifest: Manifest = JSON.parse(readFileSync(join(iconDir, "manifest.json"), "utf8"))

describe("vendored provider logos", () => {
  it("ships a file for every logo the app claims to have", () => {
    const onDisk = new Set(
      readdirSync(iconDir)
        .filter((f) => f.endsWith(".svg"))
        .map((f) => f.replace(/\.svg$/, "")),
    )
    expect([...VENDORED_MODEL_ICONS].filter((s) => !onDisk.has(s))).toEqual([])
    // And nothing stale: a file no longer listed would keep being shipped unnoticed.
    expect([...onDisk].filter((s) => !VENDORED_MODEL_ICONS.has(s))).toEqual([])
  })

  it("matches the checksum upstream was vendored at, byte for byte", () => {
    // The whole point of vendoring is that correctness comes from the source. A hand edit
    // here -- however well meant -- is a logo nobody reviewed, so it has to fail.
    const drifted = [...VENDORED_MODEL_ICONS].filter((slug) => {
      const svg = readFileSync(join(iconDir, `${slug}.svg`))
      return createHash("sha256").update(svg).digest("hex") !== manifest.icons[slug]?.sha256
    })
    expect(drifted).toEqual([])
  })

  it("flags a mark as colourless only when it really carries no colour", () => {
    // This drives whether the mark is inverted on a dark background, so getting it wrong
    // either hides a black logo or recolours a brand's own palette.
    for (const slug of VENDORED_MODEL_ICONS) {
      const svg = readFileSync(join(iconDir, `${slug}.svg`), "utf8")
      const hasColour =
        /(?:fill|stroke|stop-color)="(?:#(?!000000\b|000\b)[0-9a-f]{3,8}|rgb)/i.test(svg)
      expect(COLOURLESS_MODEL_ICONS.has(slug), `${slug} colourless flag`).toBe(!hasColour)
    }
  })

  it("only inverts marks it also ships", () => {
    expect([...COLOURLESS_MODEL_ICONS].filter((s) => !VENDORED_MODEL_ICONS.has(s))).toEqual([])
  })

  it("points every alias at a logo that exists", () => {
    // An alias naming a logo we do not ship renders an <img> to a 404: a blank space in the
    // picker with nothing in the console to explain it.
    const dangling = Object.entries(ICON_ALIASES)
      .filter(([, slug]) => !VENDORED_MODEL_ICONS.has(slug))
      .map(([provider, slug]) => `${provider} -> ${slug}`)
    expect(dangling).toEqual([])
  })
})

describe("iconSlugFor", () => {
  it("resolves a provider whose name is its logo's name", () => {
    expect(iconSlugFor("openai")).toBe("openai")
    expect(iconSlugFor("mistral")).toBe("mistral")
  })

  it("resolves a provider routed under another brand's mark", () => {
    expect(iconSlugFor("watsonx")).toBe("ibm")
    expect(iconSlugFor("sagemaker")).toBe("aws")
    expect(iconSlugFor("vertex_ai")).toBe("vertexai")
  })

  it("strips a routing suffix that is not part of the brand", () => {
    expect(iconSlugFor("fireworks_ai")).toBe("fireworks")
    expect(iconSlugFor("ollama_chat")).toBe("ollama")
    expect(iconSlugFor("text-completion-openai")).toBe("openai")
  })

  it("resolves a service named after neither its company nor its model", () => {
    expect(iconSlugFor("dashscope")).toBe("alibabacloud")
    expect(iconSlugFor("triton")).toBe("nvidia")
    expect(iconSlugFor("wandb")).toBe("weightsandbiases")
  })

  it("returns null for a provider we ship no logo for, rather than a broken path", () => {
    expect(iconSlugFor("oci")).toBeNull()
    expect(iconSlugFor("nonesuch")).toBeNull()
  })
})

describe("ProviderGlyph", () => {
  it("shows a real logo from the vendored set", () => {
    render(<ProviderGlyph model="gpt-5.6-terra" provider="openai" providerLabel="OpenAI" />)
    expect(screen.getByAltText("OpenAI").getAttribute("src")).toBe("/model-icons/openai.svg")
  })

  it("inverts a colourless mark on a dark background but never a colour one", () => {
    render(<ProviderGlyph model="m" provider="openai" providerLabel="OpenAI" />)
    expect(screen.getByAltText("OpenAI").className).toContain("dark:invert")

    render(<ProviderGlyph model="m" provider="gemini" providerLabel="Gemini" />)
    expect(screen.getByAltText("Gemini").className).not.toContain("invert")
  })

  it("uses the full-colour Claude mark for Anthropic", () => {
    render(<ProviderGlyph model="claude-opus-5" provider="anthropic" />)
    expect(screen.getByAltText("Claude").getAttribute("src")).toBe("/claude.svg")
  })

  it("shows nothing at all for a provider with no authentic logo", () => {
    // The rule that matters: no stand-in. A generic shape sitting where a logo goes reads as
    // that provider's logo, so a provider we have no mark for renders nothing.
    const { container } = render(
      <ProviderGlyph model="m" provider="oci" providerLabel="Oracle Cloud" />,
    )
    expect(container.firstChild).toBeNull()
  })

  it("shows nothing for an OpenAI-compatible endpoint, which is not OpenAI", () => {
    // `openai_like` means "some server speaking the OpenAI API" -- it could be vLLM or
    // anything else, so OpenAI's logo would name the wrong company.
    for (const provider of ["openai_like", "custom_openai", "aiohttp_openai"]) {
      const { container } = render(<ProviderGlyph model="m" provider={provider} />)
      expect(container.firstChild, provider).toBeNull()
    }
  })

  it("derives the provider from a prefixed model when none is passed", () => {
    render(<ProviderGlyph model="mistral/mistral-large" />)
    expect(screen.getByAltText("mistral").getAttribute("src")).toBe("/model-icons/mistral.svg")
  })

  it("renders nothing when there is no provider to show", () => {
    const { container } = render(<ProviderGlyph model="bare-model-name" />)
    expect(container.firstChild).toBeNull()
  })
})
