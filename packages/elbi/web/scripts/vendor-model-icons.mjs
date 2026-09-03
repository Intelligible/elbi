// Vendor the model-provider logos into public/model-icons/.
//
// The rule this script exists to enforce: a provider gets its real logo or it gets no logo.
// Nothing here is drawn, traced, approximated or recoloured. An earlier version of the app
// had hand-drawn marks and they were simply wrong -- the OpenAI mark is an interlocking knot
// described by a 1.5 KB path, and a few ellipses is not a simplification of it but a
// different shape. A provider with no authentic mark available shows no mark at all, which
// is honest; a plausible-looking stand-in is not.
//
// Three sources, all permissively licensed, all authoritative:
//
//   lobe-icons     purpose-built for AI/LLM providers, and the only set carrying marks for
//                  OpenAI, Groq, Cohere, Together and the rest of the model-serving long
//                  tail. Copied verbatim at a pinned commit.
//   simple-icons   already a dependency of this app. Covers the general-infrastructure
//                  brands lobe-icons lacks (Docker, Databricks, SAP, Weights & Biases). Its
//                  data carries each brand's official path and official hex, so the file
//                  written here is assembled from the package rather than authored.
//   gilbarbara     for brands the other two withdrew or never had. Copied verbatim at a
//                  pinned commit.
//
// Providers with no authentic mark in any of them get none. As of this writing that is a
// short tail of small or very new vendors, plus one notable case worth recording so nobody
// re-researches it: Oracle. Every permissively-licensed set has Oracle only as its wordmark,
// roughly 8:1, because Oracle has no square icon mark. Fitted to a 14px picker slot that is a
// two-pixel red smear -- authentic, and still not a logo. So `oci` shows nothing.
//
// Files are checked in rather than fetched at run time because a deployment may have no
// route to the internet, and a logo that needs fetching is a logo that is missing exactly
// where the product can least afford to look half-built. Provenance lives in manifest.json
// (source, upstream name, sha256 per file) so any change is auditable, and
// `src/lib/modelIcons.generated.ts` is written from what actually landed on disk so the app
// can never claim a logo it does not ship.
//
// A brand's colours are part of its logo, so the full-colour variant is always preferred.
// Some marks genuinely are one colour -- OpenAI, Ollama, xAI and Groq are black wordless
// marks, and black *is* the logo. Those are recorded as colourless so the app can show them
// the way their own brand guidelines ask on a dark background: the same mark in white. A
// mark that has colours of its own is never touched.
//
// Run with `npm run icons:vendor`. Re-running is idempotent.
//
// lobe-icons is MIT and simple-icons is CC0; see NOTICE at the repository root.

import { createHash } from "node:crypto"
import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import * as simpleIcons from "simple-icons"

const UPSTREAM_REPO = "lobehub/lobe-icons"
// Pinned rather than tracking master: a logo set that changes under us on an unrelated
// `npm install` is a change nobody reviewed.
const UPSTREAM_COMMIT = "f07e9be35aef452ce735f95ea8204a14ecc513f7"
const UPSTREAM_DIR = "packages/static-svg/icons"

const GILBARBARA_REPO = "gilbarbara/logos"
const GILBARBARA_COMMIT = "4de741f8503d5e81abf5dfa05214690e938296bf"

// Brands the other two sets do not carry, keyed by the filename we write to. `heroku-icon` is
// the square mark; plain `heroku` there is the wordmark, which does not survive an icon slot.
const GILBARBARA_ICONS = {
  heroku: "heroku-icon",
}

// Brands whose only authentic mark is in simple-icons, keyed by the filename we write to.
// Each value is a simple-icons title, matched exactly against the package so a retitled or
// withdrawn icon fails the run rather than silently vanishing from the picker.
const SIMPLE_ICONS = {
  clarifai: "Clarifai",
  databricks: "Databricks",
  digitalocean: "DigitalOcean",
  docker: "Docker",
  ovh: "OVH",
  sap: "SAP",
  scaleway: "Scaleway",
  weightsandbiases: "Weights & Biases",
}

// The logos we ship from lobe-icons, one slug per upstream file.
//
// Anthropic is deliberately absent: the app already ships the full-colour Claude sunburst at
// public/claude.svg, which is a better mark than the monochrome one here.
const SLUGS = [
  "ai21",
  "alibabacloud",
  "apertis",
  "aws",
  "azure",
  "baseten",
  "bedrock",
  "cerebras",
  "cloudflare",
  "cohere",
  "cometapi",
  "cursor",
  "deepinfra",
  "deepseek",
  "featherless",
  "fireworks",
  "flux",
  "friendli",
  "gemini",
  "github",
  "githubcopilot",
  "groq",
  "huggingface",
  "hyperbolic",
  "ibm",
  "inception",
  "lambda",
  "lmstudio",
  "meta",
  "minimax",
  "mistral",
  "modelscope",
  "moonshot",
  "morph",
  "nebius",
  "novita",
  "nvidia",
  "ollama",
  "openai",
  "openrouter",
  "parasail",
  "perplexity",
  "poe",
  "qwen",
  "replicate",
  "sambanova",
  "snowflake",
  "stability",
  "together",
  "v0",
  "vercel",
  "vertexai",
  "vllm",
  "volcengine",
  "xai",
  "xiaomimimo",
  "xinference",
  "zai",
]

const here = dirname(fileURLToPath(import.meta.url))
const iconDir = join(here, "..", "public", "model-icons")
const generatedFile = join(here, "..", "src", "lib", "modelIcons.generated.ts")

// Read rather than imported: the package does not export its own package.json. Recorded so
// the manifest pins both sources, not just the one fetched over the network.
const simpleIconsVersion = JSON.parse(
  await readFile(join(here, "..", "node_modules", "simple-icons", "package.json"), "utf8"),
).version

async function fetchSvg(url, label) {
  const response = await fetch(url)
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`${label}: HTTP ${response.status} from ${url}`)
  const svg = await response.text()
  // Guard against a 404 body, an HTML error page, or a truncated read being written to
  // disk as if it were a logo -- the failure would otherwise surface as a blank icon.
  if (!svg.includes("<svg") || !svg.includes("</svg>")) {
    throw new Error(`${label}: response is not an SVG document`)
  }
  return svg
}

const fetchVariant = (name) =>
  fetchSvg(
    `https://raw.githubusercontent.com/${UPSTREAM_REPO}/${UPSTREAM_COMMIT}/${UPSTREAM_DIR}/${name}.svg`,
    name,
  )

async function fetchGilbarbara(slug, name) {
  const url = `https://raw.githubusercontent.com/${GILBARBARA_REPO}/${GILBARBARA_COMMIT}/logos/${name}.svg`
  const svg = await fetchSvg(url, slug)
  if (!svg) throw new Error(`${slug}: ${GILBARBARA_REPO} has no logos/${name}.svg`)
  return { svg, upstream: `${name}.svg`, source: "gilbarbara/logos" }
}

// The full-colour mark when the brand has one, else the single-colour mark, which for these
// brands is the logo rather than a reduction of it.
async function fetchIcon(slug) {
  const colour = await fetchVariant(`${slug}-color`)
  if (colour) return { svg: colour, upstream: `${slug}-color.svg`, source: "lobe-icons" }
  const mono = await fetchVariant(slug)
  if (!mono) throw new Error(`${slug}: no ${slug}-color.svg and no ${slug}.svg upstream`)
  return { svg: mono, upstream: `${slug}.svg`, source: "lobe-icons" }
}

// simple-icons ships data, not files, so the file is assembled here -- but every part of it
// comes from the package: the brand's own path and its official hex. Looking the icon up by
// title rather than by export name means a brand that upstream retitles or withdraws (as
// happened to OpenAI) fails this run loudly instead of disappearing from the picker.
const byTitle = new Map(
  Object.values(simpleIcons)
    .filter((icon) => icon && typeof icon === "object" && "path" in icon)
    .map((icon) => [icon.title, icon]),
)

function buildSimpleIcon(slug, title) {
  const icon = byTitle.get(title)
  if (!icon) throw new Error(`${slug}: simple-icons has no icon titled "${title}"`)
  // "Weights & Biases" is a bare ampersand, which is not valid XML and would leave the file
  // unparseable -- a blank icon in the picker.
  const escaped = icon.title.replace(/&/g, "&amp;").replace(/</g, "&lt;")
  const svg =
    `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">` +
    `<title>${escaped}</title>` +
    `<path fill="#${icon.hex}" d="${icon.path}"/>` +
    `</svg>\n`
  return {
    svg,
    upstream: `${icon.slug} (simple-icons)`,
    source: "simple-icons",
    // Recorded as upstream reports it, including absent. simple-icons is CC0 as a project
    // but says that does not imply every icon in it is, and carries per-icon licence data
    // for only a fraction of them -- so `null` here means "upstream does not say", which is
    // not the same as CC0. Writing that down beats re-deriving it during a licence review.
    license: icon.license ?? null,
    guidelines: icon.guidelines ?? null,
  }
}

// Whether a mark carries no colour of its own, read from the file rather than assumed from
// its name. It matters because these are loaded with <img>, and inside an <img> the SVG is
// its own document: `currentColor` resolves against that document's initial colour, black,
// not against the page's text colour. So a `currentColor` mark paints black however the app
// is themed, and on a dark background it needs the white treatment its own brand guidelines
// specify. A mark with real colours is left exactly as the brand ships it.
function isColourless(svg) {
  const fills = [...svg.matchAll(/(?:fill|stroke)="([^"]+)"/g)].map((m) => m[1])
  return !fills.some(
    (v) => !["currentColor", "none", "#000", "#000000", "black", "inherit"].includes(v),
  )
}

await mkdir(iconDir, { recursive: true })

const allSlugs = [...SLUGS, ...Object.keys(SIMPLE_ICONS), ...Object.keys(GILBARBARA_ICONS)]
const duplicated = allSlugs.filter((slug, i) => allSlugs.indexOf(slug) !== i)
if (duplicated.length) throw new Error(`listed under more than one source: ${duplicated}`)

// Drop files from a previous run whose slug is no longer listed, so a removed entry does
// not leave an orphan that the generated set still advertises.
const existing = (await readdir(iconDir).catch(() => [])).filter((f) => f.endsWith(".svg"))
const wanted = new Set(allSlugs.map((s) => `${s}.svg`))
for (const stale of existing.filter((f) => !wanted.has(f))) {
  await rm(join(iconDir, stale))
  console.log(`  removed ${stale}`)
}

const results = await Promise.all([
  ...SLUGS.map(async (slug) => ({ slug, ...(await fetchIcon(slug)) })),
  ...Object.entries(SIMPLE_ICONS).map(async ([slug, title]) => ({
    slug,
    ...buildSimpleIcon(slug, title),
  })),
  ...Object.entries(GILBARBARA_ICONS).map(async ([slug, name]) => ({
    slug,
    ...(await fetchGilbarbara(slug, name)),
  })),
])

for (const icon of results) {
  await writeFile(join(iconDir, `${icon.slug}.svg`), icon.svg)
  icon.bytes = icon.svg.length
  icon.colourless = isColourless(icon.svg)
  icon.sha256 = createHash("sha256").update(icon.svg).digest("hex")
}
results.sort((a, b) => a.slug.localeCompare(b.slug))

await writeFile(
  join(iconDir, "manifest.json"),
  `${JSON.stringify(
    {
      note: "Written by scripts/vendor-model-icons.mjs. Do not edit by hand.",
      sources: {
        "lobe-icons": {
          url: `https://github.com/${UPSTREAM_REPO}`,
          license: "MIT",
          commit: UPSTREAM_COMMIT,
          path: UPSTREAM_DIR,
        },
        "simple-icons": {
          url: "https://github.com/simple-icons/simple-icons",
          license: "CC0-1.0",
          version: simpleIconsVersion,
        },
        "gilbarbara/logos": {
          url: `https://github.com/${GILBARBARA_REPO}`,
          license: "CC0-1.0",
          commit: GILBARBARA_COMMIT,
          path: "logos",
        },
      },
      trademarks:
        "These logos are trademarks of their respective owners, reproduced only to " +
        "identify the model provider a user has configured. The collection licences above " +
        "are copyright licences and grant no trademark rights. See NOTICE at the " +
        "repository root.",
      icons: Object.fromEntries(
        results.map((r) => [
          r.slug,
          {
            source: r.source,
            upstream: r.upstream,
            sha256: r.sha256,
            colourless: r.colourless,
            ...(r.license === undefined ? {} : { license: r.license }),
            ...(r.guidelines ? { guidelines: r.guidelines } : {}),
          },
        ]),
      ),
    },
    null,
    2,
  )}\n`,
)

await writeFile(
  generatedFile,
  `// Generated by scripts/vendor-model-icons.mjs -- do not edit.
//
// The provider logos actually present in public/model-icons/, so the icon component can tell
// a mark it ships from one it does not and fall back rather than request a 404.
//
// Source: https://github.com/${UPSTREAM_REPO} (MIT) at ${UPSTREAM_COMMIT}

export const VENDORED_MODEL_ICONS: ReadonlySet<string> = new Set([
${results.map((r) => `  "${r.slug}",`).join("\n")}
])

/**
 * Marks that carry no colour of their own, and so paint black inside an \`<img>\` whatever
 * the app's theme. These get the white treatment their brand guidelines specify on a dark
 * background. Every other mark is shown in the brand's own colours, untouched.
 */
export const COLOURLESS_MODEL_ICONS: ReadonlySet<string> = new Set([
${results
  .filter((r) => r.colourless)
  .map((r) => `  "${r.slug}",`)
  .join("\n")}
])
`,
)

const total = results.reduce((sum, r) => sum + r.bytes, 0)
const colour = results.filter((r) => !r.colourless).length
console.log(
  `vendored ${results.length} logos (${(total / 1024).toFixed(1)} KB) at ${UPSTREAM_COMMIT.slice(0, 12)}: ` +
    `${colour} in brand colours, ${results.length - colour} single-colour`,
)
