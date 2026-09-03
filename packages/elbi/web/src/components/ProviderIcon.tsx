// Provider logos for the model picker, so a model reads as itself at a glance.
//
// The rule: a provider shows its real logo or it shows nothing. There is no stand-in.
//
// An earlier version of this file drew a few marks by hand and they were simply wrong -- the
// OpenAI mark is an interlocking knot described by a 1.5 KB path, and three ellipses is not a
// simplification of it but a different shape. A later version fell back to a generic glyph
// per provider role, which is the same mistake wearing a different hat: sitting where a logo
// goes, a shape reads as that provider's logo. No logo is better than a wrong one, so a
// provider we have no authentic mark for renders null and its name carries it alone.
//
// The marks are real files, vendored from lobe-icons and simple-icons at pinned versions --
// see scripts/vendor-model-icons.mjs. Correctness comes from the source, not from anyone's
// eye, and a checksum test fails if a byte drifts. They are served from public/ rather than a
// CDN because a deployment may have no route to the internet, and a logo that needs fetching
// is a logo that is missing exactly where the product can least afford to look half-built.
//
// Each is shown in the brand's own colours. The exception is marks that carry no colour --
// OpenAI, Ollama, xAI and Groq are black wordless marks, and black is the logo. Inside an
// <img> the SVG is its own document, so `currentColor` there resolves to black whatever the
// app's theme; on a dark background those get the white treatment their brand guidelines
// specify. A mark with real colours is never recoloured.
//
// The provider is resolved on the server, not parsed here. Splitting a model string on "/" is
// wrong in both directions -- a bare `gpt-5.6-terra` has no prefix and is still OpenAI,
// `bedrock/us.anthropic.…` has one that is not the model's family -- and the server already
// knows the answer because it has to route on it. The `model` prop remains as a fallback for
// a caller with no resolved provider to hand.

import { COLOURLESS_MODEL_ICONS, VENDORED_MODEL_ICONS } from "@/lib/modelIcons.generated"

// Providers whose name is not the name of their logo. Three kinds of mismatch: a routing
// suffix that is not part of the brand (`fireworks_ai`, `ollama_chat`), a product routed
// under its vendor's mark (`watsonx` is IBM's, `triton` is NVIDIA's), and a name that is the
// service rather than the company (`dashscope` is Alibaba Cloud's).
//
// Deliberately absent: `openai_like`, `custom_openai` and `aiohttp_openai`. Those mean "some
// server speaking the OpenAI API", which could be vLLM, LM Studio or anything else -- showing
// OpenAI's logo would name the wrong company.
//
// Tests hold every key here to a provider LiteLLM can actually route and every value to a
// logo that is actually vendored, so neither side can rot silently.
export const ICON_ALIASES: Record<string, string> = {
  ai21_chat: "ai21",
  amazon_nova: "aws",
  azure_ai: "azure",
  azure_text: "azure",
  bedrock_mantle: "bedrock",
  black_forest_labs: "flux",
  chatgpt: "openai",
  codestral: "mistral",
  cohere_chat: "cohere",
  dashscope: "alibabacloud",
  docker_model_runner: "docker",
  featherless_ai: "featherless",
  fireworks_ai: "fireworks",
  friendliai: "friendli",
  github_copilot: "githubcopilot",
  gradient_ai: "digitalocean",
  hosted_vllm: "vllm",
  lambda_ai: "lambda",
  lm_studio: "lmstudio",
  meta_llama: "meta",
  nvidia_nim: "nvidia",
  ollama_chat: "ollama",
  ovhcloud: "ovh",
  sagemaker: "aws",
  sagemaker_chat: "aws",
  sagemaker_nova: "aws",
  "text-completion-codestral": "mistral",
  "text-completion-inception": "inception",
  "text-completion-openai": "openai",
  together_ai: "together",
  triton: "nvidia",
  vercel_ai_gateway: "vercel",
  vertex_ai: "vertexai",
  vertex_ai_beta: "vertexai",
  wandb: "weightsandbiases",
  watsonx: "ibm",
  watsonx_text: "ibm",
  xiaomi_mimo: "xiaomimimo",
}

/** The vendored logo for a provider, or null when we do not ship one. */
export function iconSlugFor(provider: string): string | null {
  const alias = ICON_ALIASES[provider]
  if (alias) return alias
  return VENDORED_MODEL_ICONS.has(provider) ? provider : null
}

// The Claude sunburst, shown beside a model name in the picker -- the model's own logo.
export function ClaudeIcon({ className }: { className?: string }) {
  return <img src="/claude.svg" alt="Claude" className={className} />
}

/**
 * A provider's logo, or nothing.
 *
 * Prefers the server-resolved `provider` and falls back to the model string's prefix, so a
 * caller that has not been updated still renders something. Returns null for any provider we
 * have no authentic logo for -- the model's name stands on its own rather than beside a mark
 * that belongs to nobody.
 */
export function ProviderGlyph({
  model,
  provider,
  providerLabel,
  className = "h-3.5 w-3.5",
}: {
  model: string
  provider?: string
  providerLabel?: string
  className?: string
}) {
  const resolved = (provider || (model.includes("/") ? model.split("/")[0] : "")).toLowerCase()
  if (!resolved) return null

  if (resolved === "anthropic" || resolved === "anthropic_text") {
    return <ClaudeIcon className={className} />
  }

  const slug = iconSlugFor(resolved)
  if (!slug) return null

  const label = providerLabel || resolved
  return (
    <img
      src={`/model-icons/${slug}.svg`}
      alt={label}
      title={label}
      className={`shrink-0 object-contain ${COLOURLESS_MODEL_ICONS.has(slug) ? "dark:invert" : ""} ${className}`}
    />
  )
}
