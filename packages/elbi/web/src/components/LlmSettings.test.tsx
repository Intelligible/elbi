import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { LlmProfilesManager } from "@/components/LlmSettings"
import { FeedbackProvider } from "@/components/ui/feedback"

// The wire shape GET /api/settings/llm/profiles actually returns, copied from the server's
// own assertion (test_app.py::test_llm_profile_crud_and_default) rather than written from
// this component. The API camelCases every field name on the way out (casing.py), so a
// fixture spelled in snake_case would let a client that reads snake_case pass a test the
// real server fails -- which is exactly how a stored base URL came to be invisible here.
const BASE_URL = "http://localhost:11434"

const LISTING = {
  profiles: [
    {
      name: "Local",
      model: "ollama/llama3",
      baseUrl: BASE_URL,
      reasoningEffort: "",
      apiKeySet: true,
      provider: "ollama",
      providerLabel: "Ollama",
      reasoningEfforts: [],
    },
  ],
  default: "Local",
  titleProfile: "",
  configured: true,
}

/** Bodies the panel PUT back, so the test can see what a save would persist. */
let puts: { model: string; base_url: string; api_key?: string }[]

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json" },
  })
}

beforeEach(() => {
  puts = []
  vi.stubGlobal("fetch", (_url: string, init?: RequestInit) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)))
      return Promise.resolve(json({}))
    }
    return Promise.resolve(json(LISTING))
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

it("reads a saved profile off the wire and does not wipe its base URL on save", async () => {
  const user = userEvent.setup()
  render(
    <FeedbackProvider>
      <LlmProfilesManager />
    </FeedbackProvider>,
  )

  // The row reports a stored key from `apiKeySet`; reading the wrong field name shows a
  // configured profile as having no key at all.
  expect(await screen.findByText(/key set/)).toBeInTheDocument()

  await user.click(screen.getByRole("button", { name: "Edit profile" }))
  // The editor has to seed itself from the stored URL. It starts blank when the field name
  // is wrong, which reads as "the base URL never saved".
  expect(screen.getByPlaceholderText(/custom or self-hosted/)).toHaveValue(BASE_URL)

  await user.click(screen.getByRole("button", { name: "Save" }))
  // ...and send it back untouched. The editor always includes `base_url` in the patch, so a
  // blank field does not mean "leave it alone" -- it overwrites the stored URL with "".
  await waitFor(() => expect(puts).toHaveLength(1))
  expect(puts[0].base_url).toBe(BASE_URL)
})

it("changing the title model saves the selection", async () => {
  const user = userEvent.setup()
  render(
    <FeedbackProvider>
      <LlmProfilesManager />
    </FeedbackProvider>,
  )

  const trigger = await screen.findByRole("combobox", { name: "Title model" })
  // jsdom fires a window blur on pointerdown once an earlier test in this file has run,
  // which closes a just-opened Radix Select; focusing first keeps it open (see repo notes
  // on this jsdom artefact).
  trigger.focus()
  await user.click(trigger)
  await user.click(screen.getByRole("option", { name: "Local" }))

  await waitFor(() => expect(puts).toHaveLength(1))
  expect(puts[0]).toEqual({ name: "Local" })
})

it("selecting Default sends the empty-string name, not the sentinel value", async () => {
  const user = userEvent.setup()
  // Starts with a real title model set, so picking "Default" is a genuine change back to
  // "no override" -- the sentinel used for that Radix item must not leak into the request.
  vi.stubGlobal("fetch", (_url: string, init?: RequestInit) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)))
      return Promise.resolve(json({}))
    }
    return Promise.resolve(json({ ...LISTING, titleProfile: "Local" }))
  })
  render(
    <FeedbackProvider>
      <LlmProfilesManager />
    </FeedbackProvider>,
  )

  const trigger = await screen.findByRole("combobox", { name: "Title model" })
  trigger.focus()
  await user.click(trigger)
  await user.click(screen.getByRole("option", { name: "Default" }))

  await waitFor(() => expect(puts).toHaveLength(1))
  expect(puts[0]).toEqual({ name: "" })
})
