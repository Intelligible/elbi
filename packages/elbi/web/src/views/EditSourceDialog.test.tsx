import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { EditSourceDialog } from "./EditSourceDialog"

const { getSourceConfig, getCatalog, updateSource } = vi.hoisted(() => ({
  getSourceConfig: vi.fn(),
  getCatalog: vi.fn(),
  updateSource: vi.fn(),
}))

vi.mock("@/lib/warehouse", () => ({ getSourceConfig, getCatalog, updateSource }))

// `secret` is resolved by the server: true for every password field, and for a
// credential that renders as a textarea.
const FIELDS = [
  { name: "manifest_json", label: "Manifest (JSON)", type: "textarea", options: [] },
  { name: "auth_token", label: "Bearer token", type: "password", secret: true, options: [] },
]

function setup(overrides: Partial<Parameters<typeof EditSourceDialog>[0]> = {}) {
  getSourceConfig.mockResolvedValue({
    sourceType: "custom",
    config: { manifest_json: '{"client":{}}', auth_token: "" },
    secretFields: ["auth_token"],
  })
  getCatalog.mockResolvedValue({ sources: [{ name: "custom", fields: FIELDS }] })
  updateSource.mockResolvedValue({})
  const onSaved = vi.fn()
  render(
    <EditSourceDialog
      sourceId="abc"
      sourceType="custom"
      initialName="posthog"
      initialDescription=""
      open
      onOpenChange={() => {}}
      onSaved={onSaved}
      {...overrides}
    />,
  )
  return { onSaved }
}

describe("EditSourceDialog", () => {
  beforeEach(() => vi.clearAllMocks())

  it("pre-fills the stored config so a manifest need not be retyped", async () => {
    setup()

    const manifest = await screen.findByDisplayValue('{"client":{}}')
    expect(manifest).toBeTruthy()
  })

  it("never receives the secret, and says a blank field keeps it", async () => {
    setup()
    await screen.findByDisplayValue('{"client":{}}')

    // The server sends secrets blank; the dialog must not invent a placeholder that
    // would be submitted as a new value.
    const token = screen.getByLabelText("Bearer token") as HTMLInputElement
    expect(token.value).toBe("")
    expect(screen.getByPlaceholderText(/blank to keep the stored value/i)).toBeTruthy()
  })

  it("sends the edited manifest with the secret still blank", async () => {
    const user = userEvent.setup()
    setup()
    const manifest = await screen.findByDisplayValue('{"client":{}}')

    await user.clear(manifest)
    await user.type(manifest, "edited")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(updateSource).toHaveBeenCalled())
    expect(updateSource).toHaveBeenCalledWith("abc", {
      name: "posthog",
      description: "",
      config: { manifest_json: "edited", auth_token: "" },
    })
  })

  it("renaming does not re-test the connection", async () => {
    // Sending `config` makes the server fetch from the API. A lapsed credential should
    // not stand between an operator and a typo in a name.
    const user = userEvent.setup()
    setup()
    await screen.findByDisplayValue('{"client":{}}')

    await user.clear(screen.getByLabelText("Name"))
    await user.type(screen.getByLabelText("Name"), "posthog-funnel")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(updateSource).toHaveBeenCalled())
    const [, patch] = updateSource.mock.calls[0]
    expect(patch.name).toBe("posthog-funnel")
    expect(patch).not.toHaveProperty("config")
  })

  it("edits the description without touching the connection", async () => {
    const user = userEvent.setup()
    setup()
    await screen.findByDisplayValue('{"client":{}}')

    await user.type(screen.getByLabelText("Description"), "activation funnel")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(updateSource).toHaveBeenCalled())
    const [, patch] = updateSource.mock.calls[0]
    expect(patch.description).toBe("activation funnel")
    expect(patch).not.toHaveProperty("config")
  })

  it("keeps the dialog open and shows why when the connection test fails", async () => {
    const user = userEvent.setup()
    const { onSaved } = setup()
    updateSource.mockRejectedValue(new Error("did not finish within 20s"))
    await screen.findByDisplayValue('{"client":{}}')

    await user.click(screen.getByRole("button", { name: "Save" }))

    // The failure is the reason the edit was refused, so it belongs on the form the
    // operator is about to correct — not behind a closed dialog.
    expect(await screen.findByRole("alert")).toHaveTextContent(/did not finish/)
    expect(onSaved).not.toHaveBeenCalled()
  })
})

describe("EditSourceDialog credential fields", () => {
  beforeEach(() => vi.clearAllMocks())

  const FIVE_AUTH = [
    { name: "manifest_json", label: "Manifest (JSON)", type: "textarea", options: [] },
    { name: "auth_token", label: "Bearer token", type: "password", secret: true, options: [] },
    { name: "auth_api_key", label: "API key", type: "password", secret: true, options: [] },
    {
      name: "auth_password",
      label: "Auth password",
      type: "password",
      secret: true,
      options: [],
    },
    {
      name: "auth_oauth2_client_secret",
      label: "OAuth2 client secret",
      type: "password",
      secret: true,
      options: [],
    },
  ]

  function renderWith(secretFields: string[]) {
    getSourceConfig.mockResolvedValue({
      sourceType: "custom",
      config: { manifest_json: "{}", auth_token: "" },
      secretFields,
    })
    getCatalog.mockResolvedValue({ sources: [{ name: "custom", fields: FIVE_AUTH }] })
    render(
      <EditSourceDialog
        sourceId="abc"
        sourceType="custom"
        initialName="posthog"
        initialDescription=""
        open
        onOpenChange={() => {}}
        onSaved={() => {}}
      />,
    )
  }

  it("offers only the credential the source actually holds", async () => {
    // Custom REST declares five credentials and a manifest uses one. Showing all five
    // is noise around the only field that matters.
    renderWith(["auth_token"])
    await screen.findByLabelText("Bearer token")

    expect(screen.queryByLabelText("API key")).toBeNull()
    expect(screen.queryByLabelText("Auth password")).toBeNull()
    expect(screen.queryByLabelText("OAuth2 client secret")).toBeNull()
  })

  it("offers all of them when the source holds none yet", async () => {
    // Nothing to narrow to, and hiding them all would leave no way to add one.
    renderWith([])
    await screen.findByLabelText("Bearer token")

    expect(screen.getByLabelText("API key")).toBeTruthy()
    expect(screen.getByLabelText("OAuth2 client secret")).toBeTruthy()
  })
})

describe("EditSourceDialog multi-line secrets", () => {
  beforeEach(() => vi.clearAllMocks())

  // A PEM key is a credential that needs the multi-line control. Read off the control
  // type, it was treated as ordinary text: offered without the hint that a blank box
  // keeps the stored key, and never narrowed away with the other credentials.
  const PEM_FIELDS = [
    { name: "host", label: "Host", type: "text", options: [] },
    {
      name: "ssh_password",
      label: "SSH password",
      type: "password",
      secret: true,
      options: [],
    },
    {
      name: "ssh_private_key",
      label: "SSH private key",
      type: "textarea",
      secret: true,
      options: [],
    },
  ]

  function renderWith(secretFields: string[]) {
    getSourceConfig.mockResolvedValue({
      sourceType: "postgres",
      config: { host: "db.internal", ssh_private_key: "" },
      secretFields,
    })
    getCatalog.mockResolvedValue({ sources: [{ name: "postgres", fields: PEM_FIELDS }] })
    render(
      <EditSourceDialog
        sourceId="abc"
        sourceType="postgres"
        initialName="warehouse"
        initialDescription=""
        open
        onOpenChange={() => {}}
        onSaved={() => {}}
      />,
    )
  }

  it("says a blank key keeps the stored one", async () => {
    renderWith(["ssh_private_key"])

    const key = (await screen.findByLabelText("SSH private key")) as HTMLTextAreaElement
    expect(key.value).toBe("")
    expect(key.placeholder).toMatch(/blank to keep the stored value/i)
  })

  it("narrows to the credential the source holds, textarea or not", async () => {
    renderWith(["ssh_private_key"])
    await screen.findByLabelText("SSH private key")

    expect(screen.queryByLabelText("SSH password")).toBeNull()
    expect(screen.getByLabelText("Host")).toBeTruthy()
  })
})
