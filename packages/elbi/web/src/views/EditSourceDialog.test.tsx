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

const FIELDS = [
  { name: "manifest_json", label: "Manifest (JSON)", type: "textarea", options: [] },
  { name: "auth_token", label: "Bearer token", type: "password", options: [] },
]

function setup(overrides: Partial<Parameters<typeof EditSourceDialog>[0]> = {}) {
  getSourceConfig.mockResolvedValue({
    source_type: "custom",
    config: { manifest_json: '{"client":{}}', auth_token: "" },
    secret_fields: ["auth_token"],
  })
  getCatalog.mockResolvedValue({ sources: [{ name: "custom", fields: FIELDS }] })
  updateSource.mockResolvedValue({})
  const onSaved = vi.fn()
  render(
    <EditSourceDialog
      sourceId="abc"
      sourceType="custom"
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
    expect(screen.getByText(/blank to keep the stored value/i)).toBeTruthy()
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
      config: { manifest_json: "edited", auth_token: "" },
    })
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
