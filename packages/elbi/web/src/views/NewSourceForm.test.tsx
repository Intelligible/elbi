import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { SourceConfig, SourceField } from "@/lib/warehouse"
import { createSource, uploadWarehouseFile } from "@/lib/warehouse"
import { NewSourceForm } from "./NewSourceForm"

vi.mock("@/lib/warehouse", () => ({
  createSource: vi.fn(),
  uploadWarehouseFile: vi.fn(),
}))

const field = (f: Partial<SourceField> & Pick<SourceField, "name" | "label" | "type">) =>
  ({
    required: false,
    placeholder: "",
    default: null,
    options: [],
    caption: "",
    ...f,
  }) as SourceField

const SOURCE: SourceConfig = {
  name: "postgres",
  label: "PostgreSQL",
  category: "Databases",
  caption: "",
  icon: "",
  releaseStatus: "ga",
  docsUrl: "",
  comingSoon: false,
  fields: [
    field({ name: "host", label: "Host", type: "text" }),
    field({
      name: "sslmode",
      label: "SSL mode",
      type: "select",
      default: "prefer",
      options: [
        { value: "prefer", label: "Prefer" },
        { value: "require", label: "Require" },
      ],
    }),
    field({ name: "ssh", label: "Connect through an SSH tunnel", type: "switch", default: false }),
    field({ name: "ssh_host", label: "SSH host", type: "text", dependsOn: "ssh" }),
    field({ name: "path", label: "File path", type: "text", upload: true }),
  ],
}

function renderForm(overrides: Partial<Parameters<typeof NewSourceForm>[0]> = {}) {
  const props = {
    source: SOURCE,
    error: null,
    onBack: vi.fn(),
    onCreated: vi.fn(),
    onError: vi.fn(),
    ...overrides,
  }
  render(
    <TooltipProvider>
      <NewSourceForm {...props} />
    </TooltipProvider>,
  )
  return props
}

const lastConfig = () => vi.mocked(createSource).mock.calls.at(-1)?.[0]

describe("NewSourceForm", () => {
  beforeEach(() => {
    vi.mocked(createSource)
      .mockReset()
      .mockResolvedValue({ id: "s1" } as never)
    vi.mocked(uploadWarehouseFile).mockReset()
  })

  it("submits the typed name, description, prefix and field values", async () => {
    const props = renderForm()
    const name = screen.getByLabelText("Source name")
    await userEvent.clear(name)
    await userEvent.type(name, "prod")
    await userEvent.type(screen.getByLabelText(/Description/), "Main db")
    await userEvent.type(screen.getByLabelText(/Table prefix/), "pg")
    await userEvent.type(screen.getByLabelText("Host"), "db.local")
    await userEvent.click(screen.getByRole("button", { name: "Next" }))
    expect(lastConfig()).toEqual({
      source_type: "postgres",
      name: "prod",
      description: "Main db",
      prefix: "pg",
      config: { host: "db.local", sslmode: "prefer", ssh: false },
    })
    expect(props.onCreated).toHaveBeenCalledWith({ id: "s1" })
  })

  it("sends the option chosen in a select field", async () => {
    renderForm()
    await userEvent.selectOptions(screen.getByRole("combobox"), "require")
    await userEvent.click(screen.getByRole("button", { name: "Next" }))
    expect(lastConfig()?.config).toMatchObject({ sslmode: "require" })
  })

  it("a switch field reveals its dependent field and sends its state", async () => {
    renderForm()
    expect(screen.queryByLabelText("SSH host")).toBeNull()
    await userEvent.click(screen.getByRole("checkbox", { name: "Connect through an SSH tunnel" }))
    await userEvent.type(screen.getByLabelText("SSH host"), "bastion")
    await userEvent.click(screen.getByRole("button", { name: "Next" }))
    expect(lastConfig()?.config).toMatchObject({ ssh: true, ssh_host: "bastion" })
  })

  it("an uploaded file fills its field with the stored path", async () => {
    vi.mocked(uploadWarehouseFile).mockResolvedValue({ path: "/stored/sales.csv", filename: "x" })
    renderForm()
    const file = new File(["a,b"], "sales.csv", { type: "text/csv" })
    await userEvent.upload(screen.getByLabelText("Upload file"), file)
    expect(await screen.findByText("Uploaded sales.csv")).toBeInTheDocument()
    expect(screen.getByLabelText("File path")).toHaveValue("/stored/sales.csv")
  })

  it("Next is disabled without a name; Cancel and Back leave the form", async () => {
    const props = renderForm()
    await userEvent.clear(screen.getByLabelText("Source name"))
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled()
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }))
    await userEvent.click(screen.getByRole("button", { name: "Back" }))
    expect(props.onBack).toHaveBeenCalledTimes(2)
  })
})
