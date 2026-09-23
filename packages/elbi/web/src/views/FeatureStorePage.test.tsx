// Pins the feature-view dialog's per-feature type picker: a chosen type reaches the
// define request, and the "type" placeholder choice sends no dtype at all.

import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/lib/features", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/features")>()),
  listFeatureViews: vi.fn(async () => []),
  listEntities: vi.fn(async () => [
    { name: "user", joinKey: "user_id", valueType: "string", description: null },
  ]),
  listTrainingSets: vi.fn(async () => []),
  defineFeatureView: vi.fn(async () => ({})),
}))

import { defineFeatureView } from "@/lib/features"
import { FeatureStorePage } from "./FeatureStorePage"

beforeEach(() => {
  vi.mocked(defineFeatureView).mockClear()
})

afterEach(() => {
  vi.clearAllMocks()
})

async function openDialogWithOneFeature(user: ReturnType<typeof userEvent.setup>) {
  render(
    <MemoryRouter>
      <FeatureStorePage />
    </MemoryRouter>,
  )
  await user.click(await screen.findByRole("button", { name: "Feature view" }))
  const dialog = await screen.findByRole("dialog")
  await user.type(within(dialog).getByLabelText("Name"), "user_stats")
  await user.click(within(dialog).getByRole("button", { name: "user" }))
  await user.type(within(dialog).getByLabelText("Source derivation"), "user_activity")
  await user.click(within(dialog).getByRole("button", { name: "Add feature" }))
  await user.type(within(dialog).getByPlaceholderText("clicks_7d"), "clicks_7d")
  return dialog
}

async function chooseType(
  user: ReturnType<typeof userEvent.setup>,
  dialog: HTMLElement,
  type: string,
) {
  const picker = within(dialog).getByRole("combobox")
  expect(picker).toHaveAccessibleName("Feature type")
  // jsdom only: Radix Select closes on a window blur, which jsdom fires at pointerdown
  // when nothing holds focus.
  picker.focus()
  await user.click(picker)
  await user.click(await screen.findByRole("option", { name: type }))
}

describe("FeatureStorePage feature-view dialog", () => {
  it("sends the chosen feature type", async () => {
    const user = userEvent.setup()
    const dialog = await openDialogWithOneFeature(user)
    await chooseType(user, dialog, "integer")
    await user.click(within(dialog).getByRole("button", { name: "Define" }))
    expect(defineFeatureView).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "user_stats",
        entities: ["user"],
        source: "user_activity",
        features: [{ name: "clicks_7d", dtype: "integer" }],
      }),
    )
  })

  it("sends no dtype once the type is set back to the placeholder", async () => {
    const user = userEvent.setup()
    const dialog = await openDialogWithOneFeature(user)
    await chooseType(user, dialog, "integer")
    await chooseType(user, dialog, "type")
    await user.click(within(dialog).getByRole("button", { name: "Define" }))
    expect(defineFeatureView).toHaveBeenCalledWith(
      expect.objectContaining({ features: [{ name: "clicks_7d" }] }),
    )
  })
})
