import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import type { DerivationDetail } from "@/lib/chat"

vi.mock("@/lib/notebooks", () => ({ notebookFromDerivation: vi.fn() }))

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  getDerivation: vi.fn(),
  getDerivationHistory: vi.fn(async () => []),
}))

import { getDerivation } from "@/lib/chat"
import { DerivationDetailPage } from "./DerivationDetailPage"

const DETAIL: DerivationDetail = {
  name: "eff",
  question: "does x move y?",
  verdict: "sound",
  dataHash: "abc123",
  createdAt: "2026-07-20T00:00:00Z",
  origin: "agent",
  conversationId: null,
  source: "def eff(ctx): ...",
  claim: null,
  serve: null,
  narrative: "",
  rendered: "",
  attestation: { checks: [{ name: "effect", verdict: "sound", detail: "holds" }] },
  assumptions: [],
}

const CLAIM = { x: "amount", y: "risk" }

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/derivations/eff"]}>
      <Routes>
        <Route path="/derivations/:name" element={<DerivationDetailPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe("DerivationDetailPage certificate panel", () => {
  it("offers JSON and PDF certificate downloads when attestation is present", async () => {
    vi.mocked(getDerivation).mockResolvedValue(DETAIL)
    renderPage()
    const json = await screen.findByRole("link", {
      name: "Download certificate as JSON",
    })
    expect(json.getAttribute("href")).toBe("/api/certificates/eff")
    expect(
      screen.getByRole("link", { name: "Download certificate as PDF" }).getAttribute("href"),
    ).toBe("/api/certificates/eff/pdf")
  })

  it("omits the certificate panel when there is no attestation", async () => {
    vi.mocked(getDerivation).mockResolvedValue({ ...DETAIL, attestation: null })
    renderPage()
    await screen.findByText("does x move y?")
    expect(screen.queryByRole("link", { name: "Download certificate as JSON" })).toBeNull()
  })
})

describe("DerivationDetailPage claim panel", () => {
  it("marks a claim with no verdict as not checked", async () => {
    vi.mocked(getDerivation).mockResolvedValue({ ...DETAIL, claim: CLAIM, verdict: null })
    renderPage()
    await screen.findByRole("heading", { name: "Claim" })
    expect(screen.getByText("Not checked by the oracle.")).toBeInTheDocument()
  })

  it("shows no not-checked note on a claim the oracle found sound", async () => {
    vi.mocked(getDerivation).mockResolvedValue({ ...DETAIL, claim: CLAIM, verdict: "sound" })
    renderPage()
    await screen.findByRole("heading", { name: "Claim" })
    expect(screen.queryByText("Not checked by the oracle.")).toBeNull()
  })
})

describe("DerivationDetailPage evidence export", () => {
  it("offers the record and, when there is output, the output as HTML", async () => {
    vi.mocked(getDerivation).mockResolvedValue({ ...DETAIL, rendered: "# Result" })
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole("button", { name: /Export/ }))
    expect(
      (await screen.findByRole("menuitem", { name: /record/ })).getAttribute("href"),
    ).toBe("/api/exports/derivations/eff")
    expect(
      screen.getByRole("menuitem", { name: /as an HTML page/ }).getAttribute("href"),
    ).toBe("/api/exports/derivations/eff/html")
  })

  it("offers no HTML export for a derivation with no output", async () => {
    vi.mocked(getDerivation).mockResolvedValue(DETAIL)
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole("button", { name: /Export/ }))
    await screen.findByRole("menuitem", { name: /record/ })
    expect(screen.queryByRole("menuitem", { name: /as an HTML page/ })).toBeNull()
  })
})
