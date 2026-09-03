import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { UIMessage } from "ai"
import { describe, expect, it } from "vitest"

import { Turn } from "./ChatView"

function assistantMessage(verified: boolean): UIMessage {
  return {
    id: "m1",
    role: "assistant",
    parts: [
      {
        type: "data-result",
        data: {
          verified,
          verdict: verified ? "sound" : "unverified",
          checks: [{ name: "effect", verdict: "sound", detail: "holds" }],
          assumptions: [],
          spec: { derivation: "eff" },
          data_hash: null,
        },
      },
    ],
  } as unknown as UIMessage
}

async function openReceipt() {
  await userEvent.click(screen.getByRole("button", { name: /Verification/ }))
}

describe("Turn certificate link", () => {
  it("offers a certificate download when the result is verified", async () => {
    render(<Turn message={assistantMessage(true)} busy={false} />)
    await openReceipt()
    const link = screen.getByRole("link", { name: "Download certificate" })
    expect(link.getAttribute("href")).toBe("/api/certificates/eff")
  })

  it("omits the certificate download when the result is not verified", async () => {
    render(<Turn message={assistantMessage(false)} busy={false} />)
    await openReceipt()
    expect(screen.queryByRole("link", { name: "Download certificate" })).toBeNull()
  })
})
