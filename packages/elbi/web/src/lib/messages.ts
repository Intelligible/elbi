import type { UIMessage } from "ai"

import type { StoredMessage } from "@/lib/chat"
import { uuid } from "@/lib/utils"

// Rebuild the chat view's messages from what the server persisted, so a resumed (or
// background-completed) conversation renders with its answers and their verification
// receipts, not just plain text. The assistant's result payload is carried as a
// data-result part, which the Turn renderer turns back into a verdict and receipt.
export function storedToUIMessages(stored: StoredMessage[]): UIMessage[] {
  return stored.map((m) => ({
    // Carry the server's message id so per-message actions (feedback) target the row;
    // fall back to a random id for anything the server did not persist an id for.
    id: m.id ?? uuid(),
    role: m.role === "user" ? ("user" as const) : ("assistant" as const),
    metadata: { feedback: m.feedback ?? null },
    parts: [
      ...(m.content ? [{ type: "text" as const, text: m.content }] : []),
      ...(m.result && m.role !== "user" ? [{ type: "data-result" as const, data: m.result }] : []),
    ],
  }))
}
