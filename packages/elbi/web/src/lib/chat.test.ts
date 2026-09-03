import { describe, expect, it } from "vitest"

import { certificatePdfUrl, certificateUrl } from "./chat"

describe("certificate URLs", () => {
  it("builds the JSON certificate URL", () => {
    expect(certificateUrl("churn_risk")).toBe("/api/certificates/churn_risk")
  })

  it("builds the PDF certificate URL", () => {
    expect(certificatePdfUrl("churn_risk")).toBe("/api/certificates/churn_risk/pdf")
  })

  it("encodes names with special characters", () => {
    expect(certificateUrl("a/b c")).toBe("/api/certificates/a%2Fb%20c")
    expect(certificatePdfUrl("a/b c")).toBe("/api/certificates/a%2Fb%20c/pdf")
  })
})
