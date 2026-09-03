import { describe, expect, it } from "vitest"

import { EVENT_TYPE_LABELS, type NotificationItem, notificationRoute } from "./notifications"

const item = (patch: Partial<NotificationItem>): NotificationItem => ({
  id: "n1",
  at: "2026-07-30T00:00:00+00:00",
  eventType: "run.failed",
  title: "",
  body: "",
  targetType: "",
  targetId: "",
  verdict: null,
  readAt: null,
  ...patch,
})

describe("notificationRoute", () => {
  it("routes each target type to its page", () => {
    expect(notificationRoute(item({ targetType: "metric_monitor", targetId: "m1" }))).toBe(
      "/monitors",
    )
    expect(notificationRoute(item({ targetType: "orchestration_run", targetId: "r1" }))).toBe(
      "/orchestration",
    )
    expect(notificationRoute(item({ targetType: "model_version", targetId: "churn/v3" }))).toBe(
      "/models/churn",
    )
    expect(notificationRoute(item({ targetType: "model", targetId: "churn" }))).toBe(
      "/models/churn",
    )
    expect(notificationRoute(item({ targetType: "feature_view", targetId: "users" }))).toBe(
      "/features/users",
    )
  })

  it("escapes names that would break the path", () => {
    expect(notificationRoute(item({ targetType: "model", targetId: "a/b" }))).toBe("/models/a%2Fb")
  })

  it("returns null when there is nowhere to go", () => {
    expect(notificationRoute(item({ targetType: "" }))).toBeNull()
    expect(notificationRoute(item({ targetType: "unknown", targetId: "x" }))).toBeNull()
  })

  it("falls back to the list page when the target id is empty", () => {
    expect(notificationRoute(item({ targetType: "model", targetId: "" }))).toBe("/models")
    expect(notificationRoute(item({ targetType: "feature_view", targetId: "" }))).toBe("/features")
  })
})

describe("EVENT_TYPE_LABELS", () => {
  it("covers the backend event vocabulary", () => {
    // Mirrors notifications.EVENT_TYPES on the backend; a new event type needs
    // a label here so the preferences UI and inbox rows can name it.
    const backendTypes = [
      "model_version.created",
      "training.failed",
      "metric.anomaly_detected",
      "metric.recovered",
      "run.failed",
      "run.slow",
      "drift.detected",
      "feature.drift_detected",
      "feature.expectations_failed",
    ]
    expect(Object.keys(EVENT_TYPE_LABELS).sort()).toEqual(backendTypes.sort())
    for (const meta of Object.values(EVENT_TYPE_LABELS)) {
      expect(meta.label).toBeTruthy()
      expect(meta.description).toBeTruthy()
    }
  })
})
