import { expect, test } from "@playwright/test"

// Backend-independent smoke: the app shell mounts, the product navigation is reachable,
// and client-side routing works. Deeper flows (chat -> certified answer) need the full
// stack and belong in a stack-backed E2E suite.
test("app shell renders", async ({ page }) => {
  await page.goto("/")
  await expect(page.getByRole("img", { name: "Elbi" })).toBeVisible()
  await expect(page.getByRole("button", { name: "Browse" })).toBeVisible()
  await expect(page.getByRole("button", { name: "New analysis" })).toBeVisible()
})

test("browse navigation routes to a product area", async ({ page }) => {
  await page.goto("/")
  await page.getByRole("button", { name: "Browse" }).click()
  await expect(page.getByRole("link", { name: "Derivations" })).toBeVisible()
  await page.getByRole("link", { name: "Derivations" }).click()
  await expect(page).toHaveURL(/\/derivations$/)
})
