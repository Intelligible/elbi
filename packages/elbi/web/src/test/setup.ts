import "@testing-library/jest-dom/vitest"

import { cleanup } from "@testing-library/react"
import { afterEach } from "vitest"

// Unmount React trees between tests so queries never see a previous render's DOM.
afterEach(cleanup)

// This setup file runs for every test file regardless of `@vitest-environment`, including
// the node-environment ESLint rule tests, which have no DOM at all.
if (typeof window !== "undefined") {
  // jsdom implements no Pointer Events API, and Radix's Select and Dropdown open on pointer
  // events rather than clicks. Without these an interaction test throws instead of opening
  // the menu, which reads as a component bug and is not one.
  if (!Element.prototype.hasPointerCapture) {
    Element.prototype.hasPointerCapture = () => false
    Element.prototype.setPointerCapture = () => {}
    Element.prototype.releasePointerCapture = () => {}
  }

  // Radix measures a popover before positioning it; jsdom reports every element as zero-sized
  // and has no scrollIntoView at all.
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = () => {}
  }

  // jsdom has no matchMedia, and useTheme() asks the OS for its color scheme on every
  // render. Report light so the resolved theme is deterministic, and accept the listeners
  // the hook attaches while it is following the system.
  if (!window.matchMedia) {
    window.matchMedia = (query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList
  }

  // jsdom ships no ResizeObserver, and the conversation's stick-to-bottom scroller attaches
  // one to its viewport on mount. A no-op is enough: nothing in a test has a size to observe.
  if (!globalThis.ResizeObserver) {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver
  }
}
