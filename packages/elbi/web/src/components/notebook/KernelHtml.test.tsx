import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { KernelHtml } from "./KernelHtml"

/**
 * The boundary where kernel output becomes DOM.
 *
 * Each vector below was verified to execute through a bare `innerHTML` before this
 * component existed, so these are the payloads the sanitizer has to stop rather than a
 * guess at what one might look like. `<script>` is here for the opposite reason: it
 * never executed that way, and the reassurance that it doesn't is what makes people
 * stop looking at the ones that do.
 */
describe("KernelHtml", () => {
  const html = (markup: string) => {
    const { container } = render(<KernelHtml markup={markup} />)
    return container.innerHTML
  }

  it("strips the handlers that actually fire through innerHTML", () => {
    const out = html(
      `<img src="x" onerror="window.__pwned = true">` +
        `<svg><animate onbegin="window.__pwned = true" attributeName="x" dur="1s"/></svg>` +
        `<div onclick="window.__pwned = true">click</div>` +
        `<body onload="window.__pwned = true">`,
    )
    expect(out).not.toContain("onerror")
    expect(out).not.toContain("onbegin")
    expect(out).not.toContain("onclick")
    expect(out).not.toContain("onload")
    expect(out).not.toContain("__pwned")
  })

  it("strips scripts and script-bearing URLs", () => {
    const out = html(
      `<script>window.__pwned = true</script>` +
        `<a href="javascript:window.__pwned=true">go</a>` +
        `<iframe src="javascript:window.__pwned=true"></iframe>`,
    )
    expect(out).not.toContain("<script")
    expect(out.toLowerCase()).not.toContain("javascript:")
  })

  it("keeps a DataFrame's table, its styling and its text", () => {
    const out = html(
      `<table class="dataframe" style="border: 1px solid"><thead><tr><th>region</th></tr>` +
        `</thead><tbody><tr><td>west</td></tr></tbody></table>`,
    )
    expect(out).toContain("<table")
    expect(out).toContain('class="dataframe"')
    expect(out).toContain("border: 1px solid")
    expect(screen.getByText("west")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("keeps an SVG plot, which is what matplotlib returns", () => {
    const out = html(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">` +
        `<path d="M0 0 L10 10" stroke="black"/><text>label</text></svg>`,
    )
    expect(out).toContain("<svg")
    expect(out).toContain("<path")
    expect(out).toContain('d="M0 0 L10 10"')
    expect(screen.getByText("label")).toBeInTheDocument()
  })

  it("defuses markup that arrived as data inside a cell", () => {
    // A warehouse column holding markup, escaped by pandas as it built the table.
    // Whether the entity stays text or is re-parsed is the host parser's business --
    // a browser keeps it as text, jsdom does not -- so the invariant asserted here is
    // the one that holds either way: nothing executable survives.
    const { container } = render(
      <KernelHtml markup="<table><tr><td>&lt;img onerror=alert(1)&gt;</td></tr></table>" />,
    )
    expect(container.innerHTML).not.toContain("onerror")
    expect(container.innerHTML).not.toContain("alert(1)")
  })

  it("replaces its content when the output changes, leaving nothing behind", () => {
    const { container, rerender } = render(<KernelHtml markup="<p>first</p>" />)
    expect(container.textContent).toBe("first")
    rerender(<KernelHtml markup="<p>second</p>" />)
    expect(container.textContent).toBe("second")
  })

  it("contains output that tries to lay itself over the application", () => {
    // CSS survives sanitizing on purpose -- a DataFrame's formatting rides on `style`,
    // and CSS cannot execute. What it can do is escape: a fixed, viewport-sized element
    // paints over the app and takes the clicks meant for it. Containment on the host is
    // what bounds that, so this asserts the host carries it rather than asserting on a
    // painted result jsdom does not compute.
    const { container } = render(
      <KernelHtml markup='<div style="position:fixed;top:0;width:100vw;height:100vh">x</div>' />,
    )
    const host = container.firstElementChild as HTMLElement
    expect(host.style.contain).toBe("paint")
    // The declaration is deliberately left alone; the bound is structural, not a filter.
    expect(container.innerHTML).toContain("position:fixed")
  })
})
