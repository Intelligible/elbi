import DOMPurify from "dompurify"
import { useEffect, useRef } from "react"

/**
 * Renders kernel-produced markup (a DataFrame's `_repr_html_`, matplotlib's SVG) as
 * sanitized DOM.
 *
 * DOMPurify runs with its defaults, which drop every event handler and `javascript:`
 * URL while keeping HTML, SVG, MathML, `style` and `class`. It returns a DOM fragment
 * rather than a string, so the markup is parsed once.
 *
 * The host sets `contain: paint`, which makes it the containing block for fixed
 * descendants and clips painting to its own box.
 */
export function KernelHtml({ markup, className }: { markup: string; className?: string }) {
  const host = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const node = host.current
    if (!node) return
    node.replaceChildren(DOMPurify.sanitize(markup, { RETURN_DOM_FRAGMENT: true }))
    return () => node.replaceChildren()
  }, [markup])

  // `contain: paint` is the layout half above; Tailwind has no utility for it.
  return <div ref={host} className={className} style={{ contain: "paint" }} />
}
