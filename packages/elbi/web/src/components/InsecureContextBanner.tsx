import { ShieldAlert } from "lucide-react"

/**
 * Says so when the page is not served over TLS.
 *
 * The browser withholds a set of APIs outside a secure context, so a deployment on plain
 * HTTP is quietly a different product: copy buttons do nothing, and anything needing
 * `crypto.subtle` is unavailable. Worth a banner on its own.
 *
 * The compliance point is the sharper one. HIPAA's Security Rule makes encryption of ePHI
 * in transit a required implementation as of May 2026 (including between systems inside
 * one network) and PCI DSS 4.0 requirement 4.2.1 is non-addressable. A deployment
 * reachable only from an internal network is still in scope for both. So this is not a
 * hardening suggestion; it says the deployment is misconfigured.
 *
 * Not dismissible. localhost is a secure context, so nobody evaluating the product on a
 * laptop ever sees this; anyone who does is reaching a real deployment over plaintext, and
 * that should stay visible until it is fixed.
 */
export function InsecureContextBanner() {
  // localhost is a secure context, so a laptop never sees this.
  if (window.isSecureContext) return null

  return (
    <div
      role="alert"
      className="flex items-start gap-2 border-b border-amber-300 bg-amber-50 px-4 py-2 text-xs text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200"
    >
      <ShieldAlert aria-hidden className="mt-0.5 size-4 shrink-0" />
      <p>
        <span className="font-medium">This page is not served over HTTPS.</span> Your browser
        withholds features outside a secure context, so copying to the clipboard and anything using
        the Web Crypto API will not work. If this deployment handles regulated data, encryption in
        transit is a requirement rather than a recommendation: terminate TLS at your load balancer
        or ingress.
      </p>
    </div>
  )
}
