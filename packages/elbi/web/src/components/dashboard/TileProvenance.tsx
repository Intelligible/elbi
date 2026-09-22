import { useEffect, useState } from "react"
import { Link } from "react-router-dom"

import { derivationProvenance, type Provenance } from "@/lib/dashboards"

/**
 * The trail from a tile back to where its numbers came from.
 *
 * A tile shows a figure; the derivation is what computed it and the dataset is what
 * that read. Following it is how someone checks a number instead of trusting it, so
 * it belongs beside the binding rather than three pages away.
 */
export function TileProvenance({ derivation }: { derivation: string }) {
  const [upstream, setUpstream] = useState<Provenance | null>(null)

  useEffect(() => {
    let live = true
    if (!derivation) {
      setUpstream(null)
      return
    }
    void derivationProvenance(derivation)
      .then((p) => live && setUpstream(p))
      .catch(() => live && setUpstream(null))
    return () => {
      live = false
    }
  }, [derivation])

  if (!derivation) return null

  return (
    <div className="space-y-1 rounded-md border border-border bg-surface-secondary px-3 py-2">
      <p className="text-xs text-text-tertiary">Where this tile's numbers come from</p>
      <p className="text-sm">
        <Link
          className="text-primary underline-offset-2 hover:underline"
          to={`/derivations/${derivation}`}
        >
          {derivation}
        </Link>
        <span className="text-text-tertiary"> — source, claim and result history</span>
      </p>
      {upstream && (upstream.derivation.length > 0 || upstream.dataset.length > 0) ? (
        <p className="text-xs text-text-tertiary">
          reads{" "}
          {upstream.derivation.map((name, i) => (
            <span key={name}>
              {i > 0 ? ", " : ""}
              <Link
                className="text-primary underline-offset-2 hover:underline"
                to={`/derivations/${name}`}
              >
                {name}
              </Link>
            </span>
          ))}
          {upstream.derivation.length > 0 && upstream.dataset.length > 0 ? ", " : ""}
          {upstream.dataset.map((name, i) => (
            <span key={name}>
              {i > 0 ? ", " : ""}
              <Link className="text-primary underline-offset-2 hover:underline" to="/warehouse">
                {name}
              </Link>
              <span> (dataset)</span>
            </span>
          ))}
        </p>
      ) : null}
    </div>
  )
}
