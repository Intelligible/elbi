import { useEffect, useState } from "react"
import { Link } from "react-router-dom"

import { derivationProvenance, type Provenance } from "@/lib/dashboards"

/** What a derivation reads. `null` until fetched, or if it cannot be. */
function useProvenance(derivation: string | undefined): Provenance | null {
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

  return upstream
}

function DerivationLink({ name }: { name: string }) {
  return (
    <Link className="text-primary underline-offset-2 hover:underline" to={`/derivations/${name}`}>
      {name}
    </Link>
  )
}

/**
 * The trail from a tile back to where its numbers came from, as a tab of the editor.
 *
 * A tile shows a figure; the derivation is what computed it and the datasets are what
 * that read. Following it is how someone checks a number rather than trusting it. A tab
 * rather than a panel over the fields: it costs a click, and buys room to say what each
 * link is instead of compressing the chain onto one line — and nothing is fetched until
 * someone asks for it.
 */
export function ProvenanceTab({ derivation }: { derivation: string }) {
  const upstream = useProvenance(derivation)

  if (!derivation) {
    return (
      <p className="text-sm text-text-tertiary">
        This tile carries its own content, so it reads nothing.
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <p className="text-sm font-medium">Computed by</p>
        <p className="text-sm">
          <DerivationLink name={derivation} />
        </p>
        <p className="text-xs text-text-tertiary">
          Its page carries the source, the claim it makes, the oracle's verdict and every result it
          has returned.
        </p>
      </div>
      {upstream && (upstream.derivation.length > 0 || upstream.dataset.length > 0) ? (
        <div className="space-y-1">
          <p className="text-sm font-medium">Which reads</p>
          <ul className="space-y-1 text-sm">
            {upstream.derivation.map((name) => (
              <li key={name}>
                <DerivationLink name={name} />{" "}
                <span className="text-text-tertiary">— a derivation</span>
              </li>
            ))}
            {upstream.dataset.map((name) => (
              <li key={name}>
                <Link className="text-primary underline-offset-2 hover:underline" to="/warehouse">
                  {name}
                </Link>{" "}
                <span className="text-text-tertiary">— a warehouse table</span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="text-xs text-text-tertiary">
          Nothing upstream: this derivation reads no other derivation or dataset.
        </p>
      )}
    </div>
  )
}
