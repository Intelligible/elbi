import { AlertTriangle, FileCheck2, NotebookPen } from "lucide-react"
import { useEffect, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { Streamdown } from "streamdown"
import {
  Scene,
  SceneBody,
  SceneHeader,
  ScenePanelLabel,
  SceneSection,
  SceneSkeleton,
} from "@/components/Scene"
import { Button } from "@/components/ui/button"
import { estimateText, VerdictBadge } from "@/components/VerdictBadge"

import {
  type CertifiedRun,
  certificatePdfUrl,
  certificateUrl,
  type DerivationDetail,
  derivationExportUrl,
  getDerivation,
  getDerivationHistory,
} from "@/lib/chat"
import { notebookFromDerivation } from "@/lib/notebooks"

export function DerivationDetailPage() {
  const { name = "" } = useParams()
  const navigate = useNavigate()
  const [d, setD] = useState<DerivationDetail | null | "missing">(null)
  useEffect(() => {
    getDerivation(name).then((r) => setD(r ?? "missing"))
  }, [name])

  if (d === null) return <SceneSkeleton />

  if (d === "missing") {
    return (
      <Scene>
        <SceneHeader backTo="/derivations" backLabel="Derivations" title="Not found" />
        <SceneBody>
          <p className="text-sm text-text-secondary">
            There's no derivation named <span className="font-mono">{name}</span>.
          </p>
        </SceneBody>
      </Scene>
    )
  }

  return (
    <Scene>
      <SceneHeader
        backTo="/derivations"
        backLabel="Derivations"
        icon={<FileCheck2 className="size-5" />}
        mono
        title={d.name}
        description={d.question}
        badges={d.verdict && <VerdictBadge verdict={d.verdict} />}
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              notebookFromDerivation(d.name).then((r) => navigate(`/notebooks/${r.id}`))
            }
          >
            <NotebookPen className="size-4" /> Open in notebook
          </Button>
        }
      />
      <SceneBody
        aside={
          <>
            {d.verdict && (
              <ScenePanelLabel label="Verdict">
                <VerdictBadge verdict={d.verdict} />
              </ScenePanelLabel>
            )}
            {d.dataHash && (
              <ScenePanelLabel label="Data hash">
                <span className="font-mono text-xs break-all text-text-secondary">
                  {d.dataHash}
                </span>
              </ScenePanelLabel>
            )}
            {d.attestation && (
              <ScenePanelLabel label="Certificate">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" asChild>
                    <a
                      href={certificateUrl(d.name)}
                      download
                      aria-label="Download certificate as JSON"
                    >
                      JSON
                    </a>
                  </Button>
                  <Button variant="outline" size="sm" asChild>
                    <a
                      href={certificatePdfUrl(d.name)}
                      download
                      aria-label="Download certificate as PDF"
                    >
                      PDF
                    </a>
                  </Button>
                </div>
              </ScenePanelLabel>
            )}
            {/* Not gated on an attestation, unlike the certificate above: an
                uncertified derivation still exports, with a null certificate. */}
            <ScenePanelLabel label="Evidence">
              <Button variant="outline" size="sm" asChild>
                <a
                  href={derivationExportUrl(d.name)}
                  download
                  aria-label="Export this derivation's record: source, claim, verdict, result history, and certificate"
                >
                  Export record
                </a>
              </Button>
              <p className="mt-1.5 text-xs text-text-tertiary">
                Source, claim, verdict, result history, and certificate, as one file.
              </p>
            </ScenePanelLabel>
          </>
        }
      >
        {d.assumptions.length > 0 && (
          <div className="rounded-lg border border-[var(--caution)]/25 bg-caution-tint px-3.5 py-3">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-caution">
              <AlertTriangle className="size-3.5" /> Sound under stated assumptions
            </div>
            <ul className="mt-2 space-y-1 text-sm text-foreground/90">
              {d.assumptions.map((a) => (
                <li key={a} className="flex gap-2">
                  <span className="text-caution">•</span>
                  {a}
                </li>
              ))}
            </ul>
          </div>
        )}

        {d.narrative && (
          <SceneSection title="Finding">
            <div className="prose-sm max-w-none text-sm leading-relaxed text-foreground">
              <Streamdown>{d.narrative}</Streamdown>
            </div>
          </SceneSection>
        )}

        {d.rendered && (
          // Only a sound verdict earns the word "verified": a derivation with no claim is
          // never oracle-gated, so calling its output verified would assert a check that
          // never ran.
          <SceneSection title={d.verdict === "sound" ? "Verified output" : "Output"}>
            <div className="overflow-x-auto rounded-lg border border-border bg-card p-4">
              <Streamdown>{d.rendered}</Streamdown>
            </div>
          </SceneSection>
        )}

        {d.attestation?.checks && d.attestation.checks.length > 0 && (
          <SceneSection
            title="Verification"
            description={`${d.attestation.checks.length} checks run by the oracle`}
          >
            <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-card">
              {d.attestation.checks.map((c) => (
                <li key={c.name} className="flex gap-2.5 px-3.5 py-2.5 text-sm">
                  <span
                    className={
                      c.verdict === "sound"
                        ? "text-verified"
                        : c.verdict === "unsound"
                          ? "text-danger"
                          : "text-text-tertiary"
                    }
                  >
                    {c.verdict === "sound" ? "✓" : c.verdict === "unsound" ? "✗" : "◦"}
                  </span>
                  <span className="min-w-0">
                    <span className="font-medium text-foreground">{c.name}</span>
                    <span className="text-text-secondary">: {c.detail}</span>
                  </span>
                </li>
              ))}
            </ul>
          </SceneSection>
        )}

        <HistorySection name={d.name} />

        {d.claim && Object.keys(d.claim).length > 0 && (
          <SceneSection title="Claim">
            <pre className="overflow-x-auto rounded-lg border border-border bg-surface-secondary p-3 font-mono text-xs text-foreground">
              {JSON.stringify(d.claim, null, 2)}
            </pre>
          </SceneSection>
        )}

        {d.source && (
          <SceneSection title="Source">
            <pre className="overflow-x-auto rounded-lg border border-border bg-surface-secondary p-3 font-mono text-xs text-foreground">
              {d.source}
            </pre>
          </SceneSection>
        )}
      </SceneBody>
    </Scene>
  )
}

// The derivation's certified results over time, newest first. Each row is a distinct
// verified version; every estimate shown is the oracle's certified value. When a version's
// `changed` set is non-empty, its tags name which inputs moved the number since the
// previous (next-older) version.
function HistorySection({ name }: { name: string }) {
  const [versions, setVersions] = useState<CertifiedRun[] | null>(null)
  useEffect(() => {
    getDerivationHistory(name).then(setVersions)
  }, [name])

  // Hold the section back until the history arrives, so it never flashes the empty state.
  if (versions === null || versions.length === 0) return null

  return (
    <SceneSection
      title="Result history"
      description={`${versions.length} version${versions.length === 1 ? "" : "s"}`}
    >
      <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-card">
        {versions.map((v) => (
          <li key={v.derivationVersion} className="px-3.5 py-3">
            <div className="flex items-center gap-3">
              <span className="shrink-0 font-mono text-xs text-text-secondary">
                {v.shortVersion}
              </span>
              <VerdictBadge verdict={v.verdict} />
              <span className="min-w-0 flex-1 truncate text-sm tabular-nums">
                {estimateText(v)}
              </span>
              <span className="shrink-0 text-xs text-text-tertiary tabular-nums">
                {new Date(v.createdAt).toLocaleDateString()}
              </span>
            </div>
            {v.changed.length > 0 ? (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {v.changed.map((dim) => (
                  <span
                    key={dim}
                    className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-text-tertiary"
                  >
                    {changedLabel(dim)} changed
                  </span>
                ))}
              </div>
            ) : (
              <div className="mt-1.5 text-[11px] text-text-tertiary/80">
                first certified version
              </div>
            )}
          </li>
        ))}
      </ul>
    </SceneSection>
  )
}

// A short human label for each input dimension the diff reports. Unknown dimensions fall
// back to their raw name rather than being dropped, so a new backend dimension still shows.
function changedLabel(dim: string): string {
  const labels: Record<string, string> = {
    data: "data",
    code: "code",
    controls: "controls",
    params: "params",
    claim: "claim",
  }
  return labels[dim] ?? dim
}
