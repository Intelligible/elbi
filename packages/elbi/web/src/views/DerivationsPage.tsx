import { FileCheck2, Search, ShieldCheck } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { type Column, DataTable } from "@/components/ui/data-table"
import { Input } from "@/components/ui/input"
import { VerdictBadge } from "@/components/VerdictBadge"
import { type DerivationSummary, getDerivations } from "@/lib/chat"
import { EMPTY } from "@/lib/utils"

export function DerivationsPage() {
  const navigate = useNavigate()
  const [derivations, setDerivations] = useState<DerivationSummary[] | null>(null)
  const [q, setQ] = useState("")

  useEffect(() => {
    getDerivations().then(setDerivations)
  }, [])

  const filtered = useMemo(() => {
    if (!derivations) return []
    const term = q.trim().toLowerCase()
    if (!term) return derivations
    return derivations.filter(
      (d) => d.name.toLowerCase().includes(term) || (d.question ?? "").toLowerCase().includes(term),
    )
  }, [derivations, q])

  if (derivations === null) return <SceneSkeleton />

  const columns: Column<DerivationSummary>[] = [
    {
      key: "name",
      title: "Name",
      width: "34%",
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (d) => (
        <span className="flex items-center gap-2">
          <ShieldCheck
            className={
              d.verdict === "sound"
                ? "size-4 shrink-0 text-verified"
                : "size-4 shrink-0 text-text-tertiary"
            }
          />
          <span className="truncate font-mono text-[0.8125rem] font-medium text-foreground">
            {d.name}
          </span>
        </span>
      ),
    },
    {
      key: "question",
      title: "Description",
      render: (d) => <span className="line-clamp-1 text-text-secondary">{d.question}</span>,
    },
    {
      key: "source",
      title: "Source",
      width: 96,
      render: (d) => (
        <span className="rounded-full border border-border px-2 py-0.5 text-[0.6875rem] font-medium uppercase tracking-wide text-text-tertiary">
          {d.origin === "repo" ? "Repo" : "Chat"}
        </span>
      ),
    },
    {
      key: "verdict",
      title: "Verdict",
      width: 140,
      render: (d) =>
        d.verdict ? (
          <VerdictBadge verdict={d.verdict} />
        ) : (
          <span className="text-text-tertiary">{EMPTY}</span>
        ),
    },
    {
      key: "created",
      title: "Created",
      width: 128,
      align: "right",
      sorter: (a, b) => a.createdAt.localeCompare(b.createdAt),
      render: (d) => (
        <span className="text-text-tertiary tabular-nums">
          {new Date(d.createdAt).toLocaleDateString()}
        </span>
      ),
    },
  ]

  return (
    <Scene>
      <SceneHeader
        icon={<FileCheck2 className="size-5" />}
        title="Derivations"
        description="Durable, cached derivations: authored in your repo or from a verified answer."
      />
      <SceneBody width="full" canvas>
        <DataTable
          columns={columns}
          data={filtered}
          rowKey={(d) => d.name}
          onRowClick={(d) => navigate(`/derivations/${encodeURIComponent(d.name)}`)}
          empty={
            q
              ? "No derivations match your search."
              : "None yet: author one and sync, or ask a question, and it appears here."
          }
          toolbar={
            <div className="relative max-w-xs">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-text-tertiary" />
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search derivations"
                className="h-8 pl-8"
              />
            </div>
          }
        />
      </SceneBody>
    </Scene>
  )
}
