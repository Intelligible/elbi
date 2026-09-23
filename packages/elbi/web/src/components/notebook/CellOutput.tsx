import { Check, ChevronsDownUp, ChevronsUpDown, Copy, Download, Maximize2 } from "lucide-react"
import { type ReactNode, useState } from "react"
import { VegaEmbed } from "react-vega"
import stripAnsi from "strip-ansi"
import type { VisualizationSpec } from "vega-embed"
import { IconButton } from "@/components/app/IconButton"
import { KernelHtml } from "@/components/notebook/KernelHtml"
import { NotebookMarkdown } from "@/components/notebook/NotebookMarkdown"
import { WidgetView } from "@/components/notebook/WidgetView"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { useRowKeys } from "@/hooks/useRowKeys"
import type { Output } from "@/lib/notebooks"
import { copyText } from "@/lib/utils"
import { WIDGET_MIME } from "@/lib/widget-mime"

// Renders a code cell's nbformat outputs: stream text, the last-expression result, mid-cell
// displays, and errors. For a MIME bundle (result/display) it picks the richest form it can
// render (a Vega chart, a DataFrame's HTML, an image, markdown, or the text/plain fallback
// every output carries) mirroring how Jupyter's frontend chooses a representation.

function asText(value: unknown): string {
  return Array.isArray(value) ? value.join("") : String(value ?? "")
}

// Wrap a `text/latex` payload in `$$...$$` so KaTeX renders it as display math, leaving
// it untouched when the kernel already supplied `$`, `$$`, or `\[ \]` delimiters.
function asDisplayMath(value: unknown): string {
  const text = asText(value).trim()
  const delimited =
    /^\$\$[\s\S]*\$\$$/.test(text) ||
    /^\$[\s\S]*\$$/.test(text) ||
    /^\\\[[\s\S]*\\\]$/.test(text) ||
    /^\\\([\s\S]*\\\)$/.test(text)
  return delimited ? text : `$$${text}$$`
}

function Bundle({ data }: { data: Record<string, unknown> }) {
  const widget = data[WIDGET_MIME] as { model_id?: string } | undefined
  if (widget?.model_id) {
    return <WidgetView modelId={widget.model_id} />
  }
  const vega = data["application/vnd.vegalite.v5+json"] ?? data["application/vnd.vega.v5+json"]
  if (vega) {
    return (
      <div className="overflow-x-auto">
        <VegaEmbed spec={vega as VisualizationSpec} options={{ actions: false }} />
      </div>
    )
  }
  if (data["text/html"]) {
    return (
      <KernelHtml markup={asText(data["text/html"])} className="nb-html overflow-x-auto text-sm" />
    )
  }
  if (data["image/svg+xml"]) {
    return <KernelHtml markup={asText(data["image/svg+xml"])} className="overflow-x-auto" />
  }
  for (const mime of ["image/png", "image/jpeg"] as const) {
    if (data[mime]) {
      return (
        <img
          src={`data:${mime};base64,${asText(data[mime])}`}
          alt="cell output"
          className="max-w-full"
        />
      )
    }
  }
  if (data["text/markdown"]) {
    return <NotebookMarkdown>{asText(data["text/markdown"])}</NotebookMarkdown>
  }
  if (data["text/latex"]) {
    // A `text/latex` payload is a math document; render it through the same KaTeX path
    // as markdown, wrapping in `$$` display delimiters unless the kernel already did.
    return <NotebookMarkdown>{asDisplayMath(data["text/latex"])}</NotebookMarkdown>
  }
  return (
    <pre className="whitespace-pre-wrap break-words font-mono text-compact leading-relaxed">
      {asText(data["text/plain"])}
    </pre>
  )
}

function OneOutput({ output }: { output: Output }) {
  if (output.output_type === "stream") {
    const isErr = output.name === "stderr"
    return (
      <pre
        className={`whitespace-pre-wrap break-words font-mono text-compact leading-relaxed ${
          isErr ? "text-danger" : "text-foreground/80"
        }`}
      >
        {asText(output.text)}
      </pre>
    )
  }
  if (output.output_type === "error") {
    // The maintained matcher, not a hand-rolled one: a traceback carries more than
    // colour -- progress bars leave cursor moves and erase-line sequences behind, and
    // rich/pytest emit OSC hyperlinks. Matching only SGR left those bytes on screen.
    const trace = stripAnsi((output.traceback ?? []).join("\n"))
    return (
      <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded-md bg-danger-tint p-3 font-mono text-xs leading-relaxed text-danger">
        {trace || `${output.ename}: ${output.evalue}`}
      </pre>
    )
  }
  return <Bundle data={output.data ?? {}} />
}

// The plain text of one output, for copying: stream text, an error's trace, or the
// text/plain a MIME bundle always carries.
function outputText(output: Output): string {
  if (output.output_type === "stream") return asText(output.text)
  if (output.output_type === "error") {
    // The maintained matcher, not a hand-rolled one: a traceback carries more than
    // colour -- progress bars leave cursor moves and erase-line sequences behind, and
    // rich/pytest emit OSC hyperlinks. Matching only SGR left those bytes on screen.
    const trace = stripAnsi((output.traceback ?? []).join("\n"))
    return trace || `${output.ename}: ${output.evalue}`
  }
  return asText(output.data?.["text/plain"])
}

export function CellOutputs({ outputs }: { outputs: Output[] }) {
  const rowKey = useRowKeys()
  const [collapsed, setCollapsed] = useState(false)
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)
  if (outputs.length === 0) return null

  // A DataFrame renders as text/html; a chart/figure as an image. Surface a "full view" for the
  // former and a download for the latter: the affordances a DS reaches for.
  const tableHtml = outputs.map((o) => o.data?.["text/html"]).find(Boolean)
  const chartPng = outputs.map((o) => o.data?.["image/png"]).find(Boolean)

  const copy = () => {
    void copyText(outputs.map(outputText).join("").trimEnd()).then((ok) => {
      if (!ok) return
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    })
  }

  return (
    <div className="group/out relative flex border-t border-border/60 bg-muted/25">
      {/* An "Out" rail mirrors the code cell's run gutter, so a result reads as the
          cell's output rather than floating text; clicking it collapses the output. */}
      <Button
        variant="ghost"
        onClick={() => setCollapsed((c) => !c)}
        title={collapsed ? "Show output" : "Hide output"}
        className="h-auto w-12 flex-col justify-start gap-1 rounded-none px-0 pt-2.5 font-mono has-[>svg]:px-0 text-3xs font-normal uppercase tracking-wide text-text-tertiary/60"
      >
        Out
        {collapsed ? (
          <ChevronsUpDown className="size-3.5" />
        ) : (
          <ChevronsDownUp className="size-3.5 opacity-0 transition group-hover/out:opacity-100" />
        )}
      </Button>

      {collapsed ? (
        <Button
          variant="link"
          onClick={() => setCollapsed(false)}
          className="h-auto flex-1 justify-start p-0 py-2 text-xs font-normal text-text-tertiary italic hover:text-foreground"
        >
          {outputs.length} output{outputs.length === 1 ? "" : "s"} hidden: click to show
        </Button>
      ) : (
        <div className="min-w-0 flex-1 space-y-1 py-2.5 pr-14">
          {outputs.map((output) => (
            <OneOutput key={rowKey(output)} output={output} />
          ))}
        </div>
      )}

      {/* Hover toolbar: copy, plus full-view for a table and download for a chart. */}
      <div className="absolute right-2 top-2 flex items-center gap-0.5 opacity-0 transition group-hover/out:opacity-100">
        {tableHtml ? (
          <OutputAction label="Open table full view" onClick={() => setExpanded(true)}>
            <Maximize2 className="size-3.5" />
          </OutputAction>
        ) : null}
        {chartPng ? (
          <OutputAction label="Download chart" asChild>
            <a href={`data:image/png;base64,${asText(chartPng)}`} download="chart.png">
              <Download className="size-3.5" />
            </a>
          </OutputAction>
        ) : null}
        <OutputAction label="Copy output" onClick={copy}>
          {copied ? <Check className="size-3.5 text-success" /> : <Copy className="size-3.5" />}
        </OutputAction>
      </div>

      {tableHtml ? (
        <Dialog open={expanded} onOpenChange={setExpanded}>
          <DialogContent className="max-h-[85vh] max-w-5xl overflow-hidden">
            <DialogHeader>
              <DialogTitle>Output</DialogTitle>
            </DialogHeader>
            <KernelHtml markup={asText(tableHtml)} className="nb-html overflow-auto text-sm" />
          </DialogContent>
        </Dialog>
      ) : null}
    </div>
  )
}

function OutputAction({
  label,
  onClick,
  asChild,
  children,
}: {
  label: string
  onClick?: () => void
  asChild?: boolean
  children: ReactNode
}) {
  return (
    <IconButton
      label={label}
      variant="outline"
      size="icon-xs"
      asChild={asChild}
      onClick={onClick}
      className="text-text-tertiary"
    >
      {children}
    </IconButton>
  )
}
