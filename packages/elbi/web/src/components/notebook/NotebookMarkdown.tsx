import "katex/dist/katex.min.css"
import "highlight.js/styles/github-dark.css"

import Markdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import rehypeKatex from "rehype-katex"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import type { PluggableList } from "unified"

import { cn } from "@/lib/utils"

// Markdown for notebook surfaces, matching how Jupyter renders a markdown cell and a
// `text/markdown` / `text/latex` output: GitHub-flavored markdown (tables, task lists,
// fenced code) with syntax-highlighted code blocks (highlight.js) plus TeX math, so
// `$$\int$$` renders with KaTeX. The plugin lists are module-level constants so the
// editor does not rebuild the pipeline on every keystroke.
//
// Math is display-only: `$…$` would claim every dollar amount in a notebook, and
// "$2,692.89 of $4,254.31" silently rendering as an integrand is worse than inline TeX
// needing the doubled delimiter. This is a data workbench — currency is the common case.
const REMARK_PLUGINS: PluggableList = [remarkGfm, [remarkMath, { singleDollarTextMath: false }]]
const REHYPE_PLUGINS: PluggableList = [rehypeKatex, rehypeHighlight]

export function NotebookMarkdown({
  children,
  className,
}: {
  children: string
  className?: string
}) {
  return (
    <div className={cn("prose prose-sm max-w-none dark:prose-invert", className)}>
      <Markdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>
        {children}
      </Markdown>
    </div>
  )
}
