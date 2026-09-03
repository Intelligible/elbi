import "katex/dist/katex.min.css"
import "highlight.js/styles/github-dark.css"

import Markdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import rehypeKatex from "rehype-katex"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"

import { cn } from "@/lib/utils"

// Markdown for notebook surfaces, matching how Jupyter renders a markdown cell and a
// `text/markdown` / `text/latex` output: GitHub-flavored markdown (tables, task lists,
// fenced code) with syntax-highlighted code blocks (highlight.js) plus TeX math, so
// `$x^2$` and `$$\int$$` render with KaTeX. The plugin lists are module-level constants
// so the editor does not rebuild the pipeline on every keystroke.
const REMARK_PLUGINS = [remarkGfm, remarkMath]
const REHYPE_PLUGINS = [rehypeKatex, rehypeHighlight]

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
