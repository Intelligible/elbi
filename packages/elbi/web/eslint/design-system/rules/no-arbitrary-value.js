import { baseUtility, classStringVisitor, splitClasses } from "../class-strings.js"

const FONT = /^text-[[(]/
const RADIUS = /^rounded(?:-[a-z]+)?-[[(]/
const COLOR_ONLY = /^(?:bg|fill|stroke|from|via|to|decoration|caret|accent)-[[(]/
const MAYBE_COLOR = /^(?:border(?:-[xytrblse])?|ring(?:-offset)?|outline|divide)-[[(](.*)[\])]$/
const LOOKS_LIKE_COLOR = /#|rgb|hsl|oklch|oklab|color|var\(--/

export default {
  meta: {
    type: "problem",
    docs: { description: "No arbitrary font sizes, radii or colours in class names" },
    messages: {
      fontSize: '"{{cls}}": use the type scale (text-3xs, text-2xs, text-xs, text-compact, text-sm, text-title, text-display) or a colour token.',
      radius: '"{{cls}}": use rounded-sm|md|lg|xl.',
      color: '"{{cls}}": use a semantic colour token.',
    },
    schema: [],
  },
  create(context) {
    return classStringVisitor((node, text) => {
      for (const cls of splitClasses(text)) {
        const base = baseUtility(cls)
        const messageId = FONT.test(base)
          ? "fontSize"
          : RADIUS.test(base)
            ? "radius"
            : COLOR_ONLY.test(base) || LOOKS_LIKE_COLOR.test(MAYBE_COLOR.exec(base)?.[1] ?? "")
              ? "color"
              : null
        if (messageId) context.report({ node, messageId, data: { cls } })
      }
    })
  },
}
