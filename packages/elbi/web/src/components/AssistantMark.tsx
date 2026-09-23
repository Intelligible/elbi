// The assistant's own mark: a faceted gem held inside the brand's hexagonal prism, with
// a small "live" accent node. It echoes Elbi's crystalline logo rather than the
// generic AI sparkle, so the assistant reads as part of this product. Uses currentColor
// so it inherits the surrounding text color (rail, header, active state).
export function AssistantMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true" focusable="false">
      {/* The prism: a flat-top hexagon, the brand's crystalline silhouette. */}
      <path
        d="M12 2.75 20 7.375 V16.625 L12 21.25 4 16.625 V7.375 Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      {/* The gem within: a faceted diamond, the "insight" the assistant refines. */}
      <path d="M12 7.6 15.4 12 12 16.4 8.6 12 Z" fill="currentColor" />
      {/* A crown line across the gem, giving it a facet rather than a flat lozenge. */}
      <path
        d="M9.3 10.9 H14.7"
        stroke="var(--color-background)"
        strokeWidth="1"
        strokeLinecap="round"
      />
      {/* The live accent: a node on the prism's upper-right vertex. */}
      <circle cx="20" cy="7.375" r="2.1" fill="currentColor" />
    </svg>
  )
}
