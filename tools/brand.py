"""Regenerate the Elbi brand assets from the mark and the typeface.

Run with `just brand`. The brand is a badger in profile beside the wordmark: the badger
is vendored artwork at `tools/badger.svg`, the wordmark comes from Geist, and every
other asset here is composed from those two. None of the outputs should be hand-edited,
because the next run overwrites them: to change the animal, replace `tools/badger.svg`
and run again.

Geist is fetched from Google Fonts rather than vendored, so the repository carries no
font binary. It is OFL 1.1; see NOTICE.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "packages/elbi/web/public/brand"
PUBLIC = ROOT / "packages/elbi/web/public"
DOCS = ROOT / "docs/assets"
TSX = ROOT / "packages/elbi/web/src/components/Logo.tsx"
MARK = Path(__file__).resolve().parent / "badger.svg"
MARK_SOLID = Path(__file__).resolve().parent / "badger-solid.svg"

FONT_CSS = "https://fonts.googleapis.com/css2?family=Geist:wght@600&display=swap"
WORD = "elbi"

#: Ink for the places that cannot inherit a colour: a README image, a social card.
INK = "#1A1A1A"
PAPER = "#FFFFFF"
INK_DARK = "#F2EFE9"


def outlines(word: str) -> tuple[list[str], float, float, float, float]:
    """``word`` in Geist SemiBold as SVG paths, with its bounding box."""
    from fontTools.misc.transform import Identity
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    css = _fetch(FONT_CSS).decode()
    url = re.search(r"url\((https://[^)]+\.ttf)\)", css)
    if url is None:
        raise SystemExit("could not find a TTF in the Google Fonts stylesheet")
    tmp = Path(sys.argv[0]).with_name(".geist.ttf")
    tmp.write_bytes(_fetch(url.group(1)))
    try:
        font = TTFont(tmp)
        glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
        kern = font["kern"].kernTables[0].kernTable if "kern" in font else {}

        paths: list[str] = []
        x = 0.0
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for i, ch in enumerate(word):
            name = cmap[ord(ch)]
            bounds = BoundsPen(glyphs)
            glyphs[name].draw(bounds)
            if bounds.bounds:
                a, b, c, d = bounds.bounds
                x0, x1 = min(x0, a + x), max(x1, c + x)
                y0, y1 = min(y0, b), max(y1, d)
            pen = SVGPathPen(glyphs)
            # Font space is y-up; SVG is y-down.
            glyphs[name].draw(TransformPen(pen, Identity.translate(x, 0).scale(1, -1)))
            if pen.getCommands():
                paths.append(pen.getCommands())
            advance = glyphs[name].width
            if i + 1 < len(word):
                advance += kern.get((name, cmap[ord(word[i + 1])]), 0)
            x += advance
        return paths, x0, x1, y0, y1
    finally:
        tmp.unlink(missing_ok=True)


def mark(path: Path = MARK) -> tuple[str, float, float]:
    """A vendored badger: its path data and the box it was drawn in.

    The line drawing carries counters -- the ear loops and the eye -- so the fill rule
    travels with it wherever it is placed. `badger-solid.svg` is the same animal
    flooded, for sizes where the strokes would close up on each other.
    """
    text = path.read_text()
    d = re.search(r'<path[^>]*\sd="([^"]+)"', text)
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text)
    if d is None or box is None:
        raise SystemExit(f"{MARK} is not the shape this expects")
    return d.group(1), float(box.group(1)), float(box.group(2))


def _fetch(url: str) -> bytes:
    """GET ``url`` over https.

    Google serves a different stylesheet per user agent, naming a woff2 for browsers it
    knows and a plain TTF otherwise; fontTools reads the TTF, so this asks as a client
    Google does not recognise.
    """
    if not url.startswith("https://"):
        raise SystemExit(f"refusing to fetch a non-https URL: {url}")
    response = httpx.get(
        url, timeout=30, follow_redirects=True, headers={"User-Agent": "elbi-brand"}
    )
    response.raise_for_status()
    return response.content


def svg(view_box: str, body: str, width: float, height: float, label: str) -> str:
    """Wrap ``body`` in the document every asset here shares."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}"'
        f' width="{width:g}" height="{height:g}" role="img" aria-label="{label}">\n'
        f"{body}\n</svg>\n"
    )


def main() -> None:
    """Write every brand asset, reporting each path and size."""
    paths, wx0, wx1, wy0, wy1 = outlines(WORD)
    word_w, word_h = wx1 - wx0, wy1 - wy0
    mark_d, mark_w, mark_h = mark()
    solid_d, solid_w, solid_h = mark(MARK_SOLID)
    written: list[Path] = []

    #: The badger stands taller than the word beside it, the way a drawn mark has to
    #: in order not to read as a piece of punctuation.
    MARK_RATIO, GAP_RATIO = 1.62, 0.46

    def badge(
        height: float,
        dx: float,
        dy: float,
        fill: str,
        d: str = "",
        box: tuple[float, float] = (0, 0),
    ) -> str:
        path, bh = (d, box[1]) if d else (mark_d, mark_h)
        s = height / bh
        return (
            f'  <g fill="{fill}" fill-rule="evenodd"'
            f' transform="translate({dx:.2f} {dy:.2f}) scale({s:.5f})">'
            f'\n    <path d="{path}"/>\n  </g>'
        )

    def lockup(
        height: float, pad: float, mark_fill: str, word_fill: str
    ) -> tuple[str, float, float]:
        """Badger, gap, wordmark: vertically centred on each other."""
        mh = height * MARK_RATIO
        mw = mark_w * mh / mark_h
        gap = height * GAP_RATIO
        vw = pad * 2 + mw + gap + word_w * height / word_h
        vh = pad * 2 + mh
        body = (
            badge(mh, pad, pad, mark_fill)
            + "\n"
            + word(height, pad + mw + gap, pad + (mh - height) / 2, word_fill)
        )
        return body, vw, vh

    def word(height: float, dx: float, dy: float, fill: str) -> str:
        s = height / word_h
        inner = "".join(f'\n    <path d="{d}"/>' for d in paths)
        return (
            f'  <g fill="{fill}" transform="translate({dx - wx0 * s:.2f}'
            f' {dy + wy1 * s:.2f}) scale({s:.5f})">{inner}\n  </g>'
        )

    # The wordmark, taking whatever colour surrounds it. Padding is a share of the
    # word height rather than a constant, because these boxes are in font units and a
    # fixed 8 of them is invisible next to a mark thousands of units wide.
    pad = word_h * 0.10
    vw, vh = word_w + pad * 2, word_h + pad * 2
    (BRAND / "elbi-wordmark.svg").write_text(
        svg(
            f"0 0 {vw:.2f} {vh:.2f}",
            word(word_h, pad, pad, "currentColor"),
            vw,
            vh,
            "elbi",
        )
    )
    written.append(BRAND / "elbi-wordmark.svg")

    # The mark on its own, taking whatever colour surrounds it.
    (BRAND / "elbi-badger.svg").write_text(
        svg(
            f"0 0 {mark_w:g} {mark_h:g}",
            badge(mark_h, 0, 0, "currentColor"),
            mark_w,
            mark_h,
            "elbi",
        )
    )
    written.append(BRAND / "elbi-badger.svg")

    # README images go through <img>, which cannot inherit a colour, so light and dark
    # each need their ink baked in.
    for name, ink in (("logo.svg", INK), ("logo-dark.svg", INK_DARK)):
        body, lw, lh = lockup(word_h, pad, ink, ink)
        (DOCS / name).write_text(svg(f"0 0 {lw:.2f} {lh:.2f}", body, lw, lh, "elbi"))
        written.append(DOCS / name)

    # A square is needed for a favicon and a home-screen icon. The badger sits on the
    # ink tile: it is a wide animal in a square, so it is set on the optical centre
    # rather than the geometric one, which leaves it looking low.
    tile, inset = 256.0, 26.0
    icon_w = tile - inset * 2
    icon_h = icon_w * mark_h / mark_w
    (BRAND / "elbi-icon.svg").write_text(
        svg(
            f"0 0 {tile:g} {tile:g}",
            f'  <rect width="{tile:g}" height="{tile:g}" rx="56" fill="{INK}"/>\n'
            + badge(icon_h, inset, (tile - icon_h) / 2 - tile * 0.012, PAPER),
            tile,
            tile,
            "elbi",
        )
    )
    written.append(BRAND / "elbi-icon.svg")

    # The tab icon is the flooded badger, not the line drawing: at 16 and 32px the
    # strokes of the drawing merge into each other and the animal disappears.
    fav_h = icon_w * solid_h / solid_w
    (BRAND / "elbi-favicon.svg").write_text(
        svg(
            f"0 0 {tile:g} {tile:g}",
            f'  <rect width="{tile:g}" height="{tile:g}" rx="56" fill="{INK}"/>\n'
            + badge(
                fav_h,
                inset,
                (tile - fav_h) / 2 - tile * 0.012,
                PAPER,
                solid_d,
                (solid_w, solid_h),
            ),
            tile,
            tile,
            "elbi",
        )
    )
    written.append(BRAND / "elbi-favicon.svg")

    # The documentation header sits on the primary colour, so its mark is paper rather
    # than currentColor; the tab there gets the same flooded tile as the app.
    (DOCS / "mark.svg").write_text(
        svg(
            f"0 0 {mark_w:g} {mark_h:g}",
            badge(mark_h, 0, 0, PAPER),
            mark_w,
            mark_h,
            "elbi",
        )
    )
    (DOCS / "favicon.svg").write_text((BRAND / "elbi-favicon.svg").read_text())
    written += [DOCS / "mark.svg", DOCS / "favicon.svg"]

    # The in-app brand: badger and wordmark, both inheriting the interface's text
    # colour so the pair reads on any surface.
    inner = "".join(f'\n        <path d="{d}" />' for d in paths)
    TSX.write_text(f"""/**
 * The Elbi brand, inlined rather than fetched from `/brand` so it paints with the first
 * render and cannot flash. Generated by `just brand`; regenerate rather than editing by
 * hand.
 *
 * `LogoMark` is the badger, `LogoWordmark` the word, and `Logo` the pair. All
 * three take `currentColor`, so they read on any surface.
 */

import type {{ ComponentProps }} from "react"

export function LogoMark({{ className, ...props }}: ComponentProps<"svg">) {{
  return (
    <svg
      viewBox="0 0 {mark_w:g} {mark_h:g}"
      fill="currentColor"
      fillRule="evenodd"
      xmlns="http://www.w3.org/2000/svg"
      className={{className}}
      role="img"
      aria-label="elbi"
      {{...props}}
    >
      <path d="{mark_d}" />
    </svg>
  )
}}

export function LogoWordmark({{ className, ...props }}: ComponentProps<"svg">) {{
  return (
    <svg
      viewBox="0 0 {word_w:g} {word_h:g}"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      className={{className}}
      role="img"
      aria-label="elbi"
      {{...props}}
    >
      <g transform="translate({-wx0:g} {wy1:g})">{inner}
      </g>
    </svg>
  )
}}

export function Logo({{ className }}: {{ className?: string }}) {{
  return (
    <span className={{`inline-flex items-center gap-2 ${{className ?? ""}}`}}>
      <LogoMark className="h-[27px] w-auto" aria-hidden />
      <LogoWordmark className="h-[17px] w-auto" />
    </span>
  )
}}
""")
    written.append(TSX)

    # The two places SVG is not accepted: an iOS home screen and a link preview.
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        _raster(
            rsvg, BRAND / "elbi-icon.svg", PUBLIC / "apple-touch-icon.png", 180, 180
        )
        card = Path(sys.argv[0]).with_name(".og.svg")
        og_body, og_w, og_h = lockup(word_h, pad, INK, INK)
        cs = 660.0 / og_w
        card.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630">'
            f'<rect width="1200" height="630" fill="{PAPER}"/>'
            f'<g transform="translate({(1200 - og_w * cs) / 2:.1f}'
            f' {(630 - og_h * cs) / 2:.1f}) scale({cs:.4f})">'
            f"{og_body}</g></svg>"
        )
        try:
            _raster(rsvg, card, PUBLIC / "og-image.png", 1200, 630)
        finally:
            card.unlink(missing_ok=True)
        written += [PUBLIC / "apple-touch-icon.png", PUBLIC / "og-image.png"]
    else:
        print("rsvg-convert not found; skipped apple-touch-icon.png and og-image.png")

    for path in written:
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size} bytes")


def _raster(rsvg: str, src: Path, dst: Path, w: int, h: int) -> None:
    """Rasterise ``src`` to ``dst``. ``rsvg`` is an absolute, already-resolved path."""
    subprocess.run(
        [rsvg, "-w", str(w), "-h", str(h), str(src), "-o", str(dst)], check=True
    )


if __name__ == "__main__":
    main()
