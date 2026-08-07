#!/usr/bin/env python3
"""
Aurik SVG Mystical Transformation
==================================
Transforms all Aurik SVG icons to:
  1. Transparent backgrounds (no solid dark fills)
  2. Mystical nebula-gradient backdrop (fades at edges)
  3. Enhanced glow filters (stronger halos)
  4. Ethereal sparkle particles
  5. Mystical outer aura ring
  6. Subtle star accent at icon center

Run from workspace root:
    python3 scripts/transform_svgs_mystical.py
"""

import re
from pathlib import Path

RESOURCES = Path("/media/michael/Software 4TB/Aurik_Standalone/Aurik10/resources")

# ──────────────────────────────────────────────────────────────────────────────
# Mystical SVG fragment definitions
# ──────────────────────────────────────────────────────────────────────────────

MYSTICAL_DEFS_48 = """\
  <!-- Mystical: transparent nebula backdrop -->
  <radialGradient id="m_nbg" cx="50%" cy="50%" r="50%">
    <stop offset="0%"   stop-color="#0C0720" stop-opacity="0.92"/>
    <stop offset="55%"  stop-color="#08041A" stop-opacity="0.80"/>
    <stop offset="85%"  stop-color="#05030E" stop-opacity="0.45"/>
    <stop offset="100%" stop-color="#020108" stop-opacity="0.00"/>
  </radialGradient>
  <!-- Mystical: outer cosmic ring gradient -->
  <radialGradient id="m_ring" cx="50%" cy="50%" r="50%">
    <stop offset="80%"  stop-color="#7744CC" stop-opacity="0.00"/>
    <stop offset="95%"  stop-color="#6633BB" stop-opacity="0.30"/>
    <stop offset="100%" stop-color="#4422AA" stop-opacity="0.10"/>
  </radialGradient>
  <!-- Mystical: ethereal glow filter -->
  <filter id="m_glow" x="-80%" y="-80%" width="260%" height="260%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="2.8" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <!-- Mystical: sparkle glow filter -->
  <filter id="m_spark" x="-200%" y="-200%" width="500%" height="500%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="1.6" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>"""

MYSTICAL_BG_BODY_48 = """\
<!-- Mystical nebula backdrop (transparent at edges) -->
<rect width="48" height="48" rx="6" fill="url(#m_nbg)"/>"""

MYSTICAL_OVERLAY_48 = """\
<!-- Mystical: ethereal aura rings -->
<circle cx="24" cy="24" r="22.8" fill="url(#m_ring)" opacity="1.0"/>
<circle cx="24" cy="24" r="22.5" fill="none" stroke="#7744DD" stroke-width="0.55"
  stroke-dasharray="4 3.5" opacity="0.30" filter="url(#m_glow)"/>
<circle cx="24" cy="24" r="20.5" fill="none" stroke="#3355BB" stroke-width="0.30"
  opacity="0.18" stroke-dasharray="2 5"/>
<!-- Mystical: ethereal sparkle particles -->
<g filter="url(#m_spark)">
  <circle cx="6.5"  cy="6.5"  r="0.85" fill="#B09AFF" opacity="0.65"/>
  <circle cx="41.5" cy="7.0"  r="0.70" fill="#80CCFF" opacity="0.55"/>
  <circle cx="42.0" cy="41.5" r="0.80" fill="#CC88FF" opacity="0.60"/>
  <circle cx="7.0"  cy="41.5" r="0.65" fill="#99DDFF" opacity="0.50"/>
  <circle cx="24.0" cy="3.5"  r="0.60" fill="#FFD0FF" opacity="0.40"/>
  <circle cx="44.5" cy="24.0" r="0.55" fill="#A0B8FF" opacity="0.38"/>
</g>"""

# ──────────────────────────────────────────────────────────────────────────────

MYSTICAL_DEFS_32 = """\
  <!-- Mystical: transparent nebula backdrop -->
  <radialGradient id="m_nbg" cx="50%" cy="50%" r="50%">
    <stop offset="0%"   stop-color="#0C0720" stop-opacity="0.88"/>
    <stop offset="60%"  stop-color="#08041A" stop-opacity="0.72"/>
    <stop offset="88%"  stop-color="#05030E" stop-opacity="0.38"/>
    <stop offset="100%" stop-color="#020108" stop-opacity="0.00"/>
  </radialGradient>
  <!-- Mystical: outer cosmic ring gradient -->
  <radialGradient id="m_ring" cx="50%" cy="50%" r="50%">
    <stop offset="78%"  stop-color="#7744CC" stop-opacity="0.00"/>
    <stop offset="95%"  stop-color="#6633BB" stop-opacity="0.28"/>
    <stop offset="100%" stop-color="#4422AA" stop-opacity="0.08"/>
  </radialGradient>
  <!-- Mystical: ethereal glow filter -->
  <filter id="m_glow" x="-80%" y="-80%" width="260%" height="260%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="2.2" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <!-- Mystical: sparkle glow filter -->
  <filter id="m_spark" x="-200%" y="-200%" width="500%" height="500%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="1.3" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>"""

MYSTICAL_BG_BODY_32 = """\
<!-- Mystical nebula backdrop (transparent at edges) -->
<circle cx="16" cy="16" r="16" fill="url(#m_nbg)"/>"""

MYSTICAL_OVERLAY_32 = """\
<!-- Mystical: ethereal aura rings -->
<circle cx="16" cy="16" r="15.5" fill="url(#m_ring)" opacity="1.0"/>
<circle cx="16" cy="16" r="15.2" fill="none" stroke="#7744DD" stroke-width="0.5"
  stroke-dasharray="3 3" opacity="0.28" filter="url(#m_glow)"/>
<!-- Mystical: ethereal sparkle particles -->
<g filter="url(#m_spark)">
  <circle cx="4.0"  cy="4.0"  r="0.70" fill="#B09AFF" opacity="0.60"/>
  <circle cx="28.0" cy="4.5"  r="0.55" fill="#80CCFF" opacity="0.50"/>
  <circle cx="28.5" cy="28.0" r="0.65" fill="#CC88FF" opacity="0.55"/>
  <circle cx="4.5"  cy="28.0" r="0.50" fill="#99DDFF" opacity="0.45"/>
  <circle cx="16.0" cy="2.0"  r="0.45" fill="#FFD0FF" opacity="0.35"/>
</g>"""


# ──────────────────────────────────────────────────────────────────────────────
# Filter enhancement: amplify existing feGaussianBlur stdDeviations
# ──────────────────────────────────────────────────────────────────────────────


def boost_existing_glows(content: str, factor: float = 1.65, cap: float = 4.0, floor: float = 1.0) -> str:
    """Multiply all feGaussianBlur stdDeviation values (except mystical filters already added)."""

    def replace_std(m: re.Match) -> str:
        # Skip mystical filters we just added (id=m_glow, m_spark)
        raw = m.group(1)
        try:
            val = float(raw)
            new_val = min(cap, max(floor, val * factor))
            return f'stdDeviation="{new_val:.2f}"'
        except ValueError:
            return m.group(0)  # type: ignore[no-any-return]

    return re.sub(r'stdDeviation="([0-9.]+)"', replace_std, content)


# ──────────────────────────────────────────────────────────────────────────────
# Insert mystical content into <defs> block
# ──────────────────────────────────────────────────────────────────────────────


def inject_into_defs(content: str, mystical_defs: str) -> str:
    """Append mystical defs INSIDE the existing <defs>...</defs> block.
    If no <defs> block exists, create one.
    """
    # Match closing </defs> tag
    close_defs = re.search(r"(</defs>)", content)
    if close_defs:
        pos = close_defs.start()
        return content[:pos] + mystical_defs + "\n" + content[pos:]
    # No <defs>: inject after <svg ...> opening tag
    svg_open = re.search(r"(<svg[^>]*>)", content)
    if svg_open:
        pos = svg_open.end()
        return content[:pos] + "\n<defs>\n" + mystical_defs + "\n</defs>" + content[pos:]
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Insert mystical background immediately after <defs>...</defs> (or <svg>)
# ──────────────────────────────────────────────────────────────────────────────


def inject_bg_after_defs(content: str, bg_snippet: str) -> str:
    """Insert mystical background element right after the </defs> tag."""
    close_defs = re.search(r"</defs>", content)
    if close_defs:
        pos = close_defs.end()
        # Skip any whitespace/newline after </defs>
        return content[:pos] + "\n" + bg_snippet + content[pos:]
    # Fallback: after SVG opening
    svg_open = re.search(r"(<svg[^>]*>)", content)
    if svg_open:
        pos = svg_open.end()
        return content[:pos] + "\n" + bg_snippet + content[pos:]
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Append mystical overlay just before </svg>
# ──────────────────────────────────────────────────────────────────────────────


def append_overlay(content: str, overlay: str) -> str:
    """Insert mystical overlay elements right before the closing </svg> tag."""
    pos = content.rfind("</svg>")
    if pos != -1:
        return content[:pos] + overlay + "\n" + content[pos:]
    return content + "\n" + overlay


# ──────────────────────────────────────────────────────────────────────────────
# 48x48 icon transformation
# ──────────────────────────────────────────────────────────────────────────────

# Exact background rect patterns identified in the codebase
BG_RECTS_48 = [
    '<rect width="48" height="48" rx="5" fill="#06091A"/>',
    '<rect width="48" height="48" rx="6" fill="#06091A"/>',
    '<rect width="48" height="48" rx="5" fill="#070E16"/>',
]
# Also handle with HTML comment prefix (star icons)
BG_COMMENT_48 = re.compile(
    r"<!-- dark backing -->\s*\n?" r'<rect width="48" height="48" rx="[56]" fill="#[0-9A-Fa-f]{6}"/>',
    re.IGNORECASE,
)


def transform_48(content: str, filename: str) -> str:
    """Transform a 48×48 SVG icon to mystical transparent style."""
    # 1. Remove background rects (with and without comment)
    content = BG_COMMENT_48.sub("", content)
    for pattern in BG_RECTS_48:
        content = content.replace(pattern, "")

    # 2. Amplify existing glow filters
    content = boost_existing_glows(content, factor=1.65, cap=4.0, floor=1.0)

    # 3. Inject mystical defs
    content = inject_into_defs(content, MYSTICAL_DEFS_48)

    # 4. Inject mystical nebula background
    content = inject_bg_after_defs(content, MYSTICAL_BG_BODY_48)

    # 5. Append mystical overlay (rings + sparkles)
    content = append_overlay(content, MYSTICAL_OVERLAY_48)

    return content.strip() + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# 32x32 icon transformation (phase_icons – already no solid background)
# ──────────────────────────────────────────────────────────────────────────────


def transform_32(content: str, filename: str) -> str:
    """Transform a 32×32 SVG phase icon to mystical transparent style."""
    # 1. No solid background to remove (31x32 phase icons already transparent)
    # 2. Amplify existing glow filters
    content = boost_existing_glows(content, factor=1.55, cap=3.5, floor=0.9)

    # 3. Inject mystical defs
    content = inject_into_defs(content, MYSTICAL_DEFS_32)

    # 4. Inject mystical nebula background
    content = inject_bg_after_defs(content, MYSTICAL_BG_BODY_32)

    # 5. Append mystical overlay (rings + sparkles)
    content = append_overlay(content, MYSTICAL_OVERLAY_32)

    return content.strip() + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────


def detect_viewbox_size(content: str) -> int:
    """Return 48 or 32 based on the SVG viewBox."""
    m = re.search(r'viewBox="0 0 (\d+) \d+"', content)
    if m:
        return int(m.group(1))
    return 48  # default


def process_all_svgs(dry_run: bool = False) -> None:
    svg_paths = sorted(RESOURCES.rglob("*.svg"))
    print(f"Found {len(svg_paths)} SVG files in {RESOURCES}")

    total_ok = 0
    total_err = 0

    for svg_path in svg_paths:
        try:
            original = svg_path.read_text(encoding="utf-8")
            size = detect_viewbox_size(original)

            if size == 48:
                transformed = transform_48(original, svg_path.name)
            else:
                transformed = transform_32(original, svg_path.name)

            if dry_run:
                print(f"  [DRY] {svg_path.relative_to(RESOURCES)}  ({size}px)")
            else:
                svg_path.write_text(transformed, encoding="utf-8")
                changed = original != transformed
                marker = "✓" if changed else "–"
                print(f"  {marker} {svg_path.relative_to(RESOURCES)}  ({size}px)")
            total_ok += 1

        except Exception as exc:
            print(f"  ✗ ERROR {svg_path.name}: {exc}")
            total_err += 1

    print(f"\nDone: {total_ok} transformed, {total_err} errors.")


if __name__ == "__main__":
    import sys

    dry = "--dry-run" in sys.argv
    if dry:
        print("DRY RUN — no files will be written.\n")
    process_all_svgs(dry_run=dry)
