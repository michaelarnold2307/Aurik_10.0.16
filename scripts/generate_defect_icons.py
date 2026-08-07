"""
Generates premium 128×128 px defect-type icons and star-rating icons for Aurik 10.0.0.

Output directories:
  Aurik10/resources/defect_icons/<defect>.png   — 30 defect-type icons
  Aurik10/resources/star_icons/star_{full,half,empty}.png — 3 star icons

Design: Muted dark-slate rounded-rect backgrounds with vivid colour-coded
symbols.  Each defect gets an intuitive visual metaphor (waveform shapes,
lightning bolts, magnets, spectra …).  3D look via gradients, soft glow
and drop-shadow — Premium-quality matching the carrier-icon set.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

_BASE = Path(__file__).parent.parent / "Aurik10" / "resources"
OUT_DEFECT = _BASE / "defect_icons"
OUT_STAR = _BASE / "star_icons"
OUT_DEFECT.mkdir(parents=True, exist_ok=True)
OUT_STAR.mkdir(parents=True, exist_ok=True)

SIZE = 128
HALF = SIZE // 2
PAD = 18  # inner padding for symbols


# ─── helpers ───────────────────────────────────────────────────────────
def _new() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _save(img: Image.Image, path: Path) -> None:
    img.save(path, "PNG", optimize=True)
    print(f"  ✓ {path.name}  ({SIZE}×{SIZE})")


def _font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _shadow(img: Image.Image, radius: int = 4, offset: int = 3) -> Image.Image:
    """Add soft drop-shadow behind icon."""
    canvas = Image.new("RGBA", (SIZE + 8, SIZE + 8), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for x in range(img.width):
        for y in range(img.height):
            pixel = img.getpixel((x, y))
            r, g, b, a = pixel  # type: ignore[misc]  # RGBA image: getpixel() always returns a 4-tuple
            if a > 30:
                shadow.putpixel((x, y), (0, 0, 0, min(120, a)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius))
    canvas.paste(shadow, (offset + 4, offset + 4), shadow)
    canvas.paste(img, (4, 4), img)
    return canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def _bg(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], alpha: int = 200) -> None:
    """Draw rounded background rectangle with gradient effect."""
    r, g, b = color
    # Darker bottom-right for 3D feel
    draw.rounded_rectangle([4, 4, SIZE - 5, SIZE - 5], radius=16, fill=(r, g, b, alpha))
    # Lighter top-left highlight
    draw.rounded_rectangle(
        [6, 6, SIZE - 7, HALF + 10], radius=14, fill=(min(255, r + 25), min(255, g + 25), min(255, b + 25), alpha // 3)
    )


def _glow(img: Image.Image, cx: int, cy: int, radius: int, color: tuple[int, int, int], alpha: int = 60) -> Image.Image:
    """Add soft colour glow at (cx, cy)."""
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(radius, 0, -2):
        a = int(alpha * (i / radius))
        gd.ellipse([cx - i, cy - i, cx + i, cy + i], fill=(*color, a))
    glow = glow.filter(ImageFilter.GaussianBlur(radius // 3 + 1))
    img = Image.alpha_composite(img, glow)
    return img


# ─── waveform helpers ──────────────────────────────────────────────────
def _draw_wave(
    draw: ImageDraw.ImageDraw,
    y_center: int,
    amplitude: int = 20,
    periods: float = 2.5,
    color: tuple[int, int, int, int] = (200, 220, 255, 200),
    width: int = 2,
    x_start: int | None = None,
    x_end: int | None = None,
):
    """Draw a sine wave."""
    x0 = x_start if x_start is not None else PAD
    x1 = x_end if x_end is not None else SIZE - PAD
    points = []
    for x in range(x0, x1):
        frac = (x - x0) / max(1, x1 - x0 - 1)
        y = y_center + int(amplitude * math.sin(frac * periods * 2 * math.pi))
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)


def _draw_zigzag(
    draw: ImageDraw.ImageDraw,
    y_center: int,
    amplitude: int = 20,
    teeth: int = 8,
    color: tuple[int, int, int, int] = (255, 100, 100, 220),
    width: int = 2,
):
    """Draw jagged zigzag line."""
    x0, x1 = PAD, SIZE - PAD
    step = (x1 - x0) / teeth
    points = []
    for i in range(teeth + 1):
        x = int(x0 + i * step)
        y = y_center + (amplitude if i % 2 == 0 else -amplitude)
        points.append((x, y))
    draw.line(points, fill=color, width=width)


# ─── DEFECT ICON GENERATORS ───────────────────────────────────────────


def gen_clicks():
    img, draw = _new()
    _bg(draw, (60, 45, 50))
    # Lightning bolt = sudden impulse
    bolt = [(50, PAD), (58, 52), (48, 56), (68, SIZE - PAD), (60, 72), (70, 68), (50, PAD)]
    draw.polygon(bolt, fill=(255, 220, 80, 240))
    draw.polygon(bolt, outline=(255, 180, 40, 255))
    # Small sparks
    for dx, dy in [(-20, -8), (22, 5), (-15, 18)]:
        cx, cy = HALF + dx, HALF + dy
        draw.line([(cx - 4, cy), (cx + 4, cy)], fill=(255, 230, 120, 180), width=2)
        draw.line([(cx, cy - 4), (cx, cy + 4)], fill=(255, 230, 120, 180), width=2)
    img = _glow(img, HALF, HALF, 30, (255, 200, 60))
    return _shadow(img)


def gen_crackle():
    img, draw = _new()
    _bg(draw, (55, 45, 42))
    # Scattered small spark particles
    import random

    rng = random.Random(42)
    for _ in range(25):
        x = rng.randint(PAD + 5, SIZE - PAD - 5)
        y = rng.randint(PAD + 5, SIZE - PAD - 5)
        sz = rng.randint(1, 3)
        alpha = rng.randint(140, 240)
        draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=(255, 180 + rng.randint(0, 60), 80, alpha))
    # Small fire/ember glow
    img = _glow(img, HALF, HALF, 25, (255, 160, 60))
    return _shadow(img)


def gen_pops():
    img, draw = _new()
    _bg(draw, (55, 42, 52))
    # Concentric burst rings
    for r in range(30, 10, -5):
        a = 60 + (30 - r) * 6
        draw.ellipse([HALF - r, HALF - r, HALF + r, HALF + r], outline=(255, 140, 200, a), width=2)
    # Center dot
    draw.ellipse([HALF - 5, HALF - 5, HALF + 5, HALF + 5], fill=(255, 180, 220, 220))
    img = _glow(img, HALF, HALF, 20, (255, 140, 200))
    return _shadow(img)


def gen_clipping():
    img, draw = _new()
    _bg(draw, (65, 35, 35))
    # Clipped waveform: sine with flat top/bottom
    y_c = HALF
    amp = 30
    clip_lvl = 18
    points = []
    for x in range(PAD, SIZE - PAD):
        frac = (x - PAD) / max(1, SIZE - 2 * PAD - 1)
        y_raw = amp * math.sin(frac * 3 * 2 * math.pi)
        y_clipped = max(-clip_lvl, min(clip_lvl, y_raw))
        points.append((x, y_c - int(y_clipped)))
    draw.line(points, fill=(255, 90, 90, 230), width=3)
    # Clip lines
    draw.line([(PAD, y_c - clip_lvl), (SIZE - PAD, y_c - clip_lvl)], fill=(255, 50, 50, 120), width=1)
    draw.line([(PAD, y_c + clip_lvl), (SIZE - PAD, y_c + clip_lvl)], fill=(255, 50, 50, 120), width=1)
    img = _glow(img, HALF, HALF, 25, (255, 60, 60))
    return _shadow(img)


def gen_hum():
    img, draw = _new()
    _bg(draw, (42, 48, 62))
    # 50/60 Hz sine wave = clean low-frequency wave
    _draw_wave(draw, HALF, amplitude=28, periods=2.0, color=(120, 180, 255, 220), width=3)
    # "50 Hz" label
    f = _font(11)
    draw.text((HALF - 14, SIZE - PAD - 6), "50Hz", fill=(120, 180, 255, 160), font=f)
    img = _glow(img, HALF, HALF, 22, (100, 160, 255))
    return _shadow(img)


def gen_noise():
    img, draw = _new()
    _bg(draw, (48, 48, 52))
    # Static noise pattern
    import random

    rng = random.Random(99)
    for x in range(PAD + 2, SIZE - PAD - 2, 3):
        for y in range(PAD + 8, SIZE - PAD - 8, 3):
            v = rng.randint(80, 220)
            a = rng.randint(60, 180)
            draw.rectangle([x, y, x + 2, y + 2], fill=(v, v, v + 20, a))
    img = _glow(img, HALF, HALF, 20, (180, 180, 200))
    return _shadow(img)


def gen_sibilance():
    img, draw = _new()
    _bg(draw, (52, 42, 58))
    # "S" shape with high-frequency zigzag
    _draw_zigzag(draw, HALF - 10, amplitude=8, teeth=16, color=(220, 140, 255, 200), width=2)
    _draw_zigzag(draw, HALF + 10, amplitude=6, teeth=14, color=(200, 120, 240, 160), width=2)
    f = _font(28)
    draw.text((HALF - 9, PAD - 2), "S", fill=(220, 160, 255, 200), font=f)
    img = _glow(img, HALF, HALF - 5, 22, (200, 140, 255))
    return _shadow(img)


def gen_dropout():
    img, draw = _new()
    _bg(draw, (50, 40, 45))
    # Waveform with a gap in the middle
    y_c = HALF
    for x in range(PAD, HALF - 12):
        frac = (x - PAD) / 30.0
        y = y_c + int(18 * math.sin(frac * 6))
        draw.line([(x, y - 1), (x, y + 1)], fill=(180, 220, 180, 200), width=2)
    # Gap — dashed line
    for x in range(HALF - 10, HALF + 10, 4):
        draw.line([(x, y_c), (x + 2, y_c)], fill=(255, 80, 80, 140), width=1)
    for x in range(HALF + 14, SIZE - PAD):
        frac = (x - HALF) / 30.0
        y = y_c + int(18 * math.sin(frac * 6))
        draw.line([(x, y - 1), (x, y + 1)], fill=(180, 220, 180, 200), width=2)
    # X mark in gap
    cx, cy = HALF, y_c
    draw.line([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=(255, 80, 80, 200), width=2)
    draw.line([(cx - 6, cy + 6), (cx + 6, cy - 6)], fill=(255, 80, 80, 200), width=2)
    img = _glow(img, HALF, y_c, 18, (255, 80, 80))
    return _shadow(img)


def gen_wow():
    img, draw = _new()
    _bg(draw, (45, 48, 55))
    # Slow wobble — very low frequency, large amplitude modulation
    _draw_wave(draw, HALF, amplitude=32, periods=0.7, color=(100, 200, 220, 220), width=3)
    f = _font(10)
    draw.text((PAD + 2, SIZE - PAD - 4), "<0.5 Hz", fill=(100, 200, 220, 140), font=f)
    img = _glow(img, HALF, HALF, 20, (80, 180, 200))
    return _shadow(img)


def gen_flutter():
    img, draw = _new()
    _bg(draw, (48, 45, 55))
    # Fast wobble — higher frequency, smaller amplitude
    _draw_wave(draw, HALF, amplitude=16, periods=6.0, color=(160, 130, 255, 220), width=3)
    f = _font(10)
    draw.text((PAD + 2, SIZE - PAD - 4), ">0.5 Hz", fill=(160, 130, 255, 140), font=f)
    img = _glow(img, HALF, HALF, 20, (140, 110, 240))
    return _shadow(img)


def gen_rumble():
    img, draw = _new()
    _bg(draw, (42, 42, 52))
    # Very large low-freq wave dominating the icon
    _draw_wave(draw, HALF, amplitude=36, periods=0.5, color=(100, 130, 200, 180), width=4)
    # Subsonic arrows pointing down
    for dx in [-15, 0, 15]:
        cx = HALF + dx
        draw.polygon(
            [(cx, SIZE - PAD - 2), (cx - 5, SIZE - PAD - 12), (cx + 5, SIZE - PAD - 12)], fill=(100, 130, 200, 160)
        )
    img = _glow(img, HALF, HALF + 10, 24, (80, 110, 180))
    return _shadow(img)


def gen_dc_offset():
    img, draw = _new()
    _bg(draw, (48, 50, 55))
    # Waveform shifted upward from center
    y_c = HALF - 14  # shifted up
    _draw_wave(draw, y_c, amplitude=16, periods=2.5, color=(180, 200, 140, 200), width=2)
    # True center line (dashed)
    for x in range(PAD, SIZE - PAD, 6):
        draw.line([(x, HALF), (x + 3, HALF)], fill=(255, 255, 255, 80), width=1)
    # Offset arrow
    draw.line([(SIZE - PAD - 8, HALF), (SIZE - PAD - 8, y_c)], fill=(255, 200, 80, 180), width=2)
    draw.polygon(
        [(SIZE - PAD - 8, y_c), (SIZE - PAD - 12, y_c + 6), (SIZE - PAD - 4, y_c + 6)], fill=(255, 200, 80, 180)
    )
    img = _glow(img, HALF, y_c, 18, (200, 180, 80))
    return _shadow(img)


def gen_digital_artifacts():
    img, draw = _new()
    _bg(draw, (50, 42, 55))
    # Glitch blocks
    import random

    rng = random.Random(77)
    for _ in range(12):
        x = rng.randint(PAD + 4, SIZE - PAD - 20)
        y = rng.randint(PAD + 4, SIZE - PAD - 10)
        w = rng.randint(8, 25)
        h = rng.randint(4, 10)
        r = rng.randint(120, 255)
        g = rng.randint(40, 120)
        b = rng.randint(180, 255)
        draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, rng.randint(100, 200)))
    # Small "0/1" text
    f = _font(9)
    draw.text((PAD + 4, PAD + 2), "01001", fill=(180, 140, 255, 120), font=f)
    img = _glow(img, HALF, HALF, 20, (160, 100, 220))
    return _shadow(img)


def gen_compression_artifacts():
    img, draw = _new()
    _bg(draw, (52, 45, 48))
    # Staircase quantization pattern
    y_c = HALF
    step_h = 8
    n_steps = 7
    w_per = (SIZE - 2 * PAD) // n_steps
    for i in range(n_steps):
        x0 = PAD + i * w_per
        y_val = y_c - 24 + i * step_h
        draw.rectangle(
            [x0, y_val, x0 + w_per - 2, y_val + step_h - 1], fill=(220, 160, 100, 180), outline=(240, 180, 120, 100)
        )
    f = _font(10)
    draw.text((HALF - 16, SIZE - PAD - 4), "Codec", fill=(220, 160, 100, 140), font=f)
    img = _glow(img, HALF, HALF, 18, (200, 140, 80))
    return _shadow(img)


def gen_stereo_imbalance():
    img, draw = _new()
    _bg(draw, (45, 50, 55))
    # L channel big, R channel small
    # L bar
    draw.rectangle(
        [PAD + 8, PAD + 10, HALF - 8, SIZE - PAD - 10], fill=(100, 180, 255, 180), outline=(120, 200, 255, 100)
    )
    # R bar (shorter)
    draw.rectangle(
        [HALF + 8, HALF - 5, SIZE - PAD - 8, SIZE - PAD - 10], fill=(255, 140, 100, 180), outline=(255, 160, 120, 100)
    )
    f = _font(12)
    draw.text((PAD + 16, PAD + 14), "L", fill=(255, 255, 255, 180), font=f)
    draw.text((HALF + 18, HALF), "R", fill=(255, 255, 255, 180), font=f)
    return _shadow(img)


def gen_phase_issues():
    img, draw = _new()
    _bg(draw, (48, 45, 55))
    # Two inverted sine waves
    _draw_wave(draw, HALF - 12, amplitude=16, periods=2.0, color=(100, 200, 255, 200), width=2)
    _draw_wave(draw, HALF + 12, amplitude=16, periods=2.0, color=(255, 120, 120, 200), width=2)
    # Phase inversion symbol (Ø)
    f = _font(24)
    draw.text((HALF - 8, HALF - 14), "Ø", fill=(255, 200, 100, 180), font=f)
    img = _glow(img, HALF, HALF, 18, (200, 160, 100))
    return _shadow(img)


def gen_bandwidth_loss():
    img, draw = _new()
    _bg(draw, (50, 48, 45))
    # Frequency spectrum bars declining then cut off
    n_bars = 10
    bar_w = (SIZE - 2 * PAD - 10) // n_bars
    for i in range(n_bars):
        x0 = PAD + 5 + i * bar_w
        if i < 6:
            h = int(50 - i * 4)
            c = (120, 200, 160, 200)
        else:
            h = int(max(3, 26 - (i - 6) * 8))
            c = (255, 100, 80, 140)
        y0 = SIZE - PAD - 10 - h
        draw.rectangle([x0, y0, x0 + bar_w - 2, SIZE - PAD - 10], fill=c)
    # Cutoff line
    cut_x = PAD + 5 + 6 * bar_w - 3
    draw.line([(cut_x, PAD + 8), (cut_x, SIZE - PAD - 8)], fill=(255, 80, 60, 180), width=2)
    img = _glow(img, cut_x, HALF, 16, (255, 80, 60))
    return _shadow(img)


def gen_pitch_drift():
    img, draw = _new()
    _bg(draw, (48, 50, 52))
    # Ascending/drifting line
    points = []
    for x in range(PAD, SIZE - PAD):
        frac = (x - PAD) / max(1, SIZE - 2 * PAD - 1)
        y = HALF + 20 - int(frac * 40) + int(8 * math.sin(frac * 8))
        points.append((x, y))
    draw.line(points, fill=(200, 180, 100, 220), width=3)
    # Arrows indicating drift direction
    draw.polygon(
        [(SIZE - PAD - 5, PAD + 14), (SIZE - PAD - 10, PAD + 22), (SIZE - PAD, PAD + 22)], fill=(200, 180, 100, 180)
    )
    img = _glow(img, HALF, HALF, 18, (180, 160, 80))
    return _shadow(img)


def gen_reverb_excess():
    img, draw = _new()
    _bg(draw, (45, 48, 58))
    # Initial transient + decaying echo tails
    # Impulse
    draw.line([(PAD + 10, SIZE - PAD - 15), (PAD + 10, PAD + 15)], fill=(180, 220, 255, 230), width=3)
    # Decay tails (diminishing)
    for i, dx in enumerate(range(20, 70, 12)):
        h = max(4, 45 - i * 10)
        a = max(40, 200 - i * 40)
        x = PAD + 10 + dx
        draw.line([(x, HALF - h // 2), (x, HALF + h // 2)], fill=(140, 180, 255, a), width=2)
    img = _glow(img, PAD + 30, HALF, 22, (120, 160, 240))
    return _shadow(img)


def gen_print_through():
    img, draw = _new()
    _bg(draw, (52, 48, 45))
    # Main signal (solid)
    draw.line([(HALF, PAD + 10), (HALF, SIZE - PAD - 10)], fill=(200, 220, 180, 220), width=3)
    # Ghost echo before (faint copy to the left)
    draw.line([(HALF - 22, PAD + 18), (HALF - 22, SIZE - PAD - 18)], fill=(200, 180, 140, 80), width=2)
    # Ghost echo after
    draw.line([(HALF + 22, PAD + 18), (HALF + 22, SIZE - PAD - 18)], fill=(200, 180, 140, 60), width=2)
    # Arrows from ghosts to main
    for dx, arr_a in [(-22, 100), (22, 80)]:
        cx = HALF + dx
        mid = HALF
        draw.line([(cx, HALF), (mid, HALF)], fill=(255, 200, 100, arr_a), width=1)
    f = _font(9)
    draw.text((PAD + 2, PAD + 2), "Echo", fill=(200, 180, 140, 120), font=f)
    img = _glow(img, HALF, HALF, 16, (200, 180, 120))
    return _shadow(img)


def gen_quantization_noise():
    img, draw = _new()
    _bg(draw, (50, 48, 52))
    # Staircase waveform (quantized sine)
    y_c = HALF
    n_steps = 12
    step_w = (SIZE - 2 * PAD) // n_steps
    prev_y = None
    for i in range(n_steps):
        frac = i / max(1, n_steps - 1)
        y_raw = 28 * math.sin(frac * 2 * math.pi)
        y_quant = round(y_raw / 8) * 8
        y = y_c - y_quant
        x0 = PAD + i * step_w
        x1 = x0 + step_w
        draw.line([(x0, y), (x1, y)], fill=(180, 160, 255, 200), width=2)
        if prev_y is not None and prev_y != y:
            draw.line([(x0, prev_y), (x0, y)], fill=(180, 160, 255, 200), width=2)
        prev_y = y
    f = _font(9)
    draw.text((PAD + 2, SIZE - PAD - 4), "Bits", fill=(180, 160, 255, 120), font=f)
    img = _glow(img, HALF, HALF, 16, (160, 140, 240))
    return _shadow(img)


def gen_jitter_artifacts():
    img, draw = _new()
    _bg(draw, (52, 45, 50))
    # Clock signal with jagged timing errors
    y_c = HALF
    x = PAD
    points = [(x, y_c)]
    state = 1
    nominal_w = 10
    import random

    rng = random.Random(55)
    while x < SIZE - PAD:
        jitter = rng.randint(-3, 3)
        w = max(4, nominal_w + jitter)
        y_top = y_c - 20
        y_bot = y_c + 20
        y = y_top if state else y_bot
        points.append((x, y))
        x += w
        points.append((x, y))
        state = 1 - state
    draw.line(points, fill=(255, 160, 120, 200), width=2)
    # "CLK" label
    f = _font(9)
    draw.text((PAD + 2, PAD + 2), "CLK", fill=(255, 160, 120, 120), font=f)
    img = _glow(img, HALF, HALF, 18, (240, 140, 100))
    return _shadow(img)


def gen_dynamic_compression_excess():
    img, draw = _new()
    _bg(draw, (55, 45, 42))
    # Squashed waveform — big input compressed to narrow output
    y_c = HALF
    # Original (faint, full dynamic range)
    _draw_wave(draw, y_c, amplitude=35, periods=2.0, color=(160, 160, 160, 80), width=1)
    # Compressed (loud but flat)
    _draw_wave(draw, y_c, amplitude=10, periods=2.0, color=(255, 120, 80, 220), width=3)
    # Compression arrows pointing inward
    for dy_off in [-25, 25]:
        y = y_c + dy_off
        # Arrow pointing toward center
        direction = 1 if dy_off < 0 else -1
        draw.polygon([(HALF - 4, y), (HALF + 4, y), (HALF, y + direction * 8)], fill=(255, 160, 80, 160))
    img = _glow(img, HALF, y_c, 18, (240, 100, 60))
    return _shadow(img)


def gen_pre_echo():
    img, draw = _new()
    _bg(draw, (48, 48, 55))
    # Faint ghost before a strong transient
    # Ghost
    draw.line([(HALF - 25, HALF + 15), (HALF - 25, HALF - 15)], fill=(200, 180, 255, 80), width=2)
    draw.line([(HALF - 15, HALF + 10), (HALF - 15, HALF - 10)], fill=(200, 180, 255, 100), width=2)
    # Main transient
    draw.line([(HALF + 5, SIZE - PAD - 10), (HALF + 5, PAD + 10)], fill=(200, 220, 255, 230), width=4)
    # Arrow from ghost to main
    draw.line([(HALF - 24, HALF - 18), (HALF + 2, HALF - 18)], fill=(200, 180, 255, 100), width=1)
    f = _font(9)
    draw.text((PAD, PAD + 2), "Pre", fill=(200, 180, 255, 130), font=f)
    img = _glow(img, HALF + 5, HALF, 18, (180, 160, 240))
    return _shadow(img)


def gen_transient_smearing():
    img, draw = _new()
    _bg(draw, (48, 45, 50))
    # Sharp transient blurred/smeared
    # Sharp original (faint)
    draw.line([(HALF - 20, SIZE - PAD - 15), (HALF - 20, PAD + 15)], fill=(180, 220, 180, 100), width=2)
    # Smeared version (wide, blurry)
    for dx in range(-8, 9, 2):
        a = max(30, 140 - abs(dx) * 18)
        h_red = abs(dx) * 3
        draw.line(
            [(HALF + 15 + dx, SIZE - PAD - 15 + h_red), (HALF + 15 + dx, PAD + 15 + h_red)],
            fill=(255, 160, 120, a),
            width=2,
        )
    # Arrow from sharp to smeared
    draw.line([(HALF - 15, HALF), (HALF + 5, HALF)], fill=(255, 200, 100, 140), width=1)
    draw.polygon([(HALF + 5, HALF - 4), (HALF + 5, HALF + 4), (HALF + 10, HALF)], fill=(255, 200, 100, 160))
    img = _glow(img, HALF + 15, HALF, 16, (240, 140, 100))
    return _shadow(img)


def gen_head_wear():
    img, draw = _new()
    _bg(draw, (50, 50, 48))
    # Tape head (stylized rectangle with gap)
    hx, hy = HALF, HALF
    draw.rounded_rectangle(
        [hx - 22, hy - 28, hx + 22, hy + 28], radius=4, fill=(140, 140, 150, 200), outline=(180, 180, 190, 160)
    )
    # Head gap
    draw.rectangle([hx - 18, hy - 2, hx + 18, hy + 2], fill=(60, 60, 65, 230))
    # Wear marks (scratches)
    for dy in [-12, -6, 8, 14]:
        draw.line([(hx - 15, hy + dy), (hx + 15, hy + dy)], fill=(200, 180, 120, 80), width=1)
    # Warning indicator
    draw.ellipse([SIZE - PAD - 12, PAD + 2, SIZE - PAD, PAD + 14], fill=(255, 100, 80, 200))
    img = _glow(img, hx, hy, 18, (140, 140, 150))
    return _shadow(img)


def gen_riaa_curve_error():
    img, draw = _new()
    _bg(draw, (48, 50, 48))
    # RIAA EQ curve (correct: descending) and wrong curve
    # Correct curve (faint green)
    points_ok = []
    for x in range(PAD + 5, SIZE - PAD - 5):
        frac = (x - PAD - 5) / max(1, SIZE - 2 * PAD - 10)
        y = PAD + 15 + int(frac * 60)
        points_ok.append((x, y))
    draw.line(points_ok, fill=(120, 200, 140, 100), width=2)
    # Wrong curve (red, shifted)
    points_err = []
    for x in range(PAD + 5, SIZE - PAD - 5):
        frac = (x - PAD - 5) / max(1, SIZE - 2 * PAD - 10)
        y = PAD + 25 + int(frac * 40) + int(10 * math.sin(frac * 4))
        points_err.append((x, y))
    draw.line(points_err, fill=(255, 120, 100, 180), width=2)
    f = _font(10)
    draw.text((PAD + 2, SIZE - PAD - 6), "RIAA", fill=(200, 180, 140, 140), font=f)
    img = _glow(img, HALF, HALF, 16, (200, 140, 100))
    return _shadow(img)


def gen_aliasing():
    img, draw = _new()
    _bg(draw, (50, 45, 52))
    # Original high-freq wave + folded-back alias
    _draw_wave(draw, HALF - 14, amplitude=12, periods=5.0, color=(140, 200, 255, 160), width=2)
    # Aliased fold-back (mirrored, different color)
    _draw_wave(draw, HALF + 14, amplitude=12, periods=3.5, color=(255, 120, 140, 200), width=2)
    # Fold-back arrow
    draw.line([(SIZE - PAD - 10, HALF - 14), (SIZE - PAD - 10, HALF + 14)], fill=(255, 200, 100, 140), width=1)
    draw.polygon(
        [(SIZE - PAD - 14, HALF + 10), (SIZE - PAD - 6, HALF + 10), (SIZE - PAD - 10, HALF + 16)],
        fill=(255, 200, 100, 160),
    )
    img = _glow(img, HALF, HALF, 16, (200, 140, 180))
    return _shadow(img)


def gen_bias_error():
    img, draw = _new()
    _bg(draw, (52, 48, 45))
    # Magnet symbol (horseshoe)
    cx, cy = HALF, HALF - 5
    # Magnet body (U-shape via arcs)
    draw.arc([cx - 22, cy - 10, cx + 22, cy + 30], start=0, end=180, fill=(200, 80, 80, 220), width=8)
    # Magnet poles (two rectangles)
    draw.rectangle([cx - 22, cy - 5, cx - 14, cy + 6], fill=(200, 80, 80, 220))
    draw.rectangle([cx + 14, cy - 5, cx + 22, cy + 6], fill=(80, 80, 200, 220))
    # N/S labels
    f = _font(10)
    draw.text((cx - 22, cy - 22), "N", fill=(255, 120, 120, 180), font=f)
    draw.text((cx + 14, cy - 22), "S", fill=(120, 120, 255, 180), font=f)
    # Field lines (faint arcs)
    for r_off in [8, 16]:
        draw.arc(
            [cx - 22 - r_off, cy - 10 - r_off, cx + 22 + r_off, cy + 30 + r_off],
            start=10,
            end=170,
            fill=(180, 140, 200, 50),
            width=1,
        )
    img = _glow(img, cx, cy + 10, 20, (180, 80, 140))
    return _shadow(img)


def gen_transport_bump():
    img, draw = _new()
    _bg(draw, (50, 48, 48))
    # Flat waveform with sudden bump/spike
    y_c = HALF
    _draw_wave(draw, y_c, amplitude=10, periods=3.0, color=(160, 180, 200, 160), width=2)
    # Large bump in the middle
    bump_x = HALF
    draw.line([(bump_x - 2, y_c), (bump_x, y_c - 35), (bump_x + 2, y_c)], fill=(255, 140, 80, 230), width=3)
    # Small tape-reel symbol
    draw.ellipse([PAD + 4, PAD + 4, PAD + 18, PAD + 18], outline=(180, 180, 190, 120), width=2)
    draw.ellipse([PAD + 8, PAD + 8, PAD + 14, PAD + 14], fill=(180, 180, 190, 120))
    img = _glow(img, bump_x, y_c - 15, 16, (240, 120, 60))
    return _shadow(img)


def gen_soft_saturation():
    """Tube/tape saturation — warm, vintage, to be PRESERVED."""
    img, draw = _new()
    _bg(draw, (55, 48, 40))
    # Softly clipped sine (rounded tops — NOT flat like clipping)
    y_c = HALF
    points = []
    for x in range(PAD, SIZE - PAD):
        frac = (x - PAD) / max(1, SIZE - 2 * PAD - 1)
        raw = 35 * math.sin(frac * 2.5 * 2 * math.pi)
        # Soft clip via tanh
        soft = 22 * math.tanh(raw / 18)
        points.append((x, y_c - int(soft)))
    draw.line(points, fill=(255, 180, 80, 220), width=3)
    # Warm glow
    f = _font(9)
    draw.text((PAD + 2, SIZE - PAD - 4), "Warm", fill=(255, 180, 80, 130), font=f)
    img = _glow(img, HALF, y_c, 22, (255, 160, 60))
    return _shadow(img)


def gen_azimuth_error():
    img, draw = _new()
    _bg(draw, (48, 50, 52))
    # Two channel waves slightly time-shifted (L ahead of R)
    _draw_wave(draw, HALF - 14, amplitude=14, periods=2.5, color=(100, 180, 255, 200), width=2)
    # R channel shifted right
    _draw_wave(
        draw,
        HALF + 14,
        amplitude=14,
        periods=2.5,
        color=(255, 140, 100, 200),
        width=2,
        x_start=PAD + 12,
        x_end=SIZE - PAD + 12,
    )
    # L/R labels
    f = _font(10)
    draw.text((PAD, HALF - 24), "L", fill=(100, 180, 255, 160), font=f)
    draw.text((PAD, HALF + 10), "R", fill=(255, 140, 100, 160), font=f)
    # Time-shift arrow
    draw.line([(HALF, HALF - 6), (HALF + 12, HALF + 6)], fill=(255, 200, 100, 160), width=2)
    img = _glow(img, HALF, HALF, 16, (200, 160, 100))
    return _shadow(img)


# ─── STAR ICON GENERATORS ─────────────────────────────────────────────


def _star_polygon(cx: int, cy: int, outer: int, inner: int, points: int = 5) -> list[tuple[int, int]]:
    """Return star polygon vertices."""
    verts = []
    for i in range(points * 2):
        angle = math.pi / 2 + i * math.pi / points
        r = outer if i % 2 == 0 else inner
        verts.append((int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle))))
    return verts


def gen_star_full():
    img, draw = _new()
    poly = _star_polygon(HALF, HALF, 52, 22)
    # Gold fill with radial gradient effect
    draw.polygon(poly, fill=(255, 200, 50, 240))
    # Highlight on upper portion
    poly_inner = _star_polygon(HALF, HALF - 4, 36, 16)
    draw.polygon(poly_inner, fill=(255, 230, 100, 100))
    # Outline
    draw.polygon(poly, outline=(200, 160, 30, 200))
    img = _glow(img, HALF, HALF, 28, (255, 200, 50))
    return _shadow(img)


def gen_star_half():
    img, draw = _new()
    poly = _star_polygon(HALF, HALF, 52, 22)
    # Outline only
    draw.polygon(poly, fill=(80, 80, 90, 180))
    draw.polygon(poly, outline=(200, 160, 30, 200))
    # Fill left half with gold using a mask
    mask = Image.new("L", (SIZE, SIZE), 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([0, 0, HALF, SIZE], fill=255)
    gold_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gold_layer)
    gd.polygon(poly, fill=(255, 200, 50, 240))
    poly_inner = _star_polygon(HALF, HALF - 4, 36, 16)
    gd.polygon(poly_inner, fill=(255, 230, 100, 80))
    gold_layer.putalpha(mask)
    img = Image.alpha_composite(img, gold_layer)
    img = _glow(img, HALF - 10, HALF, 22, (255, 200, 50), alpha=40)
    return _shadow(img)


def gen_star_empty():
    img, draw = _new()
    poly = _star_polygon(HALF, HALF, 52, 22)
    draw.polygon(poly, fill=(60, 60, 70, 160))
    draw.polygon(poly, outline=(140, 130, 100, 180))
    return _shadow(img)


# ─── SEVERITY INDICATOR ICONS ─────────────────────────────────────────


def gen_severity_high():
    img, draw = _new()
    # Red pulsing circle
    for r in range(35, 5, -3):
        a = max(20, 200 - (35 - r) * 6)
        draw.ellipse([HALF - r, HALF - r, HALF + r, HALF + r], fill=(220, 50, 50, a))
    draw.ellipse([HALF - 12, HALF - 12, HALF + 12, HALF + 12], fill=(255, 80, 80, 240))
    # Exclamation mark
    f = _font(18)
    draw.text((HALF - 4, HALF - 12), "!", fill=(255, 255, 255, 230), font=f)
    return _shadow(img)


def gen_severity_medium():
    img, draw = _new()
    for r in range(32, 5, -3):
        a = max(20, 180 - (32 - r) * 6)
        draw.ellipse([HALF - r, HALF - r, HALF + r, HALF + r], fill=(220, 180, 40, a))
    draw.ellipse([HALF - 12, HALF - 12, HALF + 12, HALF + 12], fill=(255, 210, 60, 240))
    # "~" mark
    f = _font(18)
    draw.text((HALF - 6, HALF - 12), "~", fill=(80, 60, 0, 220), font=f)
    return _shadow(img)


def gen_severity_low():
    img, draw = _new()
    for r in range(30, 5, -3):
        a = max(20, 160 - (30 - r) * 6)
        draw.ellipse([HALF - r, HALF - r, HALF + r, HALF + r], fill=(60, 180, 100, a))
    draw.ellipse([HALF - 10, HALF - 10, HALF + 10, HALF + 10], fill=(80, 210, 120, 240))
    # Checkmark
    draw.line([(HALF - 6, HALF), (HALF - 1, HALF + 5), (HALF + 7, HALF - 6)], fill=(255, 255, 255, 220), width=2)
    return _shadow(img)


# ─── registry ─────────────────────────────────────────────────────────

DEFECT_GENERATORS: dict[str, callable] = {  # type: ignore[valid-type]
    "clicks": gen_clicks,
    "crackle": gen_crackle,
    "pops": gen_pops,
    "clipping": gen_clipping,
    "hum": gen_hum,
    "noise": gen_noise,
    "noise_level": gen_noise,  # alias
    "sibilance": gen_sibilance,
    "dropout": gen_dropout,
    "wow": gen_wow,
    "flutter": gen_flutter,
    "rumble": gen_rumble,
    "dc_offset": gen_dc_offset,
    "digital_artifacts": gen_digital_artifacts,
    "compression_artifacts": gen_compression_artifacts,
    "stereo_imbalance": gen_stereo_imbalance,
    "phase_issues": gen_phase_issues,
    "bandwidth_loss": gen_bandwidth_loss,
    "pitch_drift": gen_pitch_drift,
    "reverb_excess": gen_reverb_excess,
    "print_through": gen_print_through,
    "quantization_noise": gen_quantization_noise,
    "jitter_artifacts": gen_jitter_artifacts,
    "dynamic_compression_excess": gen_dynamic_compression_excess,
    "pre_echo": gen_pre_echo,
    "transient_smearing": gen_transient_smearing,
    "head_wear": gen_head_wear,
    "riaa_curve_error": gen_riaa_curve_error,
    "aliasing": gen_aliasing,
    "bias_error": gen_bias_error,
    "transport_bump": gen_transport_bump,
    "soft_saturation": gen_soft_saturation,
    "azimuth_error": gen_azimuth_error,
}

STAR_GENERATORS: dict[str, callable] = {  # type: ignore[valid-type]
    "star_full": gen_star_full,
    "star_half": gen_star_half,
    "star_empty": gen_star_empty,
}

SEVERITY_GENERATORS: dict[str, callable] = {  # type: ignore[valid-type]
    "severity_high": gen_severity_high,
    "severity_medium": gen_severity_medium,
    "severity_low": gen_severity_low,
}


def main() -> None:
    print("=== Generating defect icons ===")
    for name, fn in DEFECT_GENERATORS.items():
        _save(fn(), OUT_DEFECT / f"{name}.png")  # type: ignore[misc]

    print("\n=== Generating star icons ===")
    for name, fn in STAR_GENERATORS.items():
        _save(fn(), OUT_STAR / f"{name}.png")  # type: ignore[misc]

    print("\n=== Generating severity icons ===")
    for name, fn in SEVERITY_GENERATORS.items():
        _save(fn(), OUT_DEFECT / f"{name}.png")  # type: ignore[misc]

    print(
        f"\nDone.  {len(DEFECT_GENERATORS)} defect + {len(STAR_GENERATORS)} star + {len(SEVERITY_GENERATORS)} severity icons generated."
    )


if __name__ == "__main__":
    main()
