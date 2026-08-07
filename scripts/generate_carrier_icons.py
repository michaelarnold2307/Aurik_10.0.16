"""
Generates premium 128×128 px photorealistic carrier-medium icons for Aurik 10.0.0.
Output: Aurik10/resources/carrier_icons/<medium>.png

Design: Photorealistic 3D look with material gradients, specular highlights,
grooves, shadows — Premium-quality matching iZotope RX / WaveLab aesthetics.
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent.parent / "Aurik10" / "resources" / "carrier_icons"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 128
HALF = SIZE // 2


def _new() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _save(img: Image.Image, name: str) -> None:
    path = OUT / f"{name}.png"
    img.save(path, "PNG", optimize=True)
    print(f"  ✓ {path.name}  ({SIZE}×{SIZE})")


def _rounded_rect(draw: ImageDraw.ImageDraw, bbox: list[int], fill: tuple, radius: int = 8):
    draw.rounded_rectangle(bbox, radius=radius, fill=fill)


def _text_center(
    draw: ImageDraw.ImageDraw, cx: int, cy: int, text: str, fill: tuple = (255, 255, 255, 230), size: int = 14
):
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((cx - tw // 2, cy - th // 2), text, fill=fill, font=font)


def _shadow(img: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 50), (2, 3), mask=img.split()[3])
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas = Image.alpha_composite(canvas, img)
    return canvas


def _disc(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r: int,
    color: tuple[int, int, int],
    groove_color: tuple[int, int, int],
    label_color: tuple[int, int, int],
    label_r: int = 20,
    groove_step: int = 4,
):
    # Outer shadow ring
    draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=(0, 0, 0, 80))
    # Main disc
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    # Grooves
    for gr in range(label_r + 4, r - 2, groove_step):
        alpha = 40 + int(20 * math.sin(gr * 0.3))
        draw.ellipse([cx - gr, cy - gr, cx + gr, cy + gr], outline=(*groove_color, alpha), width=1)
    # Specular highlight (top-left)
    for i in range(12):
        a = max(0, 60 - i * 5)
        off_x, off_y = -r // 3, -r // 3
        hr = r - 10 - i * 2
        if hr > 0:
            draw.ellipse([cx + off_x - hr, cy + off_y - hr, cx + off_x + hr, cy + off_y + hr], fill=(255, 255, 255, a))
    # Label
    draw.ellipse([cx - label_r, cy - label_r, cx + label_r, cy + label_r], fill=label_color)
    draw.ellipse([cx - label_r + 3, cy - label_r + 3, cx + label_r - 8, cy + label_r - 8], fill=(255, 255, 255, 30))
    # Spindle hole
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(40, 40, 40, 255))


def _draw_reel(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, tape_r: int, flange_color: tuple, hub_color: tuple):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=flange_color)
    draw.ellipse([cx - tape_r, cy - tape_r, cx + tape_r, cy + tape_r], fill=(90, 55, 30))
    for tr in range(tape_r - 2, 10, -3):
        a = 60 + int(30 * math.sin(tr * 0.5))
        draw.ellipse([cx - tr, cy - tr, cx + tr, cy + tr], outline=(110, 70, 35, a), width=1)
    draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=hub_color)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(50, 50, 55))
    # Metallic highlight
    for i in range(6):
        a = max(0, 50 - i * 9)
        draw.ellipse([cx - r + 5 + i, cy - r + 5 + i, cx - r + 20 - i, cy - r + 20 - i], fill=(255, 255, 255, a))


# ── Generators ─────────────────────────────────────────────────────────────


def gen_vinyl() -> Image.Image:
    img, draw = _new()
    _disc(draw, HALF, HALF, 56, (20, 20, 25), (60, 60, 70), label_color=(180, 40, 40), label_r=18, groove_step=3)
    return _shadow(img)


def gen_shellac() -> Image.Image:
    img, draw = _new()
    _disc(draw, HALF, HALF, 56, (60, 35, 15), (90, 60, 30), label_color=(200, 170, 80), label_r=20, groove_step=5)
    return _shadow(img)


def gen_lacquer_disc() -> Image.Image:
    img, draw = _new()
    _disc(draw, HALF, HALF, 56, (70, 55, 20), (100, 80, 35), label_color=(220, 195, 100), label_r=16, groove_step=4)
    # Golden sheen
    for i in range(8):
        a = max(0, 45 - i * 6)
        draw.ellipse([HALF - 40 + i, HALF - 50 + i, HALF + 20 - i, HALF - 10 - i], fill=(255, 230, 140, a))
    return _shadow(img)


def gen_wax_cylinder() -> Image.Image:
    img, draw = _new()
    cx, cy = HALF, HALF
    w, h = 36, 52
    _rounded_rect(draw, [cx - w, cy - h + 10, cx + w, cy + h - 10], fill=(160, 120, 60), radius=4)
    for x_off in range(-w + 4, w, 5):
        alpha = 90 + int(30 * math.sin(x_off * 0.4))
        draw.line([(cx + x_off, cy - h + 14), (cx + x_off, cy + h - 14)], fill=(120, 85, 35, alpha), width=1)
    draw.ellipse([cx - w, cy - h, cx + w, cy - h + 20], fill=(180, 140, 75))
    draw.ellipse([cx - w + 3, cy - h + 3, cx + w - 3, cy - h + 17], fill=(200, 165, 95, 120))
    draw.ellipse([cx - w, cy + h - 20, cx + w, cy + h], fill=(130, 95, 45))
    for i in range(6):
        draw.line(
            [(cx - w + 10, cy - h + 12 + i), (cx - w + 10, cy + h - 12 + i)],
            fill=(255, 230, 170, max(0, 50 - i * 10)),
            width=2,
        )
    return _shadow(img)


def gen_wire_recording() -> Image.Image:
    img, draw = _new()
    cx, cy = HALF, HALF
    draw.ellipse([cx - 44, cy - 44, cx + 44, cy + 44], fill=(120, 120, 130))
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(155, 155, 165))
    for r in range(12, 38, 3):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(190, 190, 200, 100), width=1)
    for i in range(8):
        a = max(0, 60 - i * 8)
        draw.ellipse([cx - 30 - i, cy - 35 - i, cx - 5 + i, cy - 10 + i], fill=(230, 230, 240, a))
    draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(80, 80, 90))
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(60, 60, 65))
    return _shadow(img)


def gen_reel_tape() -> Image.Image:
    img, draw = _new()
    _rounded_rect(draw, [8, 20, 120, 108], fill=(55, 50, 48), radius=6)
    _draw_reel(draw, 40, HALF, 28, 22, (170, 170, 175), (100, 100, 110))
    _draw_reel(draw, 88, HALF, 28, 16, (170, 170, 175), (100, 100, 110))
    draw.line([(40, HALF + 28), (88, HALF + 28)], fill=(90, 55, 30), width=2)
    draw.line([(40, HALF - 28), (88, HALF - 28)], fill=(90, 55, 30, 120), width=1)
    return _shadow(img)


def gen_tape() -> Image.Image:
    img, draw = _new()
    _draw_reel(draw, HALF, HALF, 46, 36, (140, 140, 150), (90, 90, 100))
    draw.arc([HALF - 46, HALF - 46, HALF + 46, HALF + 46], start=30, end=90, fill=(90, 55, 30, 200), width=3)
    return _shadow(img)


def gen_cassette() -> Image.Image:
    img, draw = _new()
    _cx, _cy = HALF, HALF
    _rounded_rect(draw, [12, 28, 116, 100], fill=(60, 58, 65), radius=8)
    _rounded_rect(draw, [16, 32, 112, 96], fill=(75, 73, 80), radius=6)
    # Tape windows
    draw.ellipse([30, 48, 58, 76], fill=(30, 30, 35))
    draw.ellipse([70, 48, 98, 76], fill=(30, 30, 35))
    # Hubs
    draw.ellipse([38, 56, 50, 68], fill=(90, 55, 30))
    draw.ellipse([78, 56, 90, 68], fill=(90, 55, 30))
    draw.ellipse([42, 60, 46, 64], fill=(60, 60, 65))
    draw.ellipse([82, 60, 86, 64], fill=(60, 60, 65))
    # Tape
    draw.rectangle([55, 70, 73, 73], fill=(90, 55, 30))
    # Label
    _rounded_rect(draw, [24, 33, 104, 47], fill=(230, 225, 210), radius=3)
    draw.line([(30, 38), (98, 38)], fill=(80, 80, 85, 100), width=1)
    draw.line([(30, 42), (80, 42)], fill=(80, 80, 85, 70), width=1)
    # Screw holes
    draw.ellipse([18, 84, 24, 90], fill=(45, 45, 50))
    draw.ellipse([104, 84, 110, 90], fill=(45, 45, 50))
    # Edge highlight
    draw.line([(16, 32), (112, 32)], fill=(255, 255, 255, 40), width=1)
    return _shadow(img)


def _gen_cd_base(color_shift: int = 0) -> Image.Image:
    img, draw = _new()
    cx, cy = HALF, HALF
    r = 54
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(225, 228, 232))
    # Rainbow reflection
    for angle in range(0, 360, 2):
        rad = math.radians(angle)
        for ring in range(20, r - 2, 2):
            hue = ((angle + ring * 3 + color_shift) % 360) / 360.0
            rgb = colorsys.hls_to_rgb(hue, 0.75, 0.35)
            c = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 40)
            x = cx + int(ring * math.cos(rad))
            y = cy + int(ring * math.sin(rad))
            if 0 <= x < SIZE and 0 <= y < SIZE:
                img.putpixel((x, y), c)
    for ring in range(18, r - 3, 3):
        draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(200, 205, 210, 50), width=1)
    # Specular highlight
    for i in range(10):
        a = max(0, 80 - i * 8)
        ox, oy = -15, -20
        hr = 35 - i * 2
        if hr > 0:
            draw.ellipse([cx + ox - hr, cy + oy - hr, cx + ox + hr, cy + oy + hr], fill=(255, 255, 255, a))
    # Center hole
    draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(0, 0, 0, 0))
    draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], outline=(180, 180, 185), width=2)
    draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], outline=(190, 190, 195, 80), width=1)
    return _shadow(img)


def gen_cd() -> Image.Image:
    return _gen_cd_base(color_shift=0)


def gen_cd_digital() -> Image.Image:
    return _gen_cd_base(color_shift=120)


def gen_dat() -> Image.Image:
    img, draw = _new()
    cx, _cy = HALF, HALF
    _rounded_rect(draw, [18, 30, 110, 98], fill=(35, 35, 42), radius=6)
    _rounded_rect(draw, [22, 34, 106, 94], fill=(50, 48, 58), radius=4)
    _rounded_rect(draw, [32, 58, 96, 80], fill=(20, 20, 25), radius=3)
    draw.rectangle([36, 62, 92, 76], fill=(85, 50, 28))
    draw.ellipse([42, 64, 54, 74], fill=(30, 30, 35))
    draw.ellipse([74, 64, 86, 74], fill=(30, 30, 35))
    draw.ellipse([46, 67, 50, 71], fill=(60, 60, 65))
    draw.ellipse([78, 67, 82, 71], fill=(60, 60, 65))
    _rounded_rect(draw, [26, 36, 102, 55], fill=(70, 130, 180), radius=3)
    _text_center(draw, cx, 45, "DAT", fill=(255, 255, 255, 230), size=12)
    draw.line([(22, 34), (106, 34)], fill=(255, 255, 255, 35), width=1)
    return _shadow(img)


def gen_minidisc() -> Image.Image:
    img, draw = _new()
    cx, cy = HALF, HALF
    _rounded_rect(draw, [16, 22, 112, 106], fill=(45, 42, 55), radius=8)
    _rounded_rect(draw, [20, 26, 108, 102], fill=(60, 55, 70), radius=6)
    draw.ellipse([34, 36, 94, 96], fill=(30, 28, 35))
    draw.ellipse([38, 40, 90, 92], fill=(200, 205, 215))
    for ring in range(8, 24, 2):
        hs = ring * 15
        rc = 180 + int(20 * math.sin(hs * 0.05))
        gc = 190 + int(20 * math.sin(hs * 0.07))
        bc = 210 + int(15 * math.sin(hs * 0.09))
        draw.ellipse([cx - ring, cy + 2 - ring, cx + ring, cy + 2 + ring], outline=(rc, gc, bc, 50), width=1)
    draw.ellipse([cx - 6, cy - 4, cx + 6, cy + 8], fill=(50, 48, 55))
    _rounded_rect(draw, [38, 80, 90, 94], fill=(70, 65, 80), radius=2)
    _text_center(draw, cx, 30, "MD", fill=(180, 175, 195, 200), size=10)
    return _shadow(img)


def _gen_digital_file(
    ext: str, color: tuple[int, int, int], accent: tuple[int, int, int], damaged: bool = False
) -> Image.Image:
    img, draw = _new()
    pts = [(30, 16), (85, 16), (100, 31), (100, 112), (30, 112)]
    draw.polygon(pts, fill=color)
    draw.polygon([(85, 16), (100, 31), (85, 31)], fill=accent)
    draw.line([(32, 18), (83, 18)], fill=(255, 255, 255, 50), width=1)
    # Sound wave bars
    wave_y = 58
    for i, h in enumerate([12, 20, 28, 20, 14, 24, 18, 10, 22, 16]):
        x = 40 + i * 6
        if damaged and i in (3, 4, 7):
            c = (200, 60, 60, 180)
        else:
            c = (*accent, 200)
        draw.rectangle([x, wave_y - h // 2, x + 3, wave_y + h // 2], fill=c)
    _rounded_rect(draw, [32, 82, 98, 104], fill=accent, radius=4)
    _text_center(draw, 65, 93, ext, fill=(255, 255, 255, 240), size=13)
    if damaged:
        draw.line([(35, 30), (95, 80)], fill=(200, 50, 50, 160), width=2)
        draw.line([(90, 30), (40, 75)], fill=(200, 50, 50, 120), width=2)
    return _shadow(img)


def gen_mp3_high() -> Image.Image:
    return _gen_digital_file("MP3", (55, 65, 85), (70, 140, 210))


def gen_mp3_low() -> Image.Image:
    return _gen_digital_file("MP3", (65, 60, 55), (180, 130, 60))


def gen_damaged_mp3() -> Image.Image:
    return _gen_digital_file("MP3", (70, 50, 50), (180, 60, 60), damaged=True)


def gen_aac() -> Image.Image:
    return _gen_digital_file("AAC", (50, 65, 60), (60, 170, 130))


def gen_streaming() -> Image.Image:
    img, draw = _new()
    cx, _cy = HALF, HALF
    # Cloud
    draw.ellipse([28, 40, 72, 78], fill=(80, 130, 190))
    draw.ellipse([52, 35, 100, 75], fill=(80, 130, 190))
    draw.ellipse([38, 30, 80, 65], fill=(90, 140, 200))
    draw.rectangle([38, 56, 90, 76], fill=(80, 130, 190))
    draw.ellipse([42, 32, 76, 58], fill=(110, 155, 215, 100))
    # Signal arcs
    for i, r_a in enumerate([14, 22, 30]):
        a = max(0, 180 - i * 50)
        draw.arc([cx - r_a, 72 - r_a // 2, cx + r_a, 72 + r_a], start=20, end=160, fill=(100, 200, 255, a), width=3)
    # Download arrow
    draw.polygon([(cx - 6, 90), (cx + 6, 90), (cx, 102)], fill=(255, 255, 255, 180))
    draw.rectangle([cx - 2, 80, cx + 2, 92], fill=(255, 255, 255, 180))
    return _shadow(img)


def gen_unknown() -> Image.Image:
    img, draw = _new()
    cx, cy = HALF, HALF
    draw.ellipse([cx - 48, cy - 48, cx + 48, cy + 48], fill=(80, 80, 88))
    draw.ellipse([cx - 44, cy - 44, cx + 44, cy + 44], fill=(95, 95, 105))
    for r in range(15, 42, 5):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(110, 110, 120, 60), width=1)
    _text_center(draw, cx, cy - 4, "?", fill=(200, 200, 210, 220), size=36)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(60, 60, 68))
    return _shadow(img)


# ── Main ───────────────────────────────────────────────────────────────────

GENERATORS: dict[str, callable] = {  # type: ignore[valid-type]
    "vinyl": gen_vinyl,
    "shellac": gen_shellac,
    "lacquer_disc": gen_lacquer_disc,
    "wax_cylinder": gen_wax_cylinder,
    "wire_recording": gen_wire_recording,
    "reel_tape": gen_reel_tape,
    "tape": gen_tape,
    "cassette": gen_cassette,
    "cd": gen_cd,
    "cd_digital": gen_cd_digital,
    "dat": gen_dat,
    "minidisc": gen_minidisc,
    "mp3_high": gen_mp3_high,
    "mp3_low": gen_mp3_low,
    "damaged_mp3": gen_damaged_mp3,
    "aac": gen_aac,
    "streaming": gen_streaming,
    "unknown": gen_unknown,
}

if __name__ == "__main__":
    print(f"Generating {len(GENERATORS)} premium carrier icons → {OUT}")
    for name, gen_fn in GENERATORS.items():
        icon = gen_fn()  # type: ignore[misc]
        _save(icon, name)
    print("Done.")
