"""Draw the PanoForge app icon (1024 px PNG + macOS .icns)."""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(size=1024):
    S = size * 4  # supersample
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # rounded tile with a vertical gradient
    grad = np.zeros((S, S, 4), np.uint8)
    t = np.linspace(0, 1, S)[:, None]
    top, bot = np.array([34, 43, 58]), np.array([14, 18, 26])
    grad[..., :3] = (top * (1 - t) + bot * t).astype(np.uint8)[:, None, :]
    grad[..., 3] = 255
    mask = Image.new("L", (S, S), 0)
    m = int(S * 0.09)
    ImageDraw.Draw(mask).rounded_rectangle([m, m, S - m, S - m], radius=int(S * 0.19), fill=255)
    tile = Image.fromarray(grad, "RGBA")
    img.paste(tile, (0, 0), mask)
    d = ImageDraw.Draw(img)
    cx, cy = S / 2, S * 0.45
    R = S * 0.255
    # globe body
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([cx - R * 1.08, cy - R * 1.08, cx + R * 1.08, cy + R * 1.08], fill=(61, 139, 253, 90))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(S * 0.03)))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(23, 64, 140, 255))
    # lat/long grid of an equirectangular sphere, seen slightly from above
    tilt = math.radians(18)
    lw = max(2, int(S * 0.006))

    def proj(lon, lat):
        x, y, z = math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)
        y, z = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
        return cx + R * x, cy - R * y, z

    def curve(pts, color):
        seg = []
        for x, y, z in pts:
            if z >= 0:
                seg.append((x, y))
            elif seg:
                if len(seg) > 1:
                    d.line(seg, fill=color, width=lw, joint="curve")
                seg = []
        if len(seg) > 1:
            d.line(seg, fill=color, width=lw, joint="curve")

    for k in range(12):
        lon = k / 12 * 2 * math.pi
        curve([proj(lon, math.radians(a)) for a in np.linspace(-90, 90, 120)], (150, 196, 255, 200))
    for lat in (-60, -30, 0, 30, 60):
        col = (255, 214, 10, 255) if lat == 0 else (150, 196, 255, 200)
        curve([proj(math.radians(a), math.radians(lat)) for a in np.linspace(-180, 180, 240)], col)
    # bright rim
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(200, 225, 255, 255), width=int(lw * 1.6))
    # a panorama band wrapping around the globe
    band = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bw = S * 0.6
    bh = S * 0.115
    bd.rounded_rectangle([cx - bw / 2, cy + R * 1.05, cx + bw / 2, cy + R * 1.05 + bh], radius=int(bh * 0.25), fill=(61, 139, 253, 255))
    # mountains on the band
    base = cy + R * 1.05 + bh
    pts = [(cx - bw / 2 + bw * x, base - bh * h) for x, h in
           [(0.04, 0.15), (0.18, 0.62), (0.3, 0.32), (0.45, 0.78), (0.6, 0.38), (0.74, 0.7), (0.86, 0.3), (0.96, 0.15)]]
    bd.polygon([(cx - bw / 2 + bw * 0.04, base - 2)] + pts + [(cx - bw / 2 + bw * 0.96, base - 2)], fill=(255, 255, 255, 235))
    img = Image.alpha_composite(img, band)
    return img.resize((size, size), Image.LANCZOS)


def main():
    png = os.path.join(HERE, "PanoForge.png")
    draw().save(png)
    if sys.platform == "darwin":
        iconset = os.path.join(HERE, "PanoForge.iconset")
        os.makedirs(iconset, exist_ok=True)
        big = Image.open(png)
        for s in (16, 32, 64, 128, 256, 512):
            big.resize((s, s), Image.LANCZOS).save(os.path.join(iconset, f"icon_{s}x{s}.png"))
            big.resize((s * 2, s * 2), Image.LANCZOS).save(os.path.join(iconset, f"icon_{s}x{s}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(HERE, "PanoForge.icns")], check=True)
    print(png)


if __name__ == "__main__":
    main()
