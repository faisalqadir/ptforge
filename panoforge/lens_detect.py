"""Automatic lens detection: circular-fisheye image circles and fisheye heuristics."""
from __future__ import annotations

import math

import cv2
import numpy as np


def detect_image_circle(rgb: np.ndarray, work: int = 1000):
    """Find the image circle of a circular fisheye.

    Returns (cx, cy, radius, confidence) in full-resolution pixels, or None when
    the frame does not look like a circular fisheye (no dark surround).
    """
    h, w = rgb.shape[:2]
    s = min(1.0, work / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(np.ascontiguousarray(small[..., :3]).astype(np.float32), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    sh, sw = g.shape
    # the corners of a circular fisheye frame are (nearly) black
    k = max(3, int(min(sh, sw) * 0.04))
    corners = [g[:k, :k], g[:k, -k:], g[-k:, :k], g[-k:, -k:]]
    dark = np.median([np.median(c) for c in corners])
    inside = np.percentile(g[sh // 3: 2 * sh // 3, sw // 3: 2 * sw // 3], 50)
    if not (dark < 0.06 and inside > dark + 0.08):
        return None
    thr = dark + 0.25 * (inside - dark)
    thr = max(thr, dark + 0.02)
    # radial scan: from a rough centre, find the outermost bright radius per angle
    m = (g > thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cy0, cx0 = cent[big][1], cent[big][0]
    rmax = math.hypot(sw, sh) / 2
    angles = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    rs = np.arange(0, rmax, 0.5)
    pts = []
    for a in angles:
        xs = cx0 + rs * math.cos(a)
        ys = cy0 + rs * math.sin(a)
        ok = (xs >= 0) & (xs < sw - 1) & (ys >= 0) & (ys < sh - 1)
        if not ok.any():
            continue
        vals = cv2.remap(g, xs[ok].astype(np.float32).reshape(-1, 1), ys[ok].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        bright = np.nonzero(vals > thr)[0]
        if not len(bright):
            continue
        last = bright[-1]
        if last >= ok.sum() - 2:  # reached the frame border: circle is cropped here
            continue
        pts.append((xs[ok][last], ys[ok][last]))
    pts = np.array(pts)
    if len(pts) < 60:
        return None
    best = None
    rng = np.random.default_rng(0)
    for _ in range(300):
        idx = rng.choice(len(pts), 3, replace=False)
        c = _circle3(pts[idx])
        if c is None:
            continue
        d = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
        inl = d < max(1.5, 0.004 * c[2])
        if best is None or inl.sum() > best[1].sum():
            best = (c, inl)
    if best is None:
        return None
    c = _fit_circle(pts[best[1]])
    conf = float(best[1].mean())
    cx, cy, r = (c[0] + 0.5) / s - 0.5, (c[1] + 0.5) / s - 0.5, c[2] / s
    if r < 0.2 * min(h, w) or conf < 0.5:
        return None
    return float(cx), float(cy), float(r), conf


def _circle3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    return ux, uy, math.hypot(x1 - ux, y1 - uy)


def _fit_circle(p):
    """Algebraic least-squares circle fit (Kasa)."""
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return cx, cy, r


def looks_fisheye(exif: dict) -> bool:
    """EXIF heuristic: very short 35mm-equivalent focal lengths are fisheyes."""
    f35 = exif.get("focal35") or (exif.get("focal_mm", 0) * exif.get("crop_factor", 0))
    lens = (exif.get("lens") or "").lower()
    return bool("fisheye" in lens or "fish-eye" in lens or (f35 and f35 < 12.0))
