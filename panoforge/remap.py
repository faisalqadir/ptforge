"""Panorama canvas and per-image warping (remapping) into it."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as G
from .masks import MASK_INCLUDE, crop_mask, load_paint_mask
from .project import Project

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "default": cv2.INTER_CUBIC,
}

WRAP_PROJECTIONS = {"equirectangular", "cylindrical", "mercator", "architectural"}


class Canvas:
    """Maps panorama pixels <-> projection plane (u, v) <-> world directions."""

    def __init__(self, p: Project, width: int | None = None):
        self.p = p
        self.proj = p.projection()
        hf = math.radians(p.panorama.hfov) / 2
        vf = math.radians(p.panorama.vfov) / 2
        self.u_ext = 2 * self.proj.u_of_hfov(hf)
        self.v_ext = 2 * self.proj.v_of_vfov(vf)
        if width is None or width <= 0:
            width = p.panorama.width
        if not width:
            from .align import optimum_size

            width = optimum_size(p)[0]
        self.width = int(width)
        self.scale = self.width / self.u_ext  # pixels per unit of u
        self.height = max(1, int(round(self.v_ext * self.scale)))
        self.wrap = p.panorama.projection in WRAP_PROJECTIONS and p.panorama.hfov >= 359.99
        self.v_offset = p.panorama.v_offset * self.v_ext

    def scaled(self, width: int) -> "Canvas":
        return Canvas(self.p, width)

    def uv(self, x, y):
        u = (x + 0.5 - self.width / 2.0) / self.scale
        v = (self.height / 2.0 - (y + 0.5)) / self.scale + self.v_offset
        return u, v

    def xy(self, u, v):
        x = u * self.scale + self.width / 2.0 - 0.5
        y = self.height / 2.0 - (v - self.v_offset) * self.scale - 0.5
        return x, y

    def dirs(self, x, y):
        u, v = self.uv(x, y)
        d, valid = self.proj.inverse(u, v)
        return d, valid

    def crop_rect(self):
        l, t, r, b = self.p.panorama.crop
        x0, y0 = int(round(l * self.width)), int(round(t * self.height))
        x1, y1 = int(round(r * self.width)), int(round(b * self.height))
        return max(0, x0), max(0, y0), min(self.width, x1), min(self.height, y1)


@dataclass
class Region:
    x0: int
    y0: int
    w: int
    h: int  # x0 may be negative / x0+w may exceed canvas width when wrapping

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def y1(self):
        return self.y0 + self.h


def footprint(p: Project, idx: int, canvas: Canvas, lowres: int = 720, cam: G.Camera | None = None) -> Region | None:
    """Bounding region of image idx on the canvas (handles 360 wrap and poles)."""
    cam = cam or p.camera(idx)
    s = min(1.0, lowres / max(canvas.width, canvas.height))
    lw, lh = max(8, int(canvas.width * s)), max(4, int(canvas.height * s))
    xs = (np.arange(lw) + 0.5) / lw * canvas.width - 0.5
    ys = (np.arange(lh) + 0.5) / lh * canvas.height - 0.5
    X, Y = np.meshgrid(xs, ys)
    d, valid = canvas.dirs(X, Y)
    x, y, ok = cam.world_to_pixel(d)
    m = valid & ok & (x > -1.5) & (x < cam.width + 0.5) & (y > -1.5) & (y < cam.height + 0.5)
    if not m.any():
        return None
    # dilate one low-res cell to be safe
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rows = np.nonzero(m.any(axis=1))[0]
    cols = m.any(axis=0)
    cell_w, cell_h = canvas.width / lw, canvas.height / lh
    y0 = max(0, int(math.floor(rows[0] * cell_h)) - 2)
    y1 = min(canvas.height, int(math.ceil((rows[-1] + 1) * cell_h)) + 2)
    if canvas.wrap and cols.all():
        return Region(0, y0, canvas.width, y1 - y0)
    if canvas.wrap:
        empty = ~cols
        ext = np.concatenate([empty, empty])
        best_len, best_start, run, start = 0, 0, 0, 0
        for k, e in enumerate(ext):
            if e:
                if run == 0:
                    start = k
                run += 1
                if run > best_len:
                    best_len, best_start = run, start
            else:
                run = 0
        best_len = min(best_len, lw)
        c0 = (best_start + best_len) % lw  # first covered column after the gap
        ncols = lw - best_len
        x0 = int(math.floor(c0 * cell_w)) - 2
        x1 = int(math.ceil((c0 + ncols) * cell_w)) + 2
        if x0 >= canvas.width:
            x0 -= canvas.width
            x1 -= canvas.width
        return Region(x0, y0, min(x1 - x0, canvas.width), y1 - y0)
    cidx = np.nonzero(cols)[0]
    x0 = max(0, int(math.floor(cidx[0] * cell_w)) - 2)
    x1 = min(canvas.width, int(math.ceil((cidx[-1] + 1) * cell_w)) + 2)
    return Region(x0, y0, x1 - x0, y1 - y0)


def compute_maps(p: Project, idx: int, canvas: Canvas, region: Region, cam: G.Camera | None = None):
    """Source pixel coordinates for every canvas pixel of the region.

    Returns (map_x, map_y, valid) float32/float32/bool, shape (h, w).
    """
    cam = cam or p.camera(idx)
    xs = np.arange(region.x0, region.x1, dtype=np.float64)
    if canvas.wrap:
        xs = np.mod(xs, canvas.width)
    ys = np.arange(region.y0, region.y1, dtype=np.float64)
    mx = np.empty((region.h, region.w), np.float32)
    my = np.empty((region.h, region.w), np.float32)
    valid = np.empty((region.h, region.w), bool)
    step = max(1, 2_000_000 // max(region.w, 1))
    for r0 in range(0, region.h, step):
        r1 = min(region.h, r0 + step)
        X, Y = np.meshgrid(xs, ys[r0:r1])
        d, v = canvas.dirs(X, Y)
        x, y, ok = cam.world_to_pixel(d)
        ok &= v & (x >= -0.5) & (x <= cam.width - 0.5) & (y >= -0.5) & (y <= cam.height - 0.5)
        mx[r0:r1] = x
        my[r0:r1] = y
        valid[r0:r1] = ok
    return mx, my, valid


@dataclass
class Layer:
    idx: int
    region: Region
    rgb: np.ndarray  # float32 (h, w, 3) linear scene (or encoded, see stitcher)
    alpha: np.ndarray  # float32 (h, w) 0..1 coverage (anti-aliased edge)
    include: np.ndarray | None = None  # bool force-include (green mask)
    map_x: np.ndarray | None = None
    map_y: np.ndarray | None = None


def _edge_alpha(map_x, map_y, valid, w, h, feather_px=1.0):
    """Soft coverage at the image border to avoid jaggies."""
    dx = np.minimum(map_x + 0.5, (w - 0.5) - map_x)
    dy = np.minimum(map_y + 0.5, (h - 0.5) - map_y)
    a = np.clip(np.minimum(dx, dy) / feather_px + 0.5, 0, 1).astype(np.float32)
    a[~valid] = 0
    return a


def warp_image(p: Project, idx: int, canvas: Canvas, region: Region | None = None, interpolator: str = "lanczos",
               source=None, photometric: bool = True, ev_ref: float | None = None, keep_maps: bool = False) -> Layer | None:
    """Warp one source image into the canvas.  Output is linear scene radiance."""
    from .imageio import load_image
    from .photometric import correct_to_scene

    cam = p.camera(idx)
    region = region or footprint(p, idx, canvas, cam=cam)
    if region is None or region.w <= 0 or region.h <= 0:
        return None
    mx, my, valid = compute_maps(p, idx, canvas, region, cam)
    if not valid.any():
        return None
    im = p.images[idx]
    if source is None:
        source = load_image(p.abspath(im.path)).rgb
    src = source
    sx = src.shape[1] / im.width
    sy = src.shape[0] / im.height
    # antialiasing: shrink the source when the output samples it sparsely
    ratio = canvas.scale / max(cam.focal_px * min(sx, sy), 1e-9)
    if ratio < 0.7:
        f = max(ratio, 1.0 / 64)
        src = cv2.resize(src, (max(1, int(round(src.shape[1] * f))), max(1, int(round(src.shape[0] * f)))), interpolation=cv2.INTER_AREA)
        sx *= src.shape[1] / source.shape[1]
        sy *= src.shape[0] / source.shape[0]
    smx = ((mx + 0.5) * sx - 0.5).astype(np.float32)
    smy = ((my + 0.5) * sy - 0.5).astype(np.float32)
    rgb = _remap(src, smx, smy, INTERP.get(interpolator, cv2.INTER_CUBIC))
    alpha = _edge_alpha(mx, my, valid, im.width, im.height, feather_px=max(1.0, 1.0 / max(ratio, 1e-3)))
    # crop + paint masks (sampled with nearest neighbour)
    include = None
    cm = crop_mask(p, idx, im.width, im.height)
    pm = load_paint_mask(p, idx, im.width, im.height)
    if cm is not None:
        a = cv2.remap(cm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255.0
        alpha *= a
    if pm is not None:
        lab = cv2.remap(pm, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha[lab == 1] = 0
        if (lab == MASK_INCLUDE).any():
            include = (lab == MASK_INCLUDE) & (alpha > 0)
    if photometric:
        rgb = correct_to_scene(p, idx, rgb, mx, my, ev_ref)
    rgb[alpha <= 0] = 0
    return Layer(idx, region, rgb, alpha, include, mx if keep_maps else None, my if keep_maps else None)


def _remap(src, mx, my, interp):
    # cv2.remap is limited to 32767 px maps; split big regions into tiles
    h, w = mx.shape
    if h < 32000 and w < 32000:
        out = cv2.remap(src, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
    else:
        out = np.empty((h, w, src.shape[2]), np.float32)
        for y0 in range(0, h, 16384):
            for x0 in range(0, w, 16384):
                out[y0:y0 + 16384, x0:x0 + 16384] = cv2.remap(src, mx[y0:y0 + 16384, x0:x0 + 16384], my[y0:y0 + 16384, x0:x0 + 16384],
                                                             interp, borderMode=cv2.BORDER_REFLECT)
    if interp in (cv2.INTER_CUBIC, cv2.INTER_LANCZOS4):
        np.maximum(out, 0, out=out)  # ringing below zero
    return out
