"""Blending: multiband (Laplacian pyramid), feather and hard-seam blenders.

Layers are fed one at a time (streaming) together with their seam mask, so
memory use is bounded by the canvas accumulators, not the number of images.
Horizontal wrap-around (360 degree panoramas) is supported at every level.
"""
from __future__ import annotations

import cv2
import numpy as np

KERNEL = np.array([1, 4, 6, 4, 1], np.float32) / 16.0


def _down(x, wrap=False):
    if wrap:
        p = np.concatenate([x[:, -4:], x, x[:, :4]], axis=1)
        d = cv2.pyrDown(p, dstsize=((p.shape[1] + 1) // 2, (p.shape[0] + 1) // 2))
        return np.ascontiguousarray(d[:, 2:-2])
    return cv2.pyrDown(x, dstsize=((x.shape[1] + 1) // 2, (x.shape[0] + 1) // 2))


def _up(x, shape, wrap=False):
    h, w = shape[:2]
    if wrap:
        p = np.concatenate([x[:, -2:], x, x[:, :2]], axis=1)
        u = cv2.pyrUp(p, dstsize=(p.shape[1] * 2, h))
        return np.ascontiguousarray(u[:, 4:4 + w])
    return cv2.pyrUp(x, dstsize=(w, h))


def push_pull_fill(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid pixels with smooth extrapolated colour (premultiplied push-pull)."""
    a = valid.astype(np.float32)
    if a.min() > 0.999:
        return img
    P = [img * a[..., None]]
    A = [a]
    while min(P[-1].shape[:2]) > 2 and A[-1].min() < 0.999:
        P.append(_down(P[-1]))
        A.append(_down(A[-1]))
    out = P[-1] / np.maximum(A[-1], 1e-6)[..., None]
    for l in range(len(P) - 2, -1, -1):
        up = _up(out, P[l].shape)
        al = np.clip(A[l], 0, 1)[..., None]
        out = P[l] + (1 - al) * up  # premultiplied "over" compositing
    return out.astype(np.float32)


def auto_levels(width, height, overlap_px: float | None = None, feather: float = 0.5, max_levels: int = 9):
    """Number of pyramid levels: softer feather -> more levels."""
    base = int(np.floor(np.log2(max(8, min(width, height)) / 8)))
    base = max(1, min(base, max_levels))
    lv = int(round(2 + feather * (base - 2))) if base > 2 else base
    if overlap_px:
        lv = min(lv, max(2, int(np.ceil(np.log2(max(overlap_px, 4))))))
    return max(1, lv)


class MultiBandBlender:
    """Streaming Laplacian-pyramid blender.

    Internally the canvas is never wrapped: for 360-degree panoramas it carries a
    margin on both sides into which wrapped copies of the layers are fed, so the
    pyramid sees continuous content across the seam and the output width needs no
    rounding.
    """

    def __init__(self, width: int, height: int, levels: int = 6, wrap: bool = False, channels: int = 3):
        self.levels = levels
        self.wrap = wrap
        self.W, self.H = width, height
        self.unit = 2 ** levels
        U = self.unit
        self.mx = 4 * U if wrap else 0  # margin (left) in internal coordinates
        self.Wp = int(np.ceil((width + 2 * self.mx) / U) * U)
        self.Hp = int(np.ceil(height / U) * U)
        self.acc, self.wacc = [], []
        w, h = self.Wp, self.Hp
        for _ in range(levels + 1):
            self.acc.append(np.zeros((h, w, channels), np.float32))
            self.wacc.append(np.zeros((h, w), np.float32))
            w, h = w // 2, h // 2
        self.cover = np.zeros((height, self.Wp), np.float32)

    def feed(self, rgb: np.ndarray, weight: np.ndarray, x0: int, y0: int, valid: np.ndarray | None = None):
        """rgb (h,w,3), weight (h,w) seam mask 0..1, placed at canvas (x0, y0)."""
        if not self.wrap:
            self._feed(rgb, weight, x0, y0, valid)
            return
        w = weight.shape[1]
        for k in (-1, 0, 1):
            xi = x0 + self.mx + k * self.W
            if xi + w > 0 and xi < self.Wp:
                self._feed(rgb, weight, xi, y0, valid)

    def _feed(self, rgb, weight, x0, y0, valid):
        h, w = weight.shape
        U = self.unit
        margin = U  # room for the coarse levels
        ax0 = max(0, int(np.floor((x0 - margin) / U) * U))
        ay0 = max(0, int(np.floor((y0 - margin) / U) * U))
        ax1 = min(self.Wp, int(np.ceil((x0 + w + margin) / U) * U))
        ay1 = min(self.Hp, int(np.ceil((y0 + h + margin) / U) * U))
        pw, ph = ax1 - ax0, ay1 - ay0
        if pw <= 0 or ph <= 0:
            return
        sx0, sy0 = max(0, ax0 - x0), max(0, ay0 - y0)
        dx0, dy0 = max(0, x0 - ax0), max(0, y0 - ay0)
        cw = min(w - sx0, pw - dx0)
        ch = min(h - sy0, ph - dy0)
        if cw <= 0 or ch <= 0:
            return
        M = np.zeros((ph, pw), np.float32)
        M[dy0:dy0 + ch, dx0:dx0 + cw] = weight[sy0:sy0 + ch, sx0:sx0 + cw]
        if not M.any():
            return
        I = np.zeros((ph, pw, rgb.shape[2]), np.float32)
        V = np.zeros((ph, pw), bool)
        I[dy0:dy0 + ch, dx0:dx0 + cw] = rgb[sy0:sy0 + ch, sx0:sx0 + cw]
        vv = valid if valid is not None else weight > 0
        V[dy0:dy0 + ch, dx0:dx0 + cw] = vv[sy0:sy0 + ch, sx0:sx0 + cw]
        I = push_pull_fill(I, V)
        gm, cur = M, I
        for l in range(self.levels + 1):
            if l < self.levels:
                nxt = _down(cur)
                lap = cur - _up(nxt, cur.shape)
            else:
                lap = cur
            yl, xl = ay0 >> l, ax0 >> l
            hh = min(gm.shape[0], self.acc[l].shape[0] - yl)
            ww = min(gm.shape[1], self.acc[l].shape[1] - xl)
            self.acc[l][yl:yl + hh, xl:xl + ww] += (lap * gm[..., None])[:hh, :ww]
            self.wacc[l][yl:yl + hh, xl:xl + ww] += gm[:hh, :ww]
            if l < self.levels:
                cur = nxt
                gm = _down(gm)
        cy1 = min(self.H, ay0 + ph)
        if cy1 > ay0:
            tgt = self.cover[ay0:cy1, ax0:ax1]
            np.maximum(tgt, V[:cy1 - ay0].astype(np.float32), out=tgt)

    def result(self):
        out = None
        for l in range(self.levels, -1, -1):
            lvl = self.acc[l] / np.maximum(self.wacc[l], 1e-8)[..., None]
            lvl[self.wacc[l] <= 1e-8] = 0
            out = lvl if out is None else _up(out, lvl.shape) + lvl
            self.acc[l] = self.wacc[l] = None  # free memory as we go
        out = np.ascontiguousarray(out[:self.H, self.mx:self.mx + self.W])
        return out, np.ascontiguousarray(self.cover[:, self.mx:self.mx + self.W])


class FeatherBlender:
    """Weighted average with blurred seam masks (alpha blending)."""

    def __init__(self, width, height, wrap=False, sigma: float = 8.0, channels=3):
        self.W, self.H, self.wrap, self.sigma = width, height, wrap, sigma
        self.acc = np.zeros((height, width, channels), np.float32)
        self.wacc = np.zeros((height, width), np.float32)
        self.cover = np.zeros((height, width), np.float32)

    def feed(self, rgb, weight, x0, y0, valid=None):
        h, w = weight.shape
        pad = int(3 * self.sigma) + 1
        M = np.pad(weight.astype(np.float32), pad)
        if self.sigma > 0:
            M = cv2.GaussianBlur(M, (0, 0), self.sigma)
        M = M[pad:-pad, pad:-pad]
        vv = valid if valid is not None else weight > 0
        M = M * vv
        self._add(rgb * M[..., None], M, vv.astype(np.float32), x0, y0)

    def _add(self, val, wt, cv, x0, y0):
        h, w = wt.shape
        cols = np.arange(x0, x0 + w)
        if self.wrap:
            cols = cols % self.W
        keep = (cols >= 0) & (cols < self.W)
        ys0, ys1 = max(0, y0), min(self.H, y0 + h)
        sy = slice(ys0 - y0, ys1 - y0)
        c = cols[keep]
        self.acc[ys0:ys1, c] += val[sy][:, keep]
        self.wacc[ys0:ys1, c] += wt[sy][:, keep]
        self.cover[ys0:ys1, c] = np.maximum(self.cover[ys0:ys1, c], cv[sy][:, keep])

    def result(self):
        out = self.acc / np.maximum(self.wacc, 1e-8)[..., None]
        return out, self.cover


class NoBlender(FeatherBlender):
    def __init__(self, width, height, wrap=False, channels=3):
        super().__init__(width, height, wrap, 0.0, channels)
