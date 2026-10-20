"""Control point fine-tuning by normalised cross-correlation.

The template is cut from image i around (x1, y1) and resampled into image j's local
geometry using the current alignment, so matching works across large rotations and
lens distortion (fisheye edges).  Without a usable alignment a plain patch is used.
"""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from .imageio import load_image
from .project import Project

_GRAY: OrderedDict = OrderedDict()


def _gray(p: Project, idx: int) -> np.ndarray:
    path = p.abspath(p.images[idx].path)
    if path not in _GRAY:
        rgb = load_image(path).rgb
        _GRAY[path] = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
        while len(_GRAY) > 6:
            _GRAY.popitem(last=False)
    _GRAY.move_to_end(path)
    return _GRAY[path]


def _subpixel(res, loc):
    x, y = loc
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        den = l - 2 * c + r
        dx = 0.5 * (l - r) / den if abs(den) > 1e-9 else 0.0
    if 0 < y < res.shape[0] - 1:
        u, c, d = res[y - 1, x], res[y, x], res[y + 1, x]
        den = u - 2 * c + d
        dy = 0.5 * (u - d) / den if abs(den) > 1e-9 else 0.0
    return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))


def finetune(p: Project, i: int, x1: float, y1: float, j: int, x2: float, y2: float, radius: int = 24, search: int = 40) -> dict:
    gi, gj = _gray(p, i), _gray(p, j)
    ci, cj = p.camera(i), p.camera(j)
    R = radius
    # where does the feature land in j according to the current alignment?
    d = ci.pixel_to_world(np.array([x1]), np.array([y1]))
    px, py, ok = cj.world_to_pixel(d)
    warped = bool(ok[0]) and abs(px[0] - x2) <= 3 * search and abs(py[0] - y2) <= 3 * search
    if warped:
        u = np.arange(-R, R + 1, dtype=np.float64)
        U, V = np.meshgrid(u, u)
        dirs = cj.pixel_to_world(px[0] + U, py[0] + V)
        mx, my, okm = ci.world_to_pixel(dirs)
        tpl = cv2.remap(gi, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        tpl = cv2.getRectSubPix(gi, (2 * R + 1, 2 * R + 1), (float(x1), float(y1)))
    S = search + R
    win = cv2.getRectSubPix(gj, (2 * S + 1, 2 * S + 1), (float(x2), float(y2)))
    if tpl.std() < 1e-4 or win.std() < 1e-4:
        return {"x2": x2, "y2": y2, "score": 0.0, "moved": False}
    res = cv2.matchTemplate(win, tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    sx, sy = _subpixel(res, loc)
    # res(0,0) corresponds to the template centred at (x2 - search, y2 - search)
    nx = x2 - search + loc[0] + sx
    ny = y2 - search + loc[1] + sy
    return {"x2": float(nx), "y2": float(ny), "score": float(score), "moved": True, "warped": warped}
