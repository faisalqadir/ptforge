"""Source-image masks: crop (rectangle / circle) and painted include/exclude masks.

Paint masks are single channel PNGs: 0 = no preference, 1 = exclude (red),
2 = force include (green).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .project import Project

MASK_NONE, MASK_EXCLUDE, MASK_INCLUDE = 0, 1, 2


def load_paint_mask(project: Project, idx: int, w: int, h: int) -> np.ndarray | None:
    im = project.images[idx]
    if not im.mask:
        return None
    path = project.abspath(im.mask)
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:  # colour-coded masks: red exclude, green include
        b, g, r = m[..., 0].astype(int), m[..., 1].astype(int), m[..., 2].astype(int)
        out = np.zeros(m.shape[:2], np.uint8)
        out[(r > 128) & (g < 100)] = MASK_EXCLUDE
        out[(g > 128) & (r < 100)] = MASK_INCLUDE
        m = out
    if m.shape[1] != w or m.shape[0] != h:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def save_paint_mask(project: Project, idx: int, mask: np.ndarray, path: str | None = None):
    if path is None:
        base = os.path.splitext(project.abspath(project.images[idx].path))[0]
        path = base + ".pfmask.png"
    cv2.imwrite(path, mask.astype(np.uint8))
    project.images[idx].mask = path
    return path


def crop_mask(project: Project, idx: int, w: int, h: int) -> np.ndarray | None:
    im = project.images[idx]
    crop = project.lenses[im.lens].crop
    if crop.mode == "none":
        return None
    sx, sy = w / max(im.width, 1), h / max(im.height, 1)
    m = np.zeros((h, w), np.uint8)
    if crop.mode == "rect":
        r = crop.right or im.width
        b = crop.bottom or im.height
        m[int(round(crop.top * sy)):int(round(b * sy)), int(round(crop.left * sx)):int(round(r * sx))] = 255
    elif crop.mode == "circle":
        # crop centre is stored in edge coordinates (W/2 = geometric centre); cv2 wants pixel centres
        cx = (crop.cx or im.width / 2) * sx - 0.5
        cy = (crop.cy or im.height / 2) * sy - 0.5
        rad = (crop.radius or min(im.width, im.height) / 2) * min(sx, sy)
        cv2.circle(m, (int(round(cx * 16)), int(round(cy * 16))), int(round(rad * 16)), 255, -1, cv2.LINE_AA, shift=4)
    return m


def source_valid_mask(project: Project, idx: int, w: int, h: int, include_paint: bool = True) -> np.ndarray | None:
    """uint8 (h,w) 255 = usable source pixel, or None when the whole image is usable."""
    m = crop_mask(project, idx, w, h)
    if include_paint:
        pm = load_paint_mask(project, idx, w, h)
        if pm is not None and (pm == MASK_EXCLUDE).any():
            if m is None:
                m = np.full((h, w), 255, np.uint8)
            m = m.copy()
            m[pm == MASK_EXCLUDE] = 0
    return m
