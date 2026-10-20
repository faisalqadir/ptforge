"""Seam finding on a reduced-resolution canvas.

1. Initial labelling: every canvas pixel goes to the covering image whose edge
   is farthest away (plus blend priority), i.e. a weighted Voronoi diagram.
2. Optional graph-cut refinement per neighbouring image pair inside a band
   around the initial seam: the seam is moved to where the images agree.
"""
from __future__ import annotations

import cv2
import numpy as np

NONE = -1


def _dist_wrap(mask: np.ndarray, wrap: bool) -> np.ndarray:
    m = mask.astype(np.uint8)
    if wrap:
        pad = m.shape[1] // 4
        mp = np.concatenate([m[:, -pad:], m, m[:, :pad]], axis=1)
        mp = np.pad(mp, ((1, 1), (0, 0)), constant_values=0)
        d = cv2.distanceTransform(mp, cv2.DIST_L2, 5)[1:-1, pad:-pad]
        return d
    mp = np.pad(m, 1, constant_values=0)
    return cv2.distanceTransform(mp, cv2.DIST_L2, 5)[1:-1, 1:-1]


def initial_labels(masks: list, wrap: bool, priorities=None, includes=None, centre_weight=None) -> np.ndarray:
    """masks: list of full-canvas bool arrays (seam resolution)."""
    H, W = masks[0].shape
    best = np.full((H, W), -1.0, np.float32)
    lab = np.full((H, W), NONE, np.int32)
    for k, m in enumerate(masks):
        if not m.any():
            continue
        d = _dist_wrap(m, wrap).astype(np.float32)
        if centre_weight is not None and centre_weight[k] is not None:
            d = d * centre_weight[k]
        if priorities is not None and priorities[k]:
            d = d + 1e6 * priorities[k] * m
        if includes is not None and includes[k] is not None:
            d = d + 1e9 * includes[k]
        d[~m] = -1
        upd = d > best
        best[upd] = d[upd]
        lab[upd] = k
    return lab


def _pair_cut(lab, a, b, img_a, img_b, ma, mb, wrap, band, fixed=None):
    """Re-label pixels currently labelled a or b by a min-cut between them."""
    import maxflow

    region = (lab == a) | (lab == b)
    both = region & ma & mb
    if both.sum() < 16:
        return False
    # restrict to a band around the current a/b boundary
    bnd = np.zeros_like(region)
    la, lb = lab == a, lab == b
    k = np.ones((3, 3), np.uint8)
    da = cv2.dilate(la.astype(np.uint8), k) > 0
    db = cv2.dilate(lb.astype(np.uint8), k) > 0
    bnd = da & db
    if wrap:
        # boundary across the wrap seam
        bnd[:, 0] |= la[:, 0] & lb[:, -1] | lb[:, 0] & la[:, -1]
        bnd[:, -1] |= la[:, -1] & lb[:, 0] | lb[:, -1] & la[:, 0]
    if not bnd.any():
        return False
    dist = _dist_wrap(~bnd, wrap)
    free = both & (dist <= band)
    if fixed is not None:
        free &= ~fixed
    if free.sum() < 4:
        return False
    ys, xs = np.nonzero(free)
    y0, y1 = max(0, ys.min() - 1), min(lab.shape[0], ys.max() + 2)
    full_width = wrap and (xs.min() == 0 or xs.max() == lab.shape[1] - 1)
    if full_width:
        x0, x1 = 0, lab.shape[1]
    else:
        x0, x1 = max(0, xs.min() - 1), min(lab.shape[1], xs.max() + 2)
    sl = (slice(y0, y1), slice(x0, x1))
    fr = free[sl]
    reg = region[sl]
    la_s, lb_s = la[sl], lb[sl]
    diff = np.abs(img_a[sl] - img_b[sl]).sum(axis=-1)
    diff = np.where(ma[sl] & mb[sl], diff, 3.0)
    diff = diff + 0.002
    h, w = fr.shape
    g = maxflow.Graph[float]()
    ids = g.add_grid_nodes((h, w))
    # horizontal edges: cost of separating (y,x) and (y,x+1)
    wh = np.zeros((h, w), np.float64)
    wh[:, :-1] = diff[:, :-1] + diff[:, 1:]
    wv = np.zeros((h, w), np.float64)
    wv[:-1, :] = diff[:-1, :] + diff[1:, :]
    st_r = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    st_d = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    g.add_grid_edges(ids, weights=wh, structure=st_r, symmetric=True)
    g.add_grid_edges(ids, weights=wv, structure=st_d, symmetric=True)
    if full_width:
        for y in range(h):
            c = float(diff[y, 0] + diff[y, -1])
            g.add_edge(ids[y, 0], ids[y, -1], c, c)
    INF = 1e9
    src = np.where(reg & ~fr & la_s, INF, 0.0)  # fixed to a (source)
    snk = np.where(reg & ~fr & lb_s, INF, 0.0)  # fixed to b (sink)
    outside = ~reg
    # pixels outside the a/b region are irrelevant: tie them to their nearest side cheaply
    src = np.where(outside, 0.0, src)
    snk = np.where(outside, 0.0, snk)
    g.add_grid_tedges(ids, src, snk)
    g.maxflow()
    seg = g.get_grid_segments(ids)  # True -> sink side (b)
    new = lab[sl].copy()
    upd = fr
    new[upd & ~seg] = a
    new[upd & seg] = b
    lab[sl] = new
    return True


def refine_graphcut(lab: np.ndarray, imgs: list, masks: list, wrap: bool, band: int = 40, includes=None, iterations: int = 1):
    """Pairwise graph-cut refinement of a label map (in place)."""
    fixed = None
    if includes is not None:
        fixed = np.zeros(lab.shape, bool)
        for inc in includes:
            if inc is not None:
                fixed |= inc
    for _ in range(iterations):
        pairs = _neighbour_pairs(lab, wrap)
        for a, b in pairs:
            _pair_cut(lab, a, b, imgs[a], imgs[b], masks[a], masks[b], wrap, band, fixed)
    return lab


def _neighbour_pairs(lab, wrap):
    pairs = set()
    a, b = lab[:, :-1], lab[:, 1:]
    m = (a != b) & (a >= 0) & (b >= 0)
    pairs |= set(zip(a[m].tolist(), b[m].tolist()))
    a, b = lab[:-1, :], lab[1:, :]
    m = (a != b) & (a >= 0) & (b >= 0)
    pairs |= set(zip(a[m].tolist(), b[m].tolist()))
    if wrap:
        a, b = lab[:, -1], lab[:, 0]
        m = (a != b) & (a >= 0) & (b >= 0)
        pairs |= set(zip(a[m].tolist(), b[m].tolist()))
    return sorted({(min(x, y), max(x, y)) for x, y in pairs})


def seam_image(lab: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Boolean image of seam lines (for display)."""
    s = np.zeros(lab.shape, bool)
    s[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    s[:-1, :] |= lab[:-1, :] != lab[1:, :]
    if wrap:
        s[:, -1] |= lab[:, -1] != lab[:, 0]
    return s & (lab >= 0)
