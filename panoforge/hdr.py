"""HDR: bracket merging (true HDR radiance), exposure fusion and tone mapping."""
from __future__ import annotations

import cv2
import numpy as np

from .blend import _down, _up, push_pull_fill
from .imageio import linear_to_srgb, srgb_to_linear


def hat_weight(z):
    """Weight for an encoded value: trust mid-tones, distrust clipped shadows/highlights."""
    z = np.clip(z, 0, 1)
    return np.clip(1.0 - np.power(2.0 * z - 1.0, 12), 1e-4, 1.0).astype(np.float32)


def merge_true_hdr(scenes: list, gains: list, alphas: list, deghost: bool = True):
    """Merge linear scene estimates of one bracket stack into radiance.

    scenes[k]: (h,w,3) linear scene radiance estimated from exposure k
    gains[k]:  multiplier that took the exposure's linear values to scene units
    """
    num = np.zeros_like(scenes[0])
    den = np.zeros(scenes[0].shape[:2], np.float32)
    # encoded value of each exposure (as the camera saw it) drives the weight
    zs = [linear_to_srgb(np.clip(s / g, 0, 1)) for s, g in zip(scenes, gains)]
    ws = []
    for z, a in zip(zs, alphas):
        w = hat_weight(z).min(axis=-1) * (a > 0)
        ws.append(w.astype(np.float32))
    if deghost and len(scenes) > 2:
        # reference = middle exposure; down-weight pixels inconsistent with it (moving objects)
        ref = len(scenes) // 2
        lref = np.log(np.maximum(scenes[ref].mean(-1), 1e-6))
        for k in range(len(scenes)):
            if k == ref:
                continue
            lk = np.log(np.maximum(scenes[k].mean(-1), 1e-6))
            usable = ws[k] > 0.2
            dev = np.abs(lk - lref)
            ws[k] = np.where(usable & (ws[ref] > 0.2) & (dev > 0.7), ws[k] * 0.02, ws[k])
    for s, w in zip(scenes, ws):
        num += s * w[..., None]
        den += w
    alpha = np.maximum.reduce([a for a in alphas])
    out = num / np.maximum(den, 1e-8)[..., None]
    # pixels where every exposure was clipped: take the darkest (highlights) / brightest (shadows)
    bad = den < 1e-3
    if bad.any():
        bright = np.mean([z.mean(-1) for z in zs], axis=0)
        pick = np.where(bright > 0.5, 0, len(scenes) - 1)  # scenes sorted dark->bright by caller
        for k in (0, len(scenes) - 1):
            m = bad & (pick == k) & (alphas[k] > 0)
            out[m] = scenes[k][m]
    return out.astype(np.float32), alpha


def fusion_weights(z: np.ndarray, alpha: np.ndarray, wc=1.0, ws=1.0, we=1.0, sigma=0.2):
    gray = cv2.cvtColor(np.ascontiguousarray(z), cv2.COLOR_RGB2GRAY)
    C = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    S = z.std(axis=-1)
    E = np.exp(-((z - 0.5) ** 2) / (2 * sigma * sigma)).prod(axis=-1)
    w = (C + 1e-3) ** wc * (S + 1e-3) ** ws * (E + 1e-6) ** we
    return (w * (alpha > 0)).astype(np.float32)


def exposure_fusion(encoded: list, alphas: list, weights=(1.0, 1.0, 1.0), sigma: float = 0.2, levels: int | None = None):
    """Mertens exposure fusion of encoded (0..1) exposures (h,w,3)."""
    h, w = encoded[0].shape[:2]
    W = [fusion_weights(z, a, *weights, sigma=sigma) for z, a in zip(encoded, alphas)]
    tot = np.sum(W, axis=0)
    W = [x / np.maximum(tot, 1e-12) for x in W]
    if levels is None:
        levels = max(1, int(np.log2(max(2, min(h, w)))) - 3)
    out_pyr = None
    for z, wk, a in zip(encoded, W, alphas):
        z = push_pull_fill(z, a > 0)
        gp = [wk]
        lp = []
        cur = z
        for _ in range(levels):
            nxt = _down(cur)
            lp.append(cur - _up(nxt, cur.shape))
            cur = nxt
            gp.append(_down(gp[-1]))
        lp.append(cur)
        contrib = [l * g[..., None] for l, g in zip(lp, gp)]
        if out_pyr is None:
            out_pyr = contrib
        else:
            for k in range(len(contrib)):
                out_pyr[k] += contrib[k]
    out = out_pyr[-1]
    for k in range(len(out_pyr) - 2, -1, -1):
        out = _up(out, out_pyr[k].shape) + out_pyr[k]
    alpha = np.maximum.reduce(list(alphas))
    return np.clip(out, 0, 1).astype(np.float32), alpha


# --------------------------------------------------------------------------- tone mapping

DEFAULT_TONEMAP = dict(brightness=0.0, boost_shadows=0.3, reduce_lights=0.3, compression=0.5, contrast=0.0, saturation=0.0)


def tonemap(hdr: np.ndarray, params: dict | None = None, alpha: np.ndarray | None = None) -> np.ndarray:
    """Local tone mapping of linear radiance to encoded 0..1 (base/detail decomposition)."""
    P = dict(DEFAULT_TONEMAP)
    P.update(params or {})
    lum = np.maximum(0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2], 1e-7)
    L = np.log2(lum)
    valid = (alpha > 0) if alpha is not None else np.ones(L.shape, bool)
    lo, mid, hi = np.percentile(L[valid], [1, 50, 99.5]) if valid.any() else (-8, 0, 2)
    # edge-preserving base layer via guided filter on log luminance
    r = max(4, int(min(L.shape) * 0.02))
    base = _guided(L, L, r, 0.05)
    detail = L - base
    comp = float(np.clip(P["compression"], 0, 1))
    target_range = 6.5 - 3.0 * comp  # stops the base layer is squeezed into
    scale = min(1.0, target_range / max(hi - lo, 1e-3))
    nb = (base - mid) * scale
    # shadows / highlights shaping on the compressed base
    nb = nb + P["boost_shadows"] * 1.5 * np.clip(-nb / 3.0, 0, 1) ** 1.5 * 2.0
    nb = nb - P["reduce_lights"] * 1.5 * np.clip(nb / 2.5, 0, 1) ** 1.5 * 2.0
    det_gain = 1.0 + P["contrast"]
    newL = nb + detail * det_gain - 1.6 + P["brightness"]
    ratio = np.power(2.0, newL - L)[..., None]
    sat = 1.0 + P["saturation"]
    col = hdr / lum[..., None]
    col = 1.0 + (col - 1.0) * sat
    out = np.maximum(col * lum[..., None] * ratio, 0)
    # soft highlight shoulder
    out = out / (1.0 + 0.15 * out)
    out = linear_to_srgb(np.clip(out / (1 / 1.15), 0, 1))
    return out.astype(np.float32)


def _guided(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    cIp = cv2.boxFilter(I * p, -1, k) - mI * mp
    vI = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cIp / (vI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def reinhard_global(hdr: np.ndarray, key: float = 0.18) -> np.ndarray:
    lum = np.maximum(0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2], 1e-7)
    lw = np.exp(np.mean(np.log(lum)))
    Ls = key / lw * lum
    Ld = Ls / (1 + Ls)
    return linear_to_srgb(np.clip(hdr * (Ld / lum)[..., None], 0, 1))
