"""Photometric model and optimisation (exposure, white balance, vignetting, response).

Model for a source pixel value ``z`` (0..1, encoded) of image i::

    scene = f_inv(z) * 2**ev_i / (V_i(r) * wb_i)           wb = (wb_r, 1, wb_b)

where ``f`` is the camera response of the lens, ``V`` the vignetting
polynomial ``1 + v1 r^2 + v2 r^4 + v3 r^6`` (r normalised by half the image
diagonal, centred at the image centre + vig offset).  Output pixels are
``f_out(scene * 2**(exposure - ev_ref))``.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import geometry as G
from .imageio import linear_to_srgb, load_image, srgb_to_linear
from .project import Project

# --------------------------------------------------------------------------- response curves


def _f(x):
    x = np.asarray(x)
    return x if x.dtype in (np.float32, np.float64) else x.astype(np.float32)


class Response:
    def __init__(self, kind: str = "srgb", params=None):
        self.kind = kind
        self.params = list(params or [])
        if kind == "poly":
            self._build_poly_lut()

    def _build_poly_lut(self):
        # monotone correction of the sRGB curve: f(x) = srgb(x) + c1 x(1-x) + c2 x(1-x)(x-0.5)
        c = (self.params + [0.0, 0.0])[:2]
        xs = np.linspace(0, 1, 4097, dtype=np.float64)
        s = linear_to_srgb(xs.astype(np.float32)).astype(np.float64)
        y = s + c[0] * xs * (1 - xs) + c[1] * xs * (1 - xs) * (xs - 0.5)
        y = np.maximum.accumulate(np.clip(y, 0, 1))
        self._fx, self._fy = xs, y

    def to_linear(self, z):
        if self.kind == "srgb":
            return srgb_to_linear(z)
        if self.kind == "linear":
            return _f(z)
        if self.kind == "gamma":
            g = self.params[0] if self.params else 2.2
            return np.power(np.maximum(_f(z), 0), g)
        if self.kind == "poly":
            z = _f(z)
            return np.interp(z, self._fy, self._fx).astype(z.dtype)
        raise ValueError(self.kind)

    def from_linear(self, x):
        if self.kind == "srgb":
            return linear_to_srgb(x)
        if self.kind == "linear":
            return _f(x)
        if self.kind == "gamma":
            g = self.params[0] if self.params else 2.2
            return np.power(np.maximum(_f(x), 0), 1.0 / g)
        if self.kind == "poly":
            x = _f(x)
            return np.interp(np.clip(x, 0, 1), self._fx, self._fy).astype(x.dtype)
        raise ValueError(self.kind)


def lens_response(p: Project, lens_idx: int) -> Response:
    ln = p.lenses[lens_idx]
    return Response(ln.response, ln.response_params if ln.response in ("gamma", "poly") else [])


# --------------------------------------------------------------------------- per pixel correction


def vignetting_factor(p: Project, idx: int, x, y):
    im = p.images[idx]
    ln = p.lenses[im.lens]
    v = ln.vig
    if not any(v):
        return None
    cx = im.width / 2.0 + ln.vig_cx
    cy = im.height / 2.0 + ln.vig_cy
    r2 = ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) / vig_norm2(p, idx)
    return (1.0 + r2 * (v[0] + r2 * (v[1] + r2 * v[2]))).astype(np.float32)


def vig_norm2(p: Project, idx: int) -> float:
    """Squared radius that maps to r=1 in the vignetting polynomial."""
    im = p.images[idx]
    crop = p.lenses[im.lens].crop
    if crop.mode == "circle":
        r = crop.radius or min(im.width, im.height) / 2
        return float(r * r)
    return (im.width ** 2 + im.height ** 2) / 4.0


def reference_ev(p: Project) -> float:
    evs = [p.images[i].ev for i in p.included()]
    return float(np.median(evs)) if evs else 0.0


def correct_to_scene(p: Project, idx: int, rgb: np.ndarray, map_x=None, map_y=None, ev_ref: float | None = None):
    """Encoded source values -> linear scene radiance (relative to the reference exposure)."""
    im = p.images[idx]
    resp = lens_response(p, im.lens)
    lin = resp.to_linear(np.clip(rgb, 0, None)) if not _is_linear_hdr(rgb) else rgb
    ev_ref = reference_ev(p) if ev_ref is None else ev_ref
    gain = 2.0 ** (im.ev - ev_ref)
    g = np.array([gain / im.wb_r, gain, gain / im.wb_b], dtype=np.float32)
    out = lin * g
    if map_x is not None:
        V = vignetting_factor(p, idx, map_x, map_y)
        if V is not None:
            out = out / np.maximum(V, 0.05)[..., None]
    if im.flare:
        out = np.maximum(out - im.flare * gain, 0)
    return out.astype(np.float32)


def _is_linear_hdr(a):
    return False


# --------------------------------------------------------------------------- optimisation


def _sample_pairs(p: Project, images, n_dirs: int = 30000, work: int = 600, rng=None):
    """Sample corresponding colours from overlapping images.

    Returns list of (i, j, zi (n,3), zj (n,3), ri2 (n,), rj2 (n,)) with r^2 normalised.
    """
    rng = rng or np.random.default_rng(1)
    # random directions on the sphere, restricted later to overlaps
    v = rng.normal(size=(n_dirs, 3))
    d = G.normalize(v)
    cams = {i: p.camera(i) for i in images}
    small = {}
    proj = {}
    for i in images:
        im = p.images[i]
        rgb = load_image(p.abspath(im.path)).rgb
        s = min(1.0, work / max(rgb.shape[:2]))
        sm = cv2.resize(rgb, (max(1, int(round(rgb.shape[1] * s))), max(1, int(round(rgb.shape[0] * s)))), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 1.0)
        small[i] = (sm, s)
        x, y, valid = cams[i].world_to_pixel(d)
        m = 0.03 * max(im.width, im.height)
        valid &= (x > m) & (x < im.width - 1 - m) & (y > m) & (y < im.height - 1 - m)
        if p.lenses[im.lens].crop.mode == "circle":
            cr = p.lenses[im.lens].crop
            cx, cy, rad = cr.cx or im.width / 2, cr.cy or im.height / 2, cr.radius or min(im.width, im.height) / 2
            valid &= np.hypot(x - cx, y - cy) < rad * 0.95
        proj[i] = (x, y, valid)
    out = []
    for a in range(len(images)):
        for b in range(a + 1, len(images)):
            i, j = images[a], images[b]
            both = proj[i][2] & proj[j][2]
            if both.sum() < 30:
                continue
            k = np.nonzero(both)[0][:2000]

            def sample(idx):
                sm, s = small[idx]
                xs = ((proj[idx][0][k] + 0.5) * s - 0.5).astype(np.float32)
                ys = ((proj[idx][1][k] + 0.5) * s - 0.5).astype(np.float32)
                vals = cv2.remap(sm, xs.reshape(-1, 1), ys.reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1, 3)
                im = p.images[idx]
                ln = p.lenses[im.lens]
                r2 = ((proj[idx][0][k] + 0.5 - im.width / 2 - ln.vig_cx) ** 2 + (proj[idx][1][k] + 0.5 - im.height / 2 - ln.vig_cy) ** 2) / vig_norm2(p, idx)
                return vals, r2

            zi, ri = sample(i)
            zj, rj = sample(j)
            zi, zj = zi.astype(np.float64), zj.astype(np.float64)
            ok = (zi.max(1) < 0.97) & (zj.max(1) < 0.97) & (zi.min(1) > 0.015) & (zj.min(1) > 0.015)
            # reject samples where local content differs a lot (misalignment/parallax/moving objects)
            ok &= np.abs(zi.mean(1) - zj.mean(1)) < 0.35
            if ok.sum() >= 20:
                out.append((i, j, zi[ok], zj[ok], ri[ok], rj[ok]))
    return out


def optimize_photometric(p: Project, images=None, verbose: int = 0) -> dict:
    """Optimise exposure (EV), white balance, vignetting and optionally response."""
    images = sorted(images if images is not None else p.included())
    s = p.exposure
    samples = _sample_pairs(p, images)
    if not samples:
        return {"success": False, "message": "no overlapping samples"}
    anchor = p.optimizer.anchor if p.optimizer.anchor in images else images[0]
    # decide what to optimise
    exif_known = all(p.images[i].exif_ev is not None for i in images)
    any_auto_exp = any(p.images[i].exif.get("exposure_auto", True) for i in images)
    # 'auto' with known manual exposure still fine-tunes brightness around the EXIF values:
    # in-camera tone processing (e.g. Active D-Lighting) changes brightness between shots
    opt_ev = s.optimize_exposure in ("full", "finetune", "auto")
    ev_prior = 0.5 if (exif_known and not any_auto_exp and s.optimize_exposure in ("auto", "finetune")) else 0.0
    any_auto_wb = any(p.images[i].exif.get("wb_auto", True) for i in images)
    opt_wb = s.optimize_wb in ("full", "on", "finetune") or (s.optimize_wb == "auto" and any_auto_wb)
    ev_exif = {}
    if exif_known and s.optimize_exposure in ("auto", "finetune"):
        evs = [p.images[i].exif_ev for i in images]
        ref = float(np.median(evs))
        for i in images:
            p.images[i].ev = float(p.images[i].exif_ev - ref)
            ev_exif[i] = p.images[i].ev
    lenses = sorted({p.images[i].lens for i in images})
    groups = p.link_groups() if s.link_brackets else {}
    # parameter layout
    names = []
    x0 = []
    if opt_ev:
        for i in images:
            if i != anchor:
                names.append(("ev", i))
                x0.append(p.images[i].ev)
    if opt_wb:
        for i in images:
            if i != anchor:
                names += [("wb_r", i), ("wb_b", i)]
                x0 += [math.log(p.images[i].wb_r), math.log(p.images[i].wb_b)]
    if s.optimize_vignetting:
        for l in lenses:
            names += [("v1", l), ("v2", l), ("v3", l)]
            x0 += list(p.lenses[l].vig[:3])
    if s.optimize_response:
        for l in lenses:
            names += [("c1", l), ("c2", l)]
            pr = p.lenses[l].response_params if p.lenses[l].response == "poly" else []
            x0 += list((pr + [0.0, 0.0])[:2])
    if s.optimize_flare:
        for i in images:
            names.append(("flare", i))
            x0.append(p.images[i].flare)
    x0 = np.array(x0, dtype=np.float64)
    if not len(x0):
        return {"success": True, "message": "nothing to optimise"}

    def unpack(x):
        st = {i: dict(ev=p.images[i].ev, wb_r=math.log(p.images[i].wb_r), wb_b=math.log(p.images[i].wb_b), flare=p.images[i].flare) for i in images}
        ls = {l: dict(v=list(p.lenses[l].vig[:3]), c=list(((p.lenses[l].response_params if p.lenses[l].response == "poly" else []) + [0.0, 0.0])[:2])) for l in lenses}
        for (nm, o), val in zip(names, x):
            if nm in ("ev", "wb_r", "wb_b", "flare"):
                st[o][nm] = val
            elif nm[0] == "v":
                ls[o]["v"][int(nm[1]) - 1] = val
            else:
                ls[o]["c"][int(nm[1]) - 1] = val
        return st, ls

    # bracket members are forced to keep their exif EV differences only if exposure is not optimised
    lut_cache = {}

    def resp_for(l, ls):
        if not s.optimize_response:
            return lens_response(p, l)
        key = (l, tuple(np.round(ls[l]["c"], 6)))
        if key not in lut_cache:
            lut_cache.clear()
            lut_cache[key] = Response("poly", ls[l]["c"])
        return lut_cache[key]

    def residuals(x):
        st, ls = unpack(x)
        out = []
        for i, j, zi, zj, ri, rj in samples:
            li, lj = p.images[i].lens, p.images[j].lens
            fi, fj = resp_for(li, ls), resp_for(lj, ls)
            Vi = 1 + ri * (ls[li]["v"][0] + ri * (ls[li]["v"][1] + ri * ls[li]["v"][2]))
            Vj = 1 + rj * (ls[lj]["v"][0] + rj * (ls[lj]["v"][1] + rj * ls[lj]["v"][2]))
            gi = np.array([math.exp(-st[i]["wb_r"]), 1.0, math.exp(-st[i]["wb_b"])]) * 2.0 ** st[i]["ev"]
            gj = np.array([math.exp(-st[j]["wb_r"]), 1.0, math.exp(-st[j]["wb_b"])]) * 2.0 ** st[j]["ev"]
            si = np.maximum(fi.to_linear(zi) - st[i]["flare"], 0) * gi / np.maximum(Vi, 0.05)[:, None]
            sj = np.maximum(fj.to_linear(zj) - st[j]["flare"], 0) * gj / np.maximum(Vj, 0.05)[:, None]
            # compare in the encoded domain (perceptually uniform-ish errors), symmetric
            pj = fj.from_linear(np.clip(si / gj * np.maximum(Vj, 0.05)[:, None], 0, 1))
            pi = fi.from_linear(np.clip(sj / gi * np.maximum(Vi, 0.05)[:, None], 0, 1))
            out.append((pj - zj).ravel())
            out.append((pi - zi).ravel())
        r = np.concatenate(out)
        # weak priors keep vignetting/response sane
        prior = []
        for (nm, o), val in zip(names, x):
            if nm[0] == "v":
                prior.append(val * 0.02)
            elif nm[0] == "c":
                prior.append(val * 0.05)
            elif nm == "ev" and ev_prior and o in ev_exif:
                prior.append((val - ev_exif[o]) * ev_prior)
        return np.concatenate([r, np.array(prior)]) if prior else r

    r0 = residuals(x0)
    res = least_squares(residuals, x0, method="trf", loss="soft_l1", f_scale=0.03, max_nfev=80, verbose=verbose, x_scale="jac", diff_step=1e-4)
    st, ls = unpack(res.x)
    for i in images:
        im = p.images[i]
        im.ev = float(st[i]["ev"])
        im.wb_r = float(math.exp(st[i]["wb_r"]))
        im.wb_b = float(math.exp(st[i]["wb_b"]))
        im.flare = float(st[i]["flare"])
    # linked bracket members share white balance with their stack leader
    for members in groups.values():
        if len(members) > 1 and not opt_ev:
            pass
    for l in lenses:
        if s.optimize_vignetting:
            p.lenses[l].vig = [float(v) for v in ls[l]["v"]]
        if s.optimize_response:
            p.lenses[l].response = "poly"
            p.lenses[l].response_params = [float(c) for c in ls[l]["c"]]
    r1 = res.fun
    return {"success": bool(res.success), "rms_before": float(np.sqrt(np.mean(r0 ** 2))), "rms_after": float(np.sqrt(np.mean(r1 ** 2))),
            "n_samples": int(sum(len(sm[2]) for sm in samples)), "optimized_ev": opt_ev, "optimized_wb": opt_wb}
