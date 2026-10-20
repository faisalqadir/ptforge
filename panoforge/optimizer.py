"""Geometric optimiser (bundle adjustment of image orientations and lens parameters).

Residuals are measured on the unit sphere and converted to approximate source
pixels with a fixed per-image scale, so they are independent of the output
projection and well behaved for every lens type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import geometry as G
from .project import CP_HORIZONTAL, CP_LINE, CP_NORMAL, CP_VERTICAL, Project

UP = np.array([0.0, 1.0, 0.0])

DISTORTION_SETS = {
    "none": (),
    "medium": ("b",),
    "heavy": ("a", "b", "c"),
    "heavy_shift": ("a", "b", "c", "d", "e"),
}

BOUNDS = {
    "yaw": (-1e4, 1e4), "pitch": (-1e4, 1e4), "roll": (-1e4, 1e4),
    "hfov": (0.5, 360.0), "fisheye_factor": (-1.0, 1.0),
    "a": (-0.3, 0.3), "b": (-0.3, 0.3), "c": (-0.3, 0.3),
    "d": (-1e4, 1e4), "e": (-1e4, 1e4), "g": (-0.5, 0.5), "t": (-0.5, 0.5),
    "vp_x": (-10, 10), "vp_y": (-10, 10), "vp_z": (-10, 10),
}


@dataclass
class Var:
    kind: str  # pose | lens | image | vp
    owner: int  # pose-group id / lens id / image id
    name: str


@dataclass
class OptimizeResult:
    success: bool
    cost_before: float
    cost_after: float
    mean_error: float
    max_error: float
    rms_error: float
    n_cps: int
    n_vars: int
    message: str = ""
    per_cp: list = field(default_factory=list)


class _Problem:
    def __init__(self, p: Project, images, variables, cps, pose_group_of):
        self.p = p
        self.images = images
        self.vars = variables
        self.cps = cps
        self.pose_group_of = pose_group_of
        self.base = {i: p.camera(i) for i in images}
        self.scale = {i: self.base[i].focal_px for i in images}
        # gather per-image endpoints
        self.pts: dict[int, list] = {i: [] for i in images}
        self.slots = []  # for each cp: (slot_i, slot_j)
        for cp in cps:
            si = len(self.pts[cp.i])
            self.pts[cp.i].append((cp.x1, cp.y1))
            sj = len(self.pts[cp.j])
            self.pts[cp.j].append((cp.x2, cp.y2))
            self.slots.append((si, sj))
        self.pts = {i: np.array(v, dtype=np.float64).reshape(-1, 2) for i, v in self.pts.items()}
        self.kind_idx = {k: [n for n, cp in enumerate(cps) if cp.kind == k] for k in (CP_NORMAL, CP_VERTICAL, CP_HORIZONTAL, CP_LINE)}
        lines: dict[int, list] = {}
        for n in self.kind_idx[CP_LINE]:
            lines.setdefault(cps[n].line, []).append(n)
        self.lines = lines
        # vectorised gather tables for normal control points
        self.off, o = {}, 0
        for i in images:
            self.off[i] = o
            o += len(self.pts[i])
        nid = self.kind_idx[CP_NORMAL]
        self.nI = np.array([cps[n].i for n in nid], dtype=np.int64)
        self.nJ = np.array([cps[n].j for n in nid], dtype=np.int64)
        self.nA = np.array([self.off[cps[n].i] + self.slots[n][0] for n in nid], dtype=np.int64)
        self.nB = np.array([self.off[cps[n].j] + self.slots[n][1] for n in nid], dtype=np.int64)
        self.nP1 = np.array([(cps[n].x1, cps[n].y1) for n in nid], dtype=np.float64).reshape(-1, 2)
        self.nP2 = np.array([(cps[n].x2, cps[n].y2) for n in nid], dtype=np.float64).reshape(-1, 2)
        self.rows_J = {i: np.nonzero(self.nJ == i)[0] for i in images}
        self.rows_I = {i: np.nonzero(self.nI == i)[0] for i in images}
        # distortion guard: per lens, one representative image and the normalised radius in use
        self.guard = []
        if any(v.name in ("a", "b", "c") for v in variables):
            for lid in sorted({p.images[i].lens for i in images}):
                rep = next(i for i in images if p.images[i].lens == lid)
                im, ln = p.images[rep], p.lenses[lid]
                if ln.crop.mode == "circle":
                    rmax = (ln.crop.radius or min(im.width, im.height) / 2) / (min(im.width, im.height) / 2)
                else:
                    rmax = math.hypot(im.width, im.height) / min(im.width, im.height)
                self.guard.append((rep, np.linspace(0.0, rmax * 1.05, 24)))

    def x0(self):
        out = []
        for v in self.vars:
            out.append(self._get(v))
        return np.array(out, dtype=np.float64)

    def _get(self, v: Var):
        p = self.p
        if v.kind == "pose":
            i = self.pose_rep[v.owner]
            return getattr(p.images[i], v.name)
        if v.kind == "lens":
            return getattr(p.lenses[v.owner], v.name)
        if v.kind == "image":
            return p.lens_value(v.owner, v.name)
        return getattr(p.images[v.owner], v.name)

    def cameras(self, x):
        cams = {}
        lens_vals: dict = {}
        img_vals: dict = {}
        pose_vals: dict = {}
        vp_vals: dict = {}
        for v, val in zip(self.vars, x):
            if v.kind == "pose":
                pose_vals.setdefault(v.owner, {})[v.name] = val
            elif v.kind == "lens":
                lens_vals.setdefault(v.owner, {})[v.name] = val
            elif v.kind == "image":
                img_vals.setdefault(v.owner, {})[v.name] = val
            else:
                vp_vals.setdefault(v.owner, {})[v.name] = val
        for i in self.images:
            c = self.base[i]
            cam = G.Camera(width=c.width, height=c.height, lens_type=c.lens_type, hfov=c.hfov, fisheye_factor=c.fisheye_factor,
                           a=c.a, b=c.b, c=c.c, d=c.d, e=c.e, g=c.g, t=c.t, yaw=c.yaw, pitch=c.pitch, roll=c.roll,
                           vp_x=c.vp_x, vp_y=c.vp_y, vp_z=c.vp_z, vp_pan=c.vp_pan, vp_tilt=c.vp_tilt)
            lid = self.p.images[i].lens
            for k, val in lens_vals.get(lid, {}).items():
                if k not in self.p.images[i].lens_overrides:
                    setattr(cam, k, val)
            for k, val in img_vals.get(i, {}).items():
                setattr(cam, k, val)
            for k, val in pose_vals.get(self.pose_group_of[i], {}).items():
                setattr(cam, k, val)
            for k, val in vp_vals.get(i, {}).items():
                setattr(cam, k, val)
            cam.update()
            cams[i] = cam
        return cams

    def rays(self, cams):
        return {i: (cams[i].pixel_to_world(self.pts[i][:, 0], self.pts[i][:, 1]) if len(self.pts[i]) else np.zeros((0, 3)))
                for i in self.images}

    def residuals(self, x):
        cams = self.cameras(x)
        R = self.rays(cams)
        # pixel-equivalent scale from the *current* focal lengths: angle * f ~ source pixels.
        # (a fixed scale would let the optimiser shrink the field of view to cut the cost)
        self.scale = {i: cams[i].focal_px for i in self.images}
        out = []
        cps = self.cps
        if len(self.nI):
            # symmetric reprojection error in source pixels
            Rall = np.concatenate([R[i] for i in self.images if len(self.pts[i])], axis=0)
            a, b = Rall[self.nA], Rall[self.nB]
            e1 = np.zeros((len(a), 2))
            e2 = np.zeros((len(a), 2))
            for k in self.images:
                rj = self.rows_J[k]
                if len(rj):
                    x, y, ok = cams[k].world_to_pixel(a[rj])
                    d = np.stack([x, y], -1) - self.nP2[rj]
                    fb = (a[rj] - b[rj])[:, :2] * self.scale[k] * 4.0  # behind camera / invalid
                    e1[rj] = np.where(ok[:, None] & np.isfinite(d), d, fb)
                ri = self.rows_I[k]
                if len(ri):
                    x, y, ok = cams[k].world_to_pixel(b[ri])
                    d = np.stack([x, y], -1) - self.nP1[ri]
                    fb = (b[ri] - a[ri])[:, :2] * self.scale[k] * 4.0
                    e2[ri] = np.where(ok[:, None] & np.isfinite(d), d, fb)
            out.append(np.concatenate([e1, e2], axis=1).ravel() * 0.7071067811865476)
        idx = self.kind_idx[CP_VERTICAL]
        if idx:
            a = np.array([R[cps[n].i][self.slots[n][0]] for n in idx])
            b = np.array([R[cps[n].j][self.slots[n][1]] for n in idx])
            nrm = G.normalize(np.cross(a, b))
            s = np.array([self.scale[cps[n].i] for n in idx])
            out.append(nrm @ UP * s)
        idx = self.kind_idx[CP_HORIZONTAL]
        if idx:
            a = np.array([R[cps[n].i][self.slots[n][0]] for n in idx])
            b = np.array([R[cps[n].j][self.slots[n][1]] for n in idx])
            s = np.array([self.scale[cps[n].i] for n in idx])
            out.append((np.arcsin(np.clip(a[:, 1], -1, 1)) - np.arcsin(np.clip(b[:, 1], -1, 1))) * s)
        for lid, members in self.lines.items():
            P = []
            for n in members:
                P.append(R[cps[n].i][self.slots[n][0]])
                P.append(R[cps[n].j][self.slots[n][1]])
            P = np.array(P)
            if len(P) < 3:
                continue
            _, _, vt = np.linalg.svd(P)
            nrm = vt[-1]
            s = self.scale[cps[members[0]].i]
            out.append(P @ nrm * s)
        # keep the radial distortion monotonic over the area actually imaged: otherwise the
        # polynomial can fold the image back on itself and "absorb" gross alignment errors
        for rep, rr in self.guard:
            c = cams[rep]
            d0 = 1.0 - (c.a + c.b + c.c)
            deriv = ((4 * c.a * rr + 3 * c.b) * rr + 2 * c.c) * rr + d0
            out.append(np.maximum(0.0, 0.3 - deriv) * 5000.0)
        return np.concatenate(out) if out else np.zeros(0)

    def sparsity(self):
        n_res = 4 * len(self.kind_idx[CP_NORMAL]) + len(self.kind_idx[CP_VERTICAL]) + len(self.kind_idx[CP_HORIZONTAL])
        n_res += sum(2 * len(m) for m in self.lines.values() if 2 * len(m) >= 3)
        n_guard = sum(len(rr) for _, rr in self.guard)
        S = lil_matrix((n_res + n_guard, len(self.vars)), dtype=np.int8)
        cols_for_image: dict[int, list] = {i: [] for i in self.images}
        for c, v in enumerate(self.vars):
            for i in self.images:
                im = self.p.images[i]
                if v.kind == "pose" and self.pose_group_of[i] == v.owner:
                    cols_for_image[i].append(c)
                elif v.kind == "lens" and im.lens == v.owner and v.name not in im.lens_overrides:
                    cols_for_image[i].append(c)
                elif v.kind in ("image", "vp") and v.owner == i:
                    cols_for_image[i].append(c)
        row = 0
        cps = self.cps
        for n in self.kind_idx[CP_NORMAL]:
            cols = set(cols_for_image[cps[n].i]) | set(cols_for_image[cps[n].j])
            for r in range(4):
                for c in cols:
                    S[row + r, c] = 1
            row += 4
        for kind in (CP_VERTICAL, CP_HORIZONTAL):
            for n in self.kind_idx[kind]:
                for c in set(cols_for_image[cps[n].i]) | set(cols_for_image[cps[n].j]):
                    S[row, c] = 1
                row += 1
        for lid, members in self.lines.items():
            if 2 * len(members) < 3:
                continue
            cols = set()
            for n in members:
                cols |= set(cols_for_image[cps[n].i]) | set(cols_for_image[cps[n].j])
            for r in range(2 * len(members)):
                for c in cols:
                    S[row + r, c] = 1
            row += 2 * len(members)
        for rep, rr in self.guard:
            cols = [c for c in cols_for_image[rep] if self.vars[c].name in ("a", "b", "c")]
            for r in range(len(rr)):
                for c in cols:
                    S[row + r, c] = 1
            row += len(rr)
        return S.tocsr()


def _upper_bound(p: Project, v: Var) -> float:
    if v.name == "hfov":
        lid = v.owner if v.kind == "lens" else p.images[v.owner].lens
        if p.lenses[lid].lens_type == G.LENS_RECTILINEAR:
            return 170.0
    return BOUNDS[v.name][1]


def _pose_groups(p: Project, images, link_brackets=True):
    """Images in the same link group share one orientation."""
    group_of, rep = {}, {}
    gid = 0
    seen = {}
    for i in images:
        lg = p.images[i].link_group
        if link_brackets and lg >= 0:
            if lg not in seen:
                seen[lg] = gid
                rep[gid] = i
                gid += 1
            group_of[i] = seen[lg]
        else:
            group_of[i] = gid
            rep[gid] = i
            gid += 1
    return group_of, rep


def cp_errors(p: Project, images=None) -> np.ndarray:
    """Per-control-point error (approx. pixels) for the current parameters (vectorised)."""
    images = images if images is not None else list(range(len(p.images)))
    iset = set(images)
    cps = p.control_points
    n = len(cps)
    errs = np.full(n, np.nan)
    if not n:
        return errs
    cams = {i: p.camera(i) for i in iset}
    # world rays of every endpoint, computed per image in one call
    end_a = np.zeros((n, 3))
    end_b = np.zeros((n, 3))
    ok = np.array([cp.i in iset and cp.j in iset for cp in cps])
    for img in iset:
        ka = [k for k in range(n) if ok[k] and cps[k].i == img]
        kb = [k for k in range(n) if ok[k] and cps[k].j == img]
        if ka:
            end_a[ka] = cams[img].pixel_to_world(np.array([cps[k].x1 for k in ka]), np.array([cps[k].y1 for k in ka]))
        if kb:
            end_b[kb] = cams[img].pixel_to_world(np.array([cps[k].x2 for k in kb]), np.array([cps[k].y2 for k in kb]))
    scale = np.array([0.5 * (cams[cp.i].focal_px + cams[cp.j].focal_px) if ok[k] else 0.0 for k, cp in enumerate(cps)])
    kinds = np.array([cp.kind for cp in cps])
    m = ok & (kinds == CP_NORMAL)
    chord = np.linalg.norm(end_a[m] - end_b[m], axis=1)
    errs[m] = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * scale[m]
    m = ok & (kinds == CP_VERTICAL)
    if m.any():
        nrm = G.normalize(np.cross(end_a[m], end_b[m]))
        errs[m] = np.abs(nrm @ UP) * scale[m]
    m = ok & (kinds == CP_HORIZONTAL)
    if m.any():
        errs[m] = np.abs(np.arcsin(np.clip(end_a[m, 1], -1, 1)) - np.arcsin(np.clip(end_b[m, 1], -1, 1))) * scale[m]
    # straight lines: distance from the fitted great circle
    lines: dict[int, list] = {}
    for k, cp in enumerate(cps):
        if ok[k] and cp.kind == CP_LINE:
            lines.setdefault(cp.line, []).append(k)
    for members in lines.values():
        P = np.concatenate([end_a[members], end_b[members]])
        e = np.zeros(len(members))
        if len(P) >= 3:
            nrm = np.linalg.svd(P)[2][-1]
            dist = np.abs(P @ nrm)
            f = np.array([cams[cps[k].i].focal_px for k in members])
            e = np.maximum(dist[:len(members)], dist[len(members):]) * f
        errs[members] = e
    for k, cp in enumerate(cps):
        if not np.isnan(errs[k]):
            cp.error = float(errs[k])
    return errs


def build_variables(p: Project, images, include_hv: bool, fixed_lens: bool = False):
    s = p.optimizer
    group_of, rep = _pose_groups(p, images, p.exposure.link_brackets)
    anchor = s.anchor if s.anchor in images else images[0]
    anchor_group = group_of[anchor]
    variables = []
    for gid in sorted(rep):
        for name in ("yaw", "pitch", "roll"):
            if gid == anchor_group:
                if name == "yaw" or not include_hv:
                    continue
            variables.append(Var("pose", gid, name))
    if not fixed_lens:
        lens_names = []
        if s.optimize_fov:
            lens_names.append("hfov")
        lens_names += list(DISTORTION_SETS.get(s.distortion, ()))
        if s.optimize_fisheye_factor:
            lens_names.append("fisheye_factor")
        if s.optimize_shear:
            lens_names += ["g", "t"]
        lenses_used = sorted({p.images[i].lens for i in images})
        for lid in lenses_used:
            lt = p.lenses[lid].lens_type
            for name in lens_names:
                if name == "fisheye_factor" and lt not in (G.LENS_FISHEYE, G.LENS_CIRCULAR_FISHEYE):
                    continue
                if lt in (G.LENS_EQUIRECT, G.LENS_CYLINDRICAL) and name != "hfov":
                    continue
                if name in s.per_image:
                    for i in images:
                        if p.images[i].lens == lid:
                            p.images[i].lens_overrides.setdefault(name, getattr(p.lenses[lid], name))
                            variables.append(Var("image", i, name))
                else:
                    variables.append(Var("lens", lid, name))
    if s.viewpoint:
        for i in images:
            if p.images[i].vp_enabled and i != anchor:
                for name in ("vp_x", "vp_y", "vp_z"):
                    variables.append(Var("vp", i, name))
    return variables, group_of, rep


def optimize(p: Project, images=None, fixed_lens: bool = False, max_nfev: int = 400, verbose: int = 0,
             loss: str | None = None, f_scale: float = 3.0) -> OptimizeResult:
    images = sorted(images if images is not None else p.included())
    iset = set(images)
    cps_all = [cp for cp in p.control_points if cp.i in iset and cp.j in iset]
    hv_mode = p.optimizer.hv_mode
    has_hv = any(cp.kind != CP_NORMAL for cp in cps_all)
    include_hv = has_hv and hv_mode == "single_pass"
    cps = [cp for cp in cps_all if cp.kind == CP_NORMAL or include_hv]
    if not cps:
        return OptimizeResult(False, 0, 0, 0, 0, 0, 0, 0, "no control points")
    variables, group_of, rep = build_variables(p, images, include_hv, fixed_lens)
    # drop lens variables that lack enough constraints
    n_normal = sum(1 for cp in cps if cp.kind == CP_NORMAL)
    if n_normal < 2 * len(variables) // 3:
        variables = [v for v in variables if v.kind == "pose" or v.name == "hfov"]
    prob = _Problem(p, images, variables, cps, group_of)
    prob.pose_rep = rep
    x0 = prob.x0()
    r0 = prob.residuals(x0)
    cost0 = float(0.5 * r0 @ r0)
    if variables:
        lo = np.array([BOUNDS[v.name][0] for v in variables])
        hi = np.array([_upper_bound(p, v) for v in variables])
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
        res = least_squares(prob.residuals, x0, jac_sparsity=prob.sparsity(), method="trf", x_scale="jac",
                            bounds=(lo, hi), loss=loss or ("soft_l1" if p.optimizer.robust else "linear"), f_scale=f_scale,
                            max_nfev=max_nfev, verbose=verbose, xtol=1e-10, ftol=1e-10)
        x = res.x
        msg, success = res.message, bool(res.success)
    else:
        x, msg, success = x0, "nothing to optimise", True
    _apply(p, prob, x, rep, group_of)
    if has_hv and hv_mode == "second_pass":
        level_with_lines(p, images)
    r1 = prob.residuals(x)
    errs = cp_errors(p, images)
    sel = np.array([e for cp, e in zip(p.control_points, errs) if cp.i in iset and cp.j in iset and not np.isnan(e)])
    return OptimizeResult(success, cost0, float(0.5 * r1 @ r1), float(sel.mean()) if len(sel) else 0.0,
                          float(sel.max()) if len(sel) else 0.0, float(np.sqrt((sel ** 2).mean())) if len(sel) else 0.0,
                          len(cps), len(variables), msg, list(map(float, errs)))


def _apply(p: Project, prob: _Problem, x, rep, group_of):
    cams = prob.cameras(x)
    for v, val in zip(prob.vars, x):
        if v.kind == "lens":
            setattr(p.lenses[v.owner], v.name, float(val))
        elif v.kind == "image":
            p.images[v.owner].lens_overrides[v.name] = float(val)
        elif v.kind == "vp":
            setattr(p.images[v.owner], v.name, float(val))
    for i in prob.images:
        c = cams[i]
        im = p.images[i]
        im.yaw, im.pitch, im.roll = float(c.yaw), float(c.pitch), float(c.roll)
    normalize_angles(p)


def normalize_angles(p: Project):
    for im in p.images:
        y, pt, r = G.matrix_to_ypr(G.rotation_matrix(im.yaw, im.pitch, im.roll))
        im.yaw, im.pitch, im.roll = y, pt, r


def rotate_panorama(p: Project, R_world: np.ndarray, images=None):
    """Apply a world rotation to all image orientations (numerical transform)."""
    for i, im in enumerate(p.images):
        if images is not None and i not in images:
            continue
        R = R_world @ G.rotation_matrix(im.yaw, im.pitch, im.roll)
        im.yaw, im.pitch, im.roll = G.matrix_to_ypr(R)


def level_with_lines(p: Project, images=None):
    """Second-pass levelling: find the global rotation that best satisfies
    vertical/horizontal/line control points, keeping relative geometry fixed."""
    images = images if images is not None else p.included()
    iset = set(images)
    hv = [cp for cp in p.control_points if cp.kind in (CP_VERTICAL, CP_HORIZONTAL) and cp.i in iset and cp.j in iset]
    if not hv:
        return False
    cams = {i: p.camera(i) for i in images}
    pairs = []
    for cp in hv:
        a = cams[cp.i].pixel_to_world(np.array([cp.x1]), np.array([cp.y1]))[0]
        b = cams[cp.j].pixel_to_world(np.array([cp.x2]), np.array([cp.y2]))[0]
        pairs.append((cp.kind, a, b))

    def res(x):
        R = G.rotation_matrix(0.0, x[0], x[1])
        out = []
        for kind, a, b in pairs:
            ra, rb = R @ a, R @ b
            if kind == CP_VERTICAL:
                out.append(G.normalize(np.cross(ra, rb)) @ UP)
            else:
                out.append(math.asin(np.clip(ra[1], -1, 1)) - math.asin(np.clip(rb[1], -1, 1)))
        return np.array(out)

    sol = least_squares(res, np.zeros(2))
    rotate_panorama(p, G.rotation_matrix(0.0, sol.x[0], sol.x[1]))
    return True


def prune_control_points(p: Project, max_error: float | None = None, worst_fraction: float | None = None,
                         keep_min_per_pair: int = 3) -> int:
    """Delete the worst control points (by last error).  Returns number removed."""
    cp_errors(p)
    errs = np.array([cp.error for cp in p.control_points])
    if not len(errs):
        return 0
    remove = np.zeros(len(errs), bool)
    if max_error is not None:
        remove |= errs > max_error
    if worst_fraction:
        k = int(len(errs) * worst_fraction)
        if k:
            remove[np.argsort(-errs)[:k]] = True
    # never strip a pair below keep_min_per_pair
    pair_counts: dict = {}
    for cp in p.control_points:
        pair_counts[(min(cp.i, cp.j), max(cp.i, cp.j))] = pair_counts.get((min(cp.i, cp.j), max(cp.i, cp.j)), 0) + 1
    order = np.argsort(-errs)
    removed = 0
    keep = np.ones(len(errs), bool)
    for n in order:
        if not remove[n]:
            continue
        cp = p.control_points[n]
        key = (min(cp.i, cp.j), max(cp.i, cp.j))
        if cp.kind == CP_NORMAL and pair_counts[key] <= keep_min_per_pair:
            continue
        pair_counts[key] -= 1
        keep[n] = False
        removed += 1
    p.control_points = [cp for cp, k in zip(p.control_points, keep) if k]
    return removed
