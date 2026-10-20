"""Control point generation: feature detection, matching and geometric verification.

Matches are verified with a *rotation* model on camera rays (using the current
lens estimate), which is correct for any lens type including fisheyes and
tolerant of a rough focal length.  Rectilinear pairs fall back to a homography
when the lens is poorly known.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as G
from .imageio import load_image
from .project import CP_NORMAL, ControlPoint, Project

DEFAULT_WORK_SIZE = 1600  # long side used for detection


@dataclass
class Features:
    xy: np.ndarray  # (n,2) full-resolution pixel coordinates (cv2 convention)
    desc: np.ndarray  # (n,128) float32 RootSIFT
    response: np.ndarray
    size: np.ndarray


@dataclass
class PairResult:
    i: int
    j: int
    n_matches: int
    n_inliers: int
    R_rel: np.ndarray | None  # maps camera-i rays into camera-j frame
    cps: list


def _gray_u8(rgb: np.ndarray, normalize: bool = True) -> np.ndarray:
    g = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.float32), cv2.COLOR_RGB2GRAY)
    if normalize:  # exposure-invariant: stretch to a common brightness range
        lo, hi = np.percentile(g, [0.5, 99.5])
        g = (g - lo) / max(hi - lo, 1e-6)
    return np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)


def detect_features(rgb: np.ndarray, work_size: int = DEFAULT_WORK_SIZE, max_features: int = 6000,
                    mask: np.ndarray | None = None) -> Features:
    h, w = rgb.shape[:2]
    s = min(1.0, work_size / max(h, w))
    g = _gray_u8(rgb)
    if s < 1.0:
        g = cv2.resize(g, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02, edgeThreshold=12)
    kps, desc = sift.detectAndCompute(g, mask)
    if desc is None or not kps:
        return Features(np.zeros((0, 2)), np.zeros((0, 128), np.float32), np.zeros(0), np.zeros(0))
    desc = desc.astype(np.float32)
    desc /= np.maximum(desc.sum(axis=1, keepdims=True), 1e-7)  # RootSIFT
    desc = np.sqrt(desc)
    pts = np.array([k.pt for k in kps], dtype=np.float64)
    # map from work-image pixel centres to full-res pixel centres
    xy = (pts + 0.5) / s - 0.5
    return Features(xy, desc, np.array([k.response for k in kps]), np.array([k.size / s for k in kps]))


def match_descriptors(d1: np.ndarray, d2: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), np.int64)
    index = dict(algorithm=1, trees=4)
    flann = cv2.FlannBasedMatcher(index, dict(checks=64))
    m12 = flann.knnMatch(d1, d2, k=2)
    m21 = flann.knnMatch(d2, d1, k=1)
    back = np.full(len(d2), -1, np.int64)
    for m in m21:
        if m:
            back[m[0].queryIdx] = m[0].trainIdx
    out = []
    for m in m12:
        if len(m) < 2:
            continue
        a, b = m
        if a.distance < ratio * b.distance and back[a.trainIdx] == a.queryIdx:
            out.append((a.queryIdx, a.trainIdx))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def kabsch(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rotation R minimising sum |b - R a|^2 for unit vectors (n,3)."""
    if w is None:
        H = a.T @ b
    else:
        H = (a * w[:, None]).T @ b
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def ransac_rotation(a: np.ndarray, b: np.ndarray, thresh_rad: float, iters: int = 600, rng=None):
    n = len(a)
    if n < 3:
        return None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(0)
    best, best_inl, best_score = None, None, -1
    cos_t = math.cos(thresh_rad)
    for _ in range(iters):
        idx = rng.choice(n, 2, replace=False)
        aa, bb = a[idx], b[idx]
        if abs(float(aa[0] @ aa[1])) > 0.9999:
            continue
        # angle between the pair must be preserved by a rotation
        if abs(math.acos(np.clip(aa[0] @ aa[1], -1, 1)) - math.acos(np.clip(bb[0] @ bb[1], -1, 1))) > 2 * thresh_rad:
            continue
        aa3 = np.vstack([aa, np.cross(aa[0], aa[1])])
        bb3 = np.vstack([bb, np.cross(bb[0], bb[1])])
        R = kabsch(G.normalize(aa3), G.normalize(bb3))
        c = np.einsum("ij,ij->i", a @ R.T, b)
        inl = c > cos_t
        score = int(inl.sum())
        if score > best_score:
            best, best_inl, best_score = R, inl, score
            if score > 0.9 * n:
                break
    if best is None or best_score < 3:
        return None, np.zeros(n, bool)
    for _ in range(3):  # refine
        R = kabsch(a[best_inl], b[best_inl])
        c = np.einsum("ij,ij->i", a @ R.T, b)
        inl = c > cos_t
        if inl.sum() < 3:
            break
        best, best_inl = R, inl
    return best, best_inl


def _spread(xy: np.ndarray, score: np.ndarray, w: int, h: int, k: int) -> np.ndarray:
    """Pick up to k indices spread over the image (grid bucketing, best score first)."""
    n = len(xy)
    if n <= k:
        return np.arange(n)
    g = max(2, int(math.ceil(math.sqrt(k * 1.5))))
    cx = np.clip((xy[:, 0] / w * g).astype(int), 0, g - 1)
    cy = np.clip((xy[:, 1] / h * g).astype(int), 0, g - 1)
    cell = cy * g + cx
    order = np.argsort(-score)
    picked, seen = [], {}
    rounds = 0
    while len(picked) < k and rounds < 50:
        for idx in order:
            if len(picked) >= k:
                break
            c = cell[idx]
            if seen.get(c, 0) == rounds and idx not in picked:
                picked.append(idx)
                seen[c] = rounds + 1
        rounds += 1
    return np.array(sorted(set(picked)))


def _valid_mask_for(project: Project, idx: int, shape) -> np.ndarray | None:
    """Detection mask from crop circle / rectangle and exclusion masks."""
    from .masks import source_valid_mask

    m = source_valid_mask(project, idx, shape[1], shape[0], include_paint=True)
    return None if m is None else (m > 0).astype(np.uint8) * 255


class ControlPointGenerator:
    def __init__(self, project: Project, work_size: int = DEFAULT_WORK_SIZE, max_cps_per_pair: int = 30,
                 max_features: int = 6000, threads: int | None = None, progress=None):
        self.p = project
        self.work_size = work_size
        self.max_cps = max_cps_per_pair
        self.max_features = max_features
        self.threads = threads or min(8, os.cpu_count() or 4)
        self.features: dict[int, Features] = {}
        self.progress = progress or (lambda *a, **k: None)

    # ----------------------------------------------------------------- detection
    def detect(self, indices):
        todo = [i for i in indices if i not in self.features]

        def work(i):
            im = self.p.images[i]
            rgb = load_image(self.p.abspath(im.path)).rgb
            ws = self.work_size
            if self.p.lenses[im.lens].lens_type in (G.LENS_FISHEYE, G.LENS_CIRCULAR_FISHEYE):
                ws = int(ws * 1.8)  # detail is compressed towards the edge of the circle
            s = min(1.0, ws / max(rgb.shape[:2]))
            m = _valid_mask_for(self.p, i, rgb.shape)
            if m is not None and s < 1.0:
                m = cv2.resize(m, (int(round(rgb.shape[1] * s)), int(round(rgb.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
            f = detect_features(rgb, ws, self.max_features, None)
            if m is not None and len(f.xy):
                xs = np.clip(((f.xy[:, 0] + 0.5) * s).astype(int), 0, m.shape[1] - 1)
                ys = np.clip(((f.xy[:, 1] + 0.5) * s).astype(int), 0, m.shape[0] - 1)
                keep = m[ys, xs] > 0
                f = Features(f.xy[keep], f.desc[keep], f.response[keep], f.size[keep])
            return i, f

        done = 0
        with ThreadPoolExecutor(self.threads) as ex:
            for i, f in ex.map(work, todo):
                self.features[i] = f
                done += 1
                self.progress("detect", done, len(todo))

    # ----------------------------------------------------------------- pairs
    def candidate_pairs(self, indices, prealigned: bool = False, link_brackets: bool = True):
        groups = {i: self.p.images[i].link_group for i in indices}
        pairs = []
        cams = {i: self.p.camera(i) for i in indices} if prealigned else None
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if link_brackets and groups[i] >= 0 and groups[i] == groups[j]:
                    continue
                if prealigned and not images_overlap(cams[i], cams[j]):
                    continue
                pairs.append((i, j))
        if not prealigned and len(indices) > 30:
            pairs = self._prune_pairs(indices, pairs)
        return pairs

    def _prune_pairs(self, indices, pairs, keep_per_image: int = 8):
        """For big projects: rank candidate partners by quick matching of strong features."""
        sub = {}
        for i in indices:
            f = self.features[i]
            o = np.argsort(-f.response)[:400]
            sub[i] = f.desc[o]
        score = {}

        def work(pr):
            i, j = pr
            return pr, len(match_descriptors(sub[i], sub[j], 0.75))

        with ThreadPoolExecutor(self.threads) as ex:
            for pr, s in ex.map(work, pairs):
                score[pr] = s
        keep = set()
        for i in indices:
            mine = sorted([pr for pr in pairs if i in pr], key=lambda pr: -score[pr])
            for pr in mine[:keep_per_image]:
                if score[pr] >= 8:
                    keep.add(pr)
        return sorted(keep)

    # ----------------------------------------------------------------- matching
    def match_pair(self, i: int, j: int, cam_i: G.Camera | None = None, cam_j: G.Camera | None = None,
                   prealigned: bool = False) -> PairResult:
        fi, fj = self.features[i], self.features[j]
        m = match_descriptors(fi.desc, fj.desc)
        if len(m) < 6:
            return PairResult(i, j, len(m), 0, None, [])
        ci = cam_i or self.p.camera(i)
        cj = cam_j or self.p.camera(j)
        pi, pj = fi.xy[m[:, 0]], fj.xy[m[:, 1]]
        a = ci.pixel_to_ray(pi[:, 0], pi[:, 1])
        b = cj.pixel_to_ray(pj[:, 0], pj[:, 1])
        fov = math.radians(max(ci.hfov, 1.0))
        # angular gate: loose enough for a rough focal length and handheld parallax.  False
        # pairs that slip through are removed later by the global consistency check.
        thresh = min(max(math.radians(0.35), 0.012 * min(fov, math.pi)), math.radians(2.5))
        if prealigned:
            # with known positions, reject matches that disagree with the layout first
            wa, wb = a @ ci.R.T, b @ cj.R.T
            ok = np.einsum("ij,ij->i", wa, wb) > math.cos(math.radians(8) + 6 * thresh)
            if ok.sum() >= 6:
                m, pi, pj, a, b = m[ok], pi[ok], pj[ok], a[ok], b[ok]
        R, inl = ransac_rotation(a, b, thresh)
        if (R is None or inl.sum() < 8) and ci.lens_type == G.LENS_RECTILINEAR and cj.lens_type == G.LENS_RECTILINEAR:
            Hm, hin = cv2.findHomography(pi.astype(np.float32), pj.astype(np.float32), cv2.USAC_MAGSAC, 3.0 + 0.004 * ci.width)
            if Hm is not None and hin is not None and hin.sum() >= 8:
                inl = hin.ravel().astype(bool)
                R = kabsch(a[inl], b[inl])
        n_in = int(inl.sum()) if R is not None else 0
        if R is None or n_in < 12 or (n_in < 30 and n_in < 0.15 * len(m)):
            return PairResult(i, j, len(m), 0, None, [])
        # reject degenerate clusters: the inlier rays must cover some angle
        ain = a[inl]
        spread = math.sqrt(float(np.mean(np.sum((ain - ain.mean(axis=0)) ** 2, axis=1))))
        if spread < math.radians(1.5) and n_in < 40:
            return PairResult(i, j, len(m), 0, None, [])
        ii = np.nonzero(inl)[0]
        # control points preferably from the tight core of the inliers (the loose gate is only
        # there to decide that the pair exists)
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", a[ii] @ R.T, b[ii]), -1, 1))
        tight = ang < max(math.radians(0.35), 0.45 * thresh)
        if tight.sum() >= max(12, 0.5 * len(ii)):
            ii = ii[tight]
        score = np.minimum(fi.response[m[ii, 0]], fj.response[m[ii, 1]])
        sel = ii[_spread(pi[ii], score, ci.width, ci.height, self.max_cps)]
        cps = [ControlPoint(i, float(pi[k, 0]), float(pi[k, 1]), j, float(pj[k, 0]), float(pj[k, 1]), CP_NORMAL) for k in sel]
        return PairResult(i, j, len(m), int(inl.sum()), R, cps)

    def run(self, indices=None, prealigned: bool = False, link_brackets: bool = True, pairs=None) -> list[PairResult]:
        indices = list(indices if indices is not None else self.p.included())
        self.detect(indices)
        if pairs is None:
            pairs = self.candidate_pairs(indices, prealigned, link_brackets)
        cams = {i: self.p.camera(i) for i in indices}
        results = []

        def work(pr):
            i, j = pr
            return self.match_pair(i, j, cams[i], cams[j], prealigned)

        done = 0
        with ThreadPoolExecutor(self.threads) as ex:
            for r in ex.map(work, pairs):
                results.append(r)
                done += 1
                self.progress("match", done, len(pairs))
        return results


def images_overlap(ci: G.Camera, cj: G.Camera, margin_deg: float = 5.0) -> bool:
    """Coarse overlap test by projecting a border/grid of rays of i into j."""
    ys, xs = np.mgrid[0:ci.height:ci.height / 8, 0:ci.width:ci.width / 8]
    d = ci.pixel_to_world(xs.ravel() + ci.width / 16, ys.ravel() + ci.height / 16)
    x, y, valid = cj.world_to_pixel(d)
    mx, my = cj.width * margin_deg / max(cj.hfov, 1), cj.height * margin_deg / max(cj.hfov, 1)
    inside = valid & (x > -mx) & (x < cj.width + mx) & (y > -my) & (y < cj.height + my)
    # also require forward hemisphere for rectilinear targets
    return bool(inside.any())


def generate_control_points(project: Project, indices=None, replace: bool = True, prealigned: bool = False,
                            max_cps_per_pair: int = 30, progress=None, generator: ControlPointGenerator | None = None):
    gen = generator or ControlPointGenerator(project, max_cps_per_pair=max_cps_per_pair, progress=progress)
    results = gen.run(indices, prealigned=prealigned, link_brackets=project.exposure.link_brackets)
    idx = set(indices if indices is not None else project.included())
    if replace:
        project.control_points = [cp for cp in project.control_points
                                  if not (cp.kind == CP_NORMAL and cp.i in idx and cp.j in idx)]
    for r in results:
        project.control_points.extend(r.cps)
    return results
