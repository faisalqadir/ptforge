"""High level project assistant: create project, align images, straighten, fit."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .features import ControlPointGenerator, generate_control_points
from .imageio import hfov_from_exif, image_size, read_exif
from .optimizer import cp_errors, normalize_angles, optimize, prune_control_points, rotate_panorama
from .project import CP_NORMAL, ControlPoint, ImageEntry, Lens, Project


def _lens_key(exif, w, h):
    return (exif.get("make", ""), exif.get("model", ""), exif.get("lens", ""), round(exif.get("focal_mm", 0.0), 1),
            max(w, h), min(w, h))


def create_project(paths, lens_type: str | None = None, hfov: float | None = None, fisheye_factor: float = 0.5,
                   detect_brackets: bool = True, detect_lens: bool = True) -> Project:
    """Build a project from image files: EXIF/auto lens init, lens grouping, bracket linking.

    With ``detect_lens`` (and no explicit ``lens_type``) circular fisheye frames are
    recognised from their image circle, which also initialises focal length, lens shift
    and the crop circle.
    """
    from .lens_detect import detect_image_circle, looks_fisheye

    p = Project()
    keys = {}
    members: dict[int, list] = {}
    for path in paths:
        w, h = image_size(path)
        ex = read_exif(path)
        key = _lens_key(ex, w, h)
        if key not in keys:
            keys[key] = len(p.lenses)
            p.lenses.append(Lens(focal_mm=ex.get("focal_mm", 0.0), crop_factor=ex.get("crop_factor", 0.0),
                                 camera=" ".join(x for x in (ex.get("make", ""), ex.get("model", "")) if x),
                                 lens_name=ex.get("lens", "")))
        lid = keys[key]
        members.setdefault(lid, []).append(len(p.images))
        p.images.append(ImageEntry(path=os.path.abspath(path), width=w, height=h, lens=lid, exif=ex, exif_ev=ex.get("ev100")))

    for lid, idxs in members.items():
        ln = p.lenses[lid]
        first = p.images[idxs[0]]
        w, h, ex = first.width, first.height, first.exif
        lt = lens_type
        circle = None
        if lt is None and detect_lens:
            from .imageio import load_image

            circles = []
            for i in idxs[:3]:  # a few frames: the circle edge depends on scene brightness
                c = detect_image_circle(load_image(p.abspath(p.images[i].path)).rgb)
                if c is not None:
                    circles.append(c)
            if circles:
                circle = tuple(float(np.median([c[k] for c in circles])) for k in range(3))
                lt = G.LENS_CIRCULAR_FISHEYE
            elif looks_fisheye(ex):
                lt = G.LENS_FISHEYE
        lt = lt or G.LENS_RECTILINEAR
        k = fisheye_factor if lt in (G.LENS_FISHEYE, G.LENS_CIRCULAR_FISHEYE) else 0.0
        fov = hfov
        if fov is None and circle is not None:
            # assume the image circle spans 180 degrees (true for nearly all circular fisheyes)
            f = circle[2] / float(G.lens_r(np.float64(math.pi / 2), k))
            fov = 2 * math.degrees(float(G.lens_theta(np.float64(w / 2 / f), k)))
        if fov is None:
            fov = hfov_from_exif(ex, w, h, lt, k if lt != G.LENS_RECTILINEAR else -1)
        if fov is None:
            fov = 50.0 if lt == G.LENS_RECTILINEAR else 180.0
        ln.lens_type, ln.hfov, ln.fisheye_factor = lt, float(fov), k
        if circle is not None:
            cx, cy, r = circle
            ln.crop.mode = "circle"
            ln.crop.cx, ln.crop.cy = cx + 0.5, cy + 0.5  # crop uses pixel-edge coordinates
            ln.crop.radius = r * 0.985  # stay clear of the flare/CA ring at the very edge
            ln.d, ln.e = cx + 0.5 - w / 2.0, cy + 0.5 - h / 2.0
        ln.init = {key: float(getattr(ln, key)) for key in _LENS_KEYS}
        if lt in (G.LENS_FISHEYE, G.LENS_CIRCULAR_FISHEYE):
            p.optimizer.optimize_fisheye_factor = True
            p.optimizer.distortion = "heavy_shift"
    if detect_brackets:
        link_brackets(p)
    return p


def link_brackets(p: Project, max_gap_s: float = 2.5) -> int:
    """Group consecutive images with differing exposure into bracket stacks."""
    evs = [im.exif_ev for im in p.images]
    if any(e is None for e in evs) or len(set(round(e, 1) for e in evs)) < 2:
        return 0
    # bracket length = number of distinct EVs in a repeating sequence
    seq = [round(e, 1) for e in evs]
    n = len(seq)
    distinct = sorted(set(seq))
    k = len(distinct)
    if n % k == 0 and all(len(set(seq[s:s + k])) == k for s in range(0, n, k)):
        gid = 0
        for s in range(0, n, k):
            for i in range(s, s + k):
                p.images[i].link_group = gid
            gid += 1
        p.exposure.hdr_mode = True
        ref = float(np.median(distinct))
        for im in p.images:
            im.ev = float(im.exif_ev - ref)
        return gid
    return 0


# --------------------------------------------------------------------------- initial orientation


def _rot_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _chain(images, tree, anchor):
    """Absolute rotations from tree edges (BFS from the anchor, then any other components)."""
    adj: dict[int, list] = {i: [] for i in images}
    for r in tree:
        adj[r.i].append((r.j, r.R_rel))  # b_cam = R a_cam
        adj[r.j].append((r.i, r.R_rel.T))
    R: dict[int, np.ndarray] = {}
    for root in [anchor] + [i for i in images if i != anchor]:
        if root in R or (root != anchor and not adj[root]):
            continue
        R[root] = np.eye(3)
        stack = [root]
        while stack:
            i = stack.pop()
            for j, Rij in adj[i]:
                if j not in R:
                    R[j] = R[i] @ Rij.T  # world d = R_i a = R_j b, b = Rij a  =>  R_j = R_i Rij^T
                    stack.append(j)
    return R


def _tree_paths(images, tree):
    """Number of tree edges between every pair of nodes (BFS per node; fine for a few hundred images)."""
    adj: dict[int, list] = {i: [] for i in images}
    for r in tree:
        adj[r.i].append(r.j)
        adj[r.j].append(r.i)
    dist = {}
    for s0 in images:
        d = {s0: 0}
        q = [s0]
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if y not in d:
                    d[y] = d[x] + 1
                    q.append(y)
        dist[s0] = d
    return dist


def _ranked_trees(pair_results, images, anchor, base_deg: float = 10.0, per_edge_deg: float = 8.0,
                  max_deg: float = 40.0, trials: int = 600, seed: int = 0):
    """Sample spanning trees over the candidate pairs and rank them by how many pairs agree.

    Repetitive texture (wallpaper, tiles, carpets) produces false pairs with many inliers, so
    a greedy maximum spanning tree can pick a wrong edge and then discard correct ones.  Each
    sampled tree defines all orientations; a pair agrees when its own rotation matches within
    a tolerance that grows with the loop it closes (the initial lens guess and handheld
    parallax bias every pairwise rotation a little, and the biases add up around a loop).

    Returns (edges, [(tree_edge_indices, agreeing_edge_indices), ...]) best first.
    """
    edges = [r for r in pair_results if r.R_rel is not None and r.n_inliers >= 8]
    if not edges:
        return [], []
    rng = np.random.default_rng(seed)
    strength = np.array([float(r.n_inliers) for r in edges])
    found = {}
    for t in range(trials):
        w = strength if t == 0 else strength * rng.random(len(edges)) ** 1.5
        parent = {i: i for i in images}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree = []
        for k in np.argsort(-w):
            ea, eb = find(edges[k].i), find(edges[k].j)
            if ea != eb:
                parent[ea] = eb
                tree.append(int(k))
        key = tuple(sorted(tree))
        if key in found:
            continue
        tedges = [edges[k] for k in tree]
        R = _chain(images, tedges, anchor)
        dist = _tree_paths(images, tedges)
        cons = []
        for k, r in enumerate(edges):
            if r.i in R and r.j in R and r.j in dist[r.i]:
                tol = min(max_deg, base_deg + per_edge_deg * max(0, dist[r.i][r.j] - 1))
                if _rot_deg((R[r.j].T @ R[r.i]).T @ r.R_rel) < tol:
                    cons.append(k)
        found[key] = (list(tree), cons, (_component(images, [edges[k] for k in cons], anchor), len(cons),
                                         float(sum(strength[k] for k in cons))))
    ranked = sorted(found.values(), key=lambda v: v[2], reverse=True)
    return edges, [(t, c) for t, c, _ in ranked]


def _component(images, edges, anchor) -> int:
    par = {i: i for i in images}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for r in edges:
        par[f(r.i)] = f(r.j)
    return sum(1 for i in images if f(i) == f(anchor))


def _pair_key(cp):
    return (min(cp.i, cp.j), max(cp.i, cp.j))


def robust_initialization(p: Project, pair_results, images, anchor: int, max_hypotheses: int = 6, progress=None):
    """Choose the image layout by testing the best spanning-tree hypotheses.

    Each hypothesis is solved (orientations + field of view, fisheye factor and main distortion)
    using only the pairs that agree with it; the solved geometry is then checked against *all*
    candidate pairs.  True loops close once the lens adapts, while a false pair can only ever be
    explained by its own hypothesis, so the layout explaining the most pairs wins.  The control
    points of pairs it cannot explain are removed.  Returns the set of rejected pairs.
    """
    edges, ranked = _ranked_trees(pair_results, images, anchor)
    if not ranked:
        return set()
    hyps = ranked[:max_hypotheses] if len(images) <= 16 else ranked[:1]
    all_cps = [ControlPoint(**cp.__dict__) for r in edges for cp in r.cps]
    f = max(p.camera(i).focal_px for i in images)
    best = None
    for h, (tree, cons) in enumerate(hyps):
        if progress:
            progress("initialize", h, len(hyps))
        q = p.copy()
        keep = {(min(edges[k].i, edges[k].j), max(edges[k].i, edges[k].j)) for k in cons}
        q.control_points = [ControlPoint(**cp.__dict__) for cp in all_cps if _pair_key(cp) in keep]
        R0 = G.rotation_matrix(q.images[anchor].yaw, q.images[anchor].pitch, q.images[anchor].roll)
        for i, Ri in _chain(images, [edges[k] for k in tree], anchor).items():
            q.images[i].yaw, q.images[i].pitch, q.images[i].roll = G.matrix_to_ypr(R0 @ Ri)
        for gid, members in q.link_groups().items():
            for m in members[1:]:
                q.images[m].yaw, q.images[m].pitch, q.images[m].roll = (q.images[members[0]].yaw, q.images[members[0]].pitch,
                                                                         q.images[members[0]].roll)
        q.optimizer.distortion = "medium"
        q.optimizer.per_image = []
        conn = sorted({cp.i for cp in q.control_points} | {cp.j for cp in q.control_points})
        if len(conn) < 2:
            continue
        before = _lens_snapshot(q)
        optimize(q, conn, fixed_lens=True, max_nfev=200)
        optimize(q, conn, max_nfev=200)
        if not _lenses_plausible(q, before):
            _lens_restore(q, before)
            optimize(q, conn, fixed_lens=True, max_nfev=200)
        # judge the solved layout against every candidate pair
        judge = q.copy()
        judge.control_points = [ControlPoint(**cp.__dict__) for cp in all_cps]
        errs = cp_errors(judge, images)
        by_pair: dict = {}
        for cp, e in zip(judge.control_points, errs):
            by_pair.setdefault(_pair_key(cp), []).append(e)
        med = {k: float(np.median(v)) for k, v in by_pair.items()}
        own = [med[k] for k in keep if k in med]
        lim = max(3.0 * float(np.median(own)) if own else 0.0, math.radians(2.5) * f)
        accepted = {k for k, m in med.items() if m < lim}
        acc_edges = [r for r in edges if (min(r.i, r.j), max(r.i, r.j)) in accepted]
        score = (_component(images, acc_edges, anchor), len(accepted),
                 -float(np.mean([med[k] for k in accepted])) if accepted else 0.0)
        if best is None or score > best[0]:
            best = (score, q, accepted)
    if best is None:
        return set()
    _, q, accepted = best
    for i in images:
        p.images[i].yaw, p.images[i].pitch, p.images[i].roll = q.images[i].yaw, q.images[i].pitch, q.images[i].roll
    for ln, lq in zip(p.lenses, q.lenses):
        for k in _LENS_KEYS:
            setattr(ln, k, getattr(lq, k))
    cand = {(min(r.i, r.j), max(r.i, r.j)) for r in pair_results if r.cps}
    bad = cand - accepted
    if bad:
        p.control_points = [cp for cp in p.control_points if cp.kind != CP_NORMAL or _pair_key(cp) not in bad]
    return bad


def drop_violated_pairs(p: Project, images, factor: float = 4.0, min_deg: float = 2.5) -> set:
    """After optimisation: remove pairs whose control points are violated far more than the rest
    (false matches that happened to agree loosely with the layout)."""
    by_pair: dict = {}
    for cp in p.control_points:
        if cp.kind == CP_NORMAL and cp.i in images and cp.j in images:
            by_pair.setdefault((min(cp.i, cp.j), max(cp.i, cp.j)), []).append(cp.error)
    if len(by_pair) < 3:
        return set()
    med = {k: float(np.median(v)) for k, v in by_pair.items()}
    typical = float(np.median(list(med.values())))
    f = max(p.camera(i).focal_px for i in images)
    lim = max(factor * typical, math.radians(min_deg) * f)
    bad = {k for k, m in med.items() if m > lim}
    # never disconnect an image completely
    deg: dict = {}
    for i, j in by_pair:
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    bad = {k for k in bad if deg[k[0]] > 1 and deg[k[1]] > 1}
    if bad:
        p.control_points = [cp for cp in p.control_points
                            if cp.kind != CP_NORMAL or (min(cp.i, cp.j), max(cp.i, cp.j)) not in bad]
    return bad


def initial_orientation(p: Project, pair_results, images, anchor: int):
    """Orient all images from the best-agreeing spanning tree (no lens solve; see robust_initialization)."""
    edges, ranked = _ranked_trees(pair_results, images, anchor)
    if not ranked:
        return set()
    tree = [edges[k] for k in ranked[0][0]]
    R0 = G.rotation_matrix(p.images[anchor].yaw, p.images[anchor].pitch, p.images[anchor].roll)
    R = {i: R0 @ Ri for i, Ri in _chain(images, tree, anchor).items()}
    for i, Ri in R.items():
        p.images[i].yaw, p.images[i].pitch, p.images[i].roll = G.matrix_to_ypr(Ri)
    for gid, members in p.link_groups().items():
        src = next((m for m in members if m in R), None)
        if src is None:
            continue
        for m in members:
            p.images[m].yaw, p.images[m].pitch, p.images[m].roll = p.images[src].yaw, p.images[src].pitch, p.images[src].roll
            R[m] = R[src]
    return set(R)


# --------------------------------------------------------------------------- straighten / fit


def straighten(p: Project, images=None):
    """Level the panorama: make the cameras' horizontal axes horizontal."""
    images = images if images is not None else p.included()
    rights, fwds = [], []
    for i in images:
        R = G.rotation_matrix(p.images[i].yaw, p.images[i].pitch, p.images[i].roll)
        rights.append(R[:, 0])
        fwds.append(R[:, 2])
    rights = np.array(rights)
    if len(rights) >= 2 and np.linalg.matrix_rank(rights, tol=0.05) >= 2:
        up = np.linalg.svd(rights)[2][-1]
    else:  # single image or all parallel: use the camera up vectors
        ups = np.array([G.rotation_matrix(p.images[i].yaw, p.images[i].pitch, p.images[i].roll)[:, 1] for i in images])
        up = G.normalize(ups.mean(axis=0))
    ups = np.array([G.rotation_matrix(p.images[i].yaw, p.images[i].pitch, p.images[i].roll)[:, 1] for i in images])
    if (ups @ up).sum() < 0:
        up = -up
    # rotation taking `up` to +Y
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, target)
    s, c = np.linalg.norm(v), float(up @ target)
    if s < 1e-12:
        Rl = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rl = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    rotate_panorama(p, Rl)
    center_panorama(p, images)
    return Rl


def center_panorama(p: Project, images=None, horizontal=True, vertical=False):
    images = images if images is not None else p.included()
    yaws = np.radians([p.images[i].yaw for i in images])
    # circular mean of yaw (weights equal)
    if horizontal:
        cov = coverage_lonlat(p, images, n=90)
        if cov is not None and cov[0] < 2 * math.pi * 0.98:
            mid_lon = cov[2]
        else:
            mid_lon = math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean())
            if cov is not None:  # full 360: put the anchor image in the centre
                a = p.optimizer.anchor if p.optimizer.anchor in images else images[0]
                mid_lon = math.radians(p.images[a].yaw)
        rotate_panorama(p, G.rotation_matrix(-math.degrees(mid_lon), 0, 0))
    if vertical:
        cov = coverage_lonlat(p, images, n=90)
        if cov is not None:
            mid_lat = (cov[3] + cov[4]) / 2
            rotate_panorama(p, G.rotation_matrix(0, -math.degrees(mid_lat), 0))


def coverage_lonlat(p: Project, images=None, n: int = 120):
    """Return (lon_span, lat_span, lon_centre, lat_min, lat_max) covered by the images (radians)."""
    images = images if images is not None else p.included()
    W, H = n * 2, n
    lon = (np.arange(W) + 0.5) / W * 2 * math.pi - math.pi
    lat = math.pi / 2 - (np.arange(H) + 0.5) / H * math.pi
    L, B = np.meshgrid(lon, lat)
    d = G._dir_from_lonlat(L, B)
    cov = np.zeros((H, W), bool)
    for i in images:
        cam = p.camera(i)
        x, y, valid = cam.world_to_pixel(d)
        cov |= valid & (x >= -0.5) & (x <= cam.width - 0.5) & (y >= -0.5) & (y <= cam.height - 0.5)
    if not cov.any():
        return None
    cols = cov.any(axis=0)
    rows = np.nonzero(cov.any(axis=1))[0]
    lat_max, lat_min = lat[rows[0]] + math.pi / H / 2, lat[rows[-1]] - math.pi / H / 2
    if rows[0] == 0:
        lat_max = math.pi / 2
    if rows[-1] == H - 1:
        lat_min = -math.pi / 2
    if cols.all():
        return (2 * math.pi, lat_max - lat_min, 0.0, lat_min, lat_max)
    # largest circular gap of empty columns
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
    best_len = min(best_len, W)
    span = (W - best_len) / W * 2 * math.pi
    gap_mid = (best_start + best_len / 2) % W
    centre = lon[0] + (gap_mid / W) * 2 * math.pi + math.pi - math.pi / W / 2
    centre = (centre + math.pi) % (2 * math.pi) - math.pi
    return (span, lat_max - lat_min, centre, lat_min, lat_max)


def choose_projection(p: Project, images=None):
    cov = coverage_lonlat(p, images)
    if cov is None:
        return
    span, vspan = math.degrees(cov[0]), math.degrees(cov[1])
    if span >= 300 or vspan >= 150:
        p.panorama.projection = "equirectangular"
    elif span > 120 and vspan <= 100:
        p.panorama.projection = "cylindrical"
    elif span > 120:
        p.panorama.projection = "equirectangular"
    elif span <= 120 and vspan <= 120:
        p.panorama.projection = "rectilinear"
    else:
        p.panorama.projection = "equirectangular"


def fit_panorama(p: Project, images=None, n: int = 200):
    """Set panorama hfov / vfov so that the images just fit."""
    images = images if images is not None else p.included()
    P = p.projection()
    # sample the borders of each image, project to the panorama plane
    us, vs = [], []
    for i in images:
        cam = p.camera(i)
        t = np.linspace(0, 1, 64)
        xs = np.concatenate([t * cam.width, np.full_like(t, cam.width), (1 - t) * cam.width, np.zeros_like(t)]) - 0.5
        ys = np.concatenate([np.zeros_like(t), t * cam.height, np.full_like(t, cam.height), (1 - t) * cam.height]) - 0.5
        gx, gy = np.meshgrid(np.linspace(0, cam.width, 16), np.linspace(0, cam.height, 12))
        xs = np.concatenate([xs, gx.ravel()])
        ys = np.concatenate([ys, gy.ravel()])
        d = cam.pixel_to_world(xs, ys)
        u, v, valid = P.forward(d)
        us.append(u[valid])
        vs.append(v[valid])
    u = np.concatenate(us)
    v = np.concatenate(vs)
    cov = coverage_lonlat(p, images)
    if p.panorama.projection in ("equirectangular", "cylindrical", "mercator", "architectural", "sinusoidal") and cov is not None and cov[0] > 2 * math.pi * 0.97:
        hfov = 360.0
    else:
        umax = float(np.max(np.abs(u)))
        hfov = _invert_extent(lambda h: P.u_of_hfov(h), umax, P.max_hfov)
    vmax = float(np.max(np.abs(v)))
    vfov = _invert_extent(lambda h: P.v_of_vfov(h), vmax, P.max_vfov)
    if cov is not None and cov[1] > math.pi * 0.97 and P.max_vfov >= 180:
        vfov = 180.0
    p.panorama.hfov = float(min(hfov, P.max_hfov))
    p.panorama.vfov = float(min(vfov, P.max_vfov))


def _invert_extent(fun, target, max_deg):
    lo, hi = 1e-4, math.radians(max_deg) / 2
    if fun(hi) <= target:
        return max_deg
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fun(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.degrees(2 * hi)


def optimum_size(p: Project, images=None) -> tuple[int, int]:
    """Panorama size at which the source resolution is preserved at the centre."""
    images = images if images is not None else p.included()
    P = p.projection()
    f = max(p.camera(i).focal_px for i in images) if images else 1000.0
    u_ext = 2 * P.u_of_hfov(math.radians(p.panorama.hfov) / 2)
    v_ext = 2 * P.v_of_vfov(math.radians(p.panorama.vfov) / 2)
    return max(16, int(round(u_ext * f))), max(16, int(round(v_ext * f)))


# --------------------------------------------------------------------------- main entry


@dataclass
class AlignReport:
    n_images: int
    n_cps: int
    connected: int
    mean_error: float
    max_error: float
    unconnected: list
    pairs: int
    message: str = ""


def align_images(p: Project, generate: bool = True, prealigned: bool = False, max_cps_per_pair: int = 30,
                 do_straighten: bool = True, do_fit: bool = True, choose_proj: bool = True, progress=None,
                 photometric: bool = True) -> AlignReport:
    progress = progress or (lambda *a, **k: None)
    images = p.included()
    anchor = p.optimizer.anchor if p.optimizer.anchor in images else images[0]
    p.optimizer.anchor = anchor
    if not prealigned:
        reset_lenses(p, images)
    results = []
    if generate:
        gen = ControlPointGenerator(p, max_cps_per_pair=max_cps_per_pair, progress=progress)
        results = generate_control_points(p, images, replace=True, prealigned=prealigned, generator=gen)
    if not generate and not prealigned:
        results = pair_results_from_cps(p, images)
    if not prealigned and results:
        p.images[anchor].yaw = p.images[anchor].pitch = p.images[anchor].roll = 0.0
        bad = robust_initialization(p, results, images, anchor, progress=progress)
        if bad:
            progress("warning", 0, 0, message=f"ignored {len(bad)} inconsistent image pair(s): {sorted(bad)}")
    connected = _connected_component(p, images, anchor)
    progress("optimize", 0, 3)
    # stage 1: orientations only (robust), stage 2: full lens model, then prune outliers
    robust = p.optimizer.robust
    p.optimizer.robust = True
    optimize(p, sorted(connected), fixed_lens=True)
    p.optimizer.robust = robust
    progress("optimize", 1, 3)
    before = _lens_snapshot(p)
    res = _optimize_and_prune(p, sorted(connected))
    bad = drop_violated_pairs(p, sorted(connected))
    if bad:
        progress("warning", 0, 0, message=f"ignored {len(bad)} image pair(s) that contradict the others: {sorted(bad)}")
        _lens_restore(p, before)
        optimize(p, sorted(connected), fixed_lens=True)
        res = _optimize_and_prune(p, sorted(connected))
    if not _lenses_plausible(p, before):
        # the full lens model ran away (weak geometry or bad pairs): retry with a simpler one
        _lens_restore(p, before)
        saved = (p.optimizer.distortion, p.optimizer.optimize_fisheye_factor)
        p.optimizer.distortion, p.optimizer.optimize_fisheye_factor = "medium", False
        res = _optimize_and_prune(p, sorted(connected))
        if not _lenses_plausible(p, before):
            _lens_restore(p, before)
            p.optimizer.distortion = "none"
            res = _optimize_and_prune(p, sorted(connected))
        p.optimizer.distortion, p.optimizer.optimize_fisheye_factor = saved
        progress("warning", 0, 0, message="lens distortion was poorly determined; a simpler lens model was used")
    progress("optimize", 3, 3)
    has_hv = any(cp.kind != CP_NORMAL for cp in p.control_points)
    if do_straighten and not (has_hv and p.optimizer.hv_mode in ("single_pass", "second_pass")):
        straighten(p, sorted(connected))
    normalize_angles(p)
    if choose_proj:
        choose_projection(p, sorted(connected))
    if do_fit:
        fit_panorama(p, sorted(connected))
    if photometric and p.exposure.optimize_exposure != "off":
        from .photometric import optimize_photometric

        try:
            optimize_photometric(p)
        except Exception as e:  # photometric failure must not break alignment
            progress("warning", 0, 0, message=f"photometric optimisation failed: {e}")
    unconnected = [i for i in images if i not in connected]
    cp_errors(p)
    return AlignReport(len(images), len(p.control_points), len(connected), res.mean_error, res.max_error, unconnected,
                       len([r for r in results if r.cps]), res.message)


def reset_lenses(p: Project, images=None):
    """Restore every unlocked lens to its initial estimate (EXIF / image circle).

    Projects from older versions have no stored estimate; it is recomputed from EXIF then.
    """
    images = images if images is not None else p.included()
    for lid, ln in enumerate(p.lenses):
        users = [i for i in images if p.images[i].lens == lid]
        if not users or ln.locked:
            continue
        if not ln.init:
            # same estimate a new project would make: equisolid fisheyes, 180 degree image circle
            im = p.images[users[0]]
            fisheye = ln.lens_type in (G.LENS_FISHEYE, G.LENS_CIRCULAR_FISHEYE)
            k = 0.5 if fisheye else -1.0
            fov = None
            if ln.crop.mode == "circle" and ln.crop.radius:
                f = ln.crop.radius / 0.985 / float(G.lens_r(np.float64(math.pi / 2), k))
                fov = 2 * math.degrees(float(G.lens_theta(np.float64(im.width / 2 / f), k)))
            fov = fov or hfov_from_exif(im.exif, im.width, im.height, ln.lens_type, k) or ln.hfov
            ln.init = {key: 0.0 for key in _LENS_KEYS}
            ln.init.update(hfov=float(fov), fisheye_factor=0.5 if fisheye else 0.0)
            if ln.crop.mode == "circle" and ln.crop.cx:
                ln.init.update(d=ln.crop.cx - im.width / 2.0, e=ln.crop.cy - im.height / 2.0)
        for key, v in ln.init.items():
            setattr(ln, key, v)
        for i in users:
            p.images[i].lens_overrides = {}


def pair_results_from_cps(p: Project, images):
    """Pairwise rotations from the existing control points (re-initialise without regenerating)."""
    from .features import PairResult, kabsch

    cams = {i: p.camera(i) for i in images}
    by_pair: dict = {}
    for cp in p.control_points:
        if cp.kind == CP_NORMAL and cp.i in cams and cp.j in cams and cp.i != cp.j:
            by_pair.setdefault((cp.i, cp.j), []).append(cp)
    out = []
    for (i, j), cps in by_pair.items():
        if len(cps) < 3:
            continue
        a = cams[i].pixel_to_ray(np.array([c.x1 for c in cps]), np.array([c.y1 for c in cps]))
        b = cams[j].pixel_to_ray(np.array([c.x2 for c in cps]), np.array([c.y2 for c in cps]))
        out.append(PairResult(i, j, len(cps), len(cps), kabsch(a, b), list(cps)))
    return out


def _optimize_and_prune(p: Project, images):
    res = optimize(p, images)
    for _ in range(3):
        errs = np.array([cp.error for cp in p.control_points if cp.kind == CP_NORMAL])
        if not len(errs):
            break
        removed = prune_control_points(p, max_error=max(3.0 * float(np.median(errs)), 2.5))
        if removed == 0:
            break
        res = optimize(p, images)
    return res


_LENS_KEYS = ("hfov", "fisheye_factor", "a", "b", "c", "d", "e", "g", "t")


def _lens_snapshot(p: Project):
    return ([{k: getattr(ln, k) for k in _LENS_KEYS} for ln in p.lenses],
            [dict(im.lens_overrides) for im in p.images])


def _lens_restore(p: Project, snap):
    lenses, overrides = snap
    for ln, vals in zip(p.lenses, lenses):
        for k, v in vals.items():
            setattr(ln, k, v)
    for im, ov in zip(p.images, overrides):
        im.lens_overrides = dict(ov)


def _lenses_plausible(p: Project, snap) -> bool:
    """Reject lens solutions no real lens has (they mean the geometry was not determined)."""
    lenses, _ = snap
    for lid, (ln, old) in enumerate(zip(p.lenses, lenses)):
        users = [im for im in p.images if im.lens == lid]
        if not users:
            continue
        im = users[0]
        ref = ln.init or old  # judge against the lens estimate, not a previous (maybe drifted) solution
        if abs(ln.hfov - ref["hfov"]) > 0.15 * ref["hfov"]:
            return False
        if max(abs(ln.a), abs(ln.b), abs(ln.c)) > 0.25:
            return False
        if abs(ln.fisheye_factor - ref["fisheye_factor"]) > 0.3:
            return False
        if max(abs(ln.d), abs(ln.e)) > 0.06 * min(im.width, im.height):
            return False
    return True


def drop_inconsistent_pairs(p: Project, results, max_deg: float = 6.0) -> int:
    """Remove control points of pairs whose measured relative rotation disagrees with
    the spanning-tree orientation (false matches on repetitive/non-overlapping images)."""
    bad = set()
    for r in results:
        if r.R_rel is None or not r.cps:
            continue
        Ri = G.rotation_matrix(p.images[r.i].yaw, p.images[r.i].pitch, p.images[r.i].roll)
        Rj = G.rotation_matrix(p.images[r.j].yaw, p.images[r.j].pitch, p.images[r.j].roll)
        D = (Rj.T @ Ri).T @ r.R_rel
        ang = math.degrees(math.acos(np.clip((np.trace(D) - 1) / 2, -1, 1)))
        if ang > max_deg:
            bad.add((r.i, r.j))
            r.cps = []
    if bad:
        p.control_points = [cp for cp in p.control_points if (cp.i, cp.j) not in bad and (cp.j, cp.i) not in bad]
    return len(bad)


def _connected_component(p: Project, images, anchor):
    adj = {i: set() for i in images}
    for cp in p.control_points:
        if cp.i in adj and cp.j in adj and cp.i != cp.j:
            adj[cp.i].add(cp.j)
            adj[cp.j].add(cp.i)
    for members in p.link_groups().values():
        for a in members:
            for b in members:
                if a != b and a in adj and b in adj:
                    adj[a].add(b)
    seen, stack = {anchor}, [anchor]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen
