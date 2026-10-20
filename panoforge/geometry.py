"""Core geometry: rotations, the source-lens camera model and output projections.

Conventions
-----------
World / camera frame: X right, Y up, Z forward (looking direction).
Rotations are given in degrees as yaw (about Y, positive turns right),
pitch (about X, positive looks up) and roll (about Z, positive rotates the
image clockwise as seen by the viewer).  ``R = Ryaw @ Rpitch @ Rroll`` maps a
camera-frame ray into the world frame.

Source lens model (per image, Panorama-Tools compatible parameters)::

    camera ray --lens mapping--> ideal image plane (pixels, about the centre)
               --radial a,b,c--> distorted plane
               --shear g,t-----> sheared plane
               --shift d,e-----> pixel coordinates (cv2 convention: integer = pixel centre)

The lens mapping is a single continuous family controlled by ``fisheye_factor`` k::

    k > 0 : r = f * sin(k*theta) / k      (k=1 orthographic, k=0.5 equisolid)
    k = 0 : r = f * theta                 (equidistant fisheye)
    k < 0 : r = f * tan(|k|*theta) / |k|  (k=-0.5 stereographic, k=-1 rectilinear)

plus two non-radial types for already-stitched sources: cylindrical and
equirectangular.  The radial polynomial follows Panorama Tools:
``r_src = r_ideal * (a r^3 + b r^2 + c r + (1 - a - b - c))`` where r is normalised
by half the shorter image side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DEG = math.pi / 180.0

# --------------------------------------------------------------------------- rotations


def rot_yaw(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_pitch(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float64)


def rot_roll(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera -> world rotation."""
    return rot_yaw(yaw_deg * DEG) @ rot_pitch(pitch_deg * DEG) @ rot_roll(roll_deg * DEG)


def matrix_to_ypr(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rotation_matrix` (degrees)."""
    # R = Ry(y) Rx(p) Rz(r).  Forward vector = R[:,2] = (sin y cos p, sin p, cos y cos p)
    sp = float(np.clip(R[1, 2], -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        yaw = math.atan2(R[0, 2], R[2, 2])
        roll = math.atan2(R[1, 0], R[1, 1])
    else:  # gimbal lock: looking straight up/down, fold roll into yaw
        roll = 0.0
        yaw = math.atan2(-R[2, 0], R[0, 0])
    return yaw / DEG, pitch / DEG, roll / DEG


def normalize(v: np.ndarray, axis=-1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-300)


# --------------------------------------------------------------------------- lens

LENS_RECTILINEAR = "rectilinear"
LENS_FISHEYE = "fisheye"  # radial family, see fisheye_factor
LENS_CIRCULAR_FISHEYE = "circular_fisheye"  # same math as fisheye, circular crop
LENS_CYLINDRICAL = "cylindrical"
LENS_EQUIRECT = "equirectangular"

LENS_TYPES = (LENS_RECTILINEAR, LENS_FISHEYE, LENS_CIRCULAR_FISHEYE, LENS_CYLINDRICAL, LENS_EQUIRECT)

FISHEYE_PRESETS = {
    "equidistant": 0.0,
    "equisolid": 0.5,
    "orthographic": 1.0,
    "stereographic": -0.5,
}


@dataclass
class Camera:
    """Everything needed to map between a source image's pixels and world rays."""

    width: int
    height: int
    lens_type: str = LENS_RECTILINEAR
    hfov: float = 50.0  # degrees, horizontal field of view (ideal, ignoring distortion)
    fisheye_factor: float = 0.0  # only for fisheye types
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0  # horizontal shift in pixels
    e: float = 0.0  # vertical shift in pixels
    g: float = 0.0  # horizontal shear
    t: float = 0.0  # vertical shear
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # viewpoint correction (parallax): camera centre offset, in units of the scene
    # plane distance; plane orientation given by vp_pan / vp_tilt (degrees)
    vp_x: float = 0.0
    vp_y: float = 0.0
    vp_z: float = 0.0
    vp_pan: float = 0.0
    vp_tilt: float = 0.0
    R: np.ndarray = field(default=None, repr=False)  # cached camera->world

    def __post_init__(self):
        self.update()

    def update(self):
        self.R = rotation_matrix(self.yaw, self.pitch, self.roll)
        return self

    # ---- intrinsics helpers
    @property
    def k(self) -> float:
        if self.lens_type == LENS_RECTILINEAR:
            return -1.0
        return float(self.fisheye_factor)

    @property
    def radial(self) -> bool:
        return self.lens_type in (LENS_RECTILINEAR, LENS_FISHEYE, LENS_CIRCULAR_FISHEYE)

    @property
    def norm_radius(self) -> float:
        return min(self.width, self.height) / 2.0

    @property
    def focal_px(self) -> float:
        h = self.hfov * DEG / 2.0
        if not self.radial:
            return (self.width / 2.0) / h
        return (self.width / 2.0) / lens_r(np.float64(h), self.k)

    def focal_mm(self, sensor_width_mm: float) -> float:
        return self.focal_px * sensor_width_mm / self.width

    @property
    def has_viewpoint(self) -> bool:
        return bool(self.vp_x or self.vp_y or self.vp_z)

    # ---- source pixel <-> camera ray
    def ray_to_pixel(self, rays: np.ndarray):
        """rays (...,3) in *camera* frame -> (x, y, valid) source pixel coords."""
        X, Y, Z = rays[..., 0], rays[..., 1], rays[..., 2]
        f = self.focal_px
        if self.radial:
            rho = np.hypot(X, Y)
            theta = np.arctan2(rho, Z)
            r, valid = lens_r_valid(theta, self.k)
            r = r * f
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.where(rho > 1e-12, r / np.maximum(rho, 1e-300), f)
            xi = X * s
            yi = -Y * s
        elif self.lens_type == LENS_CYLINDRICAL:
            lon = np.arctan2(X, Z)
            h = np.hypot(X, Z)
            xi = f * lon
            with np.errstate(divide="ignore", invalid="ignore"):
                yi = -f * (Y / np.maximum(h, 1e-12))
            valid = h > 1e-9
        else:  # equirectangular
            lon = np.arctan2(X, Z)
            lat = np.arctan2(Y, np.hypot(X, Z))
            xi = f * lon
            yi = -f * lat
            valid = np.ones(np.shape(X), dtype=bool)
        if self.a or self.b or self.c:
            # beyond the first turning point the polynomial folds back into the frame
            valid = valid & (np.hypot(xi, yi) <= self.radial_limit() * self.norm_radius)
        x, y = self.ideal_to_pixel(xi, yi)
        return x, y, valid

    def radial_limit(self) -> float:
        """Largest normalised ideal radius where the distortion is still monotonic."""
        key = (self.a, self.b, self.c)
        if getattr(self, "_rlim_key", None) == key:
            return self._rlim
        a, b, c = key
        d0 = 1.0 - (a + b + c)
        r = np.linspace(0.0, 20.0, 4001)
        deriv = ((4 * a * r + 3 * b) * r + 2 * c) * r + d0
        bad = np.nonzero(deriv <= 1e-3)[0]
        lim = float(r[bad[0]]) if len(bad) else float("inf")
        self._rlim_key, self._rlim = key, lim
        return lim

    def ideal_to_pixel(self, xi, yi):
        if self.a or self.b or self.c:
            R0 = self.norm_radius
            rn = np.hypot(xi, yi) / R0
            d0 = 1.0 - (self.a + self.b + self.c)
            sc = ((self.a * rn + self.b) * rn + self.c) * rn + d0
            xd, yd = xi * sc, yi * sc
        else:
            xd, yd = xi, yi
        if self.g or self.t:
            xs = xd + self.g * yd
            ys = yd + self.t * xd
        else:
            xs, ys = xd, yd
        x = xs + self.width / 2.0 + self.d - 0.5
        y = ys + self.height / 2.0 + self.e - 0.5
        return x, y

    def pixel_to_ideal(self, x, y):
        xs = np.asarray(x, dtype=np.float64) + 0.5 - self.width / 2.0 - self.d
        ys = np.asarray(y, dtype=np.float64) + 0.5 - self.height / 2.0 - self.e
        if self.g or self.t:
            det = 1.0 - self.g * self.t
            xd = (xs - self.g * ys) / det
            yd = (ys - self.t * xs) / det
        else:
            xd, yd = xs, ys
        if self.a or self.b or self.c:
            R0 = self.norm_radius
            rd = np.hypot(xd, yd) / R0
            ru = _invert_radial(rd, self.a, self.b, self.c)
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.where(rd > 1e-12, ru / np.maximum(rd, 1e-300), 1.0 / max(1.0 - self.a - self.b - self.c, 1e-6))
            xi, yi = xd * s, yd * s
        else:
            xi, yi = xd, yd
        return xi, yi

    def pixel_to_ray(self, x, y) -> np.ndarray:
        """source pixel coords -> unit rays in *camera* frame (...,3)."""
        xi, yi = self.pixel_to_ideal(x, y)
        f = self.focal_px
        if self.radial:
            r = np.hypot(xi, yi) / f
            theta = lens_theta(r, self.k)
            st = np.sin(theta)
            with np.errstate(invalid="ignore", divide="ignore"):
                rr = np.maximum(np.hypot(xi, yi), 1e-300)
                X = np.where(r > 1e-12, xi / rr * st, xi / f)
                Y = np.where(r > 1e-12, -yi / rr * st, -yi / f)
            Z = np.cos(theta)
            v = np.stack([X, Y, Z], axis=-1)
        elif self.lens_type == LENS_CYLINDRICAL:
            lon = xi / f
            v = np.stack([np.sin(lon), -yi / f, np.cos(lon)], axis=-1)
        else:
            lon = xi / f
            lat = -yi / f
            v = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)], axis=-1)
        return normalize(v)

    # ---- world helpers
    def world_to_pixel(self, dirs: np.ndarray):
        """world unit directions (...,3) -> source pixel (x, y, valid)."""
        cam = dirs @ self.R  # == (R^T d) for row vectors
        if self.has_viewpoint:
            cam = self._apply_viewpoint_inverse(cam)
        return self.ray_to_pixel(cam)

    def pixel_to_world(self, x, y) -> np.ndarray:
        cam = self.pixel_to_ray(x, y)
        if self.has_viewpoint:
            cam = self._apply_viewpoint_forward(cam)
        return cam @ self.R.T

    # Viewpoint correction: the scene is assumed to be a plane (e.g. the floor for
    # nadir patching).  The plane has normal n (from vp_pan/vp_tilt, world frame)
    # at unit distance from the *reference* viewpoint.  This camera sits at offset
    # T (vp_x, vp_y, vp_z) from that viewpoint.  A ray from this camera hits the
    # plane at P; the reference viewpoint sees P along direction P.
    def _plane(self):
        n = rotation_matrix(self.vp_pan, self.vp_tilt, 0.0) @ np.array([0.0, 0.0, 1.0])
        return n

    def _apply_viewpoint_forward(self, cam_rays):
        # camera ray (camera frame) -> direction seen from reference viewpoint (camera frame)
        n = self.R.T @ self._plane()
        T = self.R.T @ np.array([self.vp_x, self.vp_y, self.vp_z])
        denom = cam_rays @ n
        dist = 1.0 - T @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = dist / denom
        P = T + cam_rays * s[..., None]
        P = np.where((s > 0)[..., None], P, cam_rays)
        return normalize(P)

    def _apply_viewpoint_inverse(self, ref_rays):
        # direction from reference viewpoint (camera frame) -> ray from this camera
        n = self.R.T @ self._plane()
        T = self.R.T @ np.array([self.vp_x, self.vp_y, self.vp_z])
        denom = ref_rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = 1.0 / denom
        P = ref_rays * s[..., None]
        out = normalize(P - T)
        return np.where((s > 0)[..., None], out, ref_rays)


def lens_r(theta, k):
    if k > 1e-9:
        return np.sin(k * theta) / k
    if k < -1e-9:
        kk = -k
        return np.tan(kk * theta) / kk
    return theta


def lens_r_valid(theta, k):
    """Like lens_r but also returns where the mapping is valid/monotonic."""
    if k > 1e-9:
        lim = math.pi / (2.0 * k)
        valid = theta < min(lim, math.pi)
        return np.sin(k * theta) / k, valid
    if k < -1e-9:
        kk = -k
        lim = math.pi / (2.0 * kk)
        valid = theta < lim * 0.999
        return np.tan(np.minimum(kk * theta, math.pi / 2 * 0.999)) / kk, valid
    return theta, theta < math.pi * 0.999


def lens_theta(r, k):
    if k > 1e-9:
        return np.arcsin(np.clip(k * r, -1.0, 1.0)) / k
    if k < -1e-9:
        kk = -k
        return np.arctan(kk * r) / kk
    return r


def _invert_radial(rd, a, b, c):
    """Solve rd = ru*(a ru^3 + b ru^2 + c ru + d0) for ru (Newton, vectorised)."""
    d0 = 1.0 - (a + b + c)
    ru = rd / max(d0, 1e-6)
    for _ in range(12):
        p = ((a * ru + b) * ru + c) * ru + d0
        fval = ru * p - rd
        dp = ((4 * a * ru + 3 * b) * ru + 2 * c) * ru + d0
        dp = np.where(np.abs(dp) < 1e-9, 1e-9, dp)
        ru = ru - fval / dp
    return ru


# --------------------------------------------------------------------------- output projections
#
# Every projection maps a world direction to plane coordinates (u, v) with v up,
# scaled so the derivative at the centre is 1 (1 unit == 1 radian at the centre),
# and back.  ``inverse`` returns (dirs, valid).

PROJECTIONS: dict[str, "Projection"] = {}


class Projection:
    name = ""
    label = ""
    max_hfov = 360.0
    max_vfov = 180.0
    params: tuple = ()  # names of extra parameters

    def __init__(self, **params):
        self.p = {k: float(v) for k, v in params.items()}

    def forward(self, d):  # (...,3) -> u, v, valid
        raise NotImplementedError

    def inverse(self, u, v):  # -> dirs (...,3), valid
        raise NotImplementedError

    # extents of the canvas for a given field of view (radians)
    def u_of_hfov(self, half_h):
        u, _, _ = self.forward(_dir_from_lonlat(np.float64(half_h), np.float64(0.0)))
        return float(u)

    def v_of_vfov(self, half_v):
        _, v, _ = self.forward(_dir_from_lonlat(np.float64(0.0), np.float64(half_v)))
        return float(v)

    @staticmethod
    def register(cls):
        PROJECTIONS[cls.name] = cls
        return cls


def _lonlat(d):
    X, Y, Z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(X, Z)
    lat = np.arctan2(Y, np.hypot(X, Z))
    return lon, lat


def _dir_from_lonlat(lon, lat):
    cl = np.cos(lat)
    return np.stack([np.sin(lon) * cl, np.sin(lat), np.cos(lon) * cl], axis=-1)


def _true(u):
    return np.ones(np.shape(u), dtype=bool)


@Projection.register
class Equirectangular(Projection):
    name, label = "equirectangular", "Equirectangular"

    def forward(self, d):
        lon, lat = _lonlat(d)
        return lon, lat, _true(lon)

    def inverse(self, u, v):
        valid = (np.abs(v) <= math.pi / 2 + 1e-9) & (np.abs(u) <= math.pi + 1e-9)
        return _dir_from_lonlat(u, np.clip(v, -math.pi / 2, math.pi / 2)), valid


@Projection.register
class Cylindrical(Projection):
    name, label, max_vfov = "cylindrical", "Cylindrical", 179.0

    def forward(self, d):
        lon, lat = _lonlat(d)
        return lon, np.tan(lat), np.abs(lat) < math.pi / 2 - 1e-6

    def inverse(self, u, v):
        return _dir_from_lonlat(u, np.arctan(v)), np.abs(u) <= math.pi + 1e-9


@Projection.register
class Mercator(Projection):
    name, label, max_vfov = "mercator", "Mercator", 179.0

    def forward(self, d):
        lon, lat = _lonlat(d)
        return lon, np.arcsinh(np.tan(np.clip(lat, -1.55, 1.55))), np.abs(lat) < 1.55

    def inverse(self, u, v):
        return _dir_from_lonlat(u, np.arctan(np.sinh(v))), np.abs(u) <= math.pi + 1e-9


@Projection.register
class Rectilinear(Projection):
    name, label, max_hfov, max_vfov = "rectilinear", "Rectilinear", 179.0, 179.0

    def forward(self, d):
        Z = d[..., 2]
        valid = Z > 1e-6
        Zs = np.where(valid, Z, 1e-6)
        return d[..., 0] / Zs, d[..., 1] / Zs, valid

    def inverse(self, u, v):
        return normalize(np.stack([u, v, np.ones_like(u)], axis=-1)), _true(u)

    def u_of_hfov(self, half_h):
        return math.tan(half_h)

    def v_of_vfov(self, half_v):
        return math.tan(half_v)


class _Azimuthal(Projection):
    """Radially symmetric azimuthal projections r = g(theta)."""

    max_hfov = 360.0
    theta_max = math.pi

    def g(self, th):
        raise NotImplementedError

    def ginv(self, r):
        raise NotImplementedError

    def forward(self, d):
        X, Y, Z = d[..., 0], d[..., 1], d[..., 2]
        rho = np.hypot(X, Y)
        th = np.arctan2(rho, Z)
        r = self.g(th)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(rho > 1e-12, r / np.maximum(rho, 1e-300), 1.0)
        return X * s, Y * s, th < self.theta_max - 1e-6

    def inverse(self, u, v):
        r = np.hypot(u, v)
        th, valid = self.ginv(r)
        st = np.sin(th)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(r > 1e-12, st / np.maximum(r, 1e-300), 1.0)
        return np.stack([u * s, v * s, np.cos(th)], axis=-1), valid

    def u_of_hfov(self, half_h):
        return float(self.g(min(half_h, self.theta_max - 1e-6)))

    v_of_vfov = u_of_hfov


@Projection.register
class Fisheye(_Azimuthal):
    name, label = "fisheye", "Fisheye (equidistant)"

    def g(self, th):
        return th

    def ginv(self, r):
        return r, r <= math.pi


@Projection.register
class Stereographic(_Azimuthal):
    name, label = "stereographic", "Stereographic"
    theta_max = math.pi * 0.999

    def g(self, th):
        return 2.0 * np.tan(th / 2.0)

    def ginv(self, r):
        return 2.0 * np.arctan(r / 2.0), _true(r)


@Projection.register
class Equisolid(_Azimuthal):
    name, label = "equisolid", "Equisolid (mirror ball)"

    def g(self, th):
        return 2.0 * np.sin(th / 2.0)

    def ginv(self, r):
        return 2.0 * np.arcsin(np.clip(r / 2.0, -1, 1)), r <= 2.0


@Projection.register
class Orthographic(_Azimuthal):
    name, label, max_hfov, max_vfov = "orthographic", "Orthographic", 180.0, 180.0
    theta_max = math.pi / 2

    def g(self, th):
        return np.sin(th)

    def ginv(self, r):
        return np.arcsin(np.clip(r, -1, 1)), r <= 1.0


@Projection.register
class Panini(Projection):
    """General Panini.  d=1 is the classic Panini/vedutismo, d=0 rectilinear.

    ``vcompress`` (0..1) blends the vertical axis from straight (tan) towards
    the equirectangular-like squeeze, keeping verticals straight.
    """

    name, label, max_hfov, max_vfov = "panini", "Panini / Vedutismo", 320.0, 160.0
    params = ("d", "vcompress")

    def __init__(self, d=1.0, vcompress=0.0, **kw):
        super().__init__(d=d, vcompress=vcompress, **kw)

    def _vfun(self, lat):
        c = self.p["vcompress"]
        return (1 - c) * np.tan(lat) + c * lat

    def _vinv(self, y):
        c = self.p["vcompress"]
        if c <= 0:
            return np.arctan(y)
        lat = np.arctan(y) if c < 1 else y
        for _ in range(8):
            f = self._vfun(lat) - y
            df = (1 - c) / np.cos(lat) ** 2 + c
            lat = np.clip(lat - f / df, -1.55, 1.55)
        return lat

    def forward(self, d):
        dd = self.p["d"]
        lon, lat = _lonlat(d)
        S = (dd + 1.0) / (dd + np.cos(lon))
        valid = (dd + np.cos(lon)) > 1e-6
        # straight verticals: y uses tan(lat) scaled by S (cylindrical Panini)
        return S * np.sin(lon), S * self._vfun(lat), valid & (np.abs(lat) < 1.55)

    def inverse(self, u, v):
        dd = self.p["d"]
        k = (u * u) / (dd + 1.0) ** 2
        dscr = k * k * dd * dd - (k + 1.0) * (k * dd * dd - 1.0)
        valid = dscr >= 0
        cl = (-k * dd + np.sqrt(np.maximum(dscr, 0))) / (k + 1.0)
        lon = np.sign(u) * np.arccos(np.clip(cl, -1, 1))
        S = (dd + 1.0) / (dd + np.cos(lon))
        lat = self._vinv(v / S)
        return _dir_from_lonlat(lon, lat), valid

    def u_of_hfov(self, half_h):
        dd = self.p["d"]
        return float((dd + 1.0) / (dd + math.cos(half_h)) * math.sin(half_h))

    def v_of_vfov(self, half_v):
        return float(self._vfun(np.float64(half_v)))


@Projection.register
class Sinusoidal(Projection):
    name, label = "sinusoidal", "Sinusoidal"

    def forward(self, d):
        lon, lat = _lonlat(d)
        return lon * np.cos(lat), lat, _true(lon)

    def inverse(self, u, v):
        cl = np.cos(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            lon = np.where(cl > 1e-9, u / np.maximum(cl, 1e-9), 0.0)
        return _dir_from_lonlat(lon, v), (np.abs(lon) <= math.pi) & (np.abs(v) <= math.pi / 2)

    def u_of_hfov(self, half_h):
        return half_h


@Projection.register
class Architectural(Projection):
    """Equirectangular below the horizon, cylindrical-like (Miller) above: keeps
    verticals straight while giving buildings a natural look."""

    name, label, max_vfov = "architectural", "Architectural", 179.0

    def forward(self, d):
        lon, lat = _lonlat(d)
        lat = np.clip(lat, -1.55, 1.55)
        v = np.where(lat > 0, 1.25 * np.arcsinh(np.tan(0.8 * lat)), lat)
        return lon, v, _true(lon)

    def inverse(self, u, v):
        lat = np.where(v > 0, np.arctan(np.sinh(v / 1.25)) / 0.8, v)
        return _dir_from_lonlat(u, lat), (np.abs(u) <= math.pi) & (lat <= math.pi / 2) & (lat >= -math.pi / 2)


class _Transverse(Projection):
    base: type = Equirectangular

    def __init__(self, **kw):
        super().__init__(**kw)
        self._b = self.base(**kw)

    def forward(self, d):
        d2 = np.stack([-d[..., 1], d[..., 0], d[..., 2]], axis=-1)
        u2, v2, valid = self._b.forward(d2)
        return v2, -u2, valid

    def inverse(self, u, v):
        d2, valid = self._b.inverse(-v, u)
        d = np.stack([d2[..., 1], -d2[..., 0], d2[..., 2]], axis=-1)
        return d, valid

    def u_of_hfov(self, half_h):
        return self._b.v_of_vfov(half_h)

    def v_of_vfov(self, half_v):
        return self._b.u_of_hfov(half_v)


@Projection.register
class TransverseEquirectangular(_Transverse):
    name, label, max_hfov, max_vfov = "transverse_equirectangular", "Transverse equirectangular", 180.0, 360.0
    base = Equirectangular


@Projection.register
class TransverseCylindrical(_Transverse):
    name, label, max_hfov, max_vfov = "transverse_cylindrical", "Transverse cylindrical", 179.0, 360.0
    base = Cylindrical


@Projection.register
class TransverseMercator(_Transverse):
    name, label, max_hfov, max_vfov = "transverse_mercator", "Transverse Mercator", 179.0, 360.0
    base = Mercator


@Projection.register
class TransversePanini(_Transverse):
    name, label, max_hfov, max_vfov = "transverse_panini", "Transverse Panini / Vedutismo", 160.0, 320.0
    base = Panini
    params = Panini.params

    def __init__(self, d=1.0, vcompress=0.0, **kw):
        super().__init__(d=d, vcompress=vcompress, **kw)


def make_projection(name: str, **params) -> Projection:
    cls = PROJECTIONS[name]
    allowed = set(cls.params)
    return cls(**{k: v for k, v in params.items() if k in allowed})
