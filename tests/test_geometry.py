import math

import numpy as np
import pytest

from panoforge import geometry as G


@pytest.mark.parametrize("ypr", [(0, 0, 0), (30, 10, -5), (-170, -60, 80), (90, 45, 179), (12, 89.0, 3)])
def test_ypr_roundtrip(ypr):
    R = G.rotation_matrix(*ypr)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    y, p, r = G.matrix_to_ypr(R)
    assert np.allclose(G.rotation_matrix(y, p, r), R, atol=1e-9)


def test_rotation_semantics():
    fwd = np.array([0, 0, 1.0])
    assert np.allclose(G.rotation_matrix(90, 0, 0) @ fwd, [1, 0, 0], atol=1e-12)  # yaw right
    assert np.allclose(G.rotation_matrix(0, 90, 0) @ fwd, [0, 1, 0], atol=1e-12)  # pitch up


LENSES = [
    dict(lens_type="rectilinear", hfov=70, a=0.01, b=-0.03, c=0.005, d=12, e=-7, g=0.002, t=-0.001),
    dict(lens_type="fisheye", hfov=150, fisheye_factor=0.0, b=-0.01, d=3),
    dict(lens_type="fisheye", hfov=180, fisheye_factor=0.5),
    dict(lens_type="circular_fisheye", hfov=190, fisheye_factor=0.25, a=0.002, e=5),
    dict(lens_type="fisheye", hfov=140, fisheye_factor=-0.5),
    dict(lens_type="cylindrical", hfov=120),
    dict(lens_type="equirectangular", hfov=360),
]


@pytest.mark.parametrize("kw", LENSES)
def test_pixel_ray_roundtrip(kw):
    h = 600 if kw["lens_type"] == "equirectangular" else 800
    cam = G.Camera(width=1200, height=h, yaw=20, pitch=-10, roll=5, **kw)
    ys, xs = np.mgrid[5 : h - 5 : 37, 5:1195:41].astype(np.float64)
    if kw["lens_type"] == "circular_fisheye":
        keep = np.hypot(xs - 600, ys - 400) < 380
        xs, ys = xs[keep], ys[keep]
    d = cam.pixel_to_world(xs, ys)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1)
    x2, y2, valid = cam.world_to_pixel(d)
    assert valid.all()
    assert np.max(np.abs(x2 - xs)) < 1e-6 and np.max(np.abs(y2 - ys)) < 1e-6


def test_rectilinear_fov():
    cam = G.Camera(width=1000, height=600, hfov=90)
    assert math.isclose(cam.focal_px, 500.0, rel_tol=1e-12)
    d = cam.pixel_to_world(np.array([-0.5]), np.array([299.5]))
    assert np.allclose(d[0], G.normalize(np.array([-1.0, 0, 1])))


@pytest.mark.parametrize("name", sorted(G.PROJECTIONS))
def test_projection_roundtrip(name):
    P = G.make_projection(name)
    lon = np.radians(np.linspace(-70, 70, 29))
    lat = np.radians(np.linspace(-60, 60, 25))
    L, B = np.meshgrid(lon, lat)
    d = G._dir_from_lonlat(L, B)
    u, v, valid = P.forward(d)
    d2, valid2 = P.inverse(u, v)
    ok = valid & valid2
    assert ok.mean() > 0.5, name
    err = np.linalg.norm(d2[ok] - d[ok], axis=-1)
    assert err.max() < 1e-6, (name, err.max())
    # unit scale at centre
    eps = 1e-5
    u1, v1, _ = P.forward(G._dir_from_lonlat(np.array([eps]), np.array([0.0])))
    u2, v2, _ = P.forward(G._dir_from_lonlat(np.array([0.0]), np.array([eps])))
    assert abs(u1[0] / eps - 1) < 1e-3 and abs(v2[0] / eps - 1) < 1e-3, name


def test_viewpoint_roundtrip():
    cam = G.Camera(width=800, height=600, hfov=80, pitch=-60, vp_x=0.05, vp_y=0.02, vp_z=-0.03, vp_tilt=-90)
    ys, xs = np.mgrid[10:590:50, 10:790:50].astype(np.float64)
    d = cam.pixel_to_world(xs, ys)
    x2, y2, valid = cam.world_to_pixel(d)
    assert np.max(np.abs(x2 - xs)) < 1e-6
