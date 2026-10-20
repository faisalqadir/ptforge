"""End-to-end engine tests on a small synthetic shoot (ground truth known)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from panoforge import geometry as G  # noqa: E402
from panoforge.align import align_images, create_project  # noqa: E402
from panoforge.features import kabsch  # noqa: E402
from panoforge.project import Project  # noqa: E402


@pytest.fixture(scope="module")
def shoot(tmp_path_factory):
    d = tmp_path_factory.mktemp("rect_row")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "synth.py"), str(d / "imgs"), "--preset", "rect_row",
                    "--world-width", "4096", "--seed", "3"], check=True, capture_output=True)
    truth = json.load(open(d / "imgs" / "truth.json"))
    files = [str(d / "imgs" / t["file"]) for t in truth["images"]]
    p = create_project(files)
    p.optimizer.distortion = "heavy_shift"
    rep = align_images(p)
    p.save(str(d / "proj.pfproj"))
    return d, truth, p, rep


def test_alignment_recovers_truth(shoot):
    _, truth, p, rep = shoot
    assert rep.connected == len(truth["images"])
    assert rep.mean_error < 1.5
    assert abs(p.lenses[0].hfov - truth["lens"]["hfov"]) < 0.6
    Rt = [G.rotation_matrix(t["yaw"], t["pitch"], t["roll"]) for t in truth["images"]]
    Re = [G.rotation_matrix(im.yaw, im.pitch, im.roll) for im in p.images]
    Q = kabsch(np.concatenate(Rt, 1).T, np.concatenate(Re, 1).T)
    worst = max(math.degrees(math.acos(np.clip((np.trace((Q @ a).T @ b) - 1) / 2, -1, 1))) for a, b in zip(Rt, Re))
    # A partial single row has no loop closure: FoV and yaw spacing are correlated, so a
    # small FoV error grows linearly towards the ends (~0.1 deg centre, <1 deg at 200 deg span
    # on this deliberately low-resolution world).  360 degree shoots are far tighter.
    assert worst < 1.0


def test_stitch_and_formats(shoot, tmp_path):
    from panoforge.psd import read_psd_summary
    from panoforge.stitcher import stitch

    d, truth, p, _ = shoot
    r = stitch(p, str(tmp_path / "pano.jpg"), width=1600)
    assert os.path.exists(r.path) and r.width == 1600
    assert r.alpha.mean() > 0.5
    r2 = stitch(p.copy(), str(tmp_path / "pano.psd"), width=800, layers="layers_and_blended")
    info = read_psd_summary(r2.path)
    assert len(info["layers"]) == len(truth["images"]) + 1
    r3 = stitch(p.copy(), str(tmp_path / "pano.tif"), width=800)
    assert os.path.exists(r3.path)


def test_pto_roundtrip(shoot, tmp_path):
    from panoforge.optimizer import cp_errors
    from panoforge.ptgui_format import export_pto

    _, _, p, _ = shoot
    export_pto(p, str(tmp_path / "x.pto"))
    q = Project.load(str(tmp_path / "x.pto"))
    a = cp_errors(p.copy())
    b = cp_errors(q)
    assert np.allclose(a, b, atol=1e-4)


def test_little_planet_centre_is_nadir():
    from panoforge.output import little_planet

    eq = np.zeros((256, 512, 3), np.float32)
    eq[-20:] = 1.0  # bright floor
    lp = little_planet(eq, width=200, fov=300)
    assert lp[100, 100].mean() > 0.9  # the nadir lands in the middle
