"""Align a synthetic dataset and compare with its ground truth."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from panoforge import geometry as G  # noqa: E402
from panoforge.align import align_images, create_project  # noqa: E402


def rel_angle_err(p, truth):
    """Max rotation error (deg) between estimated and true poses after removing the global gauge."""
    Rt = [G.rotation_matrix(t["yaw"], t["pitch"], t["roll"]) for t in truth["images"]]
    Re = [G.rotation_matrix(im.yaw, im.pitch, im.roll) for im in p.images]
    # best global rotation: Re_i ~ Q Rt_i
    from panoforge.features import kabsch

    A = np.concatenate([R for R in Rt], axis=1).T  # columns as vectors
    B = np.concatenate([R for R in Re], axis=1).T
    Q = kabsch(A, B)
    errs = []
    for a, b in zip(Rt, Re):
        D = (Q @ a).T @ b
        errs.append(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))))
    return max(errs)


def main(folder, lens_type=None, photometric=False):
    truth = json.load(open(os.path.join(folder, "truth.json")))
    files = [os.path.join(folder, t["file"]) for t in truth["images"]]
    t0 = time.time()
    p = create_project(files, lens_type=lens_type, fisheye_factor=truth["lens"]["fisheye_factor"])
    print("initial hfov", [round(l.hfov, 2) for l in p.lenses], "groups", [im.link_group for im in p.images])
    if truth["lens"]["lens_type"] != "rectilinear":
        p.optimizer.distortion = "medium"
    rep = align_images(p, photometric=photometric)
    dt = time.time() - t0
    L = p.lenses[0]
    tl = truth["lens"]
    print(f"time {dt:.1f}s  cps {rep.n_cps}  connected {rep.connected}/{rep.n_images}  mean err {rep.mean_error:.3f}px max {rep.max_error:.3f}px")
    print(f"hfov {L.hfov:.3f} (true {tl['hfov']})  a {L.a:.4f}/{tl['a']} b {L.b:.4f}/{tl['b']} c {L.c:.4f}/{tl['c']}")
    print(f"pose error (max, deg): {rel_angle_err(p, truth):.4f}")
    print("projection", p.panorama.projection, "fov", round(p.panorama.hfov, 1), round(p.panorama.vfov, 1))
    p.save(os.path.join(folder, "aligned.pfproj"))
    return p


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or [None]))
