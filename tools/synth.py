"""Synthetic panorama shoot generator (ground truth for testing the engine).

Builds a procedural 360x180 world (linear radiance) and renders virtual camera
shots from it with known lens geometry, distortion, exposure, white balance and
vignetting.  Writes JPEGs with EXIF + a ground-truth JSON.

    python tools/synth.py OUTDIR --preset rect_row|fisheye|multirow|brackets
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from panoforge import geometry as G  # noqa: E402
from panoforge.imageio import linear_to_srgb  # noqa: E402


def make_world(width=8192, seed=1):
    rng = np.random.default_rng(seed)
    h = width // 2
    img = np.zeros((h, width, 3), np.float32)
    # smooth colourful base (multi-octave noise)
    for octave in range(1, 7):
        n = 2 ** octave * 4
        small = rng.random((max(2, n // 2), n, 3)).astype(np.float32)
        up = cv2.resize(small, (width, h), interpolation=cv2.INTER_CUBIC)
        img += up * (0.5 ** octave)
    img = (img - img.min()) / (img.max() - img.min())
    img = 0.15 + 0.5 * img
    # shapes give corners/blobs for feature detectors
    for _ in range(3500):
        col = tuple(float(c) for c in rng.random(3) * 1.4)
        x, y = int(rng.integers(0, width)), int(rng.integers(int(h * 0.04), int(h * 0.96)))
        s = int(rng.integers(6, 120))
        kind = rng.integers(0, 4)
        if kind == 0:
            cv2.circle(img, (x, y), s // 2, col, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.3, 1.5))), col, -1, cv2.LINE_AA)
        elif kind == 2:
            pts = (rng.random((5, 2)) * s + [x, y]).astype(np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-s, s)), y + int(rng.integers(-s, s))), col, int(rng.integers(1, 6)), cv2.LINE_AA)
    # fine texture
    img += (rng.random(img.shape).astype(np.float32) - 0.5) * 0.06
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    # vertical structures (meridians are vertical lines in the world)
    for k in range(12):
        x = int((k + 0.5) / 12 * width)
        cv2.line(img, (x, int(h * 0.2)), (x, int(h * 0.8)), (1.6, 1.6, 1.5), 5, cv2.LINE_AA)
    # sky gradient / ground tint so up and down differ
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img *= 0.7 + 0.6 * (1 - yy)
    return np.clip(img, 0.0, 3.0)


def vignette(cam: G.Camera, v):
    ys, xs = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32)
    r = np.hypot(xs + 0.5 - cam.width / 2, ys + 0.5 - cam.height / 2) / (math.hypot(cam.width, cam.height) / 2)
    r2 = r * r
    return 1 + v[0] * r2 + v[1] * r2 * r2 + v[2] * r2 ** 3


def render(world, cam: G.Camera, ev=0.0, wb=(1.0, 1.0), vig=(0, 0, 0), noise=0.004, rng=None):
    H, W = world.shape[:2]
    ys, xs = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    d = cam.pixel_to_world(xs, ys)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    mx = ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32)
    my = ((0.5 - lat / math.pi) * H - 0.5).astype(np.float32)
    out = cv2.remap(world, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    out = out * (2.0 ** (-ev))
    out[..., 0] *= wb[0]
    out[..., 2] *= wb[1]
    out *= vignette(cam, vig)[..., None]
    if rng is not None and noise:
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(linear_to_srgb(np.clip(out, 0, 1)), 0, 1)


def save_jpeg(path, rgb, focal_mm, focal35, exposure_time=1 / 125, fnumber=8.0, iso=100):
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    ex = Image.Exif()
    ex[0x010F] = "Synthetic"
    ex[0x0110] = "SynthCam"
    sub = {
        0x920A: (int(focal_mm * 100), 100),  # FocalLength
        0xA405: int(round(focal35)),  # FocalLengthIn35mmFilm
        0x829A: (1, int(round(1 / exposure_time))) if exposure_time < 1 else (int(exposure_time), 1),
        0x829D: (int(fnumber * 10), 10),
        0x8827: iso,
        0xA403: 1,  # manual WB
        0xA402: 1,  # manual exposure
    }
    ex_ifd = ex.get_ifd(0x8769)
    ex_ifd.update(sub)
    im.save(path, quality=93, exif=ex)


PRESETS = {
    # name: (lens_type, hfov, fisheye_factor, size, yaw/pitch list)
    "rect_row": dict(lens="rectilinear", hfov=62.0, k=0, size=(1500, 1000),
                     poses=[(y, 0, 0) for y in range(-100, 101, 40)], dist=(0.0, -0.045, 0.0), shift=(6.0, -4.0)),
    "multirow": dict(lens="rectilinear", hfov=70.0, k=0, size=(1200, 900),
                     poses=[(y, p, 0) for p in (-40, 5) for y in range(0, 360, 45)] + [(y, 50, 0) for y in range(0, 360, 60)] + [(0, 90, 0), (0, -90, 0)], dist=(0.0, -0.03, 0.0), shift=(0, 0)),
    "fisheye": dict(lens="fisheye", hfov=180.0, k=0.5, size=(1600, 1067),
                    poses=[(y, 0, 0) for y in (0, 90, 180, 270)] + [(45, 80, 0), (225, -80, 0)], dist=(0.0, 0.01, 0.0), shift=(0, 0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--preset", default="rect_row", choices=list(PRESETS) + ["brackets"])
    ap.add_argument("--world-width", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exposure-jitter", type=float, default=0.0, help="random EV jitter (auto-exposure simulation)")
    ap.add_argument("--vignetting", type=float, default=-0.3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    wpath = os.path.join(os.path.dirname(os.path.abspath(args.outdir)), f"_world_{args.world_width}_{args.seed}.npy")
    if os.path.exists(wpath):
        world = np.load(wpath)
    else:
        world = make_world(args.world_width, args.seed)
        np.save(wpath, world)

    preset = args.preset
    brackets = [0.0]
    if preset == "brackets":
        preset, brackets = "rect_row", [-2.0, 0.0, 2.0]
    P = PRESETS[preset]
    W, H = P["size"]
    truth = {"preset": args.preset, "images": [], "lens": {"lens_type": P["lens"], "hfov": P["hfov"], "fisheye_factor": P["k"],
             "a": P["dist"][0], "b": P["dist"][1], "c": P["dist"][2], "d": P["shift"][0], "e": P["shift"][1]},
             "vignetting": [args.vignetting, 0.0, 0.0]}
    crop = 1.5
    sensor_w = 43.2666 / crop * W / math.hypot(W, H)
    for n, (yaw, pitch, roll) in enumerate(P["poses"]):
        jr = rng.normal(0, 1.5, 3)
        pose = (yaw + jr[0], pitch + jr[1] * 0.5, roll + jr[2] * 0.7)
        cam = G.Camera(width=W, height=H, lens_type=P["lens"], hfov=P["hfov"], fisheye_factor=P["k"],
                       a=P["dist"][0], b=P["dist"][1], c=P["dist"][2], d=P["shift"][0], e=P["shift"][1],
                       yaw=pose[0], pitch=pose[1], roll=pose[2])
        focal_px = cam.focal_px
        if P["lens"] == "rectilinear":
            focal_mm = focal_px * sensor_w / W
        else:
            focal_mm = focal_px * sensor_w / W
        for b in brackets:
            ev = b + (rng.normal(0, args.exposure_jitter) if args.exposure_jitter else 0.0)
            wb = (1.0 + rng.normal(0, 0.03), 1.0 + rng.normal(0, 0.03)) if args.exposure_jitter else (1.0, 1.0)
            img = render(world, cam, ev=ev, wb=wb, vig=(args.vignetting, 0, 0), rng=rng)
            name = f"img_{n:02d}" + (f"_ev{b:+.0f}" if len(brackets) > 1 else "") + ".jpg"
            save_jpeg(os.path.join(args.outdir, name), img, focal_mm, focal_mm * crop, exposure_time=(1 / 125) * 2 ** (-b))
            truth["images"].append({"file": name, "yaw": pose[0], "pitch": pose[1], "roll": pose[2], "ev": ev, "wb": wb, "bracket": b})
    with open(os.path.join(args.outdir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"wrote {len(truth['images'])} images to {args.outdir}")


if __name__ == "__main__":
    main()
