"""PanoForge command line.

    panoforge new IMG... -o pano.pfproj        create + align (+ stitch with --stitch)
    panoforge align PROJECT                     control points + optimise + exposure
    panoforge optimize PROJECT                  re-run the optimiser
    panoforge stitch PROJECT [-o OUT]           create the panorama
    panoforge info PROJECT                      summary and control point statistics
    panoforge batch FOLDER -o DIR [--template]  batch builder + optional stitching
    panoforge convert PANO --to cube|littleplanet|PROJ ...
    panoforge export PROJECT out.pto            Hugin / Panorama Tools project
    panoforge gui [PROJECT]                     start the graphical editor
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time


class _Progress:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.t0 = time.time()
        self.last = None

    def __call__(self, stage, i, n, **kw):
        if self.quiet:
            return
        if stage == "warning":
            print(f"  ! {kw.get('message', '')}", file=sys.stderr)
            return
        if stage != self.last:
            print(f"  [{time.time() - self.t0:6.1f}s] {stage}", flush=True)
            self.last = stage


def _expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            from .imageio import is_image_file

            out += sorted(os.path.join(p, f) for f in os.listdir(p) if is_image_file(f))
        else:
            g = sorted(glob.glob(p))
            out += g if g else [p]
    return out


def _apply_stitch_args(p, a):
    o = p.output
    for key in ("format", "quality", "bit_depth", "blender", "seams", "interpolator", "layers"):
        v = getattr(a, key, None)
        if v is not None:
            setattr(o, key, v)
    if getattr(a, "feather", None) is not None:
        o.feather = a.feather
    if getattr(a, "hdr", None):
        p.exposure.hdr_mode = a.hdr != "off"
        if a.hdr != "off":
            p.exposure.hdr_method = a.hdr
    if getattr(a, "projection", None):
        p.panorama.projection = a.projection
    if getattr(a, "hfov", None):
        p.panorama.hfov = a.hfov
    if getattr(a, "vfov", None):
        p.panorama.vfov = a.vfov
    if getattr(a, "size_percent", None):
        o.size_percent = a.size_percent
    if getattr(a, "megapixels", None):
        o.max_megapixels = a.megapixels


def _print_align(rep):
    print(f"  {rep.connected}/{rep.n_images} images connected, {rep.n_cps} control points, "
          f"mean {rep.mean_error:.2f}px, max {rep.max_error:.2f}px")
    if rep.unconnected:
        print(f"  ! not connected: {rep.unconnected} (add control points or check overlap)")


def cmd_new(a):
    from .align import align_images, create_project

    files = _expand(a.images)
    if len(files) < 1:
        sys.exit("no images")
    p = create_project(files, lens_type=a.lens, hfov=a.lens_hfov, fisheye_factor=a.fisheye_factor)
    if a.template:
        from .project import Project
        from .templates import apply_template

        apply_template(p, Project.load(a.template))
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(files[0])), os.path.splitext(os.path.basename(files[0]))[0] + " Panorama.pfproj")
    p.save(out)
    ln = p.lenses[0]
    print(f"project {out}: {len(p.images)} images, lens {ln.lens_type} hfov {ln.hfov:.1f}"
          + (f", {len(p.link_groups())} bracket stacks" if p.link_groups() else ""))
    if not a.no_align:
        rep = align_images(p, prealigned=bool(a.template), progress=_Progress(a.quiet))
        _print_align(rep)
        p.save(out)
    if a.stitch:
        a.project = out
        cmd_stitch(a, p)


def cmd_align(a):
    from .align import align_images
    from .project import Project

    p = Project.load(a.project)
    rep = align_images(p, generate=not a.keep_cps, prealigned=a.prealigned, max_cps_per_pair=a.max_cps or 30,
                       progress=_Progress(a.quiet))
    _print_align(rep)
    p.save(a.output or a.project)


def cmd_optimize(a):
    from .optimizer import optimize
    from .project import Project

    p = Project.load(a.project)
    if a.distortion:
        p.optimizer.distortion = a.distortion
    r = optimize(p)
    print(f"  {r.n_cps} control points, {r.n_vars} variables: mean {r.mean_error:.2f}px max {r.max_error:.2f}px ({r.message})")
    if a.photometric:
        from .photometric import optimize_photometric

        print("  photometric:", optimize_photometric(p))
    p.save(a.output or a.project)


def cmd_stitch(a, p=None):
    from .project import Project
    from .stitcher import stitch

    p = p or Project.load(a.project)
    _apply_stitch_args(p, a)
    r = stitch(p, a.out, width=a.width, progress=_Progress(a.quiet))
    print(f"  {r.width}x{r.height} -> {r.path} ({r.seconds:.1f}s)")
    for f in r.extra_files:
        print(f"  + {f}")
    if getattr(a, "cube", False):
        from .output import save_cube_faces

        base = os.path.splitext(r.path)[0]
        for f in save_cube_faces(r.image, base, ".jpg", quality=p.output.quality):
            print(f"  + {f}")


def cmd_info(a):
    import numpy as np

    from .optimizer import cp_errors
    from .project import Project

    p = Project.load(a.project)
    print(f"{a.project}")
    for k, ln in enumerate(p.lenses):
        print(f"  lens {k}: {ln.lens_type} hfov {ln.hfov:.3f} k {ln.fisheye_factor:.3f} a {ln.a:.5f} b {ln.b:.5f} c {ln.c:.5f} "
              f"d {ln.d:.2f} e {ln.e:.2f} crop {ln.crop.mode} vig {[round(v, 3) for v in ln.vig]}")
    for i, im in enumerate(p.images):
        print(f"  [{i:3d}] {'+' if im.include else '-'} {os.path.basename(im.path):30s} y {im.yaw:8.2f} p {im.pitch:7.2f} r {im.roll:7.2f} "
              f"ev {im.ev:+.2f} wb {im.wb_r:.3f}/{im.wb_b:.3f}" + (f" link {im.link_group}" if im.link_group >= 0 else ""))
    errs = cp_errors(p)
    if len(errs):
        print(f"  control points: {len(errs)}  mean {np.nanmean(errs):.2f}px  max {np.nanmax(errs):.2f}px")
    P = p.panorama
    print(f"  panorama: {P.projection} {P.hfov:.1f} x {P.vfov:.1f} deg, crop {P.crop}")


def cmd_batch(a):
    from .project import Project
    from .stitcher import stitch
    from .templates import build_batch

    tpl = Project.load(a.template) if a.template else None
    projects = build_batch(a.folder, a.output, tpl, a.gap, a.count)
    print(f"{len(projects)} projects")
    for path in projects:
        print(f"* {path}")
        if a.align or a.stitch:
            from .align import align_images

            p = Project.load(path)
            rep = align_images(p, prealigned=tpl is not None, progress=_Progress(a.quiet))
            _print_align(rep)
            p.save(path)
            if a.stitch and (not a.only_if_ok or not rep.unconnected):
                r = stitch(p, progress=_Progress(a.quiet))
                print(f"  -> {r.path}")


def cmd_convert(a):
    import numpy as np

    from .imageio import load_image, save_image
    from .output import little_planet, reproject, save_cube_faces

    img = load_image(a.input).rgb
    base = os.path.splitext(a.output or a.input)[0]
    if a.to == "cube":
        for f in save_cube_faces(img, base, os.path.splitext(a.output or ".jpg")[1] or ".jpg", strip=a.strip):
            print(f)
    elif a.to == "littleplanet":
        out = little_planet(img, a.width or 2000, a.fov or 300.0, a.yaw or 0.0)
        save_image(a.output or base + "_planet.jpg", np.clip(out, 0, 1))
    else:
        out = reproject(img, "equirectangular", 360.0, a.to, a.fov or 120.0, a.vfov or 90.0, a.width or 2000,
                        yaw=a.yaw or 0.0, pitch=a.pitch or 0.0)
        save_image(a.output or base + f"_{a.to}.jpg", np.clip(out, 0, 1))


def cmd_export(a):
    from .project import Project
    from .ptgui_format import export_pto

    p = Project.load(a.project)
    export_pto(p, a.output)
    print(a.output)


def cmd_gui(a):
    from .server import run_server

    run_server(a.project, host=a.host, port=a.port, open_browser=not a.no_browser)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="panoforge", description="PanoForge panorama stitcher")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def stitch_opts(s):
        s.add_argument("--out", help="output file (extension selects the format)")
        s.add_argument("--width", type=int)
        s.add_argument("--format", choices=["jpeg", "png", "tiff", "psd", "exr", "hdr", "webp"])
        s.add_argument("--quality", type=int)
        s.add_argument("--bit-depth", dest="bit_depth", type=int, choices=[8, 16, 32])
        s.add_argument("--blender", choices=["multiband", "feather", "none"])
        s.add_argument("--seams", choices=["graphcut", "voronoi", "none"])
        s.add_argument("--feather", type=float)
        s.add_argument("--interpolator", choices=["nearest", "bilinear", "bicubic", "lanczos"])
        s.add_argument("--layers", choices=["blended", "layers", "layers_and_blended"])
        s.add_argument("--hdr", choices=["off", "fusion", "true_hdr"])
        s.add_argument("--projection")
        s.add_argument("--hfov", type=float)
        s.add_argument("--vfov", type=float)
        s.add_argument("--size-percent", dest="size_percent", type=float)
        s.add_argument("--megapixels", type=float)
        s.add_argument("--cube", action="store_true", help="also write cube faces")

    s = sub.add_parser("new", help="create a project from images and align it")
    s.add_argument("images", nargs="+")
    s.add_argument("-o", "--output")
    s.add_argument("--lens", choices=["rectilinear", "fisheye", "circular_fisheye", "cylindrical", "equirectangular"])
    s.add_argument("--lens-hfov", dest="lens_hfov", type=float)
    s.add_argument("--fisheye-factor", dest="fisheye_factor", type=float, default=0.5)
    s.add_argument("--template")
    s.add_argument("--no-align", action="store_true")
    s.add_argument("--stitch", action="store_true")
    stitch_opts(s)
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("align", help="generate control points and optimise")
    s.add_argument("project")
    s.add_argument("-o", "--output")
    s.add_argument("--keep-cps", action="store_true")
    s.add_argument("--prealigned", action="store_true")
    s.add_argument("--max-cps", dest="max_cps", type=int)
    s.set_defaults(func=cmd_align)

    s = sub.add_parser("optimize", help="run the optimiser")
    s.add_argument("project")
    s.add_argument("-o", "--output")
    s.add_argument("--distortion", choices=["none", "medium", "heavy", "heavy_shift"])
    s.add_argument("--photometric", action="store_true")
    s.set_defaults(func=cmd_optimize)

    s = sub.add_parser("stitch", help="create the panorama")
    s.add_argument("project")
    stitch_opts(s)
    s.set_defaults(func=cmd_stitch)

    s = sub.add_parser("info", help="project summary")
    s.add_argument("project")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("batch", help="batch builder: one project per panorama found in a folder")
    s.add_argument("folder")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--template")
    s.add_argument("--gap", type=float, default=10.0, help="max seconds between shots of one panorama")
    s.add_argument("--count", type=int, help="fixed number of images per panorama")
    s.add_argument("--align", action="store_true")
    s.add_argument("--stitch", action="store_true")
    s.add_argument("--only-if-ok", dest="only_if_ok", action="store_true")
    s.set_defaults(func=cmd_batch)

    s = sub.add_parser("convert", help="convert a stitched equirectangular panorama")
    s.add_argument("input")
    s.add_argument("--to", required=True, help="cube | littleplanet | <projection name>")
    s.add_argument("-o", "--output")
    s.add_argument("--width", type=int)
    s.add_argument("--fov", type=float)
    s.add_argument("--vfov", type=float)
    s.add_argument("--yaw", type=float)
    s.add_argument("--pitch", type=float)
    s.add_argument("--strip", action="store_true")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("export", help="export a Hugin/Panorama Tools .pto project")
    s.add_argument("project")
    s.add_argument("output")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("gui", help="start the graphical editor")
    s.add_argument("project", nargs="?")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", dest="no_browser", action="store_true")
    s.set_defaults(func=cmd_gui)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
