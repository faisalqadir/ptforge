"""Interoperability with Panorama-Tools family projects.

* Hugin ``.pto`` (Panorama Tools script): import and export.
* PTGui 13 ``.pts`` (JSON, schema ``ptgui.com/schemas/project_v*``): import pending a
  sample file -- the schema is undocumented, so mapping it needs a real project.

Sign conventions of roll and lens shift are taken as identical to PanoForge's; they
still need confirming against a project saved by PTGui/Hugin.
"""
from __future__ import annotations

import math
import os
import re
import shlex

import numpy as np

from . import geometry as G
from .project import (CP_HORIZONTAL, CP_LINE, CP_NORMAL, CP_VERTICAL, ControlPoint, ImageEntry, Lens, Project)

# Hugin source image projections
PTO_LENS = {0: G.LENS_RECTILINEAR, 1: G.LENS_CYLINDRICAL, 2: G.LENS_CIRCULAR_FISHEYE, 3: G.LENS_FISHEYE,
            4: G.LENS_EQUIRECT, 8: G.LENS_FISHEYE, 10: G.LENS_FISHEYE, 20: G.LENS_FISHEYE, 21: G.LENS_FISHEYE}
PTO_LENS_K = {2: 0.0, 3: 0.0, 8: 1.0, 10: -0.5, 20: 0.713, 21: 0.5}
# Hugin panorama projections
PTO_PROJ = {0: "rectilinear", 1: "cylindrical", 2: "equirectangular", 3: "fisheye", 4: "stereographic", 5: "mercator",
            6: "transverse_mercator", 7: "sinusoidal", 12: "panini", 13: "architectural", 14: "orthographic",
            15: "equisolid", 19: "panini"}
PROJ_PTO = {"rectilinear": 0, "cylindrical": 1, "equirectangular": 2, "fisheye": 3, "stereographic": 4, "mercator": 5,
            "transverse_mercator": 6, "sinusoidal": 7, "panini": 12, "architectural": 13, "orthographic": 14, "equisolid": 15}
CP_TYPES = {0: CP_NORMAL, 1: CP_VERTICAL, 2: CP_HORIZONTAL}


def _lens_code(ln: Lens) -> int:
    if ln.lens_type == G.LENS_RECTILINEAR:
        return 0
    if ln.lens_type == G.LENS_CYLINDRICAL:
        return 1
    if ln.lens_type == G.LENS_EQUIRECT:
        return 4
    k = ln.fisheye_factor
    best = min(((abs(k - v), code) for code, v in PTO_LENS_K.items() if code not in (2, 3)), default=(1, 3))
    if abs(k) < 0.08:
        return 2 if ln.lens_type == G.LENS_CIRCULAR_FISHEYE else 3
    return best[1]


def _hugin_vig_scale(p: Project, idx: int) -> float:
    """Factor s with r_hugin = r_ours * s  (Hugin normalises by the half diagonal)."""
    from .photometric import vig_norm2

    im = p.images[idx]
    return math.sqrt(vig_norm2(p, idx)) / (math.hypot(im.width, im.height) / 2)


def export_pto(p: Project, path: str):
    lines = ["# hugin project file", "#hugin_ptoversion 2"]
    P = p.panorama
    from .remap import Canvas

    cv = Canvas(p, P.width or None)
    l, t, r, b = cv.crop_rect()
    lines.append(f'p f{PROJ_PTO.get(P.projection, 2)} w{cv.width} h{cv.height} v{P.hfov:.6g} E0 R0 S{l},{r},{t},{b} n"TIFF_m c:LZW r:CROP"')
    lines.append("m i0")
    lines.append("")
    first_of_lens: dict[int, int] = {}
    base = os.path.dirname(os.path.abspath(path))
    for i, im in enumerate(p.images):
        ln = p.lenses[im.lens]
        s = _hugin_vig_scale(p, i)
        v = [ln.vig[0] / s ** 2, ln.vig[1] / s ** 4, ln.vig[2] / s ** 6]
        if ln.crop_factor:
            lines.append(f"#-hugin  cropFactor={ln.crop_factor:.6g}")
        # exact PanoForge values PTO cannot express (ignored by other tools)
        lines.append(f"#-panoforge fisheye_factor={ln.fisheye_factor:.10g} lens_type={ln.lens_type} crop={ln.crop.mode}")
        fields = [f"w{im.width}", f"h{im.height}", f"f{_lens_code(ln)}"]
        linked = im.lens in first_of_lens
        ref = first_of_lens.get(im.lens, i)
        for key in ("hfov", "a", "b", "c", "d", "e", "g", "t"):
            tag = "v" if key == "hfov" else key
            if linked and key not in im.lens_overrides:
                fields.append(f"{tag}={ref}")
            else:
                fields.append(f"{tag}{float(p.lens_value(i, key)):.10g}")
        fields += [f"Ra0 Rb0 Rc0 Rd0 Re0 Eev{im.ev:.6g} Er{im.wb_r:.6g} Eb{im.wb_b:.6g}",
                   f"r{im.roll:.10g} p{im.pitch:.10g} y{im.yaw:.10g}",
                   f"TrX{im.vp_x:.6g} TrY{im.vp_y:.6g} TrZ{im.vp_z:.6g} Tpy{im.vp_pan:.6g} Tpp{im.vp_tilt:.6g} j0",
                   f"Va1 Vb{v[0]:.6g} Vc{v[1]:.6g} Vd{v[2]:.6g} Vx{ln.vig_cx:.6g} Vy{ln.vig_cy:.6g} Vm5"]
        if ln.crop.mode == "rect":
            fields.append(f"S{int(ln.crop.left)},{int(ln.crop.right or im.width)},{int(ln.crop.top)},{int(ln.crop.bottom or im.height)}")
        elif ln.crop.mode == "circle":
            cx, cy, rad = ln.crop.cx or im.width / 2, ln.crop.cy or im.height / 2, ln.crop.radius or min(im.width, im.height) / 2
            fields.append(f"S{int(round(cx - rad))},{int(round(cx + rad))},{int(round(cy - rad))},{int(round(cy + rad))}")
        try:
            fn = os.path.relpath(p.abspath(im.path), base)
        except ValueError:
            fn = p.abspath(im.path)
        fields.append(f'n"{fn}"')
        if not im.include:
            lines.append("#-hugin  enabled=false")
        lines.append("i " + " ".join(fields))
        first_of_lens.setdefault(im.lens, i)
    lines.append("")
    lines.append("# specify variables that should be optimized")
    lines.append("v")
    lines.append("")
    lines.append("# control points")
    line_ids: dict[int, int] = {}
    for cp in p.control_points:
        if cp.kind == CP_LINE:
            t = line_ids.setdefault(cp.line, 3 + len(line_ids))
        else:
            t = {CP_NORMAL: 0, CP_VERTICAL: 1, CP_HORIZONTAL: 2}[cp.kind]
        lines.append(f"c n{cp.i} N{cp.j} x{cp.x1:.6f} y{cp.y1:.6f} X{cp.x2:.6f} Y{cp.y2:.6f} t{t}")
    lines.append("")
    lines.append("#hugin_optimizeReferenceImage " + str(p.optimizer.anchor))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


_TOKEN = re.compile(r'([A-Za-z]+)(=?)("[^"]*"|[^\s]*)')


def _parse_fields(rest: str) -> dict:
    out = {}
    for m in _TOKEN.finditer(rest):
        key, eq, val = m.group(1), m.group(2), m.group(3)
        if val.startswith('"'):
            val = val[1:-1]
        out[key] = ("=", val) if eq else val
    return out


def import_pto(path: str) -> Project:
    p = Project()
    base = os.path.dirname(os.path.abspath(path))
    raw_images = []
    crop_factor = None
    disabled_next = False
    pf_extra: dict = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#-panoforge"):
                pf_extra = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                continue
            if line.startswith("#-hugin"):
                m = re.search(r"cropFactor=([0-9.]+)", line)
                if m:
                    crop_factor = float(m.group(1))
                if "enabled=false" in line:
                    disabled_next = True
                continue
            if not line or line.startswith("#"):
                continue
            kind, _, rest = line.partition(" ")
            fl = _parse_fields(rest)
            if kind == "p":
                P = p.panorama
                P.projection = PTO_PROJ.get(int(float(fl.get("f", 2))), "equirectangular")
                P.hfov = float(fl.get("v", 360))
                P.width = int(float(fl.get("w", 0)))
                h = int(float(fl.get("h", 0)))
                if P.width and h:
                    proj = G.make_projection(P.projection)
                    u_ext = 2 * proj.u_of_hfov(math.radians(P.hfov) / 2)
                    v_half = h / (P.width / u_ext) / 2
                    from .align import _invert_extent

                    P.vfov = _invert_extent(proj.v_of_vfov, v_half, proj.max_vfov)
                if "S" in fl and isinstance(fl["S"], str) and fl["S"]:
                    l, r, t, b = [float(x) for x in fl["S"].split(",")]
                    if P.width and h:
                        P.crop = [l / P.width, t / h, r / P.width, b / h]
            elif kind == "i":
                raw_images.append((fl, crop_factor, disabled_next, pf_extra))
                disabled_next = False
                pf_extra = {}
            elif kind == "c":
                t = int(float(fl.get("t", 0)))
                kindcp = CP_TYPES.get(t, CP_LINE)
                p.control_points.append(ControlPoint(int(fl["n"]), float(fl["x"]), float(fl["y"]), int(fl["N"]),
                                                     float(fl["X"]), float(fl["Y"]), kindcp, t if kindcp == CP_LINE else 0))
            elif kind == "#hugin_optimizeReferenceImage":
                pass
    # resolve images and linked lens parameters ("v=0" references image 0)
    lens_of_ref: dict[int, int] = {}
    for idx, (fl, cf, disabled, extra) in enumerate(raw_images):

        def val(key, default=0.0):
            v = fl.get(key, default)
            if isinstance(v, tuple):
                return raw_value(int(v[1]), key)
            return float(v)

        def raw_value(ref, key):
            v = raw_images[ref][0].get(key, 0.0)
            return raw_value(int(v[1]), key) if isinstance(v, tuple) else float(v)

        ref_v = fl.get("v")
        ref = int(ref_v[1]) if isinstance(ref_v, tuple) else idx
        code = int(float(fl.get("f", 0)))
        if ref in lens_of_ref and ref != idx:
            lid = lens_of_ref[ref]
        else:
            ln = Lens(lens_type=PTO_LENS.get(code, G.LENS_RECTILINEAR), hfov=val("v", 50.0),
                      fisheye_factor=PTO_LENS_K.get(code, 0.0), a=val("a"), b=val("b"), c=val("c"), d=val("d"), e=val("e"),
                      g=val("g"), t=val("t"), crop_factor=cf or 0.0)
            if "fisheye_factor" in extra:
                ln.fisheye_factor = float(extra["fisheye_factor"])
            if extra.get("lens_type") in G.LENS_TYPES:
                ln.lens_type = extra["lens_type"]
            lid = len(p.lenses)
            p.lenses.append(ln)
            lens_of_ref[idx] = lid
        name = fl.get("n", "")
        imgpath = name if os.path.isabs(name) else os.path.normpath(os.path.join(base, name))
        im = ImageEntry(path=imgpath, width=int(float(fl.get("w", 0))), height=int(float(fl.get("h", 0))), lens=lid,
                        yaw=val("y"), pitch=val("p"), roll=val("r"), ev=val("Eev", 0.0), wb_r=val("Er", 1.0) or 1.0,
                        wb_b=val("Eb", 1.0) or 1.0, include=not disabled)
        for k in ("TrX", "TrY", "TrZ"):
            if val(k):
                im.vp_enabled = True
        im.vp_x, im.vp_y, im.vp_z = val("TrX"), val("TrY"), val("TrZ")
        im.vp_pan, im.vp_tilt = val("Tpy"), val("Tpp")
        p.images.append(im)
        ln = p.lenses[lid]
        if lens_of_ref.get(idx) == lid:
            S = fl.get("S")
            if isinstance(S, str) and S:
                l, r, t, b = [float(x) for x in S.split(",")]
                if ln.lens_type == G.LENS_CIRCULAR_FISHEYE or extra.get("crop") == "circle":
                    ln.crop.mode = "circle"
                    ln.crop.cx, ln.crop.cy, ln.crop.radius = (l + r) / 2, (t + b) / 2, min(r - l, b - t) / 2
                elif (l, t, r, b) != (0, 0, im.width, im.height):
                    ln.crop.mode, ln.crop.left, ln.crop.right, ln.crop.top, ln.crop.bottom = "rect", l, r, t, b
            vs = _hugin_vig_scale(p, idx)
            ln.vig = [val("Vb") * vs ** 2, val("Vc") * vs ** 4, val("Vd") * vs ** 6]
            ln.vig_cx, ln.vig_cy = val("Vx"), val("Vy")
    return p


def import_pts(d: dict, path: str) -> Project:
    """PTGui 13 JSON project import (needs a sample to map the schema)."""
    schema = str(d.get("$schema", ""))
    raise NotImplementedError(
        f"PTGui JSON project import is not mapped yet (schema {schema or 'unknown'}). "
        "Send a sample .pts saved by PTGui so its fields can be mapped exactly; "
        "meanwhile export a Hugin .pto from PTGui/Hugin and open that.")
