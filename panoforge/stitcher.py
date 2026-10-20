"""The stitching pipeline.

    footprints -> low-res seam finding -> full-res label repair
               -> streaming multiband blend (per blend plane) -> HDR merge / fusion
               -> encode, crop, write (+ optional layers)
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import seams as S
from .blend import FeatherBlender, MultiBandBlender, NoBlender, auto_levels
from .hdr import exposure_fusion, merge_true_hdr, tonemap
from .imageio import linear_to_srgb, load_image, save_image
from .photometric import reference_ev
from .project import Project
from .remap import Canvas, Layer, Region, compute_maps, footprint, warp_image


@dataclass
class Unit:
    """A thing that gets one seam label: an image, or a bracket stack (merge-then-blend)."""

    uid: int
    members: list  # image indices, sorted dark -> bright (highest ev first)
    region: Region | None = None
    priority: int = 0


@dataclass
class StitchResult:
    path: str | None
    image: np.ndarray | None  # encoded float RGB (cropped)
    alpha: np.ndarray | None
    hdr: np.ndarray | None
    width: int
    height: int
    seconds: float
    labels: np.ndarray | None = None  # seam label map (seam resolution)
    extra_files: list = field(default_factory=list)


def output_width(p: Project) -> int:
    from .align import optimum_size

    if p.panorama.width:
        return int(p.panorama.width)
    w, h = optimum_size(p)
    s = p.output.size_percent / 100.0
    if p.output.max_megapixels:
        s = min(s, math.sqrt(p.output.max_megapixels * 1e6 / max(w * h, 1)))
    return max(16, int(round(w * s)))


def _units(p: Project, images, merge_stacks: bool):
    units = []
    if merge_stacks:
        groups = p.link_groups()
        seen = set()
        for i in images:
            if i in seen:
                continue
            g = p.images[i].link_group
            if g >= 0 and g in groups:
                mem = [m for m in groups[g] if m in images]
            else:
                mem = [i]
            seen.update(mem)
            mem = sorted(mem, key=lambda m: -p.images[m].ev)
            units.append(Unit(len(units), mem, priority=max(p.images[m].blend_priority for m in mem)))
    else:
        for i in images:
            units.append(Unit(len(units), [i], priority=p.images[i].blend_priority))
    return units


def _union_region(regions, canvas):
    regions = [r for r in regions if r is not None]
    if not regions:
        return None
    if len(regions) == 1:
        return regions[0]
    y0 = min(r.y0 for r in regions)
    y1 = max(r.y1 for r in regions)
    x0 = min(r.x0 for r in regions)
    x1 = max(r.x1 for r in regions)
    w = x1 - x0
    if canvas.wrap and w >= canvas.width:
        return Region(0, y0, canvas.width, y1 - y0)
    return Region(x0, y0, w, y1 - y0)


class Stitcher:
    def __init__(self, p: Project, width: int | None = None, progress=None, seam_width: int | None = None,
                 interpolator: str | None = None, blender: str | None = None, seams: str | None = None,
                 feather: float | None = None, layers: str | None = None, tmpdir: str | None = None):
        self.p = p
        self.o = p.output
        self.progress = progress or (lambda *a, **k: None)
        self.interp = interpolator or self.o.interpolator
        self.blender_kind = blender or self.o.blender
        self.seam_kind = seams or self.o.seams
        self.feather = self.o.feather if feather is None else feather
        self.layers = layers or self.o.layers
        self.images = p.included()
        ex = p.exposure
        self.hdr = ex.hdr_mode and bool(p.link_groups()) or (ex.hdr_mode and len({round(p.images[i].ev, 1) for i in self.images}) > 1)
        self.method = ex.hdr_method if self.hdr else "ldr"
        self.flow = ex.hdr_flow if self.hdr else "merge_then_blend"
        width = width or output_width(p)
        canvas = Canvas(p, width)
        self.canvas = canvas
        # only the cropped output window is ever computed
        x0, y0, x1, y1 = canvas.crop_rect()
        self.win = Region(x0, y0, x1 - x0, y1 - y0)
        self.win_wrap = canvas.wrap and x0 == 0 and x1 == canvas.width
        self.levels = auto_levels(self.win.w, self.win.h, feather=self.feather) if self.blender_kind == "multiband" else 0
        self.seam_width = seam_width or (2400 if len(self.images) <= 40 else 1600)
        self.ev_ref = reference_ev(p)
        self.out_gain = 2.0 ** p.exposure.exposure
        self.out_wb = np.array([p.exposure.wb_r, 1.0, p.exposure.wb_b], np.float32)
        self.tmp = tmpdir
        self._sources: dict = {}

    # ------------------------------------------------------------------ helpers
    def _source(self, idx):
        if idx not in self._sources:
            self._sources = {idx: load_image(self.p.abspath(self.p.images[idx].path)).rgb}  # keep only one in memory
        return self._sources[idx]

    def _encode(self, scene):
        """linear scene -> output encoded (LDR)."""
        return linear_to_srgb(np.maximum(scene * (self.out_gain * self.out_wb), 0))

    def _gain(self, idx):
        return 2.0 ** (self.p.images[idx].ev - self.ev_ref)

    def unit_layer(self, u: Unit, canvas: Canvas, region: Region, interp: str, plane: int | None = None,
                   maps=None) -> Layer | None:
        """Build the blendable layer of a unit.  plane selects one bracket member (blend-then-merge)."""
        members = u.members if plane is None else [u.members[min(plane, len(u.members) - 1)]]
        layers = []
        for m in members:
            src = self._source(m) if canvas is self.canvas else None
            lay = warp_image(self.p, m, canvas, region, interp, source=src, ev_ref=self.ev_ref)
            if lay is not None:
                layers.append((m, lay))
        if not layers:
            return None
        if len(layers) == 1 and (plane is not None or self.method == "ldr"):
            m, lay = layers[0]
            if plane is not None and self.hdr:
                # blend plane: keep the exposure's own appearance
                lay.rgb = linear_to_srgb(np.clip(lay.rgb / self._gain(m) * self.out_wb, 0, None)) if self.method == "fusion" else lay.rgb
            elif self.method == "ldr":
                lay.rgb = self._encode(lay.rgb)
            return lay
        scenes = [l.rgb for _, l in layers]
        alphas = [l.alpha for _, l in layers]
        if self.method == "true_hdr" or self.method == "ldr":
            rad, alpha = merge_true_hdr(scenes, [self._gain(m) for m, _ in layers], alphas, self.p.exposure.deghost)
            rgb = rad if self.method == "true_hdr" else self._encode(rad)
        else:
            enc = [linear_to_srgb(np.clip(l.rgb / self._gain(m) * self.out_wb, 0, None)) for m, l in layers]
            rgb, alpha = exposure_fusion(enc, alphas, self.p.exposure.fusion_weights, self.p.exposure.fusion_sigma)
        inc = None
        for _, l in layers:
            if l.include is not None:
                inc = l.include if inc is None else (inc | l.include)
        return Layer(u.uid, region, rgb, alpha, inc)

    # ------------------------------------------------------------------ seams
    def find_seams(self, units):
        cv_full = self.canvas
        sw = min(cv_full.width, self.seam_width)
        cs = cv_full.scaled(sw)
        n = len(units)
        H, W = cs.height, cs.width
        imgs, masks, incs = [], [], []
        regions = []
        for k, u in enumerate(units):
            self.progress("seams", k, n)
            reg = _union_region([footprint(self.p, m, cs) for m in u.members], cs)
            img = np.zeros((H, W, 3), np.float16)
            msk = np.zeros((H, W), bool)
            inc = None
            if reg is not None:
                lay = self.unit_layer(u, cs, reg, "bilinear")
                if lay is not None:
                    rgb = lay.rgb if self.method != "true_hdr" else linear_to_srgb(np.clip(lay.rgb * 0.5, 0, 1))
                    cols = np.arange(reg.x0, reg.x1)
                    if cs.wrap:
                        cols = cols % W
                    ok = (cols >= 0) & (cols < W)
                    ys = slice(reg.y0, reg.y0 + reg.h)
                    img[ys, cols[ok]] = rgb[:, ok]
                    msk[ys, cols[ok]] = lay.alpha[:, ok] > 0.5
                    if lay.include is not None:
                        inc = np.zeros((H, W), bool)
                        inc[ys, cols[ok]] = lay.include[:, ok]
            imgs.append(img)
            masks.append(msk)
            incs.append(inc)
            regions.append(reg)
        prio = [u.priority for u in units]
        if self.seam_kind == "none":
            # stacking order: later images on top
            lab = np.full((H, W), -1, np.int32)
            for k, m in enumerate(masks):
                lab[m] = k
        else:
            lab = S.initial_labels(masks, cs.wrap, prio, incs)
            if self.seam_kind == "graphcut" and n > 1:
                band = max(8, int(0.03 * W))
                S.refine_graphcut(lab, [x.astype(np.float32) for x in imgs], masks, cs.wrap, band=band, includes=incs)
        self.seam_canvas = cs
        return lab, masks

    # ------------------------------------------------------------------ main
    def _clip(self, reg: Region | None) -> Region | None:
        """Intersect a full-canvas region with the output window (full-canvas coordinates)."""
        if reg is None:
            return None
        win, cv = self.win, self.canvas
        y0, y1 = max(reg.y0, win.y0), min(reg.y1, win.y1)
        if y1 <= y0:
            return None
        if self.win_wrap:
            return Region(reg.x0, y0, reg.w, y1 - y0)
        pieces = []
        for sh in ([0, -cv.width, cv.width] if cv.wrap else [0]):
            a, b = max(reg.x0 + sh, win.x0), min(reg.x1 + sh, win.x1)
            if b > a:
                pieces.append((a, b))
        if not pieces:
            return None
        a, b = min(x[0] for x in pieces), max(x[1] for x in pieces)
        return Region(a, y0, b - a, y1 - y0)

    def _wcols(self, reg: Region):
        """Window column indices of a (clipped, full-canvas) region."""
        cols = np.arange(reg.x0, reg.x1) - self.win.x0
        if self.win_wrap:
            cols = cols % self.win.w
        return np.clip(cols, 0, self.win.w - 1)

    EXT_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".tif": "tiff", ".tiff": "tiff", ".psd": "psd",
                  ".psb": "psd", ".exr": "exr", ".hdr": "hdr", ".webp": "webp"}

    def run(self, output_path: str | None = None, write: bool = True) -> StitchResult:
        t0 = time.time()
        if output_path:  # an explicit file name decides the format
            fmt = self.EXT_FORMAT.get(os.path.splitext(output_path)[1].lower())
            if fmt:
                self.o.format = fmt
        p, cv, win = self.p, self.canvas, self.win
        merge_stacks = self.hdr and self.flow == "merge_then_blend"
        units = _units(p, self.images, merge_stacks or (self.hdr and self.flow == "blend_then_merge"))
        for u in units:
            u.region = self._clip(_union_region([footprint(p, m, cv) for m in u.members], cv))
        units = [u for u in units if u.region is not None]
        for k, u in enumerate(units):
            u.uid = k
        if not units:
            raise RuntimeError("no images cover the panorama area")
        lab_small, small_masks = self.find_seams(units)
        # label map of the output window, sampled from the seam canvas
        H, W = win.h, win.w
        hs, ws = lab_small.shape
        ry = np.minimum(((np.arange(win.y0, win.y1) + 0.5) * hs / cv.height).astype(np.int64), hs - 1)
        rx = np.minimum(((np.arange(win.x0, win.x1) + 0.5) * ws / cv.width).astype(np.int64), ws - 1)
        L = lab_small[ry][:, rx].astype(np.int16)
        tmp = self.tmp or tempfile.mkdtemp(prefix="panoforge_")
        own_tmp = self.tmp is None
        try:
            # pass 1: exact validity per unit; repair labels that point at a non-covering unit
            valid_store = {}
            ok = np.zeros((H, W), bool)
            covered = np.zeros((H, W), bool)
            for k, u in enumerate(units):
                self.progress("coverage", k, len(units))
                reg = u.region
                v = np.zeros((reg.h, reg.w), bool)
                for m in u.members:
                    _, _, vm = compute_maps(p, m, cv, reg)
                    v |= vm
                valid_store[u.uid] = np.packbits(v, axis=None), v.shape
                ys = slice(reg.y0 - win.y0, reg.y1 - win.y0)
                cols = self._wcols(reg)
                ok[ys, cols] |= (L[ys][:, cols] == u.uid) & v
                covered[ys, cols] |= v
            orphan = covered & ~ok
            if orphan.any():
                for u in sorted(units, key=lambda u: -u.priority):
                    reg = u.region
                    ys = slice(reg.y0 - win.y0, reg.y1 - win.y0)
                    cols = self._wcols(reg)
                    packed, shape = valid_store[u.uid]
                    v = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
                    sub = orphan[ys][:, cols] & v
                    if sub.any():
                        blk = L[ys][:, cols]
                        blk[sub] = u.uid
                        L[ys, cols] = blk
                        o = orphan[ys][:, cols]
                        o[sub] = False
                        orphan[ys, cols] = o
            L[~covered] = -1
            del ok, valid_store
            # pass 2: blend (one pass per blend plane)
            planes = [None]
            if self.hdr and self.flow == "blend_then_merge":
                planes = list(range(max(len(u.members) for u in units)))
            plane_results, layer_files = [], []
            self._psd_layers = []
            cover = None
            for pl in planes:
                bl = self._make_blender(H, W)
                for k, u in enumerate(units):
                    self.progress("blend" if pl is None else f"blend plane {pl + 1}", k, len(units))
                    lay = self.unit_layer(u, cv, u.region, self.interp, plane=pl)
                    if lay is None:
                        continue
                    reg = u.region
                    ys = slice(reg.y0 - win.y0, reg.y1 - win.y0)
                    wmask = (L[ys][:, self._wcols(reg)] == u.uid) & (lay.alpha > 0)
                    bl.feed(self._to_blend_domain(lay.rgb), wmask.astype(np.float32), reg.x0 - win.x0, reg.y0 - win.y0,
                            valid=lay.alpha > 0)
                    if self.layers in ("layers", "layers_and_blended") and write and pl is None:
                        layer_files += self._save_layer(lay, u, output_path, tmp)
                res, cover = bl.result()
                plane_results.append(self._from_blend_domain(res))
                del bl
            alpha = np.clip(cover, 0, 1)
            hdr_img = None
            if len(plane_results) > 1:
                if self.method == "true_hdr":
                    gains = [2.0 ** (np.median([p.images[u.members[min(k, len(u.members) - 1)]].ev for u in units]) - self.ev_ref)
                             for k in range(len(plane_results))]
                    # planes already hold linear scene radiance; gains only drive the weights
                    hdr_img, _ = merge_true_hdr(plane_results, gains, [alpha] * len(plane_results), self.p.exposure.deghost)
                    image = tonemap(hdr_img, p.exposure.tonemap_params, alpha)
                else:
                    image, _ = exposure_fusion(plane_results, [alpha] * len(plane_results), p.exposure.fusion_weights, p.exposure.fusion_sigma)
            else:
                image = plane_results[0]
                if self.method == "true_hdr":
                    hdr_img = image
                    image = tonemap(hdr_img, p.exposure.tonemap_params, alpha)
            image = np.clip(image, 0, 1)
            image[alpha <= 0] = 0  # coarse pyramid levels bleed outside the covered area
            result = StitchResult(None, image, alpha, hdr_img, image.shape[1], image.shape[0], 0.0, lab_small)
            if write:
                if self._psd_layers:
                    result.path = self._write_layered_psd(image, alpha, output_path)
                    extra = []
                else:
                    result.path, extra = self._write(image, alpha, hdr_img, output_path)
                result.extra_files = extra + layer_files
            result.seconds = time.time() - t0
            self.progress("done", 1, 1)
            return result
        finally:
            if own_tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def _make_blender(self, H, W):
        if self.blender_kind == "multiband":
            return MultiBandBlender(W, H, self.levels, self.win_wrap)
        if self.blender_kind == "feather":
            sigma = 2 + self.feather * 0.01 * min(W, H)
            return FeatherBlender(W, H, self.win_wrap, sigma)
        return NoBlender(W, H, self.win_wrap)

    def _to_blend_domain(self, rgb):
        if self.method == "true_hdr":
            return np.log(np.maximum(rgb, 0) + 1e-4).astype(np.float32)
        return rgb

    def _from_blend_domain(self, rgb):
        if self.method == "true_hdr":
            return np.maximum(np.exp(rgb) - 1e-4, 0).astype(np.float32)
        return rgb

    def _write_layered_psd(self, image, alpha, output_path):
        from .psd import write_psd

        path = self._out_path(output_path)
        layers = []
        for name, fn, pos in self._psd_layers:
            d = np.load(fn)
            layers.append((name, d["rgb"], d["alpha"], pos))
        if self.layers == "layers_and_blended":
            layers.append(("Blended panorama", image, alpha, (0, 0)))
        write_psd(path, layers, bits=16 if self.o.bit_depth >= 16 else 8, width=image.shape[1], height=image.shape[0],
                  composite=(image, alpha))
        return path

    def _out_path(self, output_path):
        if output_path:
            return output_path
        if self.o.path:
            return self.p.abspath(self.o.path)
        base = os.path.splitext(self.p.path)[0] if self.p.path else os.path.join(os.getcwd(), "panorama")
        ext = {"jpeg": ".jpg", "png": ".png", "tiff": ".tif", "psd": ".psd", "exr": ".exr", "hdr": ".hdr", "webp": ".webp"}[self.o.format]
        return base + ext

    def _write(self, image, alpha, hdr_img, output_path):
        from .output import write_metadata_bytes

        path = self._out_path(output_path)
        ext = os.path.splitext(path)[1].lower()
        extra = []
        exif = write_metadata_bytes(self.p, image.shape[1], image.shape[0]) if ext in (".jpg", ".jpeg") else None
        use_alpha = alpha if (self.o.alpha and ext in (".png", ".tif", ".tiff", ".psd", ".webp", ".exr")) else None
        if ext in (".exr", ".hdr"):
            save_image(path, hdr_img if hdr_img is not None else srgb_decode(image), use_alpha, bits=32 if self.o.bit_depth == 32 else 16)
        else:
            img = image
            if use_alpha is None and alpha is not None:
                img = image * alpha[..., None]  # composite soft edges over black
            save_image(path, img, use_alpha, bits=self.o.bit_depth if self.o.bit_depth in (8, 16) else 8, quality=self.o.quality,
                       compression=self.o.compression, exif=exif)
            if ext in (".jpg", ".jpeg"):
                from .output import gpano_xmp, inject_jpeg_xmp

                xmp = gpano_xmp(self.p, image.shape[1], image.shape[0])
                if xmp:
                    inject_jpeg_xmp(path, xmp)
            if hdr_img is not None:
                hp = os.path.splitext(path)[0] + "_hdr." + ("exr" if self.o.hdr_format == "exr" else "hdr")
                save_image(hp, hdr_img, alpha if self.o.hdr_format == "exr" else None, bits=32)
                extra.append(hp)
        return path, extra

    def _save_layer(self, lay: Layer, u: Unit, output_path, tmp):
        """Store an individual remapped layer.

        PSD output collects the layers for one layered file; otherwise each layer is a TIFF
        with alpha.  Layers crossing the 360-degree edge are split in two.
        """
        rgb = lay.rgb if self.method != "true_hdr" else linear_to_srgb(lay.rgb)
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        x0 = lay.region.x0 - self.win.x0
        y0 = lay.region.y0 - self.win.y0
        pieces = []
        W = self.win.w
        if self.win_wrap and (x0 < 0 or x0 + rgb.shape[1] > W):
            cols = (np.arange(x0, x0 + rgb.shape[1]) % W)
            split = int(np.nonzero(np.diff(cols) < 0)[0][0]) + 1 if (np.diff(cols) < 0).any() else rgb.shape[1]
            pieces.append((rgb[:, :split], lay.alpha[:, :split], int(cols[0])))
            if split < rgb.shape[1]:
                pieces.append((rgb[:, split:], lay.alpha[:, split:], 0))
        else:
            pieces.append((rgb, lay.alpha, x0))
        name = os.path.basename(self.p.images[u.members[0]].path)
        out = []
        base = os.path.splitext(self._out_path(output_path))[0]
        for k, (r, a, xx) in enumerate(pieces):
            if self.o.format == "psd":
                fn = os.path.join(tmp, f"layer{u.uid:04d}_{k}.npz")
                np.savez(fn, rgb=r, alpha=a.astype(np.float32))
                self._psd_layers.append((name if k == 0 else f"{name} ({k + 1})", fn, (xx, y0)))
            else:
                path = f"{base}_layer{u.uid:04d}{'' if k == 0 else f'_{k}'}.tif"
                save_image(path, r, a, bits=16 if self.o.bit_depth >= 16 else 8)
                out.append(path)
        return out


def srgb_decode(x):
    from .imageio import srgb_to_linear

    return srgb_to_linear(np.clip(x, 0, 1))


def stitch(p: Project, output_path: str | None = None, width: int | None = None, progress=None, write: bool = True, **kw) -> StitchResult:
    return Stitcher(p, width=width, progress=progress, **kw).run(output_path, write=write)


def preview(p: Project, width: int = 1600, seams: str = "voronoi", progress=None) -> StitchResult:
    """Fast low-resolution stitch for previews."""
    return Stitcher(p, width=width, progress=progress, seams=seams, interpolator="bilinear", seam_width=min(width, 1200)).run(write=False)
