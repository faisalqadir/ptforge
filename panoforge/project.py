"""Project data model: images, lenses, control points, panorama/output settings.

A project is plain data that round-trips through JSON (``*.pfproj``).  Geometry
is evaluated by building :class:`panoforge.geometry.Camera` objects from it.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

import numpy as np

from . import geometry as G

FORMAT_VERSION = 1

LENS_PARAMS = ("hfov", "fisheye_factor", "a", "b", "c", "d", "e", "g", "t")
IMAGE_POSE_PARAMS = ("yaw", "pitch", "roll")
VIEWPOINT_PARAMS = ("vp_x", "vp_y", "vp_z")
PHOTO_IMAGE_PARAMS = ("ev", "wb_r", "wb_b", "flare")

CP_NORMAL = "normal"
CP_VERTICAL = "vertical"
CP_HORIZONTAL = "horizontal"
CP_LINE = "line"


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Crop:
    """Source-image crop.  mode: none | rect | circle (circular fisheye)."""

    mode: str = "none"
    left: float = 0.0
    top: float = 0.0
    right: float = 0.0  # absolute pixel coordinate of the right edge (0 -> image width)
    bottom: float = 0.0
    # circle: centre and radius in pixels
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0


@dataclass
class Lens:
    """Shared lens/camera parameters (images taken with the same lens link to one)."""

    lens_type: str = G.LENS_RECTILINEAR
    hfov: float = 50.0
    fisheye_factor: float = 0.0
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    g: float = 0.0
    t: float = 0.0
    crop: Crop = field(default_factory=Crop)
    # photometric lens properties
    vig: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 1 + v1 r^2 + v2 r^4 + v3 r^6
    vig_cx: float = 0.0  # vignetting centre offset (pixels)
    vig_cy: float = 0.0
    response: str = "srgb"  # srgb | gamma | linear | emor | custom
    response_params: list = field(default_factory=lambda: [2.2])  # gamma or EMoR coefficients
    # metadata
    focal_mm: float = 0.0
    crop_factor: float = 0.0
    camera: str = ""
    lens_name: str = ""
    mirrored: bool = False
    # initial estimate (EXIF / image circle): every full alignment restarts from it, so
    # re-aligning is reproducible instead of drifting from the previous solution
    init: dict = field(default_factory=dict)
    locked: bool = False  # keep the current values (e.g. a calibrated lens from the lens database)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        crop = _from_dict(Crop, d.pop("crop", {}))
        obj = _from_dict(cls, d)
        obj.crop = crop
        return obj


@dataclass
class Mask:
    """Paint mask stored as a PNG next to the project: 0 none, 1 exclude (red), 2 include (green)."""

    path: str = ""


@dataclass
class ImageEntry:
    path: str
    width: int = 0
    height: int = 0
    lens: int = 0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    include: bool = True
    # per-image overrides of lens parameters ("use individual parameters")
    lens_overrides: dict = field(default_factory=dict)
    # viewpoint correction
    vp_enabled: bool = False
    vp_x: float = 0.0
    vp_y: float = 0.0
    vp_z: float = 0.0
    vp_pan: float = 0.0
    vp_tilt: float = -90.0  # default plane: the floor
    # photometric
    ev: float = 0.0  # exposure offset in stops relative to the reference
    wb_r: float = 1.0
    wb_b: float = 1.0
    flare: float = 0.0
    # EXIF derived exposure value (EV100) when known
    exif_ev: Optional[float] = None
    # bracket stack / link group: images with the same group share position
    link_group: int = -1
    blend_priority: int = 0
    mask: str = ""  # path of mask png (relative to project)
    exif: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)


@dataclass
class ControlPoint:
    i: int
    x1: float
    y1: float
    j: int
    x2: float
    y2: float
    kind: str = CP_NORMAL
    line: int = 0  # line id for CP_LINE
    error: float = 0.0  # last optimiser distance (pixels)

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)


@dataclass
class OptimizerSettings:
    anchor: int = 0
    optimize_fov: bool = True
    distortion: str = "medium"  # none | medium | heavy | heavy_shift
    optimize_fisheye_factor: bool = False
    optimize_shear: bool = False
    per_image: list = field(default_factory=list)  # lens params optimised per image
    hv_mode: str = "single_pass"  # ignore | second_pass | single_pass
    viewpoint: bool = False  # optimise viewpoint for images with vp_enabled
    link_roll: bool = False
    link_pitch: bool = False
    robust: bool = False


@dataclass
class PanoramaSettings:
    projection: str = "equirectangular"
    proj_params: dict = field(default_factory=dict)
    hfov: float = 360.0
    vfov: float = 180.0
    width: int = 0  # 0 -> optimum
    height: int = 0  # derived
    # output crop as fractions of the canvas (left, top, right, bottom)
    crop: list = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    # vertical offset of the projection centre (fraction of vfov) for asymmetric panos
    v_offset: float = 0.0


@dataclass
class ExposureSettings:
    optimize_exposure: str = "auto"  # auto | off | full | finetune
    optimize_wb: str = "auto"
    optimize_vignetting: bool = True
    optimize_flare: bool = False
    optimize_response: bool = False
    response: str = "srgb"
    hdr_mode: bool = False
    hdr_method: str = "fusion"  # fusion | true_hdr
    hdr_flow: str = "merge_then_blend"
    link_brackets: bool = True
    fusion_sigma: float = 0.2
    fusion_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])  # contrast, saturation, exposure
    exposure: float = 0.0  # global output exposure correction (stops)
    wb_r: float = 1.0
    wb_b: float = 1.0
    tonemap: bool = False
    tonemap_params: dict = field(default_factory=dict)
    deghost: bool = True


@dataclass
class OutputSettings:
    path: str = ""
    format: str = "jpeg"  # jpeg | png | tiff | psd | exr | hdr | webp
    quality: int = 95
    bit_depth: int = 8  # 8 | 16 | 32
    compression: str = "lzw"
    alpha: bool = True
    layers: str = "blended"  # blended | layers | blend_planes | layers_and_blended
    hdr_format: str = "exr"
    blender: str = "multiband"  # multiband | feather | none
    seams: str = "graphcut"  # graphcut | voronoi | none
    feather: float = 0.5  # 0 sharp .. 1 soft (multiband levels / feather width)
    interpolator: str = "lanczos"  # nearest | bilinear | bicubic | lanczos
    size_percent: float = 100.0
    max_megapixels: float = 0.0
    color_profile: str = "srgb"


@dataclass
class Project:
    images: list = field(default_factory=list)
    lenses: list = field(default_factory=list)
    control_points: list = field(default_factory=list)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    panorama: PanoramaSettings = field(default_factory=PanoramaSettings)
    exposure: ExposureSettings = field(default_factory=ExposureSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    metadata: dict = field(default_factory=dict)
    path: str = ""  # where it was loaded from / saved to (not serialised)

    # ------------------------------------------------------------------ geometry
    def lens_value(self, img_idx: int, name: str):
        im = self.images[img_idx]
        if name in im.lens_overrides:
            return im.lens_overrides[name]
        return getattr(self.lenses[im.lens], name)

    def camera(self, idx: int) -> G.Camera:
        im = self.images[idx]
        kw = {p: float(self.lens_value(idx, p)) for p in LENS_PARAMS}
        cam = G.Camera(
            width=im.width,
            height=im.height,
            lens_type=self.lenses[im.lens].lens_type,
            yaw=im.yaw,
            pitch=im.pitch,
            roll=im.roll,
            **kw,
        )
        if im.vp_enabled:
            cam.vp_x, cam.vp_y, cam.vp_z = im.vp_x, im.vp_y, im.vp_z
            cam.vp_pan, cam.vp_tilt = im.vp_pan, im.vp_tilt
        return cam

    def cameras(self):
        return [self.camera(i) for i in range(len(self.images))]

    def projection(self) -> G.Projection:
        return G.make_projection(self.panorama.projection, **self.panorama.proj_params)

    # ------------------------------------------------------------------ helpers
    def abspath(self, p: str) -> str:
        if not p or os.path.isabs(p):
            return p
        base = os.path.dirname(self.path) if self.path else os.getcwd()
        return os.path.normpath(os.path.join(base, p))

    def included(self):
        return [i for i, im in enumerate(self.images) if im.include]

    def link_groups(self) -> dict:
        groups: dict[int, list] = {}
        for i, im in enumerate(self.images):
            if im.link_group >= 0:
                groups.setdefault(im.link_group, []).append(i)
        return groups

    def cps_for(self, i, j=None):
        out = []
        for k, cp in enumerate(self.control_points):
            if j is None:
                if cp.i == i or cp.j == i:
                    out.append(k)
            elif (cp.i, cp.j) in ((i, j), (j, i)):
                out.append(k)
        return out

    def copy(self) -> "Project":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        d = {
            "format": "panoforge-project",
            "version": FORMAT_VERSION,
            "images": [asdict(im) for im in self.images],
            "lenses": [asdict(l) for l in self.lenses],
            "control_points": [asdict(c) for c in self.control_points],
            "optimizer": asdict(self.optimizer),
            "panorama": asdict(self.panorama),
            "exposure": asdict(self.exposure),
            "output": asdict(self.output),
            "metadata": self.metadata,
        }
        return _jsonable(d)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls()
        p.images = [ImageEntry.from_dict(x) for x in d.get("images", [])]
        p.lenses = [Lens.from_dict(x) for x in d.get("lenses", [])]
        p.control_points = [ControlPoint.from_dict(x) for x in d.get("control_points", [])]
        p.optimizer = _from_dict(OptimizerSettings, d.get("optimizer"))
        p.panorama = _from_dict(PanoramaSettings, d.get("panorama"))
        p.exposure = _from_dict(ExposureSettings, d.get("exposure"))
        p.output = _from_dict(OutputSettings, d.get("output"))
        p.metadata = d.get("metadata", {})
        return p

    def save(self, path: str, relative_paths: bool = True):
        path = os.path.abspath(path)
        d = self.to_dict()
        if relative_paths:
            base = os.path.dirname(path)
            for im in d["images"]:
                ap = self.abspath(im["path"])
                try:
                    im["path"] = os.path.relpath(ap, base)
                except ValueError:
                    im["path"] = ap
                if im.get("mask"):
                    try:
                        im["mask"] = os.path.relpath(self.abspath(im["mask"]), base)
                    except ValueError:
                        pass
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        # paths in memory become relative to the new location
        self.path = path
        if relative_paths:
            for im, dd in zip(self.images, d["images"]):
                im.path, im.mask = dd["path"], dd.get("mask", "")

    @classmethod
    def load(cls, path: str) -> "Project":
        if path.lower().endswith(".pto"):
            from .ptgui_format import import_pto  # Hugin / Panorama Tools script

            p = import_pto(path)
        else:
            with open(path) as f:
                d = json.load(f)
            if d.get("format") != "panoforge-project":
                from .ptgui_format import import_pts  # PTGui 13 JSON project

                p = import_pts(d, path)
            else:
                p = cls.from_dict(d)
        p.path = os.path.abspath(path)
        return p


def _jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("latin-1")
    return o
