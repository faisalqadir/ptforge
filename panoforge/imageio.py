"""Image loading/saving and EXIF extraction.

Pixels are returned as float32 RGB (H, W, 3) [+ optional alpha].  LDR files are
scaled to 0..1 and stay in their (gamma-encoded) colour space; HDR files
(EXR / Radiance) are linear floats and flagged ``is_hdr``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ExifTags  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw", ".nrw", ".3fr", ".iiq", ".rwl", ".x3f"}
HDR_EXTS = {".exr", ".hdr"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif", ".jp2"} | RAW_EXTS | HDR_EXTS


@dataclass
class LoadedImage:
    rgb: np.ndarray  # float32 HxWx3
    alpha: np.ndarray | None
    bits: int
    is_hdr: bool
    is_raw: bool = False


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _exif_orientation(path):
    try:
        with Image.open(path) as im:
            return int(im.getexif().get(0x0112, 1))
    except Exception:
        return 1


def _apply_orientation(a: np.ndarray, o: int) -> np.ndarray:
    if o == 2:
        return a[:, ::-1]
    if o == 3:
        return a[::-1, ::-1]
    if o == 4:
        return a[::-1]
    if o == 5:
        return np.swapaxes(a, 0, 1)
    if o == 6:
        return np.swapaxes(a, 0, 1)[:, ::-1]
    if o == 7:
        return np.swapaxes(a, 0, 1)[::-1, ::-1]
    if o == 8:
        return np.swapaxes(a, 0, 1)[::-1]
    return a


def image_size(path: str, use_orientation: bool = True) -> tuple[int, int]:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        im = load_image(path)
        return im.rgb.shape[1], im.rgb.shape[0]
    if ext in HDR_EXTS:
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return a.shape[1], a.shape[0]
    with Image.open(path) as im:
        w, h = im.size
        o = int(im.getexif().get(0x0112, 1)) if use_orientation else 1
    return (h, w) if o in (5, 6, 7, 8) else (w, h)


def load_image(path: str, use_orientation: bool = True, raw_params: dict | None = None) -> LoadedImage:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        return _load_raw(path, raw_params or {})
    if ext in (".heic", ".heif"):
        try:
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("HEIC support requires pillow-heif") from e
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"))
        return LoadedImage(a.astype(np.float32) / 255.0, None, 8, False)

    a = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if a is None:
        with Image.open(path) as im:  # fallback through Pillow
            a = np.asarray(im.convert("RGBA" if "A" in im.mode else "RGB"))
            a = a[..., [2, 1, 0, 3]] if a.shape[-1] == 4 else a[..., ::-1]
    if a.ndim == 2:
        a = a[..., None]
    if a.dtype == np.uint8:
        bits, scale, hdr = 8, 1 / 255.0, False
    elif a.dtype == np.uint16:
        bits, scale, hdr = 16, 1 / 65535.0, False
    else:
        bits, scale, hdr = 32, 1.0, True
    a = a.astype(np.float32) * np.float32(scale)
    if hdr and ext not in HDR_EXTS and float(np.nanmax(a)) <= 1.0001:
        hdr = False  # float TIFF holding LDR data
    alpha = None
    c = a.shape[-1]
    if c == 1:
        rgb = np.repeat(a, 3, axis=-1)
    elif c == 2:
        rgb, alpha = np.repeat(a[..., :1], 3, axis=-1), a[..., 1]
    else:
        rgb = a[..., 2::-1]
        if c >= 4:
            alpha = a[..., 3]
    if use_orientation and ext not in HDR_EXTS:
        o = _exif_orientation(path)
        if o != 1:
            rgb = _apply_orientation(rgb, o)
            if alpha is not None:
                alpha = _apply_orientation(alpha, o)
    rgb = np.ascontiguousarray(np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0))
    if alpha is not None:
        alpha = np.ascontiguousarray(alpha)
    return LoadedImage(rgb, alpha, bits, hdr)


def _load_raw(path, params):
    try:
        import rawpy  # type: ignore
    except ImportError as e:
        raise RuntimeError("RAW support requires the 'rawpy' package (pip install rawpy)") from e
    with rawpy.imread(path) as raw:
        kw = dict(
            use_camera_wb=params.get("camera_wb", True),
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1) if params.get("linear", True) else (2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full if params.get("noise_reduction") else rawpy.FBDDNoiseReductionMode.Off,
        )
        if params.get("ca_correction"):
            kw["chromatic_aberration"] = (1.0, 1.0)
        rgb = raw.postprocess(**kw).astype(np.float32) / 65535.0
    if params.get("linear", True):
        # keep linear data but present it in sRGB encoding so the rest of the pipeline is uniform
        rgb = linear_to_srgb(rgb)
    return LoadedImage(rgb, None, 16, False, True)


# --------------------------------------------------------------------------- colour transfer


def _float(x):
    x = np.asarray(x)
    return x if x.dtype in (np.float32, np.float64) else x.astype(np.float32)


def srgb_to_linear(x):
    """sRGB decode; keeps float64 input in float64 (needed for numerical derivatives)."""
    x = _float(x)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4).astype(x.dtype)


def linear_to_srgb(x):
    x = _float(x)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055).astype(x.dtype)


# --------------------------------------------------------------------------- saving


def save_image(path: str, rgb: np.ndarray, alpha: np.ndarray | None = None, bits: int = 8, quality: int = 95,
               compression: str = "lzw", icc: bytes | None = None, exif: bytes | None = None, dpi: float | None = None):
    """Save float RGB(+alpha) data.  ``bits`` 8/16 for LDR formats, 32 for float."""
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if ext in (".exr", ".hdr"):
        a = rgb[..., ::-1].astype(np.float32)
        if alpha is not None and ext == ".exr":
            a = np.dstack([a, alpha.astype(np.float32)])
        params = []
        if ext == ".exr":
            params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if bits == 16 else cv2.IMWRITE_EXR_TYPE_FLOAT]
        if not cv2.imwrite(path, a, params):
            raise IOError(f"could not write {path}")
        return
    if ext in (".tif", ".tiff"):
        import tifffile

        data = _to_bits(rgb, bits)
        if alpha is not None:
            data = np.dstack([data, _to_bits(alpha[..., None], bits)])
        extratags = []
        if icc:
            extratags.append((34675, "B", len(icc), icc, False))
        comp = {"none": None, "lzw": "lzw", "zip": "zlib", "deflate": "zlib", "packbits": "packbits"}.get(compression, "lzw")
        if bits == 32 and comp == "lzw":
            comp = "zlib"
        res = (dpi, dpi) if dpi else None
        kw = dict(photometric="rgb", extrasamples=["unassalpha"] if alpha is not None else None,
                  extratags=extratags, resolution=res, bigtiff=data.nbytes > 3.9e9)
        try:
            tifffile.imwrite(path, data, compression=comp, **kw)
        except Exception:  # codec unavailable (imagecodecs missing or incomplete): deflate is built in
            if comp == "zlib":
                raise
            tifffile.imwrite(path, data, compression="zlib", **kw)
        return
    if ext == ".psd" or ext == ".psb":
        from .psd import write_psd

        write_psd(path, [("Panorama", rgb, alpha, (0, 0))], bits=8 if bits == 8 else 16)
        return
    # JPEG / PNG / WebP / BMP through Pillow (for metadata support)
    if ext in (".jpg", ".jpeg"):
        a8 = _to_bits(rgb, 8)
        im = Image.fromarray(a8, "RGB")
        kw = dict(quality=int(quality), subsampling=0 if quality >= 90 else 2, optimize=True)
        if icc:
            kw["icc_profile"] = icc
        if exif:
            kw["exif"] = exif
        if dpi:
            kw["dpi"] = (dpi, dpi)
        im.save(path, **kw)
        return
    if ext == ".png":
        if bits == 16:
            a = _to_bits(rgb, 16)[..., ::-1]
            if alpha is not None:
                a = np.dstack([a, _to_bits(alpha[..., None], 16)])
            cv2.imwrite(path, a, [cv2.IMWRITE_PNG_COMPRESSION, 4])
            return
        a8 = _to_bits(rgb, 8)
        im = Image.fromarray(np.dstack([a8, _to_bits(alpha[..., None], 8)]) if alpha is not None else a8)
        im.save(path, icc_profile=icc) if icc else im.save(path)
        return
    if ext == ".webp":
        a8 = _to_bits(rgb, 8)
        im = Image.fromarray(np.dstack([a8, _to_bits(alpha[..., None], 8)]) if alpha is not None else a8)
        im.save(path, quality=int(quality))
        return
    a8 = _to_bits(rgb, 8)
    cv2.imwrite(path, a8[..., ::-1])


def _to_bits(x, bits):
    if bits == 32:
        return x.astype(np.float32)
    m = 255 if bits == 8 else 65535
    return np.clip(np.rint(x * m), 0, m).astype(np.uint8 if bits == 8 else np.uint16)


# --------------------------------------------------------------------------- EXIF

_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _rat(v):
    try:
        if isinstance(v, tuple):
            return float(v[0]) / float(v[1]) if v[1] else 0.0
        return float(v)
    except Exception:
        return 0.0


def _gps_deg(v, ref):
    try:
        d = _rat(v[0]) + _rat(v[1]) / 60 + _rat(v[2]) / 3600
        return -d if ref in ("S", "W") else d
    except Exception:
        return None


def read_exif(path: str) -> dict:
    """Extract the metadata a stitcher cares about.  Missing values are omitted."""
    out: dict = {}
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in RAW_EXTS:
            return _read_exif_exifread(path)
        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return _read_exif_exifread(path)
            sub = ex.get_ifd(0x8769)
            gps = ex.get_ifd(0x8825)
            allt = dict(ex)
            allt.update(sub)
    except Exception:
        return _read_exif_exifread(path)

    def g(name):
        return allt.get(_TAGS.get(name))

    if g("Make"):
        out["make"] = str(g("Make")).strip("\x00 ")
    if g("Model"):
        out["model"] = str(g("Model")).strip("\x00 ")
    if g("LensModel"):
        out["lens"] = str(g("LensModel")).strip("\x00 ")
    fl = _rat(g("FocalLength") or 0)
    if fl:
        out["focal_mm"] = fl
    f35 = g("FocalLengthIn35mmFilm")
    if f35:
        out["focal35"] = float(f35)
    et = _rat(g("ExposureTime") or 0)
    if et:
        out["exposure_time"] = et
    fn = _rat(g("FNumber") or 0)
    if fn:
        out["fnumber"] = fn
    iso = g("ISOSpeedRatings") or g("PhotographicSensitivity")
    if iso:
        out["iso"] = float(iso[0] if isinstance(iso, (tuple, list)) else iso)
    if g("ExposureBiasValue") is not None:
        out["exposure_bias"] = _rat(g("ExposureBiasValue"))
    if g("DateTimeOriginal"):
        out["datetime"] = str(g("DateTimeOriginal"))
    if g("WhiteBalance") is not None:
        out["wb_auto"] = int(g("WhiteBalance")) == 0
    if g("ExposureMode") is not None:
        out["exposure_auto"] = int(g("ExposureMode")) == 0
    if g("Orientation"):
        out["orientation"] = int(g("Orientation"))
    fpx, fpu = _rat(g("FocalPlaneXResolution") or 0), g("FocalPlaneResolutionUnit")
    if fpx:
        unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(fpu or 2), 25.4)
        out["focal_plane_px_per_mm"] = fpx / unit_mm
    if gps:
        lat = _gps_deg(gps.get(2), gps.get(1)) if gps.get(2) else None
        lon = _gps_deg(gps.get(4), gps.get(3)) if gps.get(4) else None
        if lat is not None and lon is not None:
            out["gps_lat"], out["gps_lon"] = lat, lon
        if gps.get(6) is not None:
            out["gps_alt"] = _rat(gps.get(6)) * (-1 if gps.get(5) == 1 else 1)
        if gps.get(17) is not None:
            out["gps_heading"] = _rat(gps.get(17))
    _derive(out)
    return out


def _read_exif_exifread(path):
    out: dict = {}
    try:
        import exifread

        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return out

    def val(name):
        t = tags.get(name)
        if t is None:
            return None
        v = t.values
        if isinstance(v, list) and v:
            v = v[0]
        if hasattr(v, "num"):
            return float(v.num) / float(v.den) if v.den else 0.0
        return v

    for key, name in [("make", "Image Make"), ("model", "Image Model"), ("lens", "EXIF LensModel")]:
        v = tags.get(name)
        if v:
            out[key] = str(v).strip()
    for key, name in [("focal_mm", "EXIF FocalLength"), ("focal35", "EXIF FocalLengthIn35mmFilm"), ("exposure_time", "EXIF ExposureTime"),
                      ("fnumber", "EXIF FNumber"), ("iso", "EXIF ISOSpeedRatings"), ("exposure_bias", "EXIF ExposureBiasValue")]:
        v = val(name)
        if v:
            out[key] = float(v)
    if tags.get("EXIF DateTimeOriginal"):
        out["datetime"] = str(tags["EXIF DateTimeOriginal"])
    _derive(out)
    return out


def _derive(out):
    if out.get("focal_mm") and out.get("focal35"):
        out["crop_factor"] = out["focal35"] / out["focal_mm"]
    t, n = out.get("exposure_time"), out.get("fnumber")
    if t and n:
        iso = out.get("iso", 100.0) or 100.0
        out["ev100"] = math.log2(n * n / t) - math.log2(iso / 100.0)


def hfov_from_exif(exif: dict, width: int, height: int, lens_type: str = "rectilinear", fisheye_factor: float = 0.0):
    """Estimate the horizontal field of view in degrees, or None."""
    from . import geometry as G

    f = exif.get("focal_mm")
    if not f:
        return None
    crop = exif.get("crop_factor")
    if crop:
        diag = 43.2666 / crop
        sensor_w = diag * width / math.hypot(width, height)
    elif exif.get("focal_plane_px_per_mm"):
        sensor_w = exif.get("_orig_width", width) / exif["focal_plane_px_per_mm"]
    else:
        return None
    half = sensor_w / 2.0 / f
    if lens_type == G.LENS_RECTILINEAR:
        return 2 * math.degrees(math.atan(half))
    return 2 * math.degrees(float(G.lens_theta(np.float64(half), fisheye_factor)))
