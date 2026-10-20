"""Output helpers: metadata (EXIF / GPano XMP), cube faces, format conversions."""
from __future__ import annotations

import math
import os
import struct
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from . import geometry as G
from .project import Project

SOFTWARE = "PanoForge"


def write_metadata_bytes(p: Project, width: int, height: int) -> bytes:
    """EXIF block for the stitched panorama (camera, date and GPS of the first source)."""
    ex = Image.Exif()
    ex[0x0131] = SOFTWARE
    src = p.images[p.included()[0]].exif if p.images else {}
    if src.get("make"):
        ex[0x010F] = src["make"]
    if src.get("model"):
        ex[0x0110] = src["model"]
    ex[0x0132] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    sub = ex.get_ifd(0x8769)
    if src.get("datetime"):
        sub[0x9003] = src["datetime"]
    sub[0xA002] = width
    sub[0xA003] = height
    lat, lon = p.metadata.get("gps_lat", src.get("gps_lat")), p.metadata.get("gps_lon", src.get("gps_lon"))
    if lat is not None and lon is not None:
        gps = ex.get_ifd(0x8825)

        def dms(v):
            v = abs(v)
            d = int(v)
            m = int((v - d) * 60)
            s = (v - d - m / 60) * 3600
            return ((d, 1), (m, 1), (int(s * 1000), 1000))

        gps[1] = "N" if lat >= 0 else "S"
        gps[2] = dms(lat)
        gps[3] = "E" if lon >= 0 else "W"
        gps[4] = dms(lon)
    return ex.tobytes()


def gpano_xmp(p: Project, width: int, height: int) -> str | None:
    """Google Photo Sphere (GPano) metadata so viewers recognise the panorama."""
    if p.panorama.projection not in ("equirectangular", "cylindrical"):
        return None
    full_w = int(round(width / max(p.panorama.crop[2] - p.panorama.crop[0], 1e-6) * 360.0 / max(p.panorama.hfov, 1e-6)))
    if p.panorama.projection == "equirectangular":
        full_h = full_w // 2
        canvas_h = int(round(height / max(p.panorama.crop[3] - p.panorama.crop[1], 1e-6)))
        canvas_w = int(round(width / max(p.panorama.crop[2] - p.panorama.crop[0], 1e-6)))
        top_canvas = (full_h - canvas_h * (p.panorama.vfov / 180.0) * full_h / max(canvas_h, 1)) / 2
        top = int(round((90 - p.panorama.vfov / 2) / 180 * full_h + p.panorama.crop[1] * canvas_h))
        left = int(round((full_w - canvas_w) / 2 + p.panorama.crop[0] * canvas_w))
    else:
        full_h = height
        top, left = 0, int(round((full_w - width) / 2))
    heading = float(p.metadata.get("compass", 0.0)) % 360
    return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
   <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
   <GPano:ProjectionType>{p.panorama.projection}</GPano:ProjectionType>
   <GPano:PoseHeadingDegrees>{heading:.1f}</GPano:PoseHeadingDegrees>
   <GPano:CroppedAreaImageWidthPixels>{width}</GPano:CroppedAreaImageWidthPixels>
   <GPano:CroppedAreaImageHeightPixels>{height}</GPano:CroppedAreaImageHeightPixels>
   <GPano:FullPanoWidthPixels>{full_w}</GPano:FullPanoWidthPixels>
   <GPano:FullPanoHeightPixels>{full_h}</GPano:FullPanoHeightPixels>
   <GPano:CroppedAreaLeftPixels>{max(0, left)}</GPano:CroppedAreaLeftPixels>
   <GPano:CroppedAreaTopPixels>{max(0, top)}</GPano:CroppedAreaTopPixels>
   <GPano:StitchingSoftware>{SOFTWARE}</GPano:StitchingSoftware>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def inject_jpeg_xmp(path: str, xmp: str):
    """Insert an XMP APP1 segment into an existing JPEG file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        return
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp.encode("utf-8")
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    # place after existing APP0/APP1 segments
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] in (0xE0, 0xE1):
        ln = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        pos += 2 + ln
    with open(path, "wb") as f:
        f.write(data[:pos] + seg + data[pos:])


# --------------------------------------------------------------------------- cube faces

CUBE_FACES = {
    # name: (yaw, pitch)
    "front": (0, 0), "right": (90, 0), "back": (180, 0), "left": (-90, 0), "up": (0, 90), "down": (0, -90),
}


def equirect_to_cube(eq: np.ndarray, size: int | None = None, interp=cv2.INTER_CUBIC) -> dict:
    H, W = eq.shape[:2]
    size = size or max(16, int(round(W / math.pi)))
    t = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(t, -t)
    rays = G.normalize(np.stack([X, Y, np.ones_like(X)], -1))
    out = {}
    for name, (yaw, pitch) in CUBE_FACES.items():
        R = G.rotation_matrix(yaw, pitch, 0)
        d = rays @ R.T
        lon = np.arctan2(d[..., 0], d[..., 2])
        lat = np.arcsin(np.clip(d[..., 1], -1, 1))
        mx = ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32)
        my = ((0.5 - lat / math.pi) * H - 0.5).astype(np.float32)
        out[name] = cv2.remap(eq, mx, my, interp, borderMode=cv2.BORDER_WRAP)
    return out


def cube_to_equirect(faces: dict, width: int, interp=cv2.INTER_CUBIC) -> np.ndarray:
    H = width // 2
    lon = ((np.arange(width) + 0.5) / width - 0.5) * 2 * math.pi
    lat = (0.5 - (np.arange(H) + 0.5) / H) * math.pi
    L, B = np.meshgrid(lon, lat)
    d = G._dir_from_lonlat(L, B)
    any_face = next(iter(faces.values()))
    out = np.zeros((H, width) + any_face.shape[2:], any_face.dtype)
    size = any_face.shape[0]
    best = np.full((H, width), -np.inf)
    for name, (yaw, pitch) in CUBE_FACES.items():
        R = G.rotation_matrix(yaw, pitch, 0)
        c = d @ R  # world -> face camera
        z = c[..., 2]
        sel = z > best
        with np.errstate(divide="ignore", invalid="ignore"):
            u = c[..., 0] / z
            v = -c[..., 1] / z
        mx = ((u + 1) / 2 * size - 0.5).astype(np.float32)
        my = ((v + 1) / 2 * size - 0.5).astype(np.float32)
        img = cv2.remap(faces[name], mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
        out[sel] = img[sel]
        best[sel] = z[sel]
    return out


def save_cube_faces(eq: np.ndarray, out_base: str, ext: str = ".jpg", size: int | None = None, strip: bool = False,
                    naming: dict | None = None, **save_kw) -> list:
    from .imageio import save_image

    faces = equirect_to_cube(eq, size)
    paths = []
    if strip:
        order = ["front", "right", "back", "left", "up", "down"]
        img = np.concatenate([faces[k] for k in order], axis=1)
        path = f"{out_base}_cubestrip{ext}"
        save_image(path, img, **save_kw)
        return [path]
    naming = naming or {}
    for name, img in faces.items():
        path = f"{out_base}_{naming.get(name, name)}{ext}"
        save_image(path, img, **save_kw)
        paths.append(path)
    return paths


def reproject(img: np.ndarray, src_proj: str, src_hfov: float, dst_proj: str, dst_hfov: float, dst_vfov: float,
              width: int, yaw=0.0, pitch=0.0, roll=0.0, interp=cv2.INTER_CUBIC, **proj_params) -> np.ndarray:
    """Re-project an already stitched panorama into another projection (e.g. little planet)."""
    S = G.make_projection(src_proj)
    D = G.make_projection(dst_proj, **proj_params)
    H0, W0 = img.shape[:2]
    s_scale = W0 / (2 * S.u_of_hfov(math.radians(src_hfov) / 2))
    du = 2 * D.u_of_hfov(math.radians(dst_hfov) / 2)
    dv = 2 * D.v_of_vfov(math.radians(dst_vfov) / 2)
    d_scale = width / du
    height = max(1, int(round(dv * d_scale)))
    xs = (np.arange(width) + 0.5 - width / 2) / d_scale
    ys = (height / 2 - (np.arange(height) + 0.5)) / d_scale
    U, V = np.meshgrid(xs, ys)
    d, valid = D.inverse(U, V)
    R = G.rotation_matrix(yaw, pitch, roll)
    d = d @ R.T  # view -> world (row vectors): the output centre looks along R @ (0,0,1)
    u, v, ok = S.forward(d)
    mx = (u * s_scale + W0 / 2 - 0.5).astype(np.float32)
    my = (H0 / 2 - v * s_scale - 0.5).astype(np.float32)
    border = cv2.BORDER_WRAP if src_proj == "equirectangular" and src_hfov >= 359.9 else cv2.BORDER_CONSTANT
    out = cv2.remap(img, mx, my, interp, borderMode=border)
    out[~(valid & ok)] = 0
    return out


def little_planet(eq: np.ndarray, width: int = 2000, fov: float = 300.0, rotate: float = 0.0) -> np.ndarray:
    """Stereographic 'little planet' from a 360x180 equirectangular image."""
    return reproject(eq, "equirectangular", 360.0, "stereographic", fov, fov, width, yaw=rotate, pitch=-90.0)
