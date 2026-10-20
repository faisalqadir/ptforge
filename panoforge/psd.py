"""Layered Photoshop writer (PSD, and PSB for big panoramas).

Each layer is (name, rgb float 0..1 (h,w,3), alpha float 0..1 (h,w) or None, (left, top)).
Channel data are PackBits (RLE) compressed when ``imagecodecs`` is available.
"""
from __future__ import annotations

import struct

import numpy as np

try:  # fast PackBits (imagecodecs loads codecs lazily: probe them once)
    import imagecodecs

    imagecodecs.packbits_decode(imagecodecs.packbits_encode(b"probe"))

    def _packbits(row: bytes) -> bytes:
        return imagecodecs.packbits_encode(row)
except Exception:  # pragma: no cover - raw (uncompressed) channel data instead
    imagecodecs = None
    _packbits = None


def _to_int(a, bits):
    m = 255 if bits == 8 else 65535
    return np.clip(np.rint(np.asarray(a, np.float32) * m), 0, m).astype(np.uint8 if bits == 8 else ">u2")


def _channel_blob(plane: np.ndarray, psb: bool) -> bytes:
    """Compression code + data for one channel plane (rows)."""
    h = plane.shape[0]
    raw_rows = [plane[r].tobytes() for r in range(h)]
    if _packbits is not None and plane.dtype.itemsize == 1:
        packed = [_packbits(r) for r in raw_rows]
        fmt = ">I" if psb else ">H"
        counts = b"".join(struct.pack(fmt, len(p)) for p in packed)
        rle = struct.pack(">H", 1) + counts + b"".join(packed)
        raw_len = 2 + sum(len(r) for r in raw_rows)
        if len(rle) < raw_len:
            return rle
    return struct.pack(">H", 0) + b"".join(raw_rows)


def _pascal(name: str, pad: int = 4) -> bytes:
    b = name.encode("latin-1", "replace")[:255]
    s = bytes([len(b)]) + b
    while len(s) % pad:
        s += b"\x00"
    return s


def write_psd(path: str, layers, bits: int = 8, width: int | None = None, height: int | None = None,
              composite: tuple | None = None, psb: bool | None = None):
    """Write a layered PSD/PSB.

    layers: list of (name, rgb, alpha, (left, top)).  composite: optional (rgb, alpha) for the
    merged image; when omitted the layers are flattened in order.
    """
    if not layers and composite is None:
        raise ValueError("nothing to write")
    if width is None or height is None:
        width = max(l[3][0] + l[1].shape[1] for l in layers)
        height = max(l[3][1] + l[1].shape[0] for l in layers)
    if psb is None:
        psb = width > 30000 or height > 30000 or path.lower().endswith(".psb")
    depth = 8 if bits == 8 else 16
    L = ">Q" if psb else ">I"

    # ---- layer records + channel data
    records, chan_data = [], []
    for name, rgb, alpha, (left, top) in layers:
        h, w = rgb.shape[:2]
        planes = [(-1, _to_int(alpha if alpha is not None else np.ones((h, w), np.float32), depth))]
        q = _to_int(rgb, depth)
        planes += [(0, q[..., 0]), (1, q[..., 1]), (2, q[..., 2])]
        blobs = [(cid, _channel_blob(np.ascontiguousarray(pl), psb)) for cid, pl in planes]
        rec = struct.pack(">iiii", top, left, top + h, left + w) + struct.pack(">H", len(blobs))
        for cid, blob in blobs:
            rec += struct.pack(">h", cid) + struct.pack(L, len(blob))
        rec += b"8BIMnorm" + bytes([255, 0, 0x08, 0])  # opacity, clipping, flags (pixel data irrelevant bit), filler
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(name)
        rec += struct.pack(">I", len(extra)) + extra
        records.append(rec)
        chan_data.append(b"".join(b for _, b in blobs))
    layer_info = struct.pack(">h", -len(layers) if layers else 0) + b"".join(records) + b"".join(chan_data)
    if len(layer_info) % 2:
        layer_info += b"\x00"
    layer_info_sec = struct.pack(L, len(layer_info)) + layer_info if layers else struct.pack(L, 0)
    lmi = layer_info_sec + struct.pack(">I", 0)  # empty global layer mask info
    lmi_sec = struct.pack(L, len(lmi)) + lmi

    # ---- merged composite
    if composite is None:
        comp = np.zeros((height, width, 3), np.float32)
        ca = np.zeros((height, width), np.float32)
        for _, rgb, alpha, (left, top) in layers:
            h, w = rgb.shape[:2]
            a = alpha if alpha is not None else np.ones((h, w), np.float32)
            sl = (slice(top, top + h), slice(left, left + w))
            comp[sl] = rgb * a[..., None] + comp[sl] * (1 - a[..., None])
            ca[sl] = a + ca[sl] * (1 - a)
        comp = np.where(ca[..., None] > 0, comp / np.maximum(ca, 1e-6)[..., None], 0)
    else:
        comp, ca = composite
        if ca is None:
            ca = np.ones(comp.shape[:2], np.float32)
    q = _to_int(comp, depth)
    qa = _to_int(ca, depth)
    planes = [q[..., 0], q[..., 1], q[..., 2], qa]
    # composite image data: one compression code for all channels
    if _packbits is not None and depth == 8:
        rows = [pl[r].tobytes() for pl in planes for r in range(height)]
        packed = [_packbits(r) for r in rows]
        fmt = ">I" if psb else ">H"
        img = struct.pack(">H", 1) + b"".join(struct.pack(fmt, len(p)) for p in packed) + b"".join(packed)
    else:
        img = struct.pack(">H", 0) + b"".join(np.ascontiguousarray(pl).tobytes() for pl in planes)

    header = b"8BPS" + struct.pack(">H", 2 if psb else 1) + b"\x00" * 6 + struct.pack(">HIIHH", 4, height, width, depth, 3)
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack(">I", 0))  # colour mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(lmi_sec)
        f.write(img)


def read_psd_summary(path: str) -> dict:
    """Minimal reader used by tests: header + layer rectangles/names."""
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"8BPS"
    ver = struct.unpack(">H", d[4:6])[0]
    psb = ver == 2
    ch, h, w, depth, mode = struct.unpack(">HIIHH", d[12:26])
    pos = 26
    pos += 4 + struct.unpack(">I", d[pos:pos + 4])[0]
    pos += 4 + struct.unpack(">I", d[pos:pos + 4])[0]
    L, ls = (">Q", 8) if psb else (">I", 4)
    pos += ls  # layer & mask section length
    li_len = struct.unpack(L, d[pos:pos + ls])[0]
    pos += ls
    layers = []
    if li_len:
        n = abs(struct.unpack(">h", d[pos:pos + 2])[0])
        pos += 2
        for _ in range(n):
            t, l, b, r = struct.unpack(">iiii", d[pos:pos + 16])
            pos += 16
            nc = struct.unpack(">H", d[pos:pos + 2])[0]
            pos += 2 + nc * (2 + ls)
            pos += 12
            el = struct.unpack(">I", d[pos:pos + 4])[0]
            ex = d[pos + 4:pos + 4 + el]
            pos += 4 + el
            ml = struct.unpack(">I", ex[:4])[0]
            bl = struct.unpack(">I", ex[4 + ml:8 + ml])[0]
            nm = ex[8 + ml + bl:]
            name = nm[1:1 + nm[0]].decode("latin-1")
            layers.append(dict(name=name, top=t, left=l, bottom=b, right=r))
    return dict(version=ver, channels=ch, width=w, height=h, depth=depth, layers=layers)


def read_psd_composite(path: str) -> np.ndarray:
    """Merged (composite) RGB image of a PSD/PSB as float 0..1 (raw or RLE, 8/16 bit)."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"8BPS":
        raise ValueError("not a Photoshop file")
    psb = struct.unpack(">H", d[4:6])[0] == 2
    ch, h, w, depth, mode = struct.unpack(">HIIHH", d[12:26])
    pos = 26
    pos += 4 + struct.unpack(">I", d[pos:pos + 4])[0]  # colour mode data
    pos += 4 + struct.unpack(">I", d[pos:pos + 4])[0]  # image resources
    ls = 8 if psb else 4
    pos += ls + struct.unpack(">Q" if psb else ">I", d[pos:pos + ls])[0]  # layer & mask info
    comp = struct.unpack(">H", d[pos:pos + 2])[0]
    pos += 2
    bpp = depth // 8
    dtype = np.uint8 if bpp == 1 else ">u2"
    row_bytes = w * bpp
    planes = []
    if comp == 0:
        n = row_bytes * h
        for c in range(min(ch, 3)):
            planes.append(np.frombuffer(d, dtype, count=w * h, offset=pos + c * n).reshape(h, w))
    elif comp == 1:
        cs = 4 if psb else 2
        counts = np.frombuffer(d, ">u4" if psb else ">u2", count=ch * h, offset=pos).astype(np.int64)
        pos += ch * h * cs
        for c in range(min(ch, 3)):
            rows = []
            for r in range(h):
                n = int(counts[c * h + r])
                rows.append(_unpackbits(d[pos:pos + n], row_bytes))
                pos += n
            planes.append(np.frombuffer(b"".join(rows), dtype).reshape(h, w))
    else:
        raise ValueError(f"unsupported PSD compression {comp}")
    rgb = np.dstack(planes).astype(np.float32) / (255.0 if bpp == 1 else 65535.0)
    return rgb


def _unpackbits(data: bytes, n: int) -> bytes:
    if imagecodecs is not None:
        return imagecodecs.packbits_decode(data)[:n]
    out = bytearray()
    i = 0
    while i < len(data) and len(out) < n:
        c = data[i]
        i += 1
        if c < 128:
            out += data[i:i + c + 1]
            i += c + 1
        elif c > 128:
            out += bytes([data[i]]) * (257 - c)
            i += 1
    return bytes(out[:n])
