"""Local web server behind the PanoForge editor (single user, single open project)."""
from __future__ import annotations

import io
import json
import math
import os
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import geometry as G
from .imageio import is_image_file, load_image
from .project import CP_NORMAL, ControlPoint, Project

STATIC = os.path.join(os.path.dirname(__file__), "web")
BOOT_ID = uuid.uuid4().hex[:8]  # part of image URLs so browsers never reuse pixels from an older server
LENSDB_PATH = os.path.join(os.path.expanduser("~"), ".panoforge", "lensdb.json")


# --------------------------------------------------------------------------- session state


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.events: list[dict] = []
        self.done = False
        self.error: str | None = None
        self.result = None
        self.cancel = False
        self.cond = threading.Condition()

    def emit(self, **ev):
        with self.cond:
            ev.setdefault("t", time.time())
            self.events.append(ev)
            self.cond.notify_all()

    def finish(self, result=None, error=None):
        with self.cond:
            self.result, self.error, self.done = result, error, True
            self.events.append({"type": "done", "error": error, "result": result})
            self.cond.notify_all()


class Session:
    def __init__(self):
        self.project = Project()
        self.lock = threading.RLock()
        self.undo: list[str] = []
        self.redo: list[str] = []
        self.dirty = False
        self.jobs: dict[str, Job] = {}
        self.cache: OrderedDict = OrderedDict()  # image cache: (path, kind, size) -> bytes
        self.cache_bytes = 0
        self.version = 0

    # --- undo/redo
    def snapshot(self) -> str:
        return json.dumps(self.project.to_dict())

    def push_undo(self, label: str = ""):
        self.undo.append(self.snapshot())
        if len(self.undo) > 100:
            self.undo.pop(0)
        self.redo.clear()
        self.dirty = True
        self.version += 1

    def restore(self, snap: str):
        path = self.project.path
        self.project = Project.from_dict(json.loads(snap))
        self.project.path = path
        self.version += 1

    # --- byte cache for encoded images
    def cached(self, key, make):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        data = make()
        with self.lock:
            self.cache[key] = data
            self.cache_bytes += len(data)
            while self.cache_bytes > 768 * 2 ** 20 and len(self.cache) > 1:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= len(old)
        return data


def _finite(o):
    """Replace inf/nan (which JSON cannot carry) by null, recursively."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.generic):
        return _finite(o.item())
    return o


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_finite(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


JSONResponse = SafeJSONResponse  # noqa: F811  (every response below goes through the safe encoder)

S = Session()
app = FastAPI(title="PanoForge", default_response_class=SafeJSONResponse)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    """Report unexpected errors with their message (the editor shows it) instead of a bare 500."""
    traceback.print_exception(exc)
    return SafeJSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


@app.middleware("http")
async def _no_stale_assets(request: Request, call_next):
    """App files are revalidated on every load (ETag), so an updated editor is never served stale."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _project_json():
    p = S.project
    d = p.to_dict()
    d["path"] = p.path
    d["dirty"] = S.dirty
    d["rev"] = S.version  # session revision (d["version"] is the file format version)
    d["boot"] = BOOT_ID
    d["can_undo"], d["can_redo"] = bool(S.undo), bool(S.redo)
    # derived values the UI needs
    for i, im in enumerate(d["images"]):
        try:
            cam = p.camera(i)
            im["focal_px"] = cam.focal_px
            im["radial_limit"] = min(cam.radial_limit(), 1e9) if (cam.a or cam.b or cam.c) else 1e9
        except Exception:
            im["focal_px"] = 0
        im["abspath"] = p.abspath(p.images[i].path)
        im["name"] = os.path.basename(p.images[i].path)
        im["n_cps"] = sum(1 for cp in p.control_points if cp.i == i or cp.j == i)
    try:
        from .align import optimum_size

        d["optimum_size"] = optimum_size(p) if p.images else [0, 0]
    except Exception:
        d["optimum_size"] = [0, 0]
    return d


def _ok(**kw):
    kw.setdefault("project", _project_json())
    return JSONResponse(kw)


def _img_entry(i):
    p = S.project
    if i < 0 or i >= len(p.images):
        raise HTTPException(404, "no such image")
    return p.images[i]


def _encode_jpeg(rgb: np.ndarray, quality=90) -> bytes:
    a = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1]
    ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _source_scaled(i: int, max_side: int) -> np.ndarray:
    im = _img_entry(i)
    li = load_image(S.project.abspath(im.path))
    rgb = li.rgb
    s = min(1.0, max_side / max(rgb.shape[:2]))
    if s < 1.0:
        rgb = cv2.resize(rgb, (max(1, int(round(rgb.shape[1] * s))), max(1, int(round(rgb.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    if li.is_hdr:  # display mapping for HDR sources (decided by the file, not by pixel values)
        from .imageio import linear_to_srgb

        rgb = linear_to_srgb(rgb / (1 + rgb))
    return np.clip(rgb, 0, 1)


# --------------------------------------------------------------------------- static / index


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html")) as f:
        return f.read()


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


# --------------------------------------------------------------------------- project


@app.get("/api/project")
def get_project():
    return JSONResponse(_project_json())


@app.post("/api/project/new")
def new_project():
    with S.lock:
        S.project = Project()
        S.undo.clear()
        S.redo.clear()
        S.dirty = False
        S.version += 1
    return _ok()


@app.post("/api/project/open")
def open_project(body: dict = Body(...)):
    path = body.get("path", "")
    if not os.path.exists(path):
        raise HTTPException(404, f"not found: {path}")
    with S.lock:
        try:
            S.project = Project.load(path)
        except NotImplementedError as e:
            raise HTTPException(400, str(e))
        S.undo.clear()
        S.redo.clear()
        S.dirty = False
        S.version += 1
    return _ok()


@app.post("/api/project/save")
def save_project(body: dict = Body(default={})):
    path = body.get("path") or S.project.path
    if not path:
        raise HTTPException(400, "no path")
    if not path.lower().endswith((".pfproj", ".pto")):
        path += ".pfproj"
    with S.lock:
        if path.lower().endswith(".pto"):
            from .ptgui_format import export_pto

            export_pto(S.project, path)
        else:
            S.project.save(path)
            S.dirty = False
    return _ok(saved=path)


@app.post("/api/project/save_template")
def save_template(body: dict = Body(...)):
    path = body["path"]
    tpl = S.project.copy()
    tpl.control_points = []
    tpl.metadata["template_behavior"] = body.get("behavior", {})
    tpl.save(path if path.endswith(".pfproj") else path + ".pfproj")
    return _ok(saved=path)


@app.post("/api/project/apply_template")
def apply_template_ep(body: dict = Body(...)):
    from .templates import apply_template

    tpl = Project.load(body["path"])
    with S.lock:
        S.push_undo("apply template")
        apply_template(S.project, tpl)
    return _ok()


@app.post("/api/undo")
def undo():
    with S.lock:
        if S.undo:
            S.redo.append(S.snapshot())
            S.restore(S.undo.pop())
            S.dirty = True
    return _ok()


@app.post("/api/redo")
def redo():
    with S.lock:
        if S.redo:
            S.undo.append(S.snapshot())
            S.restore(S.redo.pop())
            S.dirty = True
    return _ok()


def _set_fields(obj, values: dict):
    for k, v in values.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _set_fields(cur, v)
        else:
            setattr(obj, k, v)


@app.patch("/api/project")
def patch_project(body: dict = Body(...)):
    """Partial update: {"lenses": {idx: {...}}, "images": {idx: {...}}, "panorama": {...}, ...}"""
    p = S.project
    with S.lock:
        if not body.pop("_no_undo", False):
            S.push_undo("edit")
        for key in ("panorama", "exposure", "output", "optimizer"):
            if key in body:
                _set_fields(getattr(p, key), body[key])
        for idx, vals in (body.get("lenses") or {}).items():
            _set_fields(p.lenses[int(idx)], vals)
        for idx, vals in (body.get("images") or {}).items():
            _set_fields(p.images[int(idx)], vals)
        if "metadata" in body:
            p.metadata.update(body["metadata"])
        S.version += 1
    return _ok()


# --------------------------------------------------------------------------- files / images


@app.get("/api/fs/list")
def fs_list(path: str = "", filter: str = "images"):
    path = os.path.expanduser(path or "~")
    if os.path.isfile(path):
        path = os.path.dirname(path)
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except OSError as e:
        raise HTTPException(400, str(e))
    entries = []
    for n in names:
        if n.startswith("."):
            continue
        full = os.path.join(path, n)
        if os.path.isdir(full):
            entries.append({"name": n, "dir": True})
        elif filter == "all" or (filter == "images" and is_image_file(n)) or (filter == "projects" and n.lower().endswith((".pfproj", ".pto", ".pts"))):
            entries.append({"name": n, "dir": False, "size": os.path.getsize(full)})
    return {"path": os.path.abspath(path), "parent": os.path.dirname(os.path.abspath(path)), "entries": entries}


# Native dialogs: the desktop app installs a provider (pywebview); in a browser on macOS the
# server, which runs on the same machine, shows the system dialog through AppleScript.
DIALOG_PROVIDER = None  # callable(kind, title, multi, filter, default_name, directory) -> list[str] | None

_TYPES = {
    "images": ["public.image", "com.adobe.raw-image", "public.camera-raw-image", "org.openexr.exr", "com.adobe.photoshop-image"],
    "projects": ["pfproj", "pto", "pts"],
}


def _osascript_dialog(kind, title, multi, filter, default_name, directory):
    import subprocess

    def q(x):
        return '"' + str(x).replace("\\", "\\\\").replace('"', '\\"') + '"'

    loc = f" default location (POSIX file {q(directory)})" if directory and os.path.isdir(directory) else ""
    if kind == "save":
        script = f"POSIX path of (choose file name with prompt {q(title)} default name {q(default_name or 'untitled')}{loc})"
    elif kind == "folder":
        script = f"POSIX path of (choose folder with prompt {q(title)}{loc})"
    else:
        types = _TYPES.get(filter)
        tclause = " of type {" + ", ".join(q(t) for t in types) + "}" if types else ""
        script = (f"set fs to choose file with prompt {q(title)}{tclause}{loc}" + (" with multiple selections allowed" if multi else "") + "\n"
                  "if class of fs is not list then set fs to {fs}\n"
                  "set out to \"\"\n"
                  "repeat with f in fs\n set out to out & POSIX path of f & linefeed\nend repeat\nreturn out")
    script = "activate\n" + script
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:  # cancelled (-128) or no GUI session
        return None
    return [line for line in r.stdout.splitlines() if line.strip()]


@app.post("/api/dialog")
def native_dialog(body: dict = Body(...)):
    import sys

    args = (body.get("kind", "open"), body.get("title", "Open"), bool(body.get("multi")), body.get("filter", "all"),
            body.get("default_name", ""), body.get("directory") or (os.path.dirname(S.project.path) if S.project.path else ""))
    if DIALOG_PROVIDER is not None:
        paths = DIALOG_PROVIDER(*args)
    elif sys.platform == "darwin":
        paths = _osascript_dialog(*args)
    else:
        return {"unsupported": True}
    return {"paths": paths or []}


RECOVERY_PATH = os.path.join(os.path.expanduser("~"), ".panoforge", "recovery.pfproj")


@app.get("/api/recovery")
def recovery_info():
    if not os.path.exists(RECOVERY_PATH):
        return {"exists": False}
    try:
        with open(RECOVERY_PATH) as f:
            d = json.load(f)
        return {"exists": True, "images": len(d.get("images", [])), "saved": os.path.getmtime(RECOVERY_PATH),
                "from": (d.get("metadata") or {}).get("recovered_from") or ""}
    except Exception:
        return {"exists": False}


@app.post("/api/recovery/restore")
def recovery_restore():
    with S.lock:
        p = Project.load(RECOVERY_PATH)
        p.path = p.metadata.pop("recovered_from", "") or ""
        S.project = p
        S.undo.clear()
        S.redo.clear()
        S.dirty = True
        S.version += 1
    os.remove(RECOVERY_PATH)
    return _ok()


@app.post("/api/recovery/discard")
def recovery_discard():
    if os.path.exists(RECOVERY_PATH):
        os.remove(RECOVERY_PATH)
    return {"ok": True}


@app.post("/api/images/add")
def add_images(body: dict = Body(...)):
    from .align import create_project

    paths = [p for p in body.get("paths", []) if os.path.isfile(p)]
    if not paths:
        raise HTTPException(400, "no images")
    with S.lock:
        S.push_undo("add images")
        p = S.project
        new = create_project(paths, lens_type=body.get("lens_type"))
        # merge: reuse an existing lens when the camera/lens/size match
        lens_map = {}
        for li, ln in enumerate(new.lenses):
            match = None
            for lj, old in enumerate(p.lenses):
                if (old.camera, round(old.focal_mm, 1), old.lens_type) == (ln.camera, round(ln.focal_mm, 1), ln.lens_type):
                    match = lj
                    break
            if match is None:
                p.lenses.append(ln)
                match = len(p.lenses) - 1
            lens_map[li] = match
        base_group = max([im.link_group for im in p.images] + [-1]) + 1
        for im in new.images:
            im.lens = lens_map[im.lens]
            if im.link_group >= 0:
                im.link_group += base_group
            p.images.append(im)
        if new.exposure.hdr_mode:
            p.exposure.hdr_mode = True
        if not p.path and p.images:
            first = p.images[0].path
            p.path = os.path.join(os.path.dirname(first), os.path.splitext(os.path.basename(first))[0] + " Panorama.pfproj")
        if new.optimizer.optimize_fisheye_factor:
            p.optimizer.optimize_fisheye_factor = True
            p.optimizer.distortion = new.optimizer.distortion
    return _ok()


@app.post("/api/images/upload")
async def upload_images(request: Request):
    """Drag-and-drop from the browser: files are copied next to the project (or to ~/PanoForge Imports)."""
    form = await request.form()
    dest = os.path.dirname(S.project.path) if S.project.path else os.path.join(os.path.expanduser("~"), "PanoForge Imports")
    os.makedirs(dest, exist_ok=True)
    saved = []
    for _, f in form.multi_items():
        if not isinstance(f, UploadFile) and not hasattr(f, "filename"):
            continue
        name = os.path.basename(f.filename or "image.jpg")
        target = os.path.join(dest, name)
        k = 1
        while os.path.exists(target):
            target = os.path.join(dest, f"{os.path.splitext(name)[0]}_{k}{os.path.splitext(name)[1]}")
            k += 1
        with open(target, "wb") as out:
            out.write(await f.read())
        saved.append(target)
    return add_images({"paths": saved})


@app.post("/api/images/remove")
def remove_images(body: dict = Body(...)):
    idx = sorted(set(int(i) for i in body.get("indices", [])), reverse=True)
    with S.lock:
        S.push_undo("remove images")
        p = S.project
        keep = [i for i in range(len(p.images)) if i not in idx]
        remap = {old: new for new, old in enumerate(keep)}
        p.images = [p.images[i] for i in keep]
        p.control_points = [cp for cp in p.control_points if cp.i in remap and cp.j in remap]
        for cp in p.control_points:
            cp.i, cp.j = remap[cp.i], remap[cp.j]
        p.optimizer.anchor = remap.get(p.optimizer.anchor, 0)
        used = sorted({im.lens for im in p.images})
        lmap = {old: new for new, old in enumerate(used)}
        p.lenses = [p.lenses[l] for l in used]
        for im in p.images:
            im.lens = lmap[im.lens]
    return _ok()


@app.post("/api/images/reorder")
def reorder_images(body: dict = Body(...)):
    order = [int(i) for i in body["order"]]
    with S.lock:
        p = S.project
        if sorted(order) != list(range(len(p.images))):
            raise HTTPException(400, "bad order")
        S.push_undo("reorder")
        remap = {old: new for new, old in enumerate(order)}
        p.images = [p.images[i] for i in order]
        for cp in p.control_points:
            cp.i, cp.j = remap[cp.i], remap[cp.j]
        p.optimizer.anchor = remap.get(p.optimizer.anchor, 0)
    return _ok()


@app.post("/api/images/link")
def link_images(body: dict = Body(...)):
    idx = [int(i) for i in body.get("indices", [])]
    with S.lock:
        S.push_undo("link")
        p = S.project
        if body.get("unlink"):
            for i in idx:
                p.images[i].link_group = -1
        elif body.get("auto"):
            from .align import link_brackets

            for im in p.images:
                im.link_group = -1
            link_brackets(p)
        else:
            g = max([im.link_group for im in p.images] + [-1]) + 1
            for i in idx:
                p.images[i].link_group = g
            ref = p.images[idx[0]]
            for i in idx[1:]:
                p.images[i].yaw, p.images[i].pitch, p.images[i].roll = ref.yaw, ref.pitch, ref.roll
    return _ok()


@app.post("/api/images/assign_lens")
def assign_lens(body: dict = Body(...)):
    """Give the selected images their own (new) lens, or link them to an existing one."""
    import copy

    idx = [int(i) for i in body.get("indices", [])]
    with S.lock:
        S.push_undo("lens assignment")
        p = S.project
        if body.get("lens") is None:
            p.lenses.append(copy.deepcopy(p.lenses[p.images[idx[0]].lens]))
            lid = len(p.lenses) - 1
        else:
            lid = int(body["lens"])
        for i in idx:
            p.images[i].lens = lid
    return _ok()


def _image_bytes(i: int, kind: str, size: int) -> bytes:
    im = _img_entry(i)
    key = (S.project.abspath(im.path), kind, size)

    def make():
        return _encode_jpeg(_source_scaled(i, size), 88 if size <= 512 else 92)

    return S.cached(key, make)


@app.get("/api/image/{i}/thumb")
def image_thumb(i: int, size: int = 256):
    return Response(_image_bytes(i, "thumb", min(size, 1024)), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/api/image/{i}/texture")
def image_texture(i: int, size: int = 2048):
    return Response(_image_bytes(i, "tex", min(size, 8192)), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/api/image/{i}/full")
def image_full(i: int):
    return Response(_image_bytes(i, "full", 1 << 20), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/image/{i}/exif")
def image_exif(i: int):
    return JSONResponse(_img_entry(i).exif)


@app.get("/api/image/{i}/mask")
def get_mask(i: int):
    from .masks import load_paint_mask

    im = _img_entry(i)
    m = load_paint_mask(S.project, i, im.width, im.height)
    rgba = np.zeros((im.height, im.width, 4), np.uint8)
    if m is not None:
        rgba[m == 1] = (0, 0, 255, 255)  # BGR(A) red
        rgba[m == 2] = (0, 255, 0, 255)
    ok, buf = cv2.imencode(".png", rgba)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.put("/api/image/{i}/mask")
async def put_mask(i: int, request: Request):
    """Body: PNG (any size) painted red (exclude) / green (include) on transparent."""
    from .masks import save_paint_mask

    im = _img_entry(i)
    data = await request.body()
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise HTTPException(400, "bad png")
    if a.shape[1] != im.width or a.shape[0] != im.height:
        a = cv2.resize(a, (im.width, im.height), interpolation=cv2.INTER_NEAREST)
    m = np.zeros(a.shape[:2], np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        alpha = a[..., 3] > 127
        m[alpha & (a[..., 2] > 127) & (a[..., 1] < 127)] = 1
        m[alpha & (a[..., 1] > 127) & (a[..., 2] < 127)] = 2
    with S.lock:
        S.push_undo("mask")
        if not m.any():
            S.project.images[i].mask = ""
        else:
            base = os.path.dirname(S.project.path) if S.project.path else os.path.dirname(S.project.abspath(im.path))
            mdir = os.path.join(base, "masks")
            os.makedirs(mdir, exist_ok=True)
            save_paint_mask(S.project, i, m, os.path.join(mdir, os.path.splitext(os.path.basename(im.path))[0] + f"_{i}.png"))
    return _ok()


@app.post("/api/image/{i}/detect_crop")
def detect_crop(i: int):
    from .lens_detect import detect_image_circle

    im = _img_entry(i)
    c = detect_image_circle(load_image(S.project.abspath(im.path)).rgb)
    if c is None:
        raise HTTPException(404, "no image circle found")
    with S.lock:
        S.push_undo("crop")
        ln = S.project.lenses[im.lens]
        ln.crop.mode = "circle"
        ln.crop.cx, ln.crop.cy, ln.crop.radius = c[0] + 0.5, c[1] + 0.5, c[2] * 0.985
    return _ok(circle=c)


# --------------------------------------------------------------------------- control points


@app.post("/api/cps/add")
def add_cp(body: dict = Body(...)):
    with S.lock:
        S.push_undo("add cp")
        cp = ControlPoint(int(body["i"]), float(body["x1"]), float(body["y1"]), int(body["j"]), float(body["x2"]), float(body["y2"]),
                          body.get("kind", CP_NORMAL), int(body.get("line", 0)))
        S.project.control_points.append(cp)
        from .optimizer import cp_errors

        cp_errors(S.project)
    return _ok(index=len(S.project.control_points) - 1)


@app.patch("/api/cps/{k}")
def edit_cp(k: int, body: dict = Body(...)):
    with S.lock:
        if not body.pop("_no_undo", False):
            S.push_undo("move cp")
        cp = S.project.control_points[k]
        for key in ("x1", "y1", "x2", "y2", "kind", "line"):
            if key in body:
                setattr(cp, key, body[key] if key in ("kind",) else (int(body[key]) if key == "line" else float(body[key])))
        from .optimizer import cp_errors

        cp_errors(S.project)
    return _ok()


@app.post("/api/cps/delete")
def delete_cps(body: dict = Body(...)):
    ks = set(int(k) for k in body.get("indices", []))
    with S.lock:
        S.push_undo("delete cps")
        if body.get("all"):
            S.project.control_points = []
        elif body.get("pair") is not None:
            i, j = body["pair"]
            S.project.control_points = [cp for cp in S.project.control_points if {cp.i, cp.j} != {i, j}]
        else:
            S.project.control_points = [cp for n, cp in enumerate(S.project.control_points) if n not in ks]
    return _ok()


@app.post("/api/cps/delete_worst")
def delete_worst(body: dict = Body(default={})):
    from .optimizer import prune_control_points

    with S.lock:
        S.push_undo("delete worst")
        n = prune_control_points(S.project, max_error=body.get("max_error"), worst_fraction=body.get("fraction"))
    return _ok(removed=n)


@app.post("/api/cps/predict")
def predict_cp(body: dict = Body(...)):
    """Where does point (x, y) of image i appear in image j with the current alignment?"""
    p = S.project
    ci, cj = p.camera(int(body["i"])), p.camera(int(body["j"]))
    d = ci.pixel_to_world(np.array([float(body["x"])]), np.array([float(body["y"])]))
    x, y, ok = cj.world_to_pixel(d)
    inside = bool(ok[0]) and -0.5 <= x[0] <= cj.width - 0.5 and -0.5 <= y[0] <= cj.height - 0.5
    return {"x": float(x[0]), "y": float(y[0]), "valid": inside}


@app.post("/api/cps/finetune")
def finetune_cp(body: dict = Body(...)):
    """Refine the position in image j by correlation with a geometry-corrected patch of image i."""
    from .cpfinetune import finetune

    r = finetune(S.project, int(body["i"]), float(body["x1"]), float(body["y1"]), int(body["j"]), float(body["x2"]), float(body["y2"]),
                 radius=int(body.get("radius", 24)), search=int(body.get("search", 40)))
    return r


# --------------------------------------------------------------------------- long running jobs


def _start_job(kind, fn):
    job = Job(kind)
    S.jobs[job.id] = job

    def progress(stage, i, n, **kw):
        if job.cancel:
            raise RuntimeError("cancelled")
        job.emit(type="progress", stage=stage, i=i, n=n, **{k: v for k, v in kw.items() if isinstance(v, (str, int, float))})

    def run():
        try:
            res = fn(progress)
            job.finish(result=res)
        except Exception as e:  # report to the UI
            traceback.print_exc()
            job.finish(error=f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return job


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = S.jobs.get(jid)
    if not job:
        raise HTTPException(404)
    return {"id": job.id, "kind": job.kind, "done": job.done, "error": job.error, "result": job.result,
            "last": job.events[-1] if job.events else None}


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    job = S.jobs.get(jid)
    if job:
        job.cancel = True
    return {"ok": True}


@app.get("/api/jobs/{jid}/events")
def job_events(jid: str):
    job = S.jobs.get(jid)
    if not job:
        raise HTTPException(404)

    def gen():
        sent = 0
        while True:
            with job.cond:
                while sent >= len(job.events) and not job.done:
                    job.cond.wait(timeout=15)
                    if sent >= len(job.events):
                        yield ": keepalive\n\n"
                evs = job.events[sent:]
                sent = len(job.events)
                done = job.done
            for ev in evs:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            if done and sent >= len(job.events):
                return

    return StreamingResponse(gen(), media_type="text/event-stream")


def _work_on_copy(fn):
    """Run fn on a copy of the project, then commit it atomically (undoable)."""

    def job(progress):
        with S.lock:
            work = S.project.copy()
            work.path = S.project.path
        res = fn(work, progress)
        with S.lock:
            S.push_undo("job")
            S.project = work
        return res

    return job


@app.post("/api/align")
def align(body: dict = Body(default={})):
    from .align import align_images

    def fn(p, progress):
        rep = align_images(p, generate=body.get("generate", True), prealigned=body.get("prealigned", False),
                           max_cps_per_pair=int(body.get("max_cps", 30)), progress=progress,
                           photometric=body.get("photometric", True))
        return {"connected": rep.connected, "n_images": rep.n_images, "n_cps": rep.n_cps, "mean_error": rep.mean_error,
                "max_error": rep.max_error, "unconnected": rep.unconnected}

    return {"job": _start_job("align", _work_on_copy(fn)).id}


@app.post("/api/cps/generate")
def generate_cps(body: dict = Body(default={})):
    from .features import generate_control_points

    def fn(p, progress):
        idx = body.get("indices")
        res = generate_control_points(p, idx, replace=body.get("replace", True), prealigned=body.get("prealigned", False),
                                      max_cps_per_pair=int(body.get("max_cps", 30)), progress=progress)
        return {"pairs": sum(1 for r in res if r.cps), "cps": sum(len(r.cps) for r in res)}

    return {"job": _start_job("cps", _work_on_copy(fn)).id}


@app.post("/api/optimize")
def run_optimize(body: dict = Body(default={})):
    from .optimizer import optimize

    def fn(p, progress):
        progress("optimize", 0, 1)
        if body.get("reinit"):
            from .align import align_images

            align_images(p, generate=False, progress=progress, photometric=False)
        r = optimize(p, fixed_lens=body.get("fixed_lens", False))
        return {"mean_error": r.mean_error, "max_error": r.max_error, "rms_error": r.rms_error, "n_cps": r.n_cps,
                "n_vars": r.n_vars, "message": r.message}

    return {"job": _start_job("optimize", _work_on_copy(fn)).id}


@app.post("/api/photometric")
def run_photometric(body: dict = Body(default={})):
    from .photometric import optimize_photometric

    def fn(p, progress):
        progress("exposure", 0, 1)
        return optimize_photometric(p)

    return {"job": _start_job("photometric", _work_on_copy(fn)).id}


@app.post("/api/stitch")
def run_stitch(body: dict = Body(default={})):
    from .stitcher import stitch

    with S.lock:
        p = S.project.copy()
        p.path = S.project.path
    if body.get("output"):
        _set_fields(p.output, body["output"])
    out = body.get("path") or None

    def fn(progress):
        r = stitch(p, out, width=body.get("width") or None, progress=progress)
        return {"path": r.path, "width": r.width, "height": r.height, "seconds": r.seconds, "extra": r.extra_files,
                "projection": p.panorama.projection, "hfov": p.panorama.hfov, "vfov": p.panorama.vfov}

    return {"job": _start_job("stitch", fn).id}


@app.post("/api/batch")
def run_batch(body: dict = Body(...)):
    """Batch stitcher: stitch a list of project files one after another."""
    from .align import align_images
    from .stitcher import stitch

    paths = body.get("paths", [])

    def fn(progress):
        done = []
        for k, path in enumerate(paths):
            progress("project", k, len(paths), message=os.path.basename(path))
            p = Project.load(path)
            if body.get("align"):
                align_images(p, prealigned=bool(body.get("prealigned")), progress=progress)
                p.save(path)
            r = stitch(p, progress=progress)
            done.append(r.path)
        return {"outputs": done}

    return {"job": _start_job("batch", fn).id}


# --------------------------------------------------------------------------- geometry edits


@app.post("/api/transform")
def numerical_transform(body: dict = Body(...)):
    """Rotate the whole panorama (or selected images): yaw/pitch/roll in degrees, or a 3x3 matrix."""
    from .optimizer import rotate_panorama

    with S.lock:
        if not body.get("_no_undo"):
            S.push_undo("transform")
        if "matrix" in body:
            R = np.array(body["matrix"], dtype=np.float64).reshape(3, 3)
        else:
            R = G.rotation_matrix(float(body.get("yaw", 0)), float(body.get("pitch", 0)), float(body.get("roll", 0)))
        idx = body.get("images")
        if idx is not None:
            idx = set(int(i) for i in idx)
            groups = S.project.link_groups()
            for i in list(idx):
                g = S.project.images[i].link_group
                if g >= 0:
                    idx |= set(groups[g])
        rotate_panorama(S.project, R, idx)
    return _ok()


@app.post("/api/straighten")
def straighten_ep(body: dict = Body(default={})):
    from .align import straighten
    from .optimizer import level_with_lines

    with S.lock:
        S.push_undo("straighten")
        if not level_with_lines(S.project):
            straighten(S.project)
    return _ok()


@app.post("/api/fit")
def fit_ep(body: dict = Body(default={})):
    from .align import center_panorama, choose_projection, fit_panorama

    with S.lock:
        S.push_undo("fit")
        if body.get("center"):
            center_panorama(S.project, horizontal=body.get("horizontal", True), vertical=body.get("vertical", False))
        if body.get("choose_projection"):
            choose_projection(S.project)
        fit_panorama(S.project)
    return _ok()


@app.post("/api/align_to_grid")
def align_to_grid(body: dict = Body(...)):
    """Position images on a grid (shooting configuration) before control point generation."""
    with S.lock:
        S.push_undo("grid")
        p = S.project
        idx = body.get("indices") or list(range(len(p.images)))
        cols, rows = int(body.get("columns", len(idx))), int(body.get("rows", 1))
        yaw0, yaw_step = float(body.get("yaw0", 0.0)), float(body.get("yaw_step", 360.0 / max(cols, 1)))
        pitch0, pitch_step = float(body.get("pitch0", 0.0)), float(body.get("pitch_step", -30.0))
        by_row = body.get("order", "row") == "row"
        snake = bool(body.get("snake", False))
        for k, i in enumerate(idx):
            r, c = (k // cols, k % cols) if by_row else (k % rows, k // rows)
            if snake and (r if by_row else c) % 2 == 1:
                if by_row:
                    c = cols - 1 - c
                else:
                    r = rows - 1 - r
            p.images[i].yaw = yaw0 + c * yaw_step
            p.images[i].pitch = pitch0 + r * pitch_step
            p.images[i].roll = 0.0
    return _ok()


# --------------------------------------------------------------------------- previews / rendering


@app.get("/api/preview")
def preview(width: int = 1600, seams: str = "voronoi"):
    from .stitcher import preview as do_preview

    with S.lock:
        p = S.project.copy()
        p.path = S.project.path
    r = do_preview(p, width=width, seams=seams)
    rgba = np.dstack([r.image, r.alpha])
    a = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA))
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/render_region")
def render_region(body: dict = Body(...)):
    """Detail viewer: stitch a small window of the final panorama at full output resolution."""
    from .stitcher import stitch

    with S.lock:
        p = S.project.copy()
        p.path = S.project.path
    width = int(body.get("width") or 0) or None
    l, t, r, b = [float(v) for v in body["crop"]]  # fractions of the full canvas
    p.panorama.crop = [max(0, l), max(0, t), min(1, r), min(1, b)]
    res = stitch(p, None, width=width, write=False, seams=body.get("seams", "graphcut"))
    rgba = np.dstack([res.image, res.alpha])
    a = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA))
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/output")
def get_output(path: str):
    if not os.path.isfile(path):
        raise HTTPException(404)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp"):
        return FileResponse(path)
    # TIFF/PSD/EXR: send a JPEG rendition for viewing
    key = (path, "view", int(os.path.getmtime(path)))

    def make():
        if ext in (".psd", ".psb"):
            from .psd import read_psd_composite

            rgb = read_psd_composite(path)
        else:
            li = load_image(path)
            rgb = li.rgb if not li.is_hdr else li.rgb / (1 + li.rgb)
            if li.is_hdr:
                from .imageio import linear_to_srgb

                rgb = linear_to_srgb(rgb)
        s = min(1.0, 8192 / max(rgb.shape[:2]))
        if s < 1:
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return _encode_jpeg(rgb, 92)

    return Response(S.cached(key, make), media_type="image/jpeg")


@app.post("/api/convert")
def convert(body: dict = Body(...)):
    """Tools: cube faces / little planet / reprojection of a stitched equirectangular file."""
    from .imageio import save_image
    from .output import little_planet, reproject, save_cube_faces

    src = body["path"]
    img = load_image(src).rgb
    base = os.path.splitext(src)[0]
    kind = body.get("to", "cube")
    if kind == "cube":
        files = save_cube_faces(img, base, body.get("ext", ".jpg"), strip=bool(body.get("strip")))
    elif kind == "littleplanet":
        out = little_planet(img, int(body.get("width", 3000)), float(body.get("fov", 300)), float(body.get("rotate", 0)))
        files = [base + "_planet.jpg"]
        save_image(files[0], np.clip(out, 0, 1))
    else:
        out = reproject(img, "equirectangular", 360.0, kind, float(body.get("hfov", 120)), float(body.get("vfov", 90)),
                        int(body.get("width", 3000)), yaw=float(body.get("yaw", 0)), pitch=float(body.get("pitch", 0)))
        files = [base + f"_{kind}.jpg"]
        save_image(files[0], np.clip(out, 0, 1))
    return {"files": files}


@app.post("/api/publish")
def publish(body: dict = Body(...)):
    """Publish to website: a self-contained folder with a WebGL 360 viewer + the panorama."""
    from .publish import publish_viewer

    return {"folder": publish_viewer(body["path"], body.get("folder"), body.get("title", "Panorama"))}


@app.post("/api/reveal")
def reveal(body: dict = Body(...)):
    """Show a file in Finder / Explorer."""
    import subprocess
    import sys

    path = body["path"]
    if not os.path.exists(path):
        raise HTTPException(404)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])
    return {"ok": True}


@app.post("/api/batch/build")
def batch_build(body: dict = Body(...)):
    """Batch Builder: one project per panorama found in a folder (grouped by capture time)."""
    from .templates import build_batch

    folder = body["folder"]
    out = body.get("out") or os.path.join(folder, "PanoForge projects")
    tpl = Project.load(body["template"]) if body.get("template") else None
    projects = build_batch(folder, out, tpl, float(body.get("gap", 10.0)), body.get("count"))
    return {"projects": projects}


# --------------------------------------------------------------------------- lens database


def _lensdb():
    if os.path.exists(LENSDB_PATH):
        with open(LENSDB_PATH) as f:
            return json.load(f)
    return []


@app.get("/api/lensdb")
def lensdb_list():
    return _lensdb()


@app.post("/api/lensdb/save")
def lensdb_save(body: dict = Body(...)):
    from dataclasses import asdict

    db = _lensdb()
    ln = S.project.lenses[int(body.get("lens", 0))]
    rec = {"name": body.get("name") or (ln.lens_name or ln.camera or "lens"), "camera": ln.camera, "focal_mm": ln.focal_mm,
           "lens": {k: v for k, v in asdict(ln).items()}}
    db = [r for r in db if r["name"] != rec["name"]] + [rec]
    os.makedirs(os.path.dirname(LENSDB_PATH), exist_ok=True)
    with open(LENSDB_PATH, "w") as f:
        json.dump(db, f, indent=1)
    return db


@app.post("/api/lensdb/apply")
def lensdb_apply(body: dict = Body(...)):
    from .project import Lens

    db = {r["name"]: r for r in _lensdb()}
    rec = db.get(body["name"])
    if not rec:
        raise HTTPException(404)
    with S.lock:
        S.push_undo("lens db")
        lid = int(body.get("lens", 0))
        new = Lens.from_dict(rec["lens"])
        new.locked = True  # a calibrated lens: alignment keeps it instead of re-estimating
        new.init = {}
        S.project.lenses[lid] = new
    return _ok()


# --------------------------------------------------------------------------- run


def run_server(project_path: str | None = None, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    import uvicorn

    if project_path:
        if os.path.isdir(project_path) or is_image_file(project_path):
            files = [project_path] if os.path.isfile(project_path) else sorted(
                os.path.join(project_path, f) for f in os.listdir(project_path) if is_image_file(f))
            add_images({"paths": files})
            S.undo.clear()
        else:
            S.project = Project.load(project_path)
    url = f"http://{host}:{port}/"
    print(f"PanoForge editor: {url}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
