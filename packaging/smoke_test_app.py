"""Smoke test a running PanoForge app (or editor server): align and stitch real images, check outputs.

    PANOFORGE_PORT=8799 dist/PanoForge.app/Contents/MacOS/PanoForge &
    .venv/bin/python packaging/smoke_test_app.py http://127.0.0.1:8799 IMG1 IMG2 ...
"""
import json
import os
import sys
import tempfile
import time
import urllib.request


def call(base, path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("content-type", "").startswith("application/json") else raw


def job(base, path, body):
    jid = call(base, path, body)["job"]
    while True:
        s = call(base, f"/api/jobs/{jid}")
        if s["done"]:
            if s["error"]:
                raise RuntimeError(s["error"])
            return s["result"]
        time.sleep(0.5)


def main():
    base, images = sys.argv[1], sys.argv[2:]
    out = tempfile.mkdtemp(prefix="panoforge_smoke_")
    for _ in range(80):
        try:
            call(base, "/api/project")
            break
        except Exception:
            time.sleep(0.25)
    assert b"PanoForge" in call(base, "/"), "editor page"
    call(base, "/api/project/new", {})
    p = call(base, "/api/images/add", {"paths": images})["project"]
    print(f"loaded {len(p['images'])} images, lens {p['lenses'][0]['lens_type']}")
    r = job(base, "/api/align", {})
    print(f"aligned {r['connected']}/{r['n_images']}  {r['n_cps']} CPs  mean {r['mean_error']:.2f}px")
    ok = True
    for name, extra in [("pano.jpg", {}), ("pano.tif", {"output": {"bit_depth": 16, "compression": "lzw"}}),
                        ("pano.psd", {"output": {"layers": "layers_and_blended"}}), ("pano.png", {})]:
        path = os.path.join(out, name)
        try:
            r = job(base, "/api/stitch", {"width": 2000, "path": path, **extra})
            size = os.path.getsize(path)
            view = call(base, "/api/output?path=" + urllib.request.quote(path))
            print(f"  {name:9s} {r['width']}x{r['height']} {size / 1e6:.1f} MB, viewable {len(view) > 1000}")
        except Exception as e:
            ok = False
            print(f"  {name:9s} FAILED: {e}")
    cube = call(base, "/api/convert", {"path": os.path.join(out, "pano.jpg"), "to": "cube"})
    print(f"  cube faces: {len(cube['files'])}")
    print("output folder:", out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
