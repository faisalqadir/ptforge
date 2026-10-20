"""PanoForge desktop app: the editor in a native window (WKWebView on macOS).

The editor's server runs inside the app on a private local port; file dialogs are the native
ones and files dropped from Finder are added by their real paths.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time

RECOVERY = os.path.join(os.path.expanduser("~"), ".panoforge", "recovery.pfproj")

IMAGE_TYPES = ("Images (*.jpg;*.jpeg;*.tif;*.tiff;*.png;*.webp;*.heic;*.heif;*.bmp;*.dng;*.cr2;*.cr3;*.nef;*.arw;*.raf;*.orf;"
               "*.rw2;*.pef;*.srw;*.exr;*.hdr;*.psd)", "All files (*.*)")
PROJECT_TYPES = ("PanoForge / Hugin projects (*.pfproj;*.pto;*.pts)", "All files (*.*)")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    import uvicorn

    from . import server

    if "--server-only" in argv:  # headless (automated tests of the bundle): just the editor server
        port = int(os.environ.get("PANOFORGE_PORT") or 8765)
        print(f"PanoForge editor server: http://127.0.0.1:{port}/", flush=True)
        uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
        return
    import webview
    from webview.dom import DOMEventHandler

    # a project or images given on the command line / "Open with"
    if argv:
        server_args = [a for a in argv if not a.startswith("-psn")]
        if server_args:
            target = server_args[0]
            try:
                if target.lower().endswith((".pfproj", ".pto")):
                    from .project import Project

                    server.S.project = Project.load(target)
                else:
                    server.add_images({"paths": [a for a in server_args if os.path.isfile(a)]})
                    server.S.undo.clear()
            except Exception as e:  # start with an empty project rather than not at all
                print(f"could not open {target}: {e}", file=sys.stderr)

    port = int(os.environ.get("PANOFORGE_PORT") or 0) or _free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True, name="panoforge-server").start()
    for _ in range(400):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.025)

    print(f"PanoForge editor server: http://127.0.0.1:{port}/", flush=True)
    window = webview.create_window("PanoForge", f"http://127.0.0.1:{port}/", width=1480, height=940, min_size=(1100, 680),
                                   background_color="#1d1f22", text_select=False, zoomable=False)

    def dialog(kind, title, multi, filter, default_name, directory):
        directory = directory if directory and os.path.isdir(directory) else os.path.expanduser("~")
        if kind == "save":
            r = window.create_file_dialog(webview.FileDialog.SAVE, directory=directory, save_filename=default_name or "")
        elif kind == "folder":
            r = window.create_file_dialog(webview.FileDialog.FOLDER, directory=directory)
        else:
            types = IMAGE_TYPES if filter == "images" else PROJECT_TYPES if filter == "projects" else ()
            r = window.create_file_dialog(webview.FileDialog.OPEN, directory=directory, allow_multiple=bool(multi), file_types=types)
        if not r:
            return None
        return [r] if isinstance(r, str) else list(r)

    server.DIALOG_PROVIDER = dialog

    def on_drop(e):
        files = (e.get("dataTransfer") or {}).get("files") or []
        paths = [f.get("pywebviewFullPath") for f in files if isinstance(f, dict) and f.get("pywebviewFullPath")]
        if not paths:
            return
        projects = [p for p in paths if p.lower().endswith((".pfproj", ".pto"))]
        try:
            if projects:
                server.open_project({"path": projects[0]})
            else:
                from .imageio import is_image_file

                expanded = []
                for pth in paths:
                    if os.path.isdir(pth):
                        expanded += sorted(os.path.join(pth, f) for f in os.listdir(pth) if is_image_file(f))
                    elif is_image_file(pth):
                        expanded.append(pth)
                if expanded:
                    server.add_images({"paths": expanded})
        except Exception as ex:
            window.evaluate_js(f"window.pfToast && window.pfToast({str(ex)!r}, 'err')")
            return
        window.evaluate_js("window.pfRefresh && window.pfRefresh()")

    def on_loaded():
        window.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, False)
        window.dom.document.events.drop += DOMEventHandler(on_drop, True, False)

    def on_closing():
        # unsaved work survives a quit: keep a recovery copy that the next launch offers to restore
        try:
            if server.S.dirty and server.S.project.images:
                os.makedirs(os.path.dirname(RECOVERY), exist_ok=True)
                p = server.S.project.copy()
                p.metadata["recovered_from"] = server.S.project.path
                p.save(RECOVERY, relative_paths=False)
        except Exception:
            pass
        return True

    window.events.loaded += on_loaded
    window.events.closing += on_closing
    webview.start(debug=bool(os.environ.get("PANOFORGE_DEBUG")))
    srv.should_exit = True


if __name__ == "__main__":
    main()
