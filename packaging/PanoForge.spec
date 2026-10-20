# PyInstaller spec for the PanoForge macOS app.   Build with: packaging/build_mac.sh
import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
VERSION = "0.1.0"

# imagecodecs imports its codec extensions lazily, so they must be listed explicitly
hidden = (collect_submodules("panoforge") + collect_submodules("uvicorn") + collect_submodules("imagecodecs") +
          ["imagecodecs", "tifffile", "exifread", "maxflow", "webview.platforms.cocoa", "multipart", "python_multipart"])

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    datas=[(os.path.join(ROOT, "panoforge", "web"), os.path.join("panoforge", "web"))],
    binaries=collect_dynamic_libs("imagecodecs"),
    hiddenimports=hidden,
    excludes=["numba", "llvmlite", "tkinter", "matplotlib", "IPython", "pytest", "psd_tools", "skimage", "pandas",
              "PyInstaller", "setuptools", "pip"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PanoForge",
    console=False,
    argv_emulation=True,  # Finder "Open With" / dropping files on the Dock icon arrive as argv
    target_arch=None,
    codesign_identity=None,
    icon=os.path.join(SPECPATH, "PanoForge.icns"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PanoForge")

IMAGE_EXTS = ["jpg", "jpeg", "tif", "tiff", "png", "webp", "heic", "heif", "dng", "cr2", "cr3", "nef", "arw", "raf", "orf",
              "rw2", "exr", "hdr"]
app = BUNDLE(
    coll,
    name="PanoForge.app",
    icon=os.path.join(SPECPATH, "PanoForge.icns"),
    bundle_identifier="app.panoforge.editor",
    version=VERSION,
    info_plist={
        "CFBundleName": "PanoForge",
        "CFBundleDisplayName": "PanoForge",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "NSHumanReadableCopyright": "PanoForge panorama stitcher",
        "CFBundleDocumentTypes": [
            {"CFBundleTypeName": "PanoForge Project", "CFBundleTypeRole": "Editor", "LSHandlerRank": "Owner",
             "CFBundleTypeExtensions": ["pfproj"], "CFBundleTypeIconFile": "PanoForge.icns"},
            {"CFBundleTypeName": "Hugin Project", "CFBundleTypeRole": "Editor", "LSHandlerRank": "Alternate",
             "CFBundleTypeExtensions": ["pto"]},
            {"CFBundleTypeName": "Image", "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
             "CFBundleTypeExtensions": IMAGE_EXTS},
        ],
    },
)
