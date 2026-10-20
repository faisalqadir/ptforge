# PtForge

**Next in line for the getcraft phenomenon:** a free, open-source panorama stitcher with a
PTGui-class feature set. Load your overlapping photos (even from a circular fisheye) and it
finds the matching points, works out how the camera was turned for each shot, corrects the
lens, evens out exposure, and blends everything into one seamless panorama, including full
360° × 180° spheres ready for VR viewers.

It ships as a Mac app with a visual editor, and as a Python library plus a command line tool
for scripting and batch jobs.

---

## Contents

1. [What is panorama stitching? (a beginner's tour)](#1-what-is-panorama-stitching-a-beginners-tour)
2. [Install and run](#2-install-and-run)
3. [Your first panorama in the app](#3-your-first-panorama-in-the-app)
4. [Using the command line](#4-using-the-command-line)
5. [Features](#5-features)
6. [How it works inside](#6-how-it-works-inside)
7. [How this project was made](#7-how-this-project-was-made)
8. [Project layout](#8-project-layout)
9. [Testing](#9-testing)
10. [Known limits and roadmap](#10-known-limits-and-roadmap)
11. [Legal](#11-legal)

---

## 1. What is panorama stitching? (a beginner's tour)

A camera only sees a slice of the world at a time. To capture a whole room, a landscape or
the full sphere around you, you take several photos while turning the camera, making sure
neighbouring photos **overlap** (about 25–50 % is ideal). A *stitcher* then merges them into
one picture. That takes several steps, and PanoForge does each of them for you:

| Step | What it means in plain words |
|---|---|
| **Lens settings** | Every lens bends the image a little. A fisheye bends it a lot. The stitcher needs a mathematical model of your lens (field of view and distortion) so it can "un-bend" each photo. PanoForge reads the focal length from the photo's EXIF data and can detect the round image of a circular fisheye by itself. |
| **Control points** | Spots that show the same real-world detail in two photos, like the corner of a window seen in both. PanoForge finds hundreds of these automatically. |
| **Optimisation (alignment)** | Using the control points, it works out exactly which way the camera was pointing for every shot (yaw = left/right, pitch = up/down, roll = tilt) and fine-tunes the lens model, until the matching points line up as closely as possible. |
| **Projection** | A sphere can't be flattened without stretching, just like a world map. The *projection* decides how to stretch it: equirectangular (the 2:1 "world map" format used for 360° photos), cylindrical, rectilinear (straight lines stay straight), little planet, and more. |
| **Exposure correction** | Photos taken one after another rarely have identical brightness and colour, and lenses get darker towards the corners (vignetting). PanoForge measures this and evens it out. |
| **Seams and blending** | Where two photos overlap, it chooses the best line along which to cut (avoiding people who moved, for example), then blends across the cut so no edge is visible. |
| **HDR (optional)** | If you shot several exposures per position (bracketing), it can merge them so both the bright window and the dark corner keep their detail. |

> **Tip for shooting:** turn the camera around the *lens*, not around your body. If the
> camera moves sideways between shots, nearby objects shift against the background
> (*parallax*) and no software can line everything up perfectly. A panoramic tripod head
> solves this. For handheld shots, keep the lens in the same spot as best you can.

---

## 2. Install and run

### Option A: the Mac app (easiest)

Requirements: a Mac with Apple Silicon (M1 or newer) on macOS 13 or later.

1. Download `PanoForge-0.1.0.dmg` from the
   [Releases page](https://github.com/faisalqadir/ptforge/releases/latest), or build it
   yourself as described below.
2. Double-click the `.dmg`. A window opens showing **PanoForge** and an **Applications** folder.
3. Drag **PanoForge** onto **Applications**.
4. Open PanoForge from Launchpad or the Applications folder.
   - The app isn't signed with an Apple developer certificate, so the first time macOS may
     say it "cannot be opened". In that case right-click (or Control-click) the app, choose
     **Open**, then click **Open** again. You only need to do this once.

### Option B: run from source (any Mac or Linux, for developers)

You need **Python 3.10 or newer** (check with `python3 --version`) and **git**.

```bash
# 1. download the code
git clone https://github.com/faisalqadir/ptforge.git
cd ptforge

# 2. create an isolated Python environment so nothing touches your system Python
python3 -m venv .venv

# 3. install PanoForge and its libraries (takes a few minutes the first time)
.venv/bin/pip install -e ".[desktop,dev]"

# 4a. start the editor in a native window
.venv/bin/python -m panoforge.desktop

# 4b. ...or start the editor in your web browser instead
.venv/bin/panoforge gui            # then open http://127.0.0.1:8765
```

On Windows, use `.venv\Scripts\` instead of `.venv/bin/`.

### Building the Mac app and disk image yourself

```bash
.venv/bin/pip install -e ".[build]"
packaging/build_mac.sh            # builds dist/PanoForge.app and ../PanoForge-0.1.0.dmg
```

This takes about a minute. `build_mac.sh` refuses to package a bundle that still points into
your development folder, so the result runs on any Apple Silicon Mac. To test a build without
opening a window, run the app binary with `--server-only`, then point
`packaging/smoke_test_app.py` at it with a few images:

```bash
dist/PanoForge.app/Contents/MacOS/PanoForge --server-only &       # serves on port 8765
.venv/bin/python packaging/smoke_test_app.py http://127.0.0.1:8765 photos/*.JPG
```

---

## 3. Your first panorama in the app

The window opens on the **Project Assistant**, which walks you through the three steps that
matter most:

1. **Load images…** Pick your photos with the normal Mac file dialog, or drag photos or a
   whole folder from Finder onto the window. PanoForge reads the EXIF data and sets up the
   lens. Check the lens type (normal, fisheye, circular fisheye) in the **Lens Settings** tab
   if your camera didn't record it.
2. **Align images.** This finds control points and solves the camera positions. A report
   tells you how well the photos fit together (the average control point error in pixels;
   under ~3 px is good).
3. **Create panorama.** Choose the projection, then set the size, file format and location
   under **Output settings…**, and render.

Want more control? Every step has its own tab on the left:

- **Panorama Editor:** a live preview of the whole panorama, drawn on your graphics card.
  Drag to re-centre the view, hold a key to roll or tilt the horizon, move single images,
  and draw the crop.
- **Control Points:** look at two photos side by side, add or delete points, and auto-tune
  them to sub-pixel accuracy. The table view sorts points by error so bad ones stand out.
- **Optimizer:** choose which lens parameters may change, and use vertical/horizontal
  control points to make walls stand straight.
- **Mask:** paint areas that must be kept (a person you want in) or removed (a tripod leg).
- **Exposure / HDR:** brightness, white balance, vignetting, and bracket merging.
- **Batch Stitcher:** queue several projects and render them in one go.

`Cmd+Z` / `Shift+Cmd+Z` undo and redo almost everything. Projects are saved as `.pfproj`
files (plain JSON, so you can read them in any text editor). Hugin `.pto` projects can be
opened and exported too. If the app closes with unsaved work, it offers to restore it the
next time you start it.

---

## 4. Using the command line

Everything in the app can be scripted. A few examples:

```bash
# make a project from photos, align it and stitch straight away
panoforge new photos/*.JPG -o room.pfproj --stitch --out room.jpg

# step by step
panoforge new photos/*.JPG -o room.pfproj
panoforge align room.pfproj
panoforge stitch room.pfproj --out room.tif --width 12000

# look at what a project contains
panoforge info room.pfproj

# turn an equirectangular 360° image into six cube faces or a "little planet"
panoforge convert room.jpg --to cube -o faces/
panoforge convert room.jpg --to littleplanet -o planet.jpg

# split a folder full of shots into panoramas (by time gap), align and stitch each
panoforge batch shoots/ -o projects/ --align --stitch

# hand a project over to Hugin
panoforge export room.pfproj room.pto
```

Run `panoforge <command> -h` for every option. In Python:

```python
from panoforge.align import create_project, align_images
from panoforge.stitcher import stitch

p = create_project(["a.jpg", "b.jpg", "c.jpg"])
align_images(p)
stitch(p, "pano.jpg", width=8000)
```

---

## 5. Features

| Area | What's included |
|---|---|
| Lens models | Rectilinear, fisheye with one continuous *fisheye factor* (equidistant, equisolid, orthographic, stereographic and everything in between), circular fisheye, cylindrical and equirectangular sources. Panorama Tools distortion a/b/c, centre shift d/e, shear g/t. |
| Lens set-up | Focal length and field of view from EXIF; automatic circular-fisheye detection (finds the image circle, its crop and its offset); a lens database for reuse. |
| Control points | SIFT/RootSIFT features, mutual ratio matching, RANSAC on a rotation model in ray space (so it works for fisheyes), spatial spreading, pair pruning for big projects. Editor with auto-add, correlation fine-tune and a magnifier. Normal, vertical, horizontal and straight-line points. |
| Optimizer | Bundle adjustment with symmetric reprojection error. Bracket stacks share one pose. Lens parameters global or per image. Distortion presets (none / medium / heavy / heavy + shift), fisheye factor, shear, viewpoint correction for flat scenes, robust loss, "delete worst control points". |
| Align images | Robust start: it tests several possible ways of chaining the photos together and rejects false matches (such as repeating wallpaper patterns), then prunes outliers, straightens the horizon, centres and fits the panorama. |
| Exposure | Exposure value, white balance, vignetting, flare and camera response curve; uses EXIF so manual exposures are only fine-tuned. |
| Projections | Equirectangular, cylindrical, Mercator, rectilinear, fisheye, stereographic (little planet), equisolid, orthographic, Panini / vedutismo (with compression), sinusoidal, architectural, and transverse equirectangular / cylindrical / Mercator / Panini. |
| Stitching | Only the cropped output area is computed. Graph-cut or Voronoi seams; multiband, feather or no blending; exact 360° wrap-around; include/exclude masks; blend priority; Lanczos / bicubic / bilinear / nearest interpolation; antialiased downscaling. |
| HDR | Bracket detection and linking. True HDR (with ghost removal) or exposure fusion, merging before or after blending. Local tone mapping. EXR / Radiance output. |
| Output | JPEG (with EXIF and GPano XMP, so Facebook, Google and VR viewers recognise 360° images), TIFF 8/16/32-bit (LZW/ZIP, alpha), PNG 8/16, WebP, layered Photoshop PSD/PSB, individual layers, cube faces, little planet, reprojection, and a self-contained web viewer you can upload to any website. |
| Projects | Native `.pfproj` (JSON), Hugin `.pto` import and export, templates, batch builder, batch stitcher, undo/redo, crash recovery. |
| Editor | Project Assistant, Source Images, Lens Settings, Crop, Mask, Image Parameters, Control Points (+ table), Optimizer, real-time Panorama Editor (drag / roll / single images / set centre / crop / numerical transform / detail viewer), Exposure/HDR, Project Settings, Preview, Metadata, Create Panorama, Batch Stitcher, 360° viewer. |

---

## 6. How it works inside

```
photos ─► EXIF + lens set-up ─► control points ─► alignment ─► exposure ─► warping ─► seams ─► blending ─► file
           (lens_detect.py)      (features.py)    (align.py,   (photo-     (remap.py)  (seams.py) (blend.py,
                                                  optimizer.py) metric.py)                         hdr.py)
```

- **Geometry (`geometry.py`).** Every pixel in every photo is turned into a direction in 3-D
  space (a "ray"), and every pixel of the output into a direction too. Lens models map
  between pixels and rays; projections map between rays and the output canvas. Everything
  else in the program builds on these two functions, and the test suite checks that each
  lens and projection maps back and forth without error.
- **Control points (`features.py`).** SIFT finds distinctive spots in each photo. Matches
  between two photos are checked with RANSAC: the program repeatedly guesses a camera
  rotation from three random matches and keeps the guess most matches agree with. Testing
  rotations of *rays*, rather than flat image warps, is what makes it work with fisheye
  lenses.
- **Alignment (`align.py`, `optimizer.py`).** Pairwise rotations are chained together along a
  spanning tree of the photo graph, trying several trees and keeping the most consistent
  one. Then a bundle adjustment (SciPy `least_squares` with a sparse Jacobian and a robust
  loss) refines all camera angles and lens parameters at once, and control points that
  still don't fit are removed.
- **Exposure (`photometric.py`).** Pixels sampled where photos overlap are compared in linear
  light, and exposure, white balance, vignetting and the response curve are solved together.
- **Seams (`seams.py`).** A graph cut (PyMaxflow) finds the cut line through each overlap
  where the photos differ least.
- **Blending (`blend.py`).** Multiband blending splits each image into coarse and fine
  detail layers (a Laplacian pyramid) and blends coarse layers over a wide area and fine
  ones over a narrow area. Colour shifts disappear and details stay sharp. The blender wraps
  around the 360° seam so the left and right edges of a sphere join without a line.
- **Editor (`server.py` + `web/`).** A small local web server (FastAPI) exposes the engine; the
  interface is plain JavaScript (no build step). The live Panorama Editor is a WebGL2
  shader that re-implements `geometry.py` on the graphics card, so dragging the panorama
  around is instant. The Mac app (`desktop.py`) runs that server privately inside the app
  and shows the interface in a native window with native file dialogs and Finder
  drag-and-drop.

---

## 7. How this project was made

PanoForge was built by an AI coding agent (Claude Code, Anthropic's Claude) working with
me. I asked for a working stand-in for PTGui with its advanced features. The agent wrote
the code, tests and tools, and I reviewed the results on my own photos.

**It is a clean-room implementation. No PTGui code was copied, decompiled or reused.**

### Starting points

1. **PTGui 13.10 trial, as a feature checklist.** The free trial from
   [ptgui.com](https://www.ptgui.com/download.html) was installed and its menus, tabs,
   settings and user-interface text were read to list *what* a professional stitcher has to
   offer (lens types, optimizer options, projections, output formats, the Project Assistant
   flow). *How* to do each of these was written from scratch using the published methods
   below. No licence keys or old installers were used.
2. **My own photos as the benchmark.** A room shot with a Nikon D5200 and a Sigma 4.5 mm
   circular fisheye (4 shots, handheld, so with real parallax), together with the panorama
   PTGui made from the same photos. PanoForge's output was compared with it for alignment,
   framing and colour. A second, harder set (night, ISO 3200, a wall with repeating
   wallpaper) uncovered the false-match problem the alignment now guards against.
3. **Published research and open formats** that the core algorithms are based on:
   - Helmut Dersch's *Panorama Tools* lens and distortion model (the a/b/c, d/e, g/t parameters
     used by most stitchers) and the documented **Hugin `.pto`** project format.
   - D. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints* (SIFT), 2004;
     R. Arandjelović & A. Zisserman, *RootSIFT*, 2012.
   - M. Fischler & R. Bolles, *RANSAC*, 1981; W. Kabsch, optimal rotation between point sets, 1976.
   - M. Brown & D. Lowe, *Automatic Panoramic Image Stitching using Invariant Features*, 2007
     (rotation-only camera model, bundle adjustment, multiband blending for panoramas).
   - B. Triggs et al., *Bundle Adjustment: A Modern Synthesis*, 2000.
   - P. Burt & E. Adelson, *A Multiresolution Spline with Application to Image Mosaics*, 1983.
   - Y. Boykov & V. Kolmogorov, max-flow/min-cut (graph-cut seams); V. Kwatra et al., *Graphcut Textures*, 2003.
   - P. Debevec & J. Malik, *Recovering High Dynamic Range Radiance Maps from Photographs*, 1997;
     T. Mertens, J. Kautz & F. Van Reeth, *Exposure Fusion*, 2007.
   - D. Goldman, *Vignette and Exposure Calibration and Compensation*, 2010 (photometric model).
   - Google's GPano XMP specification for 360° photo metadata, and Adobe's published PSD/PSB file format.
4. **Open-source libraries:** OpenCV (features, warping), NumPy and SciPy (the maths and the
   optimiser), Numba (fast loops), PyMaxflow (graph cuts), tifffile and imagecodecs (TIFF),
   Pillow and exifread (images and EXIF), FastAPI and Uvicorn (editor server), pywebview
   (native window) and PyInstaller (the Mac app).

### Method

1. **Geometry first.** The camera and projection maths was written and tested before anything
   else: every lens and projection must map a pixel to a direction and back exactly.
2. **Synthetic ground truth.** `tools/synth.py` renders fake photo shoots from a known
   panorama with known camera angles, lenses, exposures and noise. With the "right answer"
   known, every stage can be measured (`tools/eval_align.py`, `tools/eval_photo.py`), for
   example: do the recovered camera angles match the true ones within a fraction of a degree?
3. **Engine, then interface.** Alignment, exposure, seams, blending, HDR and output were built
   and tested from the command line first. The editor came after, and all of it calls the same
   engine.
4. **Real-photo validation.** The engine was run on real handheld fisheye sets and compared
   against PTGui's result. Each failure became a fix and, where possible, a test: fisheye-safe
   match checking, rejecting look-alike matches from repeating patterns, repeatable
   re-alignment, and blending that wraps across the 360° seam.
5. **Packaging.** The editor was wrapped in a native Mac window, bundled with PyInstaller, and
   the finished disk image was mounted and smoke-tested with real photos before release.

---

## 8. Project layout

```
panoforge/
  geometry.py     camera model + projections (the maths everything else uses)
  imageio.py      loading/saving, EXIF, colour transfer
  project.py      data model + JSON project files
  features.py     control point generation
  optimizer.py    geometric bundle adjustment
  align.py        project assistant pipeline (create, align, straighten, fit)
  lens_detect.py  circular fisheye detection
  photometric.py  exposure / white balance / vignetting / response
  remap.py        canvas, footprints, warping
  seams.py        Voronoi + graph-cut seams
  blend.py        multiband / feather blenders (streaming, 360°-wrap aware)
  hdr.py          true HDR merge, exposure fusion, tone mapping
  stitcher.py     the full rendering pipeline
  output.py psd.py publish.py ptgui_format.py templates.py masks.py cpfinetune.py
  cli.py          the `panoforge` command
  server.py       FastAPI backend of the editor
  desktop.py      native window (Mac app entry point)
  web/            editor frontend (ES modules + WebGL2, no build step)
packaging/        icon, PyInstaller spec, build_mac.sh, smoke test for the app bundle
tools/            synthetic shoot generator and accuracy checks
tests/            pytest suite
```

The WebGL shader (`web/js/glpano.js`) and `web/js/geom.js` re-implement `geometry.py`; keep
them in sync if you change the camera model.

---

## 9. Testing

```bash
.venv/bin/pytest -q          # 34 tests, about 20 seconds
```

The tests cover round-trips for every lens and projection, end-to-end alignment on synthetic
shoots with known answers, exposure recovery, HDR, and every output format. To generate
bigger synthetic shoots for experiments:

```bash
.venv/bin/python tools/synth.py testdata/fisheye --preset fisheye
```

---

## 10. Known limits and roadmap

- **Parallax correction** for handheld shots (in progress).
- PTGui `.pts` project import (needs sample files to work from).
- RAW files need the optional `rawpy` package (`pip install -e ".[raw]"`).
- No GPU stitching yet. A 6000 × 3000 fisheye panorama takes about 35 s on an M2.
- Very large outputs (above ~200 megapixels) need tiled blending, which isn't done yet.
- Nadir/zenith patch round-trip and a mosaic mode for flat scenes.
- The Mac app is built for Apple Silicon and isn't notarised.

Contributions and bug reports are welcome. Sample shoots that stitch badly help most.

---

## 11. Legal

PanoForge is an independent project. It is not affiliated with, endorsed by, or connected to
PTGui or New House Internet Services B.V. "PTGui" is a trademark of its owner and is mentioned
only to describe the feature set this project aims to match. No PTGui software, code or assets
are included in this repository.

---

thankmelater
