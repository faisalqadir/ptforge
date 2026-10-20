#!/bin/bash
# Build PanoForge.app and a drag-to-install disk image (PanoForge-<version>.dmg).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-$ROOT/..}"            # where the .dmg goes (default: the folder above the project)
VERSION="0.1.0"
DIST="${DIST:-dist}"              # build folder (override to avoid touching a running copy)

.venv/bin/python packaging/make_icon.py >/dev/null
rm -rf build "$DIST"
.venv/bin/pyinstaller --noconfirm --clean --distpath "$DIST" --workpath build packaging/PanoForge.spec

APP="$DIST/PanoForge.app"
# the bundle must be self-contained: refuse to package one that still points into this checkout
if grep -rl --include='*.cfg' "$ROOT/.venv" "$APP" >/dev/null 2>&1; then echo "bundle references the build venv" >&2; exit 1; fi
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true   # ad-hoc signature (runs locally)

STAGE="$DIST/dmg"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="$OUT/PanoForge-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "PanoForge $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
echo "$DMG"
du -h "$DMG" | cut -f1
