#!/usr/bin/env python3
"""Build one Blender 4.2+ installable ZIP containing ALL MPFB character tools.

Run from any working directory:
    python tools/build_onedrop.py
    python tools/build_onedrop.py --output dist/MPFB-Character-Studio.zip

No Blender install, pip or third-party dependencies are required to *build*.
This zips the existing fully populated src/mpfb extension at the ZIP ROOT,
not the entire GitHub source archive. An arbitrary GitHub "Download ZIP"
cannot be installed directly as a Blender extension.

This intentionally ships the complete original MPFB data (basemesh, targets,
shaders, rigs, UVs) plus muscularity, face fitting and foot realism.
"""
from __future__ import annotations

import argparse
import compileall
import os
from pathlib import Path
import sys
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

REPO = Path(__file__).resolve().parent.parent
ADDON = REPO / "src" / "mpfb"
DEFAULT_OUT = REPO / "dist" / "MPFB-Character-Studio.zip"
REQUIRED = (
    "__init__.py",
    "blender_manifest.toml",
    "data/3dobjs/base.obj",
    "data/targets/target.json",
    "data/targets/macrodetails/macro.json",
    "data/face_fit/mp_to_mpfb.json",
    "data/mesh_metadata/hm08_config.json",
    "ui/model/modelpanel.py",
    "ui/model/muscularitypresets.py",
    "ui/model/facephotofit.py",
    "ui/model/facephotofit_core.py",
    "ui/model/facephotofit_multiview_core.py",
    "ui/model/face_photo_landmarks.py",
    "ui/model/face_detector_setup.py",
    "ui/model/footrealism.py",
    "ui/model/footrealism_core.py",
    "ONE_DROP_README.md",
)
FORBIDDEN_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".blend1", ".blend2"}


def files_to_ship():
    if not ADDON.is_dir():
        raise RuntimeError("Missing add-on source directory: " + str(ADDON))
    result = []
    for path in ADDON.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ADDON)
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == ".DS_Store":
            continue
        result.append(path)
    return sorted(result, key=lambda path: path.as_posix())


def check_source(files):
    names = {p.relative_to(ADDON).as_posix() for p in files}
    absent = sorted(set(REQUIRED) - names)
    if absent:
        raise RuntimeError("Missing essential contents: " + ", ".join(absent))
    manifest = tomllib.loads((ADDON / "blender_manifest.toml").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0" or manifest.get("type") != "add-on":
        raise RuntimeError("Missing Blender 4.2+ extension manifest fields")
    if manifest.get("id") != "mpfb":
        raise RuntimeError("Preserve the existing MPFB extension ID so the update replaces MPFB")
    panel = (ADDON / "ui/model/modelpanel.py").read_text(encoding="utf-8")
    for panel_type in ("MPFB_PT_Muscularity_Presets_Panel", "MPFB_PT_Face_Photo_Fit_Panel",
                       "MPFB_PT_Foot_Realism_Panel"):
        if panel_type not in panel:
            raise RuntimeError("Tool not connected to Model sidebar: " + panel_type)
    for file in files:
        if file.suffix == ".py":
            # Compile without writing pyc files into distributable source.
            compile(file.read_bytes(), str(file), "exec")
    return manifest


def add_file(archive, file):
    name = file.relative_to(ADDON).as_posix()
    info = ZipInfo(name)
    info.date_time = (2026, 1, 1, 0, 0, 0)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (0o100644 << 16)
    archive.writestr(info, file.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=8)


def build(output):
    files = files_to_ship()
    manifest = check_source(files)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if ADDON.resolve() in output.parents:
        raise RuntimeError("Refusing to create ZIP inside add-on source tree")
    temporary = output.with_name(output.name + ".tmp")
    try:
        with ZipFile(temporary, "w", allowZip64=True) as archive:
            for file in files:
                add_file(archive, file)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC verification failed")
            if sorted(archive.namelist()) != sorted(p.relative_to(ADDON).as_posix() for p in files):
                raise RuntimeError("Archive contents are incomplete")
            if any(name.startswith("mpfb/") for name in archive.namelist()):
                raise RuntimeError("Blender extension ZIP must have manifest at archive root")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("Built Blender drag-and-drop extension: " + str(output))
    print("Name: " + manifest["name"] + " | v" + manifest["version"])
    print("Files: " + str(len(files)) + " | Bytes: " + str(output.stat().st_size))
    print("Install: drag the ZIP into Blender 4.2+ or Preferences > Get Extensions > Install from Disk.")
    print("Photo fitting requires one-time MediaPipe setup from the Face Photo Fit panel.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true", help="Validate entire source package without writing ZIP")
    args = parser.parse_args()
    if args.check:
        files = files_to_ship()
        manifest = check_source(files)
        print("Source package verified:", len(files), "files; version", manifest["version"])
    else:
        build(args.output)


if __name__ == "__main__":
    main()
