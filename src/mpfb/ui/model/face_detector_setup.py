"""One-time, opt-in setup of a self-contained local MediaPipe face detector.

Runs as an *external* Python process. Never installs pip packages in Blender's
active Python environment, never uploads face images and never edits source
files inside the installed MPFB extension.

The Blender UI launches this script from a button. The script creates a venv,
pins a tested Face Landmarker-compatible MediaPipe release, and downloads the
official .task model. Requires HTTPS internet access only during setup.

GPL-3.0-or-later.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MEDIAPIPE_REQUIREMENT = "mediapipe==0.10.32"
MAX_MODEL_BYTES = 100 * 1024 * 1024


def _embedded_interpreters():
    """Find Blender's bundled Python first, then a compatible system Python.

    sys.executable inside Blender points to 'blender', not necessarily python.
    sys.prefix normally points to its self-contained bundled Python directory.
    """
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots = [Path(sys.prefix), Path(sys.base_prefix)]
    candidates = []
    for root in roots:
        for name in (f"bin/{version}", "bin/python3", "bin/python",
                     "bin/python.exe", "python.exe"):
            candidates.append(root / name)
    if Path(sys.executable).name.lower().startswith("python"):
        candidates.append(Path(sys.executable))
    for executable in (shutil.which("python3"), shutil.which("python")):
        if executable:
            candidates.append(Path(executable))
    distinct = set()
    for candidate in candidates:
        if not candidate.is_file() or str(candidate) in distinct:
            continue
        distinct.add(str(candidate))
        yield str(candidate)


def _call(args, *, timeout=900):
    print("Running", str(args[0]), *[str(x) for x in args[1:3]], flush=True)
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout, check=False)
    if proc.stdout:
        print(proc.stdout[-18000:], flush=True)
    if proc.returncode:
        raise RuntimeError("Detector setup command failed (exit %s)" % proc.returncode)


def _venv_python(venv_path):
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _choose_interpreter():
    for candidate in _embedded_interpreters():
        proc = subprocess.run([candidate, "-c", "import venv, sys; print(sys.version_info[:2])"],
                              capture_output=True, text=True, timeout=20, check=False)
        if proc.returncode == 0:
            return candidate
    raise RuntimeError("Could not find a Python interpreter with venv support. "
                       "Use the External Python field instead.")


def _download_task(target):
    if target.is_file() and target.stat().st_size >= 100_000:
        print("MediaPipe task model already available.", flush=True)
        return
    tmp = target.with_suffix(".task.download")
    try:
        request = urllib.request.Request(TASK_URL, headers={"User-Agent": "MPFB-Character-Studio/2.0.19"})
        with urllib.request.urlopen(request, timeout=45) as response, tmp.open("wb") as file:
            downloaded = 0
            while True:
                data = response.read(1024 * 512)
                if not data:
                    break
                downloaded += len(data)
                if downloaded > MAX_MODEL_BYTES:
                    raise RuntimeError("Unexpectedly large landmarker file")
                file.write(data)
        if downloaded < 100_000:
            raise RuntimeError("Downloaded task file was unexpectedly small")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def install(destination):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    venv = destination / "venv"
    python = _venv_python(venv)
    model = destination / "face_landmarker.task"
    if not python.is_file():
        base = _choose_interpreter()
        print("Creating dedicated detector environment from", base, flush=True)
        _call([base, "-m", "venv", str(venv)], timeout=120)
    if not python.is_file():
        raise RuntimeError("Could not create the isolated MediaPipe Python environment")
    check = subprocess.run([str(python), "-c",
                            "import mediapipe; from mediapipe.tasks.python import vision"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=45, check=False)
    if check.returncode:
        print("Installing platform-compatible MediaPipe into isolated venv.", flush=True)
        _call([str(python), "-m", "pip", "install", "--disable-pip-version-check",
               "--no-input", "--only-binary=:all:", MEDIAPIPE_REQUIREMENT])
    _download_task(model)
    print("DONE: detector ready at", python, "and", model, flush=True)
    return python, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        install(args.destination)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
