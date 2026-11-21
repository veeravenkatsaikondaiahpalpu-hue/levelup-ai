"""
Assemble the deployable Hugging Face Space from canonical sources.

Why a build step
----------------
spaces/backend/app.py imports `anomaly_detection.models` and `sentiment.models`
— the same classes the local FastAPI app uses. A Space is a separate git repo,
so those packages and their joblib artefacts have to be present inside the
Space folder at deploy time.

Copying them in by hand would leave two divergent copies of the same code in
the repository. Instead this script copies them just before upload, and the
copies are gitignored, so the originals stay the single source of truth.

Both packages resolve their model directory as <package_parent>/models/<name>,
so placing them at the Space root makes `spaces/backend/models/...` resolve
correctly with no path changes.

Usage
-----
    python -m scripts.prepare_space            # assemble
    python -m scripts.prepare_space --clean    # remove the copied artefacts
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "backend"

# (source, destination relative to the Space root)
PACKAGES = [
    (ROOT / "anomaly_detection", SPACE / "anomaly_detection"),
    (ROOT / "sentiment", SPACE / "sentiment"),
]
MODEL_DIRS = [
    (ROOT / "models" / "anomaly", SPACE / "models" / "anomaly"),
    (ROOT / "models" / "sentiment", SPACE / "models" / "sentiment"),
]

# Everything this script creates — also what --clean removes.
GENERATED = [dst for _, dst in PACKAGES] + [SPACE / "models"]


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))}


def clean() -> None:
    for path in GENERATED:
        if path.exists():
            shutil.rmtree(path)
            print(f"  removed {path.relative_to(ROOT)}")


def assemble() -> None:
    missing = [src for src, _ in PACKAGES + MODEL_DIRS if not src.exists()]
    if missing:
        for m in missing:
            print(f"  MISSING: {m.relative_to(ROOT)}")
        raise SystemExit("Cannot assemble — train the models first.")

    clean()
    total = 0
    for src, dst in PACKAGES + MODEL_DIRS:
        shutil.copytree(src, dst, ignore=_ignore)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        total += size
        print(f"  {size/1e6:7.2f} MB  {dst.relative_to(ROOT)}")
    print(f"\n  {total/1e6:7.2f} MB copied in total")

    print("\nSpace contents:")
    for f in sorted(SPACE.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            print(f"  {f.stat().st_size/1e6:8.3f} MB  {f.relative_to(SPACE)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="remove copied artefacts and exit")
    args = ap.parse_args()

    if args.clean:
        print("Cleaning generated Space artefacts:")
        clean()
        return

    print(f"Assembling Space at {SPACE.relative_to(ROOT)}:")
    assemble()


if __name__ == "__main__":
    main()
