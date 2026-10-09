"""Prepare the Cats vs Dogs dataset into data/{train,test}/{cat,dog}/ and validate it.

What this script does (all numbers are measured, none are assumed):
  1. Downloads (git clone) a public Cats vs Dogs image repository, unless --source points at a local copy.
  2. Decodes every image to find unreadable / corrupt files.
  3. Hashes every file (SHA-256) and removes exact duplicates, keeping the first copy, so identical
     images can never appear in two different splits (no data leakage).
  4. Copies the clean images into data/train/{cat,dog} and data/test/{cat,dog}.
  5. Writes data/dataset_report.json with the real counts, image-size statistics and the checks above.

Validation images are NOT stored here: src/data_utils.py carves them out of data/train with a seeded,
stratified split, so train / validation / test stay disjoint.

Usage (from the project root):
    python scripts/prepare_dataset.py                       # clone from GitHub, then prepare
    python scripts/prepare_dataset.py --source path/to/PetImages_like_folder
    python scripts/prepare_dataset.py --limit_per_class 1500   # small subset for quick tests

Expected source layout:  <source>/train/{cats,dogs}  and  <source>/test/{cats,dogs}
(the folder names 'cat'/'dog' are accepted as well).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/laxmimerit/dog-cat-full-dataset"
EXT = {".jpg", ".jpeg", ".png"}
CLASS_ALIASES = {"cat": ("cat", "cats"), "dog": ("dog", "dogs")}


def find_class_dir(split_dir: Path, cls: str) -> Path:
    for name in CLASS_ALIASES[cls]:
        if (split_dir / name).is_dir():
            return split_dir / name
    raise FileNotFoundError(f"No folder {CLASS_ALIASES[cls]} in {split_dir}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="local folder with train/ and test/ (default: git clone REPO_URL)")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--limit_per_class", type=int, default=0, help="keep at most N images per class and split (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)

    # ---- 1. obtain the raw data
    tmp = None
    if a.source:
        src = Path(a.source)
    else:
        tmp = ROOT / "_raw_download"
        if tmp.exists():
            shutil.rmtree(tmp)
        print(f"Cloning {REPO_URL} ...")
        r = subprocess.run(["git", "clone", "--depth", "1", "-q", REPO_URL, str(tmp)])
        if r.returncode != 0:
            sys.exit("Dataset download FAILED. Check your internet connection, or download the dataset manually "
                     "and pass --source <folder> (see data/README.md).")
        src = tmp / "data"
    if not src.exists():
        sys.exit(f"Source folder not found: {src}")

    rng = np.random.RandomState(a.seed)
    report = {"source": a.source or REPO_URL, "seed": a.seed, "limit_per_class": a.limit_per_class,
              "splits": {}, "corrupt_files": [], "duplicates_removed": [], "modes": {}}
    seen_hash = {}              # sha256 -> "split/class/name" of the first copy kept
    modes, widths, heights = Counter(), [], []

    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            files = sorted(p for p in find_class_dir(src / split, cls).iterdir() if p.suffix.lower() in EXT)
            if a.limit_per_class:
                files = [files[i] for i in sorted(rng.permutation(len(files))[: a.limit_per_class])]
            dst = out / split / cls
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            kept = 0
            for p in files:
                try:
                    with Image.open(p) as im:
                        im.load()
                        mode, size = im.mode, im.size
                except Exception:
                    report["corrupt_files"].append(f"{split}/{cls}/{p.name}")
                    continue
                digest = sha256(p)
                if digest in seen_hash:
                    report["duplicates_removed"].append({"file": f"{split}/{cls}/{p.name}", "same_as": seen_hash[digest]})
                    continue
                seen_hash[digest] = f"{split}/{cls}/{p.name}"
                shutil.copy2(p, dst / p.name)
                kept += 1
                modes[mode] += 1
                widths.append(size[0])
                heights.append(size[1])
            report["splits"].setdefault(split, {})[cls] = {"found": len(files), "kept": kept}
            print(f"{split}/{cls}: found {len(files)}, kept {kept}")

    report["modes"] = dict(modes)
    report["n_images_kept"] = len(widths)
    for k, arr in (("width", widths), ("height", heights)):
        report[k] = {"min": int(min(arr)), "max": int(max(arr)), "mean": round(float(np.mean(arr)), 1),
                     "median": float(np.median(arr))}
    report["n_corrupt"] = len(report["corrupt_files"])
    report["n_duplicates_removed"] = len(report["duplicates_removed"])
    # leakage check: re-hash what is on disk and make sure no hash occurs in two places
    disk = {}
    for p in sorted(out.glob("*/*/*")):
        if p.suffix.lower() in EXT:
            disk.setdefault(sha256(p), []).append(str(p.relative_to(out)))
    report["leakage_check"] = {"files_on_disk": sum(len(v) for v in disk.values()),
                               "hash_collisions_across_files": sum(1 for v in disk.values() if len(v) > 1)}
    with open(out / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("corrupt_files", "duplicates_removed")}, indent=2))


if __name__ == "__main__":
    main()
