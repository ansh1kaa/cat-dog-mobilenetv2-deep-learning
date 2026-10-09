"""Shared data code: file listing, seeded stratified train/validation split, image loading, caching, augmentation.

Label convention used EVERYWHERE in this project:  0 = cat,  1 = dog.
The model has ONE sigmoid output = P(dog).  P(cat) = 1 - P(dog).  Predicted dog if P(dog) >= 0.5.

Folders (created by scripts/prepare_dataset.py):  data/train/{cat,dog}  data/test/{cat,dog}
Validation images are taken from data/train; data/test is only touched for the final evaluation.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["cat", "dog"]
SEED = 42
EXT = {".jpg", ".jpeg", ".png"}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def list_split(split, data_dir=DATA_DIR):
    """Sorted list of (path, label) for data/<split>/{cat,dog}."""
    items = []
    for label, name in enumerate(CLASS_NAMES):
        folder = Path(data_dir) / split / name
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} not found - run: python scripts/prepare_dataset.py")
        items += [(p, label) for p in sorted(folder.iterdir()) if p.suffix.lower() in EXT]
    return items


def make_splits(data_dir=DATA_DIR, per_class=0, val_fraction=0.2, test_per_class=0, seed=SEED):
    """Return dict split -> list of (path, label).

    train/val come from data/train (stratified 80/20, seeded); test comes from data/test.
    per_class / test_per_class = 0 means 'use everything'; otherwise a seeded random subset per class.
    """
    rng = np.random.RandomState(seed)
    out = {"train": [], "val": [], "test": []}
    for src_split, limit in (("train", per_class), ("test", test_per_class)):
        items = list_split(src_split, data_dir)
        for label in (0, 1):
            cls_items = [it for it in items if it[1] == label]
            order = rng.permutation(len(cls_items))
            if limit:
                order = order[:limit]
            chosen = [cls_items[i] for i in order]
            if src_split == "train":
                n_val = int(round(len(chosen) * val_fraction))
                out["val"] += chosen[:n_val]
                out["train"] += chosen[n_val:]
            else:
                out["test"] += chosen
    # leakage guard: no file may appear in two splits
    names = {s: {str(p) for p, _ in v} for s, v in out.items()}
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"]) and not (names["val"] & names["test"]), "split overlap!"
    return out


def load_image(path, size=IMG_SIZE):
    """Any image file / file-like -> RGB -> size x size uint8 array (bilinear). Used by training AND the app."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def cached_arrays(items, name, cache_dir=CACHE_DIR):
    """Decode + resize all images once into a memory-mapped .npy cache (keeps RAM use low). Returns (x, y)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fx, fy, fl = cache_dir / f"{name}_x.npy", cache_dir / f"{name}_y.npy", cache_dir / f"{name}_files.json"
    files = [str(p) for p, _ in items]
    if not (fx.exists() and fy.exists() and fl.exists() and json.load(open(fl)) == files):
        mm = np.lib.format.open_memmap(fx, mode="w+", dtype=np.uint8, shape=(len(items), IMG_SIZE, IMG_SIZE, 3))
        for i, (p, _) in enumerate(items):
            mm[i] = load_image(p)
        mm.flush()
        del mm
        np.save(fy, np.array([l for _, l in items], dtype="float32"))
        json.dump(files, open(fl, "w"))
    return np.load(fx, mmap_mode="r"), np.load(fy)


def build_augmentation():
    """Label-preserving augmentation for TRAINING images only: horizontal flip, ~+/-14 deg rotation,
    10% zoom, 5% translation. No vertical flip, strong crop or colour distortion (these could change the look of a class)."""
    from tensorflow import keras
    L = keras.layers
    return keras.Sequential([
        L.RandomFlip("horizontal", seed=SEED),
        L.RandomRotation(0.04, fill_mode="reflect", seed=SEED),
        L.RandomZoom(0.1, fill_mode="reflect", seed=SEED),
        L.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
    ], name="augmentation")


def make_dataset(x, y, batch_size=32, training=False, seed=SEED):
    """tf.data pipeline over (memory-mapped) arrays. Pixels stay 0-255 float32; the MODEL rescales them to [-1, 1]
    (identical to keras.applications.mobilenet_v2.preprocess_input), so training and the app use the same preprocessing.
    Augmentation is applied only when training=True."""
    import tensorflow as tf
    n = len(x)
    rng = np.random.RandomState(seed)

    def gen():
        idx = rng.permutation(n) if training else np.arange(n)
        for i in range(0, n, batch_size):
            b = np.sort(idx[i:i + batch_size])
            yield np.asarray(x[b], dtype=np.float32), y[b]

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32), tf.TensorSpec((None,), tf.float32)))
    if training:
        aug = build_augmentation()
        ds = ds.map(lambda a, b: (aug(a, training=True), b))
    return ds.prefetch(1)
