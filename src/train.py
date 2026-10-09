"""Two-stage MobileNetV2 transfer learning for cat vs dog classification.

Stage 1  feature extraction: ImageNet-pretrained backbone FROZEN, only the new head (GAP -> Dropout -> Dense sigmoid) is trained.
Stage 2  fine-tuning: the upper backbone blocks (from --fine_tune_from) are unfrozen and trained with a much lower learning rate.
         BatchNormalization layers stay frozen (their ImageNet statistics are kept).

The model is selected on VALIDATION loss only. This script never reads the test set.
Interrupted runs resume from the last finished epoch (files in checkpoints/).

Examples (from the project root):
    python src/train.py --run pilot --per_class 1500 --epochs_head 2 --epochs_ft 1    # quick pilot
    python src/train.py --run final                                                  # all images, longer
"""
import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402

CKPT = du.ROOT / "checkpoints"
WEIGHTS_URL = ("https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/"
               "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")


def model_path(run):
    """final run -> models/cat_dog_mobilenetv2.keras ; any other run -> models/<run>_cat_dog_mobilenetv2.keras"""
    return du.MODELS_DIR / ("cat_dog_mobilenetv2.keras" if run == "final" else f"{run}_cat_dog_mobilenetv2.keras")


def get_backbone(weights_file=None):
    """ImageNet-pretrained MobileNetV2 without its classifier. Returns (model, honest description of the weight source).
    Tries: --weights file, default Keras download, then the same ImageNet weights from a GitHub release.
    Raises if none is reachable: we never silently train from random weights and call it transfer learning."""
    shape = (du.IMG_SIZE, du.IMG_SIZE, 3)
    if weights_file and Path(weights_file).exists():
        return keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=str(weights_file)), f"local file {weights_file}"
    try:
        return keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet"), "keras.applications default ImageNet weights"
    except Exception as e:
        print(f"Default weight download failed ({type(e).__name__}); trying GitHub release mirror")
    CKPT.mkdir(exist_ok=True)
    dst = CKPT / "mobilenet_v2_1.0_224_no_top.h5"
    for attempt in range(1, 4):                      # retry: a one-off network error should not stop training
        if dst.exists():
            break
        try:
            req = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0 (cat-dog-pbl)"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                f.write(r.read())
        except Exception as e:
            print(f"weights download attempt {attempt}/3 failed: {e}")
            dst.unlink(missing_ok=True)
            time.sleep(3 * attempt)
    if not dst.exists():
        raise RuntimeError("Could not obtain ImageNet weights. Download the file from " + WEIGHTS_URL +
                           " and run with --weights <file>. (Not training from random weights.)")
    return keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=str(dst)), \
        "GitHub release v1.1 of JonathanCMitchell/mobilenet_v2_keras (ImageNet weights, alpha=1.0, 224x224, no top)"


def build_model(weights_file=None, dropout=0.2):
    """raw 0-255 RGB -> Rescaling to [-1,1] (= mobilenet_v2.preprocess_input) -> MobileNetV2 -> GAP -> Dropout -> Dense(1, sigmoid) = P(dog)."""
    base, source = get_backbone(weights_file)
    base.trainable = False
    inp = keras.Input((du.IMG_SIZE, du.IMG_SIZE, 3), name="image")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenetv2_preprocess")(inp)
    x = base(x, training=False)                 # training=False keeps BatchNorm in inference mode
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(dropout, seed=du.SEED, name="dropout")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="dog_probability")(x)
    return keras.Model(inp, out, name="cat_dog_mobilenetv2"), base, source


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])


def unfreeze_top(base, from_layer):
    """Make backbone layers from `from_layer` on trainable, except BatchNormalization layers. Returns (start_idx, n_trainable, n_total)."""
    base.trainable = True
    start = [l.name for l in base.layers].index(from_layer)
    for i, l in enumerate(base.layers):
        l.trainable = i >= start and not isinstance(l, keras.layers.BatchNormalization)
    return start, sum(l.trainable for l in base.layers), len(base.layers)


class EpochLog(keras.callbacks.Callback):
    """After every epoch: save the full model (for resuming) and append the metrics to a json file."""
    def __init__(self, last_path, hist_path, history):
        super().__init__()
        self.last_path, self.hist_path, self.history = last_path, hist_path, history

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            self.history.setdefault(k, []).append(float(v))
        self.model.save(self.last_path)
        json.dump(self.history, open(self.hist_path, "w"))


def run_stage(model, train_ds, val_ds, stage, run, epochs, lr, patience):
    """Train one stage (resumable). Returns (history dict, best_epoch_index, seconds)."""
    last, hist_f = CKPT / f"{run}_stage{stage}_last.keras", CKPT / f"{run}_stage{stage}_history.json"
    best_w = CKPT / f"{run}_stage{stage}_best.weights.h5"
    history, start_epoch, seconds = {}, 0, 0.0
    if last.exists() and hist_f.exists():
        history = json.load(open(hist_f))
        start_epoch = len(history["loss"])
        print(f"[stage {stage}] resuming after epoch {start_epoch}")
        if start_epoch < epochs:
            model.load_weights(last)
    if start_epoch < epochs:
        compile_model(model, lr)
        t = time.time()
        cbs = [keras.callbacks.ModelCheckpoint(str(best_w), monitor="val_loss", save_best_only=True, save_weights_only=True),
               keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=False, verbose=1),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1),
               EpochLog(last, hist_f, history)]
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=start_epoch, callbacks=cbs, verbose=2)
        seconds = time.time() - t
    best_epoch = int(np.argmin(history["val_loss"]))
    model.load_weights(best_w)                         # validation-based selection inside the stage
    return history, best_epoch, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final", help="run name: 'pilot', 'final', ...")
    ap.add_argument("--per_class", type=int, default=0, help="train+val images per class from data/train (0 = all)")
    ap.add_argument("--test_per_class", type=int, default=0, help="test images per class (only recorded here; used by evaluate.py)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_head", type=int, default=3)
    ap.add_argument("--epochs_ft", type=int, default=3)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--fine_tune_from", default="block_13_expand")
    ap.add_argument("--weights", default=os.environ.get("MOBILENET_WEIGHTS"), help="optional local MobileNetV2 no-top .h5 file")
    ap.add_argument("--fresh", action="store_true", help="ignore resume files and start over")
    a = ap.parse_args()

    keras.utils.set_random_seed(du.SEED)
    res = du.RESULTS_DIR / a.run
    res.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    du.MODELS_DIR.mkdir(exist_ok=True)
    if a.fresh:
        for f in CKPT.glob(f"{a.run}_stage*"):
            f.unlink()
    t0 = time.time()

    splits = du.make_splits(per_class=a.per_class, test_per_class=a.test_per_class)
    sizes = {s: len(v) for s, v in splits.items()}
    counts = {s: {"cat": sum(1 for _, l in v if l == 0), "dog": sum(1 for _, l in v if l == 1)} for s, v in splits.items()}
    print("split sizes:", sizes, counts)
    xtr, ytr = du.cached_arrays(splits["train"], f"pc{a.per_class}_train")
    xva, yva = du.cached_arrays(splits["val"], f"pc{a.per_class}_val")
    train_ds = du.make_dataset(xtr, ytr, a.batch_size, training=True)
    val_ds = du.make_dataset(xva, yva, a.batch_size)

    model, base, source = build_model(a.weights)
    print("weights:", source, "| total params:", model.count_params())

    # ---- Stage 1: frozen backbone
    h1, b1, t1 = run_stage(model, train_ds, val_ds, 1, a.run, a.epochs_head, a.lr_head, a.patience)
    model.save(CKPT / f"{a.run}_stage1_best.keras")

    # ---- Stage 2: fine-tune upper blocks (BatchNorm frozen)
    start, n_train, n_total = unfreeze_top(base, a.fine_tune_from)
    print(f"fine-tuning backbone layers {start}..{n_total - 1}: {n_train} trainable of {n_total} (BatchNorm frozen)")
    h2, b2, t2 = run_stage(model, train_ds, val_ds, 2, a.run, a.epochs_ft, a.lr_ft, a.patience)
    model.save(CKPT / f"{a.run}_stage2_best.keras")

    # ---- model selection on validation loss (never on test data)
    v1, v2 = h1["val_loss"][b1], h2["val_loss"][b2]
    chosen = 2 if v2 < v1 else 1
    shutil.copy(CKPT / f"{a.run}_stage{chosen}_best.keras", model_path(a.run))

    with open(res / "training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "epoch_in_stage", "epoch_overall", "loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"])
        k = 0
        for stage, h in ((1, h1), (2, h2)):
            for i in range(len(h["loss"])):
                k += 1
                w.writerow([stage, i + 1, k, h["loss"][i], h["accuracy"][i], h["val_loss"][i], h["val_accuracy"][i], h.get("learning_rate", [None] * len(h["loss"]))[i]])
    summary = {
        "run": a.run, "model_file": str(model_path(a.run).relative_to(du.ROOT)),
        "selected_stage": "stage 2 (fine-tuned)" if chosen == 2 else "stage 1 (frozen backbone)",
        "selection_rule": "lowest validation loss over the best epoch of each stage; test set not used",
        "stage1": {"best_epoch": b1 + 1, "epochs_run": len(h1["loss"]), "val_loss": v1, "val_accuracy": h1["val_accuracy"][b1], "train_accuracy_at_best": h1["accuracy"][b1], "train_loss_at_best": h1["loss"][b1], "seconds_this_session": round(t1)},
        "stage2": {"best_epoch": b2 + 1, "epochs_run": len(h2["loss"]), "val_loss": v2, "val_accuracy": h2["val_accuracy"][b2], "train_accuracy_at_best": h2["accuracy"][b2], "train_loss_at_best": h2["loss"][b2], "seconds_this_session": round(t2)},
        "config": {"image_size": du.IMG_SIZE, "batch_size": a.batch_size, "dropout": 0.2, "optimizer": "Adam", "loss": "BinaryCrossentropy",
                   "lr_head": a.lr_head, "lr_finetune": a.lr_ft, "epochs_head_max": a.epochs_head, "epochs_finetune_max": a.epochs_ft,
                   "early_stopping_patience": a.patience, "fine_tune_from": a.fine_tune_from, "backbone_layers": n_total,
                   "backbone_trainable_stage2": n_train, "seed": du.SEED, "per_class": a.per_class, "test_per_class": a.test_per_class,
                   "augmentation": "horizontal flip, rotation 0.04 turn (~14 deg), zoom 10%, translation 5% (training only)"},
        "split_sizes": sizes, "class_counts": counts, "weight_source": source, "total_params": int(model.count_params()),
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "gpu": bool(tf.config.list_physical_devices("GPU")), "cpu_cores": os.cpu_count()},
        "wall_clock_seconds_this_session": round(time.time() - t0),
        "note": "seconds_* only count training done in this session if the run was resumed",
    }
    json.dump(summary, open(res / "training_summary.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("selected_stage", "stage1", "stage2")}, indent=2))
    print("DONE. Model saved to", model_path(a.run))


if __name__ == "__main__":
    main()
