"""Evaluate a trained model on the HELD-OUT test set and write metrics + plots to results/<run>/.

Run this only after training / model selection is finished. Every number is computed from the model's predictions and the true labels.

    python src/evaluate.py --run pilot
    python src/evaluate.py --run final
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402
from train import model_path  # noqa: E402

BLUE, ORANGE, GREEN, RED = "#2F6DB5", "#E07B39", "#2E8B57", "#B3261E"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})


def predict_probs(model, x, batch_size=32):
    """P(dog) for every image (uint8 array in, probability out). Rescaling to [-1, 1] happens inside the model."""
    return np.concatenate([model.predict_on_batch(np.asarray(x[i:i + batch_size], dtype="float32")).reshape(-1)
                           for i in range(0, len(x), batch_size)])


def metrics(y, p):
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "precision_dog": float(precision_score(y, pred, zero_division=0)), "recall_dog": float(recall_score(y, pred, zero_division=0)),
            "f1_dog": float(f1_score(y, pred, zero_division=0)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y, pred, target_names=["cat", "dog"], output_dict=True, zero_division=0),
            "n_errors": int((pred != y).sum())}


def read_log(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("loss", "accuracy", "val_loss", "val_accuracy")}, np.array([int(r["stage"]) for r in rows])


def plot_curves(res):
    h, stage = read_log(res / "training_log.csv")
    ep = np.arange(1, len(stage) + 1)
    for key, label, fname in (("accuracy", "Accuracy", "accuracy_curve.png"), ("loss", "Binary cross-entropy loss", "loss_curve.png")):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(ep, h[key], "o-", color=BLUE, label="Training")
        ax.plot(ep, h["val_" + key], "s--", color=ORANGE, label="Validation")
        if (stage == 2).any():
            ax.axvline(np.argmax(stage == 2) + 0.5, color="grey", ls=":", lw=1.3)
            ax.text(np.argmax(stage == 2) + 0.55, ax.get_ylim()[0] + 0.02 * np.ptp(ax.get_ylim()), "fine-tuning starts", color="grey", fontsize=9)
        ax.set_xlabel("Epoch (stage 1 = frozen backbone, stage 2 = fine-tuning)")
        ax.set_ylabel(label)
        ax.set_title(f"Training vs validation {label.lower()}")
        ax.set_xticks(ep)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(res / fname)
        plt.close(fig)


def plot_confusion(cm, res):
    cm = np.array(cm)
    pc = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(pc, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1], ["Cat", "Dog"])
    ax.set_yticks([0, 1], ["Cat", "Dog"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (held-out test set)", fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pc[i, j]:.1f}%)", ha="center", va="center", fontsize=13, color="white" if pc[i, j] > 50 else "black")
    for s in ax.spines.values():
        s.set_visible(True)
    fig.savefig(res / "confusion_matrix.png")
    plt.close(fig)


def plot_roc(y, p, auc, res):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"MobileNetV2 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate (cat predicted as dog)")
    ax.set_ylabel("True positive rate (dog recall)")
    ax.set_title("ROC curve (held-out test set)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(res / "roc_curve.png")
    plt.close(fig)


def _tiles(axes, idxs, x, y, p):
    for ax, i in zip(axes, idxs):
        pred = int(p[i] >= 0.5)
        conf = p[i] if pred else 1 - p[i]
        ok = pred == int(y[i])
        ax.imshow(x[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"True: {du.CLASS_NAMES[int(y[i])].upper()}\nPred: {du.CLASS_NAMES[pred].upper()} ({conf * 100:.1f}%)", fontsize=9, color=GREEN if ok else RED)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GREEN if ok else RED)
            s.set_linewidth(2.2)
    for ax in axes[len(idxs):]:
        ax.axis("off")


def plot_samples(x, y, p, res, files, seed=7):
    rng = np.random.RandomState(seed)
    pred = (p >= 0.5).astype(int)
    ok, bad = np.where(pred == y.astype(int))[0], np.where(pred != y.astype(int))[0]
    idx = list(rng.choice(ok, size=min(8, len(ok)), replace=False)) + list(bad[np.argsort(-np.abs(p[bad] - 0.5))][:4])
    rows = int(np.ceil(len(idx) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(11, rows * 3.0), squeeze=False)
    _tiles(list(axes.ravel()), idx, x, y, p)
    fig.suptitle("Sample test predictions (green = correct, red = wrong; % = confidence in the predicted class)", y=1.0)
    fig.tight_layout()
    fig.savefig(res / "sample_predictions.png")
    plt.close(fig)
    # misclassified gallery (most confident mistakes first) + list of files
    if len(bad):
        worst = bad[np.argsort(-np.abs(p[bad] - 0.5))]
        show = worst[:12]
        rows = int(np.ceil(len(show) / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(11, rows * 3.0), squeeze=False)
        _tiles(list(axes.ravel()), list(show), x, y, p)
        fig.suptitle("Misclassified test images, most confident mistakes first", y=1.0)
        fig.tight_layout()
        fig.savefig(res / "misclassified_examples.png")
        plt.close(fig)
        with open(res / "misclassified_files.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "true", "predicted", "confidence_percent"])
            for i in worst:
                pr = int(p[i] >= 0.5)
                w.writerow([files[i], du.CLASS_NAMES[int(y[i])], du.CLASS_NAMES[pr], round(float((p[i] if pr else 1 - p[i]) * 100), 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    from tensorflow import keras
    res = du.RESULTS_DIR / a.run
    summ = json.load(open(res / "training_summary.json"))
    cfg = summ["config"]
    mp = Path(a.model) if a.model else model_path(a.run)

    # exactly the same split as training (seeded); only the test part is read here
    splits = du.make_splits(per_class=cfg["per_class"], test_per_class=cfg["test_per_class"])
    xt, yt = du.cached_arrays(splits["test"], f"pc{cfg['test_per_class']}_test")
    files = [str(p.relative_to(du.DATA_DIR)) for p, _ in splits["test"]]

    model = keras.models.load_model(mp)
    prob = predict_probs(model, xt)
    m = metrics(yt, prob)
    # measured single-image inference latency on this machine (batch of 1, after warm-up)
    xs = np.asarray(xt[:1], dtype="float32")
    for _ in range(3):
        model.predict_on_batch(xs)
    t = time.perf_counter()
    for i in range(30):
        model.predict_on_batch(np.asarray(xt[i:i + 1], dtype="float32"))
    m["latency_ms_per_image_batch1"] = round((time.perf_counter() - t) / 30 * 1000, 1)
    m.update({"run": a.run, "model_file": str(mp.relative_to(du.ROOT)), "selected_stage": summ["selected_stage"],
              "test_class_counts": summ["class_counts"]["test"], "threshold": 0.5, "positive_class": "dog"})

    # report both stages on the test set for comparison only (the model was already chosen on validation loss)
    comp = {}
    for stage in (1, 2):
        sp = du.ROOT / "checkpoints" / f"{a.run}_stage{stage}_best.keras"
        if sp.exists():
            comp[f"stage{stage}"] = {k: v for k, v in metrics(yt, predict_probs(keras.models.load_model(sp), xt)).items() if k != "classification_report"}
    m["stage_comparison_test_reporting_only"] = comp
    json.dump(m, open(res / "test_metrics.json", "w"), indent=2)

    plot_curves(res)
    plot_confusion(m["confusion_matrix"], res)
    plot_roc(yt, prob, m["roc_auc"], res)
    plot_samples(xt, yt, prob, res, files)
    np.save(res / "test_probabilities.npy", prob.astype("float32"))
    print(json.dumps({k: m[k] for k in ("n", "accuracy", "precision_dog", "recall_dog", "f1_dog", "roc_auc", "confusion_matrix", "n_errors", "latency_ms_per_image_batch1")}, indent=2))


if __name__ == "__main__":
    main()
