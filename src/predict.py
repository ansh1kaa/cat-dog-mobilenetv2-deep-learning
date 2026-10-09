"""Single-image prediction with the saved model (used by app.py and usable from the command line).

    python src/predict.py path/to/image.jpg
    python src/predict.py path/to/image.jpg --model models/cat_dog_mobilenetv2.keras

The model has ONE sigmoid output = P(dog); P(cat) = 1 - P(dog); the label is DOG if P(dog) >= 0.5.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402


def default_model_path():
    """Prefer the final model; fall back to the pilot model if the final run has not been done yet."""
    for name in ("cat_dog_mobilenetv2.keras", "pilot_cat_dog_mobilenetv2.keras"):
        p = du.MODELS_DIR / name
        if p.exists():
            return p
    return None


def load_trained_model(path=None):
    from tensorflow import keras
    path = Path(path) if path else default_model_path()
    if path is None or not path.exists():
        raise FileNotFoundError("No trained model found in models/. Train one with: python src/train.py --run final")
    return keras.models.load_model(path)


def prepare(image):
    """Path / file-like / PIL image -> (1, 224, 224, 3) float32 in 0-255, the same loading code used for training."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB").resize((du.IMG_SIZE, du.IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
    else:
        arr = du.load_image(image)
    return arr[None].astype("float32")


def predict_image(model, image):
    p_dog = float(model.predict(prepare(image), verbose=0).reshape(-1)[0])
    label = "DOG" if p_dog >= 0.5 else "CAT"
    return {"label": label, "confidence": (p_dog if label == "DOG" else 1 - p_dog) * 100, "p_dog": p_dog * 100, "p_cat": (1 - p_dog) * 100}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    r = predict_image(load_trained_model(a.model), a.image)
    print(f"Prediction: {r['label']} (confidence {r['confidence']:.2f}%)  |  cat {r['p_cat']:.2f}%  dog {r['p_dog']:.2f}%")
