"""Cat & Dog AI Classifier - Streamlit app. Run:  streamlit run app.py
Every number shown comes from the saved MobileNetV2 model; nothing is hard-coded or random."""
import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import default_model_path, load_trained_model, predict_image  # noqa: E402

st.set_page_config(page_title="Cat & Dog AI Classifier", page_icon="🐾", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 860px; padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg, #1B3A6B 0%, #2F6DB5 100%); color: white; padding: 24px 28px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: white;} .hero p {margin: 6px 0 0; opacity: .92;}
.result {border-radius: 14px; padding: 16px 20px; text-align: center; border: 1px solid #dfe5ee; background: #fff;}
.result .tag {font-size: .8rem; color: #6b7686; letter-spacing: .08em;} .result .label {font-size: 2.5rem; font-weight: 800; margin: 2px 0;}
.result .conf {font-size: 1.1rem; color: #333;}
.bar-row {display: flex; align-items: center; margin: 8px 0; font-size: .95rem;} .bar-name {width: 44px; font-weight: 600;}
.bar-bg {flex: 1; background: #edf1f7; border-radius: 8px; height: 20px; overflow: hidden; margin: 0 10px;} .bar-fill {height: 100%; border-radius: 8px;}
.bar-val {width: 64px; text-align: right; font-variant-numeric: tabular-nums;}
</style>
<div class="hero"><h1>🐾 Cat &amp; Dog AI Classifier</h1><p>Deep Learning-Based Image Classification Using Transfer Learning (MobileNetV2)</p></div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model(path):
    return load_trained_model(path)


model_file = default_model_path()
if model_file is None:
    st.error("No trained model found in the `models/` folder. Train one first with `python src/train.py --run final` (see README.md).")
    st.stop()
try:
    model = get_model(str(model_file))
except Exception as e:  # corrupt / incompatible model file
    st.error(f"The model file `{model_file.name}` could not be loaded: {e}")
    st.stop()

upload = st.file_uploader("Upload a cat or dog photo (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"])
if upload is None:
    st.info("Upload an image to get a prediction. You can upload another image afterwards.")
else:
    try:
        image = Image.open(upload)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG file.")
        st.stop()
    with st.spinner("Running MobileNetV2..."):
        res = predict_image(model, image)
    left, right = st.columns(2, gap="large")
    with left:
        st.image(image.convert("RGB"), caption=f"Uploaded image ({image.size[0]}x{image.size[1]} px)", use_container_width=True)
    with right:
        color = "#2F6DB5" if res["label"] == "DOG" else "#E07B39"
        st.markdown(f"""
        <div class="result"><div class="tag">PREDICTION</div>
        <div class="label" style="color:{color}">{'🐶' if res['label'] == 'DOG' else '🐱'} {res['label']}</div>
        <div class="conf">Confidence: <b>{res['confidence']:.2f}%</b></div></div>
        <div style="margin-top:10px">
          <div class="bar-row"><div class="bar-name">Cat</div><div class="bar-bg"><div class="bar-fill" style="width:{res['p_cat']:.2f}%;background:#E07B39"></div></div><div class="bar-val">{res['p_cat']:.2f}%</div></div>
          <div class="bar-row"><div class="bar-name">Dog</div><div class="bar-bg"><div class="bar-fill" style="width:{res['p_dog']:.2f}%;background:#2F6DB5"></div></div><div class="bar-val">{res['p_dog']:.2f}%</div></div>
        </div>""", unsafe_allow_html=True)
        if res["confidence"] < 70:
            st.warning("Low confidence: the image may be ambiguous, unusual, or not a cat/dog photo.")
    st.caption("The model only knows two classes, so any image (even a non-animal) is forced into CAT or DOG.")

st.warning("This model is designed for educational and demonstration purposes.")
with st.expander("Model information"):
    st.markdown(f"""
- **Architecture:** MobileNetV2 (ImageNet-pretrained backbone) → Global Average Pooling → Dropout(0.2) → Dense(1, sigmoid) = P(dog)
- **Input:** RGB image resized to 224 × 224 × 3, scaled to [-1, 1] inside the model (same preprocessing as training)
- **Framework:** TensorFlow / Keras · **Task:** binary image classification (cat vs dog)
- **Model file in use:** `{model_file.name}`
""")
    for run in ("final", "pilot"):
        f = ROOT / "results" / run / "test_metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            st.markdown(f"**Measured on held-out test images ({run} run, n={m['n']}):** accuracy {m['accuracy']*100:.2f}%, "
                        f"precision {m['precision_dog']*100:.2f}%, recall {m['recall_dog']*100:.2f}%, F1 {m['f1_dog']*100:.2f}%, ROC-AUC {m['roc_auc']:.4f}")
            break
