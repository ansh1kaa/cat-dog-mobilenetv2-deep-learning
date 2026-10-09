# Cat and Dog Image Classification Using MobileNetV2 Transfer Learning

Deep Learning (Computer Vision) · TensorFlow/Keras · Streamlit · Python

**Status: pilot version.** Only the small pilot experiment has been trained and evaluated so far; the longer final run is not part of this package yet.

## Abstract
A convolutional neural network (MobileNetV2, pretrained on ImageNet) is adapted to classify photographs as **cat** or **dog**. Stage 1 trains only a new classification head on a frozen backbone; stage 2 fine-tunes the upper backbone blocks with a low learning rate. The model is chosen on validation loss, evaluated once on a held-out test set (accuracy, precision, recall, F1, confusion matrix, ROC-AUC), and served through a Streamlit web app.

## Problem statement and objectives
Classify an RGB photograph as cat or dog accurately with a model small enough to train and run on a CPU.
1. Build a reproducible data pipeline (download, validation, duplicate/leakage checks, seeded splits).
2. Train MobileNetV2 with transfer learning, then fine-tune, with validation-based model selection.
3. Evaluate honestly on a held-out test set and analyse the mistakes.
4. Provide a working image-upload application that uses the saved model.

## Dataset (measured, not assumed)
* Source: Kaggle "Dogs vs. Cats" images (Asirra / Microsoft Research), obtained from the public GitHub repository `laxmimerit/dog-cat-full-dataset`. Terms of use: see `data/README.md` (academic, non-commercial use).
* After `scripts/prepare_dataset.py`: **24,984 images**, all RGB, 0 unreadable files, **5 exact duplicates removed** (byte-identical files; two of them appeared in both the train and test folders, which would have been leakage), re-hash of the folders found 0 remaining duplicates.
* Train folder: 9,998 cats / 9,989 dogs. Test folder: 2,499 cats / 2,498 dogs.
* Image sizes vary (width 42-1050 px, height 32-768 px), so every image is resized to 224 × 224.
* Validation images are a seeded, stratified 20% of the train folder (made in `src/data_utils.py`); the test folder is never used for training or model selection.
* This differs from the ~23,262-image TensorFlow Datasets version; counts here come from `data/dataset_report.json`.

## Method
**Deep learning concepts.** A CNN learns filters by *backpropagation*: the loss (binary cross-entropy) is differentiated with respect to every trainable weight, and Adam updates the weights. Early layers detect edges and textures, later layers detect object parts.
**MobileNetV2.** Built from *depthwise separable convolutions* (a per-channel 3×3 convolution followed by a 1×1 pointwise convolution, far cheaper than a full convolution) arranged in *inverted residual blocks* (1×1 expand → 3×3 depthwise → 1×1 linear projection, with a shortcut between the narrow ends). The reference network has about 3.4 M parameters (Sandler et al., 2018).
**Transfer learning.** *Feature extraction* (stage 1): reuse ImageNet features, train only the head. *Fine-tuning* (stage 2): unfreeze the upper blocks (from layer `block_13_expand`) and train them at a 20× lower learning rate; BatchNormalization layers stay frozen.
**Architecture:** `Input 224×224×3 → Rescaling to [-1,1] (= MobileNetV2 preprocessing) → MobileNetV2 → GlobalAveragePooling → Dropout(0.2) → Dense(1, sigmoid)`.
**Binary output.** One sigmoid unit gives P(dog); P(cat) = 1 − P(dog); label = dog if P(dog) ≥ 0.5. Label order everywhere: 0 = cat, 1 = dog.
**Augmentation (training images only):** horizontal flip, rotation ≈ ±14°, zoom 10%, translation 5%.
**Training:** Adam, binary cross-entropy, batch size 32, learning rate 0.001 (stage 1) and 5e-05 (stage 2), EarlyStopping (patience 2), ModelCheckpoint, ReduceLROnPlateau, seed 42. The pretrained weights were loaded from: local file /home/claude/w/mnv2_notop.h5.

## Results (all numbers come from files in `results/`)
| Run | Train / val / test images | Epochs (stage 1 + 2) | Selected | Test accuracy | Precision (dog) | Recall (dog) | F1 (dog) | ROC-AUC | Errors |
|---|---|---|---|---|---|---|---|---|---|
| pilot | 2,400 / 600 / 1,000 | 2 + 1 | stage 2 (fine-tuned) | 97.20% | 94.70% | 100.00% | 97.28% | 0.9992 | 28 of 1,000 |

Precision/recall/F1 are for the **dog** class (positive class). Test sets: the pilot used a seeded random subset of 500 cats + 500 dogs from the test folder. Pilot and final numbers must not be mixed up: they are different experiments.

### Stage comparison (pilot run)
| Stage | Best epoch | Val loss | Val accuracy | Test accuracy (reporting only) | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| 1 (frozen backbone) | 2 of 2 | 0.0593 | 98.17% | 98.00% | 97.99% | 0.9989 |
| 2 (fine-tuned) | 1 of 1 | 0.0499 | 98.00% | 97.20% | 97.28% | 0.9992 |

The stage was selected by **validation loss only**. The test columns are shown for transparency and were not used to choose anything.

### Confusion matrix (pilot run, test set)
cats: 472 correct, 28 predicted as dog; dogs: 500 correct, 0 predicted as cat. Latency measured on this CPU machine: 33.8 ms per image (batch of 1). See `results/pilot/` for `confusion_matrix.png`, `roc_curve.png`, `accuracy_curve.png`, `loss_curve.png`, `sample_predictions.png`, `misclassified_examples.png`, `misclassified_files.csv`, `training_log.csv`, `training_summary.json`, `test_metrics.json`.

### Honest interpretation
* Pilot only: just 3 training epochs on 2,400 training images, one fine-tuning epoch. These numbers show that the pipeline works, not the best this model can do.
* In the pilot run the selected model makes errors that are all of one kind (cats: 472 correct, 28 predicted as dog; dogs: 500 correct, 0 predicted as cat), i.e. it is biased towards one class at the 0.5 threshold, while the ROC-AUC (0.9992) shows the two classes are ranked almost perfectly. The threshold was deliberately **not** tuned on the test set.
* Inspecting `misclassified_examples.png` shows hard images: dark or blurry photos, animals held by people, cages, several animals, or an animal that is small in the frame.
* Single run, single seed, CPU training: differences of a few tenths of a percentage point are not conclusive.

## Installation and usage (tested with Python 3.13.16, TensorFlow 2.21.0, Keras 3.15.1)
```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py                 # downloads + validates the data (needs git and internet)
python src/train.py --run pilot --per_class 1500 --test_per_class 500 --epochs_head 2 --epochs_ft 1   # ~5 min on a CPU
python src/train.py --run final                    # all images, longer (see "Training time" below)
python src/evaluate.py --run pilot                 # or --run final; writes metrics and plots to results/<run>/
streamlit run app.py                               # the web app (uses models/cat_dog_mobilenetv2.keras, else the pilot model)
python src/predict.py path/to/image.jpg            # command-line prediction
```
Pretrained weights are fetched automatically (default Keras download, with a GitHub-release fallback). If both fail, `train.py` stops with an error instead of silently training from random weights. On Windows you can also double-click `run_app.bat`.
Interrupted training resumes automatically from `checkpoints/` when you run the same command again (`--fresh` starts over).

Training time (this project's 2-core CPU machine): about 75 s per epoch on 2,400 training images, so roughly 8-10 minutes per epoch on the full ~16,000-image training split.

## Project structure
```text
app.py                      Streamlit app
src/data_utils.py           split, loading, caching, augmentation
src/train.py                two-stage training (never reads the test set)
src/evaluate.py             held-out evaluation, ROC-AUC, plots
src/predict.py              single-image prediction
scripts/prepare_dataset.py  dataset download + validation
models/                     saved Keras model(s)
results/<run>/              metrics, logs and plots of each run (pilot, final)
results/app_test/           screenshots of the real browser test of the app
data/                       dataset folder (images created by the script) + dataset_report.json

VERIFICATION.md             what was actually tested
```

## Limitations
* Only two classes: every uploaded picture is forced into cat or dog.
* CPU-only, few epochs, one seed; no hyper-parameter search; no k-fold cross-validation.
* Images are mostly clear pet photos; behaviour on other kinds of images is not measured.
* Resizing to 224×224 ignores the aspect ratio.

## Future scope
Several seeds with confidence intervals; larger backbones (EfficientNet, ResNet50) under the same split; a rejection class for non-cat/dog images; Grad-CAM heat-maps; threshold calibration on the validation set; TensorFlow Lite export.

## References
* Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR. arXiv:1801.04381
* Howard et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861
* Yosinski, Clune, Bengio, Lipson (2014). How transferable are features in deep neural networks? NeurIPS. arXiv:1411.1792
* Deng et al. (2009). ImageNet: A large-scale hierarchical image database. CVPR.
* Elson, Douceur, Howell, Saul (2007). Asirra: A CAPTCHA that exploits interest-aligned manual image categorization. ACM CCS.
