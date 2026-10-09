# Verification report

Everything below was executed in this workspace (Python 3.13.16, TensorFlow 2.21.0, Keras 3.15.1, 2 CPU cores, no GPU).

## Completed and checked
| Item | Result |
|---|---|
| Dataset download | `git clone` of laxmimerit/dog-cat-full-dataset succeeded; 24,984 images kept |
| Corrupt images | 0 found (every image fully decoded) |
| Duplicates / leakage | 5 exact duplicates removed; re-hash of the output folders: 0 duplicates; train/val/test file overlap asserted to be empty in code |
| Pretrained weights | ImageNet MobileNetV2 (no top) loaded from a local copy of the GitHub release file; default Keras download host is not reachable from this environment |
| Automatic weight download | with no local file, the default Keras host failed (not reachable here), then the GitHub-release fallback downloaded and loaded the same ImageNet weights |
| Pilot training | executed; log in `results/pilot/training_log.csv` |
| Checkpoint + reload | model saved and re-loaded (input (None,224,224,3), output (None,1)) |
| Held-out evaluation | `python src/evaluate.py --run pilot` executed; metrics in `results/pilot/test_metrics.json` |
| Streamlit app | launched headless and driven by a real Chromium browser: uploaded `data/test/cat/cat.0.jpg` -> CAT 98.31%; uploaded `data/test/dog/dog.0.jpg` -> DOG 99.97%; uploaded a text file renamed .jpg -> error message shown (screenshots in `results/app_test/`) |
| CLI vs app | `src/predict.py` gives the same two predictions as the app |
| Syntax | all Python files parsed without errors |

## Not done / limitations
* The longer final training run is NOT part of this package; all reported numbers are from the pilot experiment.
* The Jupyter notebook is not included yet.
* The browser test covered only the images listed above (one cat, one dog, one invalid file), not a systematic app test.
* Single run and single random seed; GPU was not available.
* `requirements.txt` was tested only with the versions listed at the top of this file.
