# Dataset

The images are **not** stored in this project (about 600 MB). Create them with one command from the project root:

```bash
python scripts/prepare_dataset.py
```

This clones the public GitHub repository `laxmimerit/dog-cat-full-dataset` (a re-packaged copy of the Kaggle
"Dogs vs. Cats" images, which come from the Asirra data set of Microsoft Research), decodes every image,
removes unreadable files and exact duplicates (SHA-256), and writes:

```text
data/
├── train/cat/   train/dog/     (used for training + validation; validation is a seeded 80/20 split made in src/data_utils.py)
├── test/cat/    test/dog/      (held-out; only used by src/evaluate.py)
└── dataset_report.json         (real counts, image-size statistics, corrupt/duplicate/leakage checks)
```

If you already have the images locally in `train/{cats,dogs}` and `test/{cats,dogs}` folders, use
`python scripts/prepare_dataset.py --source <that folder>`.

## Source and terms of use
* Origin: Asirra CAPTCHA photographs (Microsoft Research / Petfinder.com), distributed on Kaggle as the "Dogs vs. Cats" competition data.
* The GitHub copy has **no licence file of its own**; the Kaggle competition rules limit the images to non-commercial / academic use.
  This project uses them only for an academic course assignment. Check the original terms before any other use.
* Reference: Elson, Douceur, Howell, Saul (2007), *Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization*, ACM CCS.
