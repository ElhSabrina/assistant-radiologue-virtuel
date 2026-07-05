# RSNA Pneumonia Detection 2018: dataset setup

This directory is the target for the preprocessed RSNA dataset.  
The raw DICOM files are **not** committed to the repository (too large, licence restrictions).

---

## 1. Download the dataset

Go to the Kaggle page and download:

```
https://www.kaggle.com/datasets/sovitrath/rsna-pneumonia-detection-2018
```

Requires a free Kaggle account.  
Alternative: install the Kaggle CLI and run:

```bash
pip install kaggle
kaggle datasets download -d sovitrath/rsna-pneumonia-detection-2018
unzip rsna-pneumonia-detection-2018.zip -d /path/to/rsna
```

---

## 2. Expected raw layout

After extraction the directory must contain at minimum:

```
<rsna_dir>/
  stage_2_detailed_class_info.csv   # patientId, class (3 classes)
  stage_2_train_labels.csv          # patientId, x, y, width, height, Target
  stage_2_train_images/
    *.dcm                           # ~26 000 DICOM files
```

---

## 3. Run the preprocessing script

```bash
python scripts/prepare_rsna_dataset.py \
    --rsna-dir /path/to/rsna \
    --out-dir  data/rsna \
    --db-path  medical_ai_evidence.sqlite \
    --smoke-n  20 \
    --dev-n    150 \
    --final-n  30 \
    --seed     42
```

The script will:

| Step | Action |
|------|--------|
| 1 | Load `stage_2_detailed_class_info.csv` + `stage_2_train_labels.csv` |
| 2 | Map RSNA classes → project 3-class ontology (see table below) |
| 3 | Build balanced splits (final held-out first, then smoke, then dev) |
| 4 | For each selected DICOM: normalize → CLAHE → resize 512×512 → save PNG |
| 5 | Write `data/rsna/cases.csv` and insert into `medical_ai_evidence.sqlite` |

---

## 4. Class mapping

| RSNA class | Project label |
|---|---|
| Normal | `normal` |
| Lung Opacity | `suspected_opacity` |
| No Lung Opacity / Not Normal | `uncertain` |

---

## 5. Output structure (generated)

```
data/rsna/
  cases.csv                        # case registry with splits + quality flags
  processed/
    images/
      RSNA_<patientId>.png         # 512x512 RGB, CLAHE-enhanced
```

---

## 6. Preprocessing pipeline (per image)

```
DICOM pixel array
  → DICOM windowing (WindowCenter/WindowWidth if present)
  → Min-max normalize to [0, 255] uint8
  → MONOCHROME1 flip if applicable
  → Quality assessment (contrast, brightness, sharpness)
  → CLAHE (clipLimit=2.0, tileGridSize=8×8)
  → Lanczos resize 512×512
  → Grayscale → RGB (3-channel repeat for VLM compatibility)
  → Save as PNG
```

Quality flags: `good` | `limited` (1 issue) | `poor` (2+ issues).  
Issues detected: `low_contrast`, `underexposed`, `overexposed`, `blurry`.

---

## 7. Reproducibility

The splits are deterministic given `--seed`.  
The exact subset used at each phase is recorded in `cases.csv` and in the SQLite `cases` table, satisfying the traceability requirement of the AI Act audit trail.

To reproduce exactly:

```bash
python scripts/prepare_rsna_dataset.py --rsna-dir <same_dir> --seed 42
```
