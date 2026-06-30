"""Prepare a small, balanced RSNA Pneumonia evaluation set for the prototype.

Source: RSNA Pneumonia Detection Challenge (Kaggle competition
``rsna-pneumonia-detection-challenge``). Ground-truth class comes from
``stage_2_detailed_class_info.csv``:

    "Normal"                       -> normal
    "Lung Opacity"                 -> suspected_opacity
    "No Lung Opacity / Not Normal" -> ignored (ambiguous, not used as truth)

The script downloads the competition archive once (~3.7 GB, requires that you
have accepted the competition rules on Kaggle), extracts only the handful of
DICOMs it needs, converts them to 512x512 PNG, and writes ``data/rsna_cases.csv``
with the same columns as the synthetic set.

This is an educational prototype. RSNA data is licensed for the competition and
must not be redistributed; only derived PNGs of a few cases are produced locally.

Usage:
    python data/prepare_rsna.py --per-class 15
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "rsna_raw"
IMG_DIR = DATA_DIR / "rsna_images"
CASES_CSV = DATA_DIR / "rsna_cases.csv"

COMPETITION = "rsna-pneumonia-detection-challenge"
ARCHIVE = RAW_DIR / f"{COMPETITION}.zip"

# Pre-converted local RSNA export (PNG already extracted). If present, it is used
# directly and no Kaggle download is needed.
LOCAL_DIR = DATA_DIR / "rsna_pneumonia"
LOCAL_METADATA = LOCAL_DIR / "stage2_train_metadata.csv"
LOCAL_IMAGES = LOCAL_DIR / "Training" / "Images"

CLASS_MAP = {
    "Normal": "normal",
    "Lung Opacity": "suspected_opacity",
}


def download_archive() -> None:
    """Download the competition archive via the Kaggle API if not present."""
    if ARCHIVE.exists() and ARCHIVE.stat().st_size > 0:
        print(f"[rsna] archive already present: {ARCHIVE}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[rsna] downloading competition '{COMPETITION}' (~3.7 GB)...")
    api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    if not ARCHIVE.exists():
        # Some kaggle versions name the file differently; take the only zip.
        zips = list(RAW_DIR.glob("*.zip"))
        if zips:
            zips[0].rename(ARCHIVE)
    print(f"[rsna] archive ready: {ARCHIVE}")


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        return list(csv.DictReader(text))


def select_patients(zf: zipfile.ZipFile, per_class: int, seed: int) -> list[tuple[str, str]]:
    """Return a balanced list of (patientId, label) drawn from the class info."""
    rows = _read_csv_from_zip(zf, "stage_2_detailed_class_info.csv")
    buckets: dict[str, list[str]] = {"normal": [], "suspected_opacity": []}
    seen: set[str] = set()
    for row in rows:
        pid = row["patientId"]
        if pid in seen:
            continue
        seen.add(pid)
        label = CLASS_MAP.get(row["class"])
        if label:
            buckets[label].append(pid)

    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label, pids in buckets.items():
        rng.shuffle(pids)
        selected.extend((pid, label) for pid in pids[:per_class])
    rng.shuffle(selected)
    return selected


def dicom_bytes_to_png(raw: bytes, size: int = 512) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.resize((size, size))


def _write_cases(cases: list[dict], out: Path) -> None:
    fieldnames = ["case_id", "image_path", "source", "label", "split", "quality", "notes"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(cases)
    n_norm = sum(c["label"] == "normal" for c in cases)
    n_op = sum(c["label"] == "suspected_opacity" for c in cases)
    print(f"[rsna] wrote {len(cases)} cases ({n_norm} normal / {n_op} opacity) -> {out}")


def build_from_local(per_class: int, seed: int, out: Path) -> None:
    """Build the case set from an already-extracted local RSNA PNG export."""
    with LOCAL_METADATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    buckets: dict[str, list[str]] = {"normal": [], "suspected_opacity": []}
    seen: set[str] = set()
    for row in rows:
        pid = row["patientId"]
        if pid in seen:
            continue
        seen.add(pid)
        label = CLASS_MAP.get(row["class"])
        if label and (LOCAL_IMAGES / f"{pid}.png").exists():
            buckets[label].append(pid)

    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label, pids in buckets.items():
        rng.shuffle(pids)
        selected.extend((pid, label) for pid in pids[:per_class])
    rng.shuffle(selected)

    cases = []
    for idx, (pid, label) in enumerate(selected, start=1):
        case_id = f"RSNA_{idx:03d}_{label}"
        rel = (LOCAL_IMAGES / f"{pid}.png").relative_to(ROOT).as_posix()
        split = "smoke" if idx <= (2 * per_class) // 3 else "final"
        cases.append({
            "case_id": case_id,
            "image_path": rel,
            "source": "rsna_pneumonia_challenge",
            "label": label,
            "split": split,
            "quality": "good",
            "notes": f"RSNA patientId={pid}",
        })
    _write_cases(cases, out)


def build(per_class: int, seed: int, out: Path = CASES_CSV) -> None:
    if LOCAL_METADATA.exists() and LOCAL_IMAGES.exists():
        print(f"[rsna] using local pre-converted export at {LOCAL_DIR}")
        build_from_local(per_class, seed, out)
        return
    download_archive()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE) as zf:
        names = set(zf.namelist())
        selected = select_patients(zf, per_class, seed)
        cases: list[dict] = []
        for idx, (pid, label) in enumerate(selected, start=1):
            member = f"stage_2_train_images/{pid}.dcm"
            if member not in names:
                print(f"[rsna] missing DICOM for {pid}, skipping")
                continue
            img = dicom_bytes_to_png(zf.read(member))
            case_id = f"RSNA_{idx:03d}_{label}"
            out_name = f"{case_id}.png"
            img.save(IMG_DIR / out_name)
            split = "smoke" if idx <= (2 * per_class) // 3 else "final"
            cases.append({
                "case_id": case_id,
                "image_path": f"data/rsna_images/{out_name}",
                "source": "rsna_pneumonia_challenge",
                "label": label,
                "split": split,
                "quality": "good",
                "notes": f"RSNA patientId={pid}",
            })

    _write_cases(cases, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=15, help="cases per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=CASES_CSV, help="output cases CSV")
    args = parser.parse_args()
    build(args.per_class, args.seed, args.out)


if __name__ == "__main__":
    main()
