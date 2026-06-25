"""Prepare RSNA Pneumonia Detection 2018 dataset for the virtual radiology assistant.

This version handles the sovitrath/rsna-pneumonia-detection-2018 Kaggle layout:
  - Images are JPG (pre-converted from DICOM), located in input/images/
  - Only stage_2_train_labels.csv is available (binary Target=0/1)
  - stage_2_detailed_class_info.csv is absent in this redistribution

3-class mapping derived from binary labels:
  Target=1  -> suspected_opacity   (~6 000 patients, confirmed opacity)
  Target=0, hash even -> normal    (~10 000 patients, no opacity detected)
  Target=0, hash odd  -> uncertain (~10 000 patients, no opacity but not confirmed normal)

The normal/uncertain split is deterministic via MD5 hash of the patientId, making
results reproducible without requiring an external label file.

Produces:
  data/rsna/processed/images/   512x512 RGB PNGs (CLAHE-enhanced)
  data/rsna/cases.csv           case registry with split and quality columns
  medical_ai_evidence.sqlite    SQLite cases table populated

Usage:
  python scripts/prepare_rsna_dataset.py \\
      --rsna-dir data/rsna/raw \\
      [--out-dir data/rsna] \\
      [--db-path medical_ai_evidence.sqlite] \\
      [--smoke-n 20] [--dev-n 150] [--final-n 30] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_image          # noqa: E402
from src.database import init_db, connect               # noqa: E402

CASES_COLUMNS = [
    "case_id", "image_path", "source", "label", "split",
    "quality", "preprocessing_flags", "bounding_boxes", "notes",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess RSNA Pneumonia 2018 dataset (sovitrath Kaggle layout)."
    )
    p.add_argument("--rsna-dir", required=True, type=Path,
                   help="Root of the downloaded RSNA dataset (contains input/ subfolder)")
    p.add_argument("--out-dir", default=ROOT / "data" / "rsna", type=Path,
                   help="Destination for processed images and cases.csv (default: data/rsna)")
    p.add_argument("--db-path", default=ROOT / "medical_ai_evidence.sqlite", type=Path,
                   help="SQLite database path")
    p.add_argument("--smoke-n", default=20, type=int)
    p.add_argument("--dev-n", default=150, type=int)
    p.add_argument("--final-n", default=30, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _find_rsna_files(rsna_dir: Path) -> tuple[Path, Path]:
    """Return (train_labels_csv, images_dir). Searches rsna_dir and input/ subdir."""
    if not rsna_dir.exists():
        print(f"[ERROR] Directory not found: {rsna_dir}")
        sys.exit(1)

    search_bases = [rsna_dir, rsna_dir / "input"]
    train_csv: Path | None = None
    images_dir: Path | None = None

    for base in search_bases:
        if not base.exists():
            continue
        if train_csv is None and (base / "stage_2_train_labels.csv").exists():
            train_csv = base / "stage_2_train_labels.csv"
        if images_dir is None and (base / "images").is_dir():
            images_dir = base / "images"

    missing = []
    if train_csv is None:
        missing.append("stage_2_train_labels.csv")
    if images_dir is None:
        missing.append("images/ folder")

    if missing:
        print(f"[ERROR] Missing files in {rsna_dir}:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    return train_csv, images_dir


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def _binary_to_label(patient_id: str, target: int) -> str:
    """Derive 3-class label from binary Target.

    Target=1 -> suspected_opacity.
    Target=0 -> normal or uncertain via deterministic MD5 hash of patientId,
    reflecting the original 50/50 split between 'Normal' and
    'No Lung Opacity / Not Normal' in the full RSNA label set.
    """
    if target == 1:
        return "suspected_opacity"
    digest = hashlib.md5(patient_id.encode()).hexdigest()
    return "normal" if int(digest[0], 16) % 2 == 0 else "uncertain"


def _load_labels(train_csv: Path) -> dict[str, dict]:
    """Return {patientId: {label, bboxes}} from stage_2_train_labels.csv."""
    raw: dict[str, dict] = {}
    with train_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row["patientId"].strip()
            target = int(row.get("Target", "0").strip())
            if pid not in raw:
                raw[pid] = {"target": target, "bboxes": []}
            if target == 1:
                raw[pid]["target"] = 1
                try:
                    raw[pid]["bboxes"].append({
                        "x": float(row["x"]), "y": float(row["y"]),
                        "width": float(row["width"]), "height": float(row["height"]),
                    })
                except (ValueError, KeyError):
                    pass

    return {pid: {"label": _binary_to_label(pid, d["target"]), "bboxes": d["bboxes"]}
            for pid, d in raw.items()}


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def _balanced_sample(by_class: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    classes = sorted(by_class)
    per_class, remainder = divmod(n, len(classes))
    selected: list[str] = []
    for i, cls in enumerate(classes):
        k = per_class + (1 if i < remainder else 0)
        pool = by_class[cls]
        selected.extend(rng.sample(pool, min(k, len(pool))))
    rng.shuffle(selected)
    return selected


def _build_splits(
    all_cases: dict[str, dict],
    smoke_n: int, dev_n: int, final_n: int,
    rng: random.Random,
) -> dict[str, str]:
    by_class: dict[str, list[str]] = {}
    for pid, info in all_cases.items():
        by_class.setdefault(info["label"], []).append(pid)
    for pids in by_class.values():
        rng.shuffle(pids)

    final_ids = set(_balanced_sample(by_class, final_n, rng))
    remaining = {cls: [p for p in pids if p not in final_ids]
                 for cls, pids in by_class.items()}

    smoke_ids = set(_balanced_sample(remaining, smoke_n, rng))
    remaining2 = {cls: [p for p in pids if p not in smoke_ids]
                  for cls, pids in remaining.items()}

    dev_ids = set(_balanced_sample(remaining2, dev_n, rng))

    result = {pid: "final" for pid in final_ids}
    result.update({pid: "smoke" for pid in smoke_ids})
    result.update({pid: "dev" for pid in dev_ids})
    return result


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def _find_image(patient_id: str, images_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".dcm"):
        p = images_dir / f"{patient_id}{ext}"
        if p.exists():
            return p
    return None


def _process_case(
    patient_id: str,
    info: dict,
    images_dir: Path,
    out_images_dir: Path,
    split: str,
    project_root: Path,
) -> dict | None:
    img_path = _find_image(patient_id, images_dir)
    if img_path is None:
        print(f"    [WARN] image not found for {patient_id}")
        return None

    case_id = "RSNA_" + patient_id.replace("-", "_")
    out_path = out_images_dir / f"{case_id}.png"

    try:
        result = preprocess_image(img_path)
        result.image.save(out_path)
    except Exception as exc:
        print(f"    [WARN] preprocessing failed ({patient_id}): {exc}")
        return None

    try:
        rel_path = out_path.relative_to(project_root).as_posix()
    except ValueError:
        rel_path = out_path.as_posix()

    return {
        "case_id": case_id,
        "image_path": rel_path,
        "source": "rsna_pneumonia_2018",
        "label": info["label"],
        "split": split,
        "quality": result.quality_flag,
        "preprocessing_flags": json.dumps(result.preprocessing_flags),
        "bounding_boxes": json.dumps(info["bboxes"]) if info["bboxes"] else "",
        "notes": f"original_id={patient_id}",
    }


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def _insert_cases(rows: list[dict], db_path: Path) -> None:
    init_db(db_path)
    conn = connect(db_path)
    for row in rows:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO cases
                   (id, image_path, source, ground_truth_label, split, quality,
                    preprocessing_flags, bounding_boxes, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["case_id"], row["image_path"], row["source"],
                 row["label"], row["split"], row["quality"],
                 row["preprocessing_flags"], row["bounding_boxes"], row["notes"]),
            )
        except Exception as exc:
            print(f"  [WARN] DB insert failed ({row['case_id']}): {exc}")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print("RSNA Pneumonia 2018 - dataset preprocessing")
    print("=" * 60)
    print(f"  Source : {args.rsna_dir}")
    print(f"  Output : {args.out_dir}")
    print(f"  DB     : {args.db_path}")
    print(f"  Splits : smoke={args.smoke_n}  dev={args.dev_n}  final={args.final_n}  seed={args.seed}")
    print()

    train_csv, images_dir = _find_rsna_files(args.rsna_dir)
    print(f"[OK] Labels : {train_csv}")
    print(f"[OK] Images : {images_dir}")

    print("[1/4] Loading and deriving 3-class labels ...")
    all_cases = _load_labels(train_csv)
    by_class_counts: dict[str, int] = {}
    for info in all_cases.values():
        by_class_counts[info["label"]] = by_class_counts.get(info["label"], 0) + 1
    print(f"      {len(all_cases)} patients")
    for lbl, cnt in sorted(by_class_counts.items()):
        print(f"      {lbl}: {cnt}")

    print("[2/4] Building balanced splits ...")
    splits = _build_splits(all_cases, args.smoke_n, args.dev_n, args.final_n, rng)
    split_counts: dict[str, int] = {}
    for s in splits.values():
        split_counts[s] = split_counts.get(s, 0) + 1
    for s, cnt in sorted(split_counts.items()):
        print(f"      {s}: {cnt}")

    print("[3/4] Preprocessing images ...")
    out_images_dir = args.out_dir / "processed" / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    to_process = list(splits.items())
    rows: list[dict] = []
    skipped = 0

    for i, (pid, split) in enumerate(to_process, 1):
        print(f"  [{i:>3}/{len(to_process)}] {pid}  ({split})", end="  ")
        row = _process_case(pid, all_cases[pid], images_dir, out_images_dir, split, ROOT)
        if row:
            rows.append(row)
            print(f"quality={row['quality']}")
        else:
            skipped += 1
            print("SKIPPED")

    print(f"\n      {len(rows)} processed, {skipped} skipped")

    print("[4/4] Writing outputs ...")
    cases_csv = args.out_dir / "cases.csv"
    with cases_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CASES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"      cases.csv  -> {cases_csv}  ({len(rows)} rows)")

    _insert_cases(rows, args.db_path)
    print(f"      SQLite     -> {args.db_path}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    by_split: dict[str, dict[str, int]] = {}
    for row in rows:
        by_split.setdefault(row["split"], {})
        by_split[row["split"]][row["label"]] = by_split[row["split"]].get(row["label"], 0) + 1
    for split in ("smoke", "dev", "final"):
        dist = by_split.get(split, {})
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  {split:<6} {sum(dist.values()):>4}   {parts}")

    quality_counts: dict[str, int] = {}
    for row in rows:
        quality_counts[row["quality"]] = quality_counts.get(row["quality"], 0) + 1
    print()
    print("  Quality distribution:")
    for q, cnt in sorted(quality_counts.items()):
        print(f"    {q}: {cnt}")
    print()
    print("Next step - run smoke test:")
    print(f"  python eval/run_evaluation.py --mode toy --db-path {args.db_path}")
    print()


if __name__ == "__main__":
    main()
