"""Prepare RSNA Pneumonia Detection 2018 dataset for the virtual radiology assistant.

Maps the 3 RSNA classes to the project's 3-class ontology:
  Normal                       -> normal
  Lung Opacity                 -> suspected_opacity
  No Lung Opacity / Not Normal -> uncertain

Produces:
  data/rsna/processed/images/   processed 512x512 RGB PNGs (CLAHE-enhanced)
  data/rsna/cases.csv           case registry (smoke / dev / final splits)
  medical_ai_evidence.sqlite    SQLite cases table populated

Usage:
  python scripts/prepare_rsna_dataset.py \\
      --rsna-dir /path/to/rsna-pneumonia-detection-2018 \\
      [--out-dir data/rsna] \\
      [--db-path medical_ai_evidence.sqlite] \\
      [--smoke-n 20] [--dev-n 150] [--final-n 30] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_image          # noqa: E402
from src.database import init_db, connect               # noqa: E402

LABEL_MAP: dict[str, str] = {
    "Normal": "normal",
    "Lung Opacity": "suspected_opacity",
    "No Lung Opacity / Not Normal": "uncertain",
}

CASES_COLUMNS = [
    "case_id", "image_path", "source", "label", "split",
    "quality", "preprocessing_flags", "bounding_boxes", "notes",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess RSNA Pneumonia 2018 DICOM dataset into the project pipeline."
    )
    p.add_argument("--rsna-dir", required=True, type=Path,
                   help="Root directory of the downloaded Kaggle RSNA dataset")
    p.add_argument("--out-dir", default=ROOT / "data" / "rsna", type=Path,
                   help="Destination for processed images and cases.csv (default: data/rsna)")
    p.add_argument("--db-path", default=ROOT / "medical_ai_evidence.sqlite", type=Path,
                   help="SQLite database path (default: medical_ai_evidence.sqlite)")
    p.add_argument("--smoke-n", default=20, type=int,
                   help="Cases in smoke-test split (default: 20)")
    p.add_argument("--dev-n", default=150, type=int,
                   help="Cases in dev split (default: 150)")
    p.add_argument("--final-n", default=30, type=int,
                   help="Cases in final evaluation split (default: 30)")
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed for reproducible splits (default: 42)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _find_rsna_files(rsna_dir: Path) -> tuple[Path, Path, Path]:
    """Locate the two mandatory CSVs and the DICOM image directory.

    Searches rsna_dir and its immediate children to handle both flat and
    nested Kaggle extraction layouts.
    """
    if not rsna_dir.exists():
        print(f"[ERROR] Directory not found: {rsna_dir}")
        print("  Download from: https://www.kaggle.com/datasets/sovitrath/rsna-pneumonia-detection-2018")
        sys.exit(1)

    search_bases = [rsna_dir] + [d for d in rsna_dir.iterdir() if d.is_dir()]

    detail_csv: Path | None = None
    train_csv: Path | None = None
    images_dir: Path | None = None

    for base in search_bases:
        if detail_csv is None and (base / "stage_2_detailed_class_info.csv").exists():
            detail_csv = base / "stage_2_detailed_class_info.csv"
        if train_csv is None and (base / "stage_2_train_labels.csv").exists():
            train_csv = base / "stage_2_train_labels.csv"
        if images_dir is None:
            for name in ("stage_2_train_images", "train_images", "images"):
                if (base / name).is_dir():
                    images_dir = base / name
                    break

    missing = []
    if detail_csv is None:
        missing.append("stage_2_detailed_class_info.csv")
    if train_csv is None:
        missing.append("stage_2_train_labels.csv")
    if images_dir is None:
        missing.append("stage_2_train_images/  (DICOM folder)")

    if missing:
        print(f"[ERROR] Required RSNA files not found in {rsna_dir}:")
        for m in missing:
            print(f"  - {m}")
        print("\nExpected layout:")
        print("  <rsna_dir>/")
        print("    stage_2_detailed_class_info.csv")
        print("    stage_2_train_labels.csv")
        print("    stage_2_train_images/")
        print("      *.dcm")
        sys.exit(1)

    return detail_csv, train_csv, images_dir


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _load_labels(detail_csv: Path, train_csv: Path) -> dict[str, dict]:
    """Return {patientId: {label: str, bboxes: list}} merged from both CSVs."""
    classes: dict[str, str] = {}
    with detail_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row["patientId"].strip()
            raw = row.get("class", row.get("Class", "")).strip()
            classes[pid] = LABEL_MAP.get(raw, "uncertain")

    bboxes: dict[str, list[dict]] = {}
    with train_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("Target", "0").strip() == "1":
                pid = row["patientId"].strip()
                try:
                    bboxes.setdefault(pid, []).append({
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "width": float(row["width"]),
                        "height": float(row["height"]),
                    })
                except (ValueError, KeyError):
                    pass

    return {pid: {"label": lbl, "bboxes": bboxes.get(pid, [])} for pid, lbl in classes.items()}


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def _balanced_sample(by_class: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    """Return n patient IDs distributed as evenly as possible across classes."""
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
    smoke_n: int,
    dev_n: int,
    final_n: int,
    rng: random.Random,
) -> dict[str, str]:
    """Return {patientId: split_name}.

    final is reserved first (held-out evaluation), then smoke, then dev.
    """
    by_class: dict[str, list[str]] = {}
    for pid, info in all_cases.items():
        by_class.setdefault(info["label"], []).append(pid)
    for pids in by_class.values():
        rng.shuffle(pids)

    final_ids = set(_balanced_sample(by_class, final_n, rng))
    remaining: dict[str, list[str]] = {
        cls: [p for p in pids if p not in final_ids]
        for cls, pids in by_class.items()
    }

    smoke_ids = set(_balanced_sample(remaining, smoke_n, rng))
    remaining2: dict[str, list[str]] = {
        cls: [p for p in pids if p not in smoke_ids]
        for cls, pids in remaining.items()
    }

    dev_ids = set(_balanced_sample(remaining2, dev_n, rng))

    result = {pid: "final" for pid in final_ids}
    result.update({pid: "smoke" for pid in smoke_ids})
    result.update({pid: "dev" for pid in dev_ids})
    return result


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def _process_case(
    patient_id: str,
    info: dict,
    images_dir: Path,
    out_images_dir: Path,
    split: str,
    project_root: Path,
) -> dict | None:
    dcm_path = images_dir / f"{patient_id}.dcm"
    if not dcm_path.exists():
        print(f"    [WARN] DICOM not found: {dcm_path.name}")
        return None

    case_id = "RSNA_" + patient_id.replace("-", "_")
    out_path = out_images_dir / f"{case_id}.png"

    try:
        result = preprocess_image(dcm_path)
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
                (
                    row["case_id"], row["image_path"], row["source"],
                    row["label"], row["split"], row["quality"],
                    row["preprocessing_flags"], row["bounding_boxes"], row["notes"],
                ),
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

    detail_csv, train_csv, images_dir = _find_rsna_files(args.rsna_dir)

    print("[1/4] Loading labels …")
    all_cases = _load_labels(detail_csv, train_csv)
    print(f"      {len(all_cases)} patients found")
    by_class_counts: dict[str, int] = {}
    for info in all_cases.values():
        by_class_counts[info["label"]] = by_class_counts.get(info["label"], 0) + 1
    for lbl, cnt in sorted(by_class_counts.items()):
        print(f"      {lbl}: {cnt}")

    print("[2/4] Building balanced splits …")
    splits = _build_splits(all_cases, args.smoke_n, args.dev_n, args.final_n, rng)
    split_counts: dict[str, int] = {}
    for s in splits.values():
        split_counts[s] = split_counts.get(s, 0) + 1
    for s, cnt in sorted(split_counts.items()):
        print(f"      {s}: {cnt}")

    print("[3/4] Preprocessing images …")
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

    print("[4/4] Writing outputs …")
    cases_csv = args.out_dir / "cases.csv"
    with cases_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CASES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"      cases.csv  → {cases_csv}  ({len(rows)} rows)")

    _insert_cases(rows, args.db_path)
    print(f"      SQLite     → {args.db_path}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    final_by_class: dict[str, dict[str, int]] = {}
    for row in rows:
        final_by_class.setdefault(row["split"], {})
        final_by_class[row["split"]][row["label"]] = (
            final_by_class[row["split"]].get(row["label"], 0) + 1
        )
    for split in ("smoke", "dev", "final"):
        dist = final_by_class.get(split, {})
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  {split:<6} {sum(dist.values()):>4} cases   {parts}")

    quality_counts: dict[str, int] = {}
    for row in rows:
        quality_counts[row["quality"]] = quality_counts.get(row["quality"], 0) + 1
    print()
    print("  Image quality distribution:")
    for q, cnt in sorted(quality_counts.items()):
        print(f"    {q}: {cnt}")

    print()
    print("Next step - run smoke test:")
    print(f"  python eval/run_evaluation.py --mode toy --split smoke --db-path {args.db_path}")
    print()


if __name__ == "__main__":
    main()
