"""Build a leakage-free held-out RSNA evaluation set.

Selects balanced cases from the local RSNA export while **excluding every
patientId used for fine-tuning** (``finetuning/data/train.jsonl`` and
``val.jsonl``). This gives a clean set to compare the fine-tuned ("improved")
model against the base ("baseline") model without training-set contamination.

    python eval/build_heldout_cases.py --per-class 40
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = ROOT / "data" / "rsna_pneumonia"
METADATA = LOCAL_DIR / "stage2_train_metadata.csv"
IMAGES = LOCAL_DIR / "Training" / "Images"
FT_DATA = ROOT / "finetuning" / "data"
OUT = ROOT / "data" / "rsna_cases_heldout.csv"

CLASS_MAP = {"Normal": "normal", "Lung Opacity": "suspected_opacity"}


def training_patient_ids() -> set[str]:
    pids: set[str] = set()
    for name in ("train.jsonl", "val.jsonl"):
        path = FT_DATA / name
        if path.exists():
            for line in path.open(encoding="utf-8"):
                pids.add(json.loads(line)["patientId"])
    return pids


def build(per_class: int, seed: int) -> None:
    excluded = training_patient_ids()
    with METADATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    buckets: dict[str, list[str]] = {"normal": [], "suspected_opacity": []}
    seen: set[str] = set()
    for r in rows:
        pid = r["patientId"]
        if pid in seen or pid in excluded:
            continue
        seen.add(pid)
        label = CLASS_MAP.get(r["class"])
        if label and (IMAGES / f"{pid}.png").exists():
            buckets[label].append(pid)

    rng = random.Random(seed)
    picked: list[tuple[str, str]] = []
    for label, pids in buckets.items():
        rng.shuffle(pids)
        picked.extend((pid, label) for pid in pids[:per_class])
    rng.shuffle(picked)

    fieldnames = ["case_id", "image_path", "source", "label", "split", "quality", "notes"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, (pid, label) in enumerate(picked, start=1):
            w.writerow({
                "case_id": f"RSNA_HO_{idx:03d}_{label}",
                "image_path": f"data/rsna_pneumonia/Training/Images/{pid}.png",
                "source": "rsna_pneumonia_challenge",
                "label": label,
                "split": "heldout",
                "quality": "good",
                "notes": f"RSNA patientId={pid}",
            })
    n_norm = sum(l == "normal" for _, l in picked)
    n_op = sum(l == "suspected_opacity" for _, l in picked)
    print(f"[heldout] excluded {len(excluded)} training patientIds")
    print(f"[heldout] wrote {len(picked)} cases ({n_norm} normal / {n_op} opacity) -> {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    build(args.per_class, args.seed)


if __name__ == "__main__":
    main()
