"""Build supervised image -> JSON training pairs from RSNA for QLoRA.

Each example teaches MedGemma to answer the SAME schema as inference, with the
correct class distilled from the RSNA label. Justifications are templated,
evidence-based and never invent clinical history (they respect the guardrails).

Two labels are supervised:
    Normal        -> predicted_class "normal"
    Lung Opacity  -> predicted_class "suspected_opacity"
The ambiguous "No Lung Opacity / Not Normal" class is not used.

Output: ``finetuning/data/train.jsonl`` and ``val.jsonl`` where each line is
``{"image_path": ..., "label": ..., "target": "<json string>"}``. The Kaggle
notebook uses the identical ``target_json`` logic on the competition DICOMs.

    python finetuning/build_training_data.py --per-class 200 --val-frac 0.15
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
OUT_DIR = ROOT / "finetuning" / "data"

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
CLASS_MAP = {"Normal": "normal", "Lung Opacity": "suspected_opacity"}

# Templated, evidence-based completions (no invented history). Index picks a
# variant so the model does not memorise a single string.
_TEMPLATES = {
    "normal": [
        {
            "image_quality": "good",
            "confidence": 0.82,
            "visual_evidence": ["clear lung fields", "no focal opacity or consolidation"],
            "justification": "The lung fields appear clear without a focal opacity or consolidation on this frontal view. No definitive abnormality is visible.",
        },
        {
            "image_quality": "good",
            "confidence": 0.78,
            "visual_evidence": ["symmetric lucency of both lungs", "no visible airspace opacity"],
            "justification": "Both lungs show symmetric lucency and no visible airspace opacity. The appearance is within normal limits on this single projection.",
        },
    ],
    "suspected_opacity": [
        {
            "image_quality": "good",
            "confidence": 0.76,
            "visual_evidence": ["area of increased opacity in a lung field", "loss of normal lucency"],
            "justification": "There is an area of increased opacity with loss of normal lucency in a lung field, which could correspond to airspace disease. This is an educational observation, not a diagnosis.",
        },
        {
            "image_quality": "good",
            "confidence": 0.71,
            "visual_evidence": ["hazy opacity projected over a lung zone", "ill-defined margins"],
            "justification": "A hazy, ill-defined opacity is projected over a lung zone. Such an appearance can be seen with a pulmonary opacity, but projection and technique should be considered.",
        },
    ],
}


def target_json(label: str, idx: int) -> str:
    """Return the schema-conforming JSON completion for a label."""
    variant = _TEMPLATES[label][idx % len(_TEMPLATES[label])]
    obj = {
        "image_quality": variant["image_quality"],
        "predicted_class": label,
        "confidence": variant["confidence"],
        "visual_evidence": variant["visual_evidence"],
        "justification": variant["justification"],
        "limitations": ["single frontal view", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
    }
    return json.dumps(obj, ensure_ascii=False)


def select(per_class: int, seed: int) -> list[tuple[str, str]]:
    with METADATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    buckets: dict[str, list[str]] = {"normal": [], "suspected_opacity": []}
    seen: set[str] = set()
    for r in rows:
        pid = r["patientId"]
        if pid in seen:
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
    return picked


def build(per_class: int, val_frac: float, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picked = select(per_class, seed)
    examples = []
    for i, (pid, label) in enumerate(picked):
        examples.append({
            "image_path": (IMAGES / f"{pid}.png").relative_to(ROOT).as_posix(),
            "patientId": pid,
            "label": label,
            "target": target_json(label, i),
        })
    n_val = int(len(examples) * val_frac)
    val, train = examples[:n_val], examples[n_val:]
    for name, rows in (("train", train), ("val", val)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ft-data] {name}: {len(rows)} examples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build(args.per_class, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
