"""Build a commented error register from an evaluation predictions CSV.

Reads the per-case predictions produced by ``run_evaluation.py`` (e.g.
``eval/outputs/improved_predictions.csv``) and classifies every case into one of
the error families the brief asks for:

    TP / TN  -> correct decision
    FP       -> false positive (normal called as opacity)
    FN       -> false negative (opacity called as normal)
    UA       -> acceptable uncertain (model abstained)
    FORMAT   -> invalid JSON / schema failure
    HALLU    -> unfounded text (invented clinical context)

Outputs a CSV and a Markdown table under ``eval/``.

    python eval/error_register.py --predictions eval/outputs/improved_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMENTS = {
    "TP": "opacity correctly flagged",
    "TN": "normal correctly cleared",
    "FP": "normal image overcalled as opacity",
    "FN": "opacity missed and called normal",
    "UA": "model abstained (uncertain) instead of forcing a class",
    "FORMAT": "output failed JSON/schema validation",
    "HALLU": "justification invents clinical context not visible on the image",
}
ACTIONS = {
    "TP": "keep",
    "TN": "keep",
    "FP": "tighten specificity / raise confidence bar",
    "FN": "review sensitivity, inspect prompt wording",
    "UA": "acceptable; monitor uncertain rate",
    "FORMAT": "harden JSON parsing / prompt formatting",
    "HALLU": "reinforce 'no invented history' rule in prompt",
}
SEVERITY = {"FN": "high", "FP": "medium", "HALLU": "high", "FORMAT": "medium", "UA": "low", "TP": "low", "TN": "low"}


def classify(row: dict) -> str:
    label = row["label"]
    pred = row["predicted_class"]
    json_valid = str(row.get("json_valid", "True")).lower() in {"true", "1"}
    hallucinated = str(row.get("hallucination", "False")).lower() in {"true", "1"}
    if not json_valid:
        return "FORMAT"
    if hallucinated:
        return "HALLU"
    if pred == "uncertain":
        return "UA"
    if pred == label:
        return "TP" if label == "suspected_opacity" else "TN"
    if label == "normal" and pred == "suspected_opacity":
        return "FP"
    if label == "suspected_opacity" and pred == "normal":
        return "FN"
    return "FP" if pred == "suspected_opacity" else "FN"


def build(predictions: Path, out_csv: Path, out_md: Path) -> None:
    with predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        etype = classify(row)
        records.append({
            "case_id": row["case_id"],
            "ground_truth": row["label"],
            "prediction": row["predicted_class"],
            "confidence": row.get("confidence", ""),
            "error_type": etype,
            "severity": SEVERITY[etype],
            "comment": COMMENTS[etype],
            "corrective_action": ACTIONS[etype],
            "justification": (row.get("justification", "") or "")[:240],
        })

    fieldnames = list(records[0].keys()) if records else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    counts: dict[str, int] = {}
    for r in records:
        counts[r["error_type"]] = counts.get(r["error_type"], 0) + 1

    lines = [
        f"# Registre d'erreurs — {predictions.name}",
        "",
        f"Total : {len(records)} cas.",
        "",
        "## Répartition",
        "",
        "| Type | Nombre |",
        "|---|---|",
    ]
    for etype in ("TP", "TN", "FP", "FN", "UA", "HALLU", "FORMAT"):
        if etype in counts:
            lines.append(f"| {etype} — {COMMENTS[etype]} | {counts[etype]} |")
    lines += ["", "## Cas commentés", "", "| case_id | vérité | prédiction | conf. | type | sévérité | commentaire |", "|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r['case_id']} | {r['ground_truth']} | {r['prediction']} | {r['confidence']} | "
            f"{r['error_type']} | {r['severity']} | {r['comment']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[error_register] {len(records)} cases -> {out_csv} and {out_md}")
    print(f"[error_register] counts: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "eval" / "results" / "improved_predictions.csv")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "eval" / "error_register.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "eval" / "error_register.md")
    args = parser.parse_args()
    build(args.predictions, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
