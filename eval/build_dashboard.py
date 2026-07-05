"""Render a simple baseline-vs-improved dashboard from evaluation outputs.

Reads ``eval/outputs/before_after_summary.csv`` (and the per-mode prediction
CSVs when present) and produces:

    eval/outputs/dashboard.png   - grouped bar charts of the key metrics
    eval/outputs/dashboard.md    - a compact textual comparison table

    python eval/build_dashboard.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("sensitivity", "Sensibilité"),
    ("specificity", "Spécificité"),
    ("json_valid_rate", "JSON valide"),
    ("uncertain_rate", "Taux incertain"),
    ("hallucination_rate", "Hallucination"),
]


def read_summary(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["mode"]: row for row in rows}


def build(summary_csv: Path, out_png: Path, out_md: Path) -> None:
    summary = read_summary(summary_csv)
    modes = [m for m in ("baseline", "improved") if m in summary]
    if not modes:
        raise SystemExit(f"no baseline/improved rows in {summary_csv}")

    labels = [lbl for _, lbl in METRICS]
    x = range(len(METRICS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, mode in enumerate(modes):
        vals = [float(summary[mode].get(key, 0) or 0) for key, _ in METRICS]
        offset = (i - (len(modes) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width, label=mode)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    engine = summary[modes[0]].get("engine", "")
    ax.set_title(f"Baseline vs Improved: moteur: {engine}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    lines = ["# Dashboard: baseline vs improved", "", f"Moteur : `{engine}`", "",
             "| Métrique | " + " | ".join(modes) + " | Δ (imp-base) |", "|---|" + "---|" * (len(modes) + 1)]
    for key, lbl in METRICS + [("mean_latency_ms", "Latence moy. (ms)"), ("n", "N")]:
        cells = [summary[m].get(key, "") for m in modes]
        try:
            delta = float(summary["improved"][key]) - float(summary["baseline"][key])
            delta_s = f"{delta:+.4f}" if key != "n" else ""
        except (KeyError, ValueError):
            delta_s = ""
        lines.append(f"| {lbl} | " + " | ".join(str(c) for c in cells) + f" | {delta_s} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[dashboard] wrote {out_png} and {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "eval" / "results" / "before_after_summary.csv")
    parser.add_argument("--out-png", type=Path, default=ROOT / "eval" / "results" / "dashboard.png")
    parser.add_argument("--out-md", type=Path, default=ROOT / "eval" / "results" / "dashboard.md")
    args = parser.parse_args()
    build(args.summary, args.out_png, args.out_md)


if __name__ == "__main__":
    main()
