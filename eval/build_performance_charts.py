"""Classic performance charts for the MedGemma zero-shot evaluation.

MedGemma is used zero-shot (by prompting), so there is no training curve: the
graphs here are the standard *evaluation* diagnostics on held-out RSNA cases.

From a predictions CSV (``label``, ``predicted_class``, ``confidence``,
``latency_ms``) it renders:

    1. Confusion matrix (true normal/opacity vs predicted 3 classes)
    2. Per-class precision / recall / F1
    3. Confidence distribution, correct vs incorrect
    4. Risk-coverage curve (abstain below a confidence threshold)
    5. Reliability diagram (calibration)
    6. Latency distribution

    python eval/build_performance_charts.py --predictions eval/perf/baseline_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
POSITIVE = "suspected_opacity"
TRUE_CLASSES = ["normal", "suspected_opacity"]
PRED_CLASSES = ["normal", "suspected_opacity", "uncertain"]


def load(predictions: Path) -> list[dict]:
    with predictions.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def confusion_matrix(rows: list[dict]) -> np.ndarray:
    m = np.zeros((len(TRUE_CLASSES), len(PRED_CLASSES)), dtype=int)
    for r in rows:
        if r["label"] in TRUE_CLASSES and r["predicted_class"] in PRED_CLASSES:
            m[TRUE_CLASSES.index(r["label"]), PRED_CLASSES.index(r["predicted_class"])] += 1
    return m


def per_class_prf(rows: list[dict]) -> dict[str, tuple[float, float, float]]:
    out = {}
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    for c in TRUE_CLASSES:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = (prec, rec, f1)
    return out


def _is_correct(r: dict) -> bool:
    return r["label"] == r["predicted_class"]


def plot_confusion(rows, ax) -> None:
    m = confusion_matrix(rows)
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(PRED_CLASSES)), PRED_CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(TRUE_CLASSES)), TRUE_CLASSES)
    ax.set_xlabel("prédit"); ax.set_ylabel("vérité"); ax.set_title("Matrice de confusion")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, str(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")


def plot_prf(rows, ax) -> None:
    prf = per_class_prf(rows)
    labels = list(prf.keys())
    x = np.arange(len(labels))
    w = 0.25
    for i, (name, idx) in enumerate([("précision", 0), ("rappel", 1), ("F1", 2)]):
        ax.bar(x + (i - 1) * w, [prf[c][idx] for c in labels], w, label=name)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05)
    ax.set_title("Précision / rappel / F1 par classe"); ax.legend(); ax.grid(axis="y", alpha=0.3)


def plot_conf_hist(rows, ax) -> None:
    corr = [float(r["confidence"]) for r in rows if _is_correct(r)]
    wrong = [float(r["confidence"]) for r in rows if not _is_correct(r)]
    bins = np.linspace(0, 1, 11)
    ax.hist([corr, wrong], bins=bins, label=["correct", "incorrect"], color=["#3a7", "#c33"])
    ax.set_xlabel("confiance"); ax.set_ylabel("nombre de cas")
    ax.set_title("Confiance : correct vs incorrect"); ax.legend()


def plot_risk_coverage(rows, ax) -> None:
    """Accuracy on decided cases vs coverage as we raise a confidence threshold."""
    thresholds = np.linspace(0, 0.95, 20)
    cov, acc = [], []
    n = len(rows)
    for t in thresholds:
        decided = [r for r in rows if float(r["confidence"]) >= t]
        cov.append(len(decided) / n if n else 0)
        acc.append(sum(_is_correct(r) for r in decided) / len(decided) if decided else np.nan)
    ax.plot(cov, acc, marker="o")
    ax.set_xlabel("couverture (part de cas décidés)"); ax.set_ylabel("accuracy sur cas décidés")
    ax.set_ylim(0, 1.05); ax.set_title("Courbe risque-couverture (abstention)"); ax.grid(alpha=0.3)


def plot_calibration(rows, ax) -> None:
    bins = np.linspace(0, 1, 6)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        group = [r for r in rows if lo <= float(r["confidence"]) < hi]
        if group:
            xs.append(np.mean([float(r["confidence"]) for r in group]))
            ys.append(np.mean([_is_correct(r) for r in group]))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibration parfaite")
    ax.plot(xs, ys, marker="s", label="observé")
    ax.set_xlabel("confiance moyenne"); ax.set_ylabel("accuracy empirique")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_title("Diagramme de fiabilité"); ax.legend()


def plot_latency(rows, ax) -> None:
    lat = np.array([float(r.get("latency_ms", 0) or 0) / 1000 for r in rows])
    median = float(np.median(lat)) if len(lat) else 0.0
    # Clip rare outliers (e.g. an image stuck under GPU contention) so the
    # histogram stays readable; the median reported in the title is unaffected.
    cap = np.percentile(lat, 95) if len(lat) else 0.0
    shown = lat[lat <= cap]
    n_out = int((lat > cap).sum())
    ax.hist(shown, bins=15, color="#69a")
    ax.set_xlabel("latence (s)"); ax.set_ylabel("nombre de cas")
    title = f"Latence (médiane {median:.0f} s)"
    if n_out:
        title += f": {n_out} outlier(s) masqué(s)"
    ax.set_title(title)


def build(predictions: Path, out_dir: Path) -> None:
    rows = load(predictions)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_confusion(rows, axes[0, 0])
    plot_prf(rows, axes[0, 1])
    plot_conf_hist(rows, axes[0, 2])
    plot_risk_coverage(rows, axes[1, 0])
    plot_calibration(rows, axes[1, 1])
    plot_latency(rows, axes[1, 2])
    fig.suptitle(f"MedGemma zero-shot: performances ({len(rows)} cas RSNA)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    combined = out_dir / "performance_overview.png"
    fig.savefig(combined, dpi=130)
    plt.close(fig)

    # Also save each panel individually for slides.
    panels = [
        ("confusion_matrix", plot_confusion), ("precision_recall_f1", plot_prf),
        ("confidence_hist", plot_conf_hist), ("risk_coverage", plot_risk_coverage),
        ("calibration", plot_calibration), ("latency", plot_latency),
    ]
    for name, fn in panels:
        f, a = plt.subplots(figsize=(6, 4.5))
        fn(rows, a); f.tight_layout(); f.savefig(out_dir / f"{name}.png", dpi=130); plt.close(f)

    print(f"[perf] {len(rows)} cases -> {combined} (+ 6 panels in {out_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "eval" / "perf" / "baseline_predictions.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "perf" / "charts")
    args = parser.parse_args()
    build(args.predictions, args.out_dir)


if __name__ == "__main__":
    main()
