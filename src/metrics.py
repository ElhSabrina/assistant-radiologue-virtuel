from __future__ import annotations

from collections import Counter
from typing import Iterable

CLASSES = ["normal", "suspected_opacity", "uncertain"]


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)


def sensitivity_specificity(
    y_true: Iterable[str], y_pred: Iterable[str], positive: str = "suspected_opacity"
) -> tuple[float, float]:
    """Binary sensitivity/specificity with ``positive`` as the disease class.

    Abstentions (``uncertain``) are counted conservatively: an ``uncertain``
    prediction on a positive case is not a detection (lowers sensitivity), and on
    a negative case is not a clearance (lowers specificity). This mirrors a
    cautious clinical reading where "I don't know" is not a positive call.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    pos_total = sum(t == positive for t in y_true)
    neg_total = sum(t != positive for t in y_true)
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive and p != "uncertain" for t, p in zip(y_true, y_pred))
    sensitivity = tp / pos_total if pos_total else 0.0
    specificity = tn / neg_total if neg_total else 0.0
    return sensitivity, specificity


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]
    hallucinations = [bool(r.get("hallucination", False)) for r in rows]
    sensitivity, specificity = sensitivity_specificity(y_true, y_pred)
    mean_latency = sum(float(r.get("latency_ms", 0) or 0) for r in rows) / len(rows) if rows else 0
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
        "hallucination_rate": round(sum(hallucinations) / len(rows), 4) if rows else 0,
        "mean_latency_ms": round(mean_latency, 1),
    }
