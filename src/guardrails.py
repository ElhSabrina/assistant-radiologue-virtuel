from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except Exception:
        errors.append("confidence is not numeric")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


# Words that signal invented clinical context (patient history the model cannot
# see on a single frontal image). Their presence flags an unfounded justification.
_HALLUCINATION_MARKERS = (
    "history", "patient reports", "years old", "year-old", "year old", "smoker",
    "fever", "cough", "shortness of breath", "diagnosed with", "known case",
    "clinical history", "complains of", "symptoms of", "antécédent", "fièvre",
    "toux", "fumeur", "patient présente", "diagnostiqué",
)


def detect_hallucination(pred: dict[str, Any]) -> bool:
    """Flag a prediction whose free text invents clinical context.

    A single frontal X-ray carries no history, symptoms or demographics, so any
    such claim in the justification/evidence is unsupported by the image.
    """
    text_parts = [str(pred.get("justification", ""))]
    text_parts.extend(str(x) for x in pred.get("visual_evidence", []) or [])
    blob = " ".join(text_parts).lower()
    return any(marker in blob for marker in _HALLUCINATION_MARKERS)


def apply_safety_guardrails(
    pred: dict[str, Any],
    uncertainty_threshold: float | None = None,
    screening: bool = False,
) -> dict[str, Any]:
    """Coerce a prediction into a safe, schema-valid output.

    When ``uncertainty_threshold`` is set (the improved system enables it), any
    decided class whose confidence falls below it is downgraded to ``uncertain``.
    Left as ``None`` for the baseline so its behaviour is unchanged.

    When ``screening`` is True (high-sensitivity mode), an ``uncertain`` outcome is
    escalated to ``suspected_opacity``: in a screening setting a doubtful image is
    flagged for review, never left cleared. This trades specificity for a lower
    false-negative rate: the medically cautious direction.
    """
    valid, errors = validate_prediction(pred)
    if not valid:
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(float(pred.get("confidence", 0.0) or 0.0), 0.5)
        pred.setdefault("limitations", []).append("guardrail triggered: invalid output schema")
    if pred.get("image_quality") in {"limited", "poor"} and float(pred.get("confidence", 0)) < 0.6:
        pred["predicted_class"] = "uncertain"
    if screening and pred.get("predicted_class") == "uncertain":
        pred["predicted_class"] = "suspected_opacity"
        pred.setdefault("limitations", []).append("screening bias: doubt flagged as suspected_opacity")
    if (
        uncertainty_threshold is not None
        and pred.get("predicted_class") != "uncertain"
        and float(pred.get("confidence", 0)) < uncertainty_threshold
    ):
        pred["predicted_class"] = "uncertain"
        pred.setdefault("limitations", []).append(
            f"guardrail triggered: confidence below {uncertainty_threshold}"
        )
    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    return pred
