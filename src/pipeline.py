"""Single prediction entry point shared by the API, the web apps and the eval.

Routes an image to the chosen engine (toy rule engine or MedGemma) and applies
the mode-appropriate safety guardrails, so every surface behaves identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails

# The improved system abstains below this confidence; the baseline does not.
IMPROVED_UNCERTAINTY_THRESHOLD = 0.60


def predict(image_path: str | Path, mode: str = "baseline", engine: str = "toy") -> dict[str, Any]:
    if engine == "medgemma":
        from src.medgemma import medgemma_predict

        pred = medgemma_predict(image_path, version=mode)
    else:
        pred = toy_predict(image_path, mode=mode)
    threshold = IMPROVED_UNCERTAINTY_THRESHOLD if mode == "improved" else None
    return apply_safety_guardrails(
        pred, uncertainty_threshold=threshold, screening=(mode == "screening")
    )
