from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from .preprocessing import PreprocessResult, basic_quality_flag, preprocess_image

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

ALLOWED_CLASSES = ("normal", "suspected_opacity", "uncertain")
_BASE_LIMITATIONS = ["no clinical context", "not a validated medical model"]
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


# ---------------------------------------------------------------------------
# Toy predictor: deterministic, filename-driven.
# Kept for the CI smoke tests and `--mode toy`; it is NOT medical inference.
# ---------------------------------------------------------------------------

def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


# ===========================================================================
# Étape 2: real baseline / improved predictors
# ===========================================================================
#
# baseline_v1 : MedGemma-4B vision-language model (Hugging Face Transformers)
#               prompted with prompts/baseline_prompt.txt.
# improved_v1 : the same MedGemma-4B model prompted with the stricter
#               prompts/improved_prompt.txt (explicit uncertainty checks,
#               forced "uncertain" below 0.60 confidence). A fine-tuned
#               checkpoint will replace the base model here in a later étape.
#
# Both modes are gated behind USE_MEDGEMMA and fall back to an automatic
# rule-based predictor (on the quality metrics of PreprocessResult plus a
# focal-opacity image feature) so the pipeline never breaks when the model,
# its weights or a HF token are not available (e.g. in CI).
# ---------------------------------------------------------------------------

# opacity_peak: contrast of the densest blob inside the lung-field ROI, expressed
# in median-absolute-deviation (MAD) units. A focal pneumonic opacity appears as a
# bright, dense blob -> high peak. These thresholds are calibrated on the RSNA
# dev split (étape 3). The value 2.30 was found by grid-search on 150 dev cases;
# 4.0 was the synthetic-set default and produced near-100% uncertain on real images.
_OPACITY_PEAK_THRESHOLD = 2.30
_OPACITY_MARGIN = 0.30  # |peak - threshold| below this is treated as ambiguous


def _roi(arr: np.ndarray) -> np.ndarray:
    """Crop to the central lung-field region, dropping borders/labels."""
    h, w = arr.shape
    return arr[int(h * 0.20):int(h * 0.85), int(w * 0.12):int(w * 0.88)]


def image_features(pre: PreprocessResult) -> dict[str, Any]:
    """Derive interpretable features from a preprocessed chest X-ray."""
    arr = np.asarray(pre.image.convert("L"), dtype=np.float32)
    roi = _roi(arr)
    blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(roi.shape) / 64.0)
    med = float(np.median(blur))
    mad = float(np.mean(np.abs(blur - med))) + 1e-6
    return {
        "opacity_peak": round(float((blur.max() - med) / mad), 3),
        "bright_fraction": round(float((blur > med + 2.2 * mad).mean()), 4),
        "contrast_std": pre.quality_details.get("contrast_std"),
        "brightness_mean": pre.quality_details.get("brightness_mean"),
        "sharpness_lap_var": pre.quality_details.get("sharpness_lap_var"),
    }


def _decide(
    features: dict[str, Any], quality_flag: str, conservative: bool
) -> tuple[str, float, list[str]]:
    """Map image features to (class, confidence, evidence).

    `conservative=True` enables the stricter uncertainty handling of
    improved_prompt.txt: abstain on ambiguous margins and on sub-0.60 confidence.
    """
    if quality_flag == "poor":
        return "uncertain", 0.55, ["image quality is poor; a reliable read is not possible"]

    peak = float(features["opacity_peak"])
    margin = peak - _OPACITY_PEAK_THRESHOLD
    if margin >= 0:
        label = "suspected_opacity"
        confidence = min(0.90, 0.60 + margin * 0.14)
        evidence = [f"a focal dense region stands out in the lung field (opacity_peak={peak})"]
    else:
        label = "normal"
        confidence = min(0.90, 0.60 + (-margin) * 0.12)
        evidence = ["no focal dense opacity detected in the lung fields"]

    if conservative and abs(margin) < _OPACITY_MARGIN:
        label, confidence = "uncertain", min(confidence, 0.55)
        evidence = ["evidence is too borderline to commit to a class"]

    if confidence < 0.60:
        label, confidence = "uncertain", min(confidence, 0.55)

    return label, round(confidence, 3), evidence


def _justify(label: str, features: dict[str, Any], quality_flag: str) -> str:
    base = (
        f"Rule-based read on image quality '{quality_flag}' and a focal-opacity score "
        f"(opacity_peak={features.get('opacity_peak')}, threshold={_OPACITY_PEAK_THRESHOLD})."
    )
    if label == "suspected_opacity":
        return base + " A localized dense region exceeds the opacity threshold; a clinician must confirm."
    if label == "normal":
        return base + " No focal dense region exceeds the opacity threshold in the lung fields."
    return base + " The safe output is uncertainty rather than forcing a class."


def _make_prediction(
    *,
    quality_flag: str,
    label: str,
    confidence: float,
    evidence: list[str],
    features: dict[str, Any],
    model_name: str,
    prompt_version: str,
    start: float,
    extra_limitations: list[str] | None = None,
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    limitations = list(_BASE_LIMITATIONS)
    if extra_limitations:
        limitations.extend(extra_limitations)
    return {
        "image_quality": quality_flag,
        "predicted_class": label,
        "confidence": round(float(confidence), 3),
        "visual_evidence": evidence,
        "justification": _justify(label, features, quality_flag),
        "limitations": limitations,
        "warning": WARNING,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "features": features,
        "preprocessing_flags": extra_flags or [],
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------
# baseline_v1 / improved_v1: MedGemma-4B with graceful rule-based fallback
# ---------------------------------------------------------------------------

_USE_MEDGEMMA = os.getenv("USE_MEDGEMMA", "0") == "1"
_MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
_MEDGEMMA_PIPE = None  # lazily created singleton


def _read_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def _get_medgemma_pipe():
    """Load MedGemma once. Requires `transformers`, `torch`, weights and a HF token."""
    global _MEDGEMMA_PIPE
    if _MEDGEMMA_PIPE is None:
        import torch
        from huggingface_hub import login
        from transformers import pipeline

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        _MEDGEMMA_PIPE = pipeline(
            "image-text-to-text",
            model=_MEDGEMMA_MODEL,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            token=hf_token,
        )
    return _MEDGEMMA_PIPE


def _medgemma_predict(
    pre: PreprocessResult, start: float, *, prompt_file: str, prompt_version: str
) -> dict[str, Any]:
    pipe = _get_medgemma_pipe()
    prompt = _read_prompt(prompt_file)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pre.image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    out = pipe(text=messages, max_new_tokens=400)
    generated = out[0]["generated_text"]
    raw = generated[-1]["content"] if isinstance(generated, list) else str(generated)
    data = _extract_json(raw)

    label = data.get("predicted_class")
    if label not in ALLOWED_CLASSES:
        label = "uncertain"
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    features = image_features(pre)
    evidence = data.get("visual_evidence") or ["see model justification"]
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    # The prompt asks MedGemma to include its own "warning" field, but the
    # value actually shown to the user is always the hardcoded WARNING
    # constant below (exact, audited safety text). `model_warning_present`
    # records whether the model followed that instruction, for prompt-quality
    # monitoring: it does not affect what is displayed.
    model_warning_present = bool(str(data.get("warning") or "").strip())

    return {
        "image_quality": data.get("image_quality", pre.quality_flag),
        "predicted_class": label,
        "confidence": round(confidence, 3),
        "visual_evidence": evidence,
        "justification": str(data.get("justification", "")) or _justify(label, features, pre.quality_flag),
        "limitations": data.get("limitations") or list(_BASE_LIMITATIONS),
        "warning": WARNING,
        "model_warning_present": model_warning_present,
        "model_name": _MEDGEMMA_MODEL,
        "prompt_version": prompt_version,
        "features": features,
        "preprocessing_flags": pre.preprocessing_flags,
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }


def _medgemma_predict_with_fallback(
    image_path: str | Path,
    *,
    prompt_file: str,
    prompt_version: str,
    conservative_fallback: bool,
    fallback_model_name: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    pre = preprocess_image(image_path)

    if _USE_MEDGEMMA:
        try:
            return _medgemma_predict(pre, start, prompt_file=prompt_file, prompt_version=prompt_version)
        except Exception as exc:  # model/token/weights unavailable -> degrade gracefully
            fallback_reason = f"medgemma_unavailable: {type(exc).__name__}: {exc}"
    else:
        fallback_reason = "medgemma_disabled (set USE_MEDGEMMA=1 to enable the VLM)"

    features = image_features(pre)
    label, confidence, evidence = _decide(features, pre.quality_flag, conservative=conservative_fallback)
    return _make_prediction(
        quality_flag=pre.quality_flag,
        label=label,
        confidence=confidence,
        evidence=evidence,
        features=features,
        model_name=fallback_model_name,
        prompt_version=prompt_version,
        start=start,
        extra_limitations=[fallback_reason],
        extra_flags=pre.preprocessing_flags + [fallback_reason],
    )


def baseline_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """baseline_v1: MedGemma-4B VLM prompted with baseline_prompt.txt.

    Enable the real model with `USE_MEDGEMMA=1` (needs transformers, torch,
    the MedGemma weights and a Hugging Face token for the gated repo). When it is
    disabled or fails to load, a rule-based predictor is used instead so the
    pipeline and CI stay green.
    """
    return _medgemma_predict_with_fallback(
        image_path,
        prompt_file="baseline_prompt.txt",
        prompt_version="baseline_v1",
        conservative_fallback=False,
        fallback_model_name="rule-baseline-v1-fallback",
    )


def improved_predict(image_path: str | Path, mode: str = "improved") -> dict[str, Any]:
    """improved_v1: MedGemma-4B VLM prompted with improved_prompt.txt (stricter
    uncertainty rules), with an automatic rule-based fallback.

    A fine-tuned MedGemma checkpoint will replace the base model here in a
    later étape; the prompt_version/model_name contract stays the same.

    Enable the real model with `USE_MEDGEMMA=1` (needs transformers, torch,
    the MedGemma weights and a Hugging Face token for the gated repo). When it is
    disabled or fails to load, a stricter rule-based predictor is used instead so
    the pipeline and CI stay green.
    """
    return _medgemma_predict_with_fallback(
        image_path,
        prompt_file="improved_prompt.txt",
        prompt_version="improved_v1",
        conservative_fallback=True,
        fallback_model_name="rule-improved-v1-fallback",
    )


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

PREDICTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "toy": toy_predict,
    "baseline": baseline_predict,
    "improved": improved_predict,
}


def predict(image_path: str | Path, model: str = "baseline") -> dict[str, Any]:
    """Single entry point used by the API / app layers."""
    return PREDICTORS.get(model, baseline_predict)(image_path)


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Backward-compatible alias for improved_predict.

    The `prompt` argument is accepted for API compatibility but is intentionally
    ignored: improved_predict always reads prompts/improved_prompt.txt from disk,
    so the caller cannot override the prompt at runtime.
    """
    return improved_predict(image_path)
