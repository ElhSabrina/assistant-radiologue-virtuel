from __future__ import annotations

from pathlib import Path

from src.guardrails import WARNING_TEXT, validate_prediction
from src.inference import baseline_predict, image_features, improved_predict

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "sample_images"


def _sample(name: str) -> Path:
    return SAMPLES / name


def test_baseline_predict_schema_is_valid() -> None:
    pred = baseline_predict(_sample("CXR_SYN_001_normal.png"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["model_name"] == "rule-baseline-v1"
    assert pred["prompt_version"] == "baseline_v1"
    assert pred["warning"] == WARNING_TEXT
    assert "opacity_peak" in pred["features"]
    assert pred["latency_ms"] >= 0


def test_baseline_separates_the_three_synthetic_classes() -> None:
    # Thresholds are calibrated on the synthetic smoke set, so these are stable.
    assert baseline_predict(_sample("CXR_SYN_001_normal.png"))["predicted_class"] == "normal"
    assert baseline_predict(_sample("CXR_SYN_002_suspected_opacity.png"))["predicted_class"] == "suspected_opacity"
    # Low-quality images are flagged "poor" by preprocessing -> safe abstention.
    assert baseline_predict(_sample("CXR_SYN_003_uncertain.png"))["predicted_class"] == "uncertain"


def test_improved_predict_falls_back_without_medgemma() -> None:
    pred = improved_predict(_sample("CXR_SYN_002_suspected_opacity.png"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["prompt_version"] == "improved_v1"
    assert pred["model_name"] == "rule-improved-v1-fallback"
    # The fallback must be transparent about why MedGemma was not used.
    assert any("medgemma" in str(item).lower() for item in pred["limitations"])


def test_image_features_are_bounded() -> None:
    from src.preprocessing import preprocess_image

    feats = image_features(preprocess_image(_sample("CXR_SYN_001_normal.png")))
    assert feats["opacity_peak"] >= 0
    assert 0.0 <= feats["bright_fraction"] <= 1.0
