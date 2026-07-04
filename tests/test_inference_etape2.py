from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.guardrails import WARNING_TEXT, validate_prediction
from src.inference import (
    baseline_predict,
    image_features,
    improved_predict,
    predict,
    toy_predict,
    vlm_predict_placeholder,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "sample_images"
PROMPTS_DIR = ROOT / "prompts"

_ALL_SYNTH = [
    "CXR_SYN_001_normal.png",
    "CXR_SYN_002_suspected_opacity.png",
    "CXR_SYN_003_uncertain.png",
]

RSNA_TEST_DIR = ROOT / "data" / "rsna_pneumonia" / "Test"
_RSNA_SAMPLES = [
    "0000a175-0e68-4ca4-b1af-167204a7e0bc.png",
    "000686d7-f4fc-448d-97a0-44fa9c5d3aa6.png",
    "00271e8e-aea8-4f0a-8a34-3025831f1079.png",
]

_MEDGEMMA_ENABLED = os.getenv("USE_MEDGEMMA", "0") == "1"
_HF_TOKEN_SET = bool(os.getenv("HF_TOKEN", "").strip())
_MEDGEMMA_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")


def _sample(name: str) -> Path:
    return SAMPLES / name


def _rsna(name: str) -> Path:
    return RSNA_TEST_DIR / name


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "rsna: tests that require RSNA images on disk")
    config.addinivalue_line("markers", "medgemma: tests that require USE_MEDGEMMA=1 and HF_TOKEN")
    config.addinivalue_line("markers", "no_medgemma: fallback tests that run only when USE_MEDGEMMA is disabled")


def _rsna_available() -> bool:
    return all((RSNA_TEST_DIR / n).exists() for n in _RSNA_SAMPLES)


# Marks de saut: chaque groupe de tests est isolé de l'autre.
rsna_only = pytest.mark.skipif(
    not _rsna_available(),
    reason="RSNA Test images not found: skipping real-image tests",
)

# Fallback tests : skippés si MedGemma est actif (sinon les assertions sur
# "rule-*-fallback" échoueraient car le vrai modèle répondrait).
no_medgemma = pytest.mark.skipif(
    _MEDGEMMA_ENABLED,
    reason="Fallback tests disabled when USE_MEDGEMMA=1: run medgemma tests instead",
)

# Tests MedGemma : skippés si USE_MEDGEMMA=0 ou si HF_TOKEN est absent.
medgemma_only = pytest.mark.skipif(
    not (_MEDGEMMA_ENABLED and _HF_TOKEN_SET),
    reason="MedGemma tests require USE_MEDGEMMA=1 and HF_TOKEN set in environment",
)


# ===========================================================================
# GROUPE 1: Fallback (sans MedGemma)
# Vérifie que le pipeline reste opérationnel quand le VLM n'est pas disponible.
# ===========================================================================

@no_medgemma
def test_baseline_predict_schema_is_valid() -> None:
    pred = baseline_predict(_sample("CXR_SYN_001_normal.png"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["model_name"] == "rule-baseline-v1-fallback"
    assert pred["prompt_version"] == "baseline_v1"
    assert pred["warning"] == WARNING_TEXT
    assert "opacity_peak" in pred["features"]
    assert pred["latency_ms"] >= 0
    assert any("medgemma" in str(item).lower() for item in pred["limitations"])


@no_medgemma
def test_baseline_rule_fallback_class_guarantees() -> None:
    # High-peak synthetic image -> suspected_opacity (peak ≈ 4.65, well above threshold 2.30)
    assert baseline_predict(_sample("CXR_SYN_002_suspected_opacity.png"))["predicted_class"] == "suspected_opacity"
    # Poor-quality image -> uncertain (safety trigger, regardless of peak)
    assert baseline_predict(_sample("CXR_SYN_003_uncertain.png"))["predicted_class"] == "uncertain"


@no_medgemma
def test_improved_predict_falls_back_without_medgemma() -> None:
    pred = improved_predict(_sample("CXR_SYN_002_suspected_opacity.png"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["prompt_version"] == "improved_v1"
    assert pred["model_name"] == "rule-improved-v1-fallback"
    assert any("medgemma" in str(item).lower() for item in pred["limitations"])


@no_medgemma
def test_improved_predict_schema_is_valid_on_all_samples() -> None:
    for name in _ALL_SYNTH:
        pred = improved_predict(_sample(name))
        valid, errors = validate_prediction(pred)
        assert valid, f"{name}: {errors}"
        assert pred["warning"] == WARNING_TEXT
        assert pred["latency_ms"] >= 0


@no_medgemma
def test_fallback_reason_is_medgemma_disabled() -> None:
    """Le fallback doit indiquer explicitement pourquoi MedGemma n'est pas utilisé."""
    pred = baseline_predict(_sample("CXR_SYN_001_normal.png"))
    disabled_flags = [
        item for item in pred["limitations"]
        if "medgemma_disabled" in str(item).lower()
    ]
    assert disabled_flags, (
        f"Expected 'medgemma_disabled' in limitations, got: {pred['limitations']}"
    )


@no_medgemma
def test_predict_dispatches_to_baseline_by_default() -> None:
    pred = predict(_sample("CXR_SYN_001_normal.png"))
    assert pred["model_name"] == "rule-baseline-v1-fallback"
    assert pred["prompt_version"] == "baseline_v1"


@no_medgemma
def test_predict_dispatches_to_improved() -> None:
    pred = predict(_sample("CXR_SYN_001_normal.png"), model="improved")
    assert pred["model_name"] == "rule-improved-v1-fallback"
    assert pred["prompt_version"] == "improved_v1"


# ===========================================================================
# GROUPE 2: Intégration MedGemma (USE_MEDGEMMA=1 + HF_TOKEN requis)
# Vérifie que la connexion HuggingFace fonctionne et que le vrai modèle
# répond avec un schéma valide.
# ===========================================================================

@medgemma_only
def test_medgemma_connection_is_established() -> None:
    """Vérifie que le pipeline MedGemma se charge sans erreur (connexion HF OK)."""
    pred = baseline_predict(_sample("CXR_SYN_001_normal.png"))

    # Si la connexion a échoué, le fallback se déclencherait avec "medgemma_unavailable"
    unavailable = [
        item for item in pred.get("limitations", [])
        if "medgemma_unavailable" in str(item).lower()
    ]
    assert not unavailable, (
        f"MedGemma failed to load: connexion HF ou token invalide: {unavailable}"
    )
    assert pred["model_name"] == _MEDGEMMA_MODEL_ID, (
        f"Expected real model '{_MEDGEMMA_MODEL_ID}', got fallback '{pred['model_name']}'"
    )


@medgemma_only
def test_medgemma_baseline_schema_is_valid() -> None:
    for name in _ALL_SYNTH:
        pred = baseline_predict(_sample(name))
        valid, errors = validate_prediction(pred)
        assert valid, f"{name}: {errors}"
        assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["warning"] == WARNING_TEXT
        assert pred["prompt_version"] == "baseline_v1"
        assert pred["model_name"] == _MEDGEMMA_MODEL_ID
        assert pred["latency_ms"] >= 0


@medgemma_only
def test_medgemma_improved_schema_is_valid() -> None:
    for name in _ALL_SYNTH:
        pred = improved_predict(_sample(name))
        valid, errors = validate_prediction(pred)
        assert valid, f"{name}: {errors}"
        assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["warning"] == WARNING_TEXT
        assert pred["prompt_version"] == "improved_v1"
        assert pred["model_name"] == _MEDGEMMA_MODEL_ID


@medgemma_only
def test_medgemma_no_fallback_flag_in_limitations() -> None:
    """Aucune limitation de type 'fallback' ne doit apparaître quand MedGemma tourne."""
    for predictor in (baseline_predict, improved_predict):
        pred = predictor(_sample("CXR_SYN_001_normal.png"))
        fallback_flags = [
            item for item in pred.get("limitations", [])
            if "medgemma_disabled" in str(item).lower()
            or "medgemma_unavailable" in str(item).lower()
        ]
        assert not fallback_flags, (
            f"{predictor.__name__} returned fallback flags with MedGemma enabled: {fallback_flags}"
        )


@medgemma_only
def test_medgemma_improved_is_more_conservative_than_baseline() -> None:
    """Le mode improved doit produire une confiance <= baseline sur image ambiguë."""
    baseline = baseline_predict(_sample("CXR_SYN_003_uncertain.png"))
    improved = improved_predict(_sample("CXR_SYN_003_uncertain.png"))
    assert improved["confidence"] <= baseline["confidence"] + 0.05, (
        "improved_predict devrait être plus conservateur que baseline sur image ambiguë"
    )


@medgemma_only
def test_medgemma_features_are_extracted() -> None:
    """Le vrai modèle doit aussi retourner les features image calculées côté code."""
    pred = baseline_predict(_sample("CXR_SYN_001_normal.png"))
    assert "features" in pred
    for key in ("opacity_peak", "bright_fraction"):
        assert key in pred["features"], f"feature manquante: {key}"


# ===========================================================================
# GROUPE 3: toy_predict (indépendant de MedGemma, toujours actif)
# ===========================================================================

def test_toy_predict_has_required_schema_fields() -> None:
    pred = toy_predict(_sample("CXR_SYN_001_normal.png"))
    valid, errors = validate_prediction(pred)
    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["model_name"] == "toy-rule-baseline"
    assert pred["prompt_version"] == "baseline_v1"
    assert pred["warning"] == WARNING_TEXT
    assert pred["latency_ms"] >= 0


def test_toy_predict_improved_mode_metadata() -> None:
    pred = toy_predict(_sample("CXR_SYN_001_normal.png"), mode="improved")
    assert pred["model_name"] == "toy-rule-improved"
    assert pred["prompt_version"] == "improved_v1"


def test_predict_dispatches_to_toy() -> None:
    pred = predict(_sample("CXR_SYN_001_normal.png"), model="toy")
    assert pred["model_name"] == "toy-rule-baseline"
    assert pred["prompt_version"] == "baseline_v1"


def test_predict_unknown_model_falls_back_to_baseline() -> None:
    pred = predict(_sample("CXR_SYN_001_normal.png"), model="nonexistent_model")
    assert pred["prompt_version"] == "baseline_v1"


# ===========================================================================
# GROUPE 4: Fichiers de prompts (indépendant de MedGemma)
# ===========================================================================

def test_baseline_prompt_exists_and_is_non_empty() -> None:
    path = PROMPTS_DIR / "baseline_prompt.txt"
    assert path.exists(), "baseline_prompt.txt introuvable"
    assert len(path.read_text(encoding="utf-8").strip()) > 0


def test_improved_prompt_exists_and_is_non_empty() -> None:
    path = PROMPTS_DIR / "improved_prompt.txt"
    assert path.exists(), "improved_prompt.txt introuvable"
    assert len(path.read_text(encoding="utf-8").strip()) > 0


def test_improved_prompt_is_stricter_than_baseline() -> None:
    baseline = (PROMPTS_DIR / "baseline_prompt.txt").read_text(encoding="utf-8")
    improved = (PROMPTS_DIR / "improved_prompt.txt").read_text(encoding="utf-8")
    assert "0.60" in improved, "improved_prompt.txt doit mentionner le seuil 0.60"
    assert baseline != improved


# ===========================================================================
# GROUPE 5: image_features (indépendant de MedGemma)
# ===========================================================================

def test_image_features_are_bounded() -> None:
    from src.preprocessing import preprocess_image

    feats = image_features(preprocess_image(_sample("CXR_SYN_001_normal.png")))
    assert feats["opacity_peak"] >= 0
    assert 0.0 <= feats["bright_fraction"] <= 1.0


def test_image_features_has_all_expected_keys() -> None:
    from src.preprocessing import preprocess_image

    feats = image_features(preprocess_image(_sample("CXR_SYN_001_normal.png")))
    for key in ("opacity_peak", "bright_fraction", "contrast_std", "brightness_mean", "sharpness_lap_var"):
        assert key in feats, f"image_features missing key: {key}"


def test_confidence_is_in_valid_range_for_all_predictors() -> None:
    for predictor in (baseline_predict, improved_predict):
        for name in _ALL_SYNTH:
            pred = predictor(_sample(name))
            conf = pred["confidence"]
            assert 0.0 <= conf <= 1.0, (
                f"{predictor.__name__}({name}) returned confidence={conf}"
            )


def test_warning_is_hardcoded_constant_across_all_predictors() -> None:
    for predictor in (baseline_predict, improved_predict, toy_predict):
        pred = predictor(_sample("CXR_SYN_001_normal.png"))
        assert pred["warning"] == WARNING_TEXT, (
            f"{predictor.__name__} returned unexpected warning: {pred['warning']!r}"
        )


# ===========================================================================
# GROUPE 6: vlm_predict_placeholder (alias backward-compat)
# ===========================================================================

@no_medgemma
def test_vlm_predict_placeholder_returns_improved_schema() -> None:
    prompt = (PROMPTS_DIR / "improved_prompt.txt").read_text(encoding="utf-8")
    pred = vlm_predict_placeholder(_sample("CXR_SYN_002_suspected_opacity.png"), prompt=prompt)
    valid, errors = validate_prediction(pred)
    assert valid, errors
    assert pred["prompt_version"] == "improved_v1"
    assert pred["model_name"] == "rule-improved-v1-fallback"


# ===========================================================================
# GROUPE 7: Images réelles RSNA (skippées si images absentes)
# ===========================================================================

@rsna_only
@pytest.mark.parametrize("name", _RSNA_SAMPLES)
def test_baseline_schema_valid_on_rsna_image(name: str) -> None:
    pred = baseline_predict(_rsna(name))
    valid, errors = validate_prediction(pred)
    assert valid, f"{name}: {errors}"
    assert pred["warning"] == WARNING_TEXT
    assert pred["latency_ms"] >= 0


@rsna_only
@pytest.mark.parametrize("name", _RSNA_SAMPLES)
def test_improved_schema_valid_on_rsna_image(name: str) -> None:
    pred = improved_predict(_rsna(name))
    valid, errors = validate_prediction(pred)
    assert valid, f"{name}: {errors}"
    assert pred["warning"] == WARNING_TEXT


@rsna_only
@pytest.mark.parametrize("name", _RSNA_SAMPLES)
def test_rsna_image_predicted_class_is_allowed(name: str) -> None:
    for predictor in (baseline_predict, improved_predict):
        pred = predictor(_rsna(name))
        assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}, (
            f"{predictor.__name__}({name}) returned invalid class: {pred['predicted_class']!r}"
        )


@rsna_only
@pytest.mark.parametrize("name", _RSNA_SAMPLES)
def test_rsna_image_confidence_in_valid_range(name: str) -> None:
    for predictor in (baseline_predict, improved_predict):
        pred = predictor(_rsna(name))
        assert 0.0 <= pred["confidence"] <= 1.0


@rsna_only
@pytest.mark.parametrize("name", _RSNA_SAMPLES)
def test_rsna_image_features_are_extracted(name: str) -> None:
    from src.preprocessing import preprocess_image

    feats = image_features(preprocess_image(_rsna(name)))
    for key in ("opacity_peak", "bright_fraction", "contrast_std", "brightness_mean", "sharpness_lap_var"):
        assert key in feats, f"{name}: image_features missing key '{key}'"
    assert feats["opacity_peak"] >= 0
    assert 0.0 <= feats["bright_fraction"] <= 1.0
