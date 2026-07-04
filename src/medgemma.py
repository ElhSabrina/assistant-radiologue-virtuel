"""MedGemma-4b vision-language connector for the educational prototype.

This module wraps ``google/medgemma-4b-it`` behind the same output schema as
``src.inference.toy_predict``. It is deliberately conservative: the model output
is parsed defensively, coerced to the allowed classes, and any failure degrades
to ``uncertain`` rather than raising.

Loading is lazy and cached: the model is only pulled into memory on the first
prediction, so importing this module (e.g. during tests) stays cheap. On a
CUDA GPU the weights are loaded in 4-bit (bitsandbytes) to fit a 6 GB card; on
CPU the model runs unquantised but very slowly.

Non-clinical prototype only. No output here is a medical diagnosis.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
MODEL_ID = "google/medgemma-4b-it"

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
ALLOWED_QUALITY = {"good", "limited", "poor"}

SYSTEM_PROMPT = (
    "You are an educational radiology assistant for engineering students. "
    "You are not a clinician and must never provide a definitive diagnosis. "
    "You only describe what is visible on the provided frontal chest X-ray and "
    "you always answer with a single valid JSON object, nothing else."
)

_PROMPT_FILES = {
    "baseline": "baseline_prompt.txt",
    "improved": "improved_prompt.txt",
    "screening": "screening_prompt.txt",
}


def load_prompt(version: str) -> str:
    """Return the user prompt text for a prompt version (baseline/improved)."""
    filename = _PROMPT_FILES.get(version)
    if filename is None:
        raise ValueError(f"Unknown prompt version: {version!r}")
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


DEFAULT_ADAPTER_DIR = ROOT / "finetuning" / "adapter"


def _adapter_path() -> str | None:
    """Path to a trained LoRA adapter.

    Uses ``MEDGEMMA_ADAPTER`` if set, otherwise auto-detects the repo's
    ``finetuning/adapter/`` when it holds a valid adapter — so the fine-tuned
    ("improved") model works with no environment variable to set.
    """
    import os

    path = os.environ.get("MEDGEMMA_ADAPTER")
    if not path and (DEFAULT_ADAPTER_DIR / "adapter_config.json").exists():
        path = str(DEFAULT_ADAPTER_DIR)
    return path if path and Path(path).exists() else None


def _apply_adapter(model):
    """Wrap the base model with the LoRA adapter (assumes MEDGEMMA_ADAPTER set)."""
    # On Windows, peft's deep import chain can overflow the thread stack when
    # pandas is imported lazily deep in that chain; importing it shallow first
    # avoids the crash.
    import pandas  # noqa: F401
    from peft import PeftModel

    return PeftModel.from_pretrained(model, _adapter_path())


@lru_cache(maxsize=1)
def _load_model():
    """Load MedGemma once. Returns (processor, model, device_str, has_adapter).

    Tries GPU 4-bit first, then falls back to CPU. If ``MEDGEMMA_ADAPTER`` points
    at a trained LoRA adapter it is loaded on top of the base model; callers then
    toggle it per request (base weights for the baseline, adapter for improved).
    Import of torch/transformers is done here so the rest of the repo (and the toy
    pipeline) does not depend on heavy libraries being installed.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = None
    device_tag = "cpu"

    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            # bfloat16 (not float16) is required on GTX 16xx / Turing cards without
            # tensor cores: float16 generation produces NaNs and empty output there.
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                quantization_config=quant,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                low_cpu_mem_usage=True,
            )
            device_tag = "cuda-4bit"
        except Exception:
            # bitsandbytes unavailable or OOM: fall through to CPU.
            torch.cuda.empty_cache()
            model = None

    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        device_tag = "cpu"

    has_adapter = _adapter_path() is not None
    if has_adapter:
        model = _apply_adapter(model)
    return processor, model, device_tag, has_adapter


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from model text."""
    # Strip common markdown code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Fall back to the outermost balanced-looking braces.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _coerce_schema(raw: dict[str, Any] | None, latency_ms: int, version: str) -> dict[str, Any]:
    """Coerce a possibly-messy model dict into the strict output schema."""
    parse_ok = raw is not None
    raw = raw or {}

    cls = str(raw.get("predicted_class", "")).strip().lower()
    if cls not in ALLOWED_CLASSES:
        cls = "uncertain"

    quality = str(raw.get("image_quality", "")).strip().lower()
    if quality not in ALLOWED_QUALITY:
        quality = "limited"

    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)

    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    evidence = _as_list(raw.get("visual_evidence")) or ["no explicit visual evidence returned"]
    limitations = _as_list(raw.get("limitations"))
    for base in ("not a validated medical model", "no clinical context"):
        if base not in limitations:
            limitations.append(base)
    if not parse_ok:
        limitations.append("model output was not valid JSON")

    justification = str(raw.get("justification", "")).strip() or (
        "The model did not return a usable justification; the safe output is uncertainty."
    )

    return {
        "image_quality": quality,
        "predicted_class": cls,
        "confidence": round(conf, 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": limitations,
        "warning": WARNING,
        "model_name": MODEL_ID,
        "prompt_version": f"{version}_v1",
        "latency_ms": latency_ms,
        "json_parse_ok": parse_ok,
    }


def _make_json_stopper(processor, prompt_len: int):
    """Stop generation once a balanced top-level JSON object has been produced.

    On a slow GPU this saves the baseline prompt from rambling up to
    ``max_new_tokens`` after it has already emitted the JSON answer. The brace
    depth is tracked incrementally (only the newly generated tokens are decoded
    each step) so the check stays O(n) over the whole generation.
    """
    from transformers import StoppingCriteria

    class _JsonComplete(StoppingCriteria):
        def __init__(self) -> None:
            self.seen = prompt_len
            self.started = False
            self.depth = 0

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            seq = input_ids[0]
            new = seq[self.seen:]
            self.seen = seq.shape[-1]
            for ch in processor.decode(new, skip_special_tokens=True):
                if ch == "{":
                    self.started = True
                    self.depth += 1
                elif ch == "}":
                    self.depth -= 1
                    if self.started and self.depth <= 0:
                        return True
            return False

    return _JsonComplete()


def medgemma_predict(
    image_path: str | Path,
    version: str = "baseline",
    max_new_tokens: int = 320,
) -> dict[str, Any]:
    """Run MedGemma on one chest X-ray and return the strict output schema.

    Any loading/inference/parse failure degrades to a valid ``uncertain`` result
    so the calling pipeline never crashes on a single bad image.
    """
    import contextlib
    import torch
    from transformers import StoppingCriteriaList

    prompt = load_prompt(version)
    start = time.perf_counter()
    try:
        processor, model, device, has_adapter = _load_model()
        # The fine-tuned adapter is the "improved" model; the baseline runs the
        # base weights. With an adapter loaded, we toggle it per request.
        use_adapter = has_adapter and version == "improved"
        adapter_ctx = (
            model.disable_adapter()
            if (has_adapter and not use_adapter)
            else contextlib.nullcontext()
        )
        model_name = f"{MODEL_ID}+lora" if use_adapter else MODEL_ID

        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode(), adapter_ctx:
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_make_json_stopper(processor, input_len)]),
            )
        new_tokens = generation[0][input_len:]
        text = processor.decode(new_tokens, skip_special_tokens=True)
        raw = _extract_json(text)
    except Exception as exc:  # noqa: BLE001 - degrade instead of crashing the batch
        latency_ms = int((time.perf_counter() - start) * 1000)
        result = _coerce_schema(None, latency_ms, version)
        result["confidence"] = min(result["confidence"], 0.4)
        result["device"] = "error"
        result["model_name"] = f"{MODEL_ID}+lora" if version == "improved" else MODEL_ID
        result["limitations"].append(f"inference error: {type(exc).__name__}")
        return result

    latency_ms = int((time.perf_counter() - start) * 1000)
    result = _coerce_schema(raw, latency_ms, version)
    result["device"] = device
    result["model_name"] = model_name
    return result
