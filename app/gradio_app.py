from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable no matter where the app is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.pipeline import predict

# Trois configurations : jouet instantané, MedGemma de base, MedGemma fine-tuné.
CONFIGS = {
    "Toy — règles instantanées": ("toy", "baseline"),
    "MedGemma baseline — base + prompt de base": ("medgemma", "baseline"),
    "MedGemma improved — fine-tuné (LoRA) + prompt renforcé": ("medgemma", "improved"),
    "MedGemma dépistage — haute sensibilité (min. faux négatifs)": ("medgemma", "screening"),
}


def analyze(image_path, config):
    if image_path is None:
        return {"error": "no image"}
    engine, mode = CONFIGS[config]
    return predict(image_path, mode=mode, engine=engine)


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="filepath", label="Radiographie thoracique"),
        gr.Radio(list(CONFIGS), value="Toy — règles instantanées", label="Configuration"),
    ],
    outputs=gr.JSON(label="Sortie structurée"),
    title="Assistant radiologue virtuel — prototype pédagogique",
    description="Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
)

if __name__ == "__main__":
    demo.launch()
