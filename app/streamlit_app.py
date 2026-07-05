from __future__ import annotations

import csv
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import re
from PIL import Image
import streamlit as st

# Make the repo root importable no matter where `streamlit run` is launched from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the actual prediction functions and database logging
from src.pipeline import predict
from src.database import insert_run

# Page configuration for a wide, premium layout
st.set_page_config(
    page_title="Assistant Radiologue Virtuel",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection matching the corrected template style
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 17px;
}

/* En-tête */
.app-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 20px 0 16px 0;
    border-bottom: 2px solid #1d4ed8;
    margin-bottom: 24px;
}
.app-header svg {
    flex-shrink: 0;
}
.header-text .main-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1e3a5f;
    letter-spacing: -0.5px;
    margin: 0 0 2px 0;
    line-height: 1.1;
}
.header-text .subtitle {
    font-size: 0.95rem;
    color: #4b72a0;
    font-weight: 400;
    margin: 0;
}

/* Avertissement */
.warning-encart {
    background: #f0f7ff;
    border-left: 4px solid #1d4ed8;
    border-radius: 0 6px 6px 0;
    padding: 12px 18px;
    margin-bottom: 24px;
}
.warning-encart .warn-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1d4ed8;
    margin-bottom: 4px;
}
.warning-encart p {
    margin: 0;
    font-size: 0.9rem;
    color: #1e3a5f;
    line-height: 1.5;
}

/* Labels de section */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1d4ed8;
    padding-bottom: 8px;
    border-bottom: 2px solid #dbeafe;
    margin-bottom: 14px;
}

/* Cartes metriques */
.metric-card {
    background: #f8faff;
    border: 1px solid #dbeafe;
    border-left: 4px solid #1d4ed8;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #4b72a0;
    font-weight: 600;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 4px;
    color: #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# French translation mapping for UI output
TRANSLATIONS = {
    # Classes de prédiction
    "normal": "Normal",
    "suspected_opacity": "Opacité suspectée",
    "uncertain": "Incertain",
    
    # Qualité d'image
    "good": "Bonne",
    "limited": "Limitée",
    "poor": "Médiocre",
    
    # Justifications et observations
    "image quality is poor; a reliable read is not possible": "La qualité de l'image est médiocre ; une lecture fiable n'est pas possible.",
    "evidence is too borderline to commit to a class": "Les indices sont trop limites pour s'engager sur une classe.",
    "a focal dense region stands out in the lung field": "Une zone dense focale se détache dans le champ pulmonaire.",
    "no focal dense opacity detected in the lung fields": "Aucune opacité dense focale détectée dans les champs pulmonaires.",
    "synthetic toy image": "Image de démonstration synthétique",
    "no clinical context": "Absence de contexte clinique",
    "not a validated medical model": "Modèle médical non validé",
    "medgemma_unavailable": "VLM MedGemma non disponible",
    "medgemma_disabled": "VLM MedGemma désactivé",
    "synthetic opacity-like area visible in the lung field": "Zone d'opacité synthétique visible dans le champ pulmonaire.",
    "no synthetic opacity marker detected": "Aucun marqueur d'opacité synthétique détecté.",
    "limited synthetic image quality": "Qualité d'image synthétique limitée.",
    
    # Justifications spécifiques à la démo
    "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation.": "L'image synthétique contient une zone plus brillante localisée compatible avec la classe d'opacité jouet. Il s'agit d'un résultat de validation du pipeline, sans valeur médicale.",
    "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting.": "L'image synthétique ne contient pas le marqueur d'opacité utilisé par le générateur jouet. Cette conclusion est limitée au cadre de validation synthétique.",
    "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class.": "L'image est traitée comme étant de qualité limitée dans le catalogue jouet. La sortie sécurisée est l'incertitude plutôt qu'une classe forcée.",
    
    # Garde-fous et erreurs
    "guardrail triggered: invalid output schema": "Garde-fou déclenché : schéma de sortie invalide.",
    "guardrail triggered: confidence below": "Garde-fou déclenché : confiance inférieure au seuil requis.",
    "screening bias: doubt flagged as suspected_opacity": "Biais de dépistage : doute signalé comme opacité suspectée.",
    "no explicit visual evidence returned": "Aucune preuve visuelle explicite retournée.",
    "model output was not valid JSON": "La sortie du modèle n'était pas un JSON valide.",
    "The model did not return a usable justification; the safe output is uncertainty.": "Le modèle n'a pas renvoyé de justification utilisable ; la sortie sécurisée est l'incertitude.",
    "inference error": "Erreur d'inférence",
}

def translate_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    text_lower = text.lower()
    for eng, fre in TRANSLATIONS.items():
        if eng.lower() in text_lower:
            pattern = re.compile(re.escape(eng), re.IGNORECASE)
            text = pattern.sub(fre, text)
            text_lower = text.lower()
            
    if "rule-based read on image quality" in text_lower:
        text = text.replace("Rule-based read on image quality", "Lecture automatique basée sur la qualité d'image")
        text = text.replace("and a focal-opacity score", "et le score d'opacité")
        text = text.replace("threshold", "seuil")
        
    return text

# Helper function to append inferences to a CSV file for traceability
def log_inference(image_name: str, mode: str, pred: dict):
    log_file = ROOT / "eval" / "inference_log.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists()
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image_name", "model_mode", "predicted_class", "confidence", "latency_ms", "warning"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            mode,
            pred.get("predicted_class", "unknown"),
            pred.get("confidence", 0.0),
            pred.get("latency_ms", 0),
            pred.get("warning", "")
        ])

# Page Header with inline SVG matching corrected template
st.markdown("""
<div class="app-header">
  <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="52" height="52" rx="10" fill="#dbeafe"/>
    <path d="M20 14 C20 14 14 14 14 20 L14 28 C14 33 18 36 22 36 C22 36 22 40 26 40 C30 40 30 36 30 36 C34 36 38 33 38 28 L38 26 C38 23 36 21 33 21 C30 21 28 23 28 26 L28 30 C28 32 27 33 26 33 C25 33 24 32 24 30 L24 20 C24 17 22 14 20 14 Z" stroke="#1d4ed8" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="33" cy="21" r="3" fill="#1d4ed8"/>
    <path d="M17 18 L17 28 C17 31.5 19.5 34 22 34" stroke="#1d4ed8" stroke-width="2" fill="none" stroke-linecap="round"/>
    <path d="M28 14 L32 14 M30 12 L30 16" stroke="#1d4ed8" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <div class="header-text">
    <div class="main-title">Assistant Radiologue Virtuel</div>
    <div class="subtitle">Prototype pédagogique · Analyse de radiographies thoraciques · EFREI 2025</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Safety Warning Box
st.markdown("""
<div class="warning-encart">
    <div class="warn-label">Avertissement non-clinique</div>
    <p>Prototype pédagogique à but démonstratif uniquement. Non certifié comme dispositif médical. Toute interprétation doit être validée par un radiologue qualifié.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar layout for controls and image selection
with st.sidebar:
    st.markdown("### Paramètres")

    # Engine selection radio buttons
    engine_choice = st.radio(
        "Moteur d'analyse",
        ["MedGemma-4B (VLM)", "Démonstration (Moteur de règles)"],
        help="MedGemma-4B : VLM multimodal réel (requiert un GPU). Démonstration : règles instantanées de test."
    )
    engine = "medgemma" if "MedGemma" in engine_choice else "toy"

    # Model configuration select box
    mode = st.selectbox(
        "Configuration",
        ["Baseline (Standard)", "Improved (Prompt Optimisé)", "Dépistage (Haute sensibilité)"],
        help="Baseline : prompt standard. Improved : règle d'incertitude stricte. Dépistage : haute sensibilité (minimise les faux négatifs)."
    )
    if "Baseline" in mode:
        mode_key = "baseline"
    elif "Improved" in mode:
        mode_key = "improved"
    else:
        mode_key = "screening"

    st.markdown("---")
    st.markdown("### Source de l'image")
    
    input_source = st.radio(
        "Source de l'image",
        ["Images de démonstration (30)", "Cas finaux RSNA (30)", "Images RSNA complètes (200)", "Téléverser une image"]
    )

    selected_image_path = None
    uploaded_file = None
    gt_label_display = None

    # Load Ground Truth case mapping if cases.csv exists
    import csv as _csv
    gt_map = {}
    cases_csv = ROOT / "data" / "rsna" / "cases.csv"
    if cases_csv.exists():
        with open(cases_csv, newline="", encoding="utf-8") as _f:
            for row in _csv.DictReader(_f):
                gt_map[row["case_id"]] = row

    if input_source == "Images de démonstration (30)":
        samples_dir = ROOT / "data" / "sample_images"
        if samples_dir.exists():
            sample_files = sorted(list(samples_dir.glob("*.png")))
            sample_names = [f.name for f in sample_files]
            
            selected_name = st.selectbox(
                "Choisir une image de test", 
                ["-- Sélectionner --"] + sample_names
            )
            
            if selected_name != "-- Sélectionner --":
                selected_image_path = samples_dir / selected_name
        else:
            st.error("Dossier data/sample_images introuvable.")

    elif input_source == "Cas finaux RSNA (30)":
        final_cases = [r for r in gt_map.values() if r.get("split") == "final"]
        final_cases = sorted(final_cases, key=lambda r: r["case_id"])
        
        if not final_cases:
            st.warning("⚠️ Fichier `data/rsna/cases.csv` introuvable ou vide. Veuillez exécuter la préparation des données RSNA.")
        else:
            options = ["-- Sélectionner --"] + [
                f"{r['case_id']}  [{r['label']}]" for r in final_cases
            ]
            selected_name = st.selectbox(
                f"Cas de soutenance ({len(final_cases)} disponibles)",
                options
            )
            if selected_name != "-- Sélectionner --":
                case_id = selected_name.split("  [")[0]
                row = gt_map[case_id]
                selected_image_path = ROOT / row["image_path"]
                gt_label_display = row["label"]
                label_icons = {"normal": "OK", "suspected_opacity": "OPACITE", "uncertain": "INCERTAIN"}
                st.caption(f"Label réel : **{label_icons.get(gt_label_display, gt_label_display)}** · qualité : {row.get('quality', '?')}")

    elif input_source == "Images RSNA complètes (200)":
        rsna_dir = ROOT / "data" / "rsna" / "processed" / "images"
        files = sorted(rsna_dir.glob("*.png")) if rsna_dir.exists() else []
        if not files:
            st.warning("⚠️ Dossier `data/rsna/processed/images` introuvable. Veuillez exécuter la préparation des données RSNA.")
        else:
            fname_to_label = {
                Path(r["image_path"]).name: r["label"] for r in gt_map.values()
            }
            names = [f.name for f in files]
            selected_name = st.selectbox(
                f"Image RSNA ({len(names)} disponibles)", ["-- Sélectionner --"] + names
            )
            if selected_name != "-- Sélectionner --":
                selected_image_path = rsna_dir / selected_name
                gt = fname_to_label.get(selected_name, "inconnu")
                st.caption(f"Label réel : **{gt}**")

    else:
        uploaded_file = st.file_uploader(
            "Téléverser une radiographie (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )

# Main layout content processing
image_to_process = None
image_name_log = ""

if selected_image_path:
    image_to_process = selected_image_path
    image_name_log = selected_image_path.name
elif uploaded_file:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        image_to_process = Path(tmp.name)
    image_name_log = uploaded_file.name

if image_to_process:
    col1, col2 = st.columns([4, 5])
    
    with col1:
        st.markdown('<div class="section-label">Radiographie</div>', unsafe_allow_html=True)
        st.image(Image.open(image_to_process), caption=image_name_log, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Résultat d\'analyse</div>', unsafe_allow_html=True)
        
        # Execute the prediction with a dynamic, informative progress bar
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_text.markdown("🔄 **Étape 1/5** : Initialisation du pipeline d'analyse...")
            progress_bar.progress(10)
            time.sleep(0.1)
            
            if engine == "medgemma":
                status_text.markdown("🔄 **Étape 2/5** : Préparation du GPU et chargement du modèle MedGemma (peut prendre quelques minutes lors de la première initialisation)...")
            else:
                status_text.markdown("🔄 **Étape 2/5** : Préparation du moteur de règles de démonstration...")
            progress_bar.progress(30)
            time.sleep(0.1)
            
            status_text.markdown("🔄 **Étape 3/5** : Exécution de l'inférence sur l'image...")
            progress_bar.progress(60)
            
            # Execute actual prediction
            pred = predict(image_to_process, mode=mode_key, engine=engine)
            
            status_text.markdown("🔄 **Étape 4/5** : Application des garde-fous cliniques et de sécurité...")
            progress_bar.progress(85)
            time.sleep(0.1)
            
            status_text.markdown("🔄 **Étape 5/5** : Finalisation et journalisation de l'analyse...")
            progress_bar.progress(100)
            time.sleep(0.1)
            
            status_text.empty()
            progress_bar.empty()
            
            # Log to CSV and SQLite for traceability
            log_inference(image_name_log, mode_key, pred)
            
            db_path = ROOT / "medical_ai_evidence.sqlite"
            case_id = Path(image_name_log).stem
            insert_run(db_path, case_id, str(image_to_process), pred)
            
        except Exception as exc:
            status_text.empty()
            progress_bar.empty()
            st.error(
                f"❌ **Échec de l'inférence** ({engine} / {mode_key}) : `{type(exc).__name__}`.\n\n"
                "Les configurations MedGemma requièrent un GPU NVIDIA, une build CUDA de "
                "PyTorch et l'accès au modèle (`huggingface-cli login`). La configuration "
                "**Démonstration** fonctionne sans GPU: utilise-la pour une démo qui marche à coup sûr."
            )
            st.stop()
            
        # Display GPU Warning on CPU fallback
        if pred.get("device") == "cpu":
            st.warning("⚠️ **Exécution sur CPU** : Le modèle MedGemma s'exécute sur le processeur (GPU indisponible). Cela va ralentir considérablement les analyses (~5 minutes).")
            
        # Check if MedGemma is unavailable and the rule-based fallback is active
        is_rule_fallback = engine == "toy" or any("medgemma" in str(limit).lower() for limit in pred.get("limitations", []))
        if is_rule_fallback:
            st.info("💡 **Mode fallback actif** : MedGemma non disponible (ou moteur de démonstration sélectionné). L'analyse utilise un classificateur basé sur les caractéristiques d'image (score d'opacité focale, qualité). Résultat déterministe, aucune valeur médicale.")
            
        # Display key metrics in cards
        m_col1, m_col2, m_col3 = st.columns(3)
        
        # Translate values for visual display
        class_val = pred["predicted_class"]
        translated_class = TRANSLATIONS.get(class_val, class_val.replace('_', ' ').title())
        class_colors = {
            "normal": "#059669",
            "suspected_opacity": "#dc2626",
            "uncertain": "#d97706"
        }
        text_color = class_colors.get(class_val, "#374151")

        with m_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid {text_color};">
                <div class="metric-label">Classe prédite</div>
                <div class="metric-value" style="color:{text_color};">{translated_class}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Indice de confiance</div>
                <div class="metric-value">{pred['confidence'] * 100:.0f} %</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pred['confidence'])

        with m_col3:
            latency_s = pred['latency_ms'] / 1000
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temps de traitement</div>
                <div class="metric-value">{latency_s:.1f} s</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        tab_obs, tab_just, tab_lim = st.tabs(["Observations", "Justification", "Limites"])
        
        with tab_obs:
            st.markdown("**Indices visuels détectés :**")
            for evidence in pred["visual_evidence"]:
                st.markdown(f"- {translate_text(evidence)}")
            st.markdown(f"**Qualité de l'image :** `{translate_text(pred.get('image_quality', 'unknown'))}`")
            
        with tab_just:
            st.markdown(f"*{translate_text(pred['justification'])}*")
            
        with tab_lim:
            st.markdown("**Facteurs limitants identifiés :**")
            for limit in pred["limitations"]:
                st.markdown(f"- {translate_text(limit)}")
                
        # Generate fully translated copy of JSON data for UI
        pred_translated = {
            "qualite_image": translate_text(pred.get("image_quality")),
            "classe_predite": translated_class,
            "confiance": pred.get("confidence"),
            "indices_visuels": [translate_text(e) for e in pred.get("visual_evidence", [])],
            "justification": translate_text(pred.get("justification")),
            "limites": [translate_text(l) for l in pred.get("limitations", [])],
            "avertissement": translate_text(pred.get("warning")),
            "nom_modele": pred.get("model_name"),
            "version_prompt": pred.get("prompt_version"),
            "latence_ms": pred.get("latency_ms")
        }
        if "device" in pred:
            pred_translated["appareil_execution"] = pred.get("device")
        if "guardrail_errors" in pred and pred["guardrail_errors"]:
            pred_translated["erreurs_garde_fou"] = [translate_text(e) for e in pred["guardrail_errors"]]
            
        with st.expander("JSON · Sortie structuree complete", expanded=False):
            st.json(pred_translated)
            
else:
    st.info("Sélectionnez une image dans la barre latérale ou téléversez une radiographie pour démarrer l'analyse.")
