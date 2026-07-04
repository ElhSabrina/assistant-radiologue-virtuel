# Étape 2: Inférence baseline & amélioration

Ce document décrit le prédicteur réel qui remplace `toy_predict` (squelette).

## Vue d'ensemble

| Variante | Fonction | Modèle | Idée |
|----------|----------|--------|------|
| `baseline_v1` | `baseline_predict` | règles + seuils | Option A de l'appel d'offre : classification à partir des métriques qualité de `PreprocessResult` et d'un descripteur d'opacité focale. |
| `improved_v1` | `improved_predict` | MedGemma‑4B (VLM) | Option B : `google/medgemma-4b-it` via Hugging Face Transformers, avec repli automatique sur des règles plus prudentes. |

Les deux variantes renvoient le **même schéma JSON** (`prompts/json_schema.md`) et passent par `apply_safety_guardrails`.

## baseline_v1 (règles)

1. `preprocess_image` produit l'image 512×512 CLAHE et un `quality_flag` (`good`/`limited`/`poor`).
2. `image_features` calcule un **`opacity_peak`** : contraste du blob le plus dense dans le ROI pulmonaire central, exprimé en unités de MAD (median absolute deviation).
3. Décision :
   - qualité `poor` → `uncertain` (abstention sûre) ;
   - `opacity_peak ≥ 4.0` → `suspected_opacity` ;
   - sinon → `normal`.

Les seuils sont **calibrés sur le jeu synthétique** (`data/synthetic_cases.csv`) et doivent être ré‑étalonnés sur le split `dev` RSNA à l'étape 3.

## improved_v1 (MedGemma + repli)

- Activation : `USE_MEDGEMMA=1` (nécessite `transformers`, `torch`, les poids MedGemma et un token Hugging Face pour le dépôt *gated*). Modèle configurable via `MEDGEMMA_MODEL_ID`.
- Le prompt `prompts/improved_prompt.txt` impose des règles d'incertitude strictes (projection, exposition, `confidence < 0.60 → uncertain`).
- **Repli automatique** : si le modèle/token/poids ne sont pas disponibles (ex. CI), on bascule sur le prédicteur à règles en mode *conservateur* (abstention sur marges ambiguës), et la raison du repli est journalisée dans `limitations` / `preprocessing_flags`. La sortie reste valide → la CI et les tests restent verts.

## Lancer l'évaluation

```bash
# Comparaison réelle baseline_v1 vs improved_v1 (données synthétiques par défaut)
python eval/run_evaluation.py --mode real

# Sur le vrai jeu RSNA une fois l'étape 1 fusionnée (étape 3)
python eval/run_evaluation.py --mode real --cases data/rsna/cases.csv

# Mode toy déterministe conservé pour le smoke-test CI
python eval/run_evaluation.py --mode toy
```

Sorties écrites dans `eval/outputs/` : `*_predictions.csv`, `*_metrics.json`, `before_after_summary.csv`.

## Limite connue

Sur le jeu **synthétique**, les classes sont nettement séparables : `baseline_v1` et `improved_v1` y obtiennent des scores proches. L'écart « amélioration mesurée » se constate sur le bruit réel du split `dev` RSNA (étape 3), où le mode conservateur d'`improved_v1` réduit les faux positifs en s'abstenant sur les cas ambigus.
