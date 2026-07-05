# Fine-tuning QLoRA de MedGemma sur RSNA

> Prototype pédagogique. Non destiné au diagnostic. Le fine-tuning ne change pas ce statut.

MedGemma-4b est adapté par **QLoRA** (4-bit + LoRA) pour produire le JSON du projet
avec la bonne classe. L'entraînement se fait **en cloud** (Kaggle T4 16 Go) car une
GTX 1660 (6 Go, Turing, pas de bf16 natif) est trop juste pour entraîner un modèle
4B multimodal.

## Fichiers

- `build_training_data.py`: génère les paires image→JSON (`data/train.jsonl`,
  `data/val.jsonl`) depuis le RSNA local, pour inspection. La même logique de
  cibles est reprise dans le notebook.
- `medgemma_qlora_kaggle.ipynb`: **notebook d'entraînement** à lancer sur Kaggle.
- `adapter/`: (à créer) où tu décompresses l'adaptateur entraîné.

## Étapes

### 1. Inspecter les données (local, optionnel)
```bash
python finetuning/build_training_data.py --per-class 200 --val-frac 0.15
```

### 2. Entraîner sur Kaggle
1. Sur https://kaggle.com → **Create → Notebook**, puis **File → Import** →
   `finetuning/medgemma_qlora_kaggle.ipynb`.
2. Panneau de droite :
   - **Accelerator : GPU T4 x1** ; **Internet : On**.
   - **Add Input** → la compétition *RSNA Pneumonia Detection Challenge* (accepte
     ses règles au préalable).
   - Accepte la licence MedGemma sur https://huggingface.co/google/medgemma-4b-it,
     crée un token HF, et ajoute-le en **secret Kaggle** `HF_TOKEN`.
3. **Run All**. À la fin, télécharge `medgemma_rsna_lora.zip` depuis l'onglet *Output*.

### 3. Évaluer l'adaptateur en local
```bash
# décompresse l'archive dans finetuning/adapter/
$env:MEDGEMMA_ADAPTER = "finetuning/adapter"        # PowerShell
python eval/run_evaluation.py --engine medgemma --mode baseline \
  --cases data/rsna_cases.csv --out-dir eval/results_ft --db-path eval/results_ft/runs.sqlite
python eval/build_performance_charts.py --predictions eval/results_ft/baseline_predictions.csv --out-dir eval/results_ft/charts
```
L'adaptateur placé dans `finetuning/adapter/` est **auto-détecté** : l'app et
l'éval l'utilisent pour la version *improved* sans variable à définir (les sorties
affichent alors `model_name: google/medgemma-4b-it+lora`). La variable
`MEDGEMMA_ADAPTER` reste utile pour pointer un adaptateur ailleurs. La version
*baseline* utilise toujours le modèle de base (adaptateur désactivé à la volée).
Compare `eval/results_ft` au zero-shot pour chiffrer le gain.

## Cibles d'entraînement

Chaque exemple = (image, prompt) → JSON de référence dérivé du **label RSNA**
(`Normal → normal`, `Lung Opacity → suspected_opacity`). Les justifications sont
gabarits, fondées sur le visible, sans invention d'historique clinique (elles
respectent les garde-fous). C'est une distillation du label dans le schéma, pas
une annotation d'expert : à documenter comme tel dans le rapport.
