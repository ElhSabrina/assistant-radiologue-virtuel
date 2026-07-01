# Mini-rapport — Assistant radiologue virtuel (MedGemma + RSNA)

> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.

## 1. Périmètre

Chaîne complète et prudente autour d'une radiographie thoracique frontale :
une image entre, une sortie JSON structurée sort, elle est validée par des
garde-fous, journalisée en SQLite, puis évaluée. Trois classes de sortie :
`normal`, `suspected_opacity`, `uncertain`. La classe `uncertain` est un
garde-fou (savoir ne pas conclure), pas un échec.

Le projet **ne vise pas le diagnostic** : il démontre une démarche d'ingénierie
responsable (baseline, amélioration mesurée, métriques, logs, analyse d'erreurs,
limites).

## 2. Données — RSNA Pneumonia Detection Challenge

- **Source :** Kaggle, `rsna-pneumonia-detection-challenge` (R5/R7 de l'appel d'offre).
- **Licence / accès :** données sous conditions de la compétition Kaggle,
  **non redistribuables**. Le dépôt ne contient aucun DICOM ni aucune donnée
  patient ; les images restent locales (`data/rsna_pneumonia/`, gitignoré).
- **Vérité terrain :** colonne `class` des métadonnées RSNA —
  `Normal → normal`, `Lung Opacity → suspected_opacity`. La classe ambiguë
  « No Lung Opacity / Not Normal » est écartée de la vérité terrain.
- **Échantillon :** 30 cas équilibrés (15 `normal` / 15 `suspected_opacity`),
  images PNG 512×512, sélection déterministe (seed 42). Décrit dans
  `data/rsna_cases.csv` (identifiants dé-identifiés uniquement, aucune donnée
  nominative ni démographique conservée).

## 3. Modèle

- **MedGemma-4b-it** (`google/medgemma-4b-it`, R4), modèle image-texte médical de
  Google, exécuté via `transformers`.
- **Quantification :** 4-bit NF4 (bitsandbytes) pour tenir sur une GTX 1660 (6 Go).
- **Précision de calcul :** `bfloat16`. Le `float16` produit des NaN et une sortie
  vide sur les GPU Turing GTX 16xx (sans tensor cores) ; `bfloat16` corrige ce
  comportement.
- **Génération :** déterministe (`do_sample=False`).
- **Conditions d'usage :** MedGemma est soumis aux Health AI Developer Foundations
  Terms of Use de Google ; usage strictement expérimental et non clinique.

## 4. Baseline vs amélioration

Même modèle, même jeu d'images ; seule la stratégie de prompt/garde-fou change.

| Version | Prompt | Règle d'incertitude |
|---|---|---|
| **Baseline** | `prompts/baseline_prompt.txt` (question directe 3 classes) | aucune |
| **Improved** | `prompts/improved_prompt.txt` (contrôle explicite projection/rotation/exposition, refus de diagnostic) | bascule vers `uncertain` si `confidence < 0.60` |

Garde-fous communs (`src/guardrails.py`) : validation du schéma JSON, warning
obligatoire réécrit, bascule `uncertain` si JSON invalide, bascule `uncertain` si
qualité image faible et confiance < 0.6, détection d'hallucination (invention de
contexte clinique).

## 5. Résultats

MedGemma-4b-it, 30 cas RSNA (15 `normal` / 15 `suspected_opacity`), génération
déterministe. Tableau et graphique dans `eval/results/`
(`before_after_summary.csv`, `dashboard.png`, `dashboard.md`).

| Métrique | baseline | improved | Δ (imp − base) |
|---|---|---|---|
| Accuracy | 0.933 | 0.833 | −0.100 |
| Macro-F1 (3 classes) | 0.644 | 0.579 | −0.064 |
| Sensibilité (opacité) | 0.933 | 0.667 | −0.267 |
| Spécificité (normal) | 0.933 | **1.000** | +0.067 |
| JSON valide | 1.000 | 1.000 | 0.000 |
| Warning présent | 1.000 | 1.000 | 0.000 |
| Taux incertain | 0.067 | 0.100 | +0.033 |
| Taux hallucination | 0.000 | 0.000 | 0.000 |
| Latence moyenne | ~294 s | ~57 s | −237 s |

**Lecture.** Le prompt renforcé + la règle « confiance < 0,60 → `uncertain` »
rendent le système nettement plus **prudent sur les images normales** : la
spécificité passe à **1,0** (zéro faux positif sur les 15 normaux) et le taux
d'abstention augmente. Le prix est une **chute de sensibilité** (0,93 → 0,67) :
5 opacités sur 15 ne sont plus appelées `suspected_opacity` (3 basculées en
`uncertain`, 2 en faux négatifs). Aucune hallucination détectée dans les deux
versions, JSON valide à 100 %. La version *improved* est aussi ~5× plus rapide
car ses réponses sont plus courtes et mieux cadrées.

**Décision.** Pour ce prototype, la baseline offre le meilleur équilibre
sensibilité/spécificité. L'*improved* illustre un arbitrage explicite en faveur de
la prudence : à conserver seulement si le coût d'un faux positif domine celui d'un
faux négatif — ce qui, en dépistage d'opacité, est rarement le cas. C'est
précisément le type de compromis à défendre en soutenance plutôt qu'à masquer.

> Nuance métrique : la macro-F1 est calculée sur 3 classes alors que la vérité
> terrain n'en contient que 2 ; la classe `uncertain` (support nul) y contribue
> pour 0 et tire la macro-F1 vers le bas. Accuracy, sensibilité et spécificité
> sont ici plus interprétables.

### 5.1 Diagnostics de performance sur un plus grand échantillon

Pour la présentation technique, MedGemma (prompt baseline, zero-shot) a été évalué
sur un échantillon plus large de **246 cas RSNA** (117 normal / 129 opacité).
MedGemma n'étant **pas entraîné** (utilisé par prompting), il n'y a pas de courbe
train/validation ; les graphiques sont les diagnostics d'évaluation classiques,
générés par `eval/build_performance_charts.py` (`eval/perf/charts/`) et repris dans
le notebook `notebooks/04_model_performance.ipynb` :

- **matrice de confusion**, **précision/rappel/F1 par classe** ;
- **distribution de confiance** (correct vs incorrect) ;
- **courbe risque-couverture** (base de la règle `uncertain`) ;
- **diagramme de fiabilité** (calibration) ; **distribution de latence**.

Résultats (246 cas) : accuracy 0.92, sensibilité 0.89, spécificité 0.96,
taux d'incertitude 5.7 %, latence médiane ~65 s/image (GTX 1660, 4-bit). Ces
chiffres confirment, sur un échantillon 8× plus grand, la lecture faite sur les
30 cas. La calibration montre une confiance globalement optimiste (à surveiller).

## 5.2 Fine-tuning QLoRA (extension Could)

Extension au niveau *Could* : adaptation de MedGemma-4b par **QLoRA** (4-bit + LoRA)
pour produire directement le JSON du projet avec la bonne classe.

- **Où :** entraînement **en cloud (Kaggle T4 16 Go)**, car la GTX 1660 locale
  (6 Go, Turing, pas de bf16 natif, fp16 instable en génération) ne peut pas
  entraîner un modèle 4B multimodal. Notebook : `finetuning/medgemma_qlora_kaggle.ipynb`.
- **Données :** paires image→JSON générées depuis RSNA
  (`finetuning/build_training_data.py`), la cible étant le label RSNA distillé dans
  le schéma (justifications gabarits, fondées sur le visible, sans invention
  d'historique). Limite assumée : ce n'est pas une annotation d'expert.
- **Méthode :** LoRA (r=16) sur les projections d'attention/MLP, tour de vision
  gelée, QLoRA 4-bit, 2 epochs, `paged_adamw_8bit`.
- **Évaluation :** l'adaptateur se branche en local via la variable
  `MEDGEMMA_ADAPTER` et se rejoue avec **le même harnais** que la baseline
  (`eval/run_evaluation.py`), sur les **mêmes 30 cas** que le zero-shot.

### 5.3 Résultats baseline vs improved

**a) Résultat préliminaire (30 cas, non retenu comme référence).** Un premier test
donnait un fine-tuning à 100 % (accuracy/sensibilité/spécificité). Mais l'échantillon
était petit et **1 des 30 cas figurait dans l'entraînement** : score optimiste, à ne
pas présenter comme définitif.

**b) Résultat de référence — 80 cas tenus à part, ZÉRO fuite.** Jeu construit par
`eval/build_heldout_cases.py` en excluant les 400 patientId d'entraînement (40 normal
/ 40 opacité). Comparaison `baseline` (MedGemma de base) vs `improved` (fine-tuné LoRA
+ prompt renforcé). Graphiques : `eval/results_heldout/`.

| Métrique | baseline (base) | improved (fine-tuné) |
|---|---|---|
| Accuracy | 0.875 | 0.875 |
| Macro-F1 | 0.606 | 0.582 |
| Sensibilité (opacité) | **0.875** | 0.750 |
| Spécificité (normal) | 0.875 | **1.000** |
| JSON valide | 1.000 | 1.000 |
| Taux incertain | 0.075 | 0.000 |
| Taux hallucination | 0.000 | 0.000 |

Répartition des erreurs (registres `eval/results_heldout/error_register_*.md`) :

| | TP | TN | FP | FN | incertain |
|---|---|---|---|---|---|
| baseline | 35 | 35 | 2 | **2** | 6 |
| improved | 30 | 40 | **0** | **10** | 0 |

**Lecture honnête — le fine-tuning n'améliore PAS l'accuracy sur données propres.**
Les deux versions sont à **0,875**. Le *improved* déplace seulement le point de
fonctionnement : **spécificité parfaite** (0 faux positif sur les normaux) au prix
d'une **sensibilité plus basse** (0,75) et surtout de **10 faux négatifs** contre 2
pour la baseline. En dépistage, le faux négatif (opacité manquée) est l'erreur la
plus dangereuse : de ce point de vue, l'*improved* est **plus risqué**, pas meilleur.

**Nuances à défendre :**

- **Perte du garde-fou `uncertain`.** L'*improved* n'abstient jamais (0 %) car les
  cibles d'entraînement ne contenaient pas `uncertain` : régression de prudence.
- **Confiance non calibrée** (reproduit les valeurs des gabarits d'entraînement).
- **Le 100 % initial était un artefact** de petit échantillon + fuite ; c'est
  précisément pour ça qu'on a refait l'évaluation sur un jeu tenu à part.
- **Échantillon encore modeste** (80 cas) : intervalles de confiance larges.

**Configuration de l'application.** L'app propose trois choix : *Toy* (règles
instantanées, pour la démo), *MedGemma baseline* (modèle de base + prompt de base)
et *MedGemma improved* (**fine-tuné LoRA + prompt renforcé** + règle d'incertitude).
En interne, un seul modèle est chargé et l'adaptateur LoRA est activé/désactivé
selon le choix (`src/medgemma.py`). L'*improved* n'est actif que si l'adaptateur
est présent (`MEDGEMMA_ADAPTER`) ; sinon il retombe sur le modèle de base + prompt
renforcé.

Le statut **non clinique** est inchangé par le fine-tuning.

## 6. Analyse d'erreurs

Registre commenté généré par `eval/error_register.py`
(`eval/error_register.csv` + `.md` pour *improved*, variantes `_baseline` pour la
baseline). Taxonomie : `TP/TN` (corrects), `FP` (normal sur-appelé),
`FN` (opacité manquée), `UA` (incertitude acceptable), `HALLU` (justification non
fondée), `FORMAT` (JSON invalide). Les faux négatifs sont en sévérité haute.

Répartition observée :

| Type | baseline | improved |
|---|---|---|
| TP (opacité correcte) | 14 | 10 |
| TN (normal correct) | 14 | 15 |
| UA (incertitude acceptable) | 2 | 3 |
| FN (opacité manquée) | 0 | 2 |
| FP / HALLU / FORMAT | 0 | 0 |

Cas notables (*improved*) :

- **RSNA_018** (FN, confiance 0,95) : opacité appelée `normal` avec forte
  confiance — l'erreur la plus dangereuse ici, une confiance élevée mal placée
  que la calibration ne rattrape pas.
- **RSNA_010 / 014 / 021** (UA) : le modèle signale une qualité limitée (ligne
  centrale, trachéotomie, cliché portable AP) et s'abstient — abstention légitime,
  et les dispositifs mentionnés sont réellement visibles (pas d'hallucination).

## 7. Limites

- Échantillon de 30 cas : indicatif, non représentatif d'une performance clinique.
- Confiance renvoyée par le modèle non calibrée.
- Forte sensibilité au prompt et au modèle.
- Pas de localisation ni de contexte clinique ; lecture d'une seule vue frontale.
- Aucune validation médicale ; toute sortie exige une vérification par un
  professionnel qualifié.

## 8. Reproduire

```bash
# Toutes les commandes se lancent depuis la racine du dépôt
# (le dossier qui contient README.md, src/, app/, eval/).
cd assistant-radiologue-virtuel-main

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124  # build CUDA

# 1. Jeu de cas RSNA (utilise l'export local data/rsna_pneumonia/ s'il est présent,
#    sinon télécharge via l'API Kaggle)
python data/prepare_rsna.py --per-class 15

# 2. Évaluation baseline vs improved avec MedGemma
python eval/run_evaluation.py --engine medgemma --mode toy \
  --cases data/rsna_cases.csv \
  --out-dir eval/results --db-path eval/results/runs.sqlite

# 3. Dashboard + registre d'erreurs
python eval/build_dashboard.py
python eval/error_register.py --predictions eval/results/improved_predictions.csv

# 4. Démo web
streamlit run app/streamlit_app.py   # sélectionner le moteur "medgemma"
```
