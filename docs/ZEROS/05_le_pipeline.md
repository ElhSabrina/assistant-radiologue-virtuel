# 05 — Le pipeline : comment tout s'assemble

## C'est quoi un pipeline ?

Un pipeline, c'est une chaîne d'étapes où la sortie de chaque étape est l'entrée de la suivante. Comme une chaîne de montage en usine.

Dans ce projet, la chaîne est :

```
  Image radio (.jpg / .png / .dcm)
         │
         ▼
  ┌──────────────────┐
  │  preprocessing   │  ← nettoyage, validation, redimensionnement
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   inference      │  ← le modèle analyse l'image
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   guardrails     │  ← vérification de la sortie
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   database       │  ← sauvegarde dans SQLite
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   API (FastAPI)  │  ← réponse envoyée au client
  └──────────────────┘
         │
         ▼
  JSON structuré  →  Interface (Streamlit / Gradio)
```

Chaque étape est **isolée dans son propre fichier Python**. Si on veut remplacer MedGemma par un autre modèle, on modifie uniquement `inference.py`. Le reste ne change pas.

---

## Étape 1 : preprocessing.py

### Ce que ça fait

Avant de donner l'image au modèle, on la prépare. Comme laver des légumes avant de cuisiner.

**Validations :**
- L'image existe-t-elle vraiment ?
- Est-ce bien une image (pas un PDF, pas un fichier corrompu) ?
- La résolution est-elle suffisante ?

**Transformations :**
- Redimensionnement en 512×512 pixels (taille standard pour ce modèle)
- Normalisation des valeurs de pixels (de 0-255 vers 0-1)
- Conversion en niveaux de gris si l'image est en couleur

**Flag qualité :**
Si l'image est trop petite, trop floue ou trop sombre, on lève un avertissement. On ne bloque pas, mais on note le problème dans le JSON de sortie.

```python
# Exemple simplifié de ce que fait preprocessing.py
from PIL import Image
import numpy as np

def preprocess(image_path: str):
    img = Image.open(image_path).convert("L")  # niveaux de gris
    img = img.resize((512, 512))               # redimensionnement
    array = np.array(img) / 255.0              # normalisation 0-1

    quality_flag = None
    if array.mean() < 0.05:                    # image trop sombre
        quality_flag = "image_trop_sombre"

    return array, quality_flag
```

---

## Étape 2 : inference.py

### Ce que ça fait

C'est ici que le modèle entre en jeu. On lui envoie l'image préparée et on lui pose une question via un **prompt**.

### C'est quoi un prompt ?

Un prompt, c'est l'instruction qu'on donne au modèle. Avec un VLM comme MedGemma, on peut lui parler en langage naturel.

```python
# Prompt basique (baseline_v1)
prompt = """
Analyse cette radiographie thoracique.
Réponds uniquement en JSON avec le format suivant :
{
    "prediction": "normal" | "pneumonia" | "uncertain",
    "confidence": 0.0 à 1.0,
    "explanation": "description courte"
}
"""

# Prompt amélioré (improved_v1) - avec chain-of-thought
prompt = """
Analyse cette radiographie thoracique en suivant ces étapes :
1. Décris les structures anatomiques visibles (côtes, diaphragme, cœur)
2. Identifie toute zone anormale (opacité, consolidation, épanchement)
3. Localise précisément la zone suspecte si elle existe
4. Formule ta conclusion

Réponds uniquement en JSON avec le format suivant :
{
    "prediction": "normal" | "pneumonia" | "uncertain",
    "confidence": 0.0 à 1.0,
    "zone": "lobe inférieur droit" | "lobe inférieur gauche" | null,
    "explanation": "description clinique",
    "warning": "message d'avertissement obligatoire"
}
"""
```

### Pourquoi deux versions de prompt ?

Le projet compare deux approches :

```
  baseline_v1                         improved_v1
  ──────────────────────              ──────────────────────────────────
  Prompt court, direct                Prompt avec chain-of-thought
  "Analyse et réponds"                "Décris d'abord, puis conclus"

  Avantage : rapide                   Avantage : plus précis
  Inconvénient : moins détaillé       Inconvénient : légèrement plus lent
```

Le chain-of-thought (raisonnement en chaîne), c'est forcer le modèle à "penser à voix haute" avant de conclure. Comme un étudiant en médecine qu'on forme à décrire systématiquement ce qu'il voit avant de poser un diagnostic.

---

## Étape 3 : guardrails.py

### Le problème que ça résout

Un modèle de langage peut produire n'importe quoi. Même si on lui demande du JSON, il peut :
- Écrire du texte en dehors du JSON
- Oublier un champ obligatoire
- Mettre une valeur invalide ("très probable" au lieu de 0.87)
- Inventer une pathologie qui n'est pas dans notre liste de classes

### Ce que font les guardrails

```
  Sortie brute du modèle
  ─────────────────────────────────────────────────────────
  "D'après mon analyse, il semble que... {
      "prediction": "probable pneumonia",   ← valeur invalide
      "confidence": "high",                 ← devrait être un float
      "explanation": "..."
  }"

  Après guardrails
  ─────────────────────────────────────────────────────────
  {
      "prediction": "uncertain",            ← downgrade automatique
      "confidence": null,
      "explanation": "...",
      "warning": "Sortie non conforme au schéma. Vérification humaine requise.",
      "guardrail_triggered": true
  }
```

Les guardrails appliquent trois règles :

1. **Validation JSON** : le JSON est-il valide ? Tous les champs obligatoires sont-ils présents ?
2. **Validation des valeurs** : `prediction` doit être `"normal"`, `"pneumonia"` ou `"uncertain"`. `confidence` doit être un float entre 0 et 1.
3. **Downgrade automatique** : si une règle est violée, on passe la prédiction à `"uncertain"` plutôt que de renvoyer une valeur fausse.

### Le warning obligatoire

Chaque réponse, qu'elle soit correcte ou non, contient un champ `warning` :

```json
"warning": "Outil d'aide au diagnostic uniquement. L'interprétation finale doit être réalisée par un médecin qualifié. Ce résultat ne constitue pas un diagnostic médical."
```

Ce n'est pas optionnel. C'est une exigence de l'AI Act européen pour les systèmes d'aide au diagnostic.

---

## Étape 4 : database.py

### Pourquoi enregistrer chaque prédiction ?

La traçabilité est obligatoire pour tout système médical. Si le modèle se trompe sur 100 patients d'affilée, il faut pouvoir le détecter, l'isoler et comprendre pourquoi.

### Structure SQLite

```sql
CREATE TABLE predictions (
    run_id          TEXT PRIMARY KEY,    -- identifiant unique de la prédiction
    image_path      TEXT NOT NULL,       -- chemin de l'image analysée
    prediction      TEXT NOT NULL,       -- "normal", "pneumonia", "uncertain"
    confidence      REAL,               -- 0.0 à 1.0
    explanation     TEXT,               -- explication générée
    prompt_version  TEXT NOT NULL,      -- "baseline_v1" ou "improved_v1"
    guardrail_ok    INTEGER NOT NULL,   -- 1 si JSON valide, 0 sinon
    warning_present INTEGER NOT NULL,   -- 1 si warning inclus, 0 sinon
    latency_ms      INTEGER,            -- temps de traitement en ms
    timestamp       TEXT NOT NULL       -- horodatage ISO 8601
);
```

SQLite est un fichier unique (`.db`) qui voyage avec le dépôt Git. Pas de serveur de base de données à configurer. N'importe quel client SQL (DBeaver, TablePlus, ou même Python) peut l'ouvrir directement.

---

## Étape 5 : l'API FastAPI

C'est la porte d'entrée du système. C'est ce que le monde extérieur voit.

```
  Client (Streamlit, Gradio, script Python, Postman...)
         │
         │  POST /predict
         │  Body: { "image": <fichier binaire> }
         │
         ▼
  ┌──────────────────────────────────────┐
  │           FastAPI Server             │
  │  - Valide l'entrée (Pydantic)        │
  │  - Appelle preprocessing.py          │
  │  - Appelle inference.py              │
  │  - Appelle guardrails.py             │
  │  - Appelle database.py               │
  │  - Renvoie le JSON structuré         │
  └──────────────────────────────────────┘
         │
         ▼
  Réponse JSON au client
```

FastAPI génère automatiquement une **documentation interactive** accessible à `http://localhost:8000/docs`. Tu peux y tester l'API directement depuis ton navigateur, sans écrire une seule ligne de code.

---

## Les interfaces utilisateur

Deux interfaces sont développées en parallèle :

**Streamlit** (interface principale) :
- Uploader une image depuis son ordinateur
- Voir le résultat avec l'image annotée et le JSON
- Simple, rapide à développer, idéal pour la démo

**Gradio** (interface alternative) :
- Compatible avec Hugging Face Spaces (hébergement gratuit en ligne)
- Même fonctionnalité que Streamlit
- Utile pour partager facilement la démo à des personnes extérieures

---

## Résumé visuel complet

```
  ┌─────────────────────────────────────────────────────────────┐
  │                     PIPELINE COMPLET                        │
  │                                                             │
  │  [Image]                                                    │
  │     │                                                       │
  │     ▼  preprocessing.py                                     │
  │  [512×512, normalisée, flag qualité]                        │
  │     │                                                       │
  │     ▼  inference.py (prompt + MedGemma)                     │
  │  [JSON brut du modèle]                                      │
  │     │                                                       │
  │     ▼  guardrails.py                                        │
  │  [JSON validé ou downgrade + warning]                       │
  │     │                                                       │
  │     ├──► database.py → SQLite (audit trail)                 │
  │     │                                                       │
  │     ▼  FastAPI                                              │
  │  [Réponse HTTP 200 avec JSON]                               │
  │     │                                                       │
  │     ├──► Streamlit (interface web)                          │
  │     └──► Gradio (alternative HF Spaces)                     │
  └─────────────────────────────────────────────────────────────┘
```

> La suite : comment appeler cette API depuis Python. Fichier suivant : `06_api_et_python.md`
