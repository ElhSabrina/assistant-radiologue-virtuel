# 03 — Les données : pourquoi c'est (presque) tout

## La règle d'or du machine learning

> "Garbage in, garbage out."

Si tu entraînes un modèle sur de mauvaises données, tu obtiens un mauvais modèle. Peu importe la qualité de ton code, peu importe la puissance de ton GPU. Les données, c'est le fondement.

---

## C'est quoi un dataset ?

Un dataset, c'est une collection organisée de données. Pour ce projet :

```
  dataset/
  ├── images/
  │   ├── patient_001.jpg     ← la radio
  │   ├── patient_002.jpg
  │   └── ...
  └── labels.csv              ← ce que montre chaque radio

  labels.csv :
  ┌────────────────┬───────────┬──────────────────────────┐
  │ image          │ label     │ bounding_box             │
  ├────────────────┼───────────┼──────────────────────────┤
  │ patient_001    │ pneumonia │ x=120, y=200, w=80, h=60 │
  │ patient_002    │ normal    │ (aucune)                  │
  │ patient_003    │ pneumonia │ x=300, y=350, w=100, h=90│
  └────────────────┴───────────┴──────────────────────────┘
```

Chaque ligne du CSV associe une image à ce qu'elle contient. C'est ce qu'on appelle une **annotation**.

---

## C'est quoi une bounding box ?

Une bounding box (boîte englobante), c'est un rectangle tracé autour de la zone anormale dans l'image.

```
  Image radio vue de face (512×512 pixels)
  ─────────────────────────────────────────

  (0,0)─────────────────────────────►x
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │  ░░░░░░░░░┌──────┐░░░░░░░░░░░   ← bounding box
    │  ░░░░░░░░░│OPAC. │░░░░░░░░░░░     x=300, y=200
    │  ░░░░░░░░░└──────┘░░░░░░░░░░░     w=80,  h=50
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    ▼y

  Paramètres :
  x = coin gauche de la boîte (en pixels depuis le bord gauche)
  y = coin haut de la boîte (en pixels depuis le bord haut)
  w = largeur de la boîte
  h = hauteur de la boîte
```

Grâce aux bounding boxes, le modèle apprend non seulement que "cette radio montre une pneumonie", mais aussi "elle est localisée ici". Ça rend l'apprentissage beaucoup plus précis.

---

## Les datasets disponibles

Il existe quatre datasets de référence en radiologie thoracique :

| Dataset | Volume | Ce qu'il contient | Difficulté |
|---|:---:|---|:---:|
| **RSNA Pneumonia** | ~30 000 | radios + bboxes + labels | accessible |
| CheXpert | 224 316 | radios + rapports textuels | intermédiaire |
| MIMIC-CXR | 377 110 | radios + rapports + données cliniques | avancé |
| NIH ChestXray | ~112 000 | radios + labels larges | intermédiaire |

---

## Pourquoi on a choisi RSNA Pneumonia

### 1. Périmètre maîtrisable

30 000 images, c'est beaucoup moins que les 377 000 de MIMIC-CXR. C'est une qualité : on peut télécharger, vérifier, tester et déboguer sans passer des semaines à gérer des téraoctets de données.

### 2. Annotations précises (bboxes + labels)

RSNA fournit des **bounding boxes** annotées par de vrais radiologues. Les autres datasets donnent souvent juste un label global ("pneumonie: oui/non") sans dire où dans l'image. Avec RSNA, le modèle a plus d'information pour apprendre.

### 3. C'est la voie recommandée

Dans la communauté de recherche en IA médicale, RSNA Pneumonia est le point d'entrée standard pour apprendre à travailler sur des radios thoraciques. La documentation est excellente, les prétraitements sont bien documentés, les benchmarks existent.

Commencer par RSNA, c'est comme apprendre à nager en piscine avant d'aller en mer ouverte.

### 4. Reproductibilité

Pour un projet académique ou de recherche, il faut que quelqu'un d'autre puisse reproduire exactement tes résultats. RSNA est public, accessible via Kaggle, avec des splits train/val/test standardisés.

---

## Et les autres datasets ?

Ils ne sont pas abandonnés. Ils feront partie d'une section dédiée aux **améliorations futures** du projet.

```
  Phase 1 (maintenant)       Phase future
  ──────────────────────      ──────────────────────────────────
  RSNA Pneumonia              MIMIC-CXR + CheXpert
  ~30 000 images              400 000+ images avec rapports textuels
  Pipeline validé             Modèle plus puissant, BDD plus riche
  Résultats reproductibles    Généralisation à d'autres pathologies
```

L'idée : construire un système solide sur une base maîtrisable, puis monter en puissance une fois le pipeline validé.

---

## Comment on découpe les données ?

On ne donne jamais toutes les données au modèle pendant l'entraînement. On les divise en trois groupes :

```
  Dataset RSNA (~30 000 images)
  ─────────────────────────────
  ┌──────────────────────────────────────────────────────┐
  │  Train (70%)       │  Val (15%)   │  Test (15%)      │
  │  ~21 000 images    │  ~4 500 img  │  ~4 500 images   │
  │                    │              │                   │
  │  Le modèle         │  On ajuste   │  On mesure les    │
  │  apprend dessus    │  les réglages│  vraies perfs     │
  └──────────────────────────────────────────────────────┘
```

Pourquoi ne pas tout utiliser pour l'entraînement ? Parce que sinon on ne peut pas savoir si le modèle a vraiment **appris** ou s'il a juste **mémorisé**. Un modèle qui mémorise aura 99% de précision sur les données d'entraînement et 50% sur de nouvelles données. C'est ce qu'on appelle le **surapprentissage** (overfitting).

---

## Les phases de développement dans ce projet

Pour ne pas gaspiller du temps GPU sur un pipeline cassé, on utilise une progression par étapes :

```
  Étape 1 : Smoke test (20 images)
  ─────────────────────────────────
  20 images choisies au hasard. On vérifie que le code tourne
  de bout en bout sans crash. Durée : quelques minutes.

  Étape 2 : Développement (100-150 images)
  ─────────────────────────────────────────
  Un sous-ensemble plus représentatif. On affine les paramètres,
  on corrige les bugs. Durée : quelques heures.

  Étape 3 : Évaluation finale (30 cas commentés)
  ──────────────────────────────────────────────
  30 cas soigneusement sélectionnés et annotés en détail.
  C'est sur eux qu'on mesure les vraies performances finales.
  Chaque cas est documenté et traçable.
```

Pourquoi 30 cas finaux seulement ? Parce que pour une évaluation de qualité, il vaut mieux 30 cas parfaitement documentés que 3000 cas approximatifs.

---

## La traçabilité via CSV + SQLite

Chaque image analysée est tracée. Concrètement :

```python
# Chaque prédiction est enregistrée dans SQLite
{
    "run_id": "a3f9c2d1",
    "image_path": "rsna/patient_001.jpg",
    "label_reel": "pneumonie",
    "prediction_modele": "pneumonie",
    "confiance": 0.87,
    "timestamp": "2026-06-18T14:23:00",
    "prompt_version": "improved_v1"
}
```

Pourquoi c'est important ? Si le modèle se trompe systématiquement sur un type d'images, on peut le retrouver. C'est la **traçabilité** : condition obligatoire pour tout système médical responsable.

---

## Résumé

- Un dataset = des images + des labels (qui dit quoi montre quoi)
- RSNA Pneumonia est choisi car il est précis, accessible et bien documenté
- On divise les données en train/val/test pour éviter le surapprentissage
- Les autres datasets (MIMIC-CXR, CheXpert) serviront dans les améliorations futures
- Chaque prédiction est tracée dans une base de données

> La suite : le modèle MedGemma en détail. Fichier suivant : `04_le_modele_MedGemma.md`
