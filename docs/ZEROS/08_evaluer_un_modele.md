# 08 — Évaluer un modèle : comment savoir si ça marche vraiment

## Le piège de l'accuracy

La métrique la plus intuitive pour évaluer un modèle, c'est l'**accuracy** (taux de bonnes réponses) :

```
  Accuracy = nombre de bonnes prédictions / total des prédictions
```

Exemple : "Notre modèle a 95% d'accuracy !"

Ça a l'air bien. Mais voilà le piège.

### Le problème du déséquilibre de classes

Imagine un dataset de 1000 images médicales :
- 950 images "normal"
- 50 images "pneumonie"

Un modèle qui dit **toujours "normal"** sans jamais analyser l'image obtient :

```
  Accuracy = 950 / 1000 = 95%
```

95% d'accuracy, et pourtant il manque 100% des pneumonies. En médecine, ce modèle serait catastrophique.

---

## La matrice de confusion : voir les erreurs en détail

Avant de comprendre le F1-score, il faut comprendre la matrice de confusion.

Pour un problème binaire (normal vs. pneumonie) :

```
                        PRÉDIT
                   Normal  |  Pneumonie
           ────────────────┼────────────────
  RÉEL    Normal  │   TN   │      FP        │
           ────────────────┼────────────────
          Pneumon │   FN   │      TP        │
           ────────────────┴────────────────

  TP (True Positive)   = Pneumonie prédite, c'était vraiment une pneumonie  ✓
  TN (True Negative)   = Normal prédit, c'était vraiment normal             ✓
  FP (False Positive)  = Pneumonie prédite, c'était en fait normal          ✗
  FN (False Negative)  = Normal prédit, c'était en fait une pneumonie       ✗
```

Le FN est le cas le plus dangereux en médecine : le modèle dit "normal" alors que le patient a une pneumonie.

---

## Précision et Rappel

Ces deux métriques complètent l'accuracy :

```
  Précision = TP / (TP + FP)
  ────────────────────────────
  "Parmi toutes les fois où j'ai dit pneumonie,
   combien de fois avais-je raison ?"
  → Mesure les fausses alarmes

  Rappel = TP / (TP + FN)
  ────────────────────────────
  "Parmi toutes les vraies pneumonies,
   combien est-ce que j'en ai détectées ?"
  → Mesure les cas manqués
```

### Exemple chiffré

Modèle analysant 100 images (30 avec pneumonie, 70 normales) :

```
  Matrice de confusion :
  ┌─────────────────────────────┐
  │              Prédit         │
  │          Normal│Pneumonie   │
  │ ───────────────┼─────────── │
  │ Réel Normal  60│    10      │  ← 10 fausses alarmes (FP)
  │      ──────────┼─────────── │
  │      Pneumonie  5│   25     │  ← 5 cas manqués (FN) ← DANGEREUX
  └─────────────────────────────┘

  Accuracy  = (60 + 25) / 100 = 85%
  Précision = 25 / (25 + 10)  = 71%
  Rappel    = 25 / (25 + 5)   = 83%
```

Le rappel de 83% signifie qu'on a manqué 17% des pneumonies. Pour une maladie mortelle si non traitée, c'est inacceptable. Il faudrait viser 95%+.

---

## Le F1-score : une métrique équilibrée

Le F1-score combine précision et rappel en une seule valeur :

```
  F1 = 2 × (Précision × Rappel) / (Précision + Rappel)
```

Avec notre exemple :
```
  F1 = 2 × (0.71 × 0.83) / (0.71 + 0.83)
     = 2 × 0.59 / 1.54
     = 0.766
     ≈ 76%
```

Le F1 "punit" les modèles qui sacrifient l'une au profit de l'autre. Un modèle qui dit tout le temps "pneumonie" aurait un rappel de 100% mais une précision catastrophique, et donc un F1 médiocre.

### Macro F1 vs. Weighted F1

Ce projet utilise le **macro F1** :

```
  Macro F1 = moyenne du F1 de chaque classe, sans pondération
  Weighted F1 = moyenne pondérée par le nombre d'exemples par classe

  Pourquoi Macro F1 ?
  ─────────────────────────────────────────────────────────────
  Avec un dataset déséquilibré (95% normal, 5% pneumonie),
  le weighted F1 donne plus d'importance à la classe "normal".
  On peut obtenir un bon weighted F1 en ratant toutes les pneumonies.

  Le macro F1 traite chaque classe équitablement.
  Rater les pneumonies fait autant de mal que rater les cas normaux.
  C'est la métrique adaptée aux problèmes médicaux.
```

---

## Les 5 métriques de ce projet

Ce projet ne se limite pas au F1-score. Il mesure cinq dimensions :

### 1. Macro F1-score (performance clinique)

```
  Objectif : le plus élevé possible
  Ce que ça mesure : est-ce que le modèle distingue pneumonie / normal / uncertain ?
  Cible : > 0.80 en phase finale
```

### 2. Taux de validité JSON (fiabilité du pipeline)

```
  Taux de validité JSON = prédictions avec JSON valide / total des prédictions
  Objectif : ≥ 95%

  Ce que ça mesure : le pipeline fonctionne-t-il de bout en bout ?
  Si ce taux est bas, il y a un bug dans inference.py ou guardrails.py,
  pas nécessairement un problème avec le modèle lui-même.
```

### 3. Taux de warning (conformité réglementaire)

```
  Taux de warning = prédictions avec le champ "warning" / total
  Objectif : 100%

  Ce que ça mesure : respecte-t-on l'obligation légale d'informer l'utilisateur ?
  Ce taux doit être absolument 100%. Si c'est 99%, il y a un bug.
```

### 4. Taux d'incertitude (calibration)

```
  Taux d'incertitude = prédictions "uncertain" / total
  Objectif : 5% à 20%

  Ce que ça mesure : le modèle est-il bien calibré ?
  < 5% : le modèle est sur-confiant, il prétend savoir alors qu'il doute
  > 20% : le modèle est trop prudent, il devient inutile en pratique
  5-20% : zone saine, le modèle exprime ses doutes honnêtement
```

### 5. Latence médiane (utilisabilité)

```
  Latence médiane = temps de traitement du 50e centile
  Objectif : < 5 secondes par image

  Ce que ça mesure : est-ce que le système est utilisable en pratique ?
  Un hôpital qui fait 1000 radios par jour ne peut pas attendre 30 secondes par image.
```

---

## Les 30 cas finaux commentés

L'évaluation finale repose sur **30 cas soigneusement sélectionnés** :

```
  Pourquoi 30 et pas 3000 ?
  ──────────────────────────────────────────────────────────────
  La quantité ne garantit pas la qualité de l'évaluation.
  30 cas parfaitement documentés valent mieux que 3000 cas approximatifs.

  Ces 30 cas sont choisis pour couvrir :
  ├── Cas clairs de pneumonie (10 cas)
  ├── Cas clairement normaux (10 cas)
  └── Cas limites / ambigus (10 cas)  ← les plus révélateurs
```

Chaque cas inclut :
- L'image radio
- Le label ground truth (la vraie réponse)
- La prédiction du modèle
- Si le modèle s'est trompé : une explication de pourquoi

---

## Comparer baseline et MedGemma

La progression en deux phases (voir fichier 04) permet une comparaison propre :

```
  Résultats hypothétiques
  ──────────────────────────────────────────────────────────────────
                      Baseline déterministe    MedGemma 4B
  ────────────────────────────────────────────────────────────────
  Macro F1             0.61                    0.84
  Validité JSON        100%                    97%
  Warning présent      100%                    100%
  Taux uncertain       0% (pas de classe)      12%
  Latence médiane      45 ms                   2100 ms
  ────────────────────────────────────────────────────────────────

  Conclusion : MedGemma apporte +23 points de F1 au prix d'une
  latence 47x plus élevée. Pour ce projet, c'est acceptable.
```

---

## Résumé

```
  Accuracy seule        → trompeuse avec des données déséquilibrées
  Matrice de confusion  → visualise les 4 types d'erreurs
  Précision             → mesure les fausses alarmes
  Rappel                → mesure les cas manqués (le plus critique en médecine)
  F1 macro              → équilibre entre précision et rappel, traite les classes également
  Validité JSON         → mesure la fiabilité technique du pipeline
  Warning               → mesure la conformité réglementaire
  Taux uncertain        → mesure la calibration du modèle
  Latence               → mesure l'utilisabilité pratique
```

> La suite : pourquoi l'éthique et la loi sont au coeur de ce projet. Fichier suivant : `09_ethique_et_loi.md`
