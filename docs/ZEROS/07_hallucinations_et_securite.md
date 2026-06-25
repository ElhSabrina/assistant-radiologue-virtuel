# 07 — Hallucinations et sécurité

## Le problème des hallucinations

Un modèle de langage peut **inventer des informations** avec une assurance totale.

Ce phénomène s'appelle une **hallucination**. Ce n'est pas un bug au sens classique du terme. C'est une propriété fondamentale des modèles génératifs : ils sont entraînés à produire du texte vraisemblable, pas du texte vrai.

### Exemple concret (hors médecine)

Demande à un LLM : "Cite-moi trois articles scientifiques sur les VLMs médicaux."

Il peut te sortir :

```
1. "Smith et al. (2022) - Medical Vision-Language Models for Pneumonia Detection.
   Journal of Artificial Intelligence in Medicine, vol. 14, pp. 234-251."

2. "Chen & Park (2023) - Multimodal Transformers in Clinical Radiology.
   Nature Machine Intelligence, vol. 5, pp. 112-128."

3. ...
```

Ces articles semblent parfaitement crédibles. Auteurs, journal, volume, pages. Tout a l'air vrai. Et pourtant, il est très probable qu'aucun de ces articles n'existe. Le modèle a "imaginé" des références plausibles.

---

## Pourquoi c'est particulièrement dangereux en médecine

Dans un contexte de divertissement, une hallucination est gênante. Dans un contexte médical, elle peut tuer.

```
  Hallucination médicale possible
  ────────────────────────────────

  Image radio avec une légère anomalie bénigne
         │
         ▼
  Modèle sans garde-fous
         │
         ▼
  Réponse : "Opacité dense bilatérale avec épanchement pleural massif.
             Suspicion de tuberculose active. Isolement immédiat requis."
         │
         ▼
  Médecin sans formation IA fait confiance au système
         │
         ▼
  Patient isolé inutilement, traitement antibiotique lourd prescrit à tort
```

Le modèle n'a pas "menti". Il a produit du texte médicalement cohérent, grammaticalement correct, et complètement faux.

---

## Comment les hallucinations se produisent

Un VLM comme MedGemma a deux modes de "raisonnement" :

```
  Mode 1 : Reconnaissance pattern
  ─────────────────────────────────
  L'image contient des features que le modèle reconnaît.
  Il produit une réponse basée sur ce qu'il a vraiment "vu".
  → Généralement fiable

  Mode 2 : Complétion probabiliste
  ──────────────────────────────────
  L'image est ambiguë ou hors distribution (type d'image
  jamais vu à l'entraînement). Le modèle "invente" la suite
  logique de ce que le texte serait selon ses données d'entraînement.
  → Source principale d'hallucinations
```

---

## Les trois niveaux de défense dans ce projet

### Niveau 1 : Schéma JSON strict

On impose une structure rigide à la réponse. Si le modèle hallucine hors du schéma, la violation est immédiatement détectable.

```python
# Schéma de sortie valide
SCHEMA_VALIDE = {
    "prediction": ["normal", "pneumonia", "uncertain"],  # liste fermée
    "confidence": float,                                  # 0.0 à 1.0
    "explanation": str,                                   # texte libre
    "warning": str                                        # texte obligatoire
}

# Réponse du modèle : hallucination structurelle
reponse_modele = {
    "prediction": "acute respiratory distress syndrome",  # ← PAS dans la liste
    "confidence": "very high",                            # ← devrait être un float
    "explanation": "...",
    "warning": "..."
}

# Guardrail détecte l'anomalie → downgrade vers "uncertain"
reponse_corrigee = {
    "prediction": "uncertain",
    "confidence": None,
    "explanation": "...",
    "warning": "Sortie non conforme. Vérification humaine requise.",
    "guardrail_triggered": True
}
```

### Niveau 2 : La classe `uncertain`

Dans beaucoup de systèmes d'IA, le modèle est **forcé** de choisir une classe. Il doit dire "normal" ou "pneumonie", même si l'image est ambiguë.

Ce projet ajoute une troisième option : **`uncertain`**.

```
  Sans classe uncertain              Avec classe uncertain
  ──────────────────────             ──────────────────────────────
  Image ambiguë                       Image ambiguë
       │                                   │
       ▼                                   ▼
  Modèle forcé à choisir             Modèle peut dire "je ne sais pas"
       │                                   │
  "pneumonie" (0.55 de confiance)    "uncertain" (confiance trop faible)
       │                                   │
  Médecin fait confiance             Médecin sait qu'il faut vérifier
       │                                   │
  Potentiellement dangereux          Comportement sûr
```

Le taux `uncertain` ne doit être ni nul (le modèle serait sur-confiant) ni trop élevé (il deviendrait inutile). L'objectif est un équilibre : le modèle exprime ses doutes honnêtement.

### Niveau 3 : Le warning obligatoire

Chaque réponse, quelle qu'elle soit, contient un avertissement :

```json
"warning": "Outil d'aide au diagnostic uniquement. L'interprétation finale doit être réalisée par un médecin qualifié. Ce résultat ne constitue pas un diagnostic médical."
```

Ce warning est :
- **Intégré dans la réponse API** (pas affiché séparément dans l'interface)
- **Vérifié par les guardrails** : si le champ `warning` est absent, le guardrail l'ajoute
- **Tracé dans SQLite** : une colonne `warning_present` garantit qu'on peut auditer

---

## Visualisation : bonne réponse vs. réponse avec hallucination

```
  Bonne réponse (guardrail OK)
  ──────────────────────────────────────────────────────────────
  {
    "prediction": "pneumonia",         ✓ valeur autorisée
    "confidence": 0.87,               ✓ float entre 0 et 1
    "uncertain": false,               ✓ booléen
    "zone": "lobe inférieur droit",   ✓ texte libre (OK)
    "explanation": "Opacité dense...",✓ texte libre (OK)
    "warning": "Outil d'aide...",     ✓ présent
    "guardrail_ok": true              ✓
  }

  Réponse avec hallucination structurelle (guardrail déclenché)
  ──────────────────────────────────────────────────────────────
  {
    "prediction": "ARDS",             ✗ pas dans la liste
    "confidence": "élevée",          ✗ pas un float
    "uncertain": false,               ✓ (OK)
    "explanation": "...",             ✓ (OK)
    "warning": "..."                  ✓ (OK)
  }
         │
         ▼ guardrail.py détecte les violations
         │
  {
    "prediction": "uncertain",        → downgrade automatique
    "confidence": null,               → mis à null
    "uncertain": true,                → forcé à true
    "explanation": "...",
    "warning": "Sortie non conforme. Vérification humaine requise.",
    "guardrail_ok": false             → tracé en base
  }
```

---

## Ce que les guardrails ne peuvent PAS détecter

Les guardrails protègent contre les hallucinations **structurelles** (valeurs hors schéma). Ils ne peuvent pas détecter les hallucinations **sémantiques** : une valeur qui respecte le schéma mais est médicalement fausse.

```
  Hallucination sémantique (guardrail ne peut PAS détecter)
  ───────────────────────────────────────────────────────────
  {
    "prediction": "pneumonia",    ← valeur valide
    "confidence": 0.92,           ← float valide
    "uncertain": false,           ← booléen valide
    "explanation": "Opacité dans le lobe supérieur gauche..." ← FAUX
  }
  Tout est syntaxiquement correct. Mais l'explication est inventée.
```

C'est pourquoi la supervision humaine reste **obligatoire**. Les guardrails réduisent le risque, ils ne l'éliminent pas.

---

## Résumé

```
  Hallucination = l'IA invente avec assurance
       │
       ▼ Pourquoi ?
  Les modèles génèrent du texte vraisemblable, pas nécessairement vrai
       │
       ▼ Défenses dans ce projet
  1. Schéma JSON strict → violation immédiatement visible
  2. Classe "uncertain" → le modèle peut ne pas trancher
  3. Warning obligatoire → l'utilisateur sait toujours que c'est une aide
       │
       ▼ Limites
  Les guardrails ne détectent pas les hallucinations sémantiques
  → La supervision humaine reste obligatoire
```

> La suite : comment on mesure si tout ça fonctionne vraiment. Fichier suivant : `08_evaluer_un_modele.md`
