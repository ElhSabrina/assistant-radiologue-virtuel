# ZEROS — Comprendre le projet de zéro

Ce dossier contient une explication complète du projet **Assistant Radiologue Virtuel**, écrite pour quelqu'un qui sait un peu coder mais n'a jamais touché à l'IA ni aux APIs médicales.

Aucun prérequis médical n'est nécessaire. Aucun prérequis en machine learning non plus. On repart vraiment de zéro.

---

## Ordre de lecture recommandé

```
01_le_probleme_medical.md      ← Pourquoi ce projet existe
02_lia_pour_les_nuls.md        ← Ce qu'est l'IA (vraiment)
03_les_donnees.md              ← Pourquoi les données, c'est tout
04_le_modele_MedGemma.md       ← Le cerveau du système
05_le_pipeline.md              ← Comment tout s'assemble
06_api_et_python.md            ← Comment parler au système en Python
07_hallucinations_et_securite.md ← Quand l'IA invente des choses
08_evaluer_un_modele.md        ← Comment savoir si ça marche vraiment
09_ethique_et_loi.md           ← Pourquoi l'IA médicale, c'est sérieux
```

---

## Carte du projet en un coup d'oeil

```
  PROBLÈME MÉDICAL
  (manque de radiologues)
         │
         ▼
  DONNÉES D'ENTRAÎNEMENT          ←── fichier 03
  (RSNA : ~30 000 radios)
         │
         ▼
  MODÈLE IA                       ←── fichier 04
  (MedGemma 4B)
         │
         ▼
  PIPELINE                        ←── fichier 05
  image → analyse → résultat JSON
         │
         ▼
  API FASTAPI                     ←── fichier 06
  (tu envoies une image, tu reçois un diagnostic)
         │
         ▼
  GARDE-FOUS + BASE DE DONNÉES    ←── fichiers 07, 08
  (sécurité + traçabilité)
         │
         ▼
  ÉTHIQUE + RÉGLEMENTATION        ←── fichier 09
  (AI Act, disclaimer obligatoire)
```

---

## Ce que tu seras capable de faire après avoir tout lu

- Expliquer pourquoi l'IA en radiologie est utile ET risquée
- Comprendre ce qu'est un modèle de vision-langage (VLM)
- Savoir pourquoi le choix du dataset est une décision critique
- Écrire du Python pour envoyer une radio au système et lire le résultat
- Identifier ce qu'est une hallucination IA et comment s'en protéger
- Expliquer ce que mesure un F1-score sans te perdre dans les maths
- Comprendre pourquoi l'AI Act européen concerne directement ce type de projet
