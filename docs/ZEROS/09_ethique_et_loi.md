# 09 — Éthique et loi : pourquoi c'est sérieux

## Le paradoxe de CheXNet

En 2017, des chercheurs de Stanford publient un article retentissant : leur modèle CheXNet (un CNN entraîné sur 112 000 images) surpasse, selon leurs mesures, un panel de quatre radiologues pour la détection de pneumonie.

La presse mondiale s'emballe. "L'IA remplace les radiologues !"

Sept ans plus tard, CheXNet n'est entré dans aucun hôpital, dans aucun workflow clinique réel.

Pourquoi ?

```
  CE QUE L'ÉTUDE MESURAIT          CE QUE LA VRAIE VIE DEMANDE
  ──────────────────────────────    ──────────────────────────────────────
  Accuracy sur un benchmark         Fiabilité sur des images de N'IMPORTE
  (images d'un seul hôpital,        quel hôpital, N'IMPORTE quelle machine,
  une seule machine, conditions      N'IMPORTE quel patient
  contrôlées)

  Performance moyenne               Performance dans les cas rares et complexes
                                    (ceux qui comptent vraiment)

  Score sur images fixes            Robustesse sur images jamais vues

  Confiance élevée toujours         Expression de l'incertitude
```

Un bon score sur un benchmark ne garantit pas la fiabilité clinique. C'est ce constat qui fonde l'approche de ce projet.

---

## L'AI Act européen : la loi qui s'applique à ce projet

L'Union Européenne a adopté le **Règlement sur l'Intelligence Artificielle** (AI Act) en 2024. C'est la première loi au monde qui réglemente spécifiquement les systèmes d'IA.

Il classe les systèmes en quatre niveaux de risque :

```
  NIVEAU 1 : Risque inacceptable     → INTERDIT
  ─────────────────────────────────────────────────────
  Exemples : surveillance de masse, manipulation subliminale
  Action : ces systèmes ne peuvent pas exister en Europe

  NIVEAU 2 : Haut risque             → AUTORISÉ avec obligations strictes
  ─────────────────────────────────────────────────────
  Exemples : aide au diagnostic médical, systèmes RH, crédit
  Action : transparence, traçabilité, supervision humaine obligatoires

  NIVEAU 3 : Risque limité           → AUTORISÉ avec transparence
  ─────────────────────────────────────────────────────
  Exemples : chatbots, générateurs de contenus
  Action : l'utilisateur doit savoir qu'il parle à une IA

  NIVEAU 4 : Risque minimal          → LIBRE
  ─────────────────────────────────────────────────────
  Exemples : filtres anti-spam, jeux vidéo
  Action : aucune obligation spécifique
```

**Ce projet est en NIVEAU 2.** Un système d'aide au diagnostic médical est classé "haut risque" par défaut.

---

## Les trois obligations du niveau 2

### 1. Transparence

L'utilisateur doit toujours savoir qu'il utilise un système d'IA et comprendre ses limites.

Dans ce projet : le champ `warning` est **obligatoirement présent dans chaque réponse API**. Ce n'est pas un message affiché optionnellement dans une interface. C'est une donnée dans la réponse JSON elle-même.

```json
"warning": "Outil d'aide au diagnostic uniquement. L'interprétation finale
            doit être réalisée par un médecin qualifié. Ce résultat ne
            constitue pas un diagnostic médical."
```

### 2. Traçabilité

Chaque décision du système doit être enregistrée et retrouvable.

Dans ce projet : **chaque prédiction est sauvegardée dans SQLite** avec son timestamp, sa version de prompt, si le guardrail a été déclenché, la latence, etc.

```
  Pourquoi c'est important ?
  ──────────────────────────────────────────────────────────────
  Scénario : un médecin utilise le système pendant 3 mois.
  Rétrospectivement, on découvre que le modèle se trompait
  systématiquement sur les patients de plus de 80 ans.

  SANS traçabilité : impossible de retrouver quels patients
  ont été potentiellement mal orientés.

  AVEC traçabilité : on retrouve tous les cas concernés,
  on peut prévenir les médecins, auditer les décisions.
```

### 3. Supervision humaine

Un humain qualifié doit rester décisionnaire. Le système d'IA ne peut pas prendre de décision finale autonome.

Dans ce projet :
- La réponse est toujours présentée comme une **aide**, jamais comme un **diagnostic**
- La classe `uncertain` oblige explicitement à une vérification humaine
- Le warning rappelle à chaque fois que la décision finale appartient au médecin

---

## "Responsible AI by design"

Ce terme désigne une approche où les exigences éthiques et réglementaires sont intégrées dès la conception du système, pas ajoutées à la fin comme un vernis.

```
  Approche classique                  Responsible AI by design
  ──────────────────────────────      ──────────────────────────────────────
  1. Développer le système             1. Identifier les risques (hallucinations,
  2. L'évaluer (accuracy)                  déséquilibre des données, biais)
  3. Le déployer                       2. Concevoir les garde-fous en même temps
  4. Ajouter un disclaimer             3. Intégrer la traçabilité dès le début
     en bas de page                    4. Évaluer sur des métriques multiples
  5. Espérer que ça marche             5. Documenter les limites explicitement
```

Concrètement dans ce projet :

| Ce qu'on aurait pu faire | Ce qu'on fait à la place |
|---|---|
| Optimiser uniquement le F1-score | Mesurer aussi validité JSON, warning, incertitude |
| Utiliser un modèle cloud (GPT-4V) | Modèle open-weights déployable localement |
| Stocker les logs dans un fichier texte | Base SQLite structurée et auditable |
| Afficher le warning dans l'interface | Warning intégré dans la réponse API elle-même |
| Ignorer les cas ambigus | Classe `uncertain` explicite |

---

## Les limites que ce projet assume publiquement

Un projet "responsible AI" ne prétend pas que son système est parfait. Il documente ses limites.

Les limites de ce système :

**1. Périmètre étroit**
Le modèle est évalué sur RSNA Pneumonia. Il peut se comporter différemment sur des images d'autres hôpitaux, d'autres machines, d'autres populations.

**2. Deux classes seulement**
Le système distingue "pneumonie" vs "normal". Il ne détecte pas d'autres pathologies (tuberculose, cancer, embolie pulmonaire...). Si l'image montre un cancer, le système peut dire "normal".

**3. Hallucinations sémantiques non détectables**
Les guardrails détectent les violations de schéma. Ils ne peuvent pas détecter une explication médicalement fausse mais syntaxiquement correcte.

**4. Ce n'est pas un dispositif médical certifié**
Pour être déployé dans un vrai hôpital, ce système devrait obtenir le marquage CE comme dispositif médical (Règlement 2017/745), un processus qui prend des années et coûte des millions d'euros.

---

## Pourquoi documenter ses propres limites ?

Ça semble contre-intuitif : pourquoi dire ce qu'on ne sait pas faire ?

Parce que dans un contexte médical, **une fausse confiance est plus dangereuse qu'un doute honnête**.

Un médecin qui sait que le système ne détecte que la pneumonie fera lui-même les vérifications complémentaires. Un médecin qui croit que le système couvre tout ne les fera peut-être pas.

---

## Résumé

```
  CheXNet (2017) → bon score, aucun déploiement clinique.
  Raison : performance sur benchmark ≠ fiabilité clinique.

  AI Act (2024) → classe les aides au diagnostic comme "haut risque".
  Obligations : transparence + traçabilité + supervision humaine.

  Ce projet répond à ces obligations :
  ├── Warning dans chaque réponse API        (transparence)
  ├── SQLite avec chaque prédiction tracée   (traçabilité)
  ├── Classe "uncertain" + supervision req.  (supervision humaine)
  └── Limites documentées publiquement       (honnêteté)

  "Responsible AI by design" = intégrer l'éthique dès la conception,
  pas la coller à la fin.
```

---

## Pour aller plus loin

- [AI Act officiel (EUR-Lex)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32024R1689)
- Ji et al. (2023) — Survey on Hallucination in Natural Language Generation, ACM Computing Surveys
- Rajpurkar et al. (2017) — CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays, Stanford ML Group
- Google DeepMind (2025) — MedGemma Technical Report
