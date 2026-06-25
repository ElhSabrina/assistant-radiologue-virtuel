# 04 — Le modèle MedGemma

## Pourquoi ne pas juste utiliser ChatGPT ?

C'est la première question que tout le monde pose. Réponse courte : ChatGPT (GPT-4V) peut analyser des images médicales, mais :

1. **Il n'est pas open-weights** : tu ne peux pas le déployer chez toi, tu dépends d'une API payante d'OpenAI
2. **Il n'est pas reproductible** : la version que tu appelles aujourd'hui peut changer demain sans prévenir
3. **Il n'est pas conçu pour la médecine** : il connaît la médecine comme un étudiant qui a lu Wikipedia, pas comme un modèle pré-entraîné sur des millions de cas cliniques
4. **Il coûte cher à l'échelle** : chaque appel API est facturé. Un hôpital qui fait 1000 radios par jour ne peut pas se permettre ça

---

## C'est quoi "open-weights" ?

Un modèle **open-weights** (ou open-source), c'est un modèle dont les poids (les paramètres internes) sont publiquement disponibles. Tu peux les télécharger, les utiliser, les modifier.

```
  Modèle propriétaire (ex: GPT-4V)    Modèle open-weights (ex: MedGemma)
  ─────────────────────────────────    ──────────────────────────────────
  Tu envoies ta requête à un           Tu télécharges le modèle sur ton
  serveur distant (OpenAI)             propre serveur
       │                                    │
       ▼                                    ▼
  Tu paies par appel                   Tu contrôles tout
  Tu dépends de leur infrastructure    Tes données ne quittent pas ta machine
  Tu ne peux pas modifier le modèle    Tu peux l'adapter (fine-tuning)
  Aucune garantie de stabilité         Version figée, reproductible
```

Pour un projet médical, la confidentialité des données patients est cruciale. Les données ne doivent pas quitter l'infrastructure hospitalière.

---

## MedGemma en détail

**MedGemma** est un modèle développé par Google DeepMind, publié en 2025. Il fait partie de la famille Gemma (les modèles open-weights de Google).

Ce qui le rend spécial pour ce projet :

```
  ┌─────────────────────────────────────────────────────────┐
  │                    MedGemma 4B                          │
  │                                                         │
  │  Pré-entraîné sur :                                    │
  │  ├── Radiographies + leurs rapports radiologiques      │
  │  ├── Images histologiques (microscope)                  │
  │  ├── Textes biomédicaux (PubMed, cas cliniques)        │
  │  └── Données ophtalmologiques                          │
  │                                                         │
  │  Capacités :                                            │
  │  ├── Analyser une image médicale                       │
  │  ├── Répondre à des questions sur l'image              │
  │  ├── Générer un compte-rendu structuré                 │
  │  └── Exprimer son incertitude                          │
  │                                                         │
  │  Taille : 4 milliards de paramètres (~8 Go)            │
  │  Compatible : GPU grand public (RTX 3090, Colab Pro)   │
  └─────────────────────────────────────────────────────────┘
```

---

## Pourquoi 4B et pas plus grand ?

Il existe des modèles plus grands (7B, 13B, 70B paramètres). Plus grand = plus performant en général. Mais aussi plus lourd à faire tourner.

Un modèle de 4B paramètres peut tourner sur un GPU grand public avec 16 Go de VRAM (un RTX 3090 ou RTX 4090 qu'on peut louer sur Google Colab Pro pour quelques euros par heure).

Un modèle de 70B paramètres nécessite des clusters de GPU coûteux. Hors de portée pour un projet académique.

```
  Taille    VRAM requise   Coût mensuel (Colab)   Pertinence
  ──────    ────────────   ────────────────────   ──────────
  4B        8-16 Go        < 50 €                 Ce projet
  7B        16-24 Go       ~100 €                 Possible
  13B       24-40 Go       ~200 €                 Difficile
  70B       80+ Go         > 500 €                Hors budget
```

---

## La comparaison avec les autres modèles

Le projet a évalué plusieurs alternatives avant de choisir MedGemma :

| Modèle | Open-weights | Spécialisé médecine | Multimodal | Coût GPU | Reproductible |
|---|:---:|:---:|:---:|:---:|:---:|
| **MedGemma 4B** | oui | oui | oui | faible | oui |
| LLaVA-Med | oui | oui | oui | moyen | oui |
| GPT-4V (API) | non | partiel | oui | variable | non |
| BioViL-T | oui | oui | partiel | faible | oui |
| DenseNet-121 | oui | oui | non | faible | oui |

MedGemma est le seul qui coche toutes les cases.

**Pourquoi pas LLaVA-Med ?** Il est aussi open-weights et médical, mais plus difficile à faire tourner et moins bien maintenu. MedGemma bénéficie du support de Google DeepMind.

**Pourquoi pas BioViL-T ?** Il est bon pour analyser des images, mais il ne génère pas de texte explicatif. C'est un modèle "vision seule", pas un VLM complet.

**Pourquoi pas DenseNet-121 ?** C'est le modèle de CheXNet (2017), la référence historique. Mais il ne produit qu'un score de confiance, sans explication. Pas adapté à un usage clinique sérieux.

---

## Fine-tuning, LoRA et QLoRA : adapter le modèle

MedGemma est pré-entraîné sur des données médicales générales. Pour le spécialiser sur la détection de pneumonie avec notre dataset RSNA, on fait du **fine-tuning**.

### Fine-tuning classique

On reprend tous les paramètres du modèle et on les ré-entraîne sur nos données. Problème : avec 4 milliards de paramètres, c'est extrêmement coûteux en temps et en GPU.

### LoRA (Low-Rank Adaptation)

Au lieu de modifier tous les paramètres, LoRA n'entraîne que de très petites matrices supplémentaires ajoutées au modèle.

```
  Fine-tuning classique          LoRA
  ─────────────────────          ────
  Modifie 4B paramètres          Modifie ~10M paramètres (0.25%)
  Durée : plusieurs jours        Durée : quelques heures
  GPU : cluster professionnel    GPU : 1 GPU grand public
  Coût : milliers d'euros        Coût : quelques euros
  Résultat : modèle complet      Résultat : modèle + petit adaptateur
```

LoRA, c'est comme coller un post-it sur un livre : tu n'as pas réécrit le livre, mais tes annotations changent comment tu le lis.

### QLoRA (Quantized LoRA)

QLoRA pousse encore plus loin : en plus d'utiliser LoRA, il compresse les poids du modèle en **4 bits** au lieu de 32 bits. Ça réduit de 8x la mémoire requise.

```
  Format normal (float32) : chaque poids = 32 bits = 4 octets
  Format QLoRA (int4)     : chaque poids = 4 bits  = 0.5 octet
  Réduction mémoire : ×8

  MedGemma 4B en float32 : ~16 Go
  MedGemma 4B en QLoRA   : ~2 Go
```

Avec QLoRA, on peut fine-tuner MedGemma sur un GPU de 6 Go de VRAM, c'est-à-dire un GPU d'ordinateur portable haut de gamme ou un Google Colab gratuit.

---

## La stratégie en deux phases

Ce projet ne démarre pas directement avec MedGemma. Il utilise une progression en deux étapes :

```
  PHASE 1 : Baseline déterministe
  ─────────────────────────────────────────────────────────
  Un prédicteur à base de règles simples (seuils de pixels,
  zones d'intérêt manuelles). Aucune IA.

  Pourquoi ? Pour valider que le pipeline complet fonctionne
  (envoi d'image → analyse → JSON → base de données → API)
  AVANT d'introduire la complexité du modèle.

  Si les métriques sont mauvaises en phase 1 : c'est normal,
  la baseline est simple. Si le pipeline crashe : on corrige.

  PHASE 2 : MedGemma 4B
  ─────────────────────────────────────────────────────────
  On branche MedGemma à la place de la baseline.
  Si les métriques s'améliorent : l'amélioration vient bien
  du modèle, pas d'un bug corrigé au passage.
```

Cette approche est standard dans la recherche en IA : toujours établir une baseline avant de mesurer l'apport d'un modèle complexe.

---

## Résumé

- MedGemma 4B est choisi car il est open-weights, médical, multimodal et accessible
- "Open-weights" = on peut le faire tourner chez soi, sans dépendre d'une API externe
- Fine-tuning = on adapte le modèle à notre tâche spécifique (pneumonie RSNA)
- LoRA/QLoRA = techniques pour faire ce fine-tuning sans GPU professionnel
- On démarre avec une baseline simple avant d'introduire MedGemma

> La suite : comment tous les composants s'assemblent en un pipeline. Fichier suivant : `05_le_pipeline.md`
