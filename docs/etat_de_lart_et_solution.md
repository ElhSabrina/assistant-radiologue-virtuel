# État de l'art & Solution envisagée

## Sommaire

1. [Contexte et enjeux](#1-contexte-et-enjeux)
2. [État de l'art](#2-état-de-lart)
   - 2.1 Des CNN aux Vision-Language Models
   - 2.2 Prompt engineering et contrôle des sorties
   - 2.3 Hallucinations
   - 2.4 Adaptation efficace : LoRA/QLoRA
   - 2.5 Cadre réglementaire
3. [Solution envisagée](#3-solution-envisagée)
   - 3.1 Bases de données : RSNA Pneumonia (choix retenu)
   - 3.2 Architecture
   - 3.3 Choix du modèle : MedGemma
   - 3.4 Stack technique
   - 3.5 Évaluation
   - 3.6 Positionnement

---

En 2017, Rajpurkar et al. affirmaient dans CheXNet qu'un DenseNet-121 entraîné sur 112 120 images surpassait un panel de quatre radiologues pour la détection de pneumonie. Sept ans plus tard, aucun système de ce type n'est entré en pratique clinique courante. Ce décalage n'est pas technique : il est méthodologique. Un score de précision sur benchmark n'est pas une preuve de fiabilité clinique, et la confiance d'un modèle génératif dans sa propre sortie n'est pas un signal diagnostique. Ce projet part de ce constat : construire non pas le système le plus performant, mais le plus défendable : structuré, traçable, explicitement limité dans son périmètre.

---

## 1. Contexte et enjeux

La radiographie thoracique est l'examen d'imagerie le plus prescrit au monde. L'OMS évalue à près de 2 milliards le nombre d'actes réalisés sans interprétation dans les pays à ressources limitées, un déficit structurel que la pénurie de radiologues aggrave d'année en année. Les délais de lecture explosent même dans les systèmes de santé les mieux dotés : l'automatisation partielle de l'analyse d'image est un levier réel.

Mais l'essor des modèles génératifs multimodaux a révélé un risque propre à la médecine : ces modèles produisent des textes cliniquement convaincants sans que leur exactitude soit garantie. La fluidité d'un modèle de langage n'est pas un signal de fiabilité diagnostique.

---

## 2. État de l'art

### 2.1 Des CNN aux Vision-Language Models

CheXNet a inauguré une première vague de systèmes reposant sur des réseaux convolutifs, capables d'atteindre des performances remarquables sur des benchmarks standardisés ; CheXpert (224 316 radios, Stanford, 2019) et MIMIC-CXR (227 827 études appairées à leurs rapports, PhysioNet, 2019) restent les références actuelles. Mais ces CNN ont une limite fondamentale : ils produisent un score de confiance par classe, sans aucune explication naturelle. L'interprétabilité reste une post-rationalisation insuffisante pour un usage clinique sérieux.

Le tournant est venu des Vision-Language Models (VLMs), capables d'analyser une image et de rédiger une explication en langage naturel. BioViL-T (Microsoft Research, 2023) a montré qu'un pré-entraînement sur des paires radiographie/rapport radiologue produit des représentations médicales nettement supérieures à un modèle généraliste. LLaVA-Med (NeurIPS 2023) a démontré qu'un assistant visio-linguistique biomédical complet pouvait être obtenu en affinant un VLM sur des instructions médicales synthétiques en une seule journée d'entraînement, ce qui a ouvert la voie à des approches reproductibles sans infrastructure lourde.

La contribution la plus pertinente pour ce projet est MedGemma (Google DeepMind, 2025) : un modèle open-weights pré-entraîné sur des données médicales structurées (radiographies, histologie, textes biomédicaux), capable de raisonner sur une image en suivant des instructions et de produire une réponse structurée. MedGemma représente le meilleur compromis actuel entre accessibilité, performance médicale et coût de déploiement.

### 2.2 Prompt engineering et contrôle des sorties

Depuis GPT-3 (Brown et al., 2020), le prompt engineering est devenu une discipline à part entière. Deux apports sont directement mobilisés ici. Le chain-of-thought prompting (Wei et al., 2022), qui consiste à demander au modèle de décrire les structures anatomiques avant de conclure, améliore significativement la précision sur des tâches de raisonnement médical. Les sorties JSON structurées contraignent le modèle à un schéma de sortie fixe : cela garantit la traçabilité dans le pipeline et rend les hallucinations structurellement visibles, un champ vide ou une classe invalide devenant immédiatement détectable. Ce projet compare deux variantes de prompts (`baseline_v1` vs `improved_v1`) selon cette approche d'ablation, désormais standard dans l'évaluation des LLMs médicaux.

### 2.3 Hallucinations

Les hallucinations dans les LLMs (contenu généré non fondé ou contredisant les faits) sont particulièrement dangereuses en médecine : un modèle peut décrire des opacités inexistantes avec une parfaite assurance. Ce phénomène, formalisé par Ji et al. (2023) dans une revue de référence publiée dans ACM Computing Surveys, n'épargne pas les VLMs médicaux. La réponse de l'état de l'art repose sur plusieurs niveaux de défense complémentaires : un schéma JSON strict rend les incohérences structurellement visibles, une classe `uncertain` explicite évite de forcer une réponse quand le modèle est peu confiant, et une validation post-génération (les guardrails) détecte toute sortie invalide avant qu'elle n'atteigne l'utilisateur.

### 2.4 Adaptation efficace : LoRA/QLoRA

Le fine-tuning complet d'un VLM est prohibitif en ressources. LoRA (Hu et al., 2021) résout ce problème en n'entraînant qu'une fraction des paramètres, réduisant de plusieurs ordres de grandeur le coût de calcul sans perte significative de performance. QLoRA (Dettmers et al., 2023) pousse plus loin avec la quantification 4-bit, rendant l'adaptation possible sur un simple GPU grand public. Ces techniques ont démocratisé le fine-tuning médical et sont intégrées dans la plupart des pipelines modernes d'adaptation de modèles ouverts.

### 2.5 Cadre réglementaire

L'AI Act européen, adopté en 2024 et dont les obligations pour les systèmes à haut risque s'appliquent depuis 2025, classe les outils d'aide au diagnostic médical parmi les systèmes les plus strictement encadrés. Il impose transparence, traçabilité et supervision humaine obligatoire, trois exigences directement matérialisées dans ce projet, à l'échelle d'un prototype pédagogique.

---

## 3. Solution envisagée

### 3.1 Bases de données : RSNA Pneumonia (choix retenu)

Le projet retient **RSNA Pneumonia** comme dataset principal pour cette première phase. Avec ses ~30 000 radiographies annotées (bounding boxes + labels de pneumonie), il offre un périmètre maîtrisable, une documentation solide et constitue la voie recommandée pour valider l'ensemble du pipeline avant toute montée en échelle.

Les autres bases de référence disponibles (CheXpert 224 316 images, MIMIC-CXR 377 110 images, NIH ChestXray ~112 000 images) ne sont pas abandonnées : elles feront l'objet d'une section dédiée aux améliorations du projet, envisageant une montée en puissance via un dataset plus riche ou un modèle plus avancé.

| Dataset | Volume | Annotations | Phase |
|---|:---:|---|:---:|
| **RSNA Pneumonia** | **~30 000** | **bboxes + labels** | **principal** |
| CheXpert | 224 316 | rapports associés | améliorations |
| MIMIC-CXR | 377 110 | forte richesse clinique | améliorations |
| NIH ChestXray | ~112 000 | labels larges | améliorations |

La traçabilité des cas est assurée via CSV + SQLite, ce qui permet de reproduire exactement le sous-ensemble utilisé à chaque phase (20 images smoke test → 100–150 cas développement → 30 cas finaux commentés).

### 3.2 Architecture

La solution adopte un pipeline linéaire délibérément simple : chaque étape est isolée dans un module distinct, ce qui garantit que le modèle d'inférence peut être remplacé sans toucher aux garde-fous ni à la persistance.

```
Image → preprocessing.py (validation, resize 512×512, flag qualité)
      → inference.py     (baseline rule-based → MedGemma 4B)
      → guardrails.py    (validation JSON, downgrade incertitude, warning)
      → database.py      (INSERT run → SQLite, audit trail immuable)
      → JSON structuré   → API FastAPI → Streamlit / Gradio
```

### 3.3 Choix du modèle : MedGemma

La stratégie suit une progression en deux phases. D'abord, une baseline déterministe valide le pipeline complet avec un prédicteur à base de règles simples, garantissant que les métriques observées ensuite reflètent l'apport du modèle et non des bugs d'infrastructure. Ensuite, MedGemma 4B est branché via Hugging Face Transformers. C'est le seul modèle qui cumule poids ouverts, pré-entraînement médical explicite, capacité multimodale native et taille compatible avec un déploiement sur GPU grand public ou Colab Pro.

| Modèle | Open-weights | Médical | Multimodal | Coût GPU | Reproductible |
|---|:---:|:---:|:---:|:---:|:---:|
| MedGemma 4B | ✓ | ✓ | ✓ | Faible | ✓ |
| LLaVA-Med | ✓ | ✓ | ✓ | Moyen | ✓ |
| GPT-4V (API) | ✗ | Partiel | ✓ | Variable | ✗ |
| BioViL-T | ✓ | ✓ | Partiel | Faible | ✓ |
| DenseNet-121 | ✓ | ✓ | ✗ | Faible | ✓ |

Si les ressources le permettent, une adaptation via QLoRA sur des cas synthétiques annotés est envisagée en phase 3.

### 3.4 Stack technique

FastAPI expose l'endpoint `POST /predict` avec validation automatique des entrées/sorties via Pydantic et documentation OpenAPI autogénérée. Streamlit constitue la démo principale, Gradio l'alternative compatible Hugging Face Spaces, retenus plutôt qu'une SPA React parce que le périmètre du projet est l'ingénierie IA, pas le développement frontend. SQLite stocke chaque prédiction dans un fichier portable qui voyage avec le dépôt Git : zéro infrastructure à configurer, auditabilité immédiate via n'importe quel client SQL.

### 3.5 Évaluation

Le protocole retient des métriques complémentaires plutôt qu'une accuracy seule. Le macro F1-score est robuste au déséquilibre des classes, ce qui est essentiel en médecine où chaque pathologie a une importance clinique propre. Le taux de validité JSON (cible ≥ 95 %) mesure la fiabilité du pipeline bout-en-bout, indépendamment de la performance clinique. Le taux de warning (cible 100 %) est un indicateur de conformité réglementaire : chaque sortie doit porter le disclaimer. Le taux d'incertitude ne doit être ni nul (signe d'un modèle sur-confiant) ni trop élevé (ce qui le rendrait inutile cliniquement). La latence médiane évalue enfin l'utilisabilité pratique.

### 3.6 Positionnement

Ce projet ne rivalise pas avec CheXNet sur la performance brute. Son originalité est ailleurs : une méthodologie d'ingénierie responsable reproductible pour les VLMs médicaux, combinant des pratiques rarement intégrées ensemble dans un prototype académique.

| Dimension | Pratique courante | Ce projet |
|---|---|---|
| Modèle | Cloud propriétaire, fine-tuning lourd | VLM open-weights + LoRA optionnel |
| Sorties | Texte libre ou score unique | JSON schématisé, 3 classes, incertitude explicite |
| Garde-fous | Absents ou post-hoc | Validation en pipeline, downgrade automatique |
| Traçabilité | Logs ad hoc | SQLite structuré, versioning des prompts |
| Évaluation | Accuracy seule | Macro F1 + métriques opérationnelles |
| Éthique | Disclaimer externe | Warning intégré dans la réponse API elle-même |

C'est ce que le domaine nomme "Responsible AI by design", notion centrale dans l'AI Act et les recommandations OMS, rarement matérialisée à ce niveau dans des prototypes académiques comparables.
