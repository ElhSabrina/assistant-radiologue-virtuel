# 02 — L'IA pour les nuls

## C'est quoi l'intelligence artificielle, vraiment ?

Oublie la science-fiction. Dans ce projet, "intelligence artificielle" veut dire quelque chose de très précis : **un programme qui a appris à faire une tâche en regardant beaucoup d'exemples**.

C'est tout.

### Exemple sans IA : les règles manuelles

Imagine que tu veuilles écrire un programme qui détecte la pneumonie. Approche classique :

```python
# Version "règles manuelles" (ce qu'on ne fait PAS)
def detecter_pneumonie(image):
    if zone_blanche_lobe_inferieur(image) > 0.3:
        if pas_de_cote_cassee(image):
            return "pneumonie probable"
    return "normal"
```

Le problème : les poumons ne ressemblent pas tous pareils. Les machines de radio ne produisent pas les mêmes images. Un enfant de 2 ans n'a pas les mêmes poumons qu'un adulte de 60 ans. Écrire toutes les règles à la main est impossible.

### Exemple avec IA : l'apprentissage automatique

Au lieu d'écrire les règles, on montre des milliers d'exemples au programme :

```
Exemple 1 : [image_radio_1.jpg] → label : "pneumonie"
Exemple 2 : [image_radio_2.jpg] → label : "normal"
Exemple 3 : [image_radio_3.jpg] → label : "pneumonie"
...
Exemple 30000 : [image_radio_30000.jpg] → label : "normal"
```

Le programme ajuste ses paramètres internes pour que sa réponse corresponde au label. Après des milliers d'itérations, il a "appris" à reconnaître les patterns associés à la pneumonie, sans qu'on lui ait jamais expliqué ce qu'est une alvéole.

---

## C'est quoi un modèle ?

Un modèle, c'est juste le programme après l'entraînement. Concrètement, c'est un fichier qui contient des milliards de nombres (appelés **paramètres** ou **poids**) qui encodent ce que le programme a appris.

```
  Avant entraînement         Après entraînement
  ─────────────────          ──────────────────
  poids = valeurs aléatoires  poids = valeurs optimisées
  résultat = n'importe quoi   résultat = détection fiable
```

MedGemma 4B (le modèle de ce projet) a **4 milliards de paramètres**. C'est un fichier de plusieurs gigaoctets.

---

## CNN vs VLM : quelle différence ?

### CNN (réseau de neurones convolutif) — la génération 1

Les premiers systèmes d'analyse d'images médicales utilisaient des **CNN**. Ils regardent une image et sortent un score.

```
  Image radio  →  [CNN]  →  score : 0.87 "pneumonie"
```

Problème : ce score ne dit rien de plus. Où est la zone suspecte ? Pourquoi 0.87 et pas 0.5 ? Impossible de savoir. C'est une boîte noire.

### VLM (Vision-Language Model) — la génération actuelle

Un VLM fait deux choses à la fois : il **voit** l'image ET il **génère du texte** pour l'expliquer.

```
  Image radio  →  [VLM]  →  "Opacité dans le lobe inférieur droit,
                              compatible avec une consolidation.
                              Classe : pneumonie. Confiance : haute."
```

C'est comme avoir un modèle qui peut vous écrire un compte-rendu, pas juste cocher une case.

---

## Comment un VLM "voit" une image ?

L'image est découpée en petits carrés appelés **patches** (comme des pièces de puzzle). Chaque patch est transformé en une suite de nombres. Le modèle lit ces nombres comme il lirait des mots.

```
  Image (512×512 pixels)
  ┌─────────────────────┐
  │ patch │ patch │ ... │
  │ patch │ patch │ ... │
  │  ...  │  ...  │ ... │
  └─────────────────────┘
       │
       ▼ transformation en vecteurs numériques
  [0.23, 1.45, -0.87, ...] [0.91, -0.12, 2.33, ...] ...
       │
       ▼ le modèle de langage traite ça comme une phrase
  "Que vois-je dans cette image ?"
       │
       ▼
  Réponse en langage naturel
```

---

## MedGemma : un VLM entraîné sur des données médicales

La plupart des VLMs grand public (comme ceux qui analysent des photos de vacances) sont entraînés sur des données générales. Ils ne connaissent pas l'anatomie pulmonaire.

**MedGemma** (Google DeepMind, 2025) a été pré-entraîné spécifiquement sur :
- Des radiographies et leurs rapports associés
- Des images histologiques (microscope)
- Des textes biomédicaux (articles scientifiques, cas cliniques)

Résultat : il comprend le vocabulaire médical, reconnaît les structures anatomiques, et sait ce qu'est une opacité.

---

## Le pré-entraînement vs. le fine-tuning

Deux étapes distinctes dans la vie d'un modèle :

```
  PHASE 1 : Pré-entraînement (fait par Google)
  ─────────────────────────────────────────────
  Données : millions d'images médicales
  Durée : semaines, sur des centaines de GPUs
  Résultat : MedGemma 4B (un modèle généraliste médical)

  PHASE 2 : Fine-tuning (ce qu'on peut faire dans ce projet)
  ──────────────────────────────────────────────────────────
  Données : notre dataset RSNA (~30 000 images de pneumonie)
  Durée : quelques heures, sur 1 GPU
  Résultat : un modèle spécialisé pour la détection de pneumonie
```

Le fine-tuning, c'est comme engager un médecin généraliste (MedGemma) et le former spécifiquement sur la pneumonie (notre dataset).

---

## Résumé visuel

```
  IA = programme qui apprend par l'exemple (pas par les règles)
       │
       ▼
  Modèle = le programme après apprentissage (un fichier de poids)
       │
       ▼
  VLM = modèle qui voit des images ET génère du texte
       │
       ▼
  MedGemma = VLM pré-entraîné sur des données médicales
       │
       ▼
  Fine-tuning = on l'adapte à notre tâche spécifique (pneumonie)
```

> La suite : comment on choisit les données d'entraînement, et pourquoi ce choix est critique. Fichier suivant : `03_les_donnees.md`
