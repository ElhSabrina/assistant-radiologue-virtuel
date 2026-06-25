# 06 — L'API et Python : comment parler au système

## C'est quoi une API ?

API signifie **Application Programming Interface**, ce qui veut dire "interface de programmation". En pratique, c'est un moyen standardisé pour deux programmes de se parler.

### Analogie : le restaurant

```
  TU (le client)            SERVEUR (l'API)         CUISINE (le serveur)
  ──────────────            ──────────────           ────────────────────
  "Je veux une pizza        → Le serveur prend       → La cuisine prépare
   margherita"                ta commande et la         et renvoie la pizza
                              transmet à la cuisine

  Équivalent informatique :
  ──────────────────────────────────────────────────────────────────────
  Client Python             API FastAPI              Pipeline IA
  ──────────────            ──────────────           ─────────────────────
  "Voilà une radio,         → FastAPI reçoit         → preprocessing.py
   analyse-la"                valide la requête,        inference.py
                              lance le pipeline         guardrails.py
                                                        database.py

                            ← Renvoie le résultat ←  "{ prediction: pneumonia }"
```

---

## C'est quoi REST ?

REST (Representational State Transfer) est une convention pour organiser les APIs. Dans une API REST :

- Tu envoies des requêtes **HTTP** (le même protocole que les pages web)
- Chaque action a un **verbe** : GET (lire), POST (envoyer), PUT (modifier), DELETE (supprimer)
- Les données circulent en **JSON** (un format texte structuré)

Dans ce projet, l'endpoint principal est :
```
POST http://localhost:8000/predict
```

- `POST` : on envoie quelque chose au serveur (notre image)
- `http://localhost:8000` : l'adresse du serveur (sur notre machine en local)
- `/predict` : la route (comme un chemin dans une URL)

---

## Lancer le serveur

Avant de pouvoir appeler l'API, il faut démarrer le serveur FastAPI :

```bash
# Dans le terminal, depuis la racine du projet
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Si tout va bien, tu vois :
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Tu peux maintenant aller sur `http://localhost:8000/docs` dans ton navigateur pour voir la documentation interactive.

---

## Installer les dépendances Python

```bash
pip install requests
```

C'est tout ce dont tu as besoin côté client. `requests` est la bibliothèque standard pour faire des appels HTTP en Python.

---

## Appel simple : envoyer une image et lire le résultat

```python
import requests
import json

# ─── Configuration ────────────────────────────────────────────────────────────
URL_SERVEUR = "http://localhost:8000/predict"
CHEMIN_IMAGE = "ma_radio.jpg"   # chemin vers l'image à analyser

# ─── Envoi de la requête ──────────────────────────────────────────────────────
with open(CHEMIN_IMAGE, "rb") as image_file:
    reponse = requests.post(
        URL_SERVEUR,
        files={
            "image": (CHEMIN_IMAGE, image_file, "image/jpeg")
        }
    )

# ─── Lecture du résultat ──────────────────────────────────────────────────────
if reponse.status_code == 200:
    resultat = reponse.json()
    print(json.dumps(resultat, indent=2, ensure_ascii=False))
else:
    print(f"Erreur {reponse.status_code} : {reponse.text}")
```

### Ce que tu obtiens

```json
{
  "run_id": "a3f9c2d1-4b82-4e9c-bc7a-123456789abc",
  "prediction": "pneumonia",
  "confidence": 0.87,
  "uncertain": false,
  "zone": "lobe inférieur droit",
  "explanation": "Opacité dense localisée dans le lobe inférieur droit, compatible avec une consolidation pneumonique. Pas d'épanchement pleural visible.",
  "prompt_version": "improved_v1",
  "guardrail_ok": true,
  "warning": "Outil d'aide au diagnostic uniquement. L'interprétation finale doit être réalisée par un médecin qualifié. Ce résultat ne constitue pas un diagnostic médical.",
  "latency_ms": 2340,
  "timestamp": "2026-06-18T14:23:00"
}
```

---

## Comprendre chaque champ du JSON

```
  run_id          Identifiant unique de cette analyse (pour la traçabilité)
  prediction      Le résultat : "normal", "pneumonia" ou "uncertain"
  confidence      La "confiance" du modèle, entre 0.0 et 1.0
                  0.5 = modèle incertain / 0.95 = modèle très confiant
  uncertain       true si le modèle a choisi de ne pas trancher
  zone            Localisation anatomique de l'anomalie (si détectée)
  explanation     Explication en langage naturel du raisonnement
  prompt_version  Quelle version du prompt a été utilisée
  guardrail_ok    true = JSON valide, false = sortie corrigée par les guardrails
  warning         Avertissement obligatoire (toujours présent)
  latency_ms      Temps de traitement en millisecondes
  timestamp       Date et heure de l'analyse
```

---

## Traiter plusieurs images en boucle

```python
import requests
import json
import os
from pathlib import Path

URL_SERVEUR = "http://localhost:8000/predict"
DOSSIER_IMAGES = Path("rsna/images/test/")

resultats = []

for chemin_image in DOSSIER_IMAGES.glob("*.jpg"):
    with open(chemin_image, "rb") as f:
        reponse = requests.post(
            URL_SERVEUR,
            files={"image": (chemin_image.name, f, "image/jpeg")}
        )

    if reponse.status_code == 200:
        resultat = reponse.json()
        resultat["fichier"] = chemin_image.name   # on ajoute le nom du fichier
        resultats.append(resultat)
        print(f"{chemin_image.name} → {resultat['prediction']} ({resultat['confidence']:.0%})")
    else:
        print(f"Erreur sur {chemin_image.name} : {reponse.status_code}")

# Sauvegarder tous les résultats dans un fichier JSON
with open("resultats_batch.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False)

print(f"\n{len(resultats)} images analysées. Résultats sauvegardés dans resultats_batch.json")
```

---

## Gérer les erreurs proprement

```python
import requests
from requests.exceptions import ConnectionError, Timeout

URL_SERVEUR = "http://localhost:8000/predict"

def analyser_radio(chemin_image: str) -> dict:
    """
    Envoie une radio au serveur et retourne le résultat.
    Gère les cas d'erreur courants.
    """
    try:
        with open(chemin_image, "rb") as f:
            reponse = requests.post(
                URL_SERVEUR,
                files={"image": (chemin_image, f, "image/jpeg")},
                timeout=30    # on abandonne après 30 secondes
            )
        reponse.raise_for_status()    # lève une exception si status != 200
        return reponse.json()

    except FileNotFoundError:
        return {"erreur": f"Fichier introuvable : {chemin_image}"}

    except ConnectionError:
        return {"erreur": "Impossible de joindre le serveur. Est-il lancé ?"}

    except Timeout:
        return {"erreur": "Le serveur a mis trop de temps à répondre."}

    except requests.HTTPError as e:
        return {"erreur": f"Erreur HTTP {e.response.status_code}"}


# Utilisation
resultat = analyser_radio("radio_patient.jpg")

if "erreur" in resultat:
    print(f"Problème : {resultat['erreur']}")
else:
    print(f"Diagnostic : {resultat['prediction']}")
    print(f"Confiance : {resultat['confidence']:.0%}")
    print(f"Explication : {resultat['explanation']}")
    print(f"\n⚠️  {resultat['warning']}")
```

---

## Vérifier les performances après un batch

```python
import json

# Charger les résultats sauvegardés
with open("resultats_batch.json", "r") as f:
    resultats = json.load(f)

# Compter les résultats
total = len(resultats)
pneumonie = sum(1 for r in resultats if r["prediction"] == "pneumonia")
normal = sum(1 for r in resultats if r["prediction"] == "normal")
incertain = sum(1 for r in resultats if r["prediction"] == "uncertain")
guardrail_declenche = sum(1 for r in resultats if not r["guardrail_ok"])

print(f"Total analysé   : {total}")
print(f"Pneumonie       : {pneumonie} ({pneumonie/total:.0%})")
print(f"Normal          : {normal} ({normal/total:.0%})")
print(f"Incertain       : {incertain} ({incertain/total:.0%})")
print(f"Guardrail déclenché : {guardrail_declenche} ({guardrail_declenche/total:.0%})")

# Latence moyenne
latences = [r["latency_ms"] for r in resultats if "latency_ms" in r]
if latences:
    print(f"Latence médiane : {sorted(latences)[len(latences)//2]} ms")
```

---

## Ce que FastAPI vérifie automatiquement

FastAPI utilise **Pydantic** pour valider les données entrantes. Si tu envoies autre chose qu'une image, tu reçois une erreur 422 (Unprocessable Entity) :

```json
{
  "detail": [
    {
      "loc": ["body", "image"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

C'est une sécurité automatique : le serveur ne plantera jamais sur une entrée malformée, il renverra une erreur claire.

---

## La documentation interactive

Quand le serveur tourne, va sur `http://localhost:8000/docs`. Tu verras :

```
  ┌────────────────────────────────────────────────┐
  │  POST  /predict  Analyse une radiographie       │
  │  GET   /health   Vérifie que le serveur tourne  │
  │  GET   /history  Retourne les dernières analyses│
  └────────────────────────────────────────────────┘
```

Tu peux cliquer sur "Try it out", uploader une image et voir la réponse en direct, sans écrire une seule ligne de code.

---

## Résumé

| Concept | Ce que c'est | Dans ce projet |
|---|---|---|
| API | Interface pour faire parler deux programmes | FastAPI sur port 8000 |
| REST | Convention de communication HTTP | POST /predict |
| JSON | Format de données textuelles structurées | Réponse du serveur |
| `requests` | Bibliothèque Python pour faire des appels HTTP | `pip install requests` |
| Pydantic | Validation automatique des entrées/sorties | Intégré à FastAPI |

> La suite : pourquoi l'IA invente parfois des choses et comment s'en protéger. Fichier suivant : `07_hallucinations_et_securite.md`
