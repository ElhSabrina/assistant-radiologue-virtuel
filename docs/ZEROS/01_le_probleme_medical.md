# 01 — Le problème médical

## C'est quoi une radiographie thoracique ?

Une radio thoracique (ou radio des poumons), c'est une image en noir et blanc de l'intérieur de ta poitrine. Les rayons X traversent le corps et les structures denses (os, poumons remplis de liquide) absorbent plus de rayonnement : elles apparaissent en blanc. L'air dans les poumons sains laisse passer les rayons : il apparaît en noir.

```
  Radiographie thoracique schématisée
  ────────────────────────────────────

       côtes (blanc)
      /     \
     /  ███  \     ← poumon gauche (noir = air = sain)
    │  █████  │
    │  █████  │
    │  ██░██  │    ← zone suspecte (gris = opacité possible)
     \       /
      \_____/
       colonne (blanc)
```

Un radiologue regarde ces images et cherche des anomalies : des zones qui ne devraient pas être là, des formes inhabituelles, des opacités.

---

## Le problème : il n'y a pas assez de radiologues

L'OMS estime qu'environ **2 milliards de radiographies** sont réalisées chaque année dans le monde sans jamais être lues par un spécialiste. Pas parce que personne ne veut les lire, mais parce qu'il n'y a pas assez de radiologues.

Dans les pays à ressources limitées (Afrique subsaharienne, certaines régions d'Asie), un seul radiologue peut être responsable de plusieurs hôpitaux à des centaines de kilomètres de distance. Résultat : les images attendent des jours, parfois des semaines.

Même dans les pays riches, les délais explosent. En France, certains hôpitaux attendent plusieurs heures pour avoir un résultat, même en urgence.

---

## C'est quoi la pneumonie ?

La pneumonie, c'est une infection des poumons. Les alvéoles pulmonaires (les petits sacs dans lesquels l'air entre) se remplissent de liquide ou de pus au lieu d'air.

Sur une radio, ça crée ce qu'on appelle une **opacité** : une zone blanche ou grise là où il devrait y avoir du noir.

```
  Poumon sain vs. poumon avec pneumonie
  ──────────────────────────────────────

  Sain          Pneumonie (lobe inférieur droit)
  ┌──────┐      ┌──────┐
  │ ░░░░ │      │ ░░░░ │
  │ ░░░░ │      │ ░░░░ │
  │ ░░░░ │      │ ░░██ │  ← opacité (blanc = liquide)
  └──────┘      └──────┘
  (air = noir)  (liquide = blanc)
```

La pneumonie est la première cause de mort chez les enfants de moins de 5 ans dans le monde. Détecter vite = traiter vite = sauver des vies.

---

## Ce que ce projet propose

Ce projet ne remplace pas le radiologue. Il lui donne un assistant.

L'idée : envoyer une image radio au système, et recevoir en quelques secondes une première analyse structurée :
- Y a-t-il une opacité suspecte ?
- Dans quelle zone ?
- Le système est-il certain ou non ?
- Quel avertissement doit accompagner le résultat ?

Le radiologue reste décisionnaire. L'IA fait le premier tri, réduit les délais, et permet de prioriser les cas urgents.

---

## Pourquoi c'est difficile à faire correctement

Un modèle d'IA entraîné sur des images peut très bien obtenir un score excellent sur des tests... et se tromper en conditions réelles. Plusieurs raisons :

- Les données d'entraînement ne représentent pas toujours la diversité des patients réels
- Un modèle peut "mémoriser" des artefacts liés aux machines spécifiques utilisées lors de la collecte
- L'IA peut être très confiante tout en se trompant (c'est ce qu'on appelle une hallucination)
- Un système non réglementé peut causer des dommages directs si ses erreurs ne sont pas détectées

C'est pourquoi ce projet ne vise pas juste la "performance" au sens statistique. Il vise la **défendabilité** : un système structuré, traçable, avec des garde-fous visibles.

---

## Résumé

| Fait | Chiffre |
|---|---|
| Radios annuelles dans le monde | ~2 milliards |
| Radios non lues faute de radiologues | ~2 milliards (pays à faibles ressources) |
| Images dans le dataset RSNA (notre point de départ) | ~30 000 |
| Morts annuelles d'enfants par pneumonie | ~700 000 |

> La suite : comprendre comment l'IA peut apprendre à lire ces images. Fichier suivant : `02_lia_pour_les_nuls.md`
