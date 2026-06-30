# Images d'exemple pour tester l'application

Chaque image porte le nom de sa classe attendue, pour vérifier facilement les
prédictions de l'app (`streamlit run app/streamlit_app.py`).

Les 3 classes possibles :

| Préfixe du fichier      | Classe attendue      | Signification                                  |
|-------------------------|----------------------|------------------------------------------------|
| `normal_*`              | `normal`             | Aucun signe évident d'opacité                  |
| `suspected_opacity_*`   | `suspected_opacity`  | Opacité possible / signal visuel suspect       |
| `uncertain_*`           | `uncertain`          | Image ambiguë ou de qualité insuffisante       |

Deux sources :

- **Sans préfixe `rsna_`** : images synthétiques (copiées de `data/sample_images/`),
  à utiliser avec la configuration **Toy** — fonctionne sans GPU.
- **Préfixe `rsna_`** : vraies radiographies du challenge RSNA Pneumonia
  (copiées de `data/rsna_images/`), pertinentes pour les configurations
  **MedGemma**. RSNA ne fournit que `normal` et `suspected_opacity`.
  Ces fichiers sont exclus de git (données de compétition non redistribuables).
