# Registre d'erreurs: improved_predictions.csv

Total : 30 cas.

## Répartition

| Type | Nombre |
|---|---|
| TP: opacity correctly flagged | 10 |
| TN: normal correctly cleared | 15 |
| FN: opacity missed and called normal | 2 |
| UA: model abstained (uncertain) instead of forcing a class | 3 |

## Cas commentés

| case_id | vérité | prédiction | conf. | type | sévérité | commentaire |
|---|---|---|---|---|---|---|
| RSNA_001_suspected_opacity | suspected_opacity | suspected_opacity | 0.65 | TP | low | opacity correctly flagged |
| RSNA_002_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_003_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_004_suspected_opacity | suspected_opacity | suspected_opacity | 0.65 | TP | low | opacity correctly flagged |
| RSNA_005_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_006_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_007_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_008_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_009_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_010_suspected_opacity | suspected_opacity | uncertain | 0.4 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_011_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_012_normal | normal | normal | 0.9 | TN | low | normal correctly cleared |
| RSNA_013_suspected_opacity | suspected_opacity | suspected_opacity | 0.65 | TP | low | opacity correctly flagged |
| RSNA_014_suspected_opacity | suspected_opacity | uncertain | 0.4 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_015_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_016_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_017_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_018_suspected_opacity | suspected_opacity | normal | 0.95 | FN | high | opacity missed and called normal |
| RSNA_019_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_020_suspected_opacity | suspected_opacity | normal | 0.7 | FN | high | opacity missed and called normal |
| RSNA_021_suspected_opacity | suspected_opacity | uncertain | 0.4 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_022_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_023_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_024_normal | normal | normal | 0.9 | TN | low | normal correctly cleared |
| RSNA_025_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_026_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_027_suspected_opacity | suspected_opacity | suspected_opacity | 0.65 | TP | low | opacity correctly flagged |
| RSNA_028_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_029_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_030_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
