# Registre d'erreurs — baseline_predictions.csv

Total : 30 cas.

## Répartition

| Type | Nombre |
|---|---|
| TP — opacity correctly flagged | 14 |
| TN — normal correctly cleared | 14 |
| UA — model abstained (uncertain) instead of forcing a class | 2 |

## Cas commentés

| case_id | vérité | prédiction | conf. | type | sévérité | commentaire |
|---|---|---|---|---|---|---|
| RSNA_001_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_002_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_003_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_004_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_005_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_006_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_007_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_008_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_009_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_010_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_011_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_012_normal | normal | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_013_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_014_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_015_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_016_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_017_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_018_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_019_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_020_suspected_opacity | suspected_opacity | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_021_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_022_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_023_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_024_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_025_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_026_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_027_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_028_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_029_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_030_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
