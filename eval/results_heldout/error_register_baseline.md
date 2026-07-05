# Registre d'erreurs: baseline_predictions.csv

Total : 80 cas.

## Répartition

| Type | Nombre |
|---|---|
| TP: opacity correctly flagged | 35 |
| TN: normal correctly cleared | 35 |
| FP: normal image overcalled as opacity | 2 |
| FN: opacity missed and called normal | 2 |
| UA: model abstained (uncertain) instead of forcing a class | 6 |

## Cas commentés

| case_id | vérité | prédiction | conf. | type | sévérité | commentaire |
|---|---|---|---|---|---|---|
| RSNA_HO_001_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_002_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_003_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_004_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_005_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_006_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_007_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_008_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_009_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_010_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_011_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_012_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_013_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_014_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_015_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_016_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_017_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_018_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_019_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_020_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_021_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_022_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_023_normal | normal | normal | 0.9 | TN | low | normal correctly cleared |
| RSNA_HO_024_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_025_suspected_opacity | suspected_opacity | normal | 0.9 | FN | high | opacity missed and called normal |
| RSNA_HO_026_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_027_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_028_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_029_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_030_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_031_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_032_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_033_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_034_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_035_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_036_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_037_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_038_normal | normal | suspected_opacity | 0.6 | FP | medium | normal image overcalled as opacity |
| RSNA_HO_039_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_040_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_041_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_042_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_043_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_044_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_045_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_046_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_047_suspected_opacity | suspected_opacity | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_HO_048_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_049_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_050_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_051_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_052_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_053_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_054_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_055_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_056_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_057_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_058_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_059_normal | normal | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_HO_060_normal | normal | suspected_opacity | 0.6 | FP | medium | normal image overcalled as opacity |
| RSNA_HO_061_suspected_opacity | suspected_opacity | suspected_opacity | 0.8 | TP | low | opacity correctly flagged |
| RSNA_HO_062_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_063_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_064_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_065_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_066_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_067_normal | normal | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_HO_068_suspected_opacity | suspected_opacity | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_HO_069_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_070_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_071_normal | normal | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_HO_072_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_073_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_074_normal | normal | normal | 0.9 | TN | low | normal correctly cleared |
| RSNA_HO_075_suspected_opacity | suspected_opacity | normal | 0.9 | FN | high | opacity missed and called normal |
| RSNA_HO_076_suspected_opacity | suspected_opacity | suspected_opacity | 0.7 | TP | low | opacity correctly flagged |
| RSNA_HO_077_suspected_opacity | suspected_opacity | suspected_opacity | 0.6 | TP | low | opacity correctly flagged |
| RSNA_HO_078_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
| RSNA_HO_079_suspected_opacity | suspected_opacity | uncertain | 0.2 | UA | low | model abstained (uncertain) instead of forcing a class |
| RSNA_HO_080_normal | normal | normal | 0.95 | TN | low | normal correctly cleared |
