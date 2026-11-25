# Validation study v1 (val split only)

Runs produced by code revision `f91914766761844937bd1e83e77740daa5af8a17`, data `36ee960d81516ad4c3a4b514686d13da667f393868d567b39c99557ea376f0f5`.

Caveats: configs were selected on val AP at seed 42 only, and the seed-42 final runs are the tuning runs, so val metrics of the selected configs are optimistically biased (e.g. TCN tuning AP 0.1495 vs its 3-seed mean 0.141). All val numbers come from one split of 10 races. Unbiased numbers come only from the locked test split (T12).

## Tuning (seed 42, val AP)

| model | learning rate | hidden | val AP | selected |
| --- | --- | --- | --- | --- |
| mlp | 0.0005 | 32 | 0.1350 |  |
| mlp | 0.0005 | 64 | 0.1463 |  |
| mlp | 0.001 | 32 | 0.1435 |  |
| mlp | 0.001 | 64 | 0.1492 |  |
| mlp | 0.002 | 32 | 0.1523 |  |
| mlp | 0.002 | 64 | 0.1549 | yes |
| gru | 0.0005 | 32 | 0.1298 |  |
| gru | 0.0005 | 64 | 0.1262 |  |
| gru | 0.001 | 32 | 0.1337 |  |
| gru | 0.001 | 64 | 0.1338 |  |
| gru | 0.002 | 32 | 0.1438 | yes |
| gru | 0.002 | 64 | 0.1357 |  |
| tcn | 0.0005 | 32 | 0.1322 |  |
| tcn | 0.0005 | 64 | 0.1495 | yes |
| tcn | 0.001 | 32 | 0.1395 |  |
| tcn | 0.001 | 64 | 0.1414 |  |
| tcn | 0.002 | 32 | 0.1407 |  |
| tcn | 0.002 | 64 | 0.1420 |  |

## Models, mean ± sd over seeds 42, 43, 44

| model | val AP | val F1 | event k=1 F1 | false alarms / race |
| --- | --- | --- | --- | --- |
| constant | 0.028 ± 0.000 | 0.055 ± 0.000 | 0.107 ± 0.000 | 982.400 ± 0.000 |
| gru | 0.147 ± 0.003 | 0.232 ± 0.004 | 0.332 ± 0.027 | 41.600 ± 18.093 |
| lightgbm | 0.141 ± 0.005 | 0.220 ± 0.010 | 0.297 ± 0.020 | 29.100 ± 16.229 |
| logreg | 0.087 ± 0.000 | 0.173 ± 0.000 | 0.270 ± 0.000 | 84.700 ± 0.000 |
| mlp | 0.150 ± 0.004 | 0.237 ± 0.004 | 0.337 ± 0.002 | 33.333 ± 2.750 |
| tcn | 0.141 ± 0.008 | 0.233 ± 0.005 | 0.336 ± 0.011 | 43.433 ± 11.736 |

## Green vs SC/VSC laps, mean over seeds

| group | laps | positives | races | prevalence |
| --- | --- | --- | --- | --- |
| green | 9805 | 284 | 10 | 0.029 |
| overall | 10415 | 296 | 10 | 0.028 |
| sc_vsc | 610 | 12 | 8 | 0.020 |
| special_tyre_rule | 1402 | 35 | 1 | 0.025 |

| model | green AP | green F1 | SC/VSC AP | SC/VSC F1 |
| --- | --- | --- | --- | --- |
| constant | 0.029 | 0.056 | 0.020 | 0.039 |
| gru | 0.152 | 0.239 | 0.044 | 0.000 |
| lightgbm | 0.148 | 0.227 | 0.056 | 0.042 |
| logreg | 0.091 | 0.178 | 0.029 | 0.042 |
| mlp | 0.157 | 0.243 | 0.051 | 0.000 |
| tcn | 0.148 | 0.239 | 0.063 | 0.025 |

SC/VSC has very few positives, so its AP is very noisy and no performance claim is made for it. An F1 of 0 means the frozen threshold (selected on the green-dominated val split) produces no SC/VSC hits.

## AP difference vs LightGBM (paired race bootstrap, 95% CI)

| model | seed | AP difference | 95% CI | result |
| --- | --- | --- | --- | --- |
| tcn | 42 | +0.0066 | [-0.0105, +0.0204] | tie |
| tcn | 43 | +0.0014 | [-0.0242, +0.0245] | tie |
| tcn | 44 | -0.0078 | [-0.0239, +0.0081] | tie |
| gru | 42 | +0.0009 | [-0.0201, +0.0306] | tie |
| gru | 43 | +0.0108 | [-0.0064, +0.0268] | tie |
| gru | 44 | +0.0058 | [-0.0119, +0.0249] | tie |
| mlp | 42 | +0.0120 | [+0.0007, +0.0200] | differs |
| mlp | 43 | +0.0116 | [-0.0151, +0.0372] | tie |
| mlp | 44 | +0.0050 | [-0.0072, +0.0162] | tie |
| logreg | 42 | -0.0558 | [-0.0700, -0.0280] | differs |
| logreg | 43 | -0.0478 | [-0.0662, -0.0213] | differs |
| logreg | 44 | -0.0574 | [-0.0721, -0.0326] | differs |
| constant | 42 | -0.1145 | [-0.1566, -0.0717] | differs |
| constant | 43 | -0.1065 | [-0.1595, -0.0540] | differs |
| constant | 44 | -0.1160 | [-0.1602, -0.0688] | differs |

The networks are statistically tied with LightGBM in 8 of 9 comparisons (a tie means the CI contains 0); logreg and constant are clearly worse. No network is claimed to beat LightGBM.
