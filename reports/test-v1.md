# Locked test evaluation v1 (test split, evaluated once)

Runs produced by code revision `75b39bc7d1c3ceaf01cd53ae904dabbaf81041f9`, data `36ee960d81516ad4c3a4b514686d13da667f393868d567b39c99557ea376f0f5`. Approval: Single locked test evaluation (T12) of the frozen T11 configs approved by the team; communicated by michalklamecki on 2026-11-26.

Frozen models and val thresholds; nothing was retrained or retuned. Mean over seeds 42, 43, 44 (± sample sd for test); val means from reports/validation-v1.json.

## Models, val vs test

| model | average_precision val | average_precision test | f1 val | f1 test | event_k1_f1 val | event_k1_f1 test | event_k1_false_alarms_per_race val | event_k1_false_alarms_per_race test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 0.028 | 0.024 ± 0.000 | 0.055 | 0.048 ± 0.000 | 0.107 | 0.092 ± 0.000 | 982.400 | 1069.182 ± 0.000 |
| gru | 0.147 | 0.113 ± 0.001 | 0.232 | 0.189 ± 0.010 | 0.332 | 0.294 ± 0.019 | 41.600 | 49.242 ± 18.744 |
| lightgbm | 0.141 | 0.109 ± 0.002 | 0.220 | 0.171 ± 0.014 | 0.297 | 0.263 ± 0.027 | 29.100 | 34.939 ± 14.655 |
| logreg | 0.087 | 0.095 ± 0.000 | 0.173 | 0.153 ± 0.000 | 0.270 | 0.261 ± 0.000 | 84.700 | 105.273 ± 0.000 |
| mlp | 0.150 | 0.115 ± 0.002 | 0.237 | 0.189 ± 0.004 | 0.337 | 0.286 ± 0.005 | 33.333 | 42.879 ± 2.005 |
| tcn | 0.141 | 0.105 ± 0.007 | 0.233 | 0.187 ± 0.010 | 0.336 | 0.285 ± 0.004 | 43.433 | 55.273 ± 19.066 |

## Test groups

| group | laps | positives | races |
| --- | --- | --- | --- |
| green | 11681 | 293 | 11 |
| overall | 12354 | 301 | 11 |
| sc_vsc | 673 | 8 | 7 |
| special_tyre_rule | 1042 | 40 | 1 |

| model | green AP | green F1 | SC/VSC AP | SC/VSC F1 |
| --- | --- | --- | --- | --- |
| constant | 0.025 | 0.049 | 0.012 | 0.023 |
| gru | 0.116 | 0.192 | 0.044 | 0.036 |
| lightgbm | 0.112 | 0.175 | 0.031 | 0.000 |
| logreg | 0.102 | 0.158 | 0.028 | 0.030 |
| mlp | 0.119 | 0.193 | 0.023 | 0.000 |
| tcn | 0.107 | 0.189 | 0.040 | 0.070 |

Group counts are on the test split. SC/VSC and special-tyre-rule groups can have few positives, so their metrics are noisy; no claim is made for them.

## AP difference vs LightGBM (paired race bootstrap, 95% CI)

| model | seed | AP difference | 95% CI | result |
| --- | --- | --- | --- | --- |
| tcn | 42 | -0.0116 | [-0.0378, +0.0076] | tie |
| tcn | 43 | -0.0078 | [-0.0198, +0.0033] | tie |
| tcn | 44 | +0.0047 | [-0.0087, +0.0148] | tie |
| gru | 42 | +0.0033 | [-0.0110, +0.0168] | tie |
| gru | 43 | +0.0016 | [-0.0148, +0.0225] | tie |
| gru | 44 | +0.0070 | [-0.0085, +0.0211] | tie |
| mlp | 42 | +0.0067 | [-0.0003, +0.0175] | tie |
| mlp | 43 | +0.0022 | [-0.0069, +0.0132] | tie |
| mlp | 44 | +0.0092 | [+0.0010, +0.0177] | differs |
| logreg | 42 | -0.0152 | [-0.0298, +0.0014] | tie |
| logreg | 43 | -0.0151 | [-0.0287, +0.0017] | tie |
| logreg | 44 | -0.0122 | [-0.0274, +0.0054] | tie |
| constant | 42 | -0.0861 | [-0.1186, -0.0658] | differs |
| constant | 43 | -0.0860 | [-0.1135, -0.0704] | differs |
| constant | 44 | -0.0830 | [-0.1140, -0.0666] | differs |

A CI containing 0 is a tie.
