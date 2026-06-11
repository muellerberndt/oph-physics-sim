# OPH-CMB Exact Low-ell Likelihood Gate v0.6

This package upgrades the earlier diagonal low-ell comparison to an exact full-sky TT cosmic-variance likelihood diagnostic. It still is not the official Planck low-ell pixel likelihood, but it uses the correct low-ell sampling law for a full-sky temperature power estimator:

```text
-2 ln L = sum_l (2l+1) [ Dhat_l / D_l_model + ln(D_l_model) ] + const
```

The OPH screen/freezeout project explicitly calls for the path:

```text
OPH screen covariance -> primordial bridge -> Boltzmann transfer -> TT/TE/EE/lensing
```

It separately flags parity, low-l anomalies, BipoSH, and map-space statistics as tests requiring Monte Carlo and masks. This package is the exact-TT likelihood rung of that path.

## Main v0.6 numbers, Planck PR3 TT, ell=2..29

| quantity | value |
|---|---:|
| OPH IR q_IR | 0.0982052501 |
| OPH IR ell_IR | 139.998287 |
| OPH IR improvement Delta(-2 ln L) vs LCDM | 4.414835 |
| OPH parity-only epsilon_P | -0.9800000000 |
| OPH parity-only ell_P | 5.043009 |
| OPH joint q_IR | 0.0966937975 |
| OPH joint ell_IR | 139.994806 |
| OPH joint epsilon_P | -0.9800000000 |
| OPH joint ell_P | 4.710022 |
| OPH joint improvement Delta(-2 ln L) vs LCDM | 10.569699 |

## Model selection, ell=2..29

|   ell_min |   ell_max |   n_multipoles | model_name                             | model_type   |   n_params |   neg2logL_no_const |   delta_neg2logL_improvement_vs_LCDM |   AIC_no_const |   BIC_no_const |   delta_AIC_vs_LCDM |   delta_BIC_vs_LCDM |   saturated_deviance |   PTE_saturated_deviance_MC |        q_IR |   ell_IR |   epsilon_P |     ell_P |
|----------:|----------:|---------------:|:---------------------------------------|:-------------|-----------:|--------------------:|-------------------------------------:|---------------:|---------------:|--------------------:|--------------------:|---------------------:|----------------------------:|------------:|---------:|------------:|----------:|
|         2 |        29 |             28 | LCDM_fullsky_TT                        | LCDM         |          0 |             6923.03 |                             0        |        6923.03 |        6923.03 |            0        |            0        |              26.6118 |                    0.56305  | nan         | nan      |      nan    | nan       |
|         2 |        29 |             28 | OPH_IR_fullsky_fit                     | IR           |          2 |             6918.62 |                             4.41483  |        6922.62 |        6925.28 |           -0.414835 |            2.24957  |              22.197  |                    0.787725 |   0.0982053 | 139.998  |      nan    | nan       |
|         2 |        29 |             28 | OPH_parity_fullsky_fit                 | parity       |          2 |             6916.69 |                             6.3407   |        6920.69 |        6923.36 |           -2.3407   |            0.323709 |              20.2711 |                    0.864817 | nan         | nan      |       -0.98 |   5.04301 |
|         2 |        29 |             28 | OPH_IR_plus_parity_fullsky_fit         | IR_parity    |          4 |             6912.46 |                            10.5697   |        6920.46 |        6925.79 |           -2.5697   |            2.75912  |              16.0421 |                    0.969083 |   0.0966938 | 139.995  |       -0.98 |   4.71002 |
|         2 |        29 |             28 | OPH_IR_reference_q0p15_ell6            | IR           |          0 |             6922.85 |                             0.185245 |        6922.85 |        6922.85 |           -0.185245 |           -0.185245 |              26.4266 |                    0.57635  |   0.15      |   6      |      nan    | nan       |
|         2 |        29 |             28 | OPH_IR_v05_diag_best_q0p2446_ell33p615 | IR           |          2 |             6923.76 |                            -0.731433 |        6927.76 |        6930.43 |            4.73143  |            7.39584  |              27.3433 |                    0.525142 |   0.244555  |  33.615  |      nan    | nan       |
|         2 |        29 |             28 | OPH_IR_v05_joint_q0p1670_ell67p979     | IR           |          2 |             6920.59 |                             2.4411   |        6924.59 |        6927.26 |            1.5589   |            4.22331  |              24.1707 |                    0.694825 |   0.166962  |  67.9795 |      nan    | nan       |

## Model selection, ell=2..64

|   ell_min |   ell_max |   n_multipoles | model_name                             | model_type   |   n_params |   neg2logL_no_const |   delta_neg2logL_improvement_vs_LCDM |   AIC_no_const |   BIC_no_const |   delta_AIC_vs_LCDM |   delta_BIC_vs_LCDM |   saturated_deviance |   PTE_saturated_deviance_MC |       q_IR |   ell_IR |   epsilon_P |     ell_P |
|----------:|----------:|---------------:|:---------------------------------------|:-------------|-----------:|--------------------:|-------------------------------------:|---------------:|---------------:|--------------------:|--------------------:|---------------------:|----------------------------:|-----------:|---------:|------------:|----------:|
|         2 |        64 |             63 | LCDM_fullsky_TT                        | LCDM         |          0 |             34358.1 |                             0        |        34358.1 |        34358.1 |            0        |            0        |              96.9063 |                  0.00448333 | nan        | nan      |      nan    | nan       |
|         2 |        64 |             63 | OPH_IR_fullsky_fit                     | IR           |          2 |             34355.5 |                             2.5807   |        34359.5 |        34363.8 |            1.4193   |            5.70557  |              94.3256 |                  0.00754167 |   0.120713 |  25.241  |      nan    | nan       |
|         2 |        64 |             63 | OPH_parity_fullsky_fit                 | parity       |          2 |             34351.8 |                             6.24651  |        34355.8 |        34360.1 |           -2.24651  |            2.03976  |              90.6598 |                  0.014525   | nan        | nan      |       -0.98 |   4.90672 |
|         2 |        64 |             63 | OPH_IR_plus_parity_fullsky_fit         | IR_parity    |          4 |             34349.4 |                             8.62526  |        34357.4 |        34366   |           -0.625262 |            7.94728  |              88.281  |                  0.021325   |   0.11593  |  25.5407 |       -0.98 |   4.57052 |
|         2 |        64 |             63 | OPH_IR_reference_q0p15_ell6            | IR           |          0 |             34357.9 |                             0.185245 |        34357.9 |        34357.9 |           -0.185245 |           -0.185245 |              96.721  |                  0.0052     |   0.15     |   6      |      nan    | nan       |
|         2 |        64 |             63 | OPH_IR_v05_diag_best_q0p2446_ell33p615 | IR           |          2 |             34365.6 |                            -7.49902  |        34369.6 |        34373.8 |           11.499    |           15.7853   |             104.405  |                  0.00116667 |   0.244555 |  33.615  |      nan    | nan       |
|         2 |        64 |             63 | OPH_IR_v05_joint_q0p1670_ell67p979     | IR           |          2 |             34379.7 |                           -21.6133   |        34383.7 |        34388   |           25.6133   |           29.8996   |             118.52   |                  5e-05      |   0.166962 |  67.9795 |      nan    | nan       |

## Selected exact multipole predictions

|   ell |   Planck_TT_Dell_obs_uK2 |   Planck_bestfit_TT_Dell_LCDM_uK2 |   OPH_IR_fullsky_fit_2_29_TT_Dell_pred_uK2 |   OPH_IR_plus_parity_fullsky_fit_2_29_TT_Dell_pred_uK2 |   LCDM_fullsky_TT_lower_tail_PTE_per_ell |   OPH_IR_fullsky_fit_2_29_lower_tail_PTE_per_ell |   OPH_IR_plus_parity_fullsky_fit_2_29_lower_tail_PTE_per_ell |
|------:|-------------------------:|----------------------------------:|-------------------------------------------:|-------------------------------------------------------:|-----------------------------------------:|-------------------------------------------------:|-------------------------------------------------------------:|
|     2 |                  225.895 |                          1016.73  |                                    916.912 |                                                329.784 |                               0.0468648  |                                        0.058212  |                                                    0.365219  |
|     3 |                  936.92  |                           963.727 |                                    869.141 |                                               1321.85  |                               0.550569   |                                        0.625667  |                                                    0.335344  |
|     4 |                  692.238 |                           912.608 |                                    823.076 |                                                478.86  |                               0.344848   |                                        0.421941  |                                                    0.837863  |
|     5 |                 1501.7   |                           874.477 |                                    788.729 |                                               1057.87  |                               0.936901   |                                        0.966039  |                                                    0.843967  |
|    10 |                  803.658 |                           817.174 |                                    737.369 |                                                651.987 |                               0.519676   |                                        0.650015  |                                                    0.78914   |
|    20 |                  659.863 |                           905.157 |                                    818.137 |                                                807.978 |                               0.0995711  |                                        0.19357   |                                                    0.208381  |
|    30 |                 1102.81  |                          1054.73  |                                    955.917 |                                                955.83  |                               0.621028   |                                        0.8074    |                                                    0.807543  |
|    40 |                 1715.07  |                          1228.96  |                                   1117.89  |                                               1119.38  |                               0.989178   |                                        0.998574  |                                                    0.998528  |
|    64 |                 1242.49  |                          1722.2   |                                   1585.21  |                                               1587.32  |                               0.00725721 |                                        0.0330469 |                                                    0.0323275 |

## Interpretation

The exact full-sky TT diagnostic keeps the v0.5 conclusion but makes it sharper. The best isotropic OPH IR/global-repair kernel remains broad and low-ell localized; it improves the TT low-ell likelihood while predicting negligible high-ell movement. The fitted parity envelope is a separate channel: it accounts for odd/even power structure and should not be folded into scalar primordial P_R(k). It belongs in angular covariance or a_lm-space Monte Carlo.

The strongest current parameter targets are:

```text
eta_R ~= 0.035, q_IR ~= 0.0982, ell_IR ~= 140.00
```

For the joint IR+parity angular template:

```text
q_IR ~= 0.0967, ell_IR ~= 139.99, epsilon_P ~= -0.9800, ell_P ~= 4.71
```

The derived angular scale of the IR kernel is approximately theta_IR ~= 180 deg / ell_IR.
For the IR-only v0.6 fit, theta_IR ~= 1.286 deg. Using D_* = 13800 Mpc, the corresponding rough wavenumber is k_IR ~= 0.0101448 Mpc^-1.

## Claim boundary

This is a public-spectrum, full-sky TT diagnostic. It is stronger than the earlier Gaussian error-bar proxy, but it is still not a substitute for the official Planck low-ell likelihood or a component-separated masked-map analysis. It produces exact numerical targets that the next Planck-map and finite-patch-simulator gates can try to derive or falsify.

## Files

- `data/01_fullsky_TT_wishart_model_selection_v0_6.csv`
- `data/02_fullsky_TT_bestfit_parameters_v0_6.csv`
- `data/03_IR_profile_grid_likelihood_region_v0_6.csv`
- `data/04_exact_multipole_predictions_l2_64_v0_6.csv`
- `data/05_parity_and_lowpower_exact_predictions_v0_6.csv`
- `data/06_cumulative_fullsky_TT_likelihood_gain_by_ell_v0_6.csv`
- `data/07_ACT_high_l_invariance_predictions_v0_6.csv if the high-l table is present`
- `data/08_exact_prediction_ledger_v0_6.csv`
- `figures/01_fullsky_TT_model_selection_v0_6.png`
- `figures/02_lowell_TT_exact_prediction_bands_v0_6.png`
- `figures/03_IR_q_ellIR_likelihood_contour_v0_6.png`
- `figures/04_cumulative_likelihood_gain_v0_6.png`
- `figures/05_per_ell_tail_probabilities_v0_6.png`
- `scripts/build_oph_cmb_exact_lowell_v0_6.py`

## v0.6 guardrail: exact low-l likelihood plus high-l preservation

A low-l-only full-sky fit tries to make the IR kernel almost constant across the fitted low-l window. That is not acceptable as a CMB prediction unless it preserves the acoustic spectrum. The guardrailed fit therefore minimizes exact full-sky TT likelihood for ell=2..29 plus a Gaussian high-l TT preservation term for ell=30..1200.

| model_name                                         | model_type   |   n_params |   objective_low_exact_plus_high_gaussian |   delta_objective_improvement_vs_LCDM |   low_exact_neg2logL |   delta_low_neg2logL_vs_LCDM |   high_l_chi2_30_1200 |   high_l_penalty_vs_LCDM |   saturated_deviance_low_l |   AIC_guardrail |   BIC_guardrail |   delta_AIC_vs_LCDM |   delta_BIC_vs_LCDM |       q_IR |   ell_IR |   epsilon_P |     ell_P |
|:---------------------------------------------------|:-------------|-----------:|-----------------------------------------:|--------------------------------------:|---------------------:|-----------------------------:|----------------------:|-------------------------:|---------------------------:|----------------:|----------------:|--------------------:|--------------------:|-----------:|---------:|------------:|----------:|
| LCDM_fullskyTT_highl_guardrail                     | LCDM         |          0 |                                  8190.23 |                              0        |              6923.03 |                     0        |               1267.2  |              0           |                    26.6118 |         8190.23 |         8190.23 |            0        |            0        | nan        | nan      |      nan    | nan       |
| OPH_IR_v05_diag_q0p2446_ell33p615                  | IR           |          2 |                                  8194.02 |                             -3.78152  |              6923.76 |                    -0.731433 |               1270.25 |              3.05009     |                    27.3433 |         8198.02 |         8208.19 |            7.78152  |           17.96     |   0.244555 |  33.615  |      nan    | nan       |
| OPH_IR_reference_q0p15_ell6                        | IR           |          0 |                                  8190.05 |                              0.185245 |              6922.85 |                     0.185245 |               1267.2  |             -9.09495e-13 |                    26.4266 |         8190.05 |         8190.05 |           -0.185245 |           -0.185245 |   0.15     |   6      |      nan    | nan       |
| OPH_IR_guardrailed_fullskyTT_highl_fit             | IR           |          2 |                                  8187.22 |                              3.0118   |              6919.43 |                     3.60026  |               1267.79 |              0.588461    |                    23.0116 |         8191.22 |         8201.4  |            0.9882   |           11.1667   |   0.117448 |  29.2874 |      nan    | nan       |
| OPH_IR_plus_parity_guardrailed_fullskyTT_highl_fit | IR_parity    |          4 |                                  8181.14 |                              9.09562  |              6913.31 |                     9.72107  |               1267.83 |              0.625459    |                    16.8908 |         8189.14 |         8209.5  |           -1.09562  |           19.2614   |   0.113079 |  29.7387 |       -0.98 |   4.62078 |

### Guardrailed selected multipole predictions

|   ell |   Planck_TT_Dell_obs_uK2 |   LCDM_TT_Dell_uK2 |   OPH_IR_guardrail_TT_Dell_uK2 |   OPH_IR_plus_parity_guardrail_TT_Dell_uK2 |   OPH_IR_guardrail_factor |   OPH_IR_plus_parity_guardrail_factor |
|------:|-------------------------:|-------------------:|-------------------------------:|-------------------------------------------:|--------------------------:|--------------------------------------:|
|     2 |                  225.895 |           1016.73  |                        898.122 |                                    328.786 |                  0.883343 |                              0.323376 |
|     3 |                  936.92  |            963.727 |                        852.06  |                                   1294.52  |                  0.88413  |                              1.34325  |
|     4 |                  692.238 |            912.608 |                        807.813 |                                    476.955 |                  0.88517  |                              0.522628 |
|     5 |                 1501.7   |            874.477 |                        775.187 |                                   1037.43  |                  0.886457 |                              1.18634  |
|    10 |                  803.658 |            817.174 |                        732.392 |                                    652.492 |                  0.896249 |                              0.798474 |
|    20 |                  659.863 |            905.157 |                        838.945 |                                    829.642 |                  0.92685  |                              0.916573 |
|    30 |                 1102.81  |           1054.73  |                       1011.31  |                                   1010.11  |                  0.958836 |                              0.957693 |
|    50 |                 1707.31  |           1421.27  |                       1411.85  |                                   1411.37  |                  0.993372 |                              0.993032 |
|   100 |                 2919.09  |           2697.86  |                       2697.86  |                                   2697.86  |                  0.999999 |                              0.999998 |
|   220 |                 6373.42  |           5730.14  |                       5730.14  |                                   5730.14  |                  1        |                              1        |
|   500 |                 2557.86  |           2446.6   |                       2446.6   |                                   2446.6   |                  1        |                              1        |
|  1000 |                  971.175 |           1062.45  |                       1062.45  |                                   1062.45  |                  1        |                              1        |

The CMB-compatible guardrailed IR result is: `q_IR=0.117448`, `ell_IR=29.287395`. The joint angular template is: `q_IR=0.113079`, `ell_IR=29.738682`, `epsilon_P=-0.980000`, `ell_P=4.620776`.

This is the sharper prediction target: it fits the largest-scale TT anomaly while explicitly paying the high-l acoustic penalty.
