# OPH-CMB Boltzmann Bridge v0.3

This package propagates the isotropic OPH IR/global-repair primordial correction through CAMB.

## Claim boundary

This is a diagnostic Boltzmann-transfer bridge. It is **not** the official Planck likelihood, it does **not** include map-level anomaly Monte Carlo, and it does **not** include angular parity/BipoSH covariance in the scalar primordial table. Parity and BipoSH remain `a_lm`-space effects.

## Data and settings

- CAMB Python package: `1.6.6`
- Background: `H0=67.4`, `ombh2=0.0224`, `omch2=0.12`, `tau=0.054`, `mnu=0.06`.
- Primordial baseline: `A_s=2.1e-09`, `n_s=0.965`, pivot `k0=0.05 Mpc^-1`.
- Lensing: CAMB lensed total spectra with `NonLinear_none`.
- OPH bridge: `ell ~= k D_*`, with `D_*=13800.0 Mpc`.

## Isotropic OPH correction

```text
F_IR(k) = 1 - q_IR * exp[-ell(k)(ell(k)+1)/(ell_IR(ell_IR+1))]
ell(k) = max(k * D_star, 2)
```

The two OPH runs are:

| model | q_IR | ell_IR |
|---|---:|---:|
| OPH_IR_reference_q0p15_ell6 | 0.15 | 6.0 |
| OPH_IR_bestfit_lowell | 0.2445545068 | 33.61495818 |

## Low-ell TT diagnostic chi^2 against public Planck TT table

|   ell_min |   ell_max |   n_points |    chi2 |   chi2_per_point |   pte_approx | model                       |   delta_chi2_vs_CAMB_LCDM |
|----------:|----------:|-----------:|--------:|-----------------:|-------------:|:----------------------------|--------------------------:|
|         2 |        29 |         28 | 27.7469 |         0.990961 |     0.477918 | CAMB_LCDM_powerlaw          |                   0       |
|         2 |        29 |         28 | 26.5658 |         0.94878  |     0.541972 | OPH_IR_reference_q0p15_ell6 |                   1.18108 |
|         2 |        29 |         28 | 16.8021 |         0.600074 |     0.952397 | OPH_IR_bestfit_lowell       |                  10.9448  |

## Selected multipole ratios

|   ell |   CAMB_LCDM_powerlaw_TT |   CAMB_LCDM_powerlaw_TT_ratio_to_CAMB_LCDM |   CAMB_LCDM_powerlaw_EE_ratio_to_CAMB_LCDM |   CAMB_LCDM_powerlaw_TE |   OPH_IR_reference_q0p15_ell6_TT |   OPH_IR_reference_q0p15_ell6_TT_ratio_to_CAMB_LCDM |   OPH_IR_reference_q0p15_ell6_EE_ratio_to_CAMB_LCDM |   OPH_IR_reference_q0p15_ell6_TE |   OPH_IR_bestfit_lowell_TT |   OPH_IR_bestfit_lowell_TT_ratio_to_CAMB_LCDM |   OPH_IR_bestfit_lowell_EE_ratio_to_CAMB_LCDM |   OPH_IR_bestfit_lowell_TE |
|------:|------------------------:|-------------------------------------------:|-------------------------------------------:|------------------------:|---------------------------------:|----------------------------------------------------:|----------------------------------------------------:|---------------------------------:|---------------------------:|----------------------------------------------:|----------------------------------------------:|---------------------------:|
|     2 |                1022.47  |                                          1 |                                          1 |                2.62879  |                          952.088 |                                            0.931165 |                                            0.908561 |                         2.3019   |                    802.863 |                                      0.785219 |                                      0.760176 |                   1.99207  |
|     3 |                 968.3   |                                          1 |                                          1 |                2.94358  |                          910.544 |                                            0.940353 |                                            0.933063 |                         2.64367  |                    765.333 |                                      0.790388 |                                      0.763166 |                   2.23395  |
|     4 |                 916.337 |                                          1 |                                          1 |                2.75751  |                          872.891 |                                            0.952588 |                                            0.951844 |                         2.54297  |                    728.974 |                                      0.795531 |                                      0.766309 |                   2.09804  |
|     5 |                 877.638 |                                          1 |                                          1 |                2.34598  |                          846.649 |                                            0.96469  |                                            0.965538 |                         2.21439  |                    702.974 |                                      0.800984 |                                      0.769563 |                   1.79084  |
|    10 |                 819.301 |                                          1 |                                          1 |                0.845996 |                          816.161 |                                            0.996168 |                                            0.998326 |                         0.84207  |                    684.071 |                                      0.834944 |                                      0.811395 |                   0.668882 |
|    20 |                 907.011 |                                          1 |                                          1 |                1.295    |                          907.009 |                                            0.999998 |                                            0.999998 |                         1.295    |                    826.464 |                                      0.911195 |                                      0.871917 |                   1.11228  |
|    30 |                1057.21  |                                          1 |                                          1 |                1.88274  |                         1057.21  |                                            1        |                                            1        |                         1.88274  |                   1017.36  |                                      0.962309 |                                      0.924154 |                   1.71114  |
|    50 |                1424.84  |                                          1 |                                          1 |                0.558917 |                         1424.84  |                                            1        |                                            1        |                         0.558917 |                   1418.95  |                                      0.995863 |                                      0.984803 |                   0.524026 |
|   100 |                2705.11  |                                          1 |                                          1 |              -23.2143   |                         2705.11  |                                            1        |                                            1        |                       -23.2143   |                   2705.1   |                                      0.999997 |                                      0.999986 |                 -23.2141   |
|   220 |                5739.1   |                                          1 |                                          1 |               13.4948   |                         5739.1   |                                            1        |                                            1        |                        13.4948   |                   5739.11  |                                      1        |                                      0.999979 |                  13.4945   |
|   500 |                2448.25  |                                          1 |                                          1 |              -58.7239   |                         2448.25  |                                            1        |                                            1        |                       -58.7239   |                   2448.26  |                                      1        |                                      0.999979 |                 -58.7254   |
|  1000 |                1061.96  |                                          1 |                                          1 |              -24.2164   |                         1061.96  |                                            1        |                                            1        |                       -24.2164   |                   1061.94  |                                      0.999974 |                                      1.00003  |                 -24.2175   |

## Readout

The CAMB transfer confirms the key separation: OPH's isotropic IR/global-repair correction changes the largest angular modes while leaving the acoustic peak region nearly unchanged. The best v0.2 IR correction suppresses the CAMB quadrupole-scale TT power to about `0.785` of the baseline, while by `ell=220` the TT ratio is `1.000001`.

## Files

- `data/camb_oph_bridge_spectra_v0_3.csv` — CAMB TT/EE/BB/TE spectra.
- `data/camb_oph_bridge_lowell_chi2_v0_3.csv` — low-ell TT diagnostic chi^2.
- `data/camb_oph_bridge_selected_ell_ratios_v0_3.csv` — selected multipole ratios.
- `data/camb_oph_bridge_summary_v0_3.json` — machine-readable summary.
- `figures/01_camb_lowell_TT_bridge_v0_3.png` — low-ell TT comparison.
- `figures/02_camb_TT_ratio_bridge_v0_3.png` — TT ratio after transfer.
- `figures/03_camb_TT_acoustic_bridge_v0_3.png` — acoustic-scale TT spectra.
- `figures/04_camb_EE_ratio_bridge_v0_3.png` — EE ratio after transfer.
- `scripts/run_oph_cmb_boltzmann_v0_3.py` — reproducibility script.
