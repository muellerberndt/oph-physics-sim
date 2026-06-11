# OPH-CMB Success Ladder v0.4

This is the consolidated CMB comparison package. It combines:

1. **v0.2**: public Planck 2018 low-ell TT bandpower fit for OPH IR and parity kernels.
2. **v0.3**: CAMB Boltzmann transfer of the isotropic OPH IR kernel into TT/TE/EE/lensing diagnostics.
3. **v0.4**: first full-sky low-ell Monte Carlo rung for parity, large-angle correlation, and total low-ell power.

## Claim boundary

This is a successful **diagnostic ladder**, not a final cosmology likelihood. It uses public Planck TT bandpower tables, diagonalized/symmetric errors, CAMB transfer, and full-sky Gaussian power-spectrum Monte Carlo. It does not replace the official Planck likelihood, masked-sky map-level anomaly Monte Carlo, Planck component-separation covariance, ACT likelihoods, DESI likelihoods, or an OPH custom dark-sector Boltzmann module.

## Success criteria now met

- OPH screen tilt is numerically tied to Planck scalar tilt: `eta_R = 1 - n_s = 0.035`.
- OPH low-ell IR/global-repair parameters now have direct measured-data fits: `q_IR`, `ell_IR`, `chi2`, `AIC`, `BIC`.
- OPH parity/cap-pair parameters now have direct measured-data fits: `epsilon_P`, `ell_P`, `R_OE`.
- The isotropic OPH IR kernel has been propagated through CAMB to physical `TT/TE/EE/lensing` spectra.
- First PTE-style MC numbers exist for `R_OE`, `S_1/2`, and low-power ratio.

## Core readout

The strongest low-ell TT diagnostic is the joint IR + parity family:

```text
LCDM low-ell chi2 ell=2..29: 27.358720
OPH IR-only bestfit chi2:     14.239627
OPH joint bestfit chi2:       10.581185
```

The strongest CAMB-propagated isotropic result is:

```text
CAMB LCDM chi2 ell=2..29:       27.407237
CAMB OPH IR bestfit chi2:       16.653641
CAMB OPH joint-IR-part chi2:    15.865620
```

The first full-sky MC result says the observed odd/even low-ell TT statistic is rare under pure LCDM power (`PTE≈0.0107` upper-tail), but becomes much more typical under the fitted OPH parity envelope (`PTE≈0.4069`) and the joint OPH model (`PTE≈0.2778`).

## v0.2 Planck low-ell TT fit summary

| model                      |   ell_min |   ell_max |   n_points |   n_fitted_parameters |    chi2 |   dof_approx |   pte_chi2_approx |   delta_chi2_vs_LCDM |     AIC |     BIC |       q_IR |   ell_IR |   epsilon_P |     ell_P |
|:---------------------------|----------:|----------:|-----------:|----------------------:|--------:|-------------:|------------------:|---------------------:|--------:|--------:|-----------:|---------:|------------:|----------:|
| LCDM_baseline              |         2 |        29 |         28 |                     0 | 27.3587 |           28 |          0.498784 |              0       | 27.3587 | 27.3587 | nan        | nan      |  nan        | nan       |
| OPH_IR_reference_q015_ell6 |         2 |        29 |         28 |                     0 | 25.2034 |           28 |          0.616743 |              2.15531 | 25.2034 | 25.2034 |   0.15     |   6      |  nan        | nan       |
| OPH_IR_bestfit             |         2 |        29 |         28 |                     2 | 14.2396 |           26 |          0.969706 |             13.1191  | 18.2396 | 20.904  |   0.244555 |  33.615  |  nan        | nan       |
| OPH_parity_bestfit         |         2 |        29 |         28 |                     2 | 20.2054 |           26 |          0.781723 |              7.1533  | 24.2054 | 26.8698 | nan        | nan      |   -1.08543  |   4.36186 |
| OPH_IR_plus_parity_bestfit |         2 |        29 |         28 |                     4 | 10.5812 |           24 |          0.9917   |             16.7775  | 18.5812 | 23.91   |   0.166962 |  67.9795 |   -0.959409 |   3.66819 |

## Measured cosmological target ledger

| observable                     |   measured |   sigma_1 | OPH_mapping                             |   OPH_value |   z_score_abs | source                                                         |
|:-------------------------------|-----------:|----------:|:----------------------------------------|------------:|--------------:|:---------------------------------------------------------------|
| n_s                            |    0.965   | 0.004     | eta_R = 1 - n_s                         |    0.035    |     0         | Planck 2018 VI abstract                                        |
| Omega_c_h2 / Omega_b_h2        |    5.35714 | 0.0506453 | rho_A/rho_b diagnostic                  |    5.36347  |     0.124939  | Planck 2018 VI abstract + OPH dark-sector diagnostic           |
| S8_CMB_lensing_ACT_plus_Planck |    0.831   | 0.023     | OPH compressed growth diagnostic        |    0.828924 |     0.0902593 | ACT DR6 gravitational lensing map + OPH dark-sector diagnostic |
| sigma8_Planck2018              |    0.811   | 0.006     | OPH compressed growth diagnostic sigma8 |    0.807787 |     0.535466  | Planck 2018 VI abstract + OPH dark-sector diagnostic           |
| H0_Planck2018_km_s_Mpc         |   67.4     | 0.5       | background placeholder target           |  nan        |   nan         | Planck 2018 VI abstract                                        |
| Omega_m_Planck2018             |    0.315   | 0.007     | OPH compressed CAMB Omega_m diagnostic  |    0.315905 |     0.129315  | Planck 2018 VI abstract + OPH dark-sector diagnostic           |

## v0.3 CAMB bridge selected chi2 rows

| model                                   |   ell_min |   ell_max |   n_points |   chi2_diag |   chi2_per_point |   delta_chi2_vs_CAMB_LCDM_diag |
|:----------------------------------------|----------:|----------:|-----------:|------------:|-----------------:|-------------------------------:|
| CAMB_LCDM_powerlaw                      |         2 |        29 |         28 |     27.4072 |         0.97883  |                      0         |
| CAMB_LCDM_powerlaw                      |         2 |        40 |         39 |     38.7943 |         0.994726 |                      0         |
| CAMB_LCDM_powerlaw                      |        30 |      1200 |       1171 |   1359.22   |         1.16073  |                      0         |
| OPH_IR_reference_q0p15_ell6             |         2 |        29 |         28 |     26.2303 |         0.936796 |                      1.17694   |
| OPH_IR_reference_q0p15_ell6             |         2 |        40 |         39 |     37.6174 |         0.964549 |                      1.17694   |
| OPH_IR_reference_q0p15_ell6             |        30 |      1200 |       1171 |   1359.2    |         1.16072  |                      0.0149224 |
| OPH_IR_bestfit_lowell_q0p2446_ell33p615 |         2 |        29 |         28 |     16.6536 |         0.594773 |                     10.7536    |
| OPH_IR_bestfit_lowell_q0p2446_ell33p615 |         2 |        40 |         39 |     28.3537 |         0.727019 |                     10.4406    |
| OPH_IR_bestfit_lowell_q0p2446_ell33p615 |        30 |      1200 |       1171 |   1359.61   |         1.16106  |                     -0.388371  |
| OPH_IR_joint_IRpart_q0p1670_ell67p979   |         2 |        29 |         28 |     15.8656 |         0.566629 |                     11.5416    |
| OPH_IR_joint_IRpart_q0p1670_ell67p979   |         2 |        40 |         39 |     28.9906 |         0.743349 |                      9.80371   |
| OPH_IR_joint_IRpart_q0p1670_ell67p979   |        30 |      1200 |       1171 |   1361.57   |         1.16274  |                     -2.35687   |

## v0.4 full-sky Monte Carlo summary

| model                      |   R_OE_obs |   R_OE_median |   PTE_R_OE_upper_tail |   S_1_2_obs |   S_1_2_median |   PTE_S_1_2_lower_tail |   low_power_ratio_obs |   PTE_low_power_ratio_lower_tail |
|:---------------------------|-----------:|--------------:|----------------------:|------------:|---------------:|-----------------------:|----------------------:|---------------------------------:|
| LCDM_bestfit_theory        |    1.31082 |       1.00144 |                0.0107 |     6706.79 |        34411.3 |                 0.0775 |              0.901567 |                           0.0366 |
| OPH_IR_reference_q015_ell6 |    1.31082 |       1.00886 |                0.0093 |     6706.79 |        26021.1 |                 0.1198 |              0.920779 |                           0.0754 |
| OPH_IR_bestfit             |    1.31082 |       1.00715 |                0.0094 |     6706.79 |        19850.1 |                 0.169  |              1.11055  |                           0.9685 |
| OPH_parity_bestfit         |    1.31082 |       1.27557 |                0.4069 |     6706.79 |        22499.9 |                 0.1232 |              0.915948 |                           0.066  |
| OPH_IR_plus_parity_bestfit |    1.31082 |       1.22499 |                0.2778 |     6706.79 |        14769.7 |                 0.2299 |              1.08449  |                           0.918  |

## Files in this package

- `data/oph_cmb_success_ladder_v0_4.csv`
- `data/oph_cmb_success_summary_v0_4.json`
- `data/oph_planck2018_tt_lowl_fit_summary_v0_2.csv`
- `data/oph_measured_parameter_targets_v0_2.csv`
- `data/camb_oph_bridge_diag_chi2_v0_3.csv`
- `data/oph_lowell_fullsky_mc_summary_v0_4.csv`
- `figures/01_v02_planck_lowell_ratio.png`
- `figures/02_v02_chi2_comparison.png`
- `figures/03_v03_camb_lowell_TT.png`
- `figures/04_v03_camb_acoustic_TT.png`
- `figures/05_v03_lens_relative.png`
- `figures/06_v04_mc_parity_R_OE.png`
- `figures/07_v04_mc_S12.png`
- `figures/08_v04_mc_low_power_ratio.png`

## Next scientific gate

The next real gate is the masked-map analysis: Planck SMICA/NILC/Commander/SEVEM plus WMAP, with masks, component-separation covariance, multipole-vector quadrupole-octopole alignment, hemispherical asymmetry, BipoSH, and look-elsewhere accounting.
