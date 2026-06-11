# OPH-CMB Public-Source Assessment v0.8

## Scope

This package adds a public-source comparison table on top of the v0.6 exact-kernel CMB prediction surface and the earlier v0.7 guardrailed assessment. It compares OPH quantities to public Planck PR3 spectra, Planck 2018 cosmological parameters, ACT DR6 CMB lensing, BICEP/Keck tensor limits, and DESI DR2 BAO/neutrino constraints.

Claim boundary: this is still a diagnostic comparison, not an official Planck likelihood and not a masked map-space analysis. The OPH draft itself says the proper staged route is screen covariance -> primordial bridge -> CAMB/CLASS -> TT/TE/EE/lensing, and that Monte Carlo skies, Boltzmann transfer, likelihood fitting, mask robustness, and finite-patch derivation are required for stronger claims.

## Headline public-number table

| Quantity | OPH number | Public benchmark | Current comparison |
| --- | --- | --- | --- |
| Exact n_s | 0.964841143031 | Planck 2018 n_s=0.965±0.004 | -0.040σ |
| Exact q_IR, ell_IR | 0.25, 32 | Planck PR3 low-ell TT diagnostic | Delta chi2 TT=10.590 |
| TT+TE+EE exact-kernel score | chi2=67.772 | LCDM diagonal proxy | Delta chi2=9.242; Delta AIC=-7.242; Delta BIC=-4.811 |
| Quadrupole D2 | scalar 800.1; scalar×parity 314.8 μK² | Planck 225.9±332.7; LCDM 1022.8 | pulls: LCDM +2.40σ; scalar +1.73σ; scalar×parity +0.27σ |
| Odd/even ratio 2..29 | scalar 1.011405; scalar×parity 1.216064 | Planck 1.310820; LCDM 1.002724 | angular-covariance diagnostic |
| Acoustic preservation ell=220 | TT ratio 0.999999338 | first acoustic peak region | -0.000066% shift |
| S8 | 0.828924037 | ACT+Planck lensing 0.831±0.023 | -0.090σ |
| Sum m_nu | 0.090011929645 eV | Planck+BAO <0.12 eV; DESI LCDM <0.064 eV; DESI w0wa <0.16 eV | mixed |

## The main numerical status

The strongest scalar-screen result is now the exact low-ell kernel candidate:

```text
eta_R = e alpha sqrt(pi) = 0.035158856969
n_s = 1 - eta_R = 0.964841143031
F_IR(ell) = 1 - (1/4) exp[-ell(ell+1)/(32*33)]
```

Against the public diagonal low-ell proxy for TT+TE+EE over ell=2..29, the exact scalar kernel gives Delta chi2 = 9.242, Delta AIC = -7.242, and Delta BIC = -4.811. Broken out by spectrum: TT improves by 10.590, TE improves by 0.465, and EE worsens by -1.814. The EE result is a pressure point.

The angular parity diagnostic is kept separate from scalar primordial power. It uses:

```text
F_P(ell) = 1 - (-1)^ell exp(-ell/4)
```

This is an angular covariance diagnostic, not a scalar P_R(k) export. It helps the odd/even TT structure, but must be tested with component-separated maps and masks before being promoted.

## Files

| File | Contents |
|---|---|
| `data/00_public_source_manifest_v0_8.csv` | public source ledger |
| `data/01_public_source_assessment_table_v0_8.csv` | main assessment table requested by the user |
| `data/02_odd_even_public_assessment_v0_8.csv` | odd/even TT ratio table |
| `data/03_selected_multipole_public_assessment_v0_8.csv` | selected multipoles: Planck, LCDM, OPH scalar, OPH scalar x angular parity |
| `data/04_lowell_amplitude_assessment_v0_8.csv` | multiplicative amplitude checks |
| `data/05_lowell_TT_TE_EE_score_assessment_v0_8.csv` | low-ell TT/TE/EE chi2/AIC/BIC proxy scores |
| `data/06_exact_parameter_table_v0_8.csv` | exact OPH parameter table |
| `data/07_next_success_gates_v0_8.csv` | next gates and failure modes |
| `figures/01_lowell_TT_public_assessment_v0_8.png` | public low-ell TT plot |
| `figures/02_lowell_TT_ratios_v0_8.png` | low-ell ratios to LCDM |
| `figures/03_selected_lowell_pulls_v0_8.png` | selected low-ell pulls |
| `figures/04_public_source_pulls_v0_8.png` | compressed public-source pulls |

## Current call

Green diagnostics: exact scalar tilt target, low-ell TT improvement, acoustic-region preservation, compressed Omega_m/sigma8/S8 checks.

Amber diagnostics: TT+TE+EE low-ell combined proxy improves, but EE alone worsens; parity helps odd/even structure but needs map-space covariance tests.

Open gates: official Planck likelihood, masked anomaly MC, DESI BAO geometry, and finite-patch derivation of q_IR, ell_IR, epsilon_P, and ell_P.
