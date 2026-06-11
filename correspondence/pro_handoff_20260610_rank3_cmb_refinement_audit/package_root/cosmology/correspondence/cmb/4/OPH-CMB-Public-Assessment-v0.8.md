# OPH-CMB Public Assessment v0.8

Generated: 2026-06-08

This package updates the assessment table by promoting the **v0.6 exact full-sky TT + high-l guardrail** values to the current OPH-CMB target. It preserves the older q_IR=0.2446 CAMB-transferred diagnostic as a historical/superseded check, because the newer exact guardrail preferred a milder IR kernel.

## Current numerical target

| quantity | value |
|---|---:|
| eta_R = 1 - n_s | 0.035 |
| current guardrailed q_IR | 0.117448347 |
| current guardrailed ell_IR | 29.287395 |
| theta_IR = 180/ell_IR | 6.145989 deg |
| k_IR = ell_IR / 13,800 Mpc | 0.00212227 Mpc^-1 |
| N_frz angular proxy = (ell_IR+1)^2 | 917.326 modes |
| joint q_IR | 0.113078954 |
| joint ell_IR | 29.738682 |
| joint epsilon_P | -0.980 |
| joint ell_P | 4.620776 |

## Exact guardrail comparison

| model | improvement vs LCDM | low-ell exact gain | high-l penalty | delta AIC | delta BIC |
|---|---:|---:|---:|---:|---:|
| reference q=0.15, ell=6 | 0.185 | 0.185 | -0.000 | -0.185 | -0.185 |
| current IR guardrail | 3.012 | 3.600 | 0.588 | 0.988 | 11.167 |
| current IR+parity guardrail | 9.096 | 9.721 | 0.625 | -1.096 | 19.261 |
| superseded q=0.2446, ell=33.615 | -3.782 | -0.731 | 3.050 | 7.782 | 17.960 |

## Assessment table

See:

- `data/01_public_assessment_table_v0_8.csv`
- `OPH-CMB-Public-Assessment-v0.8.xlsx`

## Claim boundary

This is a public-source numerical assessment and diagnostic likelihood ladder. It is **not** the official Planck likelihood, **not** a masked-sky component-separated map analysis, and **not** a finite-patch derivation of q_IR, ell_IR, epsilon_P, or eta_R. The next success gate is map-space Planck SMICA/NILC/Commander/SEVEM analysis plus a simulator-derived freezeout prior.
