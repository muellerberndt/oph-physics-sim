# OPH-CMB Diagnostics v0.1

This is the first numerical diagnostic package for the OPH screen/freezeout CMB program.
It implements the current v0.1 scalar screen model, generates the requested plot families, and exports an isotropic primordial curvature table for external Boltzmann transfer.

## Boundary

This package does **not** claim a physical CMB fit yet. It computes:

```text
C_l^chi screen covariance
  -> D_l^chi tilt / low-l / parity diagnostics
  -> isotropic F_OPH(k) bridge using ell ~= k D_*
  -> P_R^OPH(k) export table
```

It does not compute `TT`, `TE`, `EE`, `BB`, or `C_l^phiphi`; CAMB and CLASS were not installed in this runtime.

## Model

```math
C_\ell^\chi =
\frac{A_\chi}{[\ell(\ell+1)+\mu^2]^{1+\eta_R/2}}
W_\ell^2 F_{IR}(\ell)F_P(\ell)+N_\ell^{frz}.
```

```math
D_\ell^\chi=\frac{\ell(\ell+1)}{2\pi}C_\ell^\chi,
\qquad n_s=1-\eta_R.
```

The external scalar export uses `eta_R=0.035`, so `n_s=0.965`, with `q_IR=0.15`, `ell_IR=6`, `A_s=2.1e-9`, `k0=0.05 Mpc^-1`, and `D_star=13,800 Mpc`.

## Diagnostic 1 — screen tilt

Generated files:

- `01_screen_Dell_tilt_etaR.png`
- `diagnostic_1_screen_tilt_Dell.csv`
- `screen_tilt_fit_summary_v0_1.csv`

The high-`ell` fitted slopes agree with the expected `D_l ~ l^(-eta_R)` behavior.

| eta_R | D2/D30 | D100/D30 | D1000/D30 | D2500/D30 |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | 1 |
| 0.035 | 1.09227 | 0.95912 | 0.884995 | 0.857072 |
| 0.06 | 1.16335 | 0.930947 | 0.811038 | 0.767667 |

Fit summary:

| eta_R | ell_min | ell_max | measured slope | expected approx |
| --- | --- | --- | --- | --- |
| 0 | 30 | 1000 | -3.15121e-11 | -0 |
| 0.035 | 30 | 1000 | -0.0349106 | -0.035 |
| 0.06 | 30 | 1000 | -0.0598467 | -0.06 |

## Diagnostic 2 — low-l global repair suppression

Generated files:

- `02_low_l_IR_suppression.png`
- `diagnostic_2_low_l_IR_suppression.csv`
- `low_l_suppression_summary_v0_1.csv`

For the exported reference case `q_IR=0.15`, `ell_IR=6`:

| q_IR | ell_IR | ell | F_IR |
| --- | --- | --- | --- |
| 0.15 | 6 | 2 | 0.869968 |
| 0.15 | 6 | 3 | 0.887278 |
| 0.15 | 6 | 4 | 0.906828 |
| 0.15 | 6 | 6 | 0.944818 |
| 0.15 | 6 | 10 | 0.989069 |
| 0.15 | 6 | 20 | 0.999993 |
| 0.15 | 6 | 40 | 1 |
| 0.15 | 6 | 80 | 1 |

## Diagnostic 3 — parity envelope

Generated files:

- `03_parity_envelope.png`
- `diagnostic_3_parity_envelope.csv`
- `parity_ratio_summary_v0_1.csv`

The ratio below is

```math
R_{OE} = \frac{\sum_{\ell\,\mathrm{odd}}\ell(\ell+1)C_\ell}{\sum_{\ell\,\mathrm{even}}\ell(\ell+1)C_\ell},\quad 2\le\ell\le30.
```

Positive `epsilon_P` boosts even multipoles and suppresses odd multipoles under this sign convention; negative `epsilon_P` does the reverse.

| epsilon_P | ell_parity | R_OE, ell 2..30 |
| --- | --- | --- |
| -0.1 | 30 | 1.05388 |
| -0.05 | 30 | 0.991007 |
| 0 | 30 | 0.932007 |
| 0.05 | 30 | 0.87653 |
| 0.1 | 30 | 0.82427 |

## Diagnostic 4 — isotropic primordial bridge

Generated files:

- `04_primordial_Fk_correction.png`
- `05_primordial_power_export.png`
- `diagnostic_4_primordial_bridge.csv`
- `oph_primordial_power_CLASS_v0_1.txt`
- `oph_primordial_power_baseline_etaR0035.txt`

Selected bridge samples:

| ell equiv | k [Mpc^-1] | F_OPH | P_R_OPH |
| --- | --- | --- | --- |
| 2 | 0.000144928 | 0.869968 | 2.24154e-09 |
| 3 | 0.000217391 | 0.887278 | 2.25393e-09 |
| 4 | 0.000289855 | 0.906828 | 2.28051e-09 |
| 6 | 0.000434783 | 0.944818 | 2.34257e-09 |
| 10 | 0.000724638 | 0.989069 | 2.40883e-09 |
| 20 | 0.00144928 | 0.999993 | 2.37706e-09 |
| 30 | 0.00217391 | 1 | 2.34358e-09 |
| 1000 | 0.0724638 | 1 | 2.0729e-09 |

Parity and BipoSH corrections are intentionally **not** included in `oph_primordial_power_CLASS_v0_1.txt`. They belong in angular covariance / `a_lm` simulations, not in a scalar isotropic `P_R(k)` table.

## External transfer handoff

- `class_external_primordial_template.ini` is a CLASS starting template.
- `camb_external_primordial_note.py` is a CAMB handoff sketch.

The next real physics step is to run this table through CLASS/CAMB externally and compare the resulting `TT/TE/EE/lensing` curves against a baseline tilted spectrum.
