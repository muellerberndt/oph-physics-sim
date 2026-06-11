# OPH-CMB Prediction Ledger v0.1

**Project:** Observer Patch Holography → CMB prediction program  
**Working title:** *The CMB as an Observer-Consensus Fossil*  
**Version:** v0.1  
**Date:** 2026-06-08  
**Status:** effective-theory ledger, not a completed Planck likelihood fit

---

## 0. Claim boundary

The v0.1 claim is:

> OPH supplies a screen/freezeout effective theory for primordial observer-consensus statistics. Those statistics are mapped into a primordial curvature spectrum and angular covariance, then propagated through ordinary Boltzmann transfer to produce CMB spectra.

The v0.1 claim is **not**:

> OPH has already derived the full observed CMB from the current finite-patch/FPE simulator.

Use the staged map:

```text
finite observer-consensus on S2
  -> freezeout / screen synchronization field chi(n)
  -> C_l^chi and angular covariance
  -> P_R^OPH(k) plus angular covariance corrections
  -> CAMB / CLASS transfer
  -> TT, TE, EE, BB, phi-phi, P(k), BAO/growth checks
```

This keeps the analytic screen result separate from photon-baryon transfer, likelihood fitting, and the OPH dark-sector Boltzmann problem.

---

## 1. The CMB properties OPH should predict

### 1.1 Primary CMB observables

| Observable | Meaning | OPH route | v0.1 status |
|---|---|---|---|
| `C_l^TT` | temperature anisotropy spectrum | `P_R^OPH(k)` + photon-baryon transfer | Boltzmann bridge required |
| `C_l^TE` | temperature-E polarization cross-spectrum | same scalar initial field + Thomson scattering | Boltzmann bridge required |
| `C_l^EE` | E-mode polarization | same scalar initial field + reionization/recombination | Boltzmann bridge required |
| `C_l^BB,lens` | lensing B-modes | scalar spectra + lensing potential | Boltzmann + lensing required |
| `C_l^BB,tensor` | primordial tensor B-modes | OPH tensor/edge-repair channel, if derived | not yet derived; v0.1 null/default |
| `C_l^phiphi` | CMB lensing potential | matter growth + OPH anomaly-stress sector | custom Boltzmann module required |
| `C_l^Tphi`, `C_l^Ephi` | ISW/lensing cross-correlations | late-time metric response | custom Boltzmann module required |

### 1.2 Primordial/statistical observables

| Observable | Target role | v0.1 OPH prediction/parameterization |
|---|---|---|
| `A_s` | scalar amplitude | bridge normalization; not yet first-principles in this ledger |
| `n_s` | scalar tilt | `n_s = 1 - eta_R` |
| `alpha_s = dn_s/dlnk` | running | zero at baseline; nonzero if repair dimension runs |
| `beta_s` | running of running | zero at baseline |
| `r` | tensor-to-scalar ratio | zero/default unless OPH tensor screen mode is derived |
| `f_NL` | bispectrum amplitude | Gaussian MaxEnt baseline gives near-zero; repair defects can add low-l/localized non-Gaussianity |
| isocurvature | non-adiabatic initial modes | absent at baseline; possible if independent sector-class freezeout field exists |
| parity envelope | odd/even power modulation | low-l smooth envelope from cap-pair/orientation repair |
| BipoSH coefficients | off-diagonal covariance | structured low-L coefficients from global repair/cap holonomy |

### 1.3 Compressed cosmological observables

| Observable | Meaning | Role in OPH-CMB program |
|---|---|---|
| `100 theta_*` | acoustic angular scale | background geometry hard target |
| `Omega_b h^2` | baryon density | standard recombination input; OPH particle branch may eventually constrain |
| `Omega_A h^2` / `Omega_c h^2` | anomaly/dark component density | OPH dark-sector Boltzmann target |
| `H_0` | expansion rate today | geometry + BAO/SNe cross-check |
| `Omega_m`, `Omega_Lambda`, `Omega_K` | late geometry | must remain consistent with CMB + BAO |
| `tau` | reionization optical depth | astrophysical/nuisance input unless OPH reionization model added |
| `N_eff^rel` | effective relativistic species | reserved cosmology symbol; do **not** use for OPH freezeout capacity |
| `sum m_nu` | neutrino mass sum | OPH particle/neutrino branch input/check |
| `sigma_8`, `S_8` | growth amplitude | OPH anomaly-stress growth target |

### 1.4 Large-angle anomaly observables

| Statistic | What it tests | OPH interpretation |
|---|---|---|
| `C_2`, `C_3` | quadrupole/octopole power | global repair-mode suppression |
| `S_1/2` | large-angle correlation deficit | finite-capacity/global repair covariance |
| odd/even parity ratio `R_OE` | parity asymmetry | cap-pair/orientation synchronization defect |
| quadrupole-octopole alignment | preferred large-angle axis | low-rank repair covariance / holonomy |
| hemispherical asymmetry | dipolar modulation | low-L BipoSH repair mode |
| BipoSH `A_ll'^{LM}` | statistical isotropy violation | structured off-diagonal covariance, not arbitrary mixing |

---

## 2. Core OPH screen field

Define the freezeout observer-consensus scalar on the last observable screen:

```math
chi(n) = freezeout synchronization / repair-load / record-phase field on S^2.
```

Expand it in spherical harmonics:

```math
chi(n) = sum_{lm} a_{lm}^chi Y_{lm}(n),
```

```math
C_l^chi = (1/(2l+1)) sum_m |a_{lm}^chi|^2.
```

The first target is not the physical CMB spectrum directly. The first target is the screen covariance:

```math
C_l^chi -> P_R^OPH(k) -> C_l^{TT}, C_l^{TE}, C_l^{EE}, C_l^{phiphi}.
```

---

## 3. Analytic theorem target: MaxEnt screen scale invariance

### 3.1 Repair-cost action

Take the maximum-entropy field on `S^2` subject to a local smoothness / repair-cost constraint:

```math
S[chi] = (1/(2 sigma^2)) int_{S^2} dOmega |grad_{S^2} chi|^2.
```

Because

```math
-grad_{S^2}^2 Y_lm = l(l+1) Y_lm,
```

the action diagonalizes as

```math
S[chi] = (1/(2 sigma^2)) sum_{l>=1,m} l(l+1) |a_lm^chi|^2.
```

Therefore the Gaussian covariance is

```math
<a_lm^chi a_l'm'^chi*> = A_chi/[l(l+1)] delta_ll' delta_mm',   l >= 1.
```

So

```math
D_l^chi = l(l+1) C_l^chi / (2 pi) ≈ constant.
```

**Prediction P0:** before photon-baryon transfer, the OPH MaxEnt repair field has a scale-invariant angular spectrum.

### 3.2 Corrected red-tilt convention

Use a positive red anomalous dimension

```math
eta_R := 1 - n_s > 0.
```

Then write

```math
C_l^chi = A_chi / [l(l+1)+mu^2]^{1 + eta_R/2}.
```

At large `l`, this gives

```math
D_l^chi ∝ l^{-eta_R},
```

so the primordial bridge has

```math
P_R^OPH(k) ∝ k^{-eta_R},
```

and therefore

```math
n_s = 1 - eta_R.
```

**Prediction P1:** OPH must derive

```math
eta_R ≈ 0.035 ± 0.004
```

if it is to match the Planck scalar tilt benchmark `n_s ≈ 0.965 ± 0.004`.

---

## 4. Minimal OPH screen covariance model

The v0.1 working model is:

```math
C_l^chi =
  A_chi
  / [l(l+1)+mu^2]^{1 + eta_R/2}
  * W_l^2
  * F_IR(l)
  * F_parity(l)
  + N_l^frz.
```

where

```math
W_l = exp[-l(l+1)/(2 l_cap^2)],
```

```math
F_IR(l) = 1 - q_IR exp[-l(l+1)/(l_IR(l_IR+1))],
```

```math
F_parity(l) = 1 + epsilon_P (-1)^l exp[-l/l_parity].
```

Use `l_parity`, not `l_P`, to avoid confusion with Planck length.

Finite freezeout sampling/noise is represented as

```math
N_l^frz = sigma_frz^2 / N_frz * W_l,frz.
```

Important notation rule:

```text
N_eff^rel = cosmological effective relativistic species.
N_frz or N_cap^frz = OPH effective freezeout capacity.
```

Do not call OPH freezeout capacity `N_eff`.

---

## 5. Parameter ledger

| Symbol | OPH meaning | Expected CMB imprint | Source/status |
|---|---|---|---|
| `A_chi` | screen synchronization amplitude | fixes `A_s` after bridge normalization | bridge parameter |
| `eta_R` | positive red repair anomalous dimension | scalar tilt `n_s=1-eta_R` | must be derived; target ≈ 0.035 |
| `mu` | IR repair/capacity regulator | low-l rollover/suppression | simulator or low-l fit |
| `l_cap` | finite cell/window scale | high-l smoothing/cutoff in screen field | likely very high or unobservable unless effective freezeout scale small |
| `N_frz` | effective freezeout capacity | sampling floor, extra variance, low-l covariance | simulator-derived |
| `q_IR` | global repair suppression strength | suppressed quadrupole/octopole/large-angle power | fit/test; then derive |
| `l_IR` | angular width of global repair mode | which low multipoles are affected | fit/test; then derive |
| `epsilon_P` | parity/cap-pair residual amplitude | odd/even low-l asymmetry | fit/test; then derive |
| `l_parity` | decay scale of parity envelope | parity localized to low-l if OPH-like | fit/test; then derive |
| `A_ll'^{LM}` | BipoSH off-diagonal covariance | preferred axes, hemispherical asymmetry, alignments | map/MC analysis |
| `g_defect` | rare repair-defect strength | bispectrum/trispectrum/non-Gaussianity | simulation-derived |
| `B_A(k,a)` | anomaly-stress response kernel | lensing/growth/ISW/CMB peaks through dark sector | custom Boltzmann module |
| `Gamma_rec(k,a)` | repair relaxation rate | cluster offsets, dark stress lag, growth | custom Boltzmann module |

---

## 6. OPH-to-primordial bridge

The scalar bridge is:

```math
P_R^OPH(k) = A_s (k/k0)^{-eta_R} F_OPH(k),
```

with

```math
F_OPH(k) = F_IR(k) F_cap(k) F_defect(k).
```

A first mapping uses

```math
l ≈ k D_*,
```

where `D_*` is the comoving distance to last scattering. Then

```math
F_IR(k) = 1 - q_IR exp[-(k/k_IR)^2],
```

```math
F_cap(k) = exp[-(k/k_cap)^2].
```

Caution: parity and BipoSH effects are primarily angular covariance effects. They should not be forced into an isotropic scalar `P_R(k)` unless the angular structure has first been averaged away.

Full angular covariance:

```math
<a_lm a_l'm'^*> = C_l delta_ll' delta_mm' + Delta_lm,l'm'^OPH.
```

BipoSH representation:

```math
Delta_lm,l'm'^OPH = sum_{LM} A_ll'^{LM} (-1)^{m'} C_{l m, l' -m'}^{LM}.
```

---

## 7. First concrete OPH-CMB predictions

### P0 — MaxEnt screen scale invariance

```math
D_l^chi ≈ constant
```

before photon-baryon transfer.

**Failure mode:** finite patch simulations do not approach inverse-Laplacian covariance.

### P1 — red scalar tilt from repair anomalous dimension

```math
n_s = 1 - eta_R.
```

Benchmark target:

```math
eta_R ≈ 0.035 ± 0.004.
```

**Failure mode:** OPH derives the wrong sign or a value far from the observed tilt.

### P2 — low-l suppression from global repair modes

```math
C_l = C_l^(0) [1 - q_IR exp(-l(l+1)/(l_IR(l_IR+1)))].
```

Expected imprint: concentrated suppression in `l=2,3,4,...`, with high-l spectra mostly unchanged.

**Failure mode:** suppression is not localized, or parameters fitted in TT do not predict TE/EE directionally.

### P3 — parity asymmetry as a smooth low-l envelope

```math
C_l = C_l^(0) [1 + epsilon_P (-1)^l exp(-l/l_parity)].
```

Prediction: if OPH parity exists, it is coherent and low-l localized, not random high-l alternation.

**Failure mode:** parity preference disappears under masks/component-separated maps or is not envelope-like.

### P4 — structured off-diagonal covariance

```math
A_ll'^{LM} != 0
```

mainly at low `l,l'` and small `L`.

Prediction: preferred-axis/anomaly structure should be low-rank or low-L, not arbitrary covariance.

**Failure mode:** BipoSH coefficients are noise/systematics dominated or not reproducible across maps.

### P5 — Gaussian baseline with defect non-Gaussianity as correction

MaxEnt screen field is Gaussian at leading order. Non-Gaussianity enters through rare repair defects or nonlinear freezeout.

Prediction:

```math
f_NL ≈ 0 baseline,
```

with possible low-l/localized defect signatures.

**Failure mode:** large non-Gaussianity is required to fit the data without a simulator-derived defect law.

### P6 — acoustic peaks are transfer physics

OPH does not need to produce acoustic peaks directly from finite patch settling. The correct separation is:

```text
OPH screen field -> primordial curvature/statistical field -> photon-baryon transfer -> acoustic peaks.
```

**Failure mode:** OPH initial conditions cannot pass through CAMB/CLASS while preserving Planck-quality TT/TE/EE peak structure.

### P7 — tensor/B-mode default is null until tensor repair is derived

At v0.1, there is no derived OPH tensor freezeout channel. Therefore the default tensor prediction is:

```math
r_OPH = 0
```

or, more carefully:

```text
no primordial tensor prediction yet; any nonzero r requires an OPH tensor/edge-repair derivation.
```

**Failure mode:** a robust primordial tensor detection appears and no OPH tensor channel can be derived.

### P8 — lensing/growth require anomaly-stress Boltzmann closure

CMB lensing and matter growth require the OPH dark-sector stress variables:

```math
rho_A(a), w_A(a), c_s,A^2(k,a), sigma_A(k,a), Q_A^mu, B_A(k,a), Gamma_rec(k,a).
```

**Failure mode:** the anomaly-stress kernel fits galaxies but fails CMB lensing, BAO, weak lensing, or growth under full covariance likelihoods.

---

## 8. Benchmark data targets for v0.1/v0.2

Use these as target checks, not as OPH-derived values yet.

| Quantity | Benchmark | Use |
|---|---:|---|
| `n_s` | `0.965 ± 0.004` | sets `eta_R ≈ 0.035` target |
| `Omega_b h^2` | `0.0224 ± 0.0001` | recombination/acoustic target |
| `Omega_c h^2` | `0.120 ± 0.001` | cold dark/anomaly density target |
| `tau` | `0.054 ± 0.007` | low-l EE/reionization target |
| `100 theta_*` | `1.0411 ± 0.0003` | acoustic-scale target |
| `H_0` | `67.4 ± 0.5 km/s/Mpc` | background target |
| `Omega_m` | `0.315 ± 0.007` | background/growth target |
| `sigma_8` | `0.811 ± 0.006` | growth target |
| `N_eff^rel` | `2.99 ± 0.17` | relativistic species; distinct from `N_frz` |
| `sum m_nu` | `< 0.12 eV` in Planck+BAO baseline | neutrino branch check |
| ACT+Planck lensing `S_8` | `0.831 ± 0.023` | CMB lensing/growth check |
| tensor ratio `r_0.05` | `< 0.036` at 95% CL | B-mode/tensor null target |

DESI DR2 BAO is a cross-check, not merely an optional external comparison: the BAO distances must remain consistent with the same background and anomaly-stress model used for CMB. DESI DR2 reports mild BAO-CMB tension in flat Lambda-CDM and preference for evolving dark energy in some data combinations; OPH must not hide from that geometry test.

---

## 9. Simulation and likelihood ladder

| Tier | Task | Output | Claim status |
|---|---|---|---|
| A | analytic screen theory | `C_l^chi`, tilt, finite-capacity, parity, BipoSH forms | theorem/effective theory |
| B | primordial bridge | `P_R^OPH(k)`, optional angular covariance | effective bridge |
| C | CAMB/CLASS transfer | `TT`, `TE`, `EE`, `BB_lens`, `phi-phi`, `P(k)` | numerical prediction |
| D | low-l Monte Carlo | `S_1/2`, parity, alignment, BipoSH significance | data-analysis result |
| E | finite patch simulator | derive `eta_R`, `mu`, `q_IR`, `epsilon_P`, `N_frz`, `g_defect` | OPH derivation target |
| F | anomaly-stress Boltzmann module | dark/anomaly CMB+lensing+growth likelihoods | publication-grade cosmology |

---

## 10. Minimal code modules

Create these modules first:

```text
oph_screen.py
  cl_oph(l, params)
  d_l(l, cl)
  parity_ratio(cl)

primordial_bridge.py
  F_oph_k(k, params)
  primordial_power_oph(k, A_s, eta_R, k0, F)
  write_class_or_camb_table(k, P_R)

low_l_anomalies.py
  S_1_2(C_theta, theta_grid)
  parity_stat(cl, l_min=2, l_max=30)
  quadrupole_octopole_alignment(alm)
  biposh_estimator(alm)

monte_carlo.py
  sample_alm_from_cl(cl)
  apply_mask(map)
  run_low_l_mc(model, n_sims)

boltzmann_oph_dark.py
  rho_A(a)
  w_A(a)
  cs2_A(k,a)
  sigma_A(k,a)
  B_A(k,a)
  Gamma_rec(k,a)
```

---

## 11. Falsifiers

The clean falsifiers are:

1. **Screen covariance falsifier:** finite observer-consensus simulations do not approach the inverse-Laplacian screen covariance.
2. **Tilt falsifier:** the derived repair anomalous dimension is not near `eta_R ≈ 0.035`, or has the wrong sign.
3. **Transfer falsifier:** OPH-screen initial conditions cannot reproduce TT/TE/EE acoustic structure under standard transfer.
4. **Low-l falsifier:** low-l suppression/parity parameters fitted in TT do not predict TE/EE or map-level anomaly statistics.
5. **Covariance falsifier:** OPH off-diagonal covariance is not low-rank/low-L or fails component-separated map robustness.
6. **Dark-sector falsifier:** anomaly-stress kernels can fit galaxies only by breaking CMB lensing, BAO, SNe, RSD, or weak-lensing likelihoods.
7. **Tensor falsifier:** primordial tensors are detected at a level incompatible with any derived OPH tensor/edge-repair channel.

---

## 12. Paper sequence

### Paper I — analytic screen theory

```text
The CMB as an Observer-Consensus Fossil I:
Screen MaxEnt, Scale Invariance, and Finite-Capacity Corrections
```

Sections:

1. OPH screen/cap setup.
2. Freezeout consensus field `chi(n)`.
3. MaxEnt repair action.
4. Derivation `C_l ∝ 1/[l(l+1)]`.
5. Red tilt as `eta_R`.
6. Finite capacity and low-l repair corrections.
7. Parity and BipoSH covariance.
8. Falsifiable predictions.

### Paper II — CMB data bridge

```text
The CMB as an Observer-Consensus Fossil II:
Boltzmann Transfer and Low-l Anomaly Tests
```

Sections:

1. OPH primordial bridge.
2. CAMB/CLASS implementation.
3. Planck TT/TE/EE comparison.
4. Low-l anomaly Monte Carlo.
5. ACT lensing / DESI / growth cross-checks.
6. Falsifiers and claim boundary.

### Paper III — simulator derivation

```text
The CMB as an Observer-Consensus Fossil III:
Finite Patch Simulations of Freezeout Screen Covariance
```

Sections:

1. OPH-FPE freezeout runs.
2. Extracted repair/synchronization fields.
3. Estimated `eta_R`, `q_IR`, `epsilon_P`, `N_frz`, `g_defect`.
4. Comparison to analytic theory.
5. Bridge to physical spectra.

---

## 13. Immediate next step

Implement the v0.1 scalar screen model and produce three diagnostic plots:

1. `D_l^chi` for `eta_R = 0, 0.035, 0.06`.
2. Low-l suppression envelope for several `q_IR, l_IR`.
3. Parity envelope for several `epsilon_P, l_parity`.

Then export `P_R^OPH(k)` as a custom primordial power table for CLASS/CAMB.
