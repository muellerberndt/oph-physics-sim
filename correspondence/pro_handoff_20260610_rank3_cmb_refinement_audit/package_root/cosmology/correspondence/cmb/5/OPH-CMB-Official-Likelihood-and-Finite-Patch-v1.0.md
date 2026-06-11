# OPH-CMB Official Likelihood and Finite-Patch Derivations v1.0

**Date:** 2026-06-08  
**Scope:** official Planck-likelihood execution handoff, CAMB OPH transfer smoke test, finite-patch CMB theorem ledger, and missing-math gate list.

## 1. Executive status

This package finishes the work that can be completed inside the current runtime and turns the remaining official Planck likelihood step into a direct reproducibility handoff.

| Gate | Status | Result |
|---|---|---|
| Official Planck PR3 likelihood | **Integration complete; official execution pending external data/API** | Configs and direct evaluator written. Runtime lacks the official Planck `clik`/`clipy` likelihood files and ESA PR3 likelihood data. |
| CAMB transfer of OPH scalar kernel | **Smoke executed** | OPH exact scalar branch generated CAMB spectra through `lmax=80`; `TT_Dl_ell2 = 800.447633 uK^2`. |
| MaxEnt finite-patch screen covariance | **Closed analytically** | Repair-cost MaxEnt covariance is the graph Green function; S2 limit gives `C_l proportional to 1/[l(l+1)]`. |
| Tilt target | **Closed as quantitative target; repair-flow proof still conditional** | `eta_R = e alpha(0) sqrt(pi) = 0.035158856969`; `n_s = 0.964841143031`. |
| Exact IR branch | **Closed as finite-patch closure hypothesis** | `q_IR = 1/4`, `ell_IR = 32`, `theta_IR = 5.625 deg`, `N_frz = 1089`. |
| Parity/BipoSH covariance | **Formula closed; map-space likelihood pending** | Parity belongs in angular covariance, not scalar `P_R(k)`. |
| Dark-stress Boltzmann module | **Not closed** | Interface variables identified; full anomaly-fluid equations still required. |

## 2. Official Planck likelihood handoff

The official baseline comparison requires the ESA Planck PR3 likelihood code/data package and the official clik-compatible files. This runtime has CAMB and Cobaya, but not a usable official Planck clik API and not the ESA PR3 likelihood data directories.

Package components:

| File | Purpose |
|---|---|
| `configs/README_official_planck_install.md` | Install and run guide for official Planck 2018 PR3 likelihoods. |
| `configs/cobaya_planck2018_official_lcdm.yaml` | Baseline official Planck LCDM Cobaya configuration with clik components. |
| `configs/oph_planck2018_clik_paths.template.yaml` | Fill-in template for Commander, SimAll, Plik, and lensing clik file paths. |
| `scripts/oph_official_planck_clik_eval.py` | Direct Planck clik evaluator for CAMB-generated LCDM/OPH spectra. |
| `scripts/oph_cobaya_likelihood_selfcontained.py` | Cobaya likelihood class wrapping CAMB + OPH primordial kernel + clik. |
| `scripts/run_official_planck_pipeline.sh` | One-command skeleton for generating spectra and evaluating official likelihoods. |

Expected official data files after installation:

```text
plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik
plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik
plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik
plc_3.0/lensing/smica_g30_ftl_full_pp.clik_lensing
```

The official execution command is:

```bash
cd oph_cmb_official_v1_0
bash scripts/run_official_planck_pipeline.sh /path/to/oph_planck2018_clik_paths.yaml outputs/full_official
```

## 3. CAMB OPH scalar smoke result

The OPH scalar branch was propagated through CAMB at low resolution as a runtime smoke test. It used:

```text
H0 = 67.4
ombh2 = 0.0224
omch2 = 0.120
tau = 0.054
As = 2.1e-9
ns = 0.964841143031
q_IR = 0.25
ell_IR = 32
lmax = 80
```

Output summary:

```text
TT_Dl_ell2  = 800.4476330187244 uK^2
TT_Dl_ell30 = 1022.4988361946814 uK^2
```

This is a smoke test only. The full official run should use `lmax >= 2508`, full lensing settings, and official foreground/nuisance handling.

## 4. Finite-patch theorem ledger

The math note `math/finite_patch_cmb_derivations_v1_0.md` closes the following statements.

### 4.1 Normal-form screen field

For a finite patch quotient `Q` with a terminating and confluent repair relation, the normal-form map

```math
N:Q -> Q_nf
```

is schedule-independent. Any screen observable `chi(q)` therefore has a unique freezeout value `chi(N(q))`, and any initial ensemble has a unique pushforward distribution over `chi`.

### 4.2 MaxEnt inverse-Laplacian covariance

On a finite graph with weighted Laplacian `L`, the maximum-entropy mean-zero Gaussian ensemble constrained by repair cost

```math
E[chi^T L chi] = K
```

has precision proportional to `L` and covariance proportional to `L^+`. For a rotationally convergent `S^2` refinement,

```math
C_l^chi = A/[l(l+1)],
D_l^chi = l(l+1) C_l^chi/(2 pi) = constant.
```

### 4.3 Tilt/anomalous repair dimension

The fractional repair kernel

```math
K_eta = (-Delta_{S2}+mu^2)^(1+eta_R/2)
```

emits

```math
C_l^chi = A/[l(l+1)+mu^2]^(1+eta_R/2),
D_l^chi proportional to l^(-eta_R),
n_s = 1 - eta_R.
```

Current OPH target:

```math
eta_R = e alpha(0) sqrt(pi) = 0.035158856969,
n_s = 0.964841143031.
```

### 4.4 Exact scalar IR branch

Current exact finite-patch target:

```math
q_IR = 1/4,
ell_IR = 32,
theta_IR = 180/32 = 5.625 deg,
k_IR = ell_IR / 13800 Mpc = 2.31884e-3 Mpc^-1,
N_frz = (ell_IR+1)^2 = 1089.
```

Raw scalar kernel:

```math
F_IR(l) = 1 - (1/4) exp[-l(l+1)/(32*33)].
```

Selected values:

```text
F_IR(2)  = 0.751416427
F_IR(3)  = 0.752824829
F_IR(10) = 0.774731224
F_IR(32) = 0.908030140
```

### 4.5 Parity and off-diagonal covariance

The parity term is

```math
C_l^P = C_l^(0) [1 + epsilon_P (-1)^l exp(-l/ell_P)].
```

This is an angular covariance correction. It should not be folded into scalar `P_R(k)` except as a diagnostic toy model.

The off-diagonal repair covariance is represented by

```math
< a_lm a_l'm'^* > = C_l delta_ll' delta_mm' + Delta^repair_{lm,l'm'}.
```

The preferred observational language is BipoSH coefficients.

## 5. Missing math and execution gates

See `data/missing_math_and_likelihood_gates_v1_0.csv` for the complete gate table.

The remaining nontrivial gates are:

1. prove `eta_R = e alpha sqrt(pi)` as the repair-flow anomalous dimension;
2. derive `q_IR = 1/4` from active global repair-channel counting;
3. derive `ell_IR = 32` from a 33-level finite freezeout packet closure;
4. derive the parity envelope from cap-pair/orientation residuals;
5. run official Planck likelihood on a machine with the official clik/clipy data files;
6. run masked Planck component-separated map Monte Carlo for `S_1/2`, parity, quadrupole-octopole alignment, hemispherical asymmetry, and BipoSH;
7. write the OPH dark-stress Boltzmann module.

## 6. Claim boundary

The completed claim is:

```text
OPH-CMB has a closed screen/freezeout effective-theory derivation, an exact finite-patch target ledger, CAMB scalar-transfer code, and official Planck likelihood integration scaffolding.
```

The incomplete claim is:

```text
OPH has not yet executed the official Planck likelihood in this runtime, has not yet run masked Planck map-space anomaly likelihoods, and has not yet derived every finite-patch numerical constant from an OPH-FPE simulator without fitting.
```
