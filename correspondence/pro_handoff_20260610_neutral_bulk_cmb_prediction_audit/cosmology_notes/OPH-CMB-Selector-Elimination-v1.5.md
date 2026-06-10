# OPH-CMB Selector Elimination v1.5

## Result

This package removes the remaining CMB selectors as far as the current OPH theorem surface honestly allows.

| Old selector | v1.5 result |
|---|---|
| S2: `q_IR=1/4` | Removed as selector. Replaced by the affine-zero-mode quarter-reserve theorem on `S^2`: one protected checkpoint scalar in `ell=0` plus three dipole/global-repair modes in `ell=1`, so `q_IR=1/(1+3)=1/4`. |
| S3: `ell_IR=32` | Removed as selector. Replaced by finite visible-covariance rank: dodecahedral visible scalar channels `F+V=12+20=32`; adding identity gives 33; matching `(L+1)^2=33^2` gives `L=32`. |
| S1: `eta_R=e alpha sqrt(pi)` | Not left as a free selector, but reduced to one repair-clock certificate: `eta_R=kappa_rep alpha sqrt(pi)`. The exact branch needs `kappa_rep=e`. |

## Current exact IR kernel

```math
F_IR^OPH(ell)=1-(1/4) exp[-ell(ell+1)/(32*33)].
```

Selected values are in `data/exact_ir_kernel_values_v1_5.csv`.

## Tilt branch

Using `alpha^-1(0)=137.035999177`, the canonical `kappa_rep=e` branch gives:

```math
eta_R = 0.035158856969
n_s   = 0.964841143031
```

But the theorem note keeps the honest version visible:

```math
eta_R = kappa_rep alpha sqrt(pi),
```

where the final finite-patch task is to compute/certify `kappa_rep=e` from the scalar repair semigroup.

## Files

- `math/OPH-CMB-Selector-Elimination-v1.5.md`
- `data/selector_elimination_status_v1_5.csv`
- `data/exact_ir_kernel_values_v1_5.csv`
- `data/selector_elimination_summary_v1_5.json`
