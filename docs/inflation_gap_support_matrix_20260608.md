# Inflation/CMB Implementation Gap Support Matrix - 2026-06-08

This note maps the inflation/CMB researcher gap list in
`cosmology/correspondence/inflation/1`, `cosmology/correspondence/inflation/2`,
`cosmology/correspondence/cmb`, `cosmology/correspondence/neutrinos`, and
`cosmology/correspondence/capacity` onto the current `oph-physics-sim` codebase.

Claim boundary:

```text
The simulator can support receipt-bearing diagnostics for the inflation-replacement
stack, but it must not claim a physical CMB prediction or proven 3D bulk until the
finite lattice derives the relevant kernels/normal forms and the controls pass.
```

## Summary

| Gap | Simulator support status | Current code surface | Claim level now |
|---|---|---|---|
| Hot MaxEnt release instead of reheating | Supportable next | Add release-state readout over record/collar states | Not implemented |
| Adiabaticity and isocurvature suppression | Partly supportable now | `sync_inflation.py`, future species-record correlators | Theorem-target audit only |
| Low-k synchronization gap / same-boundary selector | Partly audited, not proved | `sync_inflation.py`, `collar_state.py`, `markov_collar.py` | Missing gate |
| Flatness as holonomy repair / zero-holonomy selector | Partly supported | `array_s3_holonomy.py`, `sync_inflation.py` | Diagnostic only |
| Scalar tilt from OPH unique prediction | Supported as target/import, not lattice-derived | `unique_predictions.py`, `cmb_derivation.py`, `camb_adapter.py` | Measurement-comparable target |
| Low-l IR/parity anomaly kernel | Supported as target/import, not lattice-derived | `unique_predictions.py`, `cmb_anomaly.py`, `camb_adapter.py` | Diagnostic target |
| Tensor/B-mode prediction | Supportable next | Need tensor screen harmonics and residual shear/holonomy readout | Not implemented |
| Non-Gaussianity from collar cumulants | Supportable next | Need 3-point/cumulant readouts over freezeout fields | Not implemented |
| Neutrino/CnuB lane | Supported as fixed target/readout | `unique_predictions.py`; extend CAMB massive-neutrino plumbing | Measurement-comparable target |
| Anomaly Boltzmann functions rho_A, B_A, Gamma_rec | Partly supported as diagnostics | `boltzmann_inputs.py`, `ba_parent.py`, `anomaly_fluid.py` | Not physical prediction |
| Global screen capacity N_scr closure | Not supported as proof | `P` local pixel exists; global capacity closure missing | Theory/selector gap |
| Baryogenesis | Not supported | Need Z6 CP phase, sphaleron, washout module | Open theorem target |
| No-trans-Planckian cutoff | Metadata supportable | `oph_pixel.py`, pixel reports | Claim audit only |
| Full likelihood contract | Partly supported | `camb_adapter.py`, compressed likelihood, comparable snapshot | Not official likelihood |

## Gaps We Can Support Directly

### Hot MaxEnt release

The notes replace inflaton reheating with a synchronized screen/collar release into
a hot Standard Model MaxEnt state:

```text
rho_rel = argmax S(rho) subject to realized SM branch, conserved charges,
anomaly load, and synchronized scalar screen record.
```

The simulator can support this as a new readout without pretending to simulate the
full SM plasma:

- detect a release surface where `Phi_sync <= epsilon_sync`;
- compute local/cap packet entropy, record entropy, and collar entropy;
- emit `T_rel_proxy`, `gstarS_assumption`, charge constraints, and release readiness;
- pass the result to the CMB bridge only as a proxy unless a physical unit mapping is
  supplied.

Recommended code surface:

```text
oph_fpe/cosmology/hot_release.py
tests/test_hot_release.py
```

### Adiabaticity and isocurvature suppression

The paper target is that all species inherit the same scalar screen displacement,
so entropy modes vanish:

```text
S_ij = delta_i/(1+w_i) - delta_j/(1+w_j) = 0
```

The finite simulator can support a proxy by assigning several observer-visible
species channels to the same record-family displacement and measuring residual
channel mismatch.

Recommended readouts:

- scalar displacement field `zeta_proxy`;
- species channels: baryon proxy, photon proxy, neutrino proxy, anomaly proxy;
- `S_ij_proxy_ell` spectra;
- decaying-mode sine/cosine phase ratio in the acoustic bridge.

Recommended code surface:

```text
oph_fpe/cosmology/adiabaticity.py
tests/test_adiabaticity.py
```

### Low-k synchronization gap

This is one of the real load-bearing gates. The current code audits repair decay,
collar CMI, and current finite reports, but it does not yet prove a nonzero low-k
gap over the CMB band.

Supportable simulator path:

- compute spherical harmonic modes of `Phi_sync`, record displacement, and repair
  load;
- fit mode-wise repair rates `Gamma_sigma(ell)`;
- report `inf_{ell in CMB band} Gamma_sigma(ell)`;
- run no-repair, shuffled-interface, shuffled-record, and random-cap controls.

Recommended code surface:

```text
oph_fpe/cosmology/sync_gap.py
tests/test_sync_gap.py
```

This should become the main finite-lattice route toward the horizon/coherence
claim.

### Tensor and non-Gaussianity

Both are supportable as finite-screen diagnostics:

- Tensor: residual spin-2/shear/holonomy field on the screen, with an `r_proxy`
  and B-mode handoff table.
- Non-Gaussianity: two-point and three-point collar cumulants for the freezeout
  scalar field, with local/equilateral/screen-shape proxy amplitudes.

These are not needed before the scalar/low-k gates, but they are implementable.

## Gaps That Are Partly Supported But Not Claimable

### Scalar tilt and low-l anomaly targets

The code now imports and reports the v0.9 target values:

```text
n_s = 0.964841143031
eta_R = 0.035158856969
q_IR = 1/4
ell_IR = 32
```

Implemented surfaces:

```text
oph_fpe/cosmology/unique_predictions.py
oph_fpe/cosmology/cmb_derivation.py
oph_fpe/cosmology/camb_adapter.py
```

Current status:

```text
measurement-comparable target: yes
finite-lattice-derived: no
physical CMB prediction: no
```

The next requirement is to make the finite screen derive `eta_R`, `q_IR`, and
`ell_IR` from freezeout/collar readouts rather than importing them.

### Anomaly Boltzmann functions

The CMB/late-structure contract needs:

```text
rho_A(a), rho_A_eq(a), B_A(k,a), Gamma_rec(k,a), stress variables
```

Implemented surfaces:

```text
oph_fpe/cosmology/boltzmann_inputs.py
oph_fpe/cosmology/ba_parent.py
oph_fpe/cosmology/anomaly_fluid.py
```

Current status:

```text
CDM-limit plumbing: yes
finite-collar diagnostic rows: yes
theorem-grade physical kernel: no
```

The current code correctly keeps `physical_cmb_prediction = false`.

### Neutrino lane

The v0.9 cosmology notes supply:

```text
N_eff = 3.044
sum_mnu = 0.090011929645 eV
f_nu ~= 0.00674
small-scale suppression ~= -5.4 percent
```

The simulator can use this as a fixed non-fit target in CAMB/CLASS style runs.
This is one of the best near-term measurement-comparable lanes, especially
against Planck/ACT/DESI constraints.

## Gaps That Need Theory Closure Before Simulation Can Prove Them

### Global screen capacity `N_scr`

The local pixel constant `P` is implemented correctly as area/entropy weighting.
The global screen capacity `N_scr` is a different object. The current OPH notes
explicitly say the capacity fixed-point theorem is missing.

The simulator can audit candidate closure maps, but it cannot prove the physical
global capacity without a declared selector:

```text
N_scr = N_OPH(N_scr)
```

### Baryogenesis

This remains an open theorem target. A simulator can implement a Z6/CP phase and
sphaleron/washout toy equation, but that would be a continuation model until the
paper supplies the CP source, rates, and freezeout computation.

### Full physical CMB likelihood

The simulator can run CAMB-style transfer and public-spectrum comparisons now.
It should not claim official Planck likelihood closure until it has:

- official low-l likelihood or faithful map-space substitute;
- component-separated map and mask pipeline;
- TT/TE/EE/lensing spectra;
- BAO/SNe/weak-lensing/RSD likelihood integration;
- OPH-derived, non-fit initial spectrum and anomaly kernels.

## Recommended Implementation Order

1. Implement `sync_gap.py` for mode-wise repair-rate spectra.
2. Implement `adiabaticity.py` for species-channel entropy/isocurvature proxies.
3. Add `hot_release.py` to define the release surface and entropy/temperature proxy.
4. Extend CAMB readout to use fixed OPH neutrino masses in addition to the scalar
   v0.9 target.
5. Implement tensor/non-Gaussian diagnostic readouts after the scalar gates are
   producing stable results.
6. Keep `physical_cmb_prediction`, `bulk_3d_established`, and
   `inflation_replacement_ready` false until the gates above pass under controls.

