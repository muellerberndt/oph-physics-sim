# OPH-FPE Theory Conformance Audit

Date: 2026-06-09

Scope:

- `cosmology/`
- `reverse-engineering-reality/paper/`
- `reverse-engineering-reality/extra/`
- `oph-physics-sim/oph_fpe/`

This audit checks whether OPH-FPE carries the latest paper and cosmology-branch requirements as executable constants, diagnostics, certificates, or explicit closed gates. It does not claim that every theorem-side branch is implemented as a finite simulator derivation.

## Executive Status

The simulator is aligned with the current theory stack as a receipt/certificate framework:

- `P` is carried as the local pixel/cell closure value.
- `N_CRC` is now carried as the global screen-capacity closure value.
- finite regulator patch count remains separate from physical cosmic screen capacity.
- selector-elimination v1.5 is represented: `q_IR=1/4`, `ell_IR=32`, and `eta_R=kappa_rep alpha sqrt(pi)` with `kappa_rep=e` still certificate-gated.
- finite CMB/cosmology certificate machinery is represented: scalar release, edge-center, homogeneous anomaly, parent collar, repair matrix, Boltzmann handoff, and no-data-use firewall.
- H0/S8, neutrino, and dark-response lanes are measurement-comparable diagnostics with physical-prediction gates closed unless finite-collar/kernel certificates are provided.
- 3+1D/Lorentz and 3D-bulk claims remain separated into chart-level BW/Lorentz receipts, H3 response diagnostics, object-population gates, and strict neutral-bulk gates.

Main conformance patch from this audit:

- `N_CRC` was added to the shared OPH cosmology constants.
- screen-capacity reports now expose active edge-center / predictive quotient / readback-map gates required by the latest capacity notes.

## Theory Requirements Matrix

| Theory surface | Latest requirement | Simulator status | Relevant implementation |
| --- | --- | --- | --- |
| `P` local closure | Treat `P` as local pixel/cell scale, area, entropy, and downstream quantitative closure. Do not use it to force dimension 3 or alter BW `2*pi`. | Implemented. | `oph_fpe/constants/oph_pixel.py`, `oph_fpe/core/pixel_scale.py`, `oph_fpe/core/screen_microphysics.py` |
| `N_CRC` global capacity closure | Treat `N_CRC = F(N_CRC)` as global cosmic record/screen-capacity closure. On observed branch `N_CRC=N_scr ~= 3.31e122`, `Lambda l_P^2=3*pi/N_CRC`. | Implemented as declared closure/readout value; finite simulator fixed-point solve remains open. | `oph_fpe/cosmology/screen_capacity.py`, `oph_fpe/cosmology/oph_constants.py` |
| Capacity variable | Use entropy capacity `N`, not raw Hilbert dimension or regulator patch count. Active capacity is the predictive quotient of edge-center records. | Implemented as gate/claim boundary; active predictive quotient not implemented. | `screen_capacity.py` |
| Readback map | Implement `F(N)=Cap_read(Obs(nf(U_N)))` to derive capacity from terminal observer sector. | Not implemented; explicitly gated false. | `screen_capacity.py` |
| Count-density selector | `N_star = MAR argmax_N [log |Omega_N^sc| - N]`, with pressure/contraction certificate. | Not implemented; explicitly gated false. | `screen_capacity.py` |
| CMB finite-patch field | Finite normal-form screen field and MaxEnt inverse-Laplacian covariance are valid diagnostics. | Implemented as diagnostic scaffolds. | `cmb_fossil/`, `oph_fpe/cosmology/oph_screen_power.py`, `cmb_transfer.py` |
| CMB selector elimination v1.5 | `q_IR=1/4` from affine zero-mode reserve; `ell_IR=32` from dodecahedral visible covariance rank; `eta_R=kappa_rep alpha sqrt(pi)`, `kappa_rep=e` remains certificate. | Implemented. Source packet audit supported. | `oph_fpe/cosmology/selector_elimination.py` |
| CMB physical prediction | Must require finite-lattice derivation, official Planck likelihood, and map-space anomaly tests before physical CMB claim. | Gates remain closed. | `camb_adapter.py`, `comparable_data.py`, `selector_elimination.py`, `inflation_certificates.py` |
| Inflation-free branch | Treat as continuation/certificate branch: flatness, same-boundary/low-k synchronization, scalar release, MaxEnt hot release, Boltzmann handoff. | Represented as certificate scaffolding and diagnostic reports; not complete physical prediction. | `sync_gap.py`, `hot_release.py`, `adiabaticity.py`, `inflation_certificates.py` |
| Finite certificate authority | Universe Simulation must emit finite certificates for `A_zeta`, `Q_A`, `B_A(k,a)`, `Gamma_rec(k,a)`, and Boltzmann handoff before likelihood use. | Implemented with validators/toy compiler/no-data-use firewall. Real certificates still require simulator outputs. | `finite_certificates.py`, `inflation_certificates.py` |
| H0/S8 | Current executable branch is low-H0 Planck-like; weak-lensing lowering needs finite-collar `B_A(k,a)` / five-of-seven projection as theorem-motivated kernel target. | Implemented as diagnostic/certificate lanes; physical prediction gates closed. | `h0s8.py`, `h0s8_certificates.py`, `oph_kernels.py` |
| Dark response | Static galaxy RAR must not be used as universal linear FLRW kernel. Parent finite-collar functional must emit `rho_A(a)` and `B_A(k,a)`. | Implemented as explicit claim boundary and kernel scaffolds. | `neutrino_background.py`, `finite_certificates.py`, `oph_kernels.py`, `galaxy_static.py` |
| Neutrinos | Carry OPH weighted-cycle neutrino target and standard relic-neutrino cosmology comparisons; finite-lattice derivation remains separate. | Implemented. | `neutrino_background.py`, `unique_predictions.py` |
| Lane 8 entropy/arrow | Use record-compatible ancestry, Fano/Hoeffding payload lower bounds, fake-history suppression, no hidden-representative inflation. | Implemented as executable certificate stack, not full cosmological arrow proof. | `h0s8_certificates.py`, `oph_universe/arrow/` |
| Lorentz / 3D bulk | OPH claims Lorentz kinematics on certified support-visible BW/geometric cap-pair branch, not from finite cells alone. Populated H3/neutral bulk remains separate gate. | Implemented as separated lanes and gates. | `bulk/conformal_spatial_chart.py`, `bulk/proof_certificate.py`, `bulk/observer_reconstruction.py`, `comparable_data.py` |
| Defects/particles | Screen/collar holonomy clusters are not matter particles until neutral bulk/object population gates pass. | Implemented as claim boundary; particle gates remain separate. | `defects/`, `bulk/proof_certificate.py`, `comparable_data.py` |

## Current Open Gates

The following are intentionally not closed by the simulator after this audit:

1. `F(N)` readback map from terminal observer normal forms.
2. active edge-center predictive quotient implementation for capacity readout.
3. count-density normal-form enumerator / pressure certificate for `N_CRC`.
4. finite-patch repair-clock proof `kappa_rep=e`.
5. finite-register noncollapse certificate for `ell_IR=32`.
6. freezeout branch certificate coupling protected reserve to affine `ell=0` sector for `q_IR=1/4`.
7. real scalar-release, parent-collar, repair-matrix, and Boltzmann certificates from non-toy simulator data.
8. official Planck likelihood and masked map-space anomaly tests.
9. finite-collar `B_A(k,a)` and `Gamma_rec(k,a)` from actual collar/repair state data.
10. strict neutral populated 3D bulk and particle/matter gates.

## Claim Boundary

The simulator conforms to the latest theory stack as a finite-regulator diagnostic and certificate framework. It should not claim:

- full physical CMB prediction,
- completed early-universe simulation,
- strict neutral 3D bulk emergence,
- production particle/matter emergence,
- or finite-simulator derivation of `N_CRC`,

until the corresponding gates above are emitted from real finite simulator data and survive controls.

