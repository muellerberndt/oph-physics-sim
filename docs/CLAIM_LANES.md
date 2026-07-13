# Claim lanes: detailed notes

Moved verbatim from the repository README (2026-07-14). Lane-by-lane
contracts, layer separations, and command references. The compact lane
summary lives in the README; the living experiment status lives in
`OPH_SIGNATURE_EXPERIMENT_TRACKER.md`.

## Screen, scale, and capacity

`P` is carried as the local pixel/cell closure value. In finite screen runs it normalizes cell area, cell entropy `P/4`, cap capacity, and residual weights.

`N_CRC` is carried as the global screen-capacity closure value. Regulator patch counts such as `4096`, `65536`, `262144`, and `1048576` are sampling counts unless a dedicated capacity readback map and terminal normal-form enumerator close the finite capacity gate.

The scale reports write dimensionless-invariant and independent-bridge receipts. `finite_simulator_derived_G_SI` remains false without an accepted dimensionful scale bridge and finite capacity proof.

## BW, H3, and bulk


The simulator separates these layers:

```text
finite settling and finite consensus: receipt-gated
BW/KMS 2*pi branch replay: receipt-gated
cap-normal H3 chart theorem: primitive-field receipt-gated
conformal Lorentz/H3 chart: receipt-gated
observer-facing H3 object population: receipt-gated
finite Lorentz theorem contract: receipt-gated proof contract
strict neutral third-person bulk: receipt-gated frontier
```

The older chart receipt records the theorem-level dimension count
`H3 = SO+(3,1)/SO(3)`, hence `6-3=3`, before any finite neutral point-cloud
estimator is consulted. The old graph-distance and modular-lift dimension
estimates are debug diagnostics. Bulk claims flow through BW/KMS,
support-visible cap/collar data, the cap-normal/conformal H3 chart receipts,
observer records, object population, and neutral-bulk gates.

## CMB and cosmology

Screen-level angular spectra are measurement-facing diagnostics. They are useful for seed/control studies and viewer payloads.

Physical CMB prediction has a stricter contract: finite OPH sources for amplitude, scalar quotient, IR selectors, finite-collar kernels, recovery rates, Boltzmann handoff, CMB1 custom-parent CDM-limit regression, Standard-Model-off control regression, frozen solver assumptions, blinded comparison setup, and official full-observable likelihood execution. Those gates are reported by the frozen transfer/likelihood closure, physical CMB frontier, and promotion-audit outputs for each concrete run.

Related commands include:

```bash
python3 -m oph_fpe.cli cmb-lite-compare --run-dir runs/<run_id> --benchmark runs/benchmarks/COM_PowerSpect_CMB-TT-binned_R3.01.txt
python3 -m oph_fpe.cli cl-from-freezeout-npz --run-dir runs/<run_id> --out runs/<run_id>/cl_recomputed
python3 -m oph_fpe.cli oph-screen-power --run-dir runs/<run_id> --out runs/screen_power
python3 -m oph_fpe.cli cmb-anomaly-report --run-dir runs/<run_id> --source-dir runs/<run_id> --out runs/cmb_anomaly
python3 -m oph_fpe.cli dark-sector-simulation-plan --run-dir runs/<run_id> --out runs/<run_id>/dark_sector_simulation_plan
python3 -m oph_fpe.cli physical-cmb-output-comparison --run-dir runs/<run_id> --out runs/physical_cmb_output_comparison
```

The dark-sector simulation plan is an integration receipt. It reads the static
galaxy, finite covariant parent, finite-collar Boltzmann bundle, Boltzmann-input,
CMB anomaly, and frozen likelihood reports, then names the first blocked
promotion gate and the next simulator command to run. It is not a dark-matter
prediction or likelihood by itself. Exact `exp(-P/24)` collar-coefficient
promotion is tracked separately through local-reserve, scalar-weighted z6-mean,
and uniform product-thickening gates; finite-thickness profile coefficients
remain the default unless those receipts close.

## Defects and particles

The screen-holonomy layer writes defect clusters, timelines, interaction proxies, H3 worldline fits, and particle-likeness reports. These are screen/collar diagnostics. Production P1 is independently recomputed as P0 proto-worldline evidence AND a classical carrier-mode receipt AND a quantum Hilbert/spectral/asymptotic receipt, with deconfinement required for colored candidates. Legacy producer booleans cannot promote it.

Useful commands:

```bash
python3 -m oph_fpe.cli controlled-defect-assay --out runs/controlled_defect_assay
python3 -m oph_fpe.cli shape-dodeca-smoke --config configs/shape_dodeca_vertex_smoke.yml --out-dir runs
python3 -m oph_fpe.cli shape-ensemble --config configs/shape_dodeca_ensemble.yml --seeds 1,2,3,4 --out-dir runs
```

## Positive geometry kernel

The amplituhedron and positive-geometry checker is a fail-closed optimization layer. The trusted path remains finite patches, records, mismatch, accepted repair, quotient normal forms, observer readout, and evidence receipts.

```bash
python3 -m oph_fpe.cli positive-geometry-kernel-report --out runs/positive_geometry_kernel
```

Normal runs can request the checker through:

```yaml
kernels:
  positive_geometry:
    enabled: true
```

The expected safe verdict is `GEOMETRY_CERTIFIED_BACKEND_NOT_ENABLED` unless OPH sector recognition, native geometry certification, observer-readout equivalence, resource accounting, provenance hashes, and fallback receipts all pass for the concrete sector.

## W/Z/H numerical backend

The fail-closed boson backend consumes hash-pinned D10/D11 theorem artifacts
and computes source-clock gaps, frozen affine RG controls, BRST-block
determinants, and Rouché pole enclosures:

```bash
python3 -m oph_fpe.bosons \
  --config configs/bosons/wzh_source_closure_diagnostic_v1.yml \
  --out runs/bosons/wzh_source_closure_diagnostic_v1
```

The tracked config is synthetic and diagnostic. Actual W/Z/H predictions
require the independently frozen carriers, source clock, physical RG packet,
BRST-complete kernels, identities, uncertainties, and prospective provenance
listed in `docs/WZH_NUMERICAL_BACKEND.md`.

## Reference vacuum baselines

Reference vacuum baselines:

```bash
python3 -m oph_fpe.cli reference-vacuum-baseline \
  --out runs/reference_vacuum_baseline \
  --ell-max 16 \
  --sample-count 256 \
  --smoothing-sigma 0.05
```

This writes a direct-sampled free-scalar harmonic Gaussian baseline, a compact-`U(1)` lattice-gauge reference sampler, deterministic replay receipts, smoothing provenance, finite-mode refinement diagnostics, and false OPH-native promotion receipts. Semantic-stream replay and canonical serial-chain replay are reported separately from pathwise partition invariance, which remains false unless a concrete commuting-event or transaction-serialization receipt passes.
