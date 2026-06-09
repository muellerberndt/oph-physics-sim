# OPH-FPE Pro Handoff: Missing Theorem-Grade Objects Audit

Date: 2026-06-09

This handoff asks for a targeted Pro audit of the missing finite objects needed to turn the current OPH-FPE simulator from a strong diagnostic stack into a theorem-aligned, measurement-comparable lattice simulation.

The attached ZIP bundles the relevant code and reports. Please assume no access to the local filesystem outside the bundle.

## Current Honest Status

The simulator now emits:

- finite repair / record receipts,
- state-derived BW/KMS sanity receipts for direct transition automorphisms,
- chart-level conformal Lorentz / H3 receipts on the paper route,
- a minimal S2 cap-profile to H3 geometry calibration,
- Planck-facing CMB diagnostic curves,
- an exact OPH CAMB TT transfer curve with local Planck TT comparison,
- finite-certificate scaffolding for `A_zeta`, `Q_A`, `B_A`, and `Gamma_rec`.

It does **not** yet honestly emit:

- a strict neutral third-person 3D bulk reconstruction,
- a theorem-grade endogenous modular generator from observer/collar state,
- a real physical CMB prediction,
- a physical `P(k)` / matter-power prediction,
- a production particle spectrum in a reconstructed bulk.

The claim boundary in the code is currently:

> measurement-facing diagnostics and theorem-assisted H3 chart receipts exist; strict neutral 3D bulk, physical CMB, physical matter power, and production particles remain open.

## Latest Validation

Full local test suite:

```text
290 passed
```

Latest comparable snapshot:

```text
runs/stage212_latest_theory_bulk_cmb_snapshot_20260609/comparable_data_snapshot.md
```

Key current numbers:

```text
physical_cmb_prediction: false
physical_matter_power_prediction: false

chart-level conformal Lorentz receipt: true
theorem-assisted populated H3 chart receipt: true
strict neutral 3D bulk receipt: false

minimal S2 caps-to-H3 receipt: true
minimal caps-to-H3 median reconstruction MSE: 5.383275121116135e-33

finite certificate stack ready: true
finite real-physics certificate: false

exact OPH CAMB TT comparable curve: true
exact OPH CAMB IR shape correlation: 0.9998566988724703
exact OPH CAMB IR chi2/bin: 0.9509132706909185
n_s: 0.9648411430307772
q_IR: 0.25
ell_IR: 32

production particle-like count: 0
controlled defect assay particle-like count: 2
```

## Why This Handoff Exists

The current implementation is no longer failing at basic geometry or CAMB plumbing. The remaining problem is object specificity.

Several lanes are still proxies:

- `finite_certificates.py` can build a finite certificate bundle, but the run-derived input is still marked `proxy_certificate: true`.
- `state_derived_bw_report` currently passes for a direct transition automorphism mode, but not for an endogenous modular generator derived from a cap/collar density state.
- The H3 route passes at chart/theorem-assisted level, but strict neutral observer-record reconstruction still returns non-3D dimensions.
- CMB/CAMB output exists, but it is not yet a physical OPH prediction because the finite lattice has not derived the required scalar/anomaly/repair certificates.

We need Pro to specify the missing finite objects precisely enough to implement them.

## Bundled Code Pointers

Important code files in the ZIP:

```text
oph_fpe/bulk/cap_geometry.py
oph_fpe/bulk/cap_profile_geometry.py
oph_fpe/bulk/collar_state.py
oph_fpe/bulk/markov_collar.py
oph_fpe/bulk/modular_probe.py
oph_fpe/bulk/modular_response_kernel.py
oph_fpe/bulk/observer_reconstruction.py
oph_fpe/bulk/proof_certificate.py
oph_fpe/bulk/record_to_h3.py

oph_fpe/cosmology/finite_certificates.py
oph_fpe/cosmology/camb_adapter.py
oph_fpe/cosmology/cmb_derivation.py
oph_fpe/cosmology/comparable_data.py
oph_fpe/cosmology/boltzmann_inputs.py
oph_fpe/cosmology/screen_capacity.py
oph_fpe/cosmology/oph_constants.py

oph_fpe/observers/objects.py
oph_fpe/defects/array_s3_holonomy.py
oph_fpe/defects/controlled_assay.py

tests/test_finite_certificates.py
tests/test_bulk_proof_certificate.py
tests/test_modular_probe.py
tests/test_observer_reconstruction.py
tests/test_cap_profile_geometry.py
tests/test_camb_adapter.py
tests/test_comparable_data.py
```

Important bundled reports:

```text
reports/comparable_data_snapshot.md
reports/comparable_data_snapshot.json
reports/oph_exact_cmb_camb_report.md
reports/oph_exact_cmb_camb_report.json
reports/finite_certificate_report.md
reports/finite_certificate_report.json
reports/caps_to_h3_minimal_report.json
reports/bulk_proof_certificate_report.md
reports/bulk_proof_certificate_report.json
reports/oph_boltzmann_input_report.md
reports/oph_boltzmann_input_report.json
reports/bulk_reconstruction_report.json
reports/observer_distance_matrix.npz
reports/observer_chart_object_h3_report.json
reports/modular_response_h3_report.json
reports/particle_likeness_report.json
```

Important theory/source notes bundled:

```text
theory/recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex
theory/reality_as_consensus_protocol.tex
theory/screen_microphysics_and_observer_synchronization.tex
theory/observers_are_all_you_need.tex
theory/deriving_the_particle_zoo_from_observer_consistency.tex
theory/OPH_falsifiability.md

cosmology/cmb/OPH-CMB-Selector-Elimination-v1.5.md
cosmology/cmb/exact_ir_kernel_values_v1_5.csv
cosmology/cmb/OPH-CMB-Official-Likelihood-and-Finite-Patch-v1.0.md
cosmology/inflation/comms.md
cosmology/h0s8/specs.md
cosmology/neutrinos/neu6.md
cosmology/paper_audit/audit.md
```

## Main Questions For Pro

Please answer as implementable finite-object specs, not just conceptual text.

### 1. Endogenous Modular Generator

Current status:

- `modular_probe.py` supports `K_a = -log(rho + aI)`.
- Current passing BW receipt is effectively a direct transition-automorphism sanity mode.
- The simulator does not yet derive the correct modular generator from the observer-visible cap/collar state in a way strong enough to count as the finite theorem surrogate.

Questions:

1. What exact finite object should be used for `rho_C`?
2. Should the basis be `|node, packet, sector>`, `|record_family, cap_bin, sector>`, collar-crossing transitions, or another support-visible algebra?
3. What off-diagonal entries are legitimate under the papers?
4. Should `M_{uv}` be co-occurrence counts, perturb/remeasure transition counts, repair-history transitions, or a Petz/Markov recovery object?
5. What exact acceptance condition makes `K_a = -log(rho_C+aI)` theorem-aligned rather than a numerical surrogate?
6. What controls must fail for this receipt?

Please provide a concrete schema:

```python
BasisState = ...
rho_C = ...
observable_family = ...
K_a = ...
R_BW = ...
errors = ...
pass_conditions = ...
```

### 2. Parent-Collar / Recovery Ladder

Current status:

- `collar_state.py` and `markov_collar.py` compute classical cap/collar packet distributions and CMI.
- `finite_certificates.py` now emits a parent-collar certificate, but run-derived values are proxies and not theorem-grade.

Questions:

1. What is the correct finite parent/collar chain?
2. Should the collar split be `A_delta / B_delta / D_delta`, nested caps, or parent-child cap refinements?
3. What quantity should be tracked under refinement: CMI, Petz recovery error, Fawzi-Renner bound, modular residual, or all of them?
4. What scaling law or threshold certifies that the finite regulator is approaching the OPH support-visible branch?
5. How should `P` and the newly derived screen capacity `N` enter this certificate?

Please provide a testable output schema:

```json
{
  "N_values": [...],
  "cap_family": "...",
  "collar_widths": [...],
  "epsilon_cmi": [...],
  "recovery_error": [...],
  "modular_error": [...],
  "scaling_pass": true/false,
  "claim_boundary": "..."
}
```

### 3. Repair Matrix Certificate

Current status:

- `repair_matrix_certificate.json` exists, but it is not based on a theorem-grade packet-state transition matrix.
- The current finite certificate lane is explicit that this is still proxy-level.

Questions:

1. What states should index the repair transition matrix?
2. Should it be a stochastic matrix over observer-visible packets, sectors, record families, or cap/collar basis states?
3. What exact matrix entries are counted from simulation traces?
4. What spectral/eigenvalue/ergodicity/detailed-balance properties need to be checked?
5. What quantity from this matrix should become `Gamma_rec(k,a)` or the repair kernel feeding the Boltzmann bridge?

Please provide:

```python
repair_state = ...
T[u, v] = ...
certificate_metrics = ...
Gamma_rec(k, a) = ...
pass_conditions = ...
```

### 4. Homogeneous Anomaly / Inflation Certificate

Current status:

- CMB/inflation diagnostics exist.
- `inflation_certificate_report` is not complete.
- The finite certificate stack can emit proxy `A_zeta`, `Q_A`, `B_A`, `Gamma_rec`.

Questions:

1. What finite simulation observable should become the homogeneous anomaly certificate?
2. How do we derive `epsilon_star`, `kappa_rel`, `A_zeta`, and `n_s` from the lattice without fitting Planck?
3. Is the current `n_s = 1 - eta_R` route acceptable if `eta_R` is derived from the finite collar ladder?
4. How should the `P/48` spectrum and selector-elimination `q_IR=1/4`, `ell_IR=32` be wired to finite lattice receipts?
5. What falsifier would show that the simulator is only importing the target rather than deriving it?

Please provide an implementable ladder:

```text
finite screen records
-> finite anomaly object
-> scalar release certificate
-> primordial P_zeta(k)
-> CAMB/CLASS source table
-> Planck likelihood
```

### 5. Strict Neutral 3D Bulk Reconstruction

Current status:

- Chart-level Lorentz/H3 route passes.
- Theorem-assisted populated H3 chart passes.
- Strict neutral observer reconstruction still does not yield a 3D window.
- The current neutral reconstruction explicitly avoids radial depth.

Questions:

1. Should strict neutral reconstruction be expected to return dimension 3 from finite records alone, or is chart-level H3 the correct paper-side bulk object?
2. If strict neutral reconstruction is required, what observer-visible features should define the distance matrix?
3. Are current object families too screen-local? If yes, what replaces support-overlap object extraction?
4. Should object identity be based on continuation classes under repair, counterfactual response, modular response signatures, sector transport, or another equivalence relation?
5. What dimension estimator is theorem-aligned: spectral, correlation, local MLE, cross-ratio/H3 chart, persistent homology, or another one?

Please provide:

```python
observer_object = ...
observer_view = ...
similarity(i, j) = ...
distance(i, j) = ...
dimension_receipt = ...
controls = ...
```

### 6. Particle / Defect Object

Current status:

- Controlled defect assay can produce particle-like controlled cases.
- Production particle-like count is currently zero.
- S3 holonomy defects remain screen/collar proxies until strict or chart-populated bulk transport is certified.

Questions:

1. What finite defect object is paper-accurate enough to count as a particle precursor?
2. Should the sector algebra be S3 for now, finite U(1), SU(2) truncation, or D10/SM branch object?
3. What exact worldline and conservation receipts are required before a defect can be called particle-like?
4. Is chart-H3 transport enough for particle worldlines, or must strict neutral bulk reconstruction pass first?
5. What controlled assay should become the production detector?

Please provide:

```python
defect_basis = ...
charge_sector = ...
worldline_metric = ...
fusion_rule_check = ...
scattering_check = ...
particle_like_pass = ...
```

### 7. Physical CMB Claim Gate

Current status:

- `oph_exact_cmb_camb_report` emits a measurement-comparable TT curve.
- It is not yet physical because the input numbers are still target/continuation diagnostics unless derived from finite certificates.

Questions:

1. What minimal finite certificate set is sufficient to mark `physical_cmb_prediction = true`?
2. Does the official Planck likelihood gate require Cobaya/clik/map-space likelihood, or is binned TT chi2 acceptable for an internal receipt?
3. What exact OPH-generated quantities must feed CAMB/CLASS:
   - `P_zeta(k)` only?
   - `B_A(k,a)`?
   - `Gamma_rec(k,a)`?
   - modified transfer equations?
4. Which measurements should be first-class gates: Planck TT/TE/EE, lensing, BAO, H0/S8, neutrinos?
5. What would falsify the OPH CMB bridge at the simulator level?

Please provide the exact minimum acceptance ladder.

## Suspected Implementation Problems

Please audit these potential code/design issues:

1. The current `bulk_3d_established` top-level flag can be confusing because it tracks chart/object bulk gate, while strict neutral bulk remains false. Should it be renamed or split further?
2. `finite_certificates.py` now emits useful certificate artifacts, but `run_proxy_certificate_input()` may be too generous. What should it refuse?
3. `modular_probe.py` may conflate direct transition automorphism sanity with a real state-derived modular generator. How should reports separate these?
4. The current H3 response fit can pass even when H3 chart dimension estimators are not a clean 3D window. Is this okay if the paper route is chart-theoretic rather than estimator-theoretic?
5. The Planck-facing CAMB lane has excellent TT shape agreement, but the finite lattice does not yet derive all inputs. What code guard should prevent overclaiming?
6. Should the new screen capacity `N` closure enter only metadata/capacity normalization, or should it constrain actual regulator size / finite capacity noise?

## Requested Pro Output

Please return:

1. A ranked list of missing finite objects that must be implemented next.
2. Concrete mathematical definitions for each object.
3. Minimal algorithms and data schemas.
4. Required controls and pass/fail thresholds.
5. Any code-level diagnosis from the bundled files.
6. A recommended shortest path to:
   - physical CMB prediction,
   - strict or paper-acceptable 3D bulk proof,
   - particle/defect output.

Please be explicit about what may be claimed at each receipt level.

## Current Working Rule For The Coding Agent

Until Pro clarifies otherwise, the coding agent should keep the following boundary:

```text
Allowed:
  measurement-facing diagnostics,
  exact CAMB transfer scaffolds,
  chart-level conformal Lorentz receipts,
  theorem-assisted H3 chart/population receipts,
  finite certificate proxy artifacts.

Not allowed:
  physical CMB prediction,
  completed strict neutral 3D bulk,
  physical P(k),
  production particle spectrum,
  proof that finite screen pixels mechanically force 3D.
```
