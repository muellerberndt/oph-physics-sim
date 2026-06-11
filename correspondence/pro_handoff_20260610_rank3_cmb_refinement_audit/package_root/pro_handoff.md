# OPH-FPE Pro Handoff: Rank-3/CMB Refinement Audit

Date: 2026-06-10

This handoff is for diagnosing the current OPH physics simulator path toward:

1. a neutral observer-record 3D bulk proof,
2. a paper-accurate 3+1D observer-facing Lorentz/H3 experience,
3. measurement-comparable CMB output,
4. eventually physical CMB/early-universe predictions.

The accompanying zip includes source code, tests, configs, selected paper/cosmology text sources, the current run artifacts, git status, and an uncommitted diff. Pro has no filesystem access, so please use the zip as the full audit bundle.

## Current Bottom Line

We now have a stronger intermediate result:

- A **control-quotient coordinate rank-3/E3 candidate** is stable across 64k and 256k observer screens.
- The cross-scale rank-refinement report passes its own candidate receipt.
- The CMB lane has a **measurement-comparable CAMB TT curve** from the scale-compressed branch with excellent shape agreement.

But the central proof is still not closed:

- `strict_neutral_third_person_bulk_established = false`
- `physical_cmb_prediction = false`
- `production_particle_matter_receipt = false`
- independent SVD rank selection does **not** select rank 3
- the control-quotient lane is not a proper negative/null control
- directional H3 strict gates are still not passed

So: the simulator has useful measurement-facing diagnostics and a 3D-compatible finite-regulator candidate, but not yet the final neutral-bulk proof or physical CMB prediction.

## Latest Verified Commands

The edited code compiles:

```bash
python3 -m py_compile \
  oph_fpe/cosmology/comparable_data.py \
  oph_fpe/bulk/proof_certificate.py \
  oph_fpe/bulk/neutral_bulk.py \
  oph_fpe/bulk/prime_geometric_response.py \
  oph_fpe/measurement_pack.py \
  oph_fpe/cli.py
```

Focused tests passed:

```bash
python3 -m pytest \
  tests/test_neutral_bulk.py::test_prime_rank_refinement_candidate_is_not_strict_without_independent_rank \
  tests/test_prime_geometric_response.py \
  tests/test_comparable_data.py::test_comparable_data_collects_control_quotient_spatial_3d_candidate \
  tests/test_bulk_proof_certificate.py::test_bulk_proof_certificate_reads_scale_compressed_branch_without_overclaiming \
  -q
```

Result:

```text
8 passed in 0.21s
```

I did not run the full test suite after the latest reporting edits.

No OPH-FPE long-running jobs were active at handoff creation.

## Current Key Artifacts

Important run artifact directories in the zip:

```text
oph-physics-sim/runs/stage318_measurement_pack_clean_two_seed_cmb_h3_20260610
oph-physics-sim/runs/stage319_prime_geometric_rank_sweep_64k_seed2026060902_20260610
oph-physics-sim/runs/stage320_prime_geometric_rank_sweep_64k_seed20261045_20260610
oph-physics-sim/runs/stage321_prime_geometric_rank_sweep_256k_seed20261049_with_rank_audit_20260610
oph-physics-sim/runs/stage322_prime_geometric_rank_sweep_256k_seed20261050_with_rank_audit_20260610
oph-physics-sim/runs/stage323_prime_rank_refinement_64k_256k_20260610
oph-physics-sim/runs/stage324_handoff_snapshot_rank_refinement_cmb_20260610
```

Most important files:

```text
runs/stage323_prime_rank_refinement_64k_256k_20260610/prime_geometric_rank_refinement_report.json
runs/stage324_handoff_snapshot_rank_refinement_cmb_20260610/comparable_data_snapshot.json
runs/stage324_handoff_snapshot_rank_refinement_cmb_20260610/comparable_data_snapshot.md
runs/stage318_measurement_pack_clean_two_seed_cmb_h3_20260610/bulk_proof_certificate_report.json
runs/stage318_measurement_pack_clean_two_seed_cmb_h3_20260610/scale_compressed_cmb_camb_report.json
runs/stage318_measurement_pack_clean_two_seed_cmb_h3_20260610/claims.json
```

## Rank-3 / Bulk Status

Latest cross-scale rank refinement:

```json
{
  "mode": "prime_geometric_rank_refinement_v0",
  "run_count": 4,
  "multi_scale": true,
  "control_quotient_rank3_refinement_candidate_receipt": true,
  "strict_neutral_bulk_refinement_receipt": false,
  "candidate_dimension_drift": 0.027657584355957443,
  "candidate_dimension_stable": true,
  "independent_rank3_selector_all": false,
  "proof_blockers": [
    "independent_svd_rank3_selector_not_stable_or_false",
    "control_quotient_lane_is_not_a_negative_control",
    "directional_h3_strict_rank_gate_not_passed"
  ]
}
```

By patch count:

```text
64k, 2 seeds:
  rank-3/E3 candidates: 2/2
  median candidate dimension: 2.8435291556015314
  median corr dimension: 2.677335068799884
  median local-MLE dimension: 3.0097232424031795
  median S2 leakage corr: 0.0011380734312799556

256k, 2 seeds:
  rank-3/E3 candidates: 2/2
  median candidate dimension: 2.815871571245574
  median corr dimension: 2.648496455763636
  median local-MLE dimension: 2.9832466867275125
  median S2 leakage corr: 0.014505440955573923
```

Important diagnosis:

The candidate is stable, but it is not strict neutral bulk proof because the selected coordinate rank is still effectively a chosen rank-3 coordinate window. I added independent rank-selection metadata from the singular spectrum; it does not select rank 3.

Observed independent-rank behavior:

```text
64k seed 2026060902:
  independent rank3: false
  largest singular gap rank: 43
  chord elbow rank: 13
  effective rank: ~98.35
  participation rank: ~77.13
  rank3 cumulative variance: ~0.07737

64k seed 20261045:
  independent rank3: false
  largest singular gap rank: 2
  chord elbow rank: 15
  effective rank: ~99.71
  participation rank: ~78.35
  rank3 cumulative variance: ~0.07645

256k seed 20261049:
  independent rank3: false
  largest singular gap rank: 1
  chord elbow rank: 11
  effective rank: ~154.49
  participation rank: ~119.87
  rank3 cumulative variance: ~0.04993

256k seed 20261050:
  independent rank3: false
  largest singular gap rank: 53
  chord elbow rank: 6
  effective rank: ~156.63
  participation rank: ~122.60
  rank3 cumulative variance: ~0.04704
```

Interpretation question: if the papers say 3D arises from the conformal/Lorentz branch, should the finite simulator select 3 by the theorem-side H3 chart/orbit dimension, or should a neutral singular spectrum also show an actual rank-3 elbow? If the latter, the code or observable is still wrong.

## CMB / Measurement Status

The current best measurement-comparable CMB artifact is:

```text
runs/stage318_measurement_pack_clean_two_seed_cmb_h3_20260610/scale_compressed_cmb_camb_report.json
```

Key values:

```text
measurement_comparable_cmb_curve: true
physical_cmb_prediction: false
n_s: 0.9660214956374176
eta_R: 0.033978504362582485
q_IR: 0.25
ell_IR: 32.0
Planck TT shape correlation: 0.9998534542805271
Planck TT normalized RMSE: 0.01711991352040137
amplitude-fit chi2/bin: 0.9615050793060024
best-fit-column chi2/bin: 0.7845969499786535
first peak ell: 225.164945
```

This is good as a diagnostic curve. It is not yet a physical OPH prediction because:

- the 24-round scale-compressed branch is not yet theorem-certified as the finite repair clock,
- amplitude normalization is not derived from the finite screen/lattice,
- anomaly-sector kernels are not finite-lattice derived,
- no official Planck likelihood is used,
- `kappa_rep = e` is not dynamically certified.

Relevant related lanes in `stage324`:

```text
scale-compressed CAMB:
  measurement-comparable curve: true
  physical prediction count: 0
  n_s: 0.9660214956374176

MaxEnt Green source:
  theorem-side/source receipt: true
  finite-lattice derived: false
  physical CMB prediction: false
  n_s: 0.9648411430307772

repair clock:
  finite repair-clock certificate: false
  median kappa_rep estimate: 16.91187547087023
  target kappa_rep: 2.718281828459045

scalar repair semigroup:
  finite transition matrix source: true
  kappa_rep: 2.592344556802311
  n_s: 0.9664700434909562
  transition clock certified: false
```

## Recent Code Changes Worth Auditing

Main touched files:

```text
oph_fpe/bulk/prime_geometric_response.py
oph_fpe/bulk/neutral_bulk.py
oph_fpe/bulk/proof_certificate.py
oph_fpe/cosmology/comparable_data.py
oph_fpe/measurement_pack.py
oph_fpe/cli.py
oph_fpe/bulk/__init__.py
tests/test_neutral_bulk.py
tests/test_prime_geometric_response.py
```

Changes:

1. Added singular-value rank-selection metadata to `response_component_spectrum`.
   - Top singular values.
   - Explained/cumulative variance.
   - Rank-3 cumulative variance.
   - Rank90/rank95.
   - spectral entropy/effective rank.
   - participation rank.
   - largest-gap rank.
   - chord-elbow rank.
   - `independent_rank3_selector_receipt`.

2. Added `prime_geometric_rank_refinement_report`.
   - Aggregates rank-sweep reports over 64k/256k.
   - Checks whether control-quotient coordinate rank-3/E3 candidate is stable.
   - Keeps strict-neutral proof false unless missing gates pass.

3. Added CLI:

```bash
python3 -m oph_fpe.cli neutral-prime-rank-refinement \
  --report <rank-sweep-report-or-dir> ... \
  --out <out-dir>
```

4. Integrated the new refinement report into:
   - comparable-data rows/lanes,
   - measurement-pack export,
   - proof certificate summary.

5. Regenerated:

```text
runs/stage324_handoff_snapshot_rank_refinement_cmb_20260610
```

The full uncommitted patch is in:

```text
current_uncommitted_diff.patch
```

## Main Questions For Pro

Please answer with concrete theorem-to-code guidance, not just conceptual reassurance.

### Q1. What should select rank 3 in a finite simulator?

The current neutral profile can show rank-3/E3-compatible coordinate windows, but the singular spectrum does not independently select rank 3. Does the OPH paper route require:

- an independent finite rank-3 elbow in support-visible data, or
- chart-level Lorentz/H3 theorem-side rank 3 plus a separate object-population receipt, or
- another finite observable entirely?

If there is an exact paper-aligned rank selector, please specify it algorithmically.

### Q2. Is the control-quotient coordinate lane legitimate?

The stable rank-3 result is in the control-quotient coordinate lane, not a negative/null-control lane. Is this quotient operation physically justified by the papers, or is it removing too much structure and making rank 3 easier to see?

What should replace it if it is not legitimate?

### Q3. What is the exact neutral observer-record distance?

We need a neutral bulk reconstruction that is not a radial-depth prior and not an S2-locality artifact. Please specify the finite formula for:

```text
observer/object record extraction
observer similarity
distance conversion
dimension estimator gate
negative controls
```

Current problem: previous object extractions were boundary dominated; rank-3 coordinate windows are promising but not yet object-populated neutral bulk.

### Q4. What exact finite object should represent “what observers see”?

The current object lanes use persistent record families, chart-object compactness, and transition/lineage variants. What does the paper stack require as the finite observer-facing object:

- cap/collar packet family?
- record-family transition class?
- modular response profile?
- sector/holonomy conserved class?
- checkpoint-continuation class?

Please specify support, persistence, overlap agreement, and counterfactual stability gates.

### Q5. How do we derive the physical CMB curve from finite lattice data?

We can currently produce an excellent measurement-comparable curve, but it is not a physical prediction. The blockers are:

- repair clock not certified,
- `kappa_rep = e` not derived dynamically,
- amplitude not derived,
- anomaly kernels not finite-lattice derived,
- official likelihood missing.

What finite computation should generate:

```text
n_s
A_s / A_zeta
q_IR
ell_IR
B_A(k,a)
Gamma_rec(k,a)
rho_A(a)
```

and which of these are theorem-side closures versus finite-run observables?

### Q6. What exactly proves “inflation symptoms without inflation”?

The current simulation/cosmology lanes expose:

- near scale invariance,
- acoustic peaks via CAMB transfer,
- low-ell / parity diagnostics,
- finite screen capacity receipts,
- H0/S8 and neutrino diagnostic scaffolds.

What is the minimal certificate set that proves the inflation symptoms arise naturally from OPH synchronization rather than from fitting?

### Q7. Does the latest N screen-capacity closure change the simulation?

P is already used as local pixel/cell area and entropy weight. There is now an N screen-capacity closure in the latest papers. Should finite runs use:

- numerical regulator mode only,
- physical-cell toy-universe mode,
- N-derived capacity weights,
- a scale-compressed capacity mapping,
- or a new finite-size extrapolation law?

Please specify the exact role of N in the simulator and CMB normalization.

### Q8. Speed/efficiency

The 64k/256k rank sweeps are manageable locally. Larger 1M+ runs need careful CPU use. Current code is mostly CPU/NumPy/SciPy/CAMB bound. Please advise:

- which diagnostics are worth scaling to 1M now,
- which should be cached,
- whether rank spectra/observer views can be reused safely,
- whether any component should move to GPU,
- what the next run matrix should be.

## Recommended Next Local Actions If Pro Agrees

1. Add a proper independent rank selector or replace the SVD-rank criterion with the paper-correct finite rank gate.
2. Replace/justify the control-quotient coordinate lane.
3. Implement the paper-correct neutral object extraction and neutral distance.
4. Run a strict 64k/256k/1M refinement with at least 4 seeds.
5. Keep the current CMB curve as a diagnostic, but work on the finite repair-clock certificate and amplitude closure before calling it physical.
6. Regenerate one measurement pack only after the proof gates are tightened.

## Package Contents

The zip should contain:

```text
pro_handoff.md
git_status.txt
current_uncommitted_diff.patch
focused_tests.log
py_compile_selected.log
package_root/
  oph-physics-sim/
    oph_fpe/...
    tests/...
    configs/...
    docs/repair_rounds.txt
    runs/stage318...
    runs/stage319...
    runs/stage320...
    runs/stage321...
    runs/stage322...
    runs/stage323...
    runs/stage324...
  reverse-engineering-reality/
    paper/*.tex selected core papers
    extra/*.tex selected cosmology/microphysics files
  cosmology/
    selected CMB/inflation/H0S8/neutrino/capacity notes
```

No `.env`, secrets, cloud credentials, private keys, or production data are intentionally included.

