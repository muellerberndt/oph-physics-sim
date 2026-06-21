# OPH Physics Simulator: Theorem-to-Code Implementation Contract

Date: 2026-06-19

## Executive verdict

The current simulator has a credible **paper-route branch-instantiation**:

- finite repair and record commitment;
- declared/branch-accurate BW/KMS `2π` cap transport;
- the conformal bridge `Conf+(S2) -> SO+(3,1) -> H3`;
- controlled, theorem-assisted population of the H3 chart in the best evidence packs.

It does **not** yet instantiate the strongest finite analogue of the OPH Lorentz theorem, because the load-bearing modular normalization and geometric transport are still supplied to important lanes rather than recovered from a finite cap operator algebra and independently fitted support-visible action. It also does not establish chart-blind neutral bulk discovery, production particles, or physical CMB prediction.

The implementation should therefore distinguish three claims:

1. **Branch replay** — the code correctly runs the declared BW/H3 route.
2. **Finite theorem-hypothesis instantiation** — the code constructs the finite carrier, records, recovery map, cap state, modular generator, support-visible covariance, rigidity, and refinement receipts from run data.
3. **Chart-blind discovery audit** — records alone recover a stable H3/3D relational geometry without receiving the theorem chart. This is a stronger, optional audit, not a prerequisite for the paper-route H3 claim.

---

## 1. Replace the ambiguous receipt ladder

Keep backward-compatible fields, but introduce this canonical ladder:

```text
C0a  finite_settle_diagnostic
C0b  finite_consensus_theorem_receipt
C1   finite_record_algebra_receipt
L0   bw_branch_replay_receipt
L1   endogenous_cap_state_receipt
L2   endogenous_modular_generator_receipt
L3   kms_geometric_clock_fit_receipt
L4   support_visible_bw_covariance_receipt
L5   ordered_cut_pair_rigidity_receipt
L6   lorentz_algebra_closure_receipt
L7   refinement_naturality_receipt
H0   conformal_h3_chart_receipt
H1   theorem_assisted_h3_population_receipt
H2   chart_blind_relational_bulk_audit
P0   bulk_localized_topological_defect_receipt
P1   production_particle_matter_receipt
CMB0 screen_cl_diagnostic
CMB1 finite_primordial_kernel_receipt
CMB2 physical_cmb_prediction_receipt
```

Mapping from current names:

```text
T0 -> C0a only
T1 -> partial C1
T2 -> L0 only
T3 -> H0
T5b -> H1
T6 -> H2
T7 -> P1
T8 -> CMB0
T9 -> CMB2
```

Do not let `final_phi == 0` imply `C0b`. Do not let a declared `2π` transition imply `L2-L7`.

---

## 2. Finite consensus theorem receipt (`C0b`)

### Current issue

`oph_fpe/dynamics/repair.py` accepts non-increasing moves (`delta_phi <= 0`) and can accept positive moves through Metropolis. That is useful for exploration but not the exact finite consensus theorem surface. `proof_certificate.py` currently promotes final settling directly to T0.

### Required split

Add two dynamics phases:

```python
phase: Literal["exploration", "theorem"]
```

Exploration may use annealing/Metropolis. The theorem phase must use:

```python
accepted = delta_phi < -strict_tol
```

Equal-score proposals are no-ops, not accepted repairs. Positive-score moves are forbidden.

### Extend `RepairEvent`

```python
@dataclass(frozen=True)
class RepairEvent:
    cycle: int
    phase: str
    node: int
    move_id: str
    touched_edges: tuple[int, ...]
    touched_phi_before: float
    touched_phi_after: float
    global_phi_before: float
    global_phi_after: float
    delta_touched_phi: float
    delta_global_phi: float
    quotient_hash_before: str
    quotient_hash_after: str
    accepted: bool
    theorem_eligible: bool
    reason: str
```

### New module

`oph_fpe/dynamics/consensus_certificate.py`

It must emit:

1. **Strict descent:** zero accepted theorem-phase moves with `delta_touched_phi >= -tol`.
2. **Disjoint commutation:** replay disjoint enabled repairs in both orders and compare quotient hashes.
3. **Local diamond:** for overlapping enabled repairs, apply both first moves, continue bounded repair, and require joinable quotient normal forms.
4. **Repair completeness:** terminal iff all visible overlap constraints are satisfied. Test both directions on small exhaustive fixtures and sampled large-run perturbations.
5. **Schedule confluence:** restart from the same saved quotient state under at least 16 randomized fair schedules; all terminal quotient hashes must match.
6. **Boundary-fiber uniqueness:** only when the boundary/sector is preserved and the selected fiber has a unique consistent extension.

Suggested default gate:

```text
strict_descent_violation_count == 0
disjoint_commutation_violation_count == 0
local_diamond_violation_count == 0
repair_completeness_violation_count == 0
unique_terminal_quotient_hash_count == 1
```

For noisy branches, emit a separate fair-block contraction receipt; never merge it into the exact receipt.

Implementation note, 2026-06-20: `oph_fpe/dynamics/consensus_certificate.py`
emits a fail-closed `FINITE_CONSENSUS_THEOREM_RECEIPT`. Ordinary BW array runs
are tagged as exploration traces, so they can pass the C0a finite-settle
diagnostic while C0b remains false with explicit missing replay/confluence
evidence. Setting `theorem_core.consensus_replay.enabled: true` now runs a
strict deterministic port-pair theorem normalizer with randomized schedule
replays and writes `finite_consensus_replay_report.json`; that C0b receipt is
for the finite array port-pair quotient only and does not certify C1 or L1-L7.

---

## 3. Finite observer record algebra (`C1`)

### Current issue

`core/records.py` commits a repeated Python tuple after `commit_cycles`. That is a useful engineering record, but it is not yet an explicit finite record algebra or checkpoint-continuation receipt.

### Minimal record algebra

For each observer subfederation `U`, construct an operator system generated by:

```text
P_r                 stable record projectors
Q_alpha             overlap/edge-sector projectors
X_(r,s,e)           Hermitian transition operators across visible interface e
Y_(r,s,e)           antisymmetric transition-current operators
J_e                 repair-current / mismatch-current operators
V_(a,b,tau)         port-a perturbation -> port-b response at lag tau
```

In a classical lane, `P_r` and `Q_alpha` are diagonal projectors. Nontrivial modular action requires the transition/current operators; a purely diagonal histogram algebra commutes with `rho` and has trivial modular dynamics.

### Extend record storage

Do not modulo or bin exact record IDs in theorem lanes. Store:

```text
record_hash
record_projector_id
local_port_id
neighbor_overlap_packet_hash
sector_id
commit_cycle
checkpoint_id
transition_from/to
lag
repair_delta
```

Binning remains allowed for visualization/diagnostics and must carry a collision report.

### Checkpoint receipt

Add `core/checkpoints.py`. A checkpoint contains:

```text
record algebra state
accessible state moments
external interface packet set
allowed future boundary schedule class
provenance hash
```

Train a continuation predictor on one subset of runs and score held-out continuation. Require it to beat shuffled-record and wrong-interface controls. This is the operational finite observer receipt.

---

## 4. Replace Gram-matrix `rho_C` with a MaxEnt cap state (`L1`)

### Current issue

Several `modular_probe.py` modes construct `rho_C` as `M M^T`, a feature Gram matrix, or a Gibbs state generated from an already-declared geometric transition. These are useful probes, not a finite realization of the paper's local MaxEnt cap state.

### New module

`oph_fpe/algebra/maxent_cap_state.py`

Input:

- a cap/collar support mask;
- local operator basis from the finite record algebra;
- empirical expectation values from training histories;
- no geometric transport target, cap tangent, H3 coordinate, or `2π` constant.

Solve the convex dual:

```math
rho_C(lambda) = exp(-sum_a lambda_a O_a) / Z(lambda)
```

with constraints:

```math
Tr(rho_C O_a) = c_a.
```

Output:

```text
rho_C
lambda vector
dual residual
PSD minimum eigenvalue
trace error
train moment error
held-out moment error
operator locality audit
geometry dependency audit
```

Default numerical gates:

```text
trace_error < 1e-10
min_eigenvalue > -1e-10
dual_residual < 1e-7
median heldout normalized moment error < 0.05
geometry_dependency_count == 0
```

The final thresholds should be frozen before the 1M run and accompanied by convergence plots.

### Strict endogenous mode

Add:

```yaml
bw:
  state_mode: maxent_record_operator_state
  prohibit_geometry_in_state_builder: true
  prohibit_declared_flow_in_state_builder: true
```

Cap membership and collar membership are allowed because the theorem is about a cap algebra. Cap tangent coordinates and the target Möbius flow are not allowed in the state builder.

---

## 5. Explicit collar recovery, not CMI alone (`L1` prerequisite)

### Current issue

The current collar report computes classical CMI on compressed packets and writes a Fawzi-Renner-style bound. It does not construct and test the finite recovery map. The current `cell_scaled` collar width also keeps `delta / l_UV` approximately constant, whereas the paper's double-scaling hypothesis requires:

```math
delta_N -> 0,
\qquad
delta_N / l_UV -> infinity.
```

### New collar scaling mode

For a spherical regulator with `l_UV ~ N^-1/2`, use for example:

```math
delta_N = c N^-1/4.
```

Then `delta_N -> 0` and `delta_N/l_UV ~ N^1/4 -> infinity`.

Config:

```yaml
bw:
  collar_width_mode: double_scaling
  angular_prefactor: 0.8
  angular_exponent: 0.25
```

The same keys may be supplied under `h3_support_profiles` for the denser
H3-population cap net.

### Recovery implementation

Add `bulk/collar_recovery.py`:

1. Build time-indexed or ensemble-indexed joint samples `(A,B,D)`; do not treat unrelated spatial nodes as IID repeats without an explicit exchangeability claim.
2. Construct the classical Markov/Petz map for diagonal packets and the matrix Petz map for operator states.
3. Apply it to held-out `rho_AB`.
4. Compare reconstructed and actual `rho_ABD` on trace distance and on held-out observable moments.
5. Emit units (`nats` or `bits`) consistently.

Receipt fields:

```text
median_cmi
p90_cmi
median_trace_recovery_error
p90_trace_recovery_error
median_observable_recovery_error
sector_conditioned_errors
collar_width
cell_scale
collar_to_cell_ratio
```

The theorem-facing gate is recovery error plus refinement, not CMI by itself.

---

## 6. Endogenous modular generator (`L2`)

Use two independent estimators and require agreement on observable action.

### A. MaxEnt modular generator

```math
K_rho(a) = -log(rho_C + a I).
```

Audit:

- several regularizers `a`;
- support projection consistency;
- nontrivial off-diagonal action;
- no geometry target in `rho_C`.

### B. Transition/Koopman generator

From observer-visible operator feature vectors `f_t`:

```math
C00 = E[f_t f_t^*],
C01 = E[f_t f_(t+tau)^*],
C11 = E[f_(t+tau) f_(t+tau)^*].
```

Whiten:

```math
T_tau = C00^-1/2 C01 C11^-1/2.
```

Take the polar factor `U_tau = polar(T_tau)` and derive:

```math
K_U(tau) = (i/tau) log(U_tau).
```

Implementation requirements:

- use Schur decomposition / stable matrix logarithm, not independent principal eigenphases;
- unwrap branches jointly across `tau, 2tau, 4tau`;
- report nonunitary defect before polar projection;
- compare `K_tau`, `K_2tau`, `K_4tau` on held-out observable action.

Suggested gates:

```text
multi_lag_action_disagreement < 0.10
regularizer_action_disagreement < 0.10
maxent_vs_koopman_action_disagreement < 0.15
geometry_dependency_count == 0
```

---

## 7. Infer the `2π` coefficient instead of declaring it (`L3-L4`)

The finite observable should be a **data-derived relation between modular time and the best support-visible cap transport parameter**.

For each cap, time, and held-out observable:

1. Evolve with the endogenous modular generator.
2. Search continuously over geometric cap-flow parameter `s` for the support-visible map that best reproduces the evolved matrix elements.
3. Record `s_hat(t)` without testing a named normalization during the fit.
4. Fit:

```math
s_hat(t) = kappa t + b.
```

5. Only after fitting, compare `kappa` with `1`, `pi`, `2pi`, and `4pi`.

Receipt:

```text
kappa_hat
kappa_95ci
intercept_hat
heldout_covariance_residual
wrong_scale_likelihood_ratios
cap_to_cap_variance
observable_to_observable_variance
```

Pass rule:

```text
2pi inside kappa_95ci
1, pi, and 4pi outside kappa_95ci
heldout covariance residual beats shuffled, no-repair, wrong-interface controls
same result on at least three consecutive regulator sizes
```

This converts `2π` from an input into a tested coefficient.

---

## 8. Ordered cut-pair rigidity and Lorentz algebra (`L5-L6`)

A single-cap fit is insufficient. Infer transformations from many ordered cap pairs.

### Ordered cut-pair checks

- preserve cap inclusion/order;
- preserve null/collar incidence;
- preserve cross-ratios on held-out cap quadruples;
- satisfy one-parameter subgroup composition;
- inverse consistency;
- cap-pair map independent of observable family.

### Lie algebra checks

Construct three independent rotation generators `J_i` and boost generators `K_i` from the inferred cap transformations. On the support-visible operator subspace, test normalized commutator residuals:

```math
[J_i,J_j] = epsilon_ijk J_k,
[J_i,K_j] = epsilon_ijk K_k,
[K_i,K_j] = -epsilon_ijk J_k.
```

Emit:

```text
JJ_residual
JK_residual
KK_residual
closure_rank
null_cone_preservation_error
group_composition_error
```

Require decreasing residuals under refinement and controls that fail in the expected direction.

Only `L0-L7` together should emit:

```text
OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1 = true
```

The wording remains “finite theorem-hypothesis and refinement instantiation,” not “proof of the continuum theorem by simulation.”

---

## 9. Refinement contract (`L7`)

### Regulator family

Use:

```text
4,096
65,536
262,144
1,048,576
```

with at least five independent seeds per size; eight is preferable. Treat seeds, not caps, as the top-level independent sampling unit.

### Keep physical quantities fixed

- same angular cap family;
- double-scaling collar width;
- same local operator labels;
- same observable holdout split policy;
- no threshold retuning after inspecting larger sizes;
- observer sampling sufficient to keep statistical error comparable.

### Replace the single negative slope

Fit each error to:

```math
epsilon(N) = epsilon_infinity + c N^-p.
```

Use hierarchical bootstrap over seeds, caps, and observables. Report confidence intervals for `epsilon_infinity` and `p`.

### Naturality checks

For explicit coarse-graining maps `C_(N->M)` test:

```math
C n_N ~= n_M C
C sigma_t^N ~= sigma_t^M C
C h_N ~= h_M C
C R_N ~= R_M C
```

where `n` is normal form, `sigma` modular flow, `h` holonomy, and `R` record readout.

Define:

```text
normal_form_naturality_error
modular_naturality_error
holonomy_naturality_error
record_naturality_error
```

Promotion rule:

- all mandatory controls pass at every size;
- no sign reversal in fitted convergence;
- `kappa` remains compatible with `2π` at every size after the first nondegenerate size;
- covariance, recovery, algebra, and naturality errors improve or plateau below frozen tolerances;
- at least three consecutive sizes pass;
- the 1M result is confirmatory, not the first tuned success.

---

## 10. Rebuild strict neutral bulk as a relational audit (`H2`)

### Status decision

Strict neutral third-person bulk is **not required** for the paper-route claim. The OPH theorem route explicitly uses the support-visible cap chart and reconstructs the H3 spatial chart from the Lorentz branch. Keep strict neutral only for the stronger claim:

> chart-blind observer records independently discover the same relational 3D/H3 continuation space.

Rename the lane to avoid suppressing the core theorem route:

```text
chart_blind_relational_bulk_audit
```

### Current failure mechanism

The current `NeutralObserverView` converts each observer into global histograms and spectra. It removes support coordinates, but it also removes local port direction, overlap incidence, lagged travel time, and causal response. A global weighted distance between those summaries is not guaranteed to preserve a local tangent manifold.

### Minimal neutral packet

For each observer and anonymous local port:

```text
stable record projector/hash
port-local overlap packet hash
port-local sector class
commit and checkpoint order
transition counts by port pair and lag
perturbation-response tensor R[a,b,tau]
repair-current tensor
first-passage / response time
counterfactual continuation score
```

Allowed:

- local port IDs as arbitrary gauge labels;
- observed overlap correspondences;
- recorded causal lags.

Forbidden:

- screen XYZ;
- cap axes or tangents;
- H3 coordinates;
- theorem H3 radius/depth;
- known icosahedral coordinates;
- geometry-derived proximity terms.

Run local-port permutation controls to prove gauge invariance.

### Relational reconstruction

1. Build the observer overlap nerve from shared packets.
2. Synchronize anonymous local port gauges from overlap correspondences.
3. Build a causal/Green kernel from perturbation-response and first-passage times.
4. Estimate local tangent rank from held-out response prediction, not only pairwise distance.
5. Select model order by cross-validated predictive likelihood/MDL.
6. Test coordinate-free geometry:
   - diffusion/heat-kernel dimension;
   - ball-volume growth;
   - thin triangles/Gromov hyperbolicity;
   - curvature consistency;
   - local isotropy/homogeneity;
   - H2/H3/H4/E2/E3/S2 held-out controls.
7. Compare recovered relational geometry with the theorem H3 chart only after the blind fit is frozen.

A rank-3 SVD candidate by itself is never a bulk receipt.

---

## 11. H3 population and the definition of “emergent bulk”

Adopt these explicit product claims:

```text
paper_route_h3_chart_established
paper_route_h3_population_established
chart_blind_relational_bulk_established
```

`paper_route_h3_population_established` should require:

- `L0` at minimum, preferably the full `L1-L7` contract;
- persistent observer objects;
- held-out H3 localization;
- H3 beats shuffled incidence;
- non-boundary localization for a declared fraction/count;
- stability across seeds and refinement;
- no use of object labels fitted on the evaluation set.

Do not call the theorem chart “neutral.” It is intentionally theorem-assisted.

---

## 12. Particle/matter gate (`P0-P1`)

Screen holonomy clusters are precursors, not particles. A production particle receipt needs:

1. localization in the shared H3 bulk chart;
2. persistent worldline with a causal speed bound;
3. topological/sector charge stable under contractible path deformations;
4. transport reproducibility;
5. fusion with charge conservation;
6. scattering reproducibility on held-out initial states;
7. survival under refinement and observer resampling;
8. failure of shuffled, boundary-only, and transient-defect controls.

Positive-geometry/amplituhedron kernels may enter only after `P0`, as an accelerator or exact representation of interaction/scattering readouts.

---

## 13. Positive geometry and P/N resonance

### Positive geometry

Keep it completely outside `C0-H2`. The current fail-closed dispatcher policy is correct.

It may affect trusted observables only after:

```text
sector recognition
boundary compiler equivalence
readout equivalence
fail-closed reproducibility
```

Until then it emits diagnostic receipts and must not change H3 fits, neutral ranks, repair outcomes, or bulk gates.

### P/N resonance

Keep it classified as numeric/theorem-side replay. Promote only when the simulator itself emits:

- the source `alpha_U(P)` proof record;
- a target-independent fixed-point/uniqueness certificate;
- the global `F(N)` capacity closure if the claim uses it;
- refinement/naturality receipts;
- a dependency audit showing no measured endpoint was read upstream.

---

## 14. CMB sequencing

Keep current CMB products as `CMB0` diagnostics. Do not promote because a shape correlation or CAMB curve looks good.

Before `CMB2`, derive from finite runs:

```text
repair clock eta_R
parent-collar response kernel B_A(k,a)
finite anomaly/source density rho_A(k,a)
freezeout/initial hypersurface
physical normalization and units
photon-baryon transfer inputs
no-data-use dependency graph
official likelihood receipt
```

The immediate implementation target is the time-resolved parent-collar response and repair clock, not a higher-ell screen transform.

---

## 15. Concrete file plan

### Modify

```text
oph_fpe/dynamics/repair.py
oph_fpe/core/records.py
oph_fpe/bulk/collar_state.py
oph_fpe/bulk/markov_collar.py
oph_fpe/bulk/modular_probe.py
oph_fpe/bulk/neutral_bulk.py
oph_fpe/bulk/proof_certificate.py
oph_fpe/scale/refinement_report.py
```

### Add

```text
oph_fpe/dynamics/consensus_certificate.py
oph_fpe/core/checkpoints.py
oph_fpe/algebra/operator_system.py
oph_fpe/algebra/maxent_cap_state.py
oph_fpe/bulk/collar_recovery.py
oph_fpe/bulk/endogenous_modular.py
oph_fpe/bulk/inferred_cap_flow.py
oph_fpe/bulk/lorentz_rigidity.py
oph_fpe/bulk/neutral_relational.py
oph_fpe/scale/refinement_naturality.py
oph_fpe/bulk/theorem_contract.py
```

### Add tests

```text
tests/test_strict_repair_phase.py
tests/test_local_diamond_certificate.py
tests/test_schedule_confluence.py
tests/test_record_operator_algebra.py
tests/test_checkpoint_continuation.py
tests/test_maxent_cap_state.py
tests/test_petz_recovery.py
tests/test_endogenous_modular_generator.py
tests/test_inferred_kms_clock.py
tests/test_ordered_cut_pair_rigidity.py
tests/test_lorentz_algebra_closure.py
tests/test_refinement_naturality.py
tests/test_neutral_relational_bulk.py
```

---

## 16. Example top-level config

```yaml
theorem_contract:
  version: v1

  consensus:
    exploration_cycles: 16
    theorem_cycles: 48
    strict_delta_tol: 1.0e-12
    schedule_replays: 16
    sampled_diamond_pairs: 10000

  records:
    exact_packet_ids: true
    allow_binning_in_theorem_lane: false
    checkpoint_horizons: [1, 2, 4, 8]
    continuation_holdout_fraction: 0.25

  cap_state:
    mode: maxent_record_operator_state
    operator_basis:
      - record_projector
      - sector_projector
      - overlap_transition_x
      - overlap_transition_y
      - repair_current
      - perturb_response
    prohibit_geometry_in_builder: true
    regularizers: [1.0e-8, 3.0e-8, 1.0e-7]

  collar:
    width_mode: double_scaling
    angular_prefactor: 0.8
    angular_exponent: 0.25
    explicit_recovery: true

  modular:
    estimators: [maxent_log_rho, koopman_polar]
    lags: [1, 2, 4]
    infer_geometric_parameter: true
    named_scale_test_after_fit: [1.0, 3.141592653589793, 6.283185307179586, 12.566370614359172]

  rigidity:
    ordered_cap_pairs: 128
    heldout_cap_quadruples: 256
    lie_algebra_generators: 6

  refinement:
    sizes: [4096, 65536, 262144, 1048576]
    seeds_per_size: 8
    fit_model: epsilon_inf_plus_power
    coarse_graining_naturality: true

  neutral_audit:
    enabled: true
    mode: relational_port_response
    prohibit_screen_coordinates: true
    local_port_permutation_controls: true
```

---

## 17. Recommended implementation order

1. **Receipt/schema cleanup:** split branch replay from theorem receipt; split settling from confluence.
2. **Strict theorem phase and confluence ledger.**
3. **Exact record/checkpoint data and unbinned transition logs.**
4. **Double-scaling collars and explicit recovery.**
5. **MaxEnt operator state and independent Koopman generator.**
6. **Continuous `s_hat(t)` fit, `kappa` inference, covariance controls.**
7. **Ordered cut-pair and Lorentz algebra closure.**
8. **Refinement naturality across 4k/64k/256k/1M.**
9. **Retest theorem-assisted H3 population using the stronger L-contract.**
10. **Build the optional chart-blind relational audit.**
11. **Only then promote particle and physical-CMB work.**

This order prevents downstream cosmology or interaction kernels from compensating for a missing finite Lorentz/modular theorem contract.
