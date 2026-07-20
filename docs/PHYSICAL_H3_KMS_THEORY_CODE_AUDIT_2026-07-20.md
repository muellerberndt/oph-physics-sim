# Physical H3/KMS simulator audit — 2026-07-20

## Status and scope

This is a read-only audit of the OPH paper stack and the simulator at
`oph-physics-sim` commit `69b2034964417fe320bc36dab3baaf1114628139`. The paper
sources were not edited. The source-line findings below remain pinned to that
baseline. Post-audit working-tree updates have since implemented several strict
finite scaffolds and fail-closed gates; their precise scope and remaining
production-integration gap are recorded in Sections 4 and "Integration status."
Other changes described under **Required correction** remain an integration
checklist unless explicitly marked implemented and tested.

The immediate conclusion is fail-closed:

- the existing run archive does **not** instantiate the paper's finite cap-normal
  BW object;
- its history-Koopman clock is a record-dynamics surrogate, not the modular
  automorphism of the support-visible prime geometric cap algebra;
- its scale controls do not all score one frozen intervention;
- its H3 fit lacks equal-footing E3 and E4 controls;
- its H3 chart is repeatedly promoted as a spatial/event chart even though the
  compact paper says H3 is a timelike-frame fiber;
- the available 4k/16k/64k artifacts do not form a three-rung frozen family, so
  they cannot support stable branch retirement;
- no important 64k/256k campaign should be run until a simulator-native
  preflight fails closed on all of these points.

The negative clock results are still useful. They say that the current record
surrogate does not select the desired normalization. They do not decide whether
a paper-conforming cap-algebra construction would do so.

## The contract that the papers actually impose

### 1. The target is a prime geometric cap algebra, not the full record state

The compact paper separates the geometric subnet from record summaries,
pointer registers, and interface-inert auxiliaries at
`paper/recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex:3680-3707`.
It then says that BW acts on a support-visible scaling-limit geometric cap pair,
not literally on the finite regulator matrices, at lines `3710-3714`.

A campaign-cap state must therefore export, for every cap and rung:

1. a concrete noncommutative finite cap algebra `M_r(C)`;
2. an explicit quotient/projection to the prime support-visible geometric subnet;
3. a faithful density matrix on that algebra;
4. `K_r,C = -log(rho_r,C)` on the same algebra;
5. evidence that record/pointer/repair auxiliaries did not enter the geometric
   generator.

Central record probabilities alone give a commutative algebra and a trivial
modular automorphism. A matrix built by placing classical records into rows and
columns does not become the paper's cap algebra merely because it is
non-diagonal.

### 2. The finite BW certificate is an eight-clause tower object

The complete `FiniteCapBWCertificate` is enumerated at compact-paper lines
`3793-3821`. It requires:

- cap and point meshes, nondegenerate radii, incidence, unit normals, and
  refinement compatibility;
- ordered, separated, oriented BW frame points;
- cap algebras with isotony, support-order separation, and a faithful
  support-action quotient;
- an exact modular-compatible reference tower and compact-time regularized
  bound;
- mixed-GNS Cauchy and support-covariance residuals, including both signs of
  time;
- identity, inverse, group law, equicontinuity, held-out complex cross-ratios,
  and an orientation witness;
- an independently parameterized geometric KMS strip residual at `2*pi`;
- nontriviality and either a wrong-beta gap or a noncentral generator-distance
  bound.

The same definition explicitly states that bare finite OPH consensus does not
imply this certificate (`3822-3826`). The consensus paper agrees: bare consensus
does not supply a null cone, round-cap support chart, conformal transport, time
orientation, or H3 observer-frame space
(`paper/reality_as_consensus_protocol.tex:237-245`).

The simulator's issue-308 module is a useful verifier: it recomputes C1-C8 and
ignores caller pass flags
(`oph_fpe/bulk/bw_certificate_308.py:34-108`). However, the main simulation
pipeline does not construct a simulator-native `BWRec_r`; the verifier is
normally exercised on supplied payloads or test fixtures. A verifier without a
producer does not instantiate the branch.

### 3. `2*pi` must be selected against an independent geometric clock

The compact paper is unambiguous. Its geometric parameter `s` is defined by an
ordered BW frame and is not modular time `t` (`3785-3790`). Modular KMS by itself
cannot choose `2*pi`, because rescaling the modular group produces any positive
inverse temperature (`3846-3849`). The theorem's conclusion is
`sigma_t = alpha_lambda(2*pi*t)` only after all certificate clauses pass
(`4061-4104`).

Consequently, a valid candidate comparison has this causal structure:

```text
one frozen source state and one frozen physical intervention
    -> one held-out response data set
    -> independently derived geometric coordinate s
    -> score kappa in {1, pi, 2*pi, 4*pi} as competing maps s = kappa t
```

Candidate `kappa` may affect a prediction. It may not affect which edges are
perturbed, the intervention dose, the endpoint side, the repair RNG stream, or
which response rows exist.

### 4. Carrier refinement and the support regulator are different towers

The compact paper requires finite algebras, faithful reference states,
refinement embeddings, state-preserving conditional expectations, and
compatible states at lines `4249-4272`. The microphysics paper gives the same
factor/detail construction and conditional expectations at
`paper/screen_microphysics_and_observer_synchronization.tex:1750-1785`; it also
warns that a finite cellulation or stable extrapolation alone is only a chart
result (`1803-1808`).

The later carrier-level architecture clarification makes an important type
distinction. Campaign rungs `4096`, `16384`, `65536`, and `262144` are exact
numbers of microscopic echosahedral carriers. A geodesic S2 mesh is a separate
support/calibration regulator. Its face counts need not equal the carrier
counts, and replacing a requested carrier count by the nearest geodesic face
count would silently change the experiment.

Independent Fibonacci point clouds are not a nested support-regulator tower.
That remains true for the archived campaign artifacts and current production
array path. The prior `graph.py` geometry defect has been repaired for the
separately typed support scaffold, but that does not construct a carrier
federation or its refinement maps.

`oph_fpe/core/icosahedral.py` now implements the geometric part of the paper's
regulator honestly:

- a true outward-oriented base icosahedron and projected shared-edge midpoint
  refinement, with shared midpoints deduplicated;
- exact spherical triangular-cell counts
  `F_m = 20 * 4**m`, alongside `E_m = 30 * 4**m` and
  `V_m = 10 * 4**m + 2` (`icosahedral.py:259-269`);
- persistent old vertices, vertex-parent support, child-to-parent face lineage,
  four children per parent, and deterministic geometry/map hashes;
- a cell-dual NetworkX graph and a large-array adapter exposing the same mesh
  topology (`374-501`);
- childwise-constant embeddings and spherical-area-weighted conditional
  expectations on a **commutative cell-value scaffold**, preserving the
  normalized spherical-area reference state (`39-109`, `174-251`);
- strict `core/graph.py` integration: arbitrary counts are rejected by default,
  while `nearest`, `floor`, or `ceil` require an explicit policy
  (`core/graph.py:38-70`).

For the support regulator only, `nominal_campaign_rung_mapping()` reports the
geodesic face-count brackets:

| nominal label | lower exact `F_m` | recommended/upper exact `F_m` |
|---:|---:|---:|
| 4k (`4096`) | 1280 | 5120 |
| 16k (`16384`) | 5120 | 20480 |
| 64k (`65536`) | 20480 | 81920 |
| 256k (`262144`) | 81920 | 327680 |

These values must not be used as substitutions for the exact campaign carrier
rungs. The geometry implementation is covered by
`tests/test_icosahedral_geometry.py`: exact closed-sphere topology, projected
midpoint nesting, area partition and lineage, state-preserving scaffold maps,
multilevel composition, deterministic hashes, graph/array agreement, strict
count handling, and nominal-rung brackets. The focused test result is
`15 passed in 0.56s`.

This is a meaningful support-regulator repair, but it is not the compact
paper's multiresolution certificate or a carrier-refinement receipt. The
implementation deliberately emits
`PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE = false` because it does not yet
instantiate:

- finite detail algebras;
- faithful detail states;
- local presentation circuits;
- noncommutative state-preserving conditional expectations.

Nor is the new mesh a valid microscopic source. `bw_array.py` still imports and
uses `fibonacci_sphere_points` (`77`, `207`, `3308`) and uses global support
points/cells as production state rows. The new federation module instead
materializes one finite 12-port carrier per exact carrier ID and typed seam
collars between carriers, but the legacy production engine is not yet realized
through it. Carrier refinement naturality and the concrete-to-abstract source
realization therefore remain false.

### 5. H3 is a frame fiber; an event base is a separate construction

The compact paper says H3 is the hyperboloid of future unit timelike frames,
not a spatial event slice (`6832-6852`), and repeats this as a proposition at
`7067-7087`. It requires event population, separation, a rank-four response
frame, and Poincare overlap cocycles (`E1`-`E4`) at `6920-6942`; without those,
H3 dimension three promotes nothing (`7180-7212`).

Record localization on H3 is itself conditional. It needs a compact domain,
point-local response factorization, positive observability constant, fixed
calibration/error budget, a finite net, and a residual minimizer
(`5644-5669`). The paper says explicitly that this does not derive point
locality, mixture resolution, a neutral bulk, stress, or Einstein entry
(`5679-5691`). It also leaves the H3 curvature radius unfixed in physical units
(`5275-5289`) and starts the localization model with arbitrary `R_H > 0`
(`5294-5302`).

A correct simulator should therefore emit two disjoint receipts:

- `H3_FRAME_FIBER_CHART_RECEIPT`: Lorentz/frame kinematics only;
- `EVENT_MANIFOLD_3P1D_RECEIPT`: semantic record events satisfying population,
  separation, rank-four, causal ancestry/time orientation, translation, and
  overlap-cocycle gates.

The first must never be accepted as the second.

### 6. Geometry controls must be equal-footing alternatives

For this campaign, `H3`, `S2`, `E3_control`, and `E4_control` mean candidate
geometries fitted to identical held-out event-response rows. The `E3_control`
and `E4_control` labels are Euclidean three- and four-dimensional controls;
they must not be confused with the paper's event receipts `mathsf(E3)` and
`mathsf(E4)`.

All four candidates must have:

- identical training and held-out row IDs;
- identical preprocessing and nuisance-variable policy;
- matched effective capacity or an explicit complexity penalty;
- one frozen hyperparameter-selection budget;
- finite scores; missing or nonfinite controls are failures, never passes.

An independently measured curvature radius and a domain spanning a material
fraction of that radius are essential. On a small domain H3 approaches E3 as
`R_H` grows, so no finite fit can identify negative curvature merely by naming
the H3 chart or fixing `R_H = 1`.

## Source-paper problems uncovered by the audit

These are paper-side specification defects or ambiguities. They are documented
here without editing the theory repository.

### A. The producer theorem cites an axiom clause that does not exist

The producer theorem consumes “the mixed-GNS state Cauchy clause of Axiom
maxent” and says that clause supplies cap-family-uniform modular convergence
(`4564-4588`). But Axiom `maxent` at `991-998` contains local homogeneous
MaxEnt constraints and refinement closure; it contains no mixed-GNS Cauchy
clause, no plus/minus-time modular convergence, and no cap-family-uniform
residual.

Required theoretical correction: define the mixed-GNS clause as a separate
assumption/receipt, with an operational finite residual, or add it explicitly to
the axiom and justify why local MaxEnt plus closure implies it. The current
cross-reference cannot be used as an implementation specification.

### B. The modular cross-ratio receipt risks embedding the answer

The cross-ratio receipt defines
`cr_r = exp(2*pi*t_r)` from a “cap modular clock”
(`4452-4476`), immediately before a separate independently normalized KMS
receipt (`4477-4481`). If the cross-ratio data are later used to establish the
geometric parameter against which `2*pi` is selected, the definition has
already inserted the target coefficient.

Required theoretical correction: derive oriented geometric cross-ratios and
the parameter `s` from incidence, ordered frames, and held-out geometry without
using `pi` or the modular clock. Keep those rows disjoint from the KMS holdout.
Only afterward compare `s` with modular `t`.

### C. H3 localization is used both as event position and as frame data

The event section first correctly separates the event base from H3 frame fibers
(`6832-6852`). It then builds event boxes as
`[tau +/- sigma] x B_H(X_hat,rho)` (`6867-6886`) and treats their hyperbolic
coordinate as the spatial component of a four-chart (`6945-6983`). Later it
says that the same H3 localization chart is fiber data, not event-base data
(`7067-7087`). A velocity/frame coordinate and a clock reading do not determine
event position; translations or an integrated position cocycle are missing.

Required theoretical correction: introduce a source-derived translation/event
position coordinate from record ancestry and the Poincare cocycle, prove its
rank and chart separation, and reserve H3 exclusively for the unit-timelike
frame variable. Until then, the event-manifold theorem cannot be used to promote
an H3 response fit to spatial `3+1D` experience.

### D. The numerical curvature scale is not derived

The paper correctly says it does not fix `R_H` in physical units
(`5280-5289`), but a campaign that compares H3 with E3 needs this scale to be
identifiable. A free large `R_H` makes H3 locally indistinguishable from E3.

Required theoretical correction: add a source-derived curvature-scale receipt
and a range/identifiability condition, or explicitly downgrade the finite result
to an H3-coordinate compatibility diagnostic.

## Full-ladder cross-paper consistency audit

The three attached papers do not specify one linear emergence ladder in which
finite repair automatically produces observers, H3 automatically becomes
spacetime, and spacetime automatically produces the Standard Model and
Einstein gravity. They specify a partially ordered set of conditional branches.
Several bridges have definitions or necessary conditions, but no theorem that
constructs an inhabited target from the simulator's current source objects.

The compact paper's own dependency ledger makes this explicit
(`paper/recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex:294-329`):

- `D1` is finite repair/confluence/refinement;
- `D2` adds collar Markov and entropy hypotheses rather than deriving them from
  `D1`;
- `D3a-D3h` are the modular/BW/geometry branch;
- `D4`, `D4b`, `D4c`, and `D4d` separately add null stress, semantic events,
  local stress/assembly, and stationarity;
- `D5` is a conditional Einstein branch whose realized nonemptiness is still
  work in progress, and `D6` leaves global closure open;
- `D7-D9` form a distinct gauge/SM branch. `D9` requires an explicitly realized
  one-generation plus one-Higgs matter package; that package and MAR are called
  explicit inputs at `325-329`, not outputs of H3/KMS;
- `D10` needs an external pixel ratio and conventions for quantitative closure.

The resulting bridge audit is:

| Bridge | What the papers actually define | Is a source-to-target producer proved? | Audited simulator baseline | Verdict |
|---|---|---|---|---|
| Physical/software patches -> repair consensus | The screen paper types a patch as `(A_i,rho_i,R_i,interfaces,U_i,Chk_i)` (`screen_microphysics_and_observer_synchronization.tex:92-107`); the consensus paper defines a finite constraint/repair code but warns that it contains no Lorentz or Einstein structure (`reality_as_consensus_protocol.tex:216-246`). | Only under the finite repair, fairness, confluence, and refinement assumptions. | `C0_finite_consensus_theorem` is replayed from theorem-phase evidence (`theorem_contract.py` baseline `243-247`). | A legitimate finite stage if its exact premises and replay lineage pass; it entails no observer, H3, SM, or gravity claim. |
| Repair consensus -> operational observer | The screen paper requires bounded interface, rereadable records, self-read/internal-state estimation, behavior changed by readback, shuffled controls, and checkpoint continuation (`170-175`, `2104-2151`). The consensus paper independently requires a persistent predictive module and predictive mutual information for the `observer-like` label (`2951-3001`). | **No.** Neither paper proves that a nonzero population satisfying this test emerges from generic repair consensus. | `record_algebra` passes from counts of support nodes, readout hashes, and transition histories alone (`theorem_contract.py` baseline `90-107`, `249-253`). | Missing producer. A row with a hash and history is not yet a self-reading observer. |
| Operational observer -> stable record/event tower | The screen paper distinguishes abstract record, physical support, and carrier (`109-129`) and gives conditional checkpoint continuation, not automatic record identity across every implementation. | Conditional only; implementation equivalence and continuation need explicit maps and error bounds (`152-168`). | Observer rows are treated as a record algebra without carrier/support equivalence, predictive control, or checkpoint-transport evidence. | Missing typed carrier-to-record and refinement-lineage receipts. |
| Consensus/records -> prime geometric algebra | The compact paper defines the prime geometric subnet by excluding record, pointer, repair, and auxiliary layers (`3680-3714`). | A quotient/producer is attempted, but its theorem invokes a nonexistent mixed-GNS Cauchy clause (`4564-4588` versus MaxEnt axiom `971-998`). Records do not themselves become the geometric algebra. | Record/repair features feed the MaxEnt state and Koopman generator. | Type error and missing producer; observer records and geometric cap algebra must fork and meet only through an explicit response channel. |
| Prime geometric algebra -> multiscale modular object | Faithful cap states, embeddings, state-preserving conditional expectations, mixed-GNS convergence, and compatible nested states are required (`4249-4272`). | Conditional; the cited producer premise is incomplete. | The new icosahedral module supplies exact nested mesh/cell lineage and a state-preserving **commutative area scaffold**, but production still uses Fibonacci points and no finite detail algebra/state tower exists. | Geometric prerequisite partially implemented; paper-level modular tower not instantiated. A nominal rung label is not a certificate. |
| Modular object -> BW normalization and `2*pi` | An independently derived geometric parameter `s` and all C1-C8 clauses are required (`3785-3826`); KMS alone cannot choose `2*pi` (`3846-3849`). | Conditional finite certificate only. The cross-ratio construction may already insert `2*pi` (`4452-4482`). | The history-Koopman scale fit competes against labels, declared replay supplies a parallel branch, and candidate scale changes intervention generation. | Current clock failure is real for that surrogate, but the surrogate is not a faithful test of the theorem object. |
| BW -> Lorentz/H3 frame kinematics | Cap-normal rigidity and Lorentz algebra yield H3 as the future-unit-timelike **frame fiber** (`6832-6852`, `7067-7087`). | Conditional on the BW/rigidity/refinement receipts. | H3 is implemented and promoted as a spatial/bulk chart. | The chart may be a frame-kinematics diagnostic; it is not an event base or spatial slice. |
| Frame fiber + records -> localized record germs | The H3 localization theorem requires point-local factorization, positive observability, fixed calibration, finite-net coverage, and residual minimization (`5644-5691`). | Conditional; it explicitly does not derive point locality, mixture resolution, neutral bulk, stress, or Einstein entry. | H3 response-fit/object-population flags are accepted with post-hoc response coordinates and incomplete controls. | Missing localization premises and equal-footing alternative-geometry tests. |
| Localized records -> semantic spacetime events | The compact paper requires event population, separation, rank-four response frame, and Poincare overlap cocycles (`6920-6942`). | Only conditionally, and the text still lacks a clean source-derived translation/position producer: it first uses H3 in event boxes (`6867-6983`) and later says H3 is fiber-only (`7067-7087`). | `B3_observer_facing_3p1d_experience` is promoted from modular time plus H3 chart/response (`theorem_contract.py` baseline `407-410`, `514-536`). | Unjustified jump. H3 dimension three plus a clock is not a 3+1D event manifold (`7180-7212`). |
| Semantic events -> kinematic metric/tetrad/connection | `D4b` conditionally supplies a semantic event manifold and local conformal/Lorentz chart after E1-E4; downstream assembly receipts are separate (`dependency ledger 312-314`). | Conditional, not an output of an H3 regression alone. | No semantic E1-E4 stage exists in the audited theorem contract before its gravity stages. | Missing common-domain event atlas and overlap-cocycle producer. |
| Repair/collar/topology -> compact gauge branch | `D7` conditionally reconstructs compact gauge data from separate refinement/sector/transport premises (`dependency ledger 317`); the consensus paper likewise presents a broader conditional bridge (`reality_as_consensus_protocol.tex:3046-3075`). | Conditional. It is a branch from the repair/collar/refinement structure, not a consequence of winning H3. | No typed gauge-stage chain in `theorem_contract.py`. | Not instantiated by this campaign. Geometry-fit success could not discharge it. |
| Compact gauge branch -> SM quotient/dynamics | `D8-D9` require MAR and an explicit realized chiral one-generation plus one-Higgs package (`dependency ledger 318-325`). The paper later concedes that physical realization, masses, couplings, and full dynamics are open (`12235-12269`). | The classification is conditional; the matter/Higgs source producer is absent. | `bw_array.py` baseline emits `SM_QUOTIENT_GATE_RECEIPT=false`, `SM_TARGET_CONFORMANCE_DIAGNOSTIC=false`, and `PHYSICAL_STANDARD_MODEL_FROM_SCREEN_RECEIPT=false` when no explicit candidate is supplied (`6015-6031`). | No SM emergence result exists. Supplying the known target as a default candidate would be target injection, not recovery. |
| Event metric + modular/entropy/stress data -> Einstein equation | `D4-D5` require null-stress identification, entropy/area response, small-ball assembly, all-timelike coverage, stationarity, and conservation before the tensor equation (`dependency ledger 311-316`). | A conditional theorem chain is stated, but an inhabited common-domain tower is explicitly WIP (`12258-12269`). | Gravity stages consume sidecars/manifest booleans; visualization assays take configured stress coupling and curvature radius (`bw_array.py` baseline `2536-2684`). | Diagnostic only. Configured coupling/curvature cannot be evidence that the simulator derived either. |
| Einstein equation -> physical metric, `G`, and `Lambda` | Scale fixing, Newton coupling, conservation, connectedness, and global closure are separate receipts; `D6` remains open and `D10` uses an external pixel ratio. | No complete source-only quantitative closure theorem. | No provenance-safe, unit-calibrated producer from patch dynamics to physical `G`, `Lambda`, and metric scale. | Not instantiated; a dimensionless H3 radius or fitted attraction is not physical metric emergence. |

There is also an internal presentation inconsistency worth fixing in the theory
stack. The compact paper's later discussion says the spatial dimension follows
from `dim H3 = 3` (`11750-11767`), while its theorem and countermodel sections
correctly say H3 is frame-fiber data and its dimension alone implies no event
base (`7067-7087`, `7180-7212`). The latter, more cautious statement must govern
the simulator contract.

### Proposed typed DAG/stage contract

The simulator needs branch-aware, artifact-backed stage types rather than a
flat collection of booleans whose names can be OR-promoted. A minimal contract
is:

```text
C0 CarrierPatchFederation
  -> C1 FiniteRepairConsensus
       -> O0 OperationalSelfReadingObserver
            -> O1 StableRecordEventTower --------------------------+
       -> Q0 PrimeGeometricQuotient                                |
            -> Q1 NestedCapStateTower                              |
                 -> Q2 IndependentC1ToC8BWCertificate              |
                      -> Q3 LorentzH3FrameFiber --------------------+-> X0 LocalizedRecordGerms
                                                                    -> X1 SemanticEventManifold
                                                                         -> X2 KinematicMetricConnection
                                                                              -> R0 NullStressCharge
                                                                              -> R1 EntropyAreaSmallBall
                                                                              -> R2 TensorEinsteinEquation
                                                                              -> R3 PhysicalMetricScaleGLambda

C1 + independent CollarMarkovTopologyRefinement
  -> G0 SectorTransportCategory
       -> G1 CompactGaugeReconstruction
            -> G2 RealizedMatterHiggsMARWitness
                 -> G3 StandardModelQuotient
                 -> G4 PhysicalGaugeCurrentAndDynamics
```

There is deliberately no arrow from `O1` to `Q0`, from `Q3` directly to `X1`,
from `Q3` to `G3`, or from `X2` directly to `R2`. Those are precisely the
unjustified jumps in the current narrative or implementation.

Each receipt must contain at least:

- an exact schema/version and claim tier;
- hashes of all primitive input artifacts and upstream receipts;
- a producer identifier and code/config family hash;
- source/target object types and explicit construction maps;
- train/calibration/holdout row IDs and leakage audit where statistical;
- units and forbidden-input audit where scale or coupling is claimed;
- quantitative residuals, thresholds frozen before the run, and negative
  controls;
- `passed=false` when a field, control, upstream receipt, or hash is absent.

The highest aggregate claim must be computed by following only these typed
edges. Aliases, declared branch replay, fixture booleans, a known target
candidate, or a visualization sidecar may be retained as diagnostics but can
never satisfy an upstream stage.

## Simulator findings

### Critical: the wrong-scale candidates are different experiments

In `oph_fpe/bulk/modular_response_kernel.py`, the primary response matrix is
generated with `transport_scale` and every wrong scale can trigger a new call to
`_perturb_resettle_matrix` (`551-731`). Inside that function the candidate scale
changes transported supports and is passed into the physical perturb/resettle
simulation (`826-894`). It then changes:

- perturbation dose in `modular_amount` mode (`997-1012`);
- selected edge support (`1014-1027`, `1238-1296`);
- endpoint side for generator-based modes (`1028-1035`, `1203-1235`);
- transported readout support (`866-879`).

Thus `1`, `pi`, `2*pi`, and `4*pi` are not hypotheses scored against one frozen
response. They can create different data. This invalidates the scale-selection
interpretation and contradicts the retained June status note, which says the
wrong-normalization controls keep the same perturbation amount
(`handoffs/pro_handover_oph_physics_sim_20260619/code/oph-physics-sim/docs/modular_response_h3_status_20260605.md:219-230`).

Required correction:

1. choose and hash one scale-free intervention support, endpoint action, dose,
   RNG streams, and repair trajectory;
2. cache one raw response tensor;
3. allow candidate scale only in the downstream prediction map;
4. export intervention and held-out-row hashes for every candidate;
5. fail unless all candidate hashes are byte-identical.

### Critical: the current “endogenous generator” is not a cap modular generator

`history_transition_density` constructs a Gram density from packet histograms,
record signatures, commit/stability masks, repair loads, modular time, and S3
fields (`oph_fpe/bulk/modular_probe.py:297-368`). The Koopman path uses the same
kind of full-record feature series (`770-858`, `2643-2670`), whitens it, takes a
polar unitary, and logs that unitary. This can be a useful record-transition
diagnostic, but it does not establish:

- a representation of the prime cap algebra;
- equality to the modular automorphism of a faithful cap state;
- isotony or support-order faithfulness;
- a modular-compatible refinement tower;
- mixed-GNS convergence;
- noncentrality on the geometric algebra.

The MaxEnt path has the same type error. Its receipt calls a state made from
record/history operators a cap state and only notes that cap support was
externally selected (`oph_fpe/algebra/maxent_cap_state.py:25-48`). Its features
are record, commit, repair, mismatch, and S3 fields (`140-165`).

The summary then labels every non-declared/direct finite row set as an
endogenous generator and can emit `ENDOGENOUS_MODULAR_GENERATOR_RECEIPT`
(`oph_fpe/bulk/modular_probe.py:1339-1443`). The theorem-contract validator
checks the emitted mode, row count, finite median, and booleans, but not the
algebraic construction (`oph_fpe/bulk/theorem_contract.py:1220-1269`).

Required correction: preserve these products as
`RECORD_KOOPMAN_SURROGATE_DIAGNOSTIC`, force the paper-level endogenous modular
receipt false, and build a separate cap-algebra producer before enabling L2/L3.

### Critical: declared replay can promote observer `3+1D`

`_observer_modular_experience_report` accepts either declared BW/KMS replay or
the finite clock as the Lorentz-clock gate, accepts several intermediate H3
aliases, and emits observer `3+1D` without any event-base E1-E4 packet
(`oph_fpe/scale/bw_array.py:4282-4377`). The postprocessor reconstructs the same
gate from cached booleans and can re-emit it after a selected refit
(`oph_fpe/pipelines/oph_universe.py:1689-1745`).

Required correction: branch replay remains a diagnostic only; physical clock
promotion requires the strict cap-algebra clock; H3 promotion requires only the
final candidate receipt; `3+1D` additionally requires the independent semantic
event-manifold receipt.

### High: H3 is encoded as a spatial/bulk chart

The chart module names H3 a “3D spatial chart” throughout and emits a
`PAPER_THEOREM_3D_BULK_CHART_RECEIPT`
(`oph_fpe/bulk/conformal_spatial_chart.py:18-55`, `60-214`). The emergence
aggregator can set `bulk_3d_established` directly from object population in H3
(`oph_fpe/scale/bw_array.py:2020-2053`). This conflicts with the paper's strict
base/fiber separation.

Required correction: rename/retier the H3 output as a frame-fiber chart,
deprecate the bulk-chart receipt, and require an event-position/cocycle producer
before any spatial or `3+1D` receipt.

### High: missing controls pass open

The H3 stage gates explicitly accept absent or nonfinite no-perturbation,
shuffled, S2, and wrong-scale metrics (`oph_fpe/bulk/h3_response_fit.py:1567-1631`).
No E3 or E4 model is present. Material-feature absence also defaults to zero,
which passes.

Required correction: require finite same-shape H3, S2, E3, E4, no-flow,
shuffle, and wrong-scale rows; require declared margins; fail on any absence,
shape mismatch, duplicated row, or nonfinite score.

### High: the object called “prime geometric” is a post-hoc response projection

`oph_fpe/bulk/prime_geometric_response.py:12-35` labels checkpoint class,
stability, and S3 sector class “prime geometric.” It then groups response
columns and applies SVD after the response tensor has been built (`38-167`). It
does not construct `A_geo`, `rho_C`, `K_C`, or a BW certificate. Its name should
not be used to bridge the algebraic gap.

### High: model selection and uncertainty are not campaign-grade

The inferred clock selects “informative” carrier rows using outcome-derived
residual advantage/gap thresholds
(`oph_fpe/bulk/modular_probe.py:1628-1674`) and then treats correlated cap/time/
observable rows as if they supplied an ordinary regression confidence interval
(`1705-1788`). This is not an independent generative split. The consensus paper
requires independent batches before preprocessing and says shared seeds,
trajectories, descendants, or repeated test inspection block a held-out theorem
(`paper/reality_as_consensus_protocol.tex:2650-2668`).

Required correction: freeze source-seed groups before feature construction,
cluster uncertainty at source-seed/trajectory level, and never select held-out
rows by their observed scale contrast.

## What the archived results actually show

Only source seed `20260751` is populated across the relevant artifacts.

| Archive | H3 receipt | H3 held-out NRMSE | S2 NRMSE | strict inferred clock | top-level state scale |
|---|---:|---:|---:|---:|---|
| 4k theory-sync | true | 0.9611948921 | 0.9907311293 | false | `1x` |
| 16k theory-sync | false | 0.9788443477 | 0.9764308984 | false | `1x` |
| 64k vn03 rerun | false | 1.0647415430 | 0.9865927193 | false | `1x` |
| 128k earned diagnostic | true | 0.8642747912 | 0.9997343399 | false | `1x` |

The selected continuous clock-fit subset sometimes has nearest label `2pi` or
`4pi`, but every strict `KMS_GEOMETRIC_CLOCK_FIT_RECEIPT` is false. The
top-level candidate residual chooses `1x` in all four reports. Several archived
observer `3+1D` receipts are nevertheless true because declared replay was
accepted while the finite clock was false. That is wiring evidence, not physical
emergence.

The 4k and 16k theory-sync configs are one comparable family: apart from run
labels and patch count, `repairs_per_cycle` scales proportionally from 1024 to
4096. The 64k vn03 config is not the next rung of that freeze. Relative to 16k,
it changes cycles `160 -> 128`, keeps repairs per cycle at `4096` (changing
repair density), moves readback stop cycle `28 -> 72`, changes observers
`64 -> 1024`, changes H3 candidate budgets, neutral-model budgets, theorem-core
sampling, visualization budgets, worker settings, and the default SM candidate.
It also came from a different simulator commit. The 256k archive lacks the
required frozen config, clock, and geometry-control reports.

Therefore:

- there are only two comparable rungs, not three;
- only one source seed exists;
- E3/E4 are absent;
- H3 outcomes are mixed across the wider incompatible archive;
- the prior “stable failure retires the branch” receipt must be corrected to
  `INCOMPLETE_FAIL_CLOSED`, not treated as evidence of retirement.

Historical successful receipts do not contradict this audit. Older configs
used `transition_response_unitary`, explicitly set
`transition_response_scale = 2*pi`, set `normalization = 2pi`, and selected
`kms_collar_transport_response`; see, for example,
`configs/e1_s3_transition_response_screen_4k.yml:50-74`. The retained June
status records successive changes from scalar response, to object-transition,
to perturb/resettle, followed by denser observer sampling after failures
(`handoffs/.../modular_response_h3_status_20260605.md:38-280`). Those runs were
useful engineering iterations, but they were not a frozen independent-emergence
campaign.

## Required simulator preflight

No important campaign should start unless a machine-readable preflight proves
all of the following:

1. **Frozen family:** multiple predeclared source seeds; exact carrier counts
   `4096`, `16384`, `65536`, and `262144`; one normalized config-family hash;
   no post-16k mutations.
2. **Production carrier federation:** every rung materializes exactly that many
   12-port echosahedral carriers; typed seams/collars and external boundaries
   are source-native; carrier count is not support-mesh cell count or primitive
   observer count; and the separate nested support regulator carries its own
   geometry, level, lineage, and map hashes without driving source seams.
3. **Operational observer:** a nonzero source-derived population passes the
   bounded-interface, rereadable-record, self-prediction, feedback-ablation,
   shuffled-control, and checkpoint-continuation tests; field presence alone is
   insufficient.
4. **Typed record tower:** record identity, physical support, carrier, and
   checkpoint/refinement transport maps are explicit and hash-linked.
5. **Nested geometric/algebraic tower:** geometric parent/child IDs and map
   hashes come from the strict icosahedral scaffold; finite detail algebras,
   faithful states, presentation circuits, noncommutative conditional
   expectations, state compatibility, cap lineage, and transported-state error
   are separately exported. The commutative area receipt cannot substitute for
   this stage.
6. **Prime cap state:** explicit noncommutative `A_geo(C)`, faithful `rho_C`,
   `K_C=-log rho_C`, and a forbidden-field audit excluding record/pointer/repair
   auxiliaries.
7. **BW C1-C8:** simulator-native primitive fields, with the issue-308 verifier
   replayed from saved artifacts.
8. **Independent geometry:** ordered frames, oriented cross-ratios, and `s`
   derived without `pi`, candidate labels, or modular target data.
9. **Frozen intervention:** byte-identical intervention support, side, dose,
   RNG, repairs, and held-out rows for `1`, `pi`, `2*pi`, and `4*pi`.
10. **Equal controls:** finite H3/S2/E3/E4 scores on identical rows and matched
   complexity; no missing-control pass.
11. **Event semantics:** read-after-write records, causal ancestry, population,
   separation, rank-four event coordinates, translations, time orientation,
   and Poincare cocycles; H3 retained as frame fiber.
12. **Independent statistics:** source-seed-grouped train/validation/test split
   and seed-level uncertainty; no carrier selection on held-out outcomes.
13. **Provenance:** code/config/input/output hashes, deterministic replay,
    mutation tests, and explicit fail-closed blockers.

Until this preflight passes, the scientifically correct action is to run unit,
mutation, and bounded smoke tests only. Scaling a semantically invalid producer
would spend compute without testing the paper's claim.

## Honest correction path

The non-cheating path is not to tune constants until `2*pi` wins. It is:

1. retire the old physical meanings while preserving its diagnostics;
2. replace the support-cell-as-carrier production path with an exact-cardinality
   echosahedral federation; keep the nested icosahedral S2 mesh as a separately
   typed support/calibration regulator and prove the realization/refinement
   maps instead of identifying the two objects;
3. implement the finite detail algebras, faithful detail states, presentation
   circuits, and noncommutative expectations needed above the now-available
   commutative geometry scaffold;
4. implement the operational self-reading-observer and stable-record-tower
   receipts instead of inferring them from row fields;
5. implement a true prime cap-algebra/refinement producer on its separate
   branch;
6. implement an independent, target-free geometric frame and cross-ratio clock;
7. freeze one physical response tensor before comparing clock candidates;
8. implement equal-capacity H3/S2/E3/E4 models;
9. implement an event-position/translation/cocycle lane separate from H3 frame
   coordinates;
10. enforce the typed DAG so absent gauge/SM and gravity producers remain false;
11. pass mutation tests that remove or perturb each primitive certificate field;
12. freeze the full campaign before the first exact-4096-carrier source run;
13. stop without retuning after the stipulated exact-16384-carrier failure;
14. retire a branch only after a predeclared, mutually comparable, multi-seed
    stable-failure criterion is actually met.

If this construction still fails, that failure is scientifically meaningful.
If the source state contains no noncentral prime geometric cap algebra or no
independent normalization carrier, the compact paper already allows that
outcome: bare consensus is underdetermined and need not inhabit the BW branch.

## Post-audit architecture ruling: a federation of echosahedrons

The user's clarification and the current microphysics paper fix the concrete
simulator ontology:

```text
abstract theorem interface: finite observer patch net
concrete simulator carrier: a very large federation of echosahedral cells
one cell: finite local algebra + P0..P11 + readout maps + central records
          + mismatch family + local repair instruments + port symmetry
```

This is supported directly by
`screen_microphysics_and_observer_synchronization.tex:1829-1850`.  The global
screen cellulation and the local carrier body are different layers.  Every
cell has its own twelve labeled port coordinates even when only a subset is
routed to neighboring cells at one cutoff.  The simulator must therefore
materialize an `N x 12` local port state, route overlaps through named local
slots, include all twelve coordinates in the patch record, and keep repair
local to the affected carrier/interface.

There are also two distinct twelve-fold A5 objects:

1. the **local carrier fiber** `P12,v` present at every echosahedral patch;
2. the **collective screen-sieve orbit** of twelve unit defects, derived only
   after the physical curvature/settlement/atomic-exposure/position-selector
   premises of the icosahedral screen-sieve theorem.

The first is an implementation choice.  The second is a conditional physical
output.  Equating them without a quotient-visible, overlap-compatible,
refinement-natural A5 intertwiner would insert the desired global symmetry by
hand.  Likewise, a federation of `N` carriers has local rank-twelve fibers
(and roughly a local gauge product before gluing); it does not collapse its
`12N` port coordinates into one global twelve-generator algebra.

The required realization arrow is now explicit in the simulator ladder:

```text
Realize_r : EchosahedralFederation_r -> AbstractPatchNet_r
```

It must preserve accessible algebra, port restrictions, records, accepted
repairs, checkpoints, semantic event history, and the physical quotient.  A
matching class name or common run directory does not prove this arrow.

## Effect of the advanced A5 theorem

The supplied Pro review agrees with the independent `survival-proof/` audit:
the compact-Lie theorem is a substantial exact shortcut at one rung, but it is
a **classifier, not a producer**.

For a compact twelve-dimensional current algebra whose A5 module is
`P12 = 1 + 3 + 3' + 5`, physical inner A5 action forces
`su(3) + su(2) + u(1)`.  Under only a group-level A5 action, compactness leaves
three alternatives; a stable noncentral commutator involving the canonical
five-band removes the abelian and `su(2)^2 + u(1)^6` alternatives.  This
eliminates the old fourteen-parameter bracket ambiguity after the hypotheses
are physically realized.

The simulator-facing missing object is local, for every sufficiently refined
patch:

```text
J_v,r : P12,v -> k_v,r
rank(J_v,r) = 12
image(J_v,r) = k_v,r
```

Here `k_v,r` must come from a closed connected compact **reversible** current
subgroup.  `J` must be A5-equivariant, well conditioned on all four irreducible
bands, reconstruct its bracket from small group commutators, and commute with
overlap transport and refinement.  Dissipative Petz/recovery/settling maps are
not gauge flows; the correct relation is repair/current covariance, not
identity.  The emergence ladder and theorem-application contract now require
these facts before applying the classifier.

This theorem does not produce a global quotient, spin lift, primitive U(1)
period, line/cocharacter lattice, matter spectrum, chirality, Higgs, physical
families, couplings, or QFT.  The imported theorem evidence closes only a
conditional finite implication.  The sole Lean artifact in `survival-proof/`
proves reconstruction no-go statements, not positive A5-to-SM emergence;
positive finite Q0 rows are exact conditional analytic/executable results.

## Additional source-theory blockers found on the current paper snapshot

These are read-only audit findings.  No paper source was edited.

### 1. H3 frame fiber is reused as event position

The compact paper correctly identifies `H3` as the future-unit-timelike frame
space (`compact.tex:5230-5289`) and later explicitly says it is a fiber, not a
spatial slice (`7067-7087`).  In between, it places record positions
`X_i(t)` in `H3`, emits hyperbolic localization balls (`5650-5669`), constructs
event boxes as `time x H3` (`6867-6886`), and uses the same reconstruction as
the three spatial coordinates of an event chart (`6945-6984`).  A timelike
frame/velocity is not an event position.  A separate source-derived event
position chart or an explicit additional identification theorem is required.

This type conflict can directly explain why an H3 event-response fit is
unstable: the evaluator may be asking frame-fiber data to predict a base-space
relation it does not encode.  The repaired simulator therefore keeps H3
strictly frame-valued and gives semantic events a separate lane.

### 2. E1-E3 do not yet prove an open four-manifold chart

The event-manifold proof asserts that every Cauchy filter of certified boxes is
a record germ and that E1 makes the chart image dense in a target open box
(`6985-6991`).  The germ definition additionally requires a compatible family
of actual record tokens (`6879-6886`); it does not show that the completion of
all box filters is populated by such tokens.  Nor does the stated E1 condition
by itself identify an ambient open box in the reconstructed coordinates.
Completeness/surjectivity or an explicit local-degree/invariance-of-domain
receipt is missing from the stated hypotheses.

### 3. A conformal celestial S2 does not force a quadratic Lorentz cone

The causal theorem states that a proper convex cone whose projectivized
boundary is conformally a round `S2` is a Lorentz quadratic cone
(`7022-7029`).  This is false without an embedding-level quadric/projective
condition: many nonquadratic strictly convex projective surfaces are abstract
two-spheres and inherit a conformal structure.  The proof silently strengthens
the premise to “boundary a quadric” (`7050-7058`).

The live #575 contract has the right repair: infer the quadratic cone and its
one-timelike signature on held-out semantic event relations, with a positive
cofinal cone-margin certificate.  The simulator now requires that direct
receipt; a conformal `S2`, H3 fit, or preassigned Lorentz matrix cannot replace
it.

### 4. Population does not imply causal reachability

The converse cone-order step says E1 population supplies a chain of populated
boxes realizing ancestry (`7048-7050`).  Density of events does not imply that
the repair dependency graph contains directed paths between the desired
events.  A reachability/controllability receipt tying derived translations to
committed read-after-write ancestry is separately needed.

### 5. The displayed trace-balanced integration has a type and sign gap

The coefficient theorem first realizes the weak block as a real
`so(3)` cross-product representation.  The trace-balanced theorem then writes
`kappa_W(d_W) - (...) I_2` (`12059-12068`) without displaying the required
`so(3) -> su(2)` spin/Pauli map; as written, a 3-by-3 real skew matrix and a
2-by-2 identity cannot be subtracted.  The exact `survival-proof/` checker
repairs this by an explicit Pauli intertwiner and correctly leaves the binary
spin lift physical receipt open.

There is also a U(1)-orientation convention mismatch between the displayed
cover `(z^2 A, z^-3 B)` (`12069-12077`) and the tensor-action kernel generator
(`12085-12098`) under the stated block hypercharges.  The two abstract
quotients are isomorphic after `z -> z^-1`, but a physical deck subgroup and
matter charges require one frozen orientation convention.

### 6. Transactional confluence needs validation-complete conflicts

Checking each proposed repair against the original snapshot is insufficient
for a protected nonlinear observable.  For example, two disjoint writes
`x:=1` and `y:=1` each preserve `B(x,y)=xy` at `(0,0)`, while their joint
commit changes `B` to one.  Conflict/read sets must include every dependency
of descent, protected boundary data, enablement, checkpoints, sector labels,
and semantic parents; the union payload must be revalidated atomically after
composition.  Otherwise schedule independence can fail and contaminate both
the modular/event and current-algebra branches.

### 7. A normal form does not select a physical state

An idempotent normalizer fixes every probability law supported on its normal
forms.  Repair convergence therefore does not choose the faithful quotient
state needed for modular theory.  The geometry branch needs an intrinsic
state/ensemble producer before a cap modular Hamiltonian exists.  A5 geometry
does not fill this gap.

## Corrected two-branch spine

The strict dependency structure is:

```text
Echosahedral federation
  -> abstract-patch realization
  -> quotient-valid transactional repair
  -> schedule-independent normal form
  -> operational self-reading observer
  -> typed common-domain source tower

common tower + independent state/cap producer
  -> noncommutative prime cap tower
  -> native BW01..BW08 certificate and independent 2pi clock
  -> Lorentz/H3 frame fiber
  -> semantic E1..E4 + held-out quadratic event cone
  -> event metric/connection
  -> stress + entropy + tensor upgrade
  -> conditional gravity

common tower + target-free screen selector + local/global A5 transport
  -> one full reversible P12 current fiber per patch
  -> compact-Lie classifier (exact theorem shortcut)
  -> local su(3)+su(2)+u(1) fiber type
  -> spin/clock/category/deck/line descent
  -> conditional global Z6 form
  -> pole/scalar/family/three-point laws
  -> conditional finite SM Q0
  -> Q1 -> Q2 -> Q3 -> Q4 (all still separate)
```

The geometry/gravity and gauge/SM branches share the repair-normal-form and
common-source trunk; neither is downstream of the other.  In particular, A5
does not prove H3/KMS, and H3 does not prove the Standard Model.

## Integration status

As of this audit snapshot, the working tree contains strict finite scaffolds for
the local 12/30/20 carrier and A5 action, typed federation seams/collars and
observer supports, the separate geodesic support-regulator tower, target-blind
reversible local carrier dynamics, strict common-source replay, the
A5-to-compact-Lie theorem application firewall, frozen clock interventions,
same-row geometry-control gates, and a fail-closed H3/KMS preflight. Focused
tests cover those components.

The production `bw_array.py` dynamics still use support-chart rows as if they
were microscopic carriers. Its source contract now reports that conflation and
keeps carrier realization, quotient invariance, refinement naturality, physical
clock, native BW, semantic event, and physical A5-current receipts false. The
finite detail algebra/state tower, native cap/BW producer, independent event
position producer, and production federation dynamics remain unimplemented.
This document must therefore not be read as a physical emergence result or as
authorization for an important scale campaign.
