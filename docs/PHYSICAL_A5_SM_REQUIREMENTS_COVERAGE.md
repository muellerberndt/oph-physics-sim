# Physical A5-to-SM simulator requirements coverage

Date: 2026-07-20

Implementation: `oph_fpe/gauge/physical_a5_sm_requirements.py`

Normative input (read-only):
`../../survival-proof/SIMULATOR_RECEIPT_REQUIREMENTS.md`

## Verdict

The requirements are realistic as a staged research program, but not as one
ordinary simulation feature.  Their manifest, provenance, exact finite
algebra, negative-control, and dependency-gating parts are implementable now.
Several later lanes require new physical source producers, exact-small
exhaustors, a finite chiral gauge construction, or independently acquired
apparatus data.  Those are research outputs, not fields that can safely be
filled in by the current simulator.

The simulator now has a fail-closed contract skeleton.  It can positively
replay only the hash/role **inventory** inside a candidate `ROOT` bundle.  That
inventory remains `OPEN`: it does not discharge typed per-role semantics, an
actual executable/build binding, or a pre-outcome commitment.  There is no
registered physical `ROOT` producer and no registered downstream physical
producer.  This prevents a checklist, theorem result, structural certificate,
synthetic fixture, or merely hash-consistent placeholder bundle from being
mistaken for empirical closure.

Every physical stage and route uses exactly `PASS`, `OPEN`, `UNRESOLVED`,
`FAIL`, or `NOT_APPLICABLE`; `passed` is true if and only if status is `PASS`.
Claim scope is explicit (`structural`, `full_interacting`, or `continuum`), so
optional downstream stages are `NOT_APPLICABLE`, never silently green.

## Claim tiers

The verifier recomputes three distinct claims.

The structural physical pass is exactly

```text
ROOT
and GEOMETRY_565
and CURRENT_566
and GLOBAL_FORM_567
and SPIN_EXCHANGE_314
and SOURCE_REGISTRY_AND_PRIMITIVE_RESIDUES
and SCALAR_CHANNEL
and FAMILY_ATTACHMENT_569
and Q1_LOCAL_ACTION
and (Q2_H or (Q2_E and POSITIVITY_OR_POSITIVE_TRANSFER))
and REFINEMENT_COMPLETENESS
and PHYSICAL_IDENTIFICATION.
```

The full interacting pass additionally requires

```text
COMPLETE_COUPLED_DYNAMICS
and FAMILY_BREAKING_OR_DESCENT
and VERTEX_1PI.
```

The nonperturbative continuum Wightman pass additionally requires `Q4_OS`.
A bare `Q2_E` result emits at most `FINITE_EUCLIDEAN_STRUCTURAL_PASS`; without
positivity it cannot satisfy the physical Q2 disjunction.  A structural/Q2
pass cannot promote either the full interacting or continuum tier.

## Implemented stage DAG

The order below is the stable serialized contract.  An `A or B` dependency is
represented as explicit alternative dependency groups, not a caller boolean.

| Stage | Hard dependencies | Receipt routes | Current physical producer |
|---|---|---|---|
| `ROOT` | none | typed, build-bound, pre-outcome immutable root packet | absent; inventory replay only |
| `GEOMETRY_565` | `ROOT` | source-derived geometry or operational hardware geometry | absent |
| `CURRENT_566` | `ROOT`, geometry | two-sided current tomography | absent |
| `GLOBAL_FORM_567` | `ROOT`, current | public carrier, volume clock, category, deck, UV selector | absent |
| `SPIN_EXCHANGE_314` | `ROOT`, geometry, global form | Spin lift, CAR, exchange, rank-15 projector | absent |
| `SOURCE_REGISTRY_AND_PRIMITIVE_RESIDUES` | `ROOT`, global form, Spin | complete registry, vacuum, quotient/filtration, residue exhaustion | absent |
| `SCALAR_CHANNEL` | `ROOT`, source registry | complete primitive scalar search | absent |
| `FAMILY_ATTACHMENT_569` | `ROOT`, current, global form, Spin, source registry | full-screen complex susceptibility | absent |
| `Q1_LOCAL_ACTION` | all finite-core producers | complete local coupled action, bound identically into Q2 | absent |
| `Q2_H` | finite core and Q1 | nontrivial finite chiral Hamiltonian | absent |
| `Q2_E` | finite core and Q1 | complete finite Euclidean coupled measure | absent |
| `POSITIVITY_OR_POSITIVE_TRANSFER` | `Q2_E` | reflection positivity or equivalent positive transfer Hamiltonian | absent |
| `REFINEMENT_COMPLETENESS` | finite core, Q1, one physical Q2 branch | complement-complete cofinal refinement | absent |
| `PHYSICAL_IDENTIFICATION` | refinement and one physical Q2 branch | frozen finite alternative class, or checked reduction theorem | absent; cannot be simulator-only |
| `COMPLETE_COUPLED_DYNAMICS` | Q1, physical Q2, refinement, physical identification | identical interacting action/measure/evolution binding | absent |
| `FAMILY_BREAKING_OR_DESCENT` | family, Q1, refinement, physical identification | exact compatibility or target-blind breaking/descent | absent |
| `VERTEX_1PI` | coupled dynamics and family compatibility/descent | amputated Grassmann 1PI vertices | absent |
| `Q4_OS` | complete positive interacting Euclidean branch | OS-compatible continuum family | absent |

Every route has a fixed list of fine-grained receipt keys in `STAGE_SPECS`.
A stage passes only when one complete route is admitted and all dependency
gates pass.  Missing routes report `no_registered_physical_producer:<receipt>`.
The admission skeleton consumes the shared generic production envelope and
requires its recomputed report marker, inventory replay, profile, root,
stage/lane, claim scope, and optional branch identity to match.  The generic
envelope also binds producer/build, subject, output, freeze, ancestry, and
firewall bytes.  That is only P0 inventory: even a matching envelope leaves
the stage `OPEN` because the current scientific producer registry is empty.
A supplied mixed-root or otherwise inconsistent envelope is `FAIL`.  Future
registered evidence still cannot bypass this boundary: stage `PASS` requires
literal-true same-root P0 identity, and a missing or demo identity is never a
passing substitute.

`FAMILY_BREAKING_OR_DESCENT` has two distinct routes.  An exact surviving
family symmetry may pass if its commutant is compatible with the emitted
nondegenerate matrices; that route does not require an artificial breaking
witness or any `FAMILY_BREAKING_*` receipt.  It instead has explicitly named
exact coupling-compatibility and refinement-stability receipts.  If exact
symmetry is incompatible, a separate target-blind
breaking/descent route requires a pre-outcome candidate freeze, raw histories,
orbit/stabilizer evidence, and refinement stability.

Both Q2 routes include a mandatory receipt binding the Q2 object to the exact
upstream Q1 action, gauge/scalar/fermion objects, source registry, and root.

## ROOT inventory replay implemented now

The exact schema is `oph.physical-a5-sm.root-manifest/1.0.0`.  It requires:

- one uniquely typed, hash-pinned artifact for code, dirty-tree state, complete
  configuration, regulator, state/quotient space, dynamics, source registry,
  boundary/sector, gauge, Spin, A5 and refinement representations,
  reproducibility, numerical policy, candidate domain, and selection law;
- a 40- or 64-hex code commit and a dirty-tree digest bound to its artifact;
- explicit unique seeds, deterministic reduction settings, precision, interval
  method, and rank-certification method;
- relative regular artifact files contained below the manifest directory, with
  no symlink component, parent traversal, duplicate path, or hash drift;
- a hash-pinned acyclic ancestry DAG in which each derived array has at least
  one parent and is reachable from source primitives;
- a distinct `source_operation` provenance kind for the dynamics generator;
  a candidate-domain or other source primitive cannot be substituted as an
  ancestry operation;
- distinct source-primitive candidate-domain and selection-law artifacts whose
  declared hashes exactly match replayed bytes; and
- the five forbidden target-selection dependencies present with the exact
  Boolean value `false`.

JSON is parsed with duplicate-key and nonfinite-number rejection, including
overflow forms such as `1e999`; normalized artifact paths reject backslashes
as well as traversal.  Stored ROOT
reports can also be replayed and must equal the recomputed report exactly.
Reports must be written beside the manifest, cannot follow an output symlink,
cannot overwrite the manifest, and cannot overwrite an existing report.

The ROOT target fields are only declared target-blindness.  They do not prove
semantic noninterference, real code/build identity, or that the packet existed
before outcomes were observed.  Consequently
`ROOT_INVENTORY_REPLAY_RECEIPT` may be true while
`ROOT_TYPED_ROLE_SEMANTICS_RECEIPT`, `ROOT_CODE_BUILD_BINDING_RECEIPT`,
`ROOT_PREOUTCOME_COMMITMENT_RECEIPT`, and the normative
`ROOT_IMMUTABLE_PACKET_REPLAY_RECEIPT` remain false.

## Evidence admission boundary

The physical producer registry is empty.  The separate inventory-verifier
registry contains `physical_a5_sm_root_inventory_replayer_v1`.  Candidate JSON
files inside an inventory-valid ROOT bundle are hash-checked and inventoried,
but their booleans are ignored.

The following are always nonpromoting in this contract:

- `PHYSICAL-FAMILY-POLE-RECEIPT-v1`, including the survival-proof semantic
  verifier's output;
- every `OPEN` receipt;
- synthetic or fixture packets;
- the conditional theorem DAG and A5 structural certificates;
- source-to-EFT conditional contracts; and
- arbitrary documents containing all-true `*_RECEIPT`, `*_PASS`, or `passed`
  fields.

These objects remain useful as theorem dependencies, diagnostics, schemas, or
test oracles.  They are not physical source producers.

## Feasibility by implementation class

| Class | Examples | Assessment |
|---|---|---|
| Contract and replay engineering | ROOT, hashes, ancestry, exact schemas, stage DAG | implemented |
| Exact-small finite mathematics | A5 actions, ranks, kernels, source closure, negative controls | realistic; build before scale-up |
| Measured simulator producers | two-sided responses, complete settlement, `J_all`, residues | realistic only after the actual local dynamics and source registry are fixed |
| Finite QFT construction | Q1, nontrivial Q2-H or complete Q2-E, positivity | substantial independent research implementation |
| Cross-regulator proofs | complement-complete refinement, all branches/diamonds | substantial; cannot be inferred from old-sector residuals |
| Physical substrate identification | apparatus histories and informational completeness | requires independent acquisition outside simulator self-inspection |
| Continuum reconstruction | complete OS Schwinger family | long-horizon optional research lane |

The recommended next positive milestone is a bounded exact-small oracle:
enumerate the finite source algebra, Hamiltonian/Gauss kernel, Riesz windows,
primitive residue images, scalar competitors, full-screen family response, and
refined complement.  Only after every required injected control fails for the
intended reason should the same producer be scaled to campaign rungs.

These are explicit nonphysical campaign gates:

- `EXACT_SMALL_ORACLE` is `OPEN` until one verified envelope covers full state
  enumeration, the complete source registry, exact Hamiltonian/Gauss kernels,
  Riesz projectors, primitive residues, scalar competitors, full-screen
  `J_all`, the refined complement, and injected negative controls;
- `SCALE_CAMPAIGN_ALLOWED` is `OPEN` and false until exact-small is `PASS` and
  the same verifier/root identity, thresholds, controls, seed schedule, and
  scale grid are frozen.

Neither gate is part of the scientific A5-to-SM conjunction.  Scale code must
check the second gate before launching important campaigns.

Forced end-to-end displays are an orthogonal UI surface.  Any JSON marked
`DEMO_ASSUMPTION` or `visualization_only` is classified diagnostic-only,
cannot satisfy a stage, and cannot change either campaign gate.  It must remain
watermarked by the visualization layer.

Issue #569 closure includes `PHYSICAL_IDENTIFICATION` at the issue aggregate,
not as an upstream dependency of family attachment; this avoids a dependency
cycle.  Issue #590 is represented as a delimitation closure with no physical
`passed` Boolean.

## Regression coverage

`tests/test_physical_a5_sm_requirements.py` covers:

- a valid fake/hash-consistent ROOT inventory remaining `OPEN`, with every
  physical stage false;
- off-disk caller mappings;
- parent-path escape;
- artifact hash drift;
- a flipped target selector;
- ancestry cycles, missing derived-array ancestry, and rejection of a
  candidate-domain artifact as a source operation;
- forged all-true lane JSON;
- an OPEN physical-family/pole fixture;
- exact stored-report replay and output-path/symlink confinement;
- duplicate JSON-key rejection;
- real on-disk generic P0 replay, same-root identity, and mixed-root rejection;
- `DEMO_ASSUMPTION`/visualization-only nonpromotion;
- the two noninterchangeable family routes and their statuses;
- claim-scope `NOT_APPLICABLE` behavior;
- full-interacting and continuum report statuses requiring their actual
  scope-specific stages rather than only the structural aggregate;
- same-root P0 identity as a mandatory condition for any future stage pass;
- overflow-number and backslash-path rejection;
- mandatory Q2 upstream-binding keys; and
- the structural/interacting/continuum tier truth table, including bare-Q2-E
  rejection without positivity and Q2-E-plus-positivity branch admission.
