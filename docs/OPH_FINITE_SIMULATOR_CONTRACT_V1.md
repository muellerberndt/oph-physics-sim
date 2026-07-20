# OPH finite simulator contract v1

Date: 2026-07-20

Status: normative implementation contract for the simulator redesign. It is
not a paper theorem, a physical-emergence receipt, or evidence that the current
production engine satisfies the contract.

## Purpose

The finite simulator must make one checkable ladder from a federation of local
carriers to observer repair, and only then to the conditional geometry/gravity
and gauge/Standard-Model branches. A suggestive diagnostic, a theorem name, a
known target, or a caller-supplied Boolean cannot fill a missing construction.

The governing rule is:

> A repair is a proof-carrying transaction on quotient-visible overlap data.

The corresponding campaign rule is:

> A physical claim is admitted only from replayed, content-addressed primitive
> artifacts whose upstream capabilities satisfy the typed dependency graph.

## 1. Canonical ontology

These types must remain distinct in code and artifacts.

| Type | Meaning | May contain hidden implementation coordinates? | Physical comparison layer? |
|---|---|---:|---:|
| `PresentationState` | concrete arrays, gauges, caches, schedules, RNG state | yes | no |
| `SemanticCarrierState` | one finite 12-port echosahedral carrier plus typed local readouts | only declared local presentation data | no |
| `QuotientState` | gauge/presentation-invariant state visible to overlap and record contracts | no | yes |
| `NormalFormState` | canonical endpoint only when uniqueness/confluence is proved | no | yes, conditionally |

The serializer for `QuotientState` and every semantic event identifier must
exclude memory addresses, worker IDs, insertion order, thread scheduling,
filesystem paths, and RNG bookkeeping. Mutation or relabeling of those fields
must leave quotient hashes and physical receipts unchanged.

## 2. Federation architecture

The microscopic source is a typed complex/hypergraph of local carriers:

```text
EchosahedralFederation
  carriers: finite SemanticCarrierState objects
  ports: P0 ... P11 on every carrier
  seams: typed port identifications with orientation/gluing maps
  collars: exact dependency neighborhoods used by validation
  higher overlaps: cycles and multi-carrier compatibility cells
  external boundary: explicit, never inferred from missing neighbors
```

Each carrier has the exact local icosahedral incidence `12 vertices / 30 edges /
20 faces` and an orientation-preserving order-60 A5 action. The local carrier
is not a global spherical-support cell and is not a primitive observer. The
support chart is a separately typed supplied/calibration regulator until a
source-derived realization map is proved.

The mandatory realization arrow is

```text
Realize_r : EchosahedralFederation_r -> AbstractPatchNet_r.
```

Its receipt must replay preservation of accessible algebras, port restrictions,
seams, accepted repairs, records, checkpoints, semantic ancestry, physical
quotient, and refinement. A common class name or matching cardinality is not a
realization proof.

Campaign rungs `4096`, `16384`, `65536`, and `262144` denote exact carrier
counts. They must not be silently replaced by the face counts of a geodesic S2
support mesh. The support regulator has its own level/count fields and hashes.

## 3. Transition taxonomy

Every state transition carries exactly one kind:

- `STRICT_REPAIR`: recovery-derived proposal, strict descent of the declared
  touched mismatch ledger, protected quantities unchanged;
- `REVERSIBLE_PROPAGATION`: quotient-compatible reversible carrier evolution;
- `RECORD_COMMIT`: semantic record/checkpoint update after a valid transition;
- `CONTROLLED_EXPLORATION`: predeclared stochastic proposal, never relabeled as
  repair or used to assert descent;
- `GAUGE_STUTTER`: presentation change with identical quotient state;
- `ROLLBACK`: aborted/stale transaction recovery.

Only `STRICT_REPAIR` is required to lower the strict repair mismatch. Forcing
unitary propagation, record formation, and sampling to lower one scalar would
conflate different physical roles.

The reversible carrier operator `U` and dissipative repair operator `R` are
separate. A5 covariance may be tested for both, but `R` is not interpreted as a
gauge flow or modular flow.

## 4. Six-stage strict repair transaction

A valid strict repair receipt is recomputed from these stages.

1. **Snapshot.** Record the complete read set and versions. The read set includes
   proposal dependencies, every affected mismatch term, protected boundary and
   sector fields, enablement predicates, semantic parents, records, and
   checkpoint dependencies.
2. **Recovery-derived proposal.** Emit all admissible quotient proposal classes
   from the declared recovery map. An arbitrary hand-authored payload is not a
   recovery receipt.
3. **Static validation.** Check types, seam maps, local algebra domains,
   conserved/source ledgers, and forbidden target inputs.
4. **Exact acceptance.** Recompute the touched mismatch vector before and after
   the proposal on the saved snapshot. Require the declared strict partial
   order, unchanged protected data, sector admissibility, and no unlisted
   writes.
5. **Atomic conflict-component commit.** Build the conflict graph from complete
   read/write dependency sets. Compose each connected component, then recompute
   the full union acceptance predicate. Primitive moves that pass separately do
   not authorize their union.
6. **Semantic commit.** Emit the quotient transition, causal record, semantic
   parent links, checkpoint, and a proof-carrying receipt binding all inputs and
   recomputed checks.

Stale snapshots, incomplete read sets, incompatible writes, protected-field
changes, target-dependent proposals, ambiguous minima, and failed union
descent abort rather than fall back to a deterministic hash selector.

## 5. Typed mismatch and source ledgers

The repair measure is a named vector, not an untyped scalar:

```text
MismatchVector(
  seam,
  cocycle,
  obstruction,
  record,
  checkpoint,
  sector,
  protected_boundary
)
```

The repair law declares which coordinates strictly decrease, may remain equal,
or are forbidden to change. The transaction artifact includes the dependency
support of every term.

Conserved quantities and allowed sources are likewise explicit. A change in a
would-be conserved charge without a declared source event is a validation
failure, not emergent dynamics.

## 6. Boundary fibers, obstruction, and ambiguity

For a fixed quotient boundary `b`, the exact finite classifier returns one of:

- `UNREALIZABLE`: no consistent interior extension;
- `UNIQUE`: one quotient extension/normal form;
- `AMBIGUOUS`: at least two distinct quotient extensions, with witnesses;
- `UNKNOWN`: bounded search or proof resources did not decide the fiber.

Cycle and higher-overlap holonomy are recorded before attempting repair. A
genuine nonzero obstruction must not be “repaired” by selecting a preferred
representative. `AMBIGUOUS` and `UNKNOWN` are first-class scientific outputs;
neither passes confluence or physical-state selection.

Repair, selection, and sampling are three different operators:

```text
repair:    presentation -> consistent quotient candidate set
selection: candidate set -> chosen physical sector, only with a selector theorem
sampling:  declared law -> ensemble draws, only with provenance and independence
```

An idempotent normalizer does not by itself select a probability law or a
faithful cap state.

## 7. Quotient lumpability and implementation independence

For quotient map `q` and transition kernel `K`, physical promotion requires
that representatives of one quotient class induce the same transition law on
quotient classes. Exact deterministic checks compare target multisets; noisy
kernels compare rational/declared-tolerance probabilities.

If lumpability fails, the engine may retain a presentation-level heuristic
diagnostic, but it cannot emit a physical quotient, observer, modular, geometry,
gravity, or Standard-Model receipt.

Implementation independence is tested by rerunning equivalent presentations
with altered carrier IDs, port labels under the allowed A5 action, insertion
order, worker schedule, storage layout, and checkpoint serialization. The
quotient transition, semantic record DAG, and downstream physical artifacts
must agree after canonicalization.

## 8. Records, observers, and clocks

Four objects must not be collapsed:

- execution logs: debugging history, nonsemantic;
- record algebra: committed observer-readable propositions;
- semantic event DAG: read-after-write ancestry between stable records;
- checkpoints: continuation state sufficient for replay/restore.

An operational observer is a connected subfederation with bounded interfaces,
readback of internal state, persistent records, prediction above shuffled and
ablated controls, feedback that changes later actions, and checkpoint
continuation. A carrier row or record hash alone is not an observer.

The clock registry keeps four independent orders:

- source/refinement clock;
- execution/scheduler clock;
- semantic causal order;
- physical clock candidate.

Only a passed, independently normalized geometric/BW construction may promote
the last. Scheduler steps, repair counts, modular labels, or a configured
`2*pi` are not a physical clock.

## 9. Proof-carrying receipt schema

Every promotable receipt includes at least:

```text
artifact_type, schema_version, claim_tier, capability
producer_id, verifier_id, code_hash, config_family_hash
primitive_input_hashes, upstream_receipt_hashes, provenance_edges
source_type, target_type, construction_map
recomputed_checks, residuals, frozen_thresholds, negative_controls
forbidden_input_audit, ambiguity_or_obstruction_witnesses
passed, blockers, claim_boundary
```

The aggregate verifier ignores caller pass flags and replays primitive files.
Deleting or changing any required primitive, evaluator, seed/config, map, or
provenance edge must demote the capability. Booleans embedded in an unrelated
JSON file are diagnostics only.

Claim tiers form a monotone lattice such as

```text
diagnostic < finite_exact < finite_statistical < conditional_theorem
           < scaling_evidence < physical_promotion
```

No alias or aggregate label may raise the tier above the weakest strict
upstream capability.

## 10. Branch-aware emergence DAG

The shared trunk is:

```text
typed echosahedral federation
  -> concrete/abstract realization
  -> complete proof-carrying repair transaction
  -> obstruction classification + quotient lumpability
  -> schedule-independent quotient normal form
  -> stable records + operational self-reading observers
  -> typed, target-free common source tower
```

The geometry/gravity branch is:

```text
common source + independent state producer
  -> noncommutative prime geometric cap tower
  -> native finite BW01..BW08
  -> independent {1, pi, 2*pi, 4*pi} clock comparison
  -> H3 timelike-frame fiber
  -> semantic event E1..E4 + held-out Lorentz-cone reconstruction
  -> metric/connection -> stress/entropy assembly -> conditional gravity
```

The gauge/Standard-Model branch is:

```text
common source + target-free screen selector + A5 transport
  -> reversible full-rank P12 current fiber
  -> exact compact-Lie classifier shortcut
  -> local su(3)+su(2)+u(1) fiber type
  -> global descent/spin/clock/deck/line data
  -> matter, chirality, Higgs, family and interaction receipts
  -> conditional finite SM rungs
```

The exact A5 classifier may be reused when its source hypotheses are replayed;
Lean or analytic proofs remove the need to numerically rediscover the theorem.
They do not prove that the simulator produced the theorem's hypotheses.

There is no edge from an H3 fit to the Standard Model, from A5 carrier geometry
to a physical gauge current, from consensus to a cap state, or from a frame
fiber directly to a 3+1-dimensional event manifold.

## 11. Physical H3/KMS campaign contract

The campaign remains blocked unless all strict preflight capabilities pass.
Its immutable family contains multiple frozen source seeds and exact carrier
rungs `4096/16384/65536/262144`. `H3`, `S2`, `E3`, and `E4` consume identical
held-out event rows with matched capacity and a common preprocessing budget.

One target-free physical intervention is generated once. Its support, side,
dose, RNG, repair trajectory, and held-out response tensor are byte-identical
for clock candidates `1`, `pi`, `2*pi`, and `4*pi`. Candidate normalization may
enter only the downstream prediction map. The geometric clock coordinate is
derived without candidate labels, `pi`, or modular target data.

No parameter may be retuned after the stipulated 16k failure. A branch is
retired only by the frozen multi-seed stable-failure rule. Invalid or incomplete
instrumentation yields `INCOMPLETE_FAIL_CLOSED`, not physical branch failure.

## 12. Verification strategy

Before any expensive run:

1. exhaust exact small carrier/federation models;
2. run countermodels for stale snapshots, incomplete read supports, nonlinear
   union conflicts, nonzero holonomy, ambiguous fibers, non-lumpable kernels,
   target leakage, ID/schedule dependence, fake records, and fake clocks;
3. run receipt-deletion and artifact-mutation tests;
4. replay the common-source and emergence-ladder verifiers in a clean bundle;
5. run only bounded smoke tests while any strict capability is false.

Source production, blind evaluation, and aggregate verification must be
separate code paths. Content-addressed campaign manifests bind the source
family before the first rung, and a universal provenance DAG labels edges such
as `DERIVES`, `REFINES`, `CALIBRATES`, `COMPARES_TO`, and `VALIDATES`.

## 13. Current implementation boundary

Implemented finite scaffolds include exact local icosahedral/A5 carrier
incidence, typed federation seams and observer supports, a strict geodesic
support-regulator tower, target-blind reversible local A5 dynamics, a replayed
common-source verifier, modular-gearing no-go firewalls, and a strict H3/KMS
campaign preflight. The transaction, ontology, obstruction, and lumpability
modules are being hardened against the rules above.

The existing production `bw_array` source still conflates support-chart cells
with microscopic carriers and therefore fails the carrier realization and
physical source-instrument receipts. The finite detail algebra/state tower,
native cap/BW producer, independent clock producer, semantic event base, and
full physical A5 current producer are also absent. Consequently no important
4k/16k/64k/256k campaign is authorized by this contract, and no emergence
branch may be retired from the archived runs.

