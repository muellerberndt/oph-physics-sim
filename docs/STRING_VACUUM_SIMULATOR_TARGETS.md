# OPH string-vacuum simulator specifications and receipt targets

Status: normative target registry and fail-closed verification specification.

The simulator does not derive a string compactification from finite OPH patch
dynamics. It ingests artifacts from string-geometry, vacuum, spectrum,
threshold, low-energy, and verified-numerics producers. A code-owned verifier
replays those artifacts and derives scoped receipts. Producer verdicts and
caller-supplied Booleans are never promotion evidence.

The canonical machine-readable files are:

- `configs/string_vacuum/receipt_targets_v1.json`;
- `schemas/string_vacuum_receipt_targets_v1.schema.json`;
- `configs/string_vacuum/oph_observable_targets_v1.json`;
- `schemas/string_vacuum_observable_targets_v1.schema.json`.

The registry loader rejects duplicate receipt IDs, unknown dependencies,
dependency cycles, scope-inverting dependencies, unknown semantic gates,
unknown catalogue proofs, invalid compatibility aliases, and observable-row
role drift.

## Observable target registry

The frozen comparison values are:

| Row | Comparison value | Role | Rank eligible |
|---|---:|---|---:|
| `oph.alpha2_mz` | `0.03377843630219015` | promoted OPH surface coordinate | yes |
| `oph.alphaY_mz` | `0.010131601067241624` | promoted OPH surface coordinate | yes |
| `oph.v_gev` | `246.76711732749683 GeV` | promoted OPH surface coordinate | yes |
| `comparison.mH_pole_gev` | `125.1995304097179 GeV` | candidate-only comparison | no |
| `comparison.mt_pole_gev` | `172.3523553288312 GeV` | candidate-only comparison | no |

The registry state is `OPEN_INCOMPLETE` and `promotion_allowed=false`.
It lacks a common running scheme, frozen observable-definition artifacts,
acceptance intervals, and a joint covariance or equivalent joint acceptance
rule. The compare-only `alpha3_mz` and `m_Z` inputs are not target rows.

A candidate packet cannot choose its own target. A positive target-binding
receipt requires an allowlisted registry ID, version, and hash; a precommitment
artifact; source freeze before target loading; and a code-owned
source-target-separation verifier.

## Candidate receipt ladder

Structural and algebraic receipts:

1. `STRING_PACKET_CONTRACT_INTEGRITY_RECEIPT`
2. `STRING_ARTIFACT_HASH_RECEIPT`
3. `STRING_OPH_TARGET_REGISTRY_BINDING_RECEIPT`
4. `STRING_EVALUATOR_ENCLOSURE_RECEIPT`
5. `STRING_INTERVAL_CONTRACTION_ALGEBRA_RECEIPT`
6. `STRING_FULL_SYSTEM_CLOSURE_RECEIPT`
7. `STRING_FLAT_DIRECTION_CLASSIFICATION_RECEIPT`
8. `STRING_PHYSICAL_LOCAL_ISOLATION_RECEIPT`

Candidate-physics receipts:

1. `STRING_CRITICAL_EDGE_CFT_RECEIPT`
2. `STRING_FULL_MASSLESS_SECTOR_RECEIPT`
3. `STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT`
4. `STRING_SUPERPOTENTIAL_SAFETY_RECEIPT`
5. `STRING_THRESHOLD_SPECTRUM_RECEIPT`
6. `STRING_COMPLETED_PHYSICAL_SLICE_RECEIPT`
7. `STRING_CANDIDATE_CONSISTENCY_RECEIPT`

Selection receipts:

1. `STRING_MODULI_LOCKING_RECEIPT`
2. `LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT`

The moduli-locking receipt requires all of the following:

- a completed stable physical slice and quotient-descent proof;
- the allowlisted OPH target-registry binding;
- a code-owned proof that residual and Jacobian intervals enclose the physical
  evaluator on the whole box;
- exact interval contraction and existence;
- proof that the selected square equations generate the full zero system;
- complete classification of residual directions, with no visible or
  unclassified flat direction;
- complete target matching and source-target separation.

This is the simulator form of

```text
ker DC(p) intersect ker DF(p) = invisible-orbit tangent at p.
```

The local receipt is false unless `candidate_status=PASS`. A determinant or
producer rank flag alone cannot satisfy it.

## Branch receipt ladder

The branch tier separates coverage from verdict:

1. `STRING_CANDIDATE_REPLAY_RECEIPT`
2. `STRING_BRANCH_GLOBAL_COVERAGE_RECEIPT`
3. `STRING_BRANCH_VERDICT_REPLAY_RECEIPT`
4. `STRING_BRANCH_GLOBAL_UNIQUENESS_RECEIPT`

Coverage proves that quotient charts, boundaries, singular strata, discrete
completions, and noncompact ends are represented. The verdict ledger separately
classifies every covered cell as a certified passing root, a certified failing
region, or unresolved. Any unresolved cell blocks branch selection. A failed
local packet plus a geometric cover cannot exclude an entire branch. Replayed
candidate rows are grouped by their verified branch-definition hash. A
branch-global uniqueness receipt is true only when every row in that branch is
terminal and exactly one physical equivalence class passes.

## Catalogue and unrestricted receipts

The catalogue tier contains:

1. `STRING_CATALOGUE_ENUMERATION_RECEIPT`
2. `STRING_EQUIVALENCE_PARTITION_RECEIPT`
3. `COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT`

Comparative uniqueness requires a semantically verified complete catalogue,
a complete physical-equivalence partition, replayed terminal verdicts for
every row and branch, no unresolved region, and exactly one passing physical
equivalence class.

The unrestricted tier contains:

1. `STRING_UNRESTRICTED_UNIVERSE_COVERAGE_RECEIPT`
2. `OPH_NATIVE_STRING_VACUUM_RECEIPT`

The unrestricted coverage receipt verifies the reduction theorem independently
of comparative uniqueness. The native receipt requires both. Failure of the
unrestricted theorem does not retract a valid catalogue-relative selection.

`OPH_CORE_STRING_SELECTOR_INDEPENDENCE_RECEIPT` is a separate claim-DAG target.
It certifies that string-selector failure has no dependency path into the
recovered OPH core.

## Executable status

The schemas, content-hash replay, subject binding, registry dependency audit,
exact-rational interval contraction, candidate-report replay, and fail-closed
CLI exits are implemented. The string-physics and catalogue-proof semantic
verifier registries are empty, the observable registry is `OPEN_INCOMPLETE`,
and the selector-independence proof is not installed. Consequently, the
executable verifier cannot emit a positive local, catalogue-relative, or
unrestricted string-selection receipt from producer labels. This is an
intentional scientific gate, not a simulator error.

## Compatibility aliases

The verifier emits generic `STRING_*` receipt IDs. The previous BD names remain
aliases for the BD adapter. Alias values must exactly equal their canonical
generic values and cannot define separate promotion paths.

## Evidence bundle requirements

Every candidate bundle includes:

- candidate, branch, equivalence-class, source-freeze, constraint-registry,
  target-registry, and receipt-registry hashes;
- defining geometry, bundle, Wilson-line, hidden-sector, five-brane, flux, and
  safety-symmetry artifacts;
- exact worldsheet, modular, BRST, cohomology, anomaly, Bianchi, and tadpole
  replay inputs;
- complete effective potential, stationarity intervals, physical Hessian or
  BF margin, and correction-control bounds;
- full threshold-relevant spectrum with degeneracies and a certified tail;
- normalized couplings, matching scales, common-scheme thresholds, RG flow,
  and decoupling data;
- frozen physical coordinates, constraints, target rows, augmented row
  bindings, interval box, residual and Jacobian enclosures, preconditioner, and
  full-system closure proof;
- a flat-direction ledger and branch-domain verdict ledger;
- content-addressed regular files, environment and dependency locks, replay
  arguments, and discovery-only random seeds.

Every promoted report binds its Boolean receipts to a subject and scope hash.
Context-free Booleans are diagnostics only.

Each flat-direction row names a nonzero basis vector, physical meaning, and
content-addressed proof artifact. `OPH_INVISIBLE_QUOTIENTED` requires a
group-orbit proof, `STABILIZED` requires a certified mass bound, and a visible
flat family requires an exact curve or constant-rank proof. Visible and
unclassified directions block moduli locking even when their existence is
proved. A well-formed visible-flat receipt gives candidate status `FAIL`; a
well-formed unclassified direction gives `INCONCLUSIVE`. Neither makes the
evidence envelope structurally invalid.

## Commands

Export both canonical target registries:

```bash
python3 -m oph_fpe.string_vacuum describe-targets \
  --out runs/string_vacuum/target_specification.json
```

Verify one candidate:

```bash
python3 -m oph_fpe.string_vacuum verify-candidate \
  runs/string_vacuum/candidate/evidence.json \
  --bundle-root runs/string_vacuum/candidate \
  --out runs/string_vacuum/candidate/verification.json
```

Verify a catalogue:

```bash
python3 -m oph_fpe.string_vacuum verify-catalogue \
  runs/string_vacuum/catalogue/evidence.json \
  --bundle-root runs/string_vacuum/catalogue \
  --out runs/string_vacuum/catalogue/verification.json
```

The verification commands return exit status zero only for a passing candidate
or a catalogue-scoped selection. `FAIL`, `INCONCLUSIVE`, and `INVALID` return a
nonzero status. The target-description command returns zero after both
registries validate.
