# String-vacuum selection receipt contract

Status: normative interface and fail-closed verifier contract.

The simulator has no string compactification solver. It cannot certify a BD
vacuum from its finite patch dynamics. Its role in this lane is to ingest
candidate-specific outputs from geometry, effective-action, spectral,
renormalization-group, and verified-numerics producers, then recompute a public
receipt bundle without trusting producer verdicts.

The Bouchard--Donagi one-Higgs model is a structural candidate. The available
sources do not supply its stabilized vacuum, full massive spectrum, physical
threshold map, nonsupersymmetric low-energy continuation, or rank/isolation
certificate. This is an unsupported selection result, not a proof that the
compactification is physically impossible.

## 1. Claim tiers

The verifier keeps four claims separate.

| Claim | Required result |
|---|---|
| Candidate consistency | One presentation satisfies the exact string, compactification, safety, spectrum, and low-energy gates. |
| Local vacuum witness | A stable physical solution exists and is isolated in a certified quotient-chart box. |
| Branch uniqueness | Every other chart box, boundary, singular stratum, discrete completion, and noncompact end on the branch is excluded. |
| Catalogue selection | Every class in a declared complete catalogue has a replayed branch verdict, exactly one class passes, and none is inconclusive. |

The final label is `selected within catalogue C`. A claim about a wider
candidate universe requires a reduction theorem proving that every eligible
presentation is represented in `C` up to the declared physical equivalence.
Known string dualities mean that the selected object is a physical equivalence
class, not a unique perturbative presentation.

## 2. Mathematical certificate

Let `X` be an ambient physical quotient chart with real dimension `n`. Exact
gauge, diffeomorphism, bundle-isomorphism, basis, scheme, and proved-duality
directions are removed before `X` is declared. Let

```text
C : X -> R^s
```

contain the completed anomaly, vacuum, stabilization, safety-realization,
spectrum, matching, and decoupling equations. Let

```text
F : X -> R^k
```

be the observer-visible forward map, with target registry `O_OPH`. At a
solution `x*`, the transverse locking condition is

```text
rank DC(x*) = s
rank [DC(x*); DF(x*)] = n
```

or, equivalently,

```text
ker DC(x*) intersect ker DF(x*) = {0}
```

on the quotient chart. Before quotienting, the right side is exactly the
tangent to the independently proved invisible orbit.

This condition is equivalent to injectivity of `DF` on the completed tangent
`ker DC`. It prevents a short target vector from being compared directly with
an unstabilized ambient family. For the documented 142-real-dimensional BD
one-Higgs slice, even five independently promoted target rows would require at
least 137 net real cuts from completion constraints and additional invisible
quotient directions. Only three rows in the frozen comparison packet have
promoted OPH status; the other two are candidate-only and rank-forbidden. The
three admissible rows require at least 139 net cuts. These counts are necessary
conditions.

### No self-targeting

Every target row is registered before candidate evaluation with its observable
definition, units, scheme, scale, acceptance region, covariance, and status.
The source model is frozen and hashed before the evaluator loads target data.
A producer may not append local candidate coordinates and use their values at
the candidate point as targets. Such padding manufactures an identity Jacobian
for any point. Candidate-only and diagnostic rows do not contribute to the
rank certificate.

### Dynamical stability is separate

Observable rank proves parameter identifiability. It does not give a scalar a
mass. A Minkowski or de Sitter receipt requires

```text
grad V_eff(x*) = 0
Hess_phys V_eff(x*) >= mu^2 I,  mu^2 > 0
```

in canonically normalized physical coordinates after gauge and Goldstone
directions are removed. An anti-de Sitter route records and verifies the
appropriate Breitenlohner--Freedman bound. The receipt also records the domain
of validity and bounds on omitted alpha-prime, loop, instanton, and truncation
terms.

## 3. Interval existence and isolation

A pointwise floating-point rank or nonzero determinant is diagnostic. The
proof packet supplies a rational box `B`, center `x0`, invertible rational
preconditioner `A`, outward-rounded residual enclosure `[H(x0)]`, and
outward-rounded Jacobian enclosure `[J(B)]` for a selected square subsystem.
The independent verifier recomputes

```text
K(B) = x0 - A[H(x0)] + (I - A[J(B)])(B - x0)
q    = sup(J in [J(B)]) ||I - AJ||_infinity
```

and requires

```text
K(B) is strictly inside B
q < 1
```

The contraction theorem then proves existence and uniqueness of the selected
root in `B`. The selected equations must generate the full physical zero
system on `B`, or every omitted equation needs an independent exact identity
or full-solution receipt. Experimental outputs not used in the square system
need certified containment in their predeclared joint acceptance region.

`oph_fpe/string_vacuum/verified_numerics.py` recomputes this algebra with exact
rational arithmetic. It does not prove that the producer's residual and
Jacobian intervals enclose the physical evaluator. That obligation belongs to
an allowlisted primitive enclosure verifier bound to the evaluator source,
input box, arithmetic backend, and code hash.

## 4. Candidate evidence packet

The strict schema is
`schemas/string_vacuum_candidate_evidence_v1.schema.json`. The packet contains
the following sections.

### Candidate identity

- theory family, presentation, discrete branch, and OPH-equivalence-class ID;
- geometry, bundle, equivariant structure, Wilson line, hidden bundle,
  five-brane class, fluxes, and safety symmetry;
- hashes for defining polynomials, transition maps, monads or extensions,
  source datasets, and branch conventions.

### Presentation and quotient

- complete real coordinate registry, complex pairings, units, domains, and
  chart transitions;
- every redundancy generator, its action, and its classification;
- gauge fixing or an intrinsic quotient chart;
- proof that `C`, `F`, spectra, thresholds, and records descend to the quotient;
- duality canonicalization and fixed-orbit-type strata;
- separate failure for an unclassified stabilizer, singular quotient, Gribov
  ambiguity, or unproved scheme equivalence.

### Exact string consistency

- critical-edge CFT, central charges, Virasoro and Sugawara exhaustion,
  supercurrent, GSO and spin structures, modular invariance, level matching,
  BRST and no-ghost checks;
- flux and charge quantization;
- local and integrated Bianchi, anomaly, and tadpole equations;
- explicit perturbative order, control parameter, and remainder status when an
  exact worldsheet construction is absent.

A supergravity solution is not an exact worldsheet receipt.

### Compactification and massless sector

- exact Calabi--Yau, vector-bundle, Wilson-line, hidden-sector, and five-brane
  data;
- stability chamber and the hypotheses used for Hermitian Yang--Mills
  existence;
- reproducible cohomology complexes, bases, maps, ranks, equivariant
  projections, indices, representations, and multiplicities;
- direct realization of the safety symmetry on the compactification;
- superpotential and operator calculation, including nonperturbative sectors
  when they affect the gate.

### Completed vacuum

- complete effective `K`, `W`, gauge kinetic functions, D-terms,
  nonperturbative contributions, scalar metric, and provenance;
- all stationarity residuals;
- physical Hessian or mass enclosure and stability lower bound;
- stabilized dilaton, Kahler, complex-structure, bundle, hidden, and five-brane
  coordinates;
- weak-coupling, large-volume, KK and string-scale separation, and correction
  bounds;
- every remaining modulus or axion, with physical meaning and classification.

### Spectrum and couplings

- complete visible, hidden, five-brane, vectorlike, KK, winding, oscillator,
  and other threshold-relevant sectors;
- spectral enclosures, degeneracies, representations, and a certified tail
  bound, or an equivalent regulated determinant or worldsheet partition
  calculation with an error bound;
- normalized kinetic terms and physical Yukawa integrals;
- no-light-exotics and no-extra-visible-gauge-factor checks;
- conventional supersymmetry breaking, mediation, and soft boundary data, or a
  separately consistent nonsupersymmetric continuation;
- proof that the continuation preserves the cited BD visible and safety data.

Massless cohomology does not stand in for the nonzero spectrum.

### Threshold and low-energy map

- one regularization and renormalization scheme;
- an ordered scale registry for string, compactification, KK, unification,
  supersymmetry breaking, electroweak, and decoupling thresholds;
- RG equations, loop orders, matching maps, field census, and truncation
  errors;
- source-derived output intervals for the complete target registry;
- pole versus running conventions, units, covariance, and error propagation;
- a dependency DAG proving that benchmark or measured targets do not enter the
  source calculation as adjustable inputs.

### Rank, isolation, and flat directions

- a complete constraint registry, frozen before solving, with a definition
  artifact for every physical equation;
- coordinate and typed augmented-row ordering hashes;
- every augmented row bound either to that constraint registry or to a
  precommitted target-registry row;
- interval enclosures of the constraint and target Jacobians;
- the named square augmented row set and rational preconditioner;
- strict interval contraction evidence;
- full-system closure identities;
- a content-addressed proof artifact and nonzero physical basis vector for
  every residual direction;
- every kernel or residual flat direction classified as one of:
  `OPH_INVISIBLE_QUOTIENTED`, `STABILIZED`, `VISIBLE_FLAT`, or `UNCLASSIFIED`.

An infinitesimal kernel vector becomes a proved flat family only with constant
rank on a neighborhood or an explicit target-preserving curve. Any
`VISIBLE_FLAT` or `UNCLASSIFIED` direction blocks moduli locking.

## 5. Catalogue evidence packet

The strict schema is
`schemas/string_vacuum_catalogue_evidence_v1.schema.json`. It requires:

- a precise scope statement and enumeration proof;
- a physical equivalence relation and partition proof;
- one independently regenerated candidate verification report per row;
- full branch-domain coverage per row;
- explicit treatment of disconnected roots, chart boundaries, singular
  strata, discrete topology, bundle, flux, hidden, five-brane, and safety
  choices;
- no unresolved region;
- exactly one passing equivalence class for a positive comparative receipt.

A producer label does not establish enumeration, equivalence, branch-domain
coverage, or exclusion. Each proof is replayed by a code-owned semantic
verifier. Branch-domain coverage and the branch verdict ledger are separate
proofs. A coverage `FAIL` or `INCONCLUSIVE` blocks selection and cannot suppress
an otherwise passing candidate. Within a covered domain, every cell needs a
replayed terminal verdict. An unresolved verdict remains in the possible
survivor set and is not treated as an excluded box. One generic `Spin(32)/Z2`
row cannot exclude all compactifications on that heterotic branch. The
rank-sixteen modular-lattice
theorem leaves both `E8 x E8` and `Spin(32)/Z2` presentations available until
an observer-visible discriminator, equivalence theorem, or exhaustive branch
audit is supplied.

## 6. Provenance and replay

Every packet records:

- producer and verifier commits, dirty-tree state, dependency lock, container
  or environment hash, arithmetic backend, precision, and outward rounding;
- source-freeze, branch-definition, coordinate-registry, dependency-lock, and
  environment artifacts bound to their declared hashes;
- a target-registry hash recomputed from the inline registry;
- content-addressed regular files beneath one bundle root;
- a typed provenance DAG and semantic role for every artifact;
- seeds and stochastic search settings as discovery provenance only;
- a fixed replay argument vector.

The structural `contract_integrity_receipt` covers these bindings and packet
invariants. Commit labels remain provenance until a build-attestation verifier
is installed; structural integrity is not a scientific gate. The verifier
rejects path escapes, symlinks, oversized files, hash mismatches, non-string
proof-critical numbers, malformed dimensions, unregistered augmented rows,
candidate-only rank rows, missing closure proofs, visible flat directions, and
producer pass hints without code-owned semantic verifiers. It does not execute
commands supplied by a manifest.

## 7. Derived receipts

The independent report derives these literal gates:

```text
STRING_CRITICAL_EDGE_CFT_RECEIPT
STRING_FULL_MASSLESS_SECTOR_RECEIPT
STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT
STRING_SUPERPOTENTIAL_SAFETY_RECEIPT
STRING_THRESHOLD_SPECTRUM_RECEIPT
STRING_COMPLETED_PHYSICAL_SLICE_RECEIPT
STRING_CANDIDATE_CONSISTENCY_RECEIPT
STRING_MODULI_LOCKING_RECEIPT
LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT
COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT
STRING_UNRESTRICTED_UNIVERSE_COVERAGE_RECEIPT
OPH_NATIVE_STRING_VACUUM_RECEIPT
```

The generic IDs are canonical. The previous BD IDs are exact compatibility
aliases for the BD adapter. The catalogue verifier reloads each candidate evidence
packet, reruns the candidate verifier, and rejects a candidate report that does
not match the recomputed bytes.

`COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT` means selection only within the
declared, semantically verified catalogue scope. `OPH_NATIVE_STRING_VACUUM_RECEIPT`
also requires the independently verified unrestricted-coverage receipt. A
failed unrestricted theorem does not retract a valid catalogue-relative
selection.

The complete machine-readable ladder, observable target values, branch verdict
rules, dependency graph, and failure scopes are specified in
`docs/STRING_VACUUM_SIMULATOR_TARGETS.md` and
`configs/string_vacuum/receipt_targets_v1.json`.

The candidate and catalogue scientific-verifier registries in
`oph_fpe/string_vacuum/contract.py` are empty. The simulator therefore cannot
emit a positive native string-vacuum receipt from JSON labels and hashes. Each
gate needs an allowlisted semantic verifier over primitive artifacts. The exact
interval contraction audit is implemented, but it stays below physical
promotion until its evaluator-enclosure proof has such a verifier.

## 8. Replay commands

Export the validated receipt and observable targets:

```bash
python3 -m oph_fpe.string_vacuum describe-targets \
  --out runs/string_vacuum/target_specification.json
```

Candidate verification:

```bash
python3 -m oph_fpe.string_vacuum verify-candidate \
  runs/string_vacuum/candidate/evidence.json \
  --bundle-root runs/string_vacuum/candidate \
  --out runs/string_vacuum/candidate/verification.json
```

Catalogue verification:

```bash
python3 -m oph_fpe.string_vacuum verify-catalogue \
  runs/string_vacuum/catalogue/evidence.json \
  --bundle-root runs/string_vacuum/catalogue \
  --out runs/string_vacuum/catalogue/verification.json
```

Both commands write their report for audit, but return shell exit status zero
only for a passing candidate or a catalogue-scoped selection. `FAIL`,
`INCONCLUSIVE`, and `INVALID` reports return nonzero.

Focused tests:

```bash
python3 -m pytest -q tests/test_string_vacuum_selection_contract.py
```

## 9. Mapping to the BD source packet

The reverse-engineering-reality BD packet has separate slots for moduli points
and mass matrices, hidden and five-brane completion, the nonzero spectrum,
scales, normalized Yukawas, low-energy continuation, conventional breaking or
a BD-equivalent nonsupersymmetric route, spectrum generation, thresholds, and
the physical Jacobian. Future adapters bind verified artifacts from this
contract into those slots.

The `physical_moduli_jacobian` slot alone is insufficient. The adapter also
needs the physical quotient, completed constraints, stability receipt,
precommitted target registry, augmented interval-isolation report, and branch
coverage ledger. The existing BD receipt builder verifies path, hash, branch,
issue, and slot envelopes only. Its promotion flags remain false until a
scientific evaluator consumes and verifies the complete packet.

## 10. Technical references and tool boundaries

- [Bouchard--Donagi, 2005](https://arxiv.org/abs/hep-th/0512149) supplies the
  visible heterotic Standard Model construction and Higgs strata. It does not
  supply the completed spectrum or vacuum certificate.
- [Bouchard--Cvetic--Donagi, 2006](https://arxiv.org/abs/hep-th/0602096)
  computes classical trilinear couplings. It does not supply normalized
  low-energy thresholds or pole observables.
- [Bouchard--Donagi, 2008](https://arxiv.org/abs/0804.2096) treats hidden-sector
  completions as additional branch data.
- [Kaplunovsky, 1992](https://arxiv.org/abs/hep-th/9205070) and
  [Kaplunovsky--Louis, 1995](https://arxiv.org/abs/hep-th/9502077) show the
  vacuum and moduli dependence of heterotic threshold matching.
- [Krawczyk--Neumaier, 1986](https://doi.org/10.1016/0022-247X(86)90303-3)
  gives the validated interval root framework used by the isolation contract.
- [alphaCertified](https://arxiv.org/abs/1011.1091) and
  [NumericalCertification](https://arxiv.org/abs/2208.01784) provide
  complementary certification for polynomial roots.
- [CYTools](https://arxiv.org/abs/2211.03823),
  [cohomCalg](https://arxiv.org/abs/1003.5217),
  [STRINGVACUA](https://arxiv.org/abs/0801.1508), and
  [cymetric](https://arxiv.org/abs/2205.13408) cover pieces of topology,
  massless cohomology, effective polynomial vacuum equations, and approximate
  metrics. None produces the complete BD receipt by itself.
- [SOFTSUSY](https://arxiv.org/abs/hep-ph/0104145) and
  [FlexibleSUSY](https://arxiv.org/abs/1406.2319) evolve supplied low-energy
  models and boundary conditions. They do not derive BD-specific breaking or
  string thresholds.
