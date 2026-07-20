# Source-to-EFT Contract

Status: `CONDITIONAL_STRICT_1L_POLE_MAP_NOT_OPH_NATIVE_PHYSICAL`

This contract implements the finite, exact part of the bridge

```text
echosahedral quotient source
  -> quotient-visible moments
  -> identifiable EFT candidate
  -> field-censused RG/matching tower
```

It does not implement or claim the producer chain from an OPH source to a
physical FJ, covariance, BRST, or pole packet.  A mathematically valid
candidate is still an untrusted candidate until an independent source
producer and replay verifier bind it to the quotient artifacts.

The implementation is in
`oph_fpe/gauge/source_to_eft.py`.  It intentionally has no parameter for
caller-authored receipt booleans.

## Exact checks implemented

- `QuotientMomentPacket` contains quotient-visible rational moments and has no
  EFT-parameter field.  Its parser rejects additional fields such as `theta`.
- `verify_exact_affine_identifiability` checks an exact rational left inverse
  `B A = I`.  In the infinity norm it derives
  `sigma = 1 / ||B||_inf`, checks the candidate residual exactly, and reports
  the two-candidate bound `2 epsilon / sigma`.
- `FiniteSourceLawPacket` distinguishes a uniquely selected deterministic
  point from a source-weighted stochastic ensemble.  A deterministic
  enclosure is not accepted as covariance, and a branch-ambiguous
  deterministic packet has no covariance.
- `CoherentSourceCommitments` binds the source root, branch, full field census,
  schemes, thresholds, FJ convention, perturbative monomial mask, analytic
  sheet, and units/clock.  Every required source component must carry the same
  tuple.  Equality is structural, not proof that the hashes were physically
  produced.
- `PerturbativeMask` verifies a finite downward-closed set of exact
  multiindices.  Direct and converted calculations require exact mask
  equality, not a shared numeric order label.
- `ExactMatchingTower` requires a complete field census on each EFT interval,
  an explicit RG flow and threshold matching map, one common mask, and
  commitments to the computed census/scheme/threshold manifests.  It composes
  affine Jacobians and the deterministic recurrence
  `e_(k+1) <= L_M,k L_Phi,k e_k + epsilon_k` with rational arithmetic.

## Provenance lanes

`IMPORTED_SM_STRICT_1L_VALIDATION` is reserved for convention-labelled inputs
used to validate a separate field-theory calculator.  It is never an OPH
prediction.

`OPH_NATIVE_STRICT_1L` may consume only a coherent packet produced from OPH
primitive artifacts and independently replayed.  This producer/verifier chain
does not yet exist here.

Consequently, even when every finite algebraic check passes, these receipts
remain false:

```text
OPH_SOURCE_PRIMITIVE_PRODUCER_RECEIPT
OPH_QUOTIENT_MOMENT_PHYSICAL_RECEIPT
OPH_NATIVE_EFT_REALIZATION_RECEIPT
MATCH1_RECEIPT
FJ1_RECEIPT
COV1_RECEIPT
BRST1_RECEIPT
PHYSICAL_POLE_RECEIPT
PHYSICAL_COHOMOLOGY_POLE_RECEIPT
OPH_NATIVE_STRICT_1L_PHYSICAL_RECEIPT
```

No pole engine or Standard Model target values are embedded in this contract.
The focused mutation tests are in `tests/test_source_to_eft_contract.py`.
