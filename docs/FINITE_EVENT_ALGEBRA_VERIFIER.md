# Exact finite diagonal event-algebra verifier

`oph_fpe.algebra.finite_event_algebra` is a narrow executable client of the
machine-checked projection-event, Lueders-update, and partition-pinching
results. It matches the object the current simulator can honestly provide: a
finite central-record basis partition with exact diagonal zero/one event
projectors and an exact rational diagonal state.

The public APIs are:

```python
verify_diagonal_record_event_algebra(...)
verify_exact_diagonal_partition_pinching(...)
```

The first verifies, with `Fraction` arithmetic:

- a positive, unit-trace diagonal state;
- an exact orthogonal projective partition summing to identity;
- the selected event's Born weight and its required nonzero guard;
- normalized Lueders conditioning, event certainty, and repeatability;
- the fixed-state iff Born-weight-one criterion; and
- the conditional unital, positive, trace-preserving, idempotent pinching and
  exact commutant-range conclusions.

The second applies the diagonal partition pinching to an exact rational matrix
and checks trace preservation, idempotence, fixed-point iff block-commutant
membership, Hilbert--Schmidt Pythagoras, and contraction.

## Boundary

This module does not turn repair into pinching. It does not select a state from
an ambiguous observation fiber, authenticate an external record census,
construct a general complex matrix partition, bundle complete positivity,
prove a spectral gap, or promote physics. Floats and booleans are rejected.

The caller's `state_space_complete=True` is surfaced only as
`state_space_complete_declared`; `external_record_census_authenticated`
remains false. The supplied formal archive also has a stale `BUILD_RECEIPT.md`
entry in its hash list, so `theorem_archive_hash_bundle_authenticated` remains
false even though the individual Lean theorem-source hashes verify.

Run the focused checks with:

```bash
python3 -m pytest -q tests/test_finite_event_algebra.py
```
