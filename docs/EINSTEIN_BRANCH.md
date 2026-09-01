# Einstein effective-description bridge

This guide separates the exact source-causal result from the additional
hypotheses needed to use Einstein curvature as a smooth effective
description. It is a reproduction and claim-boundary document, not a status
tracker.

## Division of labour

- The companion Lean library proves conditional mathematical implications,
  including the Einstein composition theorem, port-frame Gram identities,
  the order-sixty rotation action, and universal coupling for an
  `A5`-equivariant source law.
- This simulator generates finite observer-like self-reading systems and
  checks whether concrete source dynamics attain named hypotheses. Each such
  system has bounded local state, ports or boundaries, readback, records,
  feedback or repair moves, and a public evidence bundle.
- The smooth Einstein equation is an effective-description consumer. It does
  not create the physical event carrier, source-selected regulator family,
  Lorentz cone, volume calibration, topology, or continuum limit that its
  hypotheses require.

## Canonical source-causal input

The source-order producer removes declared ancestry and reconstructs the
finite informational order from authenticated read-after-write provenance.
The history-family producer runs the registered capture independently at 4,
8, 16, 32, and 64 complete rounds. Its independent verifier reconstructs each
raw cutoff log separately and checks that every adjacent direct and transitive
order is the exact induced restriction.

```bash
.venv/bin/python -m oph_fpe.bulk.source_derived_causal_order \
  --out data/causal_order/source_derived_causal_order_receipt.json
.venv/bin/python -m oph_fpe.bulk.verify_source_derived_causal_order_independent \
  --receipt data/causal_order/source_derived_causal_order_receipt.json
.venv/bin/python -m oph_fpe.bulk.source_causal_history_family \
  --out data/causal_order/source_causal_history_family_receipt.json \
  --publication-out data/causal_order/source_causal_history_family_publication_projection.json
.venv/bin/python -m oph_fpe.bulk.verify_source_causal_history_family_independent \
  --receipt data/causal_order/source_causal_history_family_receipt.json \
  --projection data/causal_order/source_causal_history_family_publication_projection.json
```

The positive result is exact informational-history custody. The checked
family has width two at every cutoff and lengthens a fixed-width history; it
is not a fixed-region density refinement. It therefore supplies no physical
causal attachment, faithful embedding, manifoldlikeness, 3+1 dimension,
count-volume law, Lorentzian metric, topology, or continuum limit.

## Prescribed rank-three placement diagnostic

The finite receipt also tests one stipulated placement. Record commits use
rank-three icosahedral port anchors in a single shared reference frame;
readback and feedback events use consumed-record barycentres; time is one
global scale times source-derived longest-path rank. The resulting exact
two-direction cone test is noninjective and has no admissible global time
scale on the bounded member.

Neither inter-carrier frame gluing nor the barycentre-selection rule is
source-derived. Consequently this result rejects that prescribed ansatz only.
It is not a no-go theorem for other source-selected placements and cannot be
used as evidence against a future physical causal refinement.

## Requirements for an Einstein continuum interpretation

A physical promotion requires one common source-bound chain that supplies:

1. a physical repair or propagation event carrier rather than instrumentation
   events over source snapshots;
2. a source-selected regulator family that increases event density and
   spatial antichain capacity in comparable physical regions;
3. faithful causal embedding and manifoldlikeness diagnostics on that family;
4. stable 3+1 dimension, count-volume calibration, and Lorentz-cone recovery;
5. compatible chart gluing, topology, and a controlled smooth limit; and
6. the stress-response hypotheses consumed by the conditional Einstein
   theorem.

Finite H3, BW/KMS, stress-coupling, and Einstein-bridge artifacts remain
useful compatibility diagnostics, but none can substitute for these upstream
source-causal requirements.

## Focused verification

```bash
.venv/bin/python -m pytest -q \
  tests/test_source_derived_causal_order.py \
  tests/test_source_causal_history_family.py \
  tests/test_einstein_tower_producer.py \
  tests/test_modular_normalization_producer.py \
  tests/test_gns_tower_producer.py \
  tests/test_stress_coupling_producer.py
```

Every physical-promotion flag remains fail-closed unless the corresponding
producer and independent verifier reproduce the complete common-source chain.
