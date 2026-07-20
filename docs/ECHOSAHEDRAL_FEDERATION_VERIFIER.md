# Echosahedral federation verifier contract

This note documents what
`oph_fpe.core.echosahedral_federation` currently verifies, and what it does
not verify. It is an implementation contract, not a physical-emergence
claim.

## Cardinality

`carrier_count` is the exact number of entries in the declared finite source
federation. It becomes `exact_source_carrier_count` only when the carrier-ID
ledger is nonempty, string-typed, and collision-free. This is the cardinality
used for an exact 4k/16k/64k/256k carrier rung.

The carrier count is deliberately separate from all of the following:

- a support-mesh or chart regulator;
- an S2 chart-cell count;
- screen entropy capacity `N_star`;
- primitive-observer count;
- H3 point count; and
- event count.

The compact bundle therefore carries `support_regulator_count: null`. A
separate, independently replayed realization receipt would be needed to relate
the carrier federation to a support regulator.

## Shared local template

Canonical carriers are ID wrappers over one cached immutable regular
icosahedron template. The 12 port labels, 12 coordinate triples, 30 edges, 20
oriented faces, six antipodal pairs, and 60 A5 permutations are tuple-valued
and shared by identity. Local conformance is cached on the exact presentation
with the carrier ID normalized away.

At federation verification time, identical structural presentations are
grouped and audited once. The report states the exact number of unique
presentations and local audits. A malformed presentation remains a distinct
group and fails even when its carrier occurs beyond the report-detail limit.

## Scalable report behavior

The verifier visits every input row, but its JSON report is bounded:

- carrier detail, seam detail, boundary detail, observer detail, and blocker
  examples are capped at 64 rows;
- every seam, boundary, and observer row contributes to a length-delimited
  canonical SHA-256 stream digest;
- late failures are retained separately as bounded failure examples;
- sewn and external port occupancy uses one 12-bit mask per referenced
  carrier rather than a set of 12 endpoint tuples per carrier; and
- carrier-graph connectivity uses a disjoint-set ledger rather than a full
  NetworkX graph.

The bounded 130-carrier tests exercise the same report path that a 256k
federation would use. This removes the known report-amplification bottleneck,
but it is not a measured 256k physical-realization receipt. The input carrier
IDs and topology still require memory proportional to federation size.

## Interface scope

For each seam, equal endpoint hashes prove only that the declared interface
schema has the same content address at both endpoints. They do not construct
or verify:

- an algebra homomorphism or star-homomorphism;
- preservation of products, adjoints, states, or dynamics;
- composition or associator laws;
- triple-overlap cocycles; or
- higher-overlap coherence.

Accordingly these receipts remain false:

- `INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT`;
- `HIGHER_OVERLAP_COCYCLE_RECEIPT`;
- `FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT`; and
- `PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT`.

`FEDERATION_SEWING_RECEIPT` is only the structural finite sewing receipt: port
bijections, inverse maps, orientation signs, collar incidence, explicit
boundary coverage, connected carrier graph, connected declared observer
support, and interface-schema hash equality.

## Compact-bundle firewall

The reference bundle encodes the local template once and lists carrier IDs.
The verifier rejects:

- non-string or colliding IDs;
- a declared carrier count that differs from the ID array;
- a non-null support regulator folded into the carrier bundle;
- non-integer port/orientation arrays, including booleans;
- coerced numeric seam, boundary, or observer identifiers; and
- per-carrier coordinate, incidence, antipode, or A5 tables smuggled into the
  instrument surface.

Even a valid compact bundle does not earn source realization, refinement
naturality, emergent S2/H3/event geometry, a BW/KMS clock, or a physical
campaign admission receipt.
