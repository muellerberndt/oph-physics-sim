# Capacity-indexed source-family replay

`capacity_indexed_source_family_projection.json` is the byte-canonical
producer projection consumed by the independent simulator-side checker. The checker
rebuilds four deterministic channels without importing the producer:

- reversible identity, with \(M_0=24k\);
- copy collapse, with \(M_0=24\);
- a two-class cap, with \(M_0=24\min(k,2)\);
- hidden spectator copies, with raw dimension \(24ks\) and
  \(M_0=24k\).

For every declared finite rung, the checker constructs the confusability graph
and proves its exact independence number from a disjoint-clique decomposition.
It also verifies the common source signature, common semantic pins, complete
sample grid, target-cleanliness flags, formula values, and bounded zero sets.

Run the replay with:

```bash
python -m oph_fpe.cosmology.capacity_indexed_family_verifier \
  data/capacity_readback/capacity_indexed_source_family_projection.json
```

The JSON Schema is
`schemas/cosmology/capacity_indexed_source_family_projection.schema.json`.
The checked-in projection is byte-identical to the issue 551 producer
projection in the research repository.

This is an independent finite replay of the declared branch grammar. It is not
an all-rung proof, an independent source construction, or a physical
cosmological closure.
