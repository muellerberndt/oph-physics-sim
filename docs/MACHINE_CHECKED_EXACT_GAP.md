# A machine-checked proof of `exact_gap.positive` for the frozen local domain

This note describes an external Lean 4 development that proves, by kernel evaluation on
this repository's own frozen data, the statement that `source_gap_receipt.json` records as
`exact_gap.positive`.

It adds no code and changes no behaviour. It is a pointer plus an honest statement of what
is and is not proved.

Repository: https://github.com/velvetmonkey/oph-gap-lean (Apache-2.0)

## Why this might be useful

The receipt states its own basis plainly. From `data/local_domain/source_gap_receipt.json`,
`exact_gap.argument`:

> "positive semidefinite by the kinetic identity and kernel-free by the signed-incidence rank
> theorem, with an explicit negative-cycle witness for every frustrated component"

and `oph_fpe/local_domain/stage3_typed_domain.py` says the same thing about scope:

> "on a fully frustrated domain the kernel-zero direction rests on the rank theorem, not on
> this loop"

So the kernel-free step is carried by a cited theorem, and the repository says so rather than
hiding it. The Lean development discharges that citation for this one frozen graph: it proves positive-definiteness directly
from a negative-cycle witness and a spanning tree, both decided by the Lean kernel on the
literal edge list, so the conclusion no longer rests on a citation.

## What is proved

For the domain pinned by `domain_freeze_sha256 = a0be6fc64aecf9ca375fd91c57315e8af5e5cf161c99611f4844ba8f452ae7ff`, with 8,662 carriers, 11,816
seams and all seam signs `-1`:

```lean
theorem oph_original_gram_posDef :
  OrigGramPosDef origNodes origEdges oph_endpoints

theorem oph_original_eigenvalues_pos :
  OrigEigenvaluesPos origNodes origEdges oph_endpoints oph_original_gram_posDef
```

`origNodes` and `origEdges` are this repository's own carrier ids and its own seam list in
its own order. The second theorem says every eigenvalue of `Dᵀ D` is strictly positive,
which is the receipt's "gap".

The chain is:

1. `check_true` evaluates the checker, by the Lean kernel, on the literal 11,816-edge list
   and the receipt's own eleven-index witness cycle
   `[5923, 14860, 1662, 11420, 12610, 188, 16358, 15744, 16262, 8315, 15104, 5923]`.
2. A passing check yields a spanning tree and a negative closed walk.
3. Those two facts force `ker D = 0`, hence `Dᵀ D` positive definite.
4. Every eigenvalue of a positive definite matrix is strictly positive.

The receipt's `component_count: 1`, `twisted_kernel_dimension: 0`,
`signed_incidence_rank: 8662` and `negative_cycle_witnesses_verified: true` all follow for
this graph from the same theorems.

## How to check it

```
git clone https://github.com/velvetmonkey/oph-gap-lean
cd oph-gap-lean && lake build
```

Zero `sorry`. Zero `native_decide`; every decision is `decide +kernel`. No axioms beyond
`propext`, `Classical.choice` and `Quot.sound`, which `#print axioms` reports for each
top-level theorem.

The data module is generated from this repository's Stage 3 output and carries the same
`domain_freeze_sha256`. If that hash changes, the proof is about a different graph and must
be regenerated.

## What is NOT proved, stated plainly

* **This is one frozen domain, not a general result about the pipeline.** The general
  theorem (a signed graph with a negative cycle and one connected component has positive
  definite `Dᵀ D`) is also in the repository, but the machine-checked instance is this
  graph only.
* **Nothing here validates the physics.** It proves a linear-algebra statement about a
  matrix built from a seam list. Whether that matrix is the right object for the claim it
  supports is not addressed.
* **The data extraction is Python.** The seam list is generated from this repository by a
  script; the Lean side then proves things about the resulting literal list. The generator
  is not itself verified, so the honest claim is "given this edge list, the gap is
  positive", not "the pipeline is correct".
* **Nothing is claimed about `physical_promotion_allowed`** or any downstream verdict.

## Negative controls

The development includes controls that must FAIL, so that a vacuous encoding would be
visible:

* an all-negative 4-cycle with no negative cycle: positive-definiteness is refused;
* the witness with one edge sign flipped: the checker returns `false`;
* a deliberately wrong vertex set and a colliding relabelling: both refused.

## Contact

Raised by Ben Cassie (@velvetmonkey). Happy to move this into the repository proper, to run it in
CI, or to close it if the pointer is not wanted.
