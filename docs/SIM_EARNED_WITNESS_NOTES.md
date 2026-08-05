# Sim-earned witness payload for Lean literals

`docs/SIM_EARNED_WITNESS_PAYLOAD.json` is a small exact extract from the B12
preregistered run `runs/b12_prereg_16k_20260806`. It exists so that a later
Lean witness can mirror realized finite run data as literals: an access cut
whose observer regions are these supports and whose records are these
classes. Producer: `scripts/extract_lean_witness_payload.py`.

## Contents

- Four patch observers (ids 36, 60, 64, 92), each with its support-node
  list truncated to the first 8 nodes, the integer record class and
  companion class of each truncated node, and the record-class counts over
  its full 96-node support.
- The realized record-class counts over the whole screen: 32 classes, 512
  patches each, total 16384.
- One exact joint table of (record class, companion class) counts over the
  union of the four full supports: 245 patches, 145 occupied cells.
- A class-structure block confirming agreement with the receipt's pinned
  reference: 32 fibers, 256 joint states, total mass 16384. The extraction
  fails hard on any disagreement.

Every datum is an integer or a pinned identifier. The file is 13 KB.

## Provenance chain

1. Run `b12_prereg_16k_20260806`, seed 20260806, git commit
   `b39b78fa`, wrote `freezeout_fields.npz` (committed per-patch fields)
   and `observer_views.jsonl` (observer supports and record readouts).
2. `conditional_resampling_realization_receipt.json` binned
   `record_signature` into at most 32 classes and
   `cumulative_repair_load` into at most 16 classes via
   `oph_fpe.dynamics.conditional_resampling.realization_inputs_from_freezeout`
   and pinned the realized joint frequency table as its common reference.
3. The extraction script calls the same producer function on the same
   freezeout file, so the payload's class labels are the receipt's class
   labels by construction, cross-checked against the receipt's pinned
   counts. Observer selection is the first 4 `patch_observer` rows of
   `observer_views.jsonl` in file order; file order is pinned by the
   sha256 in the provenance block.
4. The payload records sha256 digests (via `hashlib`) of the receipt, the
   freezeout bundle, and the views file, plus every extraction parameter.
   A rerun of the script reproduces the payload byte for byte.

## What a Lean consumer must and must not read into this

Must: the integers are exact realized counts and class labels on one
committed run, reproducible from the pinned inputs by the pinned script.
A Lean structure whose literals equal these integers mirrors realized
finite data.

Must not: no physical claim rides on the payload. The classes are outputs
of a declared binning (quantile edges over realized values) with no
canonical meaning; the observer selection and the 8-node truncation are
declared extraction parameters with no distinguished status; the counts
carry no probability claim beyond realized frequencies; nothing here
asserts a continuum limit, a spacetime region, an instrument, or a
measurement. The kernel-recognition certificate lives in the receipt, and
the payload does not restate it. Mirroring the integers in Lean carries no
claim that the mirrored structure is forced by the run.

## Candidate mapping onto the Lean interfaces

Target structures: `ObserverAccessCut` (`Lean/QFT/ObserverAccessCut.lean`)
and `OperationalObserver` (`Lean/Tower/OperationalObserver.lean`) in
`reverse-engineering-reality`.

- Each selected observer maps to one observer label `T.Observer r` at a
  single regulator; `observer_id` is the label literal.
- `support_nodes_first8` maps to the declared `observerRegion o`: a finite
  region literal whose node set is the truncated support.
- The record classes realized on a support map to generators of the
  commutative public record algebra `T.publicAlgebra r o`, one projector
  per class; `public_le_accessible` mirrors readability of the record
  classes on the support.
- The companion classes map to diagonal projectors inside
  `accessibleAlgebra o`, the coordinate the conditional-resampling kernel
  leaves unconstrained; `accessible_le_region` localizes them on the
  declared region.
- `realized_record_class_counts` map, as rationals `count / 16384`, to the
  fiber masses of the pinned reference law consumed by the recognizer
  interface (`kernel_eq_conditionalResamplingKernel_iff_recognition`,
  referenced by the receipt).
- `selected_support_joint_table` maps to a finite joint frequency literal
  over (record class, companion class) on the union region, a candidate
  pinned common reference for the cut's regions.
- For `OperationalObserver`: `label` is the constant family at one
  regulator; a candidate `readback` is the fiberwise conditional
  expectation onto record classes; `record_durable` mirrors the receipt's
  `unchanged_by_resampling` flag for the protected record. The `control`
  and `predict` fields have no counterpart in this payload and stay
  declared data on the Lean side.
