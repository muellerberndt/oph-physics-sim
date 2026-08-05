# Theory-drift audit, 2026-08-05

Scope: the simulator audited against the theory repository's Lean stack
at `reverse-engineering-reality` commit `2bfa9b6b` (release r2015 plus
the V2 bounded-closure wave). Method: five parallel audit lanes (repair
kernel, thermal claims, seam conservation, receipt freshness, claim
boundaries) with every serious finding adversarially re-verified against
the code before acceptance. This file records what was fixed, what was
confirmed as minor drift for follow-up, and which correspondences were
verified as sound.

## Fixed in this commit

1. Receipt-chain custody break (root cause of the failing test
   collection): commit `4aa2ce7` updated cross-receipt pin hashes
   without rerunning the producers, so the vertex12 atomic receipt's
   internal source-capture sections no longer matched the current
   deterministic source engine, and the fail-closed verifier chain
   (atomic, directed transport, A2 endpoint commutator, constructive
   source law, ancestry inventory, CR-0 capability matrix, birefinement
   preflight) rejected on import. Every producer was rerun in
   dependency order and every independent verifier passes.
2. Lyapunov receipt semantics (`oph_fpe/consensus/lyapunov.py`): the
   receipt compared only within-row `phi_before -> phi` pairs, so a
   trajectory whose broken-edge count was raised between cycles by the
   observer-readback drive could not fail it; the shipped 128k earned
   run has 71 of 128 cycles rising between rows while the receipt
   recorded `max_delta = 0.0`. The receipt is chained across
   consecutive rows: cross-cycle injections are detected, reported per
   row, flagged as `driven_trajectory`, and fail the receipt. Rows
   without an explicit `phi_before`/`phi_prev` raise instead of
   silently comparing a value with itself. The earned-run bundles are
   immutable history; their recorded receipts predate this correction
   and must be read as within-cycle statements only.
3. Transition-clock reversibilization
   (`oph_fpe/cosmology/finite_repair_transition_clock.py`): the
   `reversible_empirical` matrix was built by symmetrizing raw counts,
   whose stationary law generally differs from the raw chain's. For an
   irreducible raw chain the additive reversibilization
   `(P + P*)/2` under the raw stationary law is used; the reducible
   fallback keeps the count-symmetrized estimator and is labeled
   through the report field `reversibilization_method`.
4. Transition-clock `min_transition_count` filter: windows were counted
   before the filter ran, so the filter filtered nothing while the
   skipped counter double-counted; windows now commit only when they
   meet the threshold.
5. Config self-description: three 4096-patch configs described
   themselves as `Compact 64k`; corrected to `Compact 4k`, and a stale
   legacy flag was dropped from the local smoke config.

## New correspondence checks

`tests/test_theory_correspondence.py` replays exact Lean statements
against simulator code paths in both directions: the conditional
resampling kernel of the finite thermodynamics package round-trips
through the simulator's exact recognizer with idempotence,
stationarity, and chi-squared contraction replayed over the rationals,
and a perturbed kernel is rejected; the fixed-word dependency cone of
the locality package holds bit-exactly for the simulator's own local
repair move (perturbing a node outside the cone leaves the probe
unchanged); cold repair traces pass the chained Lyapunov receipt and
driven traces fail it.

## Confirmed minor drift, tracked for follow-up

- The exogenous quiescent fallback in `RepairKernel.step` selects an
  arbitrary node when no mismatch is active, and the `local_best`
  proposal reads gauge-hidden neighbor interiors; both diverge from the
  proven trigger and congruence contract and should be documented or
  gated where receipts consume them.
- The `local_best` proposal is asymmetric, so the hot-Metropolis chain
  has no Gibbs stationarity guarantee; no current receipt claims one,
  and the `beta` parameter of the array engines is inert, but the
  parameter naming invites the wrong reading.
- The eight-state repair-load quotient's detailed-balance eligibility is
  satisfied by the reversibilized projection by construction; eligibility
  wording should name the projection rather than the raw chain.
- Aperiodicity in `_matrix_summary` is inferred from a positive
  diagonal, which is sufficient only; the label should say so.
- The simulator has no producer of the conditional-resampling kernel on
  observation fibres (only the recognizer); the four-law realization
  receipts continue to wait on the bounded new-source contract stated
  in the theory-side issue for the thermodynamic receipts.
- The B5 continuity witness is certified statically (incidence algebra)
  with no per-step regional-balance assertion in the run loops.
- Documentation and provenance items: stale `CLOSURE_LEDGER.md` anchors
  in the signature tracker, four e5 configs pinning a pre-migration
  theory-repo artifact surface, the configs README inventory drifting
  from the tracked set, floating-ref RER citations of sim receipts, the
  earned-run README describing rewritten bundles as byte-for-byte
  copies, orphaned gallium schema tags, and one clipped
  synthesis-table row.

## Theory implementation pass, same day

Following the audit, the current axiom basis and the conditional-
resampling package were implemented into the simulator directly.

- `data/theory/axiom_registry_pin.json` carries the canonical A1--A3
  statements verbatim from the theory repository's machine registry,
  pinned by commit and content hash; `oph_fpe/axioms.py` builds a
  per-run `axiom_manifest.json` mapping each axiom to the simulator
  structures realizing a finite fragment of it, with realization status
  stated explicitly. A test verifies the pin byte-for-byte against the
  theory checkout when one is present.
- `oph_fpe/dynamics/conditional_resampling.py` is the producer the
  audit listed as missing: the exact fiber-resampling kernel on a
  nonconstant protected record (the run's committed record classes)
  with a pinned common reference (the realized joint frequency table
  over the rationals). The kernel is replayed through the independent
  recognizer, its idempotence, stationarity, and chi-squared
  contraction are verified exactly, and an integer-count empirical
  trajectory from a displaced start is reported per sweep. The producer
  fails closed on a constant record and on all-singleton fibers, and
  selects the companion coordinate from a declared candidate chain,
  skipping fields frozen to a single value.
- `run_oph_universe_pipeline` writes both artifacts into every run and
  surfaces the realization receipt in the run summary. The e6
  dense-observer configs exercise the lane at 16k and 64k.
- The inventory catalogs in `source_operator_inventory.py` and
  `verify_source_operator_inventory_independent.py` declare the axiom
  pin, and the full custody cascade (capability matrix, vertex12 chain,
  bridges, preflight, inventory fixpoint) was regenerated after the
  final source edit.

## Verified sound

The canonical 30-seam/12-port tables and signed boundary conventions
match the Lean tables exactly, including the source-to-RER relabeling
bridge; the transactional repair layer implements strict-descent
acceptance with locality and dependency contracts; the consensus
certificate refuses schedule independence without its premises; the
graph Laplacian and conservative-mean operators match the typed
diffusion conventions; the KMS/BW, Boltzmann-bundle, and arrow lanes
attach their thermal vocabulary only to declared laws rather than to
the Metropolis kernel; and the four RER-cited receipts
(metric quotient, carrier action, CHSH candidate, source gap) verify
against their pinned producers.
