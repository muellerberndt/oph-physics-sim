# B12 preregistered bounded source run (declared 2026-08-05, run id `b12_prereg_16k_20260806`)

This document is the preregistration for the bounded simulator source run
required by issue #688 (B12) in `reverse-engineering-reality`: a separately
preregistered bounded source run with explicit observer and transition caps,
a nonconstant protected record, and one pinned common reference used by both
the state and transition optimizers. Every declaration below is written and
hashed before the run command is issued. The 20260805 e6 runs referenced on
issue #688 were executed without preregistration and serve only as existence
witnesses; nothing from them enters the gates below except the pinned
producer code and the pinned base configuration.

## 1. Preregistration integrity rules

1. Exactly one run is performed under this preregistration. There are no
   seed re-draws, no reruns on gate failure, and no post-hoc edits to the
   gates or to this document.
2. A failed run is a valid negative result and will be reported as such.
   Failure of any gate closes this preregistration negatively; it does not
   justify adaptive quotient mining, alternative binnings, alternative
   companion fields, or alternative references.
3. The comparison receipt (Section 8) recomputes every recomputable gate
   independently from the raw run artifacts, exactly over the rationals
   where the gate is exact.

## 2. Bounded run configuration (declared before execution)

- Base configuration (pinned, unmodified):
  `configs/e6_axiom_manifest_16k_dense_observers.yml`,
  sha256 `0a23c3dd269d7f0479801e59cb42f9fabe1489b7de840294c7831dc416001be3`.
- Derived run configuration (byte-identical to the base configuration except
  for the two declared lines `seed: 20260806` and
  `run_id: b12_prereg_16k_20260806`):
  `configs/local/b12_prereg_16k_20260806.yml`,
  sha256 `ba1ec33e87f7920742b2f8491fc923a8d8f37bc60b7e4621a06d59f67f827302`.
  The seed is placed in the configuration file because the pipeline seeds the
  conditional-resampling producer from the loaded configuration, and the
  declared seed must govern both the base run and the producer sweep.
- Seed: **20260806**. This seed is fresh for this preregistration: no run,
  configuration, or script in the repository references it at declaration
  time. It differs from the 20260805 campaign seed.
- Caps: patch count **16384** (screen cap), observer sample count **1024**
  (observer cap), 128 dynamics cycles, 2048 repairs per cycle, 12-neighbor
  Fibonacci-sphere graph, group S3. All inherited verbatim from the pinned
  base configuration.
- Run id: `b12_prereg_16k_20260806`; output directory `runs/`.
- Exact command (visualization export is skipped; it feeds no gate):

```
.venv/bin/python -m oph_fpe.cli run-oph-universe \
  --config configs/local/b12_prereg_16k_20260806.yml \
  --out-dir runs \
  --run-id b12_prereg_16k_20260806 \
  --seed 20260806 \
  --skip-visualizations
```

  The `--run-id` and `--seed` flags repeat the values in the derived
  configuration; they must agree with it.
- Producer execution: `run_oph_universe_pipeline` invokes
  `write_conditional_resampling_realization` on the run directory as part of
  the pipeline. If the pipeline terminates after the base run without
  writing the receipt, the single declared fallback is to invoke
  `oph_fpe.dynamics.conditional_resampling.write_conditional_resampling_realization`
  on the run directory with `seed=20260806` and default sweep count, once.
- Pinned producer code (read-only, untouched by this campaign):
  - `oph_fpe/dynamics/conditional_resampling.py`,
    sha256 `3828608b860a3c5c3df223e71044bc73a5ae02c22c4849baee71a505557c8ed4`;
  - recognizer `oph_fpe/quotient/observable_normal_form.py`,
    sha256 `a5575469646880ac908955f6f970f7d95c86041ffe05dd4961585382c5c89959`.
- Environment: the repository venv `.venv/bin/python`; simulator tree at
  commit `b39b78f` plus uncommitted files outside `oph_fpe/`.

## 3. Protected record definition (declared)

The protected record is the run's committed freezeout `record_signature`
field (`runs/b12_prereg_16k_20260806/freezeout_fields.npz`, array
`record_signature`), mapped to integer classes by the producer's declared
binning `_class_bins` with `_MAX_RECORD_CLASSES = 32`: distinct realized
values map to distinct classes when there are at most 32 of them; otherwise
quantile edges over the realized values define the classes. The record class
of every patch is the protected datum; the resampling dynamics must leave it
unchanged at every patch through every sweep.

## 4. Companion coordinate (declared)

The resampled companion coordinate is the first field in the producer's
declared candidate chain (`cumulative_repair_load`, `stable_count`,
`repair_load`, `s3_class_density`) whose realized freezeout values are
nonconstant, binned by `_class_bins` with `_MAX_COMPANION_CLASSES = 16`.
The chain and its order are fixed by the pinned producer; no other companion
is admissible under this preregistration.

## 5. The single pinned common reference (declared)

The pinned reference is the **realized joint frequency table** over
(record class, companion class) pairs, built from the run's freezeout fields
with exact rational weights `count / total_count`. Only realized cells enter
the state space, so every weight is strictly positive and no regularization
is applied. This one table is the only reference object in the entire
package. Both optimizers below consume it; neither is permitted a private
or re-fitted reference.

## 6. The two consumers of the one reference (declared)

- **State optimizer** (information projection onto the reference): the
  state-side objective is the chi-squared divergence of a state law to the
  pinned reference table. Two instances are produced, both against the same
  table: (a) the exact replay, in which a declared perturbed start (mass
  moved between the two heaviest states) is pushed through the kernel once
  and its chi-squared to the reference is compared before and after, exactly
  over the rationals; (b) the empirical trajectory, in which every patch's
  companion class is displaced to its fiber's least-likely realized class
  and then resampled sweep by sweep, with the exact integer-count
  chi-squared to the reference reported per sweep.
- **Transition optimizer** (the resampling kernel): the transition kernel is
  `P(x, y) = 1[b(y) = b(x)] * pi(y) / pi(F_b(x))`, built from the same
  pinned reference `pi` restricted to each record fiber. Its required
  stationary law is that same reference, checked exactly over the
  rationals, together with exact idempotence and replay through the
  independent recognizer (R1 fiber support, R2 constant fiber rows, R3
  weighted detailed balance, explicit formula match).

The common-reference requirement is discharged structurally: the comparison
receipt rebuilds the joint table once from the raw freezeout arrays and
feeds that single object to both the state-side chi-squared recomputation
and the transition-side stationarity recomputation.

## 7. Pass/fail gates (declared, exact where stated)

Precondition P0 (run conformance): the run directory
`runs/b12_prereg_16k_20260806` exists; its manifest reports run seed
20260806 and patch count 16384; its stored configuration reports observer
sample count 1024; the realization receipt reports empirical seed 20260806
and provenance patch count 16384. P0 failure voids the run (wrong artifact
executed) and is reported as an execution failure, distinct from a
gate-negative result.

- **G1 (recognizer receipt)**: the receipt file
  `conditional_resampling_realization_receipt.json` has
  `CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT` true, is not a labeled skip,
  and every recognizer flag (`exact_table_recognition_receipt`,
  `r1_fiber_supported`, `r2_fiber_rows_constant`,
  `r3_weighted_detailed_balance`, `explicit_formula_match`) is true.
- **G2 (nonconstant protected record, at least 8 classes)**: the receipt
  reports `protected_record.nonconstant` true,
  `protected_record.unchanged_by_resampling` true, and
  `protected_record.class_count >= 8`; the comparison receipt independently
  recomputes the realized record class count from `freezeout_fields.npz`
  and requires the recomputed count to be at least 8 and to equal the
  receipt's count.
- **G3 (kernel stationary law equals the state-side reference exactly over
  the rationals)**: the receipt reports
  `exact_kernel_package.reference_stationary` true and
  `exact_kernel_package.idempotent` true; the comparison receipt
  independently rebuilds the joint table and the fiber kernel from the raw
  freezeout arrays with `fractions.Fraction` arithmetic and verifies
  `sum_x pi(x) P(x, y) == pi(y)` for every state `y` as exact rational
  equality, with `pi` the same table object used in the G4 recomputation.
- **G4 (chi-squared contraction)**: the receipt reports
  `exact_kernel_package.chi_squared_contracts` true; the comparison receipt
  parses `chi_squared_before` and `chi_squared_after_one_step` as exact
  rationals and verifies `after <= before`; additionally the empirical
  trajectory must exhibit the measured one-step collapse
  (`empirical_realization.one_step_collapse_measured` true, with the sweep-1
  exact chi-squared strictly below the displaced-start exact chi-squared as
  rationals).

The run passes if and only if P0 holds and G1 through G4 all pass.

## 8. Comparison receipt (declared output)

A standalone script `scripts/b12_prereg_source_run_receipt.py` (outside the
pinned `oph_fpe/` tree) reads this document's declared constants, the run
artifacts, and the realization receipt, recomputes the recomputable gates
from `freezeout_fields.npz` as described above, and writes
`docs/B12_PREREGISTERED_SOURCE_RUN_2026-08-05_receipt.json` with a per-gate
verdict and the exact witness values. The receipt records the sha256 of
this preregistration document and of the derived configuration.

## 9. Claim boundary

This run supplies the preregistered bounded-source-run ingredient of B12:
a bounded run with a nonconstant protected record and one pinned common
reference consumed by both the state-side information projection and the
transition-side resampling kernel, with exact receipts. It does not by
itself supply the remaining B12 receipts (global objective representation,
the collar-matrix realization on a recurrent mixing chain, refinement-uniform
low-temperature control), and it promotes no physical claim.
