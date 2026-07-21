# Einstein Branch: What Is Proven, What Is Measured, How To Run It

This guide lets anyone, including an automated agent with no context, verify
the Einstein-branch status of OPH end to end. Every command is copy-paste.
Every expected output is stated. If an output differs from what this page
says, that difference is a finding to report, never something to hide.

## The division of labor

The rule of this program: **the simulator is used only for what mathematics
and Lean cannot decide.**

- **Proven (Lean, no simulation needed).** The conditional Einstein
  composition theorem (typed implication, sorry-free), the icosahedral
  port-frame Gram identities, the explicit order-sixty rotation action, and
  the universal-coupling theorem: for every `A5`-equivariant source law, the
  per-cap coupling ratios are equal with zero spread
  (`Lean/ObserverPatchHolography/Screen/A5CouplingSymmetry.lean` in the
  reverse-engineering-reality repository).
- **Measured (simulator instruments, this repository).** Whether the
  *current implemented source dynamics* attains the branch clauses. Five
  fail-closed instruments measure this. Their present verdicts are recorded
  below and frozen in the test suite.
- **The frontier.** Making the measured clauses true by construction:
  issue [#595](https://github.com/FloatingPragma/observer-patch-holography/issues/595)
  owns the five targets, with a Lean-first route named for each.

## Setup (one time)

```bash
cd oph-physics-sim
# The project virtual environment must exist; all commands use it.
.venv/bin/python --version   # expect Python 3.11+
```

No network access is needed. All runs are deterministic from declared seeds.

## Step 1: run the full Einstein-branch test battery

```bash
.venv/bin/python -m pytest -q \
  tests/test_einstein_tower_producer.py \
  tests/test_modular_normalization_producer.py \
  tests/test_gns_tower_producer.py \
  tests/test_event_manifold_producer.py \
  tests/test_stress_coupling_producer.py \
  tests/test_einstein_branch_countermodels.py
```

**Expected: all tests pass** (38 tests, a few minutes). A failure means
either the environment is broken or a frozen verdict has changed; both are
findings.

## Step 2: produce and verify the typed source tower (issue #572)

```bash
.venv/bin/python - << 'PY'
import tempfile
from oph_fpe.bulk.einstein_tower_producer import (
    produce_common_source_tower_bundle, verify_physical_source_binding)
with tempfile.TemporaryDirectory() as tmp:
    result = produce_common_source_tower_bundle(tmp)
    report = result["verifier_report"]
    print("tower receipt:", report["COMMON_DOMAIN_SOURCE_TOWER_RECEIPT"])
    print("refinement:", report["SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT"])
    print("splice rejection:", report["SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT"])
    print("overall receipt (pinned):", report["receipt"])
    binding = verify_physical_source_binding(result["manifest_path"])
    print("source binding replay:", binding["passed"])
PY
```

**Expected output:**

```
tower receipt: True
refinement: True
splice rejection: True
overall receipt (pinned): False
source binding replay: True
```

The overall receipt is `False` **by design**: the verifier pins two physical
receipts until the generator-code firewall and federation binding land
(owned by #595). `True` for those two lines before #595 closes would itself
be a bug report.

## Step 3: read the five measured clause verdicts

Run each instrument and compare with the frozen verdicts.

```bash
.venv/bin/python - << 'PY'
from oph_fpe.bulk.modular_normalization_producer import produce_modular_normalization_report
from oph_fpe.bulk.gns_tower_producer import produce_gns_tower_report
from oph_fpe.bulk.event_manifold_producer import produce_event_manifold_report
from oph_fpe.bulk.stress_coupling_producer import produce_stress_coupling_report
from oph_fpe.bulk.einstein_branch_countermodels import produce_countermodel_matrix

r573 = produce_modular_normalization_report()
print("573 normalization:", r573["verdict"], r573["normalization_interval"])
r574 = produce_gns_tower_report()
print("574 gns clauses:", r574["verdict"], r574["clause_verdicts"])
r575 = produce_event_manifold_report()
print("575 event manifold:", r575["verdict"], r575["held_out_quadratic_fit"]["inertia"])
r576 = produce_stress_coupling_report()
print("576 coupling:", r576["verdict"], round(r576["coupling_relative_spread"], 3))
r577 = produce_countermodel_matrix()
print("577 countermodels isolated:", r577["all_countermodels_isolated"])
PY
```

**Expected verdicts for the current source dynamics** (frozen 2026-07-22):

| Instrument | Verdict | Measured detail |
| --- | --- | --- |
| 573 normalization | `NOT_ATTAINED` | interval near `(-1.36, 0.95)`, acceptance band `(5.34, 7.23)` |
| 574 GNS clauses | `NOT_ATTAINED` | cyclicity/separation/intersection True; future cone False (3 of 4 candidates positive) |
| 575 event manifold | `NOT_ATTAINED` | held-out inertia `(2, 2)`, one time direction per observer chain |
| 576 coupling | `NOT_ATTAINED` | ratio spread about `0.68` against envelope `0.10` |
| 577 countermodels | isolated: `True` | each clause family flips alone |

These `NOT_ATTAINED` verdicts are the honest status of the heuristic
dynamics, not defects in the instruments: every instrument carries negative
controls that all fail closed, so a passing verdict cannot be faked.

## Step 4: what would change these verdicts

Only a change to the source law, tracked in #595. The Lean-first routes:

1. **Thermalization (573):** a detailed-balance repair law against a
   geometric cap Hamiltonian makes the `2*pi` normalization a theorem.
2. **Positivity (574):** choose the generator family positive by
   construction; prove it the way `PortFrameGram.lean` proves its identities.
3. **Cone merge (575):** prove a finite overlap-density-implies-merge lemma;
   the density hypothesis is a counted quantity.
4. **Universality (576):** already a theorem (`A5CouplingSymmetry.lean`);
   the only open hypothesis is `A5`-equivariance of the implemented law,
   a finite check.
5. **Record spanning (577 baseline):** the source must emit snapshots that
   span port space, so faithfulness stops depending on the regularizer.

When a target is attained, the corresponding frozen assertion in the test
suite must be flipped **deliberately, one commit per target, citing #595**.

## Campaign cells (larger runs)

Frozen H3/KMS campaign cells require the canonical single-threaded runtime:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m oph_fpe.bulk.physical_h3_kms_campaign \
  --seed 20260751 --rung 4096 --execute-physical-cell /tmp/cell_4k
```

Expected: `instrument_status: VALID_PASS`, `cell_scientific_status:
INCOMPLETE` (about 30 seconds). Rung 16384 and above are refused until the
lower cells are scientifically decisive; that refusal is the fail-closed
design working, and more compute cannot change it.

## The declared v2 law and the many-observer scaling result

`oph_fpe/dynamics/geometric_law_v2.py` declares a second source law designed
so the #595 targets hold by construction where mathematics allows. Frozen
status (`tests/test_geometric_law_v2.py`):

| Target | v2 status | How |
| --- | --- | --- |
| 2. Generator positivity | attained | positive by construction, 4/4 candidates |
| 4. Coupling universality | attained | `A5`-equivariant law; zero spread, per the Lean theorem |
| 5. Record spanning | attained | spanning snapshot family, full-rank raw moment |
| 1. Thermalization | open | two recorded v2 defects: non-Moebius geometry rows, framed-projection mismatch |
| 3. Cone merge | open, supported | see the scaling table |

The many-observer scaling probe (cross-reading round-robin observers on a
shared carrier) confirms the effective-description hypothesis: the held-out
event-form signature climbs with observer count, reaching the Lorentzian
inertia `(1,3)` at sixteen observers, though not yet stably at other scales
and with the cone margin still negative:

| observers | events | inertia | margin |
| --- | --- | --- | --- |
| 2 (v1 heuristic law) | 36 | (2,2) | negative |
| 2 (v2) | 72 | (1,0) | negative |
| 4 (v2) | 144 | (1,1) | negative |
| 16 (v2) | 576 | (1,3) | negative |
| 32 (v2) | 3456 | (1,2) | negative |
| 128 (v1 fixed capture, 16384 carriers, dense supports) | 2304 | (1,3) | negative |

Reading: Einstein spacetime behaves as an effective many-observer
description, and dense cross-observer record reading is the mechanism that
merges causal cones. The strongest row is the last: with the three audited
capture defects fixed (cross-observer reads, spanning snapshots, held-out
geometry transport), the REAL v1 repair dynamics at a 16384-carrier
federation with 128 densely packed observers produces 348 cross-observer
ancestry edges and a held-out form with Lorentzian inertia (1,3). At the
same scale the coupling spread falls to 0.19 (from 0.68 at small cutoff),
trending toward the zero-spread symmetric limit the Lean theorem fixes. A stable `(1,3)` with positive margin needs a better
event chart and denser overlap schedules, tracked in #595.

## Audited v1 simulator findings that affect results

The v1 capture audit found four result-affecting limitations, all now
documented: record snapshots sample only the observer-support carriers (four
of thirty-two at defaults); observer supports are small disjoint balls, and
the observer loop never reads records committed by other observers, so
cross-observer ancestry is structurally zero regardless of dynamics; and the
geometry rows derive the flow parameter from the cross ratio, making the
geometric rate fit circular. The instrument verdicts on the heuristic law
were re-measured under widened sampling and remain NOT_ATTAINED, so the
verdicts stand; the structural cross-read gap is fixed by the v2 law design.

## Claim boundary

Nothing on this page is a physical promotion. Every instrument prints
`physical_promotion_allowed: false`, every report is deterministic from
declared seeds, and no measured constant (couplings, masses, cosmological
values) appears anywhere in the source path.
