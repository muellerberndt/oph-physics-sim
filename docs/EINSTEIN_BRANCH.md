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
- **The frontier.** Making the five measured clauses true requires the
  source-law conditions stated below; each proposed route is Lean-first where
  the condition is mathematical.

## Setup (one time)

```bash
cd oph-physics-sim
# The project virtual environment must exist; all commands use it.
.venv/bin/python --version   # expect Python 3.11+
```

No network access is needed. All runs are deterministic from declared seeds.

## Run the full Einstein-branch test battery

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

## Produce and verify the typed source tower

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
receipts until the generator-code firewall and federation binding are
attained. `True` for those two lines without those conditions would itself be
an invalid promotion.

## Read the five measured clause verdicts

Run each instrument and compare with the frozen verdicts.

```bash
.venv/bin/python - << 'PY'
from oph_fpe.bulk.modular_normalization_producer import produce_modular_normalization_report
from oph_fpe.bulk.gns_tower_producer import produce_gns_tower_report
from oph_fpe.bulk.event_manifold_producer import produce_event_manifold_report
from oph_fpe.bulk.stress_coupling_producer import produce_stress_coupling_report
from oph_fpe.bulk.einstein_branch_countermodels import produce_countermodel_matrix

r_normalization = produce_modular_normalization_report()
print("normalization:", r_normalization["verdict"], r_normalization["normalization_interval"])
r_gns = produce_gns_tower_report()
print("gns clauses:", r_gns["verdict"], r_gns["clause_verdicts"])
r_manifold = produce_event_manifold_report()
print("event manifold:", r_manifold["verdict"], r_manifold["held_out_quadratic_fit"]["inertia"])
r_coupling = produce_stress_coupling_report()
print("coupling:", r_coupling["verdict"], round(r_coupling["coupling_relative_spread"], 3))
r_countermodels = produce_countermodel_matrix()
print("countermodels isolated:", r_countermodels["all_countermodels_isolated"])
PY
```

**Expected verdicts for the current bounded instruments** (source-round
observer revision):

| Instrument | Verdict | Measured detail |
| --- | --- | --- |
| Normalization | `NOT_ATTAINED` | interval near `(-1.36, 0.95)`, acceptance band `(5.34, 7.23)` |
| GNS clauses | `NOT_ATTAINED` | cyclicity/separation/intersection True; future cone False (3 of 4 candidates positive) |
| Event manifold | `NOT_ATTAINED` | source-round provenance gives held-out inertia `(4, 0)` and a negative cone margin on the prescribed depth-plus-three-spectral-coordinate ansatz |
| Coupling | `NOT_ATTAINED` | ratio spread about `0.68` against envelope `0.10` |
| Countermodels | isolated: `True` | each clause family flips alone |

These `NOT_ATTAINED` verdicts are the honest status of the heuristic
dynamics, not defects in the instruments: every instrument carries negative
controls that all fail closed, so a passing verdict cannot be faked.

## Conditions that would change these verdicts

The relevant source-law hypotheses are:

1. **Thermalization:** a detailed-balance repair law against a
   geometric cap Hamiltonian makes the `2*pi` normalization a theorem.
2. **Positivity:** choose the generator family positive by
   construction; prove it the way `PortFrameGram.lean` proves its identities.
3. **Cone merge:** prove a finite overlap-density-implies-merge lemma;
   the density hypothesis is a counted quantity.
4. **Universality:** already a theorem (`A5CouplingSymmetry.lean`);
   the only open hypothesis is `A5`-equivariance of the implemented law,
   a finite check.
5. **Record spanning:** the source must emit snapshots that
   span port space, so faithfulness stops depending on the regularizer.

An attained condition changes a frozen assertion only after the corresponding
producer and independent verifier reproduce the new result.

## The declared v2 law

`oph_fpe/dynamics/geometric_law_v2.py` declares a second source law designed
so the measured conditions hold by construction where mathematics allows. Frozen
status (`tests/test_geometric_law_v2.py`):

| Condition | v2 status | How |
| --- | --- | --- |
| 2. Generator positivity | attained | positive by construction, 4/4 candidates |
| 4. Coupling universality | attained | `A5`-equivariant law; zero spread, per the Lean theorem |
| 5. Record spanning | attained | spanning snapshot family, full-rank raw moment |
| 1. Thermalization | not attained | two recorded v2 defects: non-Moebius geometry rows and framed-projection mismatch |
| 3. Cone merge | not attained | the current bounded event-manifold candidate has a negative cone margin |

These statuses concern the current finite source and cutoff. They do not
assert nonexistence, scaling convergence, or a continuum limit.

## Audited v1 simulator findings that affect results

The historical v1 audit found result-affecting sampling, cross-read, and
geometric-fit limitations. The current observer producer now emits complete
source rounds: every round commits all records, then performs same-round reads
on declared shared support carriers, then emits feedback; the next round's
records consume that feedback. Observer order within each phase is certified
as presentation metadata. This removes executor-order contamination, but the
result remains an observer instrumentation history over source snapshots, not
the complete physical repair-event order. A repair-only carrier is an
antichain in the current grammar because its reads consume version-zero roots.
Physical causet work therefore requires eventized, interleaved recurrent
propagation and seam repair. The prescribed three-coordinate spectral block
is an ansatz and supplies no independent dimension evidence.

## Claim boundary

Nothing on this page is a physical promotion. Every instrument prints
`physical_promotion_allowed: false`, every report is deterministic from
declared seeds, and no measured constant (couplings, masses, cosmological
values) appears anywhere in the source path.
