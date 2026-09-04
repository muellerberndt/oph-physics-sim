# Standard simulation-instrument preregistration template, v1

Status: committed template (declared 2026-09-01). Owning lane: V3 issue
[#737](https://github.com/FloatingPragma/observer-patch-holography/issues/737)
in `FloatingPragma/observer-patch-holography`. Companion machine-checkable
surface: `schemas/instruments/instrument_freeze_manifest_v1.schema.json`.

This template standardizes the preregistration structure first exercised by
`docs/OL_A1_PREREGISTERED_SIGNATURE_REPLICATION_2026-08-12.md` (instrument
INS-01). It extends that instance into a reusable contract; it does not
replace or reinterpret any committed preregistration. Every future simulation
instrument of the emergent-adequacy program is preregistered as one markdown
document following the numbered sections below, plus one freeze manifest JSON
conforming to the companion schema. Both are committed to this repository;
registration of the freeze in the upstream instrument register
(`docs/INSTRUMENT_REGISTER_V3.md` in `observer-patch-holography`) is a
separate upstream act and is not performed by this repository's lane.

Claim-tag rule, binding on every instance and on this template itself: every
claim carries one of **PROVEN** (a machine-checked theorem that ships in a
build graph), **TESTED** (verified by a committed, rerunnable check),
**ASSUMED** (a declared design choice or premise), or **UNKNOWN** (stated as
open). A preregistration is design; nothing in it is empirical validation,
and no instance may describe itself or its future verdict as empirical
validation of physics. An instrument verdict concerns what simulated
observers' records exhibit inside the architecture and makes no physical,
laboratory, or continuum claim.

## Lifecycle stages

An instrument moves through explicit stages. Each stage transition is one
commit, and later stages may not edit earlier-stage artifacts.

- **S0 SPECIFIED**: mutable design. No freeze artifacts, no authority.
- **S1 DESIGN-FROZEN**: the preregistration document and freeze manifest are
  committed. Everything analysis-relevant is fixed as text: binding,
  observables, analysis, controls, decision rule, seed *derivation
  procedure*, target/data access, and cost boundary. No seed integer exists
  yet. The manifest pins the document and all referenced code by sha256 and
  records `execution_authorized: false`.
- **S2 IMPLEMENTATION-FROZEN**: the frozen driver implementing S1 with no
  semantic discretion is committed, plus an addendum manifest pinning its
  sha256. The addendum may add pins only; any change to S1 semantics voids
  the instrument to S0 (a new campaign id is required, and the void is
  reported).
- **S3 SEED-COMMITTED**: execution has been separately authorized (see cost
  boundary). The seed table is committed per the pinned derivation procedure
  *before any run*, referencing the S1 document sha256 and the S2 driver
  sha256. After this commit the seeds are immutable no matter what.
- **RUNNING / verdict**: the campaign executes once under the pinned
  configuration; receipts are committed regardless of outcome; the frozen
  decision rule yields REPLICATED, FAILED, or INCONCLUSIVE, or the
  conformance block yields VOIDED_EXECUTION_ERROR.

S1 is the freeze this phase can perform when the owning issue authorizes
design only. S3 and RUNNING require the run authorization that the owning
issue withholds in a design-only phase.

## Required sections of an instance

**1. Identity and binding.** Instrument id from the upstream register (or
"pending registration"), campaign id, exactly one observation-ledger row,
exactly one composition lane, lineage predecessor and its status, and the
declared ledger consequence of each verdict. The instance must state that a
verdict applies to the ledger only through the upstream ledger-control
lineage's explicit selection, and which completed instrument currently
controls the row.

**2. Preregistration integrity rules.** At minimum: exactly one campaign per
preregistration; no seed re-draws, no reruns on any outcome, no post-hoc
edits to the document, driver, or decision rule; FAILED and INCONCLUSIVE
reported with the same structure and prominence as REPLICATED; a P0
nonconformance voids the campaign as an execution error and authorizes no
re-draw; the resolution limits of the replicate count stated plainly.

**3. Pinned code and environment.** Every code path the analysis depends on,
pinned by repository path and sha256 at the design-freeze commit, with the
repository base commit (an ancestor of the freeze-artifact commit, not
necessarily its direct parent or the freeze commit itself) recorded. Feature
flags that change analysis behaviour (for
example `standardize` on the quadratic-form fit) are pinned to explicit
values here, not left to defaults. The runtime environment pinning (thread
variables, single recorded environment) is declared.

Every instance also carries an **architecture-change declaration**: whether
the instrument needs any architecture change (new exports, conserved
labels, producer capabilities). If it does, the upstream premise register's
recorded-decision process must complete before stage S3, the recorded
decision id is added to the seed table, and a run without it is a P0
nonconformance. If it does not, the instance says so and identifies the
declared direct analysis paths pinned by the freeze manifest; this pin is not
a complete executable-dependency closure.

**4. Declared observables.** Each observable gets an id, an exact
computational definition in terms of the pinned code, a declared reference
value or band with the provenance and claim tag of that reference, and an
explicit **gating** or **recorded** role. A reference derived from archived
committed artifacts is admissible and tagged with its source; a reference
with no committed provenance is ASSUMED and says so. Recorded observables
are reported with the same prominence as gating ones.

**5. Declared controls.** Each control gets an id, an exact construction
(including its own derived-seed procedure), a quantitative fire/pass
criterion, and a stated authority: a control can gate the verdict or void
the campaign, but no control result may be silently dropped. Two control
roles are mandatory in every instance:

- an **invariance sham**: a transformation under which the analysis must be
  exactly invariant; any observable mismatch is an implementation defect and
  a P0 nonconformance, not a scientific outcome;
- a **sensitivity synthetic**: a planted effect of known sign that the same
  estimator must recover through the same pipeline; failure to recover it
  caps the verdict at INCONCLUSIVE (instrument insensitive). A synthetic
  recovery has no promotion authority.

**6. Frozen decision rule.** A complete, machine-evaluable rule naming
REPLICATED, FAILED, and INCONCLUSIVE, with the evaluation order stated
(conformance void first, then sham, then synthetic sensitivity, then FAILED,
then REPLICATED, else INCONCLUSIVE). No component pass may overwrite a
failed control or endpoint; no component failure may erase a separately
reported component pass; every component outcome appears in the campaign
summary. Any prospective power or precision computation supporting the
replicate count is shown with its assumptions tagged ASSUMED.

**7. Target and data access (target-clean declaration).** An explicit list
of what the run may read and a declaration that it reads nothing else: no
external measurement data, no archived campaign outputs as inputs, no target
values visible to the executing process. Declared references enter the
frozen decision rule as committed text only, never as runtime inputs.

**8. Seed protocol.** The master-seed freshness rule (the integer appears
nowhere in the named repositories at draw time, verified by grep and
recorded), the exact per-run seed derivation (a hash-based derivation from
the master seed and cell/replicate labels, so one committed integer
determines every stream), the execution-order randomization, and the
commit-before-run requirement. Every receipt must record the seed-table
commit hash; a receipt whose seed does not reproduce from the committed
table under the pinned derivation is a P0 nonconformance.

**9. Execution plan and cost boundary.** The exact command, environment,
output layout, and a laptop-scale cost cap. Runtime-only cost pilots may be
declared: their outputs are restricted to wall-clock and memory, they must
suppress every scientific field (no signature, margin, inertia, or control
outcome may be computed into any retained or displayed output), and their
receipts are committed. If the projected campaign cost exceeds the declared
cap, execution requires the separate larger-campaign authorization (owner
authorization, cost cap, complete receipt retrieval, resource shutdown) and
this instance's S3 stage is blocked until that exists.

**10. Receipts, manifest, and conformance block.** Declared receipt and
summary schema ids; one receipt per run committed regardless of outcome; one
campaign manifest with the sha256 of every output file; capture hashes for
raw outputs (raw arrays recomputable from seed determinism may be retained
as hashes, and the instance must say whether independent reconstruction is
possible without a rerun); and a P0 conformance block listing every
mechanical check (clean tree at run start, HEAD recorded, document and
pinned-code sha256 matches, thread environment, runtime-version uniformity,
executed cell count, sham invariance, seed-table reproduction). All checks
true gives CONFORMANT; any check false gives VOIDED_EXECUTION_ERROR,
reported with the same prominence as any verdict, authorizing no re-draw.

**11. Claim boundary.** What a REPLICATED verdict promotes, verbatim; what
it does not claim (no open-chart topology, no continuum limit, no physical
metric, no laboratory prediction, no entry into the frozen-prediction
ladder); which cautions every public sentence must carry; and the statement
that negative structural findings are bounded — in particular, a passive
rank defect or any other single-configuration negative reading is never
promoted to a universal obstruction.

**12. Fail-closed mapping.** A table mapping each no-cheating clause of the
owning issue to the mechanical check in this instance that catches it. The
table below is the template's minimum; instances copy it and fill the
right-hand column with their concrete artifacts.

| Clause | Mechanical check |
| --- | --- |
| Post-hoc analysis labelled as validation | Analysis fixed at S1; receipts carry the S1 document sha256; any analysis not derivable from the pinned text is outside the campaign and says so. |
| Instrument parameters revised after data | S1/S2 artifacts immutable by stage rule; P0 check hashes them at run start; semantic change voids to S0 with a new campaign id. |
| Silent reruns | One campaign per preregistration; seeds immutable after S3; receipts committed regardless of outcome; the manifest enumerates every run; a missing or extra receipt is a P0 nonconformance. |
| Controls omitted | Controls are gating or voiding by declaration; the campaign summary must report every declared control; an absent control result is a P0 nonconformance. |
| Passive rank defect promoted to universal obstruction | Claim-boundary section bounds every negative reading to its configuration; the summary's verdict sentence is copied from the frozen rule and admits no generalization. |
| Run not reproducible from committed artifacts | Seed-table commit-before-run; hash-derived per-run seeds; pinned code and environment; P0 seed-reproduction check. |
| Negative/control outcomes under-reported | Equal-prominence rule in the integrity section; the summary schema requires every component outcome; FAILED and VOIDED reports use the same structure as REPLICATED. |

## Freeze manifest

Each instance commits one JSON manifest conforming to
`schemas/instruments/instrument_freeze_manifest_v1.schema.json`, pinning the
instance document, this template, and every pinned code path by sha256, and
recording the stage, the freeze UTC time, the seed-policy state
(`declared` at S1/S2, `committed` at S3), and `execution_authorized`
(`false` until the owning issue's run authorization exists). The manifest and
the required validation procedure are the machine-checkable freeze surface:
schema validation checks its structure, and the digest-validation check below
refuses any working-tree file whose recorded sha256 does not match. Any receipt
that does not reference a committed manifest by sha256 is nonconformant by
construction.

Run both commands from the repository root for every manifest validation; a
zero exit status requires both checks to pass.

```sh
python3 -m jsonschema -i docs/OL_A1_INS02_FREEZE_MANIFEST_2026-09-01.json \
  schemas/instruments/instrument_freeze_manifest_v1.schema.json
python3 - docs/OL_A1_INS02_FREEZE_MANIFEST_2026-09-01.json <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
entries = [manifest["preregistration"], manifest["template"], *manifest["pinned_code"]]
mismatches = []
for entry in entries:
    path = pathlib.Path(entry["path"])
    if not path.is_file():
        mismatches.append(f"missing file: {path}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        mismatches.append(f"sha256 mismatch: {path}: expected {entry['sha256']}, got {actual}")
if mismatches:
    print("\n".join(mismatches), file=sys.stderr)
    raise SystemExit(1)
print(f"digest validation passed for {len(entries)} pinned files")
PY
```
