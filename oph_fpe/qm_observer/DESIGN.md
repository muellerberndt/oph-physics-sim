# Observer-frame quantum statistics probe (lane D3)

Status: exploratory, non-evidential. No instrument is frozen, no run is
evidential, INS-01 remains the controlling OL-A1 verdict, and OL-C5 status is
untouched by this lane. This document precedes and pins the code in
`oph_fpe/qm_observer/`.

## 1. Purpose

The probe demonstrates quantum statistics in the observer frame: the exact
committed Born weights of the eight PR-04 measurement contexts are reproduced
as integer count ratios over a finite deterministic record ensemble driven by
the declared branch tables, which are transcriptions of the committed
conditional weights Tr(p F) (section 3.3), with no probability postulate
anywhere in the code path, and measurement collapse is realized as
conditioning on the observer's record. The probe feeds a visualizer through
a documented JSON export.

## 2. Committed sources, per modeling choice

Every modeling choice below carries its source. Committed modules are
imported read-only; nothing outside `oph_fpe/qm_observer/` and
`tests/test_qm_observer.py` is edited.

| Choice | Source |
| --- | --- |
| The eight contexts, their exact effect matrices, the run literals (111, 68) at mass 179, the exhaustive-deterministic semantics, the Q(sqrt(3), i) machinery, `canonical_sha256` | `oph_fpe/quantum/phase_operation.py` (sim, committed PR-04 module) and `oph_fpe/core/charged_response.py` |
| The committed outcome weights: 111/179 on the three diagonal contexts, 315/716 on the four rotated contexts, 1/2 on the phase context, with counts (111, 68), (315, 401), (179, 179) at masses 179, 716, 358 | `reverse-engineering-reality/code/phase_operation_producer/PHASE_OPERATION_RECEIPT.v1.json`, payload sha256 `71a06f1c15192123cd09feb2386da702b572c8ac57c9b7633f5aa60c5d404e22` |
| Conditioning rule: Lueders update `rho -> P rho P / Tr(rho P)`, repeatability, idempotence | `reverse-engineering-reality/Lean/EventAlgebra/Lueders.lean` |
| Every count frequency of a committed-core inhabitant is an exact affine function of one free coordinate; the integer receipt window `32041*(a-b)^2 <= 30192*(a+b)^2` is scoped to committed-core phase count pairs | `reverse-engineering-reality/Lean/EventAlgebra/PhaseInstrumentDetermination.lean` |
| The phase lift is a declared architecture operation (register row PR-04, disposition axiomatize, decision date 2026-08-18, lane issue 730), never derived from source dynamics; the completion ambiguity is one conjugation orbit | `oph_fpe/quantum/phase_operation.py` docstring, `Lean/QFT/ConjugationGauge.lean`, `Lean/QFT/SourceOrientedCompletion.lean` |
| Claim boundary language: projector webs alone underdetermine the weight; frequencies alone do not prove the universal valuation law; the current source supplies no phase operation, rotated/phase outcome receipts, or common-preparation instrument validation | `reverse-engineering-reality/paper/observers_are_all_you_need.tex`, Born passages, and `Lean/EventAlgebra/FiniteBornFrame.lean` |

The exhaustive-deterministic semantics of the committed corpus reads: every
count pair is the exact Born weight of the context effect under the committed
record-diagonal run state diag(111/179, 68/179), computed in exact arithmetic
over Q(sqrt(3), i) and scaled to the least positive integer multiple of the
committed run mass 179 that makes both counts integers. Probabilities are
count ratios over a finite deterministic ensemble. The committed corpus does
not claim a Born-rule derivation: the operation is declared, the counts are
produced by the declared semantics, and the physical-attachment premises of
the register stay open.

## 3. Record-ensemble construction

### 3.1 Micro-configurations

A micro-configuration is a tuple (index, class, history). The class is a
record class label naming an exact rank-one projector or basis projector; the
history is the tuple of (context, outcome) records the configuration carries.

The base ensemble is the committed run population: 179 micro-configurations,
111 in class `rec0` (record bit 0) and 68 in class `rec1` (record bit 1).
Source: the committed run literals `run_counts = [111, 68]`, `run_mass = 179`
of the reference receipt inputs, which are the record state of a 16k run;
reading them as an exhaustive record population is the declared convention of
section 8.1. No amplitude enters this construction.

### 3.2 Context application by integer branch rules

Applying a measurement context to an ensemble proceeds by uniform
refinement and deterministic outcome assignment:

1. For every class present, the declared branch table supplies an exact
   rational pair (t, 1 - t) with t = num/den in lowest terms: the outcome-0
   fraction on that class.
2. L is the least common multiple of the class denominators. Every
   micro-configuration splits into exactly L sub-configurations, indexed
   j = 0 .. L-1.
3. Sub-configuration j of a configuration in class p receives outcome 0 when
   j < L * num(p) / den(p), outcome 1 otherwise. The threshold is an integer
   by construction of L.
4. The recorded outcome is appended to the history; the class of a
   sub-configuration with recorded outcome k under context c becomes the
   conditioned class of (c, k) (section 4).
5. Counts per outcome are obtained by explicit enumeration: iterating the
   ensemble and tallying integer multiplicities. No multiplication of a
   weight by a mass stands in for the tally.

Each micro-configuration produces exactly one recorded outcome. The
observer-frame statistic of an outcome is its exact integer count over the
ensemble; a ratio of counts is a `Fraction` of two integers.

### 3.3 Declared branch tables

The branch tables live in `tables.py` as integer literals. Each entry is the
exact conditional outcome-0 weight Tr(p F) of probe effect F on class
projector p, transcribed from the committed exact matrices and cited here.
The distinct effects among the eight committed contexts:

- `P_rec = [[1, 0], [0, 0]]` (contexts web_diagonal, web_conjugated_0,
  web_conjugated_1),
- `E_A = [[1/4, -sqrt(3)/4], [-sqrt(3)/4, 3/4]]` (web_conjugated_2,
  web_conjugated_3),
- `E_B = [[1/4, sqrt(3)/4], [sqrt(3)/4, 3/4]]` (web_conjugated_4,
  web_conjugated_5),
- `Y_plus = [[1/2, -i/2], [i/2, 1/2]]` (phase).

Classes and their outcome-0 fractions per probe effect (P_rec, E_A, E_B,
Y_plus):

| class | projector | P_rec | E_A | E_B | Y_plus |
| --- | --- | --- | --- | --- | --- |
| rec0 | `[[1,0],[0,0]]` | 1 | 1/4 | 1/4 | 1/2 |
| rec1 | `[[0,0],[0,1]]` | 0 | 3/4 | 3/4 | 1/2 |
| A0 | E_A | 1/4 | 1 | 1/4 | 1/2 |
| A1 | I - E_A | 3/4 | 0 | 3/4 | 1/2 |
| B0 | E_B | 1/4 | 1/4 | 1 | 1/2 |
| B1 | I - E_B | 3/4 | 3/4 | 0 | 1/2 |
| Y0 | Y_plus | 1/2 | 1/2 | 1/2 | 1 |
| Y1 | I - Y_plus | 1/2 | 1/2 | 1/2 | 0 |

The table is declared data on the counting path. Its verification against
the trace formula happens only in the receipt layer (section 6), never
inside the counting module.

### 3.4 The receipted identity

For every committed context, three separately computed quantities are
receipted equal:

1. the exact integer count ratio over the enumerated ensemble
   (`ensemble.py`, integers and Fractions only, no matrix, no trace),
2. the squared-amplitude value Tr(rho E) computed in Q(sqrt(3), i)
   (`amplitudes.py`, matrices and traces only, no ensemble, no count),
3. the committed exact rational weight of the reference receipt row.

The counting path never references amplitudes and the amplitude path never
references counts; section 6 receipts that separation by an import-graph
check. The inputs of the counting path are the declared branch tables of
section 3.3, which are transcriptions of the amplitude conditionals Tr(p F),
so the identity is a consistency receipt over the transcription, the
refinement rule, and the enumeration, not an independent confirmation of the
committed weights. Expected identities on the base ensemble:

| context | counts | mass | ratio = committed weight |
| --- | --- | --- | --- |
| web_diagonal, web_conjugated_0, web_conjugated_1 | (111, 68) | 179 | 111/179 |
| web_conjugated_2 .. web_conjugated_5 | (315, 401) | 716 | 315/716 |
| phase | (179, 179) | 358 | 1/2 |

## 4. Collapse as conditioning

Conditioning on outcome k of context c selects the sub-ensemble whose record
shows (c, k). Every micro-configuration of the sub-ensemble is in the
conditioned class of (c, k): the class whose projector is E_c for k = 0 and
I - E_c for k = 1 (all eight committed effects are projectors, so both
outcome effects of every context are projectors).

Receipted properties, each an exact integer identity:

1. Record persistence: an immediate repeat of context c on the conditioned
   sub-ensemble yields outcome k on every micro-configuration, count
   fraction exactly 1. On the counting path this is the branch-table entry
   1 (respectively 0) of the conditioned class under its own effect; on the
   amplitude path this is Lueders repeatability, Tr(P' E_c) = 1 for
   P' = E_c.
2. Projection statistics: a different context d applied to the conditioned
   sub-ensemble yields counts whose ratio equals the exact Born weight
   Tr(rho' E_d) of the Lueders-updated state rho' = E_ck rho E_ck /
   Tr(rho E_ck). For rank-one projectors the updated state is the projector
   itself, so the amplitude path computes Tr(P_ck E_d) exactly in
   Q(sqrt(3), i).
3. Fail-closed conditioning: conditioning on an outcome realized by zero
   micro-configurations raises a typed error; no empty ensemble is
   constructed.

The collapse chains of the standard run: for each of the eight contexts c
and each outcome k realized on the base measurement, condition, repeat c,
then apply each probe context d in the fixed probe set {web_diagonal,
web_conjugated_3, phase}.

Scope convention: the committed integer window
`32041*(a-b)^2 <= 30192*(a+b)^2` applies to phase count pairs of the
committed core (`PhaseInstrumentDetermination.lean` scopes it to
committed-core inhabitants). Conditioned sub-ensembles carry pure states
outside the committed core, so the window clause is checked on the base
phase pair (179, 179) only.

## 5. Interference receipt

Context pair: first = web_conjugated_3 (the committed rotated context),
second = web_diagonal (its base context).

- Direct: the second context on the base ensemble counts (111, 68) at mass
  179; at the common comparison mass 2864 this is (1776, 1088).
- Mediated: the first context on the base ensemble counts (315, 401) at
  mass 716; conditioning on each outcome and applying the second context
  gives (315, 945) on the A0 sub-ensemble and (1203, 401) on the A1
  sub-ensemble; the sewn totals at mass 2864 are (1518, 1346).
- Exact gap: 1776 - 1518 = 258 at mass 2864, the Fraction 129/1432.

Classical mixture family: every model in which each micro-configuration
carries simultaneous definite outcome values for both contexts and
measurement reveals the value without rewriting the record satisfies the
counting identity direct = mediated (a sub-population tally sews back to the
whole-population tally). The recorded gap 129/1432 differs from 0, so the
recorded statistics differ from every member of the family. The receipt
carries the exact integers and the exact rational gap. The tests carry a
reveal-only classical mock whose mediated counts equal its direct counts and
which the interference check therefore rejects.

The gap has a companion reading on the amplitude path: measurement rewrites
the record because the conditioned classes A0, A1 are non-diagonal
projectors, and Tr(P_rec E_A-conditioned mixture) differs from
Tr(rho_base P_rec) by the exact off-diagonal transfer. No Bell-type or
locality claim is made; the receipt is a measurement-disturbance identity.

## 6. Module layout and independence receipt

- `tables.py`: declared integer literals only: class labels, branch tables,
  context table, scenario declarations, citations. Imports nothing from this
  package and nothing from `oph_fpe.quantum`.
- `ensemble.py`: micro-configuration enumeration, context application,
  conditioning. Imports `tables` and the standard library only. Never
  imports `amplitudes`, never imports `oph_fpe.quantum`.
- `amplitudes.py`: exact Q(sqrt(3), i) state and effect matrices, Born
  weights, Lueders updates, all through the committed `phase_operation`
  machinery. Never imports `ensemble`, never imports `tables`.
- `receipt.py`: runs both paths, demands the identities of sections 3 to 5,
  verifies the branch tables against the trace formula, performs the
  import-graph independence check by parsing the module sources with `ast`,
  builds the receipt JSON and the visualizer export, carries the CLI.
- `__main__.py`: argument parsing for `python -m oph_fpe.qm_observer`.

The independence receipt records, per module, the absolute import set found
by AST parsing, the dynamic-import call set, and the sha256 of the inspected
source, and demands: `ensemble` imports neither `amplitudes` nor any matrix
machinery; `amplitudes` imports neither `ensemble` nor `tables`; none of the
three inspected sources carries an `__import__` or `importlib.import_module`
call. The check therefore covers static cross-imports and those named
dynamic-import forms; the source hashes pin the audited code, so the
receipted import sets and the counts belong to the same bytes. A mutant that
drops one micro-configuration breaks the count-ratio identity because the
enumerated mass and tally no longer match the exact rational weight.

## 7. Visualizer export schema

Schema id: `oph.sim.qm_observer_viz.v1`. One JSON object:

```
{
  "schema": "oph.sim.qm_observer_viz.v1",
  "labels": {
    "exploratory": true,
    "evidential": false,
    "statement": "..."
  },
  "boundary": "...",
  "fraction_encoding": "every exact rational is a two-integer array [numerator, denominator]",
  "scenarios": [Scenario, ...]
}
```

The `boundary` field carries the claim-boundary statement of section 9,
byte-identical to the `boundary` field of the receipt JSON.

Scenario object:

```
{
  "scenario_id": "...",
  "kind": "base_context" | "collapse_chain" | "interference",
  "context_sequence": ["web_conjugated_3", "web_diagonal", ...],
  "tree": Node,
  "collapse_events": [CollapseEvent, ...],
  "display": DisplayBlock
}
```

Node object (the branch tree; exact integers and [num, den] rationals):

```
{
  "node_id": "...",
  "context": "web_conjugated_3" | null,
  "class_counts": {"rec0": 111, "rec1": 68},
  "mass": 716,
  "counts": [315, 401],
  "weights": [[315, 716], [401, 716]],
  "children": [Node, ...]
}
```

The root node carries `context: null`, `counts: null`, and `weights: null`:
an unmeasured population has no recorded outcome to tally. Its `class_counts`
and `mass` fields are populated. A child node describes one context
application to its parent's (possibly conditioned) population.

Interference scenarios carry one additional `comparison` object with
the common mass, both count vectors at the common mass, the exact gap
as `[num, den]`, and the classical-family statement.

CollapseEvent object:

```
{
  "after_context": "web_conjugated_3",
  "conditioned_outcome": 0,
  "before": {"mass": 716, "counts": [315, 401]},
  "after": {"class": "A0", "mass": 315}
}
```

DisplayBlock: `{"derived_for_display": true, "float_weights": {...}}`, float
renderings of the exact weights for convenience only, labeled derived; every
float is reproducible from the exact fields and carries no receipted
content. Total export size is far below the few-MB target: the standard run
holds 8 base scenarios, the collapse chains, and one interference scenario.

The separate receipt JSON (schema `oph.sim.qm_observer_probe.v1`) carries
the identities, the independence receipt, the interference integers, the
export body sha256, and a `receipt_sha256` over its own canonical body via
the committed `canonical_sha256`. Byte reproducibility: both files are
`json.dumps(..., sort_keys=True, indent=2)` over content free of paths,
timestamps, and environment data.

## 8. Declared conventions beyond the committed corpus

1. The base ensemble population (111 in rec0, 68 in rec1) reads the
   committed run literals as an exhaustive record population. The literals
   are the record state of a 16k run (`REFERENCE_RUN_ID`
   `b12_prereg_16k_20260806`), not a 179-member population; the exhaustive
   reading is this convention and carries no source claim.
2. The branch tables of section 3.3 are declared literal data on the
   counting path, transcribed from the exact conditional weights Tr(p F).
3. Uniform refinement by the least common multiple of class denominators.
4. The record-persistence rule: a conditioned class answers its own context
   with its conditioning outcome on every micro-configuration.
5. The probe set {web_diagonal, web_conjugated_3, phase} for collapse
   chains.
6. The class labels rec0, rec1, A0, A1, B0, B1, Y0, Y1.
7. The interference comparison at the common mass 2864.
8. The committed-window scope convention of section 4.
9. The export schema `oph.sim.qm_observer_viz.v1` and the receipt schema
   `oph.sim.qm_observer_probe.v1`.

## 9. Boundaries

The phase operation is a declared architecture operation under the PR-04
recorded decision (2026-08-18, disposition axiomatize, lane issue 730), not
derived from source dynamics. This probe establishes that observer-frame
record counting reproduces the exact quantum weights without a probability
postulate, and that conditioning reproduces projection under the declared
record-persistence and conditioned-class conventions. It does not derive
the Hilbert-space structure, does not source-produce a phase operation, and
does not create evidence: no instrument, no freeze, no evidential run.
Projector webs alone underdetermine the weight and frequencies alone do not
prove the universal valuation law; the flagship boundary stands unchanged.
INS-01 remains the controlling OL-A1 verdict; OL-C5 status is untouched;
the PR-52 physical attachments and the source-production obligation on
register row PR-04 stay open.
