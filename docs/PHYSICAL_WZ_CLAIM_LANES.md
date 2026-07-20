# Physical W/Z claim lanes

This document is the simulator-side claim contract for the W/Z source-to-pole
stack. It implements the lane and gate rules in
`survival-proof-3/simulator/UPSTREAM_MATH_SIMULATOR_REQUIREMENTS.md`; it does
not supply the missing QFT or OPH source calculations.

## Exclusive lanes

Every requirements audit selects exactly one lane:

| Lane | Licensed result | Never licensed |
|---|---|---|
| `OPH_CHART_ONLY` | Source/chart or WZH0 diagnostic coordinates | A resonance pole |
| `EXTERNAL_SM_EFT_VALIDATION` | Imported-SM QFT and pole validation | An OPH-native derivation |
| `OPH_NATIVE_DIMENSIONLESS` | Native `s_W/E_star^2` and/or `s_Z/E_star^2` | GeV masses or widths |
| `OPH_NATIVE_PHYSICAL` | Native physical-unit pole validation | A prospective prediction for already exposed W/Z targets |
| `TARGET_COMPARISON_ONLY` | Immutable post-processing comparison | Any new source or pole promotion |

The exposure classification is fixed to `post_exposure_validation`. It is not
an input switch. Target comparison cannot mutate or promote the source lane,
and success in the external lane never flips either native lane.

The claim scope is exactly one of `W_ONLY`, `Z_ONLY`, `WZ`, or `WZH`. The W and
Z physical-current pole gates are independent. A W receipt cannot close Z and
vice versa. The H pole is required only for `WZH`; it is `NOT_APPLICABLE` for
the other three scopes.

## External conjunction

The external common gate is the conjunction of:

- `IMPORTED_MINKOWSKI_CHART`
- `IMPORTED_SM_EFT_ACTION`
- `ANOMALY_AND_PERTURBATIVE_BRST_1`
- `FULL_YUKAWA_1`
- `EFT_MATCHING_1`
- `RULE_EQUIVALENCE_1`
- `RENORMALIZATION_ST_1`
- `FJ_DIRECT_1`
- `FJ_CONVERTED_1`
- `FJ_EQUIVALENCE_1`
- `GENERAL_GAUGE_BRST_1`
- `WARD_ST_NIELSEN_1`
- `POLE_SERIES_1`
- `ANALYTIC_CONTINUATION_1`
- `RUNTIME_SUBJECT_BINDING_1`
- `TARGET_FIREWALL_1`

The scope then adds `PHYSICAL_CURRENT_POLE_W_1`,
`PHYSICAL_CURRENT_POLE_Z_1`, or both. `WZH` additionally adds
`PHYSICAL_SCALAR_POLE_H_1`. This is the Section 2.1 external conjunction with
the species gates split so a W-only or Z-only audit remains meaningful.

## Native conjunctions

The native dimensionless common gate replaces imported chart/action/matching
parents with:

- `EVENT_GEOMETRY_OR_EXPLICIT_NATIVE_LOCAL_CHART`
- `OPH_SM_EFT_ACTION_1`
- `OPH_SOURCE_PARAMETER_JET_1`
- `OPH_SOURCE_TO_SM_MATCHING_1`
- `SOURCE_LAW_1`
- `SOURCE_COVARIANCE_1`
- `NO_TARGET_ANCESTRY_1`

It retains the source-independent QFT checks from the external conjunction:
`ANOMALY_AND_PERTURBATIVE_BRST_1`, `FULL_YUKAWA_1`,
`RULE_EQUIVALENCE_1`, `RENORMALIZATION_ST_1`, the two FJ engines and their
equivalence, general-gauge BRST, Ward/ST/Nielsen, strict pole series, analytic
continuation, runtime subject binding, target firewall, and the requested
species pole gates. Every retained receipt must be regenerated or wrapped with
the native parent family; an imported-lane receipt is not reusable merely
because its numerical payload agrees.

`OPH_NATIVE_PHYSICAL` adds exactly:

- `SOURCE_CLOCK_1`
- `POLE_UNCERTAINTY_FREEZE_1`

The dimensionless outputs remain distinct from physical-unit outputs.

## Evidence and status rules

Gate evidence is supplied only as an on-disk production-envelope verification
artifact. The requirements aggregator re-runs the production-envelope
verifier and reads its recomputed stage, lane, scope, subject, family, checker,
and status fields. A mapping containing `passed: true`, a hash-shaped string,
or a diagnostic envelope is not evidence.

All relevant envelopes must agree exactly on branch, freeze, source root,
action, scheme, FJ convention, term mask, analytic sheet, units basis, and
runtime subject family. Wrong-stage, wrong-lane, wrong-scope, mixed-root, and
cross-freeze artifacts fail closed.  Unknown stages, nonliteral scientific
flags, inventory-only evidence rows, rows with blockers, and any
`DEMO`/`FORCED`/`FROZEN_TARGET`/`SYNTHETIC` label are nonpromoting.  An
integrity blocker zeros every scientific receipt even when an otherwise valid
subset is present, and top-level promotion requires the complete audit report
itself to be `PASS`.

Statuses have only these meanings:

- `PASS`: the strict production verifier replayed a passing envelope;
- `OPEN`: required evidence was not supplied;
- `UNRESOLVED`: evidence is diagnostic or the upstream source selection is
  explicitly non-identifiable;
- `FAIL`: supplied evidence is invalid, mismatched, or a prerequisite failed;
- `NOT_APPLICABLE`: the gate is outside the selected lane or scope.

Only literal `status == PASS` satisfies a gate. In particular `OPEN`,
`UNRESOLVED`, and `NOT_APPLICABLE` are never truthy substitutes. Any required
`FAIL` dominates later successes, so a pole cannot mask a failed action,
matching, BRST, or source prerequisite.

## WZH0 boundary

The existing polynomial blocks, sampled contour test, affine RG transport,
generic matrix gap, caller booleans, and trusted sidecars remain WZH0
synthetic controls. The aggregator may inventory them as diagnostics, but
`WZH0` is not a claim lane and cannot satisfy any production gate. Empty input
therefore leaves all physical promotions false.

A forced end-to-end rendering must use a profile containing
`DEMO_ASSUMPTION` or `VISUALIZATION_ONLY`. It must be visibly watermarked by
the rendering surface. The requirements aggregator classifies that envelope
as `UNRESOLVED`, and both scientific promotion and scale authorization remain
false. A demo is useful for testing plumbing; it is not a substitute for any
gate in the conjunctions above.
