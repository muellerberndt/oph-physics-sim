# Physical W/Z source-to-pole requirements coverage

Date: 2026-07-20

Normative/audit input (read-only):
`../../survival-proof-3/simulator/UPSTREAM_MATH_SIMULATOR_REQUIREMENTS.md`

Input SHA-256:
`7ec07e225bb1223616040094162727e323b1320aaba74337603c054025f29f61`

## Verdict

The updated material is useful and materially changes the simulator boundary.
It supplies a detailed draft sufficiency contract for an imported or OPH-native
W/Z source-to-pole computation. It does not supply the primitive action,
renormalized source packet, independent QFT engines, physical amplitudes, or
operational clock.

The existing `oph_fpe.bosons` backend is therefore classified as `WZH0`:

```text
synthetic polynomial/block controls
+ affine RG numerical controls
+ finite matrix-gap/unit-conversion controls
+ sampled polynomial zero controls
= diagnostic only, unconditionally nonpromoting
```

All legacy helper and aggregate `promotion_allowed` fields are now hard-false.
Caller identity flags, `promotion_allowed` sidecars, and hash-shaped strings
cannot produce a BRST block, matching, source-clock, or physical-pole receipt.

## Claim lanes

Exactly one lane must be selected:

| Lane | Licensed output | Current status |
|---|---|---|
| `OPH_CHART_ONLY` | chart or WZH0 synthetic controls | available, diagnostic only |
| `EXTERNAL_SM_EFT_VALIDATION` | imported-SM QFT/pole validation | contract specified; engines and proof envelope absent |
| `OPH_NATIVE_DIMENSIONLESS` | source-derived dimensionless W/Z poles | blocked on native action/parameters/matching/law and QFT parents |
| `OPH_NATIVE_PHYSICAL` | dimensionful operational W/Z poles | additionally blocked on source clock and uncertainty freeze |
| `TARGET_COMPARISON_ONLY` | read-only post-processing | must never write source artifacts |

The already exposed W/Z values make a current native calculation
`post_exposure_validation`; changing the metadata to “prospective” is not a
scientific gate.

## Exact external gate

The imported field-theory validation lane requires the conjunction:

```text
IMPORTED_MINKOWSKI_CHART
and IMPORTED_SM_EFT_ACTION
and ANOMALY_AND_PERTURBATIVE_BRST_1
and FULL_YUKAWA_1
and EFT_MATCHING_1
and RULE_EQUIVALENCE_1
and RENORMALIZATION_ST_1
and FJ_DIRECT_1
and FJ_CONVERTED_1
and FJ_EQUIVALENCE_1
and GENERAL_GAUGE_BRST_1
and WARD_ST_NIELSEN_1
and POLE_SERIES_1
and ANALYTIC_CONTINUATION_1
and PHYSICAL_CURRENT_POLE_W_1
and PHYSICAL_CURRENT_POLE_Z_1
and RUNTIME_SUBJECT_BINDING_1
and TARGET_FIREWALL_1
```

H is not a hidden prerequisite for a W/Z-only claim. A joint WZH claim may add
its own H gate.

## Native additions

The dimensionless OPH-native lane must replace every imported parent with the
same-root native parents and additionally require:

```text
EVENT_GEOMETRY_OR_EXPLICIT_NATIVE_LOCAL_CHART
OPH_SM_EFT_ACTION_1
OPH_SOURCE_PARAMETER_JET_1
OPH_SOURCE_TO_SM_MATCHING_1
FULL_YUKAWA_1
SOURCE_LAW_1
SOURCE_COVARIANCE_1
NO_TARGET_ANCESTRY_1
```

Its outputs are only `s_W/E_star^2` and `s_Z/E_star^2`. Physical units require
`SOURCE_CLOCK_1` and `POLE_UNCERTAINTY_FREEZE_1`.

The BW `2pi` clock, A5 oriented-volume clock, and W/Z operational source clock
are distinct typed objects. None substitutes for another without an explicit
same-source intertwiner.

## Production evidence envelope

No producer may self-promote. The P0 common envelope now binds and resolves:

- schema ID/version and actual schema bytes/hash;
- artifact ID, exclusive claim lane/scope, branch and freeze IDs;
- source-root hash and exact canonical runtime subject/digest;
- producer source tree, executable and environment-lock bytes/hashes;
- independently authored checker source tree, executable and independence
  evidence;
- every parent receipt ID, file hash, subject digest and output digest;
- common action, field-census, scheme, FJ, term-mask, sheet and units hashes;
- numerical precision/rounding, output digest, terminal status and blockers;
  and
- target-firewall and exposure metadata.

Every referenced hash must resolve inside an immutable contained bundle. All
parents must share the same root, branch, freeze, subject conventions and
order mask. Mixed-root splicing is a hard failure. Passing producer booleans
are retained only as ignored/unverified claims. The generic resolver remains
P0 inventory infrastructure: with no registered independent scientific
checker, its scientific-replay and promotion receipts are always false.

`oph_fpe.bosons.physical_wz_requirements` now replays that on-disk bundle and
serializes the five exclusive lanes and four species scopes. A mapping, stored
report, WZH0 diagnostic, visualization object, or `DEMO_ASSUMPTION` cannot be
used as evidence.

The production terminal classes are `PASS`, `OPEN`, `UNRESOLVED`, `FAIL`, and
explicitly optional `NOT_APPLICABLE`. `OPEN` and `UNRESOLVED` never promote.

## Mathematical/engine boundary

| Surface | Finite implementation can do | Still missing |
|---|---|---|
| invariant action | canonicalize and type-check an imported AST | OPH theorem selecting it and its coefficients |
| flavor | verify complete `Yu,Yd,Ye`, SVDs and CKM | source-derived full matrices |
| matching | verify census-derived beta functions and threshold maps | native parameter root and full D10-style matching |
| FJ | compare direct and fully converted engines at fixed order | two independent production engines |
| renormalization | generate/check one-loop counterterms and ST restoration | implementation and complete proof artifacts |
| BRST/gauge | check Ward/ST/Nielsen identities and frozen stress grid | analytic kernels and independent diagram/rule universe |
| poles | certify a simple zero on a continuous contour | complex-ball kernels, sheet replay, Laurent data and current amplitudes |
| source law | propagate a supplied deterministic/stochastic law | target-independent native law or unique selector |
| clock | verify a supplied operational transition packet | source selection and same-root physical calibration |

## Immediate implementation order

1. `P0`: production-envelope schema, immutable resolver, runtime subject
   binding, mixed-root rejection, exclusive lanes, target process isolation,
   diagnostic/passing separation, and clean-room/mutation tests — implemented
   as a nonpromoting inventory/replay layer.
2. `P1`: imported SM action, census, Yukawa/CKM, matching, anomaly/beta and
   strict-one-loop order-mask packets.
3. `P2`: independent rules, diagrams, counterterms, direct/converted FJ, and
   complex integral engines.
4. `P3`: independent BRST/ST/Ward/Nielsen checker, analytic continuation,
   interval argument count, Laurent denominator and physical-current poles.
5. `P4`: replace imported parents with source-produced OPH action, parameters,
   matching, law/covariance and operational clock.

Only P0 safety hardening and the nonpromoting stage/envelope skeleton belong in
the present architecture phase. P1-P3 are substantial independent QFT
engineering; P4 additionally needs new OPH source mathematics.

## Current receipt status

```text
WZH0_SYNTHETIC_CONTROL = available
WZH_PRODUCTION_ENVELOPE_INVENTORY_REPLAY = implemented
WZH_INDEPENDENT_SCIENTIFIC_REPLAY = false
EXTERNAL_SM_EFT_VALIDATION = false
OPH_NATIVE_DIMENSIONLESS = false
OPH_NATIVE_PHYSICAL = false
PHYSICAL_CURRENT_POLE_W = false
PHYSICAL_CURRENT_POLE_Z = false
SOURCE_CLOCK_1 = false
```

This document is an implementation map, not a physical W/Z receipt.
