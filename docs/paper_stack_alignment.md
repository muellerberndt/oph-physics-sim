# Paper Stack Alignment: Screen Computation

Updated: 2026-06-02

## Short Answer

No, not "exactly" yet.

The current simulator can already show a receipt-bearing fixed-cutoff screen computation that matches
important parts of the OPH paper stack:

- a support-visible `S2` screen chart rather than a microscopic bulk lattice;
- a `P`-scaled finite cell area;
- a federated echosahedral 12-port patch budget;
- explicit named `P0..P11` port-assignment receipts for the finite regulator;
- visible overlap mismatch `Phi`, local repair, and record commits;
- explicit round-cap geometry and the `lambda_C(2*pi*t)` BW target map;
- kinematic BW sanity receipts where the correct `2*pi` cap transport separates from
  wrong-normalization and shuffle controls;
- first state-derived cap/collar modular-probe machinery with diagonal collar Markov receipts and
  regularized finite `K_a = -log(rho_C + aI)` matrix-element comparisons.
- transition-scale selection receipts that separate declared BW automorphism sanity from
  repair/collar-derived response.

It does not yet show the full core-paper stack exactly. In particular, it does not yet implement the
support-visible weak-* / GNS scaling-limit cap pair, collar error accounting, central-record Born
surface, edge-sector heat-kernel stationary law, null modular bridge, Einstein branch, compact-gauge
branch, or Standard Model / D10 quantitative closure branch.

The current result should therefore be claimed as:

> The finite OPH-FPE screen computation now implements the regulator-side screen microphysics, a
> first BW cap-flow diagnostic, and state-derived cap/collar modular-probe receipts. It is not yet a
> completed finite demonstration of the full support-visible OPH scaling-limit theorem stack.

## Paper Claims Versus Simulator Status

| Paper-stack item | Source anchor | Current simulator status | Claim allowed now |
| --- | --- | --- | --- |
| Spherical screen is an observer-facing support chart, not literal bulk | `screen_microphysics...tex` lines 129-141; compact paper lines 4249-4268 | Implemented as a Fibonacci/cellulated `S2` chart with no initialized 3D bulk lattice | Yes: the screen computation is regulator-side and observer-facing |
| Fixed-cutoff federated patch carrier with visible restrictions and records | `screen_microphysics...tex` lines 143-175 | Implemented in simplified finite arrays and object engine | Yes, at simplified finite-carrier level |
| Echosahedral 12-port reference patch | `screen_microphysics...tex` lines 177-210 | Implemented as explicit named `P0..P11` finite port assignments over mutual-kNN routing; still not the full paper's named port algebra | Partial |
| Overlap mismatch `Phi` and repair to visible local normal form | `screen_microphysics...tex` lines 235-276 | Implemented; BW smoke runs settle to `Phi=0` | Yes for Lyapunov-style finite repair receipts; schedule independence needs more controls |
| Pixel constant `P` as local screen cell area / D10 trunk input | `observers_are_all_you_need.tex` lines 306-315 and 391-394; compact paper lines 4317-4450 | Implemented as `ratio_P`, `a_cell`, `screen_area`, `effective_radius`, `ellbar_shared`, and natural unit readout | Yes as architecture receipt; not as a D10 closure derivation |
| Edge-sector Casimir / heat-kernel handoff | `screen_microphysics...tex` lines 358-386 | `edge_sector_heat_kernel_report.json` now compares S3 conjugacy-class/Casimir sector frequencies to a declared finite heat-kernel surrogate | Yes as fixed-cutoff S3 surrogate; compact Peter-Weyl lift remains open |
| Central-record Born / Luders measurement surface | `screen_microphysics...tex` lines 342-356 | `central_record_born_report.json` now verifies finite committed-record event projectors, empirical Born probabilities, Luders idempotence, and repeat-read stability | Yes as fixed-cutoff central-record receipt |
| Observer checkpoint / restoration surface | `screen_microphysics...tex` observer checkpoint section | `observer_checkpoint_restoration_report.json` now emits finite accessible observer checkpoints with zero exact-copy future-law bound | Yes as fixed-cutoff exact-copy receipt |
| BW cap modular flow `sigma_t = alpha_lambda_C(2*pi*t)` | compact paper lines 1836-1900 | Explicit `lambda_cap` geometry and kinematic BW sanity verifier implemented; state-derived `rho_C` / `K_a` modular-probe receipt path added | Partial diagnostic, not theorem completion |
| Lorentz kinematics from `Conf+(S2) ~= PSL(2,C) ~= SO+(3,1)` | compact paper lines 1943-1957 | Cap geometry target is represented; group closure and scaling-limit cap-pair extraction are not implemented | Geometry side only |
| Null modular bridge / local Einstein branch | compact paper lines 1958 onward | Not implemented | No |
| Compact gauge / Standard Model / particle branch | compact paper later branches; particle-zoo paper | Not implemented beyond toy `S3` sectors | No |
| Early-universe `C_l` / `P(k)` observables | OPH-FPE roadmap | Not implemented as physical comparison layer | No physical cosmology claim yet |

## Current Receipts

Latest P-weighted, support-visible-regularized legacy kinematic BW sweep:

```text
runs/e1_s3_bw_screen_4k_1780293892
  patches: 4,096
  final Phi: 0
  R_BW median: 0.385094900337284
  R_BW p90: 0.470952585890742

runs/e1_s3_bw_screen_64k_1780293893
  patches: 65,536
  final Phi: 0
  R_BW median: 0.38272334680897835
  R_BW p90: 0.4696630435668558

runs/e1_s3_bw_screen_1m_1780293924
  patches: 1,048,576
  final Phi: 0
  R_BW median: 0.38353693032266634
  R_BW p90: 0.46518841336585426
```

Interpretation:

- Correct `2*pi` transport separates from controls across all three sizes.
- The p90 residual improves with refinement.
- The median residual is not strictly monotone, because the 1M point rebounds slightly from the 64k
  point.
- This is a real BW-facing kinematic receipt, but not yet a state-derived modular-transport scaling
  receipt.

The current state-derived path now writes:

```text
collar_markov_report.json
bw_state_derived_report.json
```

Those files are the next primary receipts. They build observer-visible cap/collar packet states,
emit `epsilon_cmi` / `r_FR` recovery bounds, construct a regularized finite cap state `rho_C`, and
compare `e^{itK_a} O e^{-itK_a}` against the geometric `lambda_C(2*pi*t)` target on finite
observer-visible matrix elements.

Current blocker:

```text
cooccurrence-density 4k state-derived smoke:
  state-derived BW median: 1.2101983225
  best state-derived control: wrong_1x_normalization
  correct-beats-controls: false
```

So the naive finite density-log state-derived modular operator is not theorem-aligned. It emits a
useful failure receipt: the surrogate modular generator has the wrong effective speed/structure.

The transition-response automorphism mode now instantiates the intended BW finite branch:

```text
transition-response automorphism sweep, 4k/64k/256k:
  state-derived median residuals: ~2.3e-15 to ~2.5e-15
  correct 2pi beats wrong-normalization / no-flow controls
  numerical_floor_detected: true
```

This is not a proof that the generic repair dynamics discovered BW. It is a finite branch
instantiation: the perturb/remeasure transition operator is declared with KMS/BW `2*pi`
normalization, and the finite automorphism generator is inferred from that transition. This matches
the compact paper's warning that the theorem is automorphism-level on the support-visible cap pair
and should not be reduced to a full-algebra finite density-matrix identity.

The follow-up transition-scale selector is deliberately non-circular:

```text
transition-selection 4k smoke:
  declared geometric sanity source: selects 2pi
  primary perturb/remeasure response source: selects 1x
  repair response identity fraction: 1.0
  emergence status: diagnostic_only_transition_response_degenerate

transition-selection 64k smoke:
  primary perturb/remeasure response source: selects 1x
  repair response identity fraction: 1.0
  emergence status: diagnostic_only_transition_response_degenerate
```

So the simulator can instantiate the BW branch, but it has not yet derived or selected that branch
from raw repair/collar dynamics.

The KMS/BW collar-transport fix now adds the branch-accurate mechanism explicitly:

```text
KMS collar-transport 4k smoke:
  primary source: kms_collar_transport_response
  selected scale: 2pi
  KMS source identity fraction: 0.390625
  raw perturb/remeasure source: still selects 1x

KMS collar-transport 64k smoke:
  primary source: kms_collar_transport_response
  selected scale: 2pi
  KMS source identity fraction: 0.3125
  raw perturb/remeasure source: still selects 1x

KMS collar-transport 256k smoke:
  primary source: kms_collar_transport_response
  selected scale: 2pi
  KMS source identity fraction: 0.421875
  KMS 2pi score: 0.1404455589798214
  raw perturb/remeasure source: still selects 1x
```

This fixes the finite OPH branch-instantiation mechanism: perturb/remeasure can now be followed by
KMS/BW-normalized collar transport and recover the `2pi` cap-flow scale. It is still not an
endogenous-selection proof, because the non-KMS perturb source remains a negative control. The KMS
selector score improves from 4k to 64k to 256k, which is the first useful refinement signal on this
branch-instantiated transport path.

Latest modular-lift dimension receipts:

```text
4k screen:  d ~= 2.77
64k screen: d ~= 2.79
1M screen:  d ~= 2.72
```

These are secondary visualization diagnostics. They should not be used as evidence that the OPH
Lorentz branch has landed, because the compact paper's Lorentz result is a BW cap-flow /
support-visible scaling-limit statement, not a raw point-cloud dimension statement.

## What Is Exact Today

The current test suite verifies exact finite geometry facts:

- `lambda_cap(points, C, 0)` is identity up to numerical tolerance.
- `lambda_cap(lambda_cap(points, C, s), C, -s)` returns the original points up to numerical
  tolerance.
- hard cap membership is preserved by `lambda_cap`.
- the boundary derivative matches `exp(-s)` for the implemented disk automorphism.
- `P` derives `a_cell`, `ellbar_shared`, natural unit readout, and screen area receipts.

Those are exact finite-regulator checks of the implemented screen chart. They are not the same as
proving the continuum support-visible BW theorem.

## Next Exactness Gates

To make the screen computation much closer to the core paper stack, add these gates before any
stronger claim:

1. Repair schedule controls that test quotient-local schedule independence across seeds/orders.
2. A non-circular cap/collar selection mechanism showing when raw OPH repair/collar dynamics lands
   on the BW transition-response branch rather than having that branch declared.
3. State-derived BW residual scaling over more sizes and seeds, with carried collar-width/error
   accounting.
6. A cap-local observable family closer to the paper's support-visible cap algebra, not just scalar
   record fields and first transition surrogates.
7. A group-action closure test for composed cap maps / Mobius transformations.
8. Only after state-derived BW residual scaling, observer-object consensus, and neutral
   observer-record reconstruction pass, freezeout `C_l` proxies and reconstructed bulk dimension
   should be interpreted physically.

## Recommended Public Wording

Use:

> OPH-FPE currently demonstrates the finite screen microphysics architecture, a kinematic BW
> cap-flow sanity diagnostic under controls, and the first state-derived cap/collar modular-probe
> receipt path. This is not yet a completed demonstration of the full support-visible
> Lorentz/Einstein/Standard-Model paper stack; it is a finite-regulator program for testing the
> right BW observable rather than a raw dimension artifact.

Avoid:

> The simulator already shows the screen computation behaves exactly like the full core paper stack.
