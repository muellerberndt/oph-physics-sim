# Proof-packet audits and Lean-mirrored fixtures

Moved verbatim from the repository README (2026-07-14) to keep the front
page compact. Each block documents a fail-closed audit that recomputes a
paper-side proof packet from primitive fields; producer-supplied booleans
are ignored by every one of them.

The [issue #361](https://github.com/FloatingPragma/observer-patch-holography/issues/361) continuum bridge is represented by `oph_fpe.scale.reference_tower` and
`docs/oph_issue361_certificate_schema.json`. A passing finite reference-tower identity check is
only the finite-regulator gate. Continuum correlations, BW modular convergence, Lorentzian
unitarity, and Yang-Mills identification remain closed or conditional until the emitted
certificate includes Cauchy envelopes, transported-state/cutoff bounds, positive-transfer plus
transfer-tower convergence, and the four-dimensional OS/gauge certificate.

[Issue #307](https://github.com/FloatingPragma/observer-patch-holography/issues/307) has a separate fail-closed collar-CMI audit. Run
`python3 -m oph_fpe.cli issue-307-collar-cmi-decay --source <primitive.json> --out <report.json>`.
The audit recomputes five clauses: finite-range Gibbs evidence, uniform strong conditional matrix
mixing, regional CMI in nats, the boundary-prefactored exponential bound, and the sharp scaling
margin

```text
delta / xi - log(|partial C|_UV).
```

The ratio `delta / ell_UV -> infinity` does not pass the scaling gate by itself. Caller-provided
pass flags are ignored. A passing `ISSUE_307_COLLAR_CMI_DECAY_FINITE_RECEIPT` is a finite
branch-instantiation sanity check. It does not certify the continuum limit, turn local packet CMI
into regional quantum CMI, produce a stress tensor, or open the Einstein gate.

[Issue #308](https://github.com/FloatingPragma/observer-patch-holography/issues/308) is represented by the finite cap-normal BW certificate audit:

```text
BWRec_r = (
  CapNormal_r,
  Frame_r,
  Order_r,
  Support_r,
  CrossRatio_r,
  Matrix_r,
  KMS_r,
  Error_r
)
```

Use `python3 -m oph_fpe.cli issue-308-bw-certificate --source <BWRec_r.json> --out <report.json>`
to recompute BW0-BW3 from primitive fields. A renderer cap, fitted boost, finite cap-ID
permutation, or numerical coefficient near `2*pi` is not a BW theorem receipt. BW3 requires the
primitive fields and a passing refinement error envelope; the audit ignores producer-supplied
`bw_passed` or `tier` booleans.

[Issue #309](https://github.com/FloatingPragma/observer-patch-holography/issues/309) is represented by `CAP_NORMAL_H3_CHART_RECEIPT`. It recomputes
`q(Omega)=(1,Omega)`, `n_C=(cot(alpha), csc(alpha) c)`, boundary incidence,
Lorentz equivariance `n_{gC}=Lambda_g n_C`, and the future-sheet `H3` checks
from primitive chart fields. A sampled/fitted display without a global
round-cap certificate is `CAP_NORMAL_H3_CHART_APPROXIMATE`, not theorem
evidence.

```bash
python3 -m oph_fpe.cli cap-normal-h3-chart \
  --source runs/<run_id>/cap_normal_h3_chart_source.json \
  --out runs/<run_id>/cap_normal_h3_chart_report.json
```

[Issue #310](https://github.com/FloatingPragma/observer-patch-holography/issues/310) is represented by `MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT`. The
allowed path is:

```text
record projector
-> R_i(C,t,O)
-> calibrated response vector
-> conditioned residual inverse
-> H3 localization ball
```

The audit recomputes cap-frame rank and singular values, compact-domain/net
metadata, total error, the residual minimizer, the localization radius, and
`Delta_loc`. A unique finite point requires `Delta_loc > 0`; otherwise the
state is `H3_LOCALIZATION_AMBIGUOUS`. Existing H3 point fields, viewer
coordinates, or object packets may be injected controls or ground truth for a
self-consistency test, but they cannot set `H3LOC=true` by themselves.

```bash
python3 -m oph_fpe.cli modular-response-h3-localization \
  --source runs/<run_id>/modular_response_h3_localization_source.json \
  --out runs/<run_id>/modular_response_h3_localization_report.json
```

## Lean-mirrored consensus fixture

Lean-mirrored consensus fixture:

```bash
python3 -m oph_fpe.cli rule90-consensus-fixture \
  --out runs/rule90_consensus_fixture_report.json
```

This writes an exact finite Rule-90 carrier receipt mirroring the Lean consensus
fixtures: good boundary-fiber uniqueness, bad-boundary failure, nontrivial
gauge equivalence, and the local H1-H2-H3 repair no-go witness. It is a
regression contract for simulator receipt logic, not a physical prediction and
not a runtime dependency on Lean.

## Matscheko proof-chain import gate

Matscheko proof-chain import gate:

```bash
python3 -m oph_fpe.cli matscheko-proof-chain-import \
  --out runs/matscheko_proof_chain_import_report.json
```

This writes a compact receipt for the additional finite audit imports: finite
modular flow/KMS, the scalar channel bridge, twelve-port surface bookkeeping,
the QBFT quorum caveat, the two-P provenance, the chi-nu G9/G10 gates, and the
v10 Rule-90 theorems T38--T41 (parity splitting, power-of-two adjacent-pair
universality, and sharp lightlike-diagonal screens). The Rule-90 entries remain
finite binary audit-fixture status imports, not microscopic physics or
spacetime-causality claims. G9 is false when no record-DeltaS to gravity-DeltaS
calibration is supplied.
