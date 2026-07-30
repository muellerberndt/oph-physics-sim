# Source selector decision tests

These two executable experiments ask whether finite screen structure selects
an angular transfer or a gauge-like response ray. They use no observational
targets. Their JSON receipts keep every physical-prediction gate false.

## Angular transfer

Run:

```bash
python3 -m oph_fpe.cosmology.angular_transfer_decision \
  --output runs/source_selector_decision/angular_transfer_decision.json
```

The experiment reconstructs the twelve icosahedral ports, checks all sixty
proper \(A_5\) rotations, and reproduces the exact equal-port moments

\[
I_6=\frac{11}{25},\qquad
I_{10}=\frac{247}{1875},\qquad
I_{12}=\frac{1071}{3125}.
\]

It then constructs a continuous family of smooth, linear,
\(A_5\)-equivariant right inverses from port samples to sphere fields. Every
member reproduces the same twelve samples and preserves the same spherical
mean, while its degree-six and degree-ten content changes. The receipt verdict
is:

```text
NONIDENTIFIABLE_WITHOUT_DYNAMICAL_TRANSFER_SELECTOR
```

This is a static source-side nonidentifiability result. It does not say that a
repair or readback dynamics cannot select one transfer. It says that port
geometry, smoothness, equivariance, sample reproduction, and mean preservation
do not select one by themselves.

## Finite response selector

Run:

```bash
python3 -m oph_fpe.gauge.kinetic_selector_sweep \
  --output runs/source_selector_decision/kinetic_selector_sweep.json
```

The frozen grammar contains positive quadratic response laws constructed from
the twelve-port graph Laplacian. In particular,

\[
H_1=I+L,\qquad H_2=I+2L
\]

are both genuinely nearest-neighbour and commute with all sixty proper
icosahedral actions. Their four-band response rays remain different after one
common scale is removed. Radius-two controls also give different grouped rays
while passing the explicitly scoped direct-isometric block-isotropy test.
The receipt status is:

```text
FINITE_SOURCE_FILTERS_DO_NOT_SELECT_A_UNIQUE_PORT_RESPONSE_RAY
```

The tested filters are positivity, finite-carrier locality, and \(A_5\)
covariance. They are not an exhaustive grammar of A1--A3 dynamics. The band
responses are not continuum gauge couplings, and the finite block-isotropy
check is not a Ward identity. The exact one-generation representation trace
ratio \(5/3:1:1\) is reported separately as conditional arithmetic and is not
used to choose a response law.

## Verification

Both reports carry payload hashes and recomputing verifiers. Their mutation
tests reject changed numerical content and any attempt to promote a physical
prediction flag.

Run the focused suite with:

```bash
python3 -m pytest -q \
  tests/test_angular_transfer_decision.py \
  tests/test_gauge_kinetic_selector_sweep.py
```

The experiments establish that the static finite filters are insufficient.
A positive prediction requires a source-derived dynamics that selects one
transfer or one response law, followed by a separately justified map to a
physical observable.

The [canonical repair-law candidate](CANONICAL_REPAIR_LAW.md) fixes the
one-step scalar seam operator inside the narrower linear, seam-local,
completed-reconciliation grammar. Uniform scheduling then gives the finite
graph-Laplacian ray up to a common clock scale. No bridge identifies that
working-reading channel with the response matrices tested above. The
sphere-transfer refinement theorem and the physical current-to-field
attachment remain open.
