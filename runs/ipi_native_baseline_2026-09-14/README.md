# Twelve-port execution diagnostics

This package exposes two bounded software executions relevant to a question
about a native equation of state and displacement current. Neither execution
measures a native substrate equation of state. A JSON `null` denotes an
undefined or unprovided observable, never a measured zero.

OPH models bounded observer-like self-reading systems: local state, ports,
readback, retained records and feedback or repair moves. The native diagnostic
below isolates one scalar repair primitive and records it from outside. The
separate Maxwell instrument executes record-conditioned feedback. These are
different programs with different supplied assumptions.

## Native scalar repair

`run_native.py` calls the existing committed implementation in
`oph_fpe/dynamics/canonical_seam_repair.py`. One isolated carrier has 12 scalar
port readings and 30 seams. On a selected seam, readings a and b become
(a+b)/2 at both endpoints. All updates and recorded state values use exact
rational arithmetic. The existing carrier topology builder uses numerical
geometric construction; the independent verifier reconstructs its incidence
with exact golden-ratio arithmetic.

The declared diagnostic schedule visits each seam once in each of ten shuffled
sweeps: exactly 300 attempts, with no outcome-based stopping. This is a
reproducible schedule choice, not a physical clock or an IID sequence of seams.
The pulse input is (12,0,...,0), and the constant control is (1,...,1).

| Case | Sum of readings, initial and final | Initial sum of squares | Final sum of squares | Final centered sum of squares |
| --- | --- | --- | --- | --- |
| Pulse, schedule seed 7 | 12 exactly | 144 | 12.000000000000105 | 1.0431058872423822e-13 |
| Pulse, schedule seed 11 | 12 exactly | 144 | 12.000000000000016 | 1.5396467015853833e-14 |
| Constant input, repair enabled | 12 exactly | 12 | 12 exactly | 0 exactly |
| Pulse, repair disabled | 12 exactly | 144 | 144 exactly | 132 exactly |

Decimals summarize the authoritative fractions in `native_trace.json`.
Every repair preserves sum(x) and changes sum(x^2) by exactly -(a-b)^2/2.
The centered sum of squares is sum(x^2)-sum(x)^2/12. The uncentered sum of
squares is a descent functional, not identified here with physical energy.
The separate sum of squared seam differences can increase on an individual
update; it is retained in the trace and is not asserted to be monotone.

No thermodynamic stress, volume/work law, or identification of energy density
is supplied by this primitive. It therefore provides no numerical pressure,
temperature or w=p/rho, even in dimensionless model units. Turning off repair
leaves the initial scalar state unchanged; it does not expose an underlying
streaming/collision gas. This small diagnostic is not a complete self-reading
federation or a claim of finite exact convergence under every schedule.

`verify_native.py` independently reconstructs incidence and replays every
update without simulator imports. `native_verification.json` records its
checks and deliberate corruptions it rejects.

## Supplied Maxwell instrument

`em/run_em.py` calls the existing RER producer
`code/electromagnetism/serial_maxwell_readout.py` and its independent verifier.
The executable uses local scalar slots, readback, retained records and feedback;
the third potential slice is advanced from decoded records. The Maxwell action,
field typing, h=1/2, two initial slices and charged paths are supplied inputs.
This is not Maxwell dynamics derived from the bare scalar averaging operation.

The bundle contains two gauge-related executions, each with 585 events and
180 probe/feedback cycles. Exact Ampere, Faraday, Gauss and continuity checks
pass. In the supplied dimensionless field convention, seam 1 has
J=0 and (E1-E0)/h = curl(B1) = -5024/1149, with zero Ampere residual.
This is a nonzero displacement term in the supplied Maxwell model.

The recorded staggered field form is
H[n] = (|E[n]|^2 + <B[n],B[n+1]>)/2. Its values are
H0=93833/2298 and H1=40915/1149; the change equals the supplied source work
-4001/766 exactly. These are algebraic quantities of that action, not
measurements of the native repair substrate's thermodynamic energy or pressure.
No temperature, pressure, energy density or equation-of-state result is given.

`em/raw_execution.json` retains both complete event logs. `em/summary.json`
contains fields, assumptions and commits. `em/verification.json` records the
independent replay and rejection of a deliberately changed electric readout.

## Reproduction

Requirements: Python 3.11 or newer, the simulator's declared dependencies and
SymPy. This package contains diagnostic drivers and outputs; the existing
producer and EM verifier sources are supplied by these repository checkouts:

- https://github.com/muellerberndt/oph-physics-sim at
  `0cdc4e5ad693422f3f710e79f4513b25c136fc67`
- https://github.com/FloatingPragma/observer-patch-holography at
  `8281894cbbd712cc5c1ab65bb3015958565d7997`

Source-file hashes are recorded in the outputs. The consumed native and EM
provider files match their recorded commits, despite unrelated working-tree
changes elsewhere. Python 3.13.7, NumPy 2.3.5, SciPy 1.16.3 and SymPy 1.14.0
were used for this bundle; the exact Python version matters for replaying the
random shuffle from its seed. The complete seam sequence is also in the trace.

Use a virtual environment and install the simulator and SymPy:

```bash
python3 -m pip install -e /path/to/oph-physics-sim sympy
```

From the extracted package directory:

```bash
python3 -B run_native.py --sim-root /path/to/oph-physics-sim
python3 -B verify_native.py --sim-root /path/to/oph-physics-sim
python3 -B em/run_em.py --sim-root /path/to/oph-physics-sim --rer-root /path/to/reverse-engineering-reality
```

The drivers write only this diagnostic package by default. They do not replace
canonical scientific receipts. `SHA256SUMS.json` binds the packaged files;
rerunning a driver with another environment may legitimately change provenance
and therefore output hashes.
