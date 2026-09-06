# Physical-model queries

This package connects a byte-pinned RER charged-field execution to queries
for observer records, local fields and action energy. Its five bounded
software patches have local registers, ring ports, destructive readback,
retained responses and feedback. Field reconstruction consumes their joint
decoded records. Their labels are computational coordinates.

Run from the simulator checkout with the sibling RER checkout present:

```sh
python3 -m oph_fpe.physical_systems describe
python3 -m oph_fpe.physical_systems observer 2 40
python3 -m oph_fpe.physical_systems field 40 0 .25 .25 .25 .25
python3 -m oph_fpe.physical_systems energy 40
python3 -m oph_fpe.physical_systems clock 80
python3 -m oph_fpe.physical_systems quantum
python3 -m oph_fpe.physical_systems free-state --time 1
python3 -m oph_fpe.physical_systems bridges
python3 -m oph_fpe.physical_systems packet --sample 1 --width 1/2
python3 -m oph_fpe.physical_systems packet-overlap .3
python3 -m oph_fpe.physical_systems magnetic-continuum --refinement 4 --steps 64
python3 -m oph_fpe.physical_systems magnetic-field 1 2 3
python3 -m oph_fpe.physical_systems charged-enclosure 1/80
```

`--rer-root PATH` before the query selects another checkout. Loading requires
the exact receipt and source bytes pinned to RER `c711eef1`; unrelated later
commits are allowed. It replays the authenticated event history, evaluates
the imported dressed action at requested barycentric positions, and checks
cell energy integrals against the recorded totals. It does not rerun the ODE
or interpolate between recorded times. Every result carries model units and
its source/interpretation boundary. `mesh`, `event`, `real_continuum` and the
Python methods on `WhitneySystem` expose the remaining interface.

`FreeScalar(system.mesh())` constructs all finite-element mass and stiffness
modes of the declared **e=g=0** model. For `u=sqrt(2) Re(Psi)`, this is an
independent real tensor factor of the decoupled complex scalar. With supplied
canonical quantization, its state space is `L2(R^N,dQ)`, equivalently bosonic
Fock space over all N modes. A `prepare(field, velocity)` call specifies a
coherent state. Its `state(t)`, `field(t, cell, barycentric)` and
`smeared(t, nodal_test_function)` methods compute time evolution, quantum
variances and energy without an occupation cutoff. The CLI preparation uses
a declared Gaussian-shaped nodal mean and vacuum covariance.

`two_point(t, cell, barycentric, s, other_cell, other_barycentric)` queries
the ordered field correlation at two positions and times.
`smeared_two_point(t, f, s, g)` uses two independently chosen real nodal test
functions. Both return the full and connected Wightman functions, the
commutator and half the connected anticommutator. Complex results use
`[real, imaginary]`. These are finite-mesh correlations; continuum
microcausality and a physical detector response are not established.

Normal-ordered energy subtracts the finite-mesh vacuum contribution;
pointwise vacuum variances and zero-point energy depend on the spatial
cutoff. Numerical eigenpair defects are checked. These calculations carry
no certified continuum error or measured-system comparison. They use a
different model/preparation from the interacting quantum trial returned by
`quantum`. The latter retains its own short error-controlled time interval.
The equations, geometry, couplings, quantization and physical-unit conversion
are supplied inputs, and computational patch histories do not constitute
laboratory observer placement. The free-field formulas follow the standard
[mode quantization](https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S2.html).

`ControlledBridges.load()` in `physical_systems.bridges` adds separately pinned
research packages. It authenticates all consumed receipt and source bytes,
including the parent receipts, without executing research code or rerunning
their verifiers. Its local Gaussian checks reproduce the projection and moment
identities. Every query returns an independent copy with source hashes.

`packet(sample, width)` returns all 56 Coulomb coordinates, the correctly
recharted velocity and cotangent, and full-rank seed position/momentum
covariances. The widths are `"1/4"`, `"1/2"` and `"1"`; samples 0 and 1 label
**separate neutral state preparations** at two recorded classical centers.
Circle projection changes scalar moments, so the seed covariance is not the
neutral state's covariance. `packet_half_density(point, sample, width)`
evaluates the normalized neutral state at a 56-coordinate point in the
Lebesgue half-density representation. It reports a numerical circle-quadrature
comparison, without a certified pointwise error. The physical Hilbert measure
and curved kinetic operator retain the supplied metric. These queries neither
propagate the interacting state nor certify its requested time interval.

`magnetic_continuum(refinement, steps=...)` exposes the conditional complex
scalar convergence theorem, refinement diagnostics and separate finite pulse
outputs. `refinement` selects spatial approximation checks; `steps` selects
the recorded pulse integration on its fixed `n=2` mesh. The pulse's nodal
initialization is distinct from the theorem's magnetic Ritz hypothesis.
`magnetic_field(position)` evaluates the prescribed background
`A=c+B cross x/2`; matter backreaction and full Maxwell evolution are absent.
The theorem supplies no numerical value of its continuum error constant.
These are mathematical model readouts, not measured systems or a common
quantum/classical observer history.

`charged_enclosure("1/80")` evaluates the authenticated classical polynomial
with exact rational arithmetic and returns exact rational coordinate bounds,
using its certified uniform error of `1e-20` throughout model times `[0,2]`.
Strings, integers and `Fraction` inputs specify time; floats are rejected.
The query imports the research interval certificate and authenticates its
bytes; it does not repeat the proof. Its nine volume-normalized canonical
coordinates differ from the packet's Coulomb coordinates, and the result
contains no join to raw observer states or certified nonlinear field error.

Reproduce the focused independent analytic, matrix-exponential, quadrature
and byte-tampering checks with:

```sh
python3 -m pytest -q tests/test_physical_systems_whitney.py tests/test_physical_systems_free_scalar.py tests/test_physical_systems_bridges.py
```

The charged tests require the pinned sibling research bytes; the free-mode
tests run on an independent tetrahedral fixture without that checkout.
