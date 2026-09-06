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

Reproduce the focused independent analytic, matrix-exponential, quadrature
and byte-tampering checks with:

```sh
python3 -m pytest -q tests/test_physical_systems_whitney.py tests/test_physical_systems_free_scalar.py
```

The charged tests require the pinned sibling research bytes; the free-mode
tests run on an independent tetrahedral fixture without that checkout.
