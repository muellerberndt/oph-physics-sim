# What the free routing comparison proves

The Readout tab has two different programs. Its left view replays stored
load and centre-flux records. Its right view prepares finite free Dirac
packets using those record IDs. The packet Hamiltonian, spinor, detector,
clock and spatial grid are declared. Reusing an ID preserves provenance;
it is not an operator transfer from the interacting load history.

In the routing comparison, set
`H = alpha.p + theta beta`, where the four matrices anticommute and square
to identity. Two alternating ticks in the half-mass frame form the
symmetric product

```
M(theta/2) S1 S2 S3 M(theta) S3 S2 S1 M(theta/2),
Si=exp(-i p_i alpha_i),  M(u)=exp(-i u beta).
```

The order may be reversed without changing the following bound. The
linear and quadratic Taylor coefficients equal those of `exp(-2iH)`.
Put `S=2(|theta|+|p1|+|p2|+|p3|)`. Multinomial expansion bounds each
higher product coefficient by `S^n/n!`. Thus

```
||W - exp(-2iH)|| <= 2 sum_(n>=3) S^n/n!
                 <= min(2, exp(S) S^3/3).
```

The minimum also uses that both operators are unitary. Over n pairs of
ticks their difference is at most n times this bound. If momenta and mass
angle scale together with a spacing epsilon, n is proportional to
1/epsilon for fixed physical time and the error is O(epsilon²). Shrinking
momentum alone while holding the mass angle fixed does not give this
same cubic one-pair estimate in momentum. The executable independently
checks both scalings on the actual routing matrices.

`readout_check.mjs` also compares one long-wavelength packet over 24 ticks.
A fidelity near 0.997 is a finite numerical control for that input, not an
exact equality, a uniform theorem for every packet, or an interacting
continuum construction. The displayed grid retains a finite momentum menu;
its free Fourier evolution is analytic up to floating-point arithmetic.
The menu's radial edge is sqrt(3) times its one-axis edge.

The Dirac energy is `E²=|p|²+theta²` in these units. The real-space packet
and Fourier momentum probabilities come from the same state. Complete
Gaussian records use the [retained detector instrument](readout_instrument.md).
There is no claim that a centre class determines spin, full gauge weights,
fermion generations or a measured rest mass. The viewer's conversion to
MeV uses its supplied electron-based time scale.

Run `node readout_limit_check.mjs` from this directory to replay the
matrix and scaling controls. The bound itself follows from the stated
finite Taylor identities and unitary norm inequality.
