# Complete finite Dirac detector

`readout_instrument.js` defines a complete measurement on the finite carrier
used by `readout_field.js`. It retains the free evolution, grid, mass, spinor
convention and positive-energy projector. A second recorded apparatus outcome
converts a negative-band component created by localization back into the
retained carrier. This conversion is part of the detector operation and is
distinct from free Dirac propagation.

## Audit of the field reset

The removed `ContinuumField.record()` implementation sampled a point with probability
proportional to its prior density, multiplies the whole state by a Gaussian
centered there, projects onto positive energy, and normalizes. These choices
do not form one Born instrument. The Gaussian operation has a convolution
probability, which differs from the density at its center. Positive projection
also has an input-dependent probability which that sampling omits. Returning
the window's prior weight does not repair either mismatch. A normalized state
after an operation does not establish completeness of its event probabilities.

The base field class now rejects record requests. The viewer and its detector
checks use `BornReadoutField`, whose complete instrument is defined below.

## Carrier and exact measurement operators

Let the grid be the periodic group \(\mathbb Z_n^3\), with four complex
spinor components at each point and inner product
\(\langle\psi,\psi\rangle=\sum_x\|\psi(x)\|^2\).
The forward Fourier transform in `fft3.js` is unnormalized; consequently
the same norm is \(N^{-1}\sum_k\|\widehat\psi(k)\|^2\), where
\(N=n^3\).

At each nonzero energy,

\[
 H(k)=X_{\rm record}\otimes\sigma\cdot k
       +mY_{\rm record}\otimes I,
 \qquad E(k)=\sqrt{k^2+m^2},\qquad
 \Pi(k)=\tfrac12(I+H(k)/E(k)).
\]

The source retains the full zero-energy fiber, with \(\Pi=I\), rather
than discarding two of its four components. The detector uses the identical
`projectPlus` convention, including its existing \(E<10^{-12}\) numerical
zero threshold. Strict positive-energy statements apply above that threshold;
the full retained numerical fiber is covered by the completeness statement.
Input states must lie in this retained carrier. The code checks their rejected
norm before mutating the source.

For width \(\sigma>0\), define

\[
 g_j(x)=\exp\left[-\sum_a d_n(x_a,j_a)^2/(4\sigma^2)\right],
 \qquad C=\sum_jg_j(x)^2,
 \qquad M_j=g_j/\sqrt C.
\]

Periodic translation makes \(C\) independent of \(x\). It is the cube
of the one-dimensional Gaussian-weight sum. Hence
\(\sum_jM_j^\dagger M_j=I\) on the complete finite spinor space.

Set \(B=Z_{\rm record}\otimes I\). The actual matrices obey
\(B^\dagger B=I\) and \(BH=-HB\). Therefore
\(B(I-\Pi)\) maps into the positive-energy carrier; the rejected operator
is zero on the retained zero-energy fiber. The two operators at each center are

\[
 K_{j,+}=\Pi M_j\Pi,\qquad
 K_{j,c}=B(I-\Pi)M_j\Pi.
\]

Both outputs belong to the retained carrier. Their effects satisfy

\[
 \sum_{j,b\in\{+,c\}}K_{j,b}^\dagger K_{j,b}
 =\Pi\sum_jM_j^\dagger[\Pi+(I-\Pi)]M_j\Pi
 =\Pi.
\]

This is the identity on the declared source space. Thus the detector is a
trace-preserving quantum instrument when all center and band records are
retained. No branch is silently retried or discarded. The isometry
\(\psi\mapsto\sum_{j,b}K_{j,b}\psi\otimes|j,b\rangle\) supplies a
finite apparatus realization and admits a unitary extension. The apparatus
operation is declared independently of the free source Hamiltonian.

## Sampling and state update

For a normalized retained state, the center probability and conditional
band probabilities are

\[
 p(j)=\frac1C\sum_xg_j(x)^2\|\psi(x)\|^2,
 \qquad
 p(+\mid j)=\frac{\|\Pi g_j\psi\|^2}{\|g_j\psi\|^2},
 \qquad
 p(c\mid j)=\frac{\|(I-\Pi)g_j\psi\|^2}{\|g_j\psi\|^2}.
\]

The returned `probability` is the original joint probability
\(p(j,b)=p(j)p(b\mid j)\). The returned state is
\(K_{j,b}\psi/\sqrt{p(j,b)}\). Its normalization uses the measured
branch norm with the Fourier factor \(1/N\); it does not alter the event
probability.

The center distribution is sampled in linear time: draw a latent point from
the density, then add three independent periodic Gaussian offsets. Marginalizing
the latent point gives exactly the convolution above. The latent point is an
internal classical sampling variable, with no quantum projection, no retained
apparatus record and no effect on the conditional state. Only the resulting
center selects the Gaussian Kraus operator.

`BornReadoutField.frame` uses the caller's random generator for all five draws:
the density point, three offsets and the band. It preserves reproducibility
for a seeded generator. Direct `record(...,u)` calls accept one uniform number;
each inverse-CDF selection passes its conditional residual to the next draw.
For an ideal uniform input this residual is uniform and independent of the
selected discrete outcome. Reusing the original number for the band decision
would not have this property. Floating arithmetic and finite random-number
resolution produce the usual finite precision sampling error.

The detector computes the selected center in \(O(N)\), the projection with
four forward Fourier transforms in \(O(N\log N)\), and the complete band
weights in \(O(N)\). It computes no grid-sized matrix square root and does
not enumerate all centers during an event. State preparation at the requested
time uses the source's existing inverse transforms. Gaussian profiles have a
bounded cache.

## Energy and returned records

The converted outcome acts with \(B\) on the rejected band. If its final
positive energy expectation is \(\bar E\), the conversion increases
source energy by \(2\bar E\) relative to the selected negative component.
The record exposes this quantity as `conversionWork`. Localization can also
change energy; `sourceEnergyChange` is the complete final-minus-initial source
energy expectation. These fields identify the detector's energetic action.
The code makes no claim that the free Hamiltonian supplies that work or that
the measurement conserves source energy.

Each event retains `x`, `band`, `convertedNegativeBand`, `positionProbability`,
`bandProbability`, `probability`, both conditional band probabilities, initial
and final source energy, `sourceEnergyChange`, `conversionWork`, the numerical
norm-completeness residual, and the instrument identifier. The compatibility
field `weight` equals `positionProbability`; it is not the unnormalized window
weight returned by the field reset. Inherited `frame` adds the particle ID and
time and preserves every additional field.

## Integration and checks

The worker can use the detector by importing
`BornReadoutField as ContinuumField` from `readout_instrument.js`. The base
field, frame renderer and shared UI files are not modified by this module.
The exported `recordPositiveEnergy` function also supports direct use with a
base field instance.

Run `node oph_defects/viewer2/readout_instrument_check.mjs` from the simulator
root. The checks reconstruct every Kraus effect on the full two-by-two-by-two
grid, including all massless zero modes, and verify their sum equals the
retained projector. They also test posterior carrier membership and norm,
both original joint outcome probabilities, seeded complete-record statistics,
independent random draws, invalid-input rejection, Gaussian partition
normalization, and a counterexample to sampling the prior point density.
These are finite-model tests; they do not establish a continuum detector,
particle identification or relativistic measurement apparatus.


## Replaying a displayed packet

“Save this readout” exports the complete history of the selected packet,
including its initial spinor, Gaussian preparation, finite grid, detector
widths, source times and the uniform variates used by the display sampler.
The visible trail is limited to eighteen records; the exported transcript
retains every record since this packet was prepared. Sampling variates are
viewer apparatus inputs, not values read from the native repair tape.

Run `node readout_replay.mjs saved-readout.json` from this viewer directory.
The replay recomputes every position and energy-band outcome, its original
joint probability and the detector's energy transfer. It rejects incomplete
histories and altered probabilities. Floating computations are checked to
a relative/absolute tolerance of `2e-10`; the exact completeness identity
in the proof is independent of those numerical replay checks.
