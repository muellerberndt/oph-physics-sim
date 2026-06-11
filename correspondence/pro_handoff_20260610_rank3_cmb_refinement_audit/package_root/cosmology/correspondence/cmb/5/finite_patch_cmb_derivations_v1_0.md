# OPH-CMB finite-patch derivations v1.0

This note closes the analytic finite-patch side as far as it can be closed without importing a populated OPH-H3 simulator or a private Planck likelihood installation. It is written as a theorem ledger: each theorem states its hypotheses, result, proof, and remaining certificate if any.

## 0. Objects and status

Let \(Q_r\) be a finite observer-facing packet quotient at regulator stage \(r\). Its patch states are finite, its overlap readouts are finite, and its accepted repair relation is terminating and confluent on the declared physical quotient.

The normal-form map is

\[
N_r:Q_r\to Q_{r,{\rm nf}}.
\]

A screen/freezeout observable is any quotient-local scalar

\[
X_r:Q_{r,{\rm nf}}\to \mathbb R^{V_r}
\]

on screen cells \(V_r\subset S^2\). The OPH CMB scalar is the continuum or high-resolution limit of a centered field

\[
\chi_r(i)=X_r(N_r(q))_i-\langle X_r\rangle.
\]

The physical CMB comparison is not \(\chi\) itself. The comparison chain is

\[
\chi(\hat n)\to C_\ell^\chi\to P_{\mathcal R}^{\rm OPH}(k)\to C_\ell^{TT,TE,EE,BB,\phi\phi}.
\]

## 1. Finite patch normal-form field theorem

**Theorem 1.** Suppose a fixed-cutoff packet quotient \(Q_r\) has a terminating and confluent repair relation. Then every packet distribution \(p\in \Delta(Q_r)\) has a unique freezeout pushforward distribution on normal forms:

\[
\mathcal C_{Q_r}(p)=\sum_{q\in Q_r}p_q\delta_{N_r(q)}.
\]

Every screen scalar \(X_r\) therefore induces well-defined freezeout moments

\[
\mathbb E[X_r]=\sum_{q\in Q_r}p_qX_r(N_r(q)),
\]

\[
\mathrm{Cov}_r(X_i,X_j)=
\sum_{q\in Q_r}p_q
\bigl(X_i(N_r(q))-\bar X_i\bigr)
\bigl(X_j(N_r(q))-\bar X_j\bigr).
\]

**Proof.** Termination and confluence imply that every repair sequence from \(q\) reaches the same terminal normal form \(N_r(q)\). The map \(q\mapsto N_r(q)\) is therefore a well-defined quotient map. Pushforward of a probability distribution by a well-defined finite map is unique. Moments of any scalar observable on the terminal quotient are ordinary finite sums. □

**CMB meaning.** The finite-patch simulator does not need to output a physical temperature sky. It needs to output quotient-normal-form screen fields whose covariance is stable under schedule changes.

## 2. MaxEnt inverse-Laplacian theorem

Let \(G_r=(V_r,E_r)\) be a screen graph approximating \(S^2\), with graph Laplacian \(L_r\). Let \(\chi\in\mathbb R^{|V_r|}\) be centered, \(\sum_i w_i\chi_i=0\). Impose a fixed repair-smoothness budget

\[
\mathbb E[\chi^TL_r\chi]=\kappa_r.
\]

**Theorem 2.** The maximum-entropy distribution on centered finite screen fields with fixed repair-smoothness budget is Gaussian,

\[
p_r(\chi)\propto
\exp\left[-\frac{\beta_r}{2}\chi^TL_r\chi\right],
\]

with covariance

\[
\Sigma_r=\beta_r^{-1}L_r^+,
\]

where \(L_r^+\) is the Moore-Penrose inverse on the subspace orthogonal to constants.

**Proof.** MaxEnt with normalization, zero mean, and one quadratic expectation constraint gives an exponential-family density with quadratic sufficient statistic \(\chi^TL_r\chi\). Because constants are removed, \(L_r\) is positive definite on the centered subspace. The resulting Gaussian has precision \(\beta_r L_r\), hence covariance \(\beta_r^{-1}L_r^+\). □

For an isotropic refinement whose graph Laplacian spectrally converges to \(-\Delta_{S^2}\), the eigenvectors converge to spherical harmonics and eigenvalues converge to \(\ell(\ell+1)\). Therefore

\[
\boxed{
\langle a_{\ell m}a^*_{\ell'm'}\rangle
=\frac{A_\chi}{\ell(\ell+1)}\delta_{\ell\ell'}\delta_{mm'},\qquad \ell\ge1.
}
\]

Thus

\[
\boxed{D_\ell^\chi\equiv\frac{\ell(\ell+1)}{2\pi}C_\ell^\chi\simeq {\rm constant}.}
\]

This is the screen/freezeout analogue of scale invariance.

## 3. Fractional repair and scalar tilt

A local repair process with anomalous long-range refinement memory is represented by a fractional repair operator

\[
\mathcal K_{\eta_R}=(-\Delta_{S^2}+\mu^2)^{1+\eta_R/2}.
\]

The MaxEnt action is

\[
S[\chi]=\frac{1}{2A_\chi}\int_{S^2}\chi\mathcal K_{\eta_R}\chi\,d\Omega.
\]

Then

\[
\boxed{
C_\ell^\chi=
\frac{A_\chi}{[\ell(\ell+1)+\mu^2]^{1+\eta_R/2}}.
}
\]

For \(\ell\gg1\),

\[
D_\ell^\chi\propto \ell^{-\eta_R}.
\]

Under the first screen-to-primordial bridge \(\ell\simeq kD_*\),

\[
P_{\mathcal R}^{\rm OPH}(k)
=A_s\left(\frac{k}{k_0}\right)^{-\eta_R}F_{\rm OPH}(k),
\]

so

\[
\boxed{n_s=1-\eta_R.}
\]

The exact OPH target promoted in the v0.9 gate is

\[
\boxed{\eta_R=e\alpha(0)\sqrt\pi=0.035158856969,}
\]

\[
\boxed{n_s=0.964841143031.}
\]

This is an OPH-only value if \(\alpha(0)\) is emitted by the pixel fixed-point branch rather than inserted from the CMB fit.

## 4. Finite freezeout capacity theorem

Let the effective freezeout subspace include all spherical harmonics with

\[
0\le\ell\le L_f.
\]

The number of real angular modes is

\[
N_{\rm frz}^{\rm angular}=
\sum_{\ell=0}^{L_f}(2\ell+1)=(L_f+1)^2.
\]

Therefore

\[
\boxed{L_f=\sqrt{N_{\rm frz}^{\rm angular}}-1.}
\]

A finite angular freezeout window may be written

\[
W_\ell=\exp\left[-\frac{\ell(\ell+1)}{2\ell_{\rm cap}^2}\right].
\]

A finite-capacity covariance floor has the generic form

\[
N_\ell^{\rm frz}=\frac{\sigma_{\rm frz}^2}{N_{\rm frz}^{\rm angular}}W_\ell^{\rm frz}.
\]

The exact branch used in v0.9 chooses

\[
\ell_{\rm IR}=32,
\qquad
N_{\rm frz}^{\rm angular}=(32+1)^2=1089.
\]

This does **not** mean total de Sitter screen capacity. It is an effective low-angle freezeout subspace for the observer-facing CMB screen.

## 5. Global repair IR kernel

The lowest modes are sensitive to global screen repair because they are comparable to the whole visible cap. Model the global repair operator as a rank-one or low-rank positive subtraction concentrated at low \(\ell\):

\[
C_\ell\mapsto C_\ell F_{\rm IR}(\ell),
\]

\[
\boxed{
F_{\rm IR}(\ell)=
1-q_{\rm IR}
\exp\left[-\frac{\ell(\ell+1)}{\ell_{\rm IR}(\ell_{\rm IR}+1)}\right].
}
\]

The exact v0.9 branch sets

\[
\boxed{q_{\rm IR}=\frac14,
\qquad \ell_{\rm IR}=32.}
\]

Interpretation of the exact values:

1. \(q_{\rm IR}=1/4\) is the scalar projection of a four-channel global cap-pair repair reserve when one channel is used by the global closure constraint.
2. \(\ell_{\rm IR}=32\) is equivalent to a closed freezeout packet count \((\ell_{\rm IR}+1)^2=1089=33^2\), i.e. 33 angular levels including the monopole bookkeeping level.

This is a **finite-patch closure hypothesis** now expressed as a theorem conditional on the active-channel and 33-level freezeout clauses. The remaining simulator certificate is to show that the OPH-FPE freezeout dynamics emits those two clauses without fitting to CMB data.

For the exact kernel:

\[
F_{\rm IR}(2)=0.751416427,
\quad
F_{\rm IR}(3)=0.752824829,
\quad
F_{\rm IR}(10)=0.774731224,
\quad
F_{\rm IR}(32)=0.908030140.
\]

The physical TT transfer response is weaker than these raw primordial ratios because radiative transfer mixes neighboring \(k\)-modes, but the kernel is designed to vanish in the acoustic domain.

## 6. Parity/cap-pair covariance theorem

Let \(\Pi\) be the orientation-reversal operator on the screen. Decompose the field into parity-even and parity-odd pieces:

\[
\chi_\pm=\frac12(\chi\pm\Pi\chi).
\]

If a residual cap-pair synchronization defect shifts the relative precision of these two subspaces by a smooth low-angle envelope, then the diagonal angular covariance becomes

\[
\boxed{
C_\ell^{P}=C_\ell^{(0)}
\left[1+\epsilon_P(-1)^\ell e^{-\ell/\ell_P}\right].
}
\]

This term is not a scalar primordial \(P(k)\) correction. It is an angular covariance correction. It must be tested in \(a_{\ell m}\)-space with sky masks and component-separated maps.

The v0.9 diagnostic target is

\[
\epsilon_P\simeq -0.98,
\qquad
\ell_P\simeq4.62,
\]

but this should be treated as a map-space target, not as a final physical amplitude.

## 7. Off-diagonal covariance and BipoSH form

The general OPH covariance is

\[
\langle a_{\ell m}a^*_{\ell'm'}\rangle
=C_\ell\delta_{\ell\ell'}\delta_{mm'}+
\Delta^{\rm OPH}_{\ell m,\ell'm'}.
\]

The support-visible, rotation-broken part should be written in bipolar spherical harmonics:

\[
\Delta^{\rm OPH}_{\ell m,\ell'm'}=
\sum_{LM} A^{LM}_{\ell\ell'}(-1)^{m'}
C^{LM}_{\ell m\,\ell' -m'}.
\]

Finite global repair predicts low-rank, low-\(L\), low-\(\ell\) support. Therefore the next public-data target is not a scalar \(P(k)\) fit but

\[
A^{LM}_{\ell\ell'}\quad\text{for small }L,\ell,\ell'.
\]

## 8. Full OPH primordial kernel for Boltzmann transfer

The scalar isotropic handoff to CAMB/CLASS is

\[
\boxed{
P_{\mathcal R}^{\rm OPH}(k)=
A_s\left(\frac{k}{k_0}\right)^{n_s-1}
\left[1-q_{\rm IR}
\exp\left(-\frac{\ell(k)(\ell(k)+1)}{\ell_{\rm IR}(\ell_{\rm IR}+1)}\right)\right]
F_{\rm cap}(k).
}
\]

with

\[
\ell(k)=\max(kD_*,2).
\]

Use the parity and BipoSH pieces separately at angular-map level.

## 9. What remains mathematically missing

The following are not closed by the finite-patch derivations above:

1. **Active-channel theorem for \(q_{\rm IR}=1/4\).** Need a finite-patch proof that exactly one scalar global channel of a four-channel cap-pair reserve is consumed by global closure.
2. **Freezeout-level theorem for \(\ell_{\rm IR}=32\).** Need a finite-patch proof that the CMB freezeout packet subspace has 33 angular levels, not a fitted value.
3. **Anomalous dimension theorem for \(\eta_R=e\alpha\sqrt\pi\).** Need a renormalization/repair-flow proof that the fractional repair exponent equals the OPH pixel detuning.
4. **Official Planck likelihood execution.** Needs the external PR3 likelihood data and compiled `clik` library.
5. **Map-space parity/BipoSH likelihood.** Needs Planck component-separated maps, masks, and Monte Carlo skies.
6. **Dark-stress Boltzmann module.** Needs \(\rho_A(a), w_A(a), c_{s,A}^2(k,a), \sigma_A(k,a), Q_A^\mu, B_A(k,a), \Gamma_{\rm rec}(k,a)\).

Everything else in the screen covariance model is now reduced to ordinary finite-dimensional normal-form, MaxEnt, spectral-convergence, and Boltzmann-transfer steps.
