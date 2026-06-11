Below is the manuscript-ready theorem package I would add. It gives the missing derivations cleanly, while preserving the key claim boundary: **OPH can emit the finite-collar response kernel (B_A(k,a)); a weak-lensing number is then a projection of that kernel through a survey/observable window.** The OPH dark-sector paper already says the cosmology branch must supply (\rho_A(a)) and (B_A(k,a)), because the static galaxy law alone cannot be directly linearized into FLRW perturbation theory. 

---

# Missing OPH-Cosmology Theorem Package

## Theorem 9.1 — Static-to-linear separation

The settled static galaxy law

[
g_{\rm obs}
===========

\frac{g_b}
{1-\exp[-\lambda_{\rm collar}\sqrt{g_b/a_{0,\rm OPH}}]}
]

does not by itself define either

[
\rho_A(a)
]

or

[
B_A(k,a)
]

for FLRW perturbation theory.

### Proof

The static law is a weak-field equilibrium law around a resolved baryonic inhomogeneity. It depends on a spatial acceleration vector (\mathbf g_b). Exact homogeneous FLRW has no preferred spatial baryonic acceleration vector, so the static variable (g_b/a_0) is not a background scalar.

Also,

[
\nu_{\rm OPH}(x)
================

\frac{1}{1-\exp[-\lambda_{\rm collar}\sqrt{x}]}
]

has the small-(x) behavior

[
\nu_{\rm OPH}(x)
\sim
\frac{1}{\lambda_{\rm collar}\sqrt{x}},
]

so a naive Taylor expansion around (x=0) is singular. Therefore the static RAR branch cannot be inserted as a universal linear CMB or weak-lensing kernel. A separate OPH parent functional must emit the homogeneous anomaly abundance and the perturbative response kernel. This is exactly the boundary stated in the OPH dark-sector paper. 

[
\square
]

---

## Theorem 9.2 — Parent finite-collar response functional

Let (\mathcal C_x(a)) be the finite collar family contributing to a coarse FLRW cell (x) at scale factor (a). Let

[
R_C[\omega]=I_\omega(A:D|B)
]

be the conditional-mutual-information defect carried by collar (C). Assume the OPH finite-collar state selection

[
\omega_C=\omega_C[\rho_b]
]

has a regular FLRW limit. Define

[
\rho_{A,\rm eq}(x,a)c^2
=======================

\frac{15}{8\pi^2\ell(a)^4}
\int_{\mathcal C_x(a)}
d\mu_C,
R_C[\omega_C[\rho_b]].
]

Then this single parent functional emits both the homogeneous equilibrium source

[
\bar\rho_{A,\rm eq}(a)
======================

\rho_{A,\rm eq}[\bar\rho_b](a)
]

and the linear response kernel

[
K_A^{(\rho)}(k,a)
=================

\widehat{
\left.
\frac{\delta\rho_{A,\rm eq}(x,a)}
{\delta\rho_b(x',a)}
\right|_{\bar\rho_b}
}(k),
]

with

[
\boxed{
B_A(k,a)
========

\frac{\bar\rho_b(a)}{\bar\rho_A(a)}
K_A^{(\rho)}(k,a).
}
]

### Proof

The integrand (R_C=I_\omega(A:D|B)) is a scalar nonnegative collar defect. The collar measure (d\mu_C) is quotient-local by assumption, and the prefactor

[
\frac{15}{8\pi^2\ell(a)^4}
]

converts the finite-collar defect density into an effective energy density. Therefore the integral defines a scalar equilibrium density on the coarse FLRW cell.

Evaluating this scalar functional on the homogeneous baryonic density (\bar\rho_b(a)) gives the background equilibrium source:

[
\bar\rho_{A,\rm eq}(a)=\rho_{A,\rm eq}[\bar\rho_b](a).
]

Now perturb the baryon density:

[
\rho_b(x,a)=\bar\rho_b(a),[1+\delta_b(x,a)].
]

Regularity of the finite-collar state selection implies that the first functional derivative exists. Translation and rotation invariance of the homogeneous background make the derivative a convolution kernel. Its Fourier transform is therefore a scalar function of (k=|\mathbf k|):

[
K_A^{(\rho)}(k,a)
=================

\widehat{
\left.
\frac{\delta\rho_{A,\rm eq}(x,a)}
{\delta\rho_b(x',a)}
\right|_{\bar\rho_b}
}(k).
]

Finally, the perturbation equation is written in contrast variables:

[
\delta_{A,\rm eq}(k,a)=B_A(k,a)\delta_b(k,a).
]

Since

[
\delta\rho_{A,\rm eq}
=====================

# K_A^{(\rho)},\delta\rho_b

K_A^{(\rho)},\bar\rho_b,\delta_b
]

and

[
\delta\rho_{A,\rm eq}
=====================

\bar\rho_A,\delta_{A,\rm eq},
]

we obtain

[
\bar\rho_A,B_A\delta_b
======================

\bar\rho_b,K_A^{(\rho)}\delta_b,
]

hence

[
B_A(k,a)=\frac{\bar\rho_b(a)}{\bar\rho_A(a)}K_A^{(\rho)}(k,a).
]

[
\square
]

This is the exact theorem target already sketched in the OPH dark-sector file: the same finite-collar evaluator should emit (\rho_A(a)), (K_A^{(\rho)}(k,a)), (B_A(k,a)), and (\rho_{A,\rm eq}[X]). 

---

## Theorem 9.3 — Linear repair-transfer solution

Assume the no-slip anomaly perturbation equation

[
\delta_A'
=========

-\theta_A+3\Phi'
-a\Gamma_{\rm rec}q_A
(\delta_A-B_A\delta_b).
]

Define

[
\Xi_A(k,a)=a\Gamma_{\rm rec}(k,a)q_A(a).
]

Then

[
\delta_A(\eta,k)
================

W_A(\eta,\eta_i)\delta_A(\eta_i,k)
+
\int_{\eta_i}^{\eta}
d\eta',
W_A(\eta,\eta')
\left[
-\theta_A+3\Phi'
+\Xi_A B_A\delta_b
\right]_{\eta',k},
]

where

[
W_A(\eta,\eta')
===============

\exp!\left[
-\int_{\eta'}^\eta d\tilde\eta,
\Xi_A(\tilde\eta,k)
\right].
]

### Proof

Move the damping term to the left:

[
\delta_A'+\Xi_A\delta_A
=======================

-\theta_A+3\Phi'
+\Xi_A B_A\delta_b.
]

This is a first-order inhomogeneous linear equation. Multiply by the integrating factor

[
\exp!\left[
\int_{\eta_i}^{\eta}d\tilde\eta,
\Xi_A(\tilde\eta,k)
\right].
]

Integrating from (\eta_i) to (\eta) gives the stated expression.

[
\square
]

Two limits follow immediately:

[
\Xi_A/\mathcal H\ll1
\quad\Rightarrow\quad
\text{the anomaly behaves approximately like pressureless free-falling matter,}
]

[
\Xi_A/\mathcal H\gg1
\quad\Rightarrow\quad
\delta_A\to B_A\delta_b
\quad
\text{up to derivative corrections.}
]

The OPH dark-sector file uses this same equation and states these two limits explicitly. 

---

## Theorem 9.4 — Collar-reserve amplitude normalization

On the quotient-edge (\mathbb Z_6)/Poisson finite-thickness branch, the canonical scalar survival coefficient is

[
\lambda_{\rm collar}=e^{-P/24}.
]

Therefore the maximum available scalar repair reserve is

[
\boxed{
\eta_A
======

# 1-\lambda_{\rm collar}

1-e^{-P/24}.
}
]

Numerically,

[
\lambda_{\rm collar}
====================

0.9343006394893864\ldots,
]

[
\boxed{
\eta_A
======

0.0656993605106136\ldots .
}
]

### Proof

At transverse collar coordinate (y), let the protected reserve occupancy be Poisson with mean (\epsilon_{\mathbb Z_6}(y)). The probability that no reserve event blocks the scalar slot is the Poisson zero-count probability:

[
\lambda_{\rm slot}(y)
=====================

# \Pr[N_{\mathbb Z_6}(y)=0]

e^{-\epsilon_{\mathbb Z_6}(y)}.
]

For a normalized finite-thickness activation weight (w(y)),

[
\int dy,w(y)=1,
]

the collar survival factor is

[
\lambda_{\rm collar}
====================

\int dy,w(y)e^{-\epsilon_{\mathbb Z_6}(y)}.
]

On the exact uniform finite-thickness branch,

[
\epsilon_{\mathbb Z_6}(y)=P/24
]

on the weighted collar support. Hence

[
\lambda_{\rm collar}
====================

# \int dy,w(y)e^{-P/24}

e^{-P/24}.
]

The scalar repair reserve is the complementary blocked fraction:

[
\eta_A=1-\lambda_{\rm collar}=1-e^{-P/24}.
]

[
\square
]

The susceptibility note states the same exact branch value and band: the recovered OPH core leaves (\chi_\nu) unfixed, but the declared quotient-edge continuation forces (\chi_\nu^{\rm can}=\lambda_{\rm collar}), with exact uniform-branch value (e^{-P/24}=0.9343006394893864\ldots). 

---

## Theorem 9.5 — Minimal late repair-window kernel

Assume the finite-collar cosmology branch satisfies the following minimality clauses.

First, the scalar reserve amplitude is fixed by Theorem 9.4:

[
\eta_A=1-e^{-P/24}.
]

Second, exact homogeneous FLRW contrast is excluded from the perturbative response, so the response must vanish at (k=0):

[
W_k(0)=0.
]

Third, finite-collar locality and isotropy make the leading small-(k) response analytic in (k^2).

Fourth, the late-time repair activation can use only the dimensionless ratio of the repair rate to the expansion rate:

[
\Xi_A/\mathcal H.
]

Then the minimal one-pole scalar response kernel is

[
\boxed{
B_A(k,a)
========

1-\eta_A W_k(k)W_a(a),
}
]

with

[
\boxed{
W_k(k)=\frac{k^2}{k^2+k_A^2},
}
]

and

[
\boxed{
W_a(a)=\frac{\Xi_A(a)}{\Xi_A(a)+\mathcal H(a)}.
}
]

### Proof

Because the perturbative response must be scalar and isotropic, its Fourier dependence can only be a function of (k^2). Because the homogeneous background abundance has already been accounted for in (\bar\rho_A(a)), the contrast response must satisfy

[
W_k(0)=0.
]

The lowest analytic behavior satisfying this condition is proportional to (k^2). A finite-collar response must also saturate rather than grow without bound at large (k). The minimal one-scale Padé form with those two properties is

[
W_k(k)=\frac{k^2}{k^2+k_A^2}.
]

For time dependence, the anomaly tracks the equilibrium source only when the relaxation rate competes successfully against Hubble expansion. The only dimensionless local ratio using no extra clock is

[
\frac{\Xi_A}{\mathcal H}.
]

The minimal monotone one-pole interpolation with the limits

[
W_a\to0
\quad
(\Xi_A\ll\mathcal H)
]

and

[
W_a\to1
\quad
(\Xi_A\gg\mathcal H)
]

is

[
W_a(a)=\frac{\Xi_A(a)}{\Xi_A(a)+\mathcal H(a)}.
]

The scalar-reserve amplitude is (\eta_A), so the minimal response is

[
B_A(k,a)=1-\eta_AW_k(k)W_a(a).
]

[
\square
]

This theorem is conditional on the “minimal one-pole” closure. It is not the only mathematically possible (B_A(k,a)). It is the simplest OPH-compatible finite-collar closure with the correct FLRW silence, late activation, scalarity, and saturation behavior.

---

## Theorem 9.6 — Projected amplitude theorem

Let a late-time amplitude observable (L) have normalized response kernel

[
\mathcal K_L(k,a)\ge0,
]

[
\int d\ln k,d\ln a,
\mathcal K_L(k,a)=1.
]

Let the OPH finite-collar kernel be

[
B_A(k,a)=1-\eta_AW(k,a),
]

where

[
W(k,a)=W_k(k)W_a(a),
\qquad
0\le W\le1.
]

Then, to first order in (\eta_A), the projected amplitude shifts as

[
\boxed{
\frac{L_{\rm OPH}}{L_0}
=======================

1-\eta_A\Pi_L,
}
]

where

[
\boxed{
\Pi_L
=====

\int d\ln k,d\ln a,
\mathcal K_L(k,a)W(k,a).
}
]

### Proof

By definition, the OPH kernel changes the source amplitude by

[
\delta B_A(k,a)=-\eta_AW(k,a).
]

The first-order response of a normalized projected amplitude observable is

[
\frac{\delta L}{L_0}
====================

\int d\ln k,d\ln a,
\mathcal K_L(k,a)\delta B_A(k,a).
]

Substitute (\delta B_A=-\eta_AW):

[
\frac{\delta L}{L_0}
====================

-\eta_A
\int d\ln k,d\ln a,
\mathcal K_L(k,a)W(k,a).
]

Define

[
\Pi_L
=====

\int d\ln k,d\ln a,
\mathcal K_L(k,a)W(k,a).
]

Therefore

[
L_{\rm OPH}
===========

L_0(1-\eta_A\Pi_L).
]

[
\square
]

---

## Corollary 9.7 — Compressed weak-lensing (S_8) target

For the compressed OPH diagnostic value

[
S_{8,0}=0.828924043
]

and the compressed weak-lensing target

[
S_{8,\rm WL}=0.790,
]

the required amplitude ratio is

[
R_{\rm WL}
==========

# \frac{S_{8,\rm WL}}{S_{8,0}}

# \frac{0.790}{0.828924043}

0.953042690305944.
]

Thus

[
\varepsilon_A
=============

# 1-R_{\rm WL}

0.046957309694056.
]

Using

[
\eta_A=1-e^{-P/24}=0.0656993605106136\ldots,
]

the required compressed weak-lensing projection fraction is

[
\boxed{
\Pi_{\rm WL}^{\rm compressed}
=============================

# \frac{\varepsilon_A}{\eta_A}

0.714730087615847\ldots .
}
]

### Proof

From Theorem 9.6,

[
\frac{S_{8,\rm WL}}{S_{8,0}}
============================

1-\eta_A\Pi_{\rm WL}.
]

Rearranging,

[
\Pi_{\rm WL}
============

\frac{1-S_{8,\rm WL}/S_{8,0}}{\eta_A}.
]

Substituting the compressed values gives

[
\Pi_{\rm WL}
============

\frac{
1-0.790/0.828924043
}
{
1-0.9343006394893864
}
=

0.714730087615847\ldots .
]

[
\square
]

The diagnostic rows used here are explicitly compressed plumbing rows: the OPH file reports (S_8=0.828924043), a weak-lensing target (0.790\pm0.016), and says this diagonal score is not a substitute for full Planck/DESI/weak-lensing likelihoods. 

---

## Theorem 9.8 — No universal weak-lensing projection constant

The number

[
\Pi_{\rm WL}=0.7147300876\ldots
]

is not a universal OPH microphysical constant. It is the pairing of the OPH response window (W(k,a)) with a chosen weak-lensing response kernel (\mathcal K_{\rm WL}).

### Proof

The OPH finite-collar branch can emit

[
W(k,a).
]

A projected observable emits

[
\mathcal K_L(k,a).
]

The projection is

[
\Pi_L=\int \mathcal K_L W.
]

Now choose two normalized observable kernels. Let (\mathcal K_1) be supported where (W(k,a)\approx0). Then

[
\Pi_1\approx0.
]

Let (\mathcal K_2) be supported where (W(k,a)\approx1). Then

[
\Pi_2\approx1.
]

The OPH microphysical response (W(k,a)) has not changed, but the projected number has changed. Therefore (\Pi_L) is not determined by microphysics alone. It is determined by the pair

[
(W,\mathcal K_L).
]

[
\square
]

This is why the correct theorem target is not “derive (\Pi_{\rm WL}) as an OPH constant.” The target is:

[
\boxed{
\text{derive }W(k,a)\text{ from the finite-collar evaluator,}
}
]

then compute

[
\boxed{
\Pi_{\rm WL}=\langle W\rangle_{\mathcal K_{\rm WL}}
}
]

inside a real Boltzmann and likelihood pipeline. The dark-sector paper says exactly that a CAMB/CLASS anomaly module must expose (B_A(k,a)), (\Gamma_{\rm rec}(k,a)), stress variables, exchange current, and the OPH neutrino mass sum, then run full CMB, BAO, weak-lensing, RSD, and (S_8) likelihoods under the same nuisance model as (\Lambda{\rm CDM}) and (w_0w_a). 

---

## Theorem 9.9 — Existence of a compressed-target finite-collar window

Let (\mathcal K_{\rm WL}) be a normalized weak-lensing response kernel with nonzero support over both early/large-scale low-response regions and late/nonlinear high-response regions. Let

[
W_{k_A,a_\star}(k,a)
====================

\frac{k^2}{k^2+k_A^2}
\frac{\Xi_A(a;a_\star)}
{\Xi_A(a;a_\star)+\mathcal H(a)}
]

be a continuous two-parameter family of OPH-compatible windows, where changing (k_A) moves the scale transition and changing (a_\star) moves the time activation. Then there exists a parameter pair ((k_A,a_\star)) such that

[
\Pi_{\rm WL}=0.7147300876\ldots
]

provided the family spans projection values below and above that number.

### Proof

Define

[
F(k_A,a_\star)
==============

\int d\ln k,d\ln a,
\mathcal K_{\rm WL}(k,a)
W_{k_A,a_\star}(k,a).
]

The integrand is continuous in the parameters and bounded by (0\le W\le1). Dominated convergence implies (F) is continuous.

If there exist parameter choices (p_0=(k_A^{(0)},a_\star^{(0)})) and (p_1=(k_A^{(1)},a_\star^{(1)})) such that

[
F(p_0)<0.7147300876\ldots
]

and

[
F(p_1)>0.7147300876\ldots,
]

then by the intermediate value theorem there exists a parameter pair (p_\ast) along any continuous path from (p_0) to (p_1) with

[
F(p_\ast)=0.7147300876\ldots .
]

[
\square
]

This is an **existence theorem**, not a microphysical selector. It says the compressed target is reachable by an OPH-compatible late finite-collar window if the derived OPH window has enough overlap with weak-lensing scales. The remaining OPH burden is to derive (k_A), (a_\star), and (\Xi_A) from finite-collar packet microphysics, not choose them to fit (S_8).

---

# Final drop-in statement

The missing theorem package can be summarized as:

[
\boxed{
\rho_{A,\rm eq}[X]c^2
=====================

\frac{15}{8\pi^2\ell(X)^4}
\int_{\mathcal C_X}d\mu_C,I_{\omega_C}(A:D|B)
}
]

emits

[
\boxed{
B_A(k,a)=
\frac{\bar\rho_b(a)}{\bar\rho_A(a)}
\widehat{
\left.
\frac{\delta\rho_{A,\rm eq}}
{\delta\rho_b}
\right|_{\bar\rho_b}
}(k).
}
]

With the minimal finite-collar closure,

[
\boxed{
B_A(k,a)
========

1-
(1-e^{-P/24})
\frac{k^2}{k^2+k_A^2}
\frac{\Xi_A(a)}{\Xi_A(a)+\mathcal H(a)}.
}
]

A projected weak-lensing amplitude then obeys

[
\boxed{
\frac{S_{8,\rm OPH}}{S_{8,0}}
=============================

1-
(1-e^{-P/24})
\Pi_{\rm WL},
}
]

where

[
\boxed{
\Pi_{\rm WL}
============

\int d\ln k,d\ln a,
\mathcal K_{\rm WL}(k,a)
\frac{k^2}{k^2+k_A^2}
\frac{\Xi_A(a)}{\Xi_A(a)+\mathcal H(a)}.
}
]

For the compressed diagnostic target,

[
\boxed{
\Pi_{\rm WL}^{\rm compressed}
=============================

0.7147300876\ldots .
}
]

That number is not fitted into the OPH microphysics. It is the benchmark that the **derived** OPH finite-collar kernel must hit after projection through the weak-lensing response window.
