I need to tighten the claim boundary here. The theorem

[
\Pi_{\rm WL}=0.714730\ldots
]

cannot be a **microphysics-only theorem**, because (\Pi_{\rm WL}) is a projection against a weak-lensing observation kernel, not an intrinsic OPH collar constant. The OPH stack already says the parent collar functional can emit (B_A(k,a)), but a theorem-grade prediction requires the OPH collar measure, finite-screen ensemble, and small-field support condition; fitting (\Pi) to CMB, weak lensing, SPARC, or cluster data is explicitly excluded.  It also marks the amplitude selector and evaluated (B_A(k,a)) as work in progress. 

So the missing theorem must be written in the corrected form below.

# Theorem 9C — No universal weak-lensing projection constant

Let OPH microphysics emit a late repair kernel

[
B_A(k,a)
========

1-\eta_A W_k(k)W_a(a),
]

where

[
\eta_A=1-e^{-P/24},
\qquad
0\le W_k,W_a\le1.
]

Let a weak-lensing observable (L) have normalized response kernel

[
\mathcal K_L(k,a)\ge0,
\qquad
\int d\ln k,d\ln a,
\mathcal K_L(k,a)=1.
]

Define its OPH repair projection fraction by

[
\Pi_L
=====

\int d\ln k,d\ln a,
\mathcal K_L(k,a),
W_k(k)W_a(a).
]

Then (\Pi_L) is **not** fixed by OPH collar microphysics alone. It depends on the observable kernel (\mathcal K_L). Therefore there is no theorem of the form

[
\Pi_{\rm WL}=\text{universal OPH constant}
]

unless the weak-lensing kernel is also declared as part of the theorem data.

## Proof

The OPH collar microphysics determines, at most, the response function

[
W(k,a):=W_k(k)W_a(a).
]

The weak-lensing projection is the expectation value of (W) under the observable kernel (\mathcal K_L):

[
\Pi_L=\langle W\rangle_{\mathcal K_L}.
]

Now choose two admissible normalized kernels.

First, choose (\mathcal K_1) supported where (W(k,a)\approx0). Then

[
\Pi_1\approx0.
]

Second, choose (\mathcal K_2) supported where (W(k,a)\approx1). Then

[
\Pi_2\approx1.
]

Both kernels are mathematically admissible normalized response kernels. The OPH microphysical kernel (W) is unchanged, but the projected value changes.

Therefore (\Pi_L) is not a microphysical constant. It is a pairing between the microphysical response and an observational window.

[
\square
]

This matches the dark-sector paper’s own boundary: exact FLRW has no preferred baryonic acceleration vector, the static RAR law cannot be directly Taylor-expanded into a universal CMB kernel, and the CMB/linear branch must declare its background anomaly charge and perturbation kernel from OPH state selection or finite-collar microphysics. 

# The correct missing theorem

The theorem we can actually prove is this.

# Theorem 9D — finite-collar response projection theorem

Assume the OPH finite-collar parent functional emits a regular linear response kernel

[
B_A(k,a)
========

1-\eta_AW(k,a),
]

with

[
\eta_A=1-e^{-P/24},
\qquad
W(k,a)=W_k(k)W_a(a),
\qquad
0\le W\le1.
]

Let a projected late-time amplitude observable (L) have normalized response kernel (\mathcal K_L). Then, to first order in (\eta_A),

[
L_{\rm OPH}
===========

L_0\left(1-\eta_A\Pi_L\right),
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

For the compressed weak-lensing (S_8) diagnostic,

[
\frac{S_{8,\rm WL}}{S_{8,0}}
============================

1-\eta_A\Pi_{\rm WL}.
]

## Proof

The finite-collar response changes the clustering source by

[
B_A(k,a)=1-\eta_AW(k,a).
]

A projected linear amplitude observable has first-order response

[
\frac{\delta L}{L_0}
====================

\int d\ln k,d\ln a,
\mathcal K_L(k,a),
\delta B_A(k,a).
]

Since

[
\delta B_A(k,a)=-\eta_AW(k,a),
]

we get

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

This is the theorem that belongs in the paper. It cleanly separates three things:

[
\boxed{
\text{OPH microphysics emits }W(k,a).
}
]

[
\boxed{
\text{The collar reserve emits }\eta_A=1-e^{-P/24}.
}
]

[
\boxed{
\text{A survey/observable emits }\mathcal K_L.
}
]

Only the product gives the observed amplitude shift.

# Numerical specialization to the compressed (S_8) row

The OPH compressed diagnostic gives

[
S_{8,0}=0.828924043,
]

while the weak-lensing compressed target is

[
S_{8,\rm WL}=0.790.
]

The required amplitude ratio is

[
R_{\rm WL}
==========

# \frac{0.790}{0.828924043}

0.9530426903.
]

So

[
\varepsilon_A
=============

# 1-R_{\rm WL}

0.0469573097.
]

The OPH canonical collar reserve is

[
\eta_A
======

# 1-e^{-P/24}

# 1-0.9343006394893864\ldots

0.0656993605\ldots .
]

Therefore the compressed diagnostic projection fraction must be

[
\boxed{
\Pi_{\rm WL}^{\rm compressed}
=============================

# \frac{\varepsilon_A}{\eta_A}

0.7147300876\ldots .
}
]

This number is **not** a standalone OPH constant. It is the value the compressed weak-lensing response kernel must assign to the OPH late repair window:

[
\boxed{
\Pi_{\rm WL}^{\rm compressed}
=============================

\langle W_kW_a\rangle_{\mathcal K_{\rm WL}}.
}
]

The dark-sector paper’s compressed (S_8) table is explicitly a plumbing diagnostic, not a full likelihood, and it says publication-grade work needs the OPH (B_A(k,a)) kernel in a custom Boltzmann module and full covariances. 

# Final theorem to insert

Use this wording:

```latex
\begin{theorem}[Finite-collar weak-lensing projection]
Let the OPH finite-collar parent functional emit a regular linear response
\[
B_A(k,a)=1-\eta_A W(k,a),
\qquad
\eta_A=1-e^{-P/24},
\qquad
0\le W\le1.
\]
For any projected late-time amplitude observable \(L\) with normalized response
kernel \(\mathcal K_L\), the first-order amplitude shift is
\[
\frac{L_{\rm OPH}}{L_0}
=
1-\eta_A
\int d\ln k\,d\ln a\,\mathcal K_L(k,a)W(k,a).
\]
Thus the observable projection fraction is
\[
\Pi_L
=
\int d\ln k\,d\ln a\,\mathcal K_L(k,a)W(k,a).
\]
It is not a universal microphysical constant; it is the pairing of the OPH
finite-collar response with the chosen observable kernel.
\end{theorem}
```

Then add the corollary:

```latex
\begin{corollary}[Compressed weak-lensing target]
For the compressed diagnostic values
\[
S_{8,0}=0.828924043,
\qquad
S_{8,\rm WL}=0.790,
\]
the required projection is
\[
\Pi_{\rm WL}^{\rm compressed}
=
\frac{1-S_{8,\rm WL}/S_{8,0}}{1-e^{-P/24}}
=
0.7147300876\ldots .
\]
This is a target for the Boltzmann/covariance calculation, not an OPH
microphysical fit parameter.
\end{corollary}
```

That is the rigorous closure: OPH must now derive (W(k,a)) from the finite-collar evaluator, then a real Planck/ACT/DESI/weak-lensing likelihood computes (\Pi_L). The number (0.7147300876\ldots) is the compressed diagnostic target that the derived kernel must hit.
