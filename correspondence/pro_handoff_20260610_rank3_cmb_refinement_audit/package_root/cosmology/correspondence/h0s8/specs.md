Yes. The remaining closure can be packaged as a **certificate theorem stack**. The key point is that the last gaps are not new metaphysics; they are audit/certification gaps. OPH already has fixed-cutoff checkpoints, record algebras, quotient normal forms, refinement/coarse-graining controls, and de Sitter screen capacity on their declared branches. In particular, the checkpoint theorem says that matching accessible record algebra, accessible state, external interface, and future schedule class fixes the future observer-accessible law up to trace-distance error; the consensus theorem gives schedule-independent normal forms only when termination, local confluence, and repair completeness hold; and the de Sitter branch identifies (S_{\rm dS}=N_{\rm scr}=A_{\rm dS}/(4\ell_P^2)).   

Below is the remaining theorem package.

# Lane 8 Addendum III — Remaining closure theorems

Use base-2 logarithms unless stated otherwise.

Let (C_t) be the present cosmological checkpoint. Let (X) be a past macrovariable or past record-source variable, and let

[
Y=(Y_1,\ldots,Y_m)
]

be the present record readout extracted from (\mathcal R_t). A decoder

[
D:Y\to \mathcal X
]

has error

[
\Pr[D(Y)\neq X]\le \varepsilon.
]

Define

[
I_{\rm rec}^{\varepsilon}(X:Y)
==============================

H(X)-h_2(\varepsilon)-\varepsilon\log(|\mathcal X|-1).
]

The fake deficit of an ancestry (A_s\to C_t) is

[
F_{\rm fake}(A_s\to C_t)
========================

\left[
I_{\rm rec}^{\varepsilon}(C_t)
------------------------------

## N_{\rm res}(A_s)

## B_{\rm hid}^{\max}(s,t)

## I_{\rm pre}(A_s)

\Delta_{\rm approx}
\right]_+ .
]

Here (N_{\rm res}) is available record-writing negentropy, (B_{\rm hid}^{\max}) is allowed hidden entropy export, (I_{\rm pre}) is pre-existing provenance, and (\Delta_{\rm approx}) is the accumulated record/recovery/refinement error.

---

# Theorem L8.21 — Audited record-payload lower bound

## Statement

Suppose (Y) decodes (X) with error at most (\varepsilon). Then

[
\boxed{
I(X:Y)\ge I_{\rm rec}^{\varepsilon}(X:Y).
}
]

If (\varepsilon) is not known exactly but is estimated from (n) independent calibration samples with empirical error (\widehat\varepsilon), then with probability at least (1-\delta),

[
\varepsilon
\le
\widehat\varepsilon
+
\sqrt{\frac{\ln(1/\delta)}{2n}},
]

and therefore a certified payload lower bound is

[
\boxed{
I_{\rm rec}^{\rm cert}
======================

## H(X)

## h_2(\varepsilon_\delta)

\varepsilon_\delta\log(|\mathcal X|-1),
}
]

where

[
\varepsilon_\delta
==================

\widehat\varepsilon
+
\sqrt{\frac{\ln(1/\delta)}{2n}}.
]

## Proof

Fano’s inequality gives

[
H(X|Y)
\le
h_2(\varepsilon)+\varepsilon\log(|\mathcal X|-1).
]

Thus

[
I(X:Y)=H(X)-H(X|Y)
\ge
H(X)-h_2(\varepsilon)-\varepsilon\log(|\mathcal X|-1).
]

For the empirical version, Hoeffding’s inequality gives

[
\Pr[
\varepsilon-\widehat\varepsilon
\ge a
]
\le
e^{-2na^2}.
]

Set

[
a=\sqrt{\frac{\ln(1/\delta)}{2n}}.
]

Then

[
\Pr[
\varepsilon
\le
\widehat\varepsilon+a
]
\ge
1-\delta.
]

Substituting the certified upper bound (\varepsilon_\delta) into the Fano lower bound proves the result.

[
\Box
]

This is the theorem that turns “present records contain information” into an auditable lower bound.

---

# Theorem L8.22 — Redundancy does not inflate payload, but suppresses fake histories

## Statement

Let

[
Y=(Y_1,\ldots,Y_m)
]

be a redundant present record family. Then

[
\boxed{
I(X:Y_1,\ldots,Y_m)
===================

\sum_{j=1}^{m}
I(X:Y_j\mid Y_{<j}).
}
]

So duplicate records contribute no extra nonredundant payload once their content is already conditioned on.

However, suppose fake fragments are independently generated under a fake-history hypothesis (H_B). Let

[
\widehat X_j=D_j(Y_j).
]

If each fake decoded fragment has maximum point probability

[
\max_x\Pr[\widehat X_j=x]\le p_j,
]

then

[
\boxed{
\Pr_{H_B}[\widehat X_1=\cdots=\widehat X_m]
\le
\prod_{j=2}^{m}p_j.
}
]

If (p_j\le p_\star) for all (j\ge2), then

[
\boxed{
\Pr_{H_B}[\widehat X_1=\cdots=\widehat X_m]
\le
p_\star^{m-1}.
}
]

## Proof

The first identity is the chain rule for mutual information:

[
I(X:Y_1,\ldots,Y_m)
===================

\sum_{j=1}^{m}
I(X:Y_j\mid Y_1,\ldots,Y_{j-1}).
]

If (Y_j) is a pure duplicate of earlier records, then

[
I(X:Y_j\mid Y_{<j})=0.
]

So redundancy does not inflate payload.

For fake agreement, independence gives

[
\Pr[\widehat X_1=\cdots=\widehat X_m]
=====================================

\sum_x
\Pr[\widehat X_1=x]
\prod_{j=2}^{m}
\Pr[\widehat X_j=x].
]

Using

[
\Pr[\widehat X_j=x]\le p_j,
]

we get

[
\Pr[\widehat X_1=\cdots=\widehat X_m]
\le
\sum_x
\Pr[\widehat X_1=x]
\prod_{j=2}^{m}p_j
==================

\prod_{j=2}^{m}p_j.
]

The uniform bound (p_j\le p_\star) gives (p_\star^{m-1}).

[
\Box
]

So cosmological redundancy should be handled in two different ways:

[
\boxed{
\text{payload is joint information, but fake agreement is exponentially suppressed.}
}
]

---

# Theorem L8.23 — Provenance-cut inequality with hidden sectors

## Statement

Let (B_s) be an accessible provenance cut at time (s<t), and let (H_s) be a hidden sector. Suppose every faithful influence from (X) to the present record (Y) factors through ((B_s,H_s)):

[
X\longrightarrow (B_s,H_s)\longrightarrow Y.
]

Then

[
\boxed{
I(X:Y)
\le
I(X:B_s)+I(X:H_s\mid B_s).
}
]

If (H_s) contains no additional provenance about (X) beyond (B_s), namely

[
I(X:H_s\mid B_s)=0,
]

then

[
\boxed{
I(X:Y)\le I(X:B_s).
}
]

## Proof

By data processing for the Markov chain

[
X\to(B_s,H_s)\to Y,
]

we have

[
I(X:Y)\le I(X:B_s,H_s).
]

By the chain rule,

[
I(X:B_s,H_s)
============

I(X:B_s)+I(X:H_s\mid B_s).
]

Therefore

[
I(X:Y)
\le
I(X:B_s)+I(X:H_s\mid B_s).
]

If

[
I(X:H_s\mid B_s)=0,
]

then

[
I(X:Y)\le I(X:B_s).
]

[
\Box
]

This is the clean hidden-sector rule:

[
\boxed{
\text{hidden information helps only if it already carries provenance.}
}
]

If it does, it must be counted as (I_{\rm pre}) or as hidden capacity. If it does not, it cannot explain the present record payload.

---

# Theorem L8.24 — No-free-hidden-export theorem

## Statement

Let (H_s) be a hidden implementation sector excluded from the observer-facing checkpoint data at time (s). Suppose varying (H_s) while holding the checkpoint data fixed does not change

[
\mathcal R_s,\quad
\rho_s^{\rm acc},\quad
\mathfrak I_s^{\rm ext},\quad
\nu_{\ge s}.
]

Then (H_s) cannot change the future observer-accessible law.

Conversely, if (H_s) can change the future record distribution (Y), then (H_s) is not purely hidden relative to the ancestry problem: its influence must cross an accessible interface, enter a provenance bundle, or be charged against (B_{\rm hid}^{\max}).

## Proof

The checkpoint continuation theorem states that exact agreement on the accessible record algebra, accessible state, external interface, and future schedule class induces the same future observer-accessible probability law. Approximate checkpoint agreement gives the corresponding total-variation bound by trace-distance contractivity. 

So if two states differ only in (H_s), while all checkpoint data agree, then they induce the same future observer-accessible law. Thus (H_s) cannot change the distribution of present records (Y).

Conversely, suppose changing (H_s) changes the law of (Y). Then the two states do not agree as checkpoints for the ancestry problem. The influence of (H_s) must therefore appear somewhere in the observer-facing continuation: through an interface, through an update schedule class, through a provenance bundle, or through hidden export that later becomes accessible. In the Lane 8 accounting, that contribution is charged to (I_{\rm pre}) or (B_{\rm hid}^{\max}).

[
\Box
]

This blocks the escape move:

[
\text{“the past was high entropy, but the missing records were hidden.”}
]

If hidden variables never affect records, they do no explanatory work. If they do affect records, they are part of the provenance budget.

---

# Theorem L8.25 — Finite hidden-export capacity bound

## Statement

Let (E_{s,t}) be the hidden entropy sink available between (s) and (t), with finite Hilbert-space dimension

[
d_E=\dim\mathcal H_E.
]

Then

[
\boxed{
B_{\rm hid}^{\max}(s,t)
\le
\log d_E-S(E_s)+\Delta_{\rm cont}.
}
]

In particular,

[
\boxed{
B_{\rm hid}^{\max}(s,t)\le \log d_E+\Delta_{\rm cont}.
}
]

On the de Sitter screen-capacity branch, if (E) is a subregister of the static-patch screen capacity, then

[
\boxed{
B_{\rm hid}^{\max}(s,t)
\le
N_{\rm hid}^{\rm scr}
+
\Delta_{\rm cont},
\qquad
N_{\rm hid}^{\rm scr}\le N_{\rm scr}.
}
]

## Proof

The entropy exported into (E) is

[
\Delta S_E=S(E_t)-S(E_s).
]

Since (E_t) is supported on a finite-dimensional Hilbert space,

[
S(E_t)\le \log d_E.
]

Therefore

[
\Delta S_E\le \log d_E-S(E_s).
]

Taking the supremum over hidden processes compatible with the same present observer-facing normal form gives

[
B_{\rm hid}^{\max}(s,t)
\le
\log d_E-S(E_s)+\Delta_{\rm cont}.
]

Since (S(E_s)\ge0),

[
B_{\rm hid}^{\max}(s,t)\le \log d_E+\Delta_{\rm cont}.
]

On the de Sitter branch, the whole static-patch record capacity is finite,

[
N_{\rm scr}=S_{\rm dS}=\frac{A_{\rm dS}}{4\ell_P^2},
]

so any hidden subregister has capacity at most its screen-capacity share (N_{\rm hid}^{\rm scr}\le N_{\rm scr}). 

[
\Box
]

This theorem does not by itself prove that hidden export is small enough. It proves that hidden export is a finite audited quantity.

---

# Theorem L8.26 — Faithful checkpoint-chain certificate

## Statement

Let

[
C_s=C^{(0)}
\preceq
C^{(1)}
\preceq
\cdots
\preceq
C^{(n)}=C_t
]

be a finite checkpoint chain. Let (R_k) be the record readout at (C^{(k)}), and define the newly written nonredundant information

[
\Delta I_k
==========

I(X:R_k)-I(X:R_{k-1}).
]

Let interval (k) have resource

[
Q_k
===

N_k+\Delta S_{E,k}+I_{{\rm pre},k}.
]

If

[
Q_k\ge \Delta I_k
\qquad
\text{for all }k,
]

the total hidden export stays within the declared bound, and (C^{(n)}) agrees with (C_t) on the observer-facing checkpoint data, then (C_s) is a faithful ancestry of (C_t).

## Proof

For each interval, the record-writing Landauer bound gives

[
N_k+\Delta S_{E,k}+I_{{\rm pre},k}\ge \Delta I_k.
]

By hypothesis this holds interval by interval. Summing,

[
\sum_{k=1}^{n} Q_k
\ge
\sum_{k=1}^{n}\Delta I_k.
]

The increments telescope:

[
\sum_{k=1}^{n}\Delta I_k
========================

I(X:R_n)-I(X:R_0).
]

Since (R_n) is the present record readout,

[
I(X:R_n)\ge I_{\rm rec}^{\varepsilon}(C_t).
]

Thus the chain supplies enough resource to write the present record payload from the initial record content (R_0), without positive fake deficit.

Because the endpoint agrees with (C_t) on the observer-facing checkpoint data, checkpoint continuation gives the same future observer-accessible law.  Therefore (C_s) is faithful.

[
\Box
]

This is the constructive ancestry theorem: exhibit the chain and the interval resource ledger, and faithfulness follows.

---

# Theorem L8.27 — Selector dominance theorem

## Statement

Let

[
\mathcal J_{\rm arrow}
======================

\Phi_{\rm cont}
+
\lambda_FF_{\rm fake}
+
\lambda_HK_{\rm hid}
+
\lambda_IK_{\rm impl}
+
\lambda_RR_{\rm unsupported}.
]

Let

[
J_F^\star
=========

\min_{A\in\operatorname{Anc}^{\rm faithful}*s(C_t)}
\mathcal J*{\rm arrow}(A\to C_t)
]

be the best faithful cost. Suppose every nonfaithful ancestry (B) satisfies

[
\mathcal J_{\rm arrow}(B\to C_t)
\ge
J_F^\star+\mu
]

for some margin (\mu>0). Then every minimizer of (\mathcal J_{\rm arrow}) is faithful.

## Proof

Assume a minimizer (\widehat C_s) is nonfaithful. Then by hypothesis,

[
\mathcal J_{\rm arrow}(\widehat C_s\to C_t)
\ge
J_F^\star+\mu.
]

But by definition of (J_F^\star), there exists a faithful ancestry (A^\star) with

[
\mathcal J_{\rm arrow}(A^\star\to C_t)=J_F^\star.
]

Thus

[
\mathcal J_{\rm arrow}(\widehat C_s\to C_t)

>

\mathcal J_{\rm arrow}(A^\star\to C_t),
]

contradicting minimality of (\widehat C_s). Therefore no minimizer is nonfaithful.

[
\Box
]

This is the precise theorem needed to show that the selector does not merely prefer low entropy. It prefers low fake-deficit, low unsupported-record load, low continuation mismatch, and low hidden implementation complexity.

---

# Theorem L8.28 — Certified low-entropy ancestry theorem

## Statement

Assume:

1. (C_t) has reliable record payload (I_{\rm rec}^{\varepsilon}(C_t));
2. (\widehat C_s(C_t)) is selected by (\mathcal J_{\rm arrow});
3. the selector dominance theorem makes (\widehat C_s) faithful;
4. hidden export, pre-existing provenance, and approximation error are bounded by

[
B_{\rm hid}^{\max},\qquad
I_{\rm pre},\qquad
\Delta_{\rm approx}.
]

Then

[
\boxed{
N_{\rm OF}(\widehat C_s)
\ge
I_{\rm rec}^{\varepsilon}(C_t)
------------------------------

## B_{\rm hid}^{\max}

## I_{\rm pre}

\Delta_{\rm approx}.
}
]

Equivalently,

[
\boxed{
S_{\rm OF}(\widehat C_s)
\le
S_{\max}(s)
-----------

I_{\rm rec}^{\varepsilon}(C_t)
+
B_{\rm hid}^{\max}
+
I_{\rm pre}
+
\Delta_{\rm approx}.
}
]

## Proof

Because (\widehat C_s) is faithful,

[
F_{\rm fake}(\widehat C_s\to C_t)=0.
]

By definition,

[
0=
\left[
I_{\rm rec}^{\varepsilon}(C_t)
------------------------------

## N_{\rm res}(\widehat C_s)

## B_{\rm hid}^{\max}

## I_{\rm pre}

\Delta_{\rm approx}
\right]_+.
]

Therefore

[
N_{\rm res}(\widehat C_s)
\ge
I_{\rm rec}^{\varepsilon}(C_t)
------------------------------

## B_{\rm hid}^{\max}

## I_{\rm pre}

\Delta_{\rm approx}.
]

Record-writing resource is part of observer-facing or record-supporting negentropy, so

[
N_{\rm OF}(\widehat C_s)\ge N_{\rm res}(\widehat C_s).
]

Thus

[
N_{\rm OF}(\widehat C_s)
\ge
I_{\rm rec}^{\varepsilon}(C_t)
------------------------------

## B_{\rm hid}^{\max}

## I_{\rm pre}

\Delta_{\rm approx}.
]

Using

[
N_{\rm OF}=S_{\max}-S_{\rm OF},
]

we get

[
S_{\rm OF}(\widehat C_s)
\le
S_{\max}(s)
-----------

I_{\rm rec}^{\varepsilon}(C_t)
+
B_{\rm hid}^{\max}
+
I_{\rm pre}
+
\Delta_{\rm approx}.
]

[
\Box
]

This is the central theorem in its certified form.

---

# Theorem L8.29 — Fake-history trial dominance theorem

## Statement

Let (\mathcal B) be a family of fake ancestries. Suppose every fake ancestry has missing provenance

[
F_{\rm fake}(B\to C_t)\ge f.
]

Assume each fake trial succeeds with probability at most (2^{-f}), and that the total number of fake trials is bounded by

[
N_{\rm trial}.
]

Then

[
\boxed{
P(\exists\ \text{accepted fake ancestry})
\le
N_{\rm trial}2^{-f}.
}
]

If

[
f>\log_2 N_{\rm trial}+\gamma,
]

then

[
\boxed{
P(\exists\ \text{accepted fake ancestry})
<
2^{-\gamma}.
}
]

If redundant agreement contributes an additional fake-agreement factor (p_\star^{m-1}), then

[
\boxed{
P(\exists\ \text{accepted redundant fake ancestry})
\le
N_{\rm trial}2^{-f}p_\star^{m-1}.
}
]

## Proof

Let (E_j) be the event that fake trial (j) lands on the present record normal form. By assumption,

[
P(E_j)\le 2^{-f}.
]

By the union bound,

[
P\left(\bigcup_{j=1}^{N_{\rm trial}}E_j\right)
\le
\sum_{j=1}^{N_{\rm trial}}P(E_j)
\le
N_{\rm trial}2^{-f}.
]

If

[
f>\log_2N_{\rm trial}+\gamma,
]

then

[
N_{\rm trial}2^{-f}<2^{-\gamma}.
]

The redundant-record version multiplies each fake-trial success probability by the fake-agreement probability from Theorem L8.22.

[
\Box
]

This is the exact Boltzmann-record condition:

[
\boxed{
\text{missing provenance must beat trial count.}
}
]

---

# Theorem L8.30 — De Sitter-only trial-count no-go

## Statement

Suppose the only trial-count information available is the crude finite-capacity bound

[
N_{\rm trial}\le 2^{N_{\rm scr}^{\rm bits}},
]

and suppose the fake deficit obeys

[
f\le N_{\rm scr}^{\rm bits}.
]

Then the union-bound method alone cannot prove fake-history suppression.

## Proof

The fake-history union bound gives

[
P(\exists\ \text{fake})
\le
N_{\rm trial}2^{-f}.
]

Using the crude capacity bound,

[
N_{\rm trial}2^{-f}
\le
2^{N_{\rm scr}^{\rm bits}-f}.
]

If

[
f\le N_{\rm scr}^{\rm bits},
]

then

[
2^{N_{\rm scr}^{\rm bits}-f}\ge 1.
]

A probability upper bound greater than or equal to (1) gives no suppression.

[
\Box
]

This is important. Lane 8 cannot rely only on “the static patch is finite.” To suppress Boltzmann records, it needs a sharper (N_{\rm trial}) certificate: finite time horizon, observer-template restriction, dynamical measure, recurrence cutoff, or selector-cost domination.

---

# Theorem L8.31 — Observer-template trial-count certificate

## Statement

Let fake observers or fake record-normal-forms be restricted to a template class (\mathcal T). Suppose:

1. each template has description length at most (K_{\mathcal T}) bits;
2. each trial requires at least (\tau_{\min}) branch time;
3. the branch time interval under consideration has length (T);
4. at most one independent fake draw occurs per template per (\tau_{\min}).

Then

[
\boxed{
N_{\rm trial}
\le
2^{K_{\mathcal T}}\left\lceil \frac{T}{\tau_{\min}}\right\rceil.
}
]

Thus fake records are suppressed whenever

[
\boxed{
F_{\rm fake}

>

K_{\mathcal T}
+
\log_2\left\lceil \frac{T}{\tau_{\min}}\right\rceil
+
\gamma.
}
]

## Proof

There are at most (2^{K_{\mathcal T}}) binary descriptions of length at most (K_{\mathcal T}), up to a fixed prefix-coding constant. Each template can generate at most

[
\left\lceil \frac{T}{\tau_{\min}}\right\rceil
]

independent trials in time (T). Therefore

[
N_{\rm trial}
\le
2^{K_{\mathcal T}}
\left\lceil \frac{T}{\tau_{\min}}\right\rceil.
]

Apply Theorem L8.29:

[
P(\exists\ \text{fake})
\le
N_{\rm trial}2^{-F_{\rm fake}}.
]

If

[
F_{\rm fake}

>

K_{\mathcal T}
+
\log_2\left\lceil \frac{T}{\tau_{\min}}\right\rceil
+
\gamma,
]

then

[
P(\exists\ \text{fake})<2^{-\gamma}.
]

[
\Box
]

This theorem states the missing Boltzmann-record certificate in usable form.

---

# Theorem L8.32 — Arrow refinement-stability theorem

## Statement

Let (r\preceq s) be two cutoffs, with restriction map

[
\rho_{sr}:Q_s\to Q_r.
]

Let (\mathcal J_s,\mathcal J_r) be the arrow functionals at the two cutoffs. Suppose

[
\left|
\mathcal J_r(\rho_{sr}(A))
--------------------------

## \mathcal J_s(A)

c_{sr}
\right|
\le
\delta_{sr}
]

for every fine ancestry (A), where (c_{sr}) is independent of (A).

Then the restricted fine minimizer is a (2\delta_{sr})-minimizer of the coarse problem:

[
\boxed{
\mathcal J_r(\rho_{sr}(\widehat C^s))
\le
\mathcal J_r(B_r)+2\delta_{sr}
}
]

for every coarse competitor (B_r) that has a fine lift.

If the coarse minimizer gap is

[
\gamma_r>2\delta_{sr},
]

then

[
\boxed{
\rho_{sr}(\widehat C^s)=\widehat C^r.
}
]

## Proof

Let (\widehat C^s) minimize (\mathcal J_s). Let (B_r) be a coarse competitor, and choose a fine lift (B_s) with

[
\rho_{sr}(B_s)=B_r.
]

Because (\widehat C^s) is a fine minimizer,

[
\mathcal J_s(\widehat C^s)\le \mathcal J_s(B_s).
]

Approximate naturality gives

[
\mathcal J_r(\rho_{sr}(\widehat C^s))
\le
\mathcal J_s(\widehat C^s)+c_{sr}+\delta_{sr},
]

and

[
\mathcal J_s(B_s)+c_{sr}
\le
\mathcal J_r(B_r)+\delta_{sr}.
]

Combining,

[
\mathcal J_r(\rho_{sr}(\widehat C^s))
\le
\mathcal J_r(B_r)+2\delta_{sr}.
]

If the coarse minimizer gap exceeds (2\delta_{sr}), then no non-minimizer can lie within (2\delta_{sr}) of the coarse optimum. Hence the restricted fine minimizer must be the coarse minimizer.

[
\Box
]

This is the arrow-specific version of the OPH refinement/coarse-graining theorem surface, where exact naturality gives zero defect and approximate coarse-graining gives declared errors. 

---

# Theorem L8.33 — Approximate record/readout stability theorem

## Statement

Let exact record projectors (\widehat P_a) define the reference central record algebra, and let practical projectors (P_a) satisfy

[
|P_a-\widehat P_a|\le \delta_{\rm rec}.
]

Let the actual accessible state (\widetilde\rho) be trace-distance close to the reference state (\rho):

[
|\widetilde\rho-\rho|_1\le \varepsilon.
]

Then for every record event (a),

[
\boxed{
\left|
\operatorname{Tr}(\widetilde\rho P_a)
-------------------------------------

\operatorname{Tr}(\rho\widehat P_a)
\right|
\le
\varepsilon+\delta_{\rm rec}.
}
]

Consequently, any payload lower bound degrades only by the finite-alphabet continuity penalty induced by (\varepsilon+\delta_{\rm rec}).

## Proof

Compute

[
\left|
\operatorname{Tr}(\widetilde\rho P_a)
-------------------------------------

\operatorname{Tr}(\rho\widehat P_a)
\right|
]

[
\le
\left|
\operatorname{Tr}((\widetilde\rho-\rho)P_a)
\right|
+
\left|
\operatorname{Tr}(\rho(P_a-\widehat P_a))
\right|.
]

The first term is bounded by

[
|\widetilde\rho-\rho|_1|P_a|\le \varepsilon,
]

because projectors have operator norm at most (1). The second term is bounded by

[
|\rho|*1|P_a-\widehat P_a|\le \delta*{\rm rec}.
]

Therefore

[
\left|
\operatorname{Tr}(\widetilde\rho P_a)
-------------------------------------

\operatorname{Tr}(\rho\widehat P_a)
\right|
\le
\varepsilon+\delta_{\rm rec}.
]

The OPH record-stability surface uses this same kind of ((\varepsilon,\delta_{\rm rec})) control for approximate records. 

[
\Box
]

---

# Theorem L8.34 — Gravitational unused-capacity theorem

## Statement

On the de Sitter screen-capacity branch, write

[
N_{\rm scr}
===========

S_{\rm spent}(t_0)
+
N_{\rm unused}(t_0).
]

Let the later record family require gravitational record-writing resource

[
I_{\rm grav}
============

## I_{\rm rec}^{\varepsilon}

## B_{\rm nongrav}^{\max}

## I_{\rm pre}

\Delta_{\rm approx}.
]

Then every faithful early ancestry satisfies

[
\boxed{
N_{\rm unused}(t_0)\ge I_{\rm grav}.
}
]

Equivalently,

[
\boxed{
S_{\rm spent}(t_0)
\le
N_{\rm scr}-I_{\rm grav}.
}
]

## Proof

After non-gravitational hidden export, pre-existing provenance, and approximation slack are subtracted, the remaining record-writing resource required is

[
I_{\rm grav}
============

## I_{\rm rec}^{\varepsilon}

## B_{\rm nongrav}^{\max}

## I_{\rm pre}

\Delta_{\rm approx}.
]

On the gravitational branch, the available gravitational resource is unused screen capacity:

[
N_{\rm grav}(t_0)=N_{\rm unused}(t_0).
]

Faithful record generation requires

[
N_{\rm grav}(t_0)\ge I_{\rm grav}.
]

Therefore

[
N_{\rm unused}(t_0)\ge I_{\rm grav}.
]

Since

[
N_{\rm scr}=S_{\rm spent}+N_{\rm unused},
]

we obtain

[
S_{\rm spent}(t_0)\le N_{\rm scr}-I_{\rm grav}.
]

[
\Box
]

This is the gravitational OPH Past Hypothesis in capacity form:

[
\boxed{
\text{early smoothness means unused gravitational record capacity.}
}
]

---

# Final closure theorem

## Theorem L8.35 — Conditional closure of the OPH cosmological arrow

Let (C_t) be a present cosmological checkpoint. Suppose there is a Lane 8 certificate

[
\mathfrak C_{\rm arrow}
=======================

\left(
I_0,,
B_0,,
P_0,,
A_0,,
f,,
N_{\rm trial},,
\mu,,
\delta_{\rm ref}
\right)
]

such that:

[
I_{\rm rec}^{\varepsilon}(C_t)\ge I_0,
]

[
B_{\rm hid}^{\max}\le B_0,
]

[
I_{\rm pre}\le P_0,
]

[
\Delta_{\rm approx}\le A_0,
]

[
F_{\rm fake}\ge f
\quad
\text{for all nonfaithful fake ancestries},
]

[
f>\log_2N_{\rm trial}+\gamma,
]

and the selector dominance margin satisfies

[
\mu>0.
]

Then, except with fake-history probability at most (2^{-\gamma}), the selected OPH ancestry satisfies

[
\boxed{
S_{\rm OF}(\widehat C_s)
\le
S_{\max}(s)
-----------

I_0
+
B_0
+
P_0
+
A_0.
}
]

If

[
I_0>B_0+P_0+A_0+\Gamma,
]

then

[
\boxed{
S_{\rm OF}(\widehat C_s)\le S_{\max}(s)-\Gamma.
}
]

On the gravitational branch,

[
\boxed{
S_{\rm spent}(t_0)
\le
N_{\rm scr}
-----------

\left(
I_0
---

## B_{\rm nongrav}^{\max}

## P_0

A_0
\right).
}
]

## Proof

By Theorem L8.21, the present record family has certified payload at least (I_0).

By Theorems L8.23–L8.25, hidden provenance and hidden export either do no observer-facing work or are charged against (B_0) and (P_0).

By Theorem L8.27, the positive selector margin (\mu) makes the selected ancestry faithful.

By Theorem L8.28, every faithful selected ancestry satisfies

[
S_{\rm OF}(\widehat C_s)
\le
S_{\max}(s)
-----------

I_{\rm rec}^{\varepsilon}(C_t)
+
B_{\rm hid}^{\max}
+
I_{\rm pre}
+
\Delta_{\rm approx}.
]

Using the certificate inequalities,

[
S_{\rm OF}(\widehat C_s)
\le
S_{\max}(s)
-----------

I_0
+
B_0
+
P_0
+
A_0.
]

If

[
I_0>B_0+P_0+A_0+\Gamma,
]

then

[
S_{\rm OF}(\widehat C_s)\le S_{\max}(s)-\Gamma.
]

By Theorem L8.29,

[
P(\exists\ \text{accepted fake})
\le
N_{\rm trial}2^{-f}<2^{-\gamma}.
]

The gravitational version follows from Theorem L8.34.

[
\Box
]

# What is now closed

The formal Lane 8 result is now:

[
\boxed{
\textbf{If a present cosmological record certificate exists with }
I_0>B_0+P_0+A_0,
\textbf{ then OPH selects a low-entropy past.}
}
]

More explicitly:

[
\boxed{
\text{present records}
+
\text{bounded hidden export}
+
\text{faithful selector dominance}
+
\text{fake-trial suppression}
\Rightarrow
\text{low-entropy selected ancestry}.
}
]

The remaining non-formal work is no longer “find more proof structure.” It is to fill the certificate values:

[
I_0,\quad
B_0,\quad
P_0,\quad
A_0,\quad
f,\quad
N_{\rm trial},\quad
\mu,\quad
\delta_{\rm ref}.
]

The decisive inequality is:

[
\boxed{
I_0

>

B_0+P_0+A_0+\Gamma.
}
]

The decisive fake-history inequality is:

[
\boxed{
f>\log_2N_{\rm trial}+\gamma.
}
]

So Lane 8 is now theorem-closed in conditional form. The only way to defeat it is to produce a high-entropy ancestry with

[
F_{\rm fake}=0,\qquad
B_{\rm hid}\le B_0,\qquad
\Phi_{\rm cont}=0,
]

and lower (\mathcal J_{\rm arrow}) than the faithful ancestry, or to show that the required certificate inequalities fail.
