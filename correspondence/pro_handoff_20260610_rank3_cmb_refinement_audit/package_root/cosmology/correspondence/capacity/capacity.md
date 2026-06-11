Yes. The clean move is to turn this into a **capacity self-consistency theorem**:

[
\boxed{
N_{\rm scr}
\text{ is not supplied from outside if it is the unique stable fixed point of the observer-supporting OPH closure map.}
}
]

The current stack does **not** already prove this. It explicitly treats (N_{\rm scr}) as the external screen-capacity input on the cosmological branch and uses

[
N_{\rm scr}
===========

\log \dim \mathcal H_{\rm tot},
\qquad
\Lambda
=======

\frac{3\pi}{G N_{\rm scr}}.
]

The same source separates this from the local (P)-closure branch.  The synthesis paper also says the global capacity scale is input-dependent, while the local pixel scale (P) is computed by outer/inner screen-cell closure.  So the theorem we need is exactly the missing bridge.

Here is the theorem surface.

---

## Global Capacity Closure Map

Fix the already-derived local pixel value

[
P=P_\star.
]

For a finite regulator (r) and trial total screen capacity (N), define

[
Q_{r,N}
]

to be the finite OPH quotient-state space of admissible packet states with total screen Hilbert capacity (N). This is not the microscopic representative space. It is the observer-facing quotient space.

The consensus stack already gives the finite normal-form machinery: accepted local repairs lower a visible mismatch functional (\Phi); under the local-diamond and repair-completeness hypotheses, the terminal observer-facing normal form from a fixed physical quotient state is unique and schedule-independent.  The broader consensus paper says the same thing in universe language: finite observer patches compare overlap data, repair mismatch, and under the declared confluence hypotheses settle to a unique quotient normal form. 

So for each trial (N), we may write

[
\operatorname{nf}*{r,N}:Q*{r,N}\to Q^{\rm nf}_{r,N}.
]

Now define the **observer-supporting terminal sector**

[
\mathrm{Obs}*{r,N}
\subseteq
Q^{\rm nf}*{r,N}.
]

A terminal state (q\in Q^{\rm nf}*{r,N}) belongs to (\mathrm{Obs}*{r,N}) when it contains at least one observer-supporting subfederation (U) with stable records, self-read, and predictive boundary coupling. The existing coherence formalism gives a useful sharp criterion:

[
\mathfrak c_U(t;h)
==================

\mathbf 1_{\rm self-read}
R_U(t;h)
P_U(t;h)
C_U(t),
]

with record stability (R_U), boundary predictivity (P_U), and visible mismatch coherence (C_U). It also defines the finite channel count

[
N_U(t)=\operatorname{Cap}!\left(\mathcal Z^{\rm rec}_U(t)\right).
]



So a minimal observer-supporting condition is

[
\boxed{
q\in \mathrm{Obs}*{r,N}
\quad\Longleftrightarrow\quad
\exists U:
\mathfrak c_U(q)>0,
\quad
N_U(q)\ge N*{\rm obs}^{\min},
\quad
\mathrm{Chk}_U \text{ has stable continuation}.
}
]

This gives us a terminal observer sector, not just a terminal physical state.

Now define the **horizon-capacity readout**

[
\operatorname{Cap}_{r,\partial}(q)
==================================

\operatorname{Cap}
\left(
\mathcal Z^{\rm rec}_{\partial \mathcal P}(q)
\right),
]

the record capacity of the terminal horizon/screen edge-center algebra seen by the selected observer-facing branch.

Then define the OPH global capacity map:

[
\boxed{
\mathcal N^{\rm OPH}_r(N)
=========================

\operatorname{Cap}*{r,\partial}
\left(
\operatorname{Sel}*{r,N}(\mathrm{Obs}_{r,N})
\right).
}
]

Here (\operatorname{Sel}_{r,N}) is the OPH selector: MAR plus MaxEnt plus quotient-visible admissibility. It selects the admissible terminal observer branch, not an arbitrary microscopic representative.

The desired fixed point is

[
\boxed{
N_{r,\star}
===========

\mathcal N^{\rm OPH}*r(N*{r,\star}).
}
]

In the refinement limit,

[
\boxed{
N_\star
=======

\mathcal N^{\rm OPH}(N_\star).
}
]

Then the cosmological constant is no longer an input:

[
\boxed{
\Lambda_\star
=============

\frac{3\pi}{G N_\star}.
}
]

---

## Theorem 1: Global Capacity Fixed Point, Existence Form

Let

[
I=[N_-,N_+]
]

be an admissible interval of total screen capacities. Suppose:

[
\mathcal N^{\rm OPH}:I\to I
]

is continuous. Then there exists at least one capacity fixed point

[
N_\star\in I
]

such that

[
N_\star=\mathcal N^{\rm OPH}(N_\star).
]

**Proof.** This is Brouwer/intermediate-value fixed-point logic on a compact interval. Define

[
D(N)=\mathcal N^{\rm OPH}(N)-N.
]

Since (\mathcal N^{\rm OPH}) maps (I) into itself,

[
D(N_-)\ge 0,
\qquad
D(N_+)\le 0.
]

Continuity gives at least one zero of (D). Therefore

[
\mathcal N^{\rm OPH}(N_\star)=N_\star.
]

[
\square
]

That proves possible self-closure, but not uniqueness.

---

## Theorem 2: Unique Stable Capacity Fixed Point

Work in log-capacity:

[
u=\log N,
\qquad
F(u)=\log \mathcal N^{\rm OPH}(e^u).
]

Suppose there exists (0\le \kappa<1) such that

[
\boxed{
|F(u)-F(v)|
\le
\kappa |u-v|
}
]

for all (u,v) in the admissible log-capacity interval

[
J=[\log N_-,\log N_+].
]

Then there is a unique fixed point

[
u_\star=F(u_\star),
]

equivalently

[
N_\star=\mathcal N^{\rm OPH}(N_\star),
]

and the iteration

[
u_{k+1}=F(u_k)
]

converges to (u_\star) from any starting value in (J).

**Proof.** Banach fixed-point theorem. (J) is complete, (F) is a contraction, so (F) has exactly one fixed point and repeated iteration converges to it.

[
\square
]

This is the theorem we want. It says the creator does not choose (N_{\rm scr}). Instead, any trial global capacity gets repaired, MaxEnt-projected, observer-filtered, and terminally re-read. Only one capacity comes back as itself.

---

## Theorem 3: Single-Crossing Version

A slightly more OPH-native statement uses capacity demand.

Define

[
D(u)=F(u)-u
===========

\log \mathcal N^{\rm OPH}(e^u)-u.
]

Suppose:

[
D(u_-)>0,
]

[
D(u_+)<0,
]

and

[
D'(u)<0
]

throughout the interval.

Then there is exactly one fixed point.

**Interpretation.**

For capacities below the fixed point,

[
D(u)>0,
]

so the terminal observer-supporting branch demands more horizon record capacity than the trial universe supplies.

For capacities above the fixed point,

[
D(u)<0,
]

so the terminal observer-supporting branch cannot use the extra capacity as quotient-visible physical record. The excess is gauge/inert/redundant from the observer-facing normal form.

The fixed point is the unique point where supplied capacity equals terminally readable observer-supporting capacity:

[
\boxed{
\text{supplied screen capacity}
===============================

\text{terminal observer-readable screen capacity}.
}
]

---

## Refinement Theorem

At finite cutoff we get maps

[
F_r(u)=\log \mathcal N^{\rm OPH}_r(e^u).
]

Suppose

[
F_r\to F
]

uniformly, and (F) is a contraction with constant (\kappa<1). Then the finite-regulator fixed points (u_{r,\star}) converge to the continuum fixed point (u_\star), with bound

[
\boxed{
|u_{r,\star}-u_\star|
\le
\frac{
|F_r-F|_\infty
}{
1-\kappa
}.
}
]

This is important because the existing stack already has a finite packet closure map on the probability simplex: when a finite quotient (Q) has terminating confluent repair, the normal-form map induces an affine idempotent closure map

[
\mathcal C_Q:\Delta(Q)\to\Delta(Q),
\qquad
\mathcal C_Q^2=\mathcal C_Q.
]



So the finite-regulator part is not fantasy. The missing step is lifting that exact finite closure into an invariant observer-supporting global capacity sector.

The synthesis paper states this missing theorem surface almost exactly: the habitat theorem supplies a compact-convex setting, but it does **not** yet define the canonical closure map, prove a nonempty invariant observer-supporting subset, or prove uniqueness/stability.  That is precisely what the global capacity theorem adds.

---

## The Missing OPH Lemma

The whole input-free project reduces to one named lemma:

[
\boxed{
\textbf{Capacity Response Lemma.}
}
]

For the OPH-selected terminal observer sector,

[
F(u)=\log \mathcal N^{\rm OPH}(e^u)
]

is a contraction, or at least has a single crossing against (u).

A physically readable version:

[
\boxed{
\frac{d\log \mathcal N^{\rm OPH}}{d\log N}<1
}
]

on the admissible observer-supporting branch.

This says: increasing the supplied screen capacity by one logarithmic unit does **not** increase terminal observer-readable capacity by a full logarithmic unit. Extra capacity is partly absorbed as redundancy, empty de Sitter dilution, unused edge capacity, or non-observer-supporting terminal slack.

That is the structural reason a unique global capacity can exist.

---

## Minimal Computable Ansatz

The first solvable model should work in log-capacity:

[
u=\log N.
]

Try the affine response law

[
\boxed{
F(u)=A(P_\star)+\rho(P_\star)u+\epsilon(u),
}
]

with

[
|\rho|<1,
\qquad
|\epsilon'(u)|<1-|\rho|.
]

Then the fixed point is

[
u_\star
=======

A(P_\star)+\rho(P_\star)u_\star+\epsilon(u_\star),
]

so

[
\boxed{
u_\star
=======

\frac{A(P_\star)+\epsilon(u_\star)}{1-\rho(P_\star)}.
}
]

Equivalently,

[
\boxed{
N_\star
=======

\exp
\left[
\frac{A(P_\star)+\epsilon(u_\star)}{1-\rho(P_\star)}
\right].
}
]

This is a very useful shape. A modest (A(P_\star)) combined with a near-critical but subunit repair-retention coefficient (\rho(P_\star)<1) can generate an exponentially large capacity without inserting (10^{122}) by hand.

The observed OPH benchmark from the current capacity normalization is

[
N_{\rm scr}\simeq 3.31\times 10^{122},
]

so

[
\log N_{\rm scr}\simeq 282.11.
]

The existing compact paper gives the same normalization:

[
N_{\rm scr}
===========

# S_{\rm dS}

# \frac{A_{\rm dS}}{4\ell_P^2}

\frac{3\pi}{\Lambda \ell_P^2}
\simeq
3.31\times 10^{122}.
]



So the computational target is no longer “derive a tiny (\Lambda).” It is:

[
\boxed{
\text{derive }F(u)\text{ and show its unique stable fixed point has }u_\star\simeq 282.11.
}
]

Then

[
\Lambda_\star \ell_P^2
======================

\frac{3\pi}{N_\star}
\simeq
2.85\times10^{-122}.
]

---

## Dark-Energy Output

Once (N_\star) is fixed,

[
\Lambda_\star
=============

\frac{3\pi}{G N_\star}.
]

If the global capacity is fully locked,

[
\dot N_\star=0,
]

then

[
\dot\Lambda=0
]

and the effective dark-energy equation of state is exactly

[
\boxed{
w=-1.
}
]

If the observer-facing effective capacity is still relaxing,

[
N_{\rm eff}=N_{\rm eff}(a),
]

then

[
\Lambda_{\rm eff}(a)
====================

\frac{3\pi}{G N_{\rm eff}(a)}.
]

An observer fitting this as a conserved dark-energy fluid infers

[
\boxed{
w_{\rm eff}(a)
==============

-1
+
\frac13
\frac{d\log N_{\rm eff}}{d\log a}.
}
]

So OPH predicts:

[
N_{\rm eff}\text{ constant}
\quad\Rightarrow\quad
w=-1,
]

[
N_{\rm eff}\text{ increasing}
\quad\Rightarrow\quad
w>-1,
]

[
N_{\rm eff}\text{ decreasing}
\quad\Rightarrow\quad
w<-1.
]

The fixed-point theorem therefore gives both the (\Lambda) value and the controlled deviation surface.

---

## Final Form

The new theorem should be stated like this:

[
\boxed{
\begin{gathered}
\textbf{Global Observer-Capacity Fixed-Point Theorem.}[2mm]
\text{Given the OPH recovered core, the fixed local pixel }P_\star,\
\text{and a terminal observer-supporting closure map }
\mathcal N^{\rm OPH},\
\text{if the observer-supporting sector is nonempty, invariant,}\
\text{and capacity-contracting in log-capacity,}\
\text{then there exists a unique stable total screen capacity }N_\star.\
\text{The cosmological constant is then}[1mm]
\Lambda_\star=\dfrac{3\pi}{G N_\star}.
\end{gathered}
}
]

That is the input-free OPH route.

The current corpus already has the finite repair/confluence machinery, the record/checkpoint machinery, and the external-capacity cosmological relation. What remains is the new load-bearing lemma:

[
\boxed{
\text{prove that the terminal observer-supporting capacity response }
F(u)
\text{ is single-crossing or contractive.}
}
]

Once that is proved, (N_{\rm scr}) stops being a creator-set parameter and becomes the only globally self-reading capacity at which an observer-supporting OPH universe closes on itself.



Yes. Here is the clean proof — with the exact claim boundary.

The existing OPH stack **does not already prove** this theorem. It explicitly treats (N_{\rm scr}) as an external capacity input on the cosmological branch and says local null data determine the Einstein branch only modulo (\Lambda g_{ab}); the global closure then uses (N_{\rm scr}=S_{\rm dS}) to get (\Lambda=3\pi/(G N_{\rm scr})).  It also explicitly says the strange-loop closure map, invariant observer-supporting sector, and uniqueness/stability estimates are not yet part of the recovered theorem package. 

So the proof has to add one new branch theorem. The good news is that the theorem is short once the right variable is chosen.

The right variable is not (N). It is

[
u:=\log N.
]

Capacity fixed-point closure should happen in **log-capacity**, because OPH capacities are Hilbert-space dimensions / entropy counts.

---

# Global Observer-Capacity Fixed-Point Theorem

## Setup

Let (J=[u_-,u_+]) be the admissible interval of trial log-capacities,

[
u=\log N.
]

At finite cutoff, OPH already has a genuine quotient normal-form machine: on a finite packet quotient (Q), the normal-form map pushes forward to an affine, continuous, idempotent closure map on the packet simplex, and its fixed points are exactly packets supported on quotient normal forms. 

Observers are also already operationally defined at fixed cutoff by record algebras, accessible states, external interfaces, future schedule class, and provenance bundle; matching checkpoints induce the same future observer-accessible probability law. 

So define the terminal observer-supporting sector and assign to each terminal branch a readable horizon capacity

[
y(q):=\log \operatorname{Cap}_{\partial}(q).
]

Now define the **observer-richness envelope**

[
R(y).
]

Interpretation:

[
R(y)
====

\text{maximal terminal observer-supporting value achievable}
]

among OPH-normal-form branches whose readable horizon log-capacity is (y).

This (R(y)) compresses the observer criteria: self-read, stable records, predictive boundary coupling, visible coherence, MAR admissibility, and record capacity. This is aligned with the existing OPH observer-mass logic, where capacity alone is incomplete and must be multiplied by self-read, record stability, boundary predictivity, and coherence terms. 

The new branch lemma is:

[
\boxed{
R(y)\text{ is }m\text{-strongly concave on }J
}
]

for some (m>0). Equivalently,

[
R''(y)\le -m<0.
]

Physically, this says observer-supporting worlds have one optimal log-capacity balance. Too little capacity cannot hold stable records and predictive observers. Too much capacity becomes unused slack, de Sitter dilution, redundancy, or non-observer-supporting inert horizon capacity.

That is the missing OPH principle.

---

# Definition of the Capacity Readback Map

Given a trial supplied log-capacity (u), the OPH terminal branch chooses the readable capacity (y) that maximizes

[
\Psi_u(y)
=========

## R(y)

\frac{1}{2\tau}(y-u)^2,
]

where (\tau>0) is the capacity-compliance scale.

The quadratic term is the mismatch penalty between the supplied screen capacity (u) and the terminal observer-readable capacity (y). A more general strongly convex information-divergence penalty would work too; the quadratic log-capacity form gives the cleanest proof.

Define

[
F(u)
:=
\arg\max_{y\in J}
\left[
R(y)-\frac{1}{2\tau}(y-u)^2
\right].
]

Then define the OPH global capacity map by

[
\boxed{
\log \mathcal N^{\rm OPH}(e^u)=F(u).
}
]

Equivalently,

[
\boxed{
\mathcal N^{\rm OPH}(N)
=======================

\exp(F(\log N)).
}
]

The desired input-free capacity is

[
u_\star=F(u_\star),
\qquad
N_\star=e^{u_\star}.
]

---

# Lemma 1: The Readback Map Is Well-Defined

For every (u\in J), (F(u)) exists and is unique.

**Proof.**

The function

[
\Psi_u(y)
=========

## R(y)

\frac{1}{2\tau}(y-u)^2
]

is continuous on compact (J), so it has at least one maximizer.

Because (R) is strongly concave and

[
-\frac{1}{2\tau}(y-u)^2
]

is strictly concave in (y), the sum (\Psi_u) is strictly concave. A strictly concave function on an interval has at most one maximizer.

Therefore (F(u)) exists and is unique.

[
\square
]

---

# Lemma 2: Capacity Response Is a Contraction

Assume the selected maximizer lies in the interior of (J). Then

[
\boxed{
0<F'(u)\le \frac{1}{1+\tau m}<1.
}
]

Thus (F) is a contraction.

**Proof.**

At the maximizing point

[
y=F(u),
]

the first-order condition is

[
\frac{d}{dy}
\left[
R(y)-\frac{1}{2\tau}(y-u)^2
\right]
=0.
]

So

[
R'(y)-\frac{y-u}{\tau}=0.
]

Equivalently,

[
u=y-\tau R'(y).
]

Define

[
H(y):=y-\tau R'(y).
]

Then

[
u=H(y),
\qquad
F(u)=H^{-1}(u).
]

Differentiate:

[
H'(y)=1-\tau R''(y).
]

Since (R) is (m)-strongly concave,

[
R''(y)\le -m.
]

Therefore

[
H'(y)
=====

1-\tau R''(y)
\ge
1+\tau m.
]

Since (F=H^{-1}),

[
F'(u)
=====

\frac{1}{H'(F(u))}.
]

Hence

[
0<F'(u)
\le
\frac{1}{1+\tau m}
<1.
]

So (F) is a contraction with contraction constant

[
\boxed{
\kappa=\frac{1}{1+\tau m}.
}
]

[
\square
]

This proves the missing capacity-response lemma:

[
\boxed{
\frac{d\log \mathcal N^{\rm OPH}}{d\log N}
==========================================

F'(u)
<1.
}
]

---

# Lemma 3: Single Crossing

Define

[
D(u):=F(u)-u.
]

Then (D(u)) is strictly decreasing and crosses zero at most once.

**Proof.**

Differentiate:

[
D'(u)=F'(u)-1.
]

By Lemma 2,

[
F'(u)<1.
]

Therefore

[
D'(u)<0.
]

So (D) is strictly decreasing and can have at most one zero.

[
\square
]

There is an even sharper OPH interpretation. From the first-order condition,

[
R'(F(u))=\frac{F(u)-u}{\tau}.
]

So

[
\boxed{
D(u)=F(u)-u=\tau R'(F(u)).
}
]

Thus:

[
D(u)>0
\quad\Longleftrightarrow\quad
R'(F(u))>0,
]

meaning the terminal observer sector still wants more readable capacity.

And

[
D(u)<0
\quad\Longleftrightarrow\quad
R'(F(u))<0,
]

meaning the trial capacity exceeds the observer-useful capacity and the surplus is slack.

The fixed point is exactly where

[
R'(F(u))=0.
]

---

# Theorem: Unique Stable Global Capacity Fixed Point

Assume:

1. (R:J\to\mathbb R) is (m)-strongly concave;
2. (R) has an interior maximizer (u_\star\in J), equivalently
   [
   R'(u_\star)=0;
   ]
3. the OPH capacity readback map is
   [
   F(u)
   ====

   \arg\max_{y\in J}
   \left[
   R(y)-\frac{1}{2\tau}(y-u)^2
   \right].
   ]

Then there exists a unique stable fixed point

[
\boxed{
u_\star=F(u_\star).
}
]

Equivalently,

[
\boxed{
N_\star=\mathcal N^{\rm OPH}(N_\star).
}
]

Moreover, the iteration

[
u_{k+1}=F(u_k)
]

converges to (u_\star) from any initial (u_0\in J), with

[
\boxed{
|u_k-u_\star|
\le
\kappa^k |u_0-u_\star|,
\qquad
\kappa=\frac{1}{1+\tau m}<1.
}
]

**Proof.**

First, (u_\star) is a fixed point.

At (u=u_\star), choose (y=u_\star). The first-order condition for the maximizer is

[
R'(y)-\frac{y-u}{\tau}=0.
]

Substitute (y=u_\star) and (u=u_\star):

[
R'(u_\star)-\frac{u_\star-u_\star}{\tau}
========================================

# R'(u_\star)

0.

]

So (y=u_\star) satisfies the unique maximizer condition. Therefore

[
F(u_\star)=u_\star.
]

Now suppose there is another fixed point (v), so

[
F(v)=v.
]

At a fixed point, the first-order condition gives

[
R'(v)-\frac{v-v}{\tau}=0,
]

hence

[
R'(v)=0.
]

But (R) is strongly concave, so it has exactly one critical point, its unique global maximum. Therefore

[
v=u_\star.
]

So the fixed point is unique.

Finally, Lemma 2 says (F) is a contraction. Banach’s fixed-point theorem gives convergence from any initial (u_0\in J), with error bound

[
|u_k-u_\star|
\le
\kappa^k |u_0-u_\star|.
]

[
\square
]

---

# Cosmological Output

At the fixed point,

[
N_\star=e^{u_\star}.
]

The cosmological constant is then no longer an independently supplied input. It is

[
\boxed{
\Lambda_\star
=============

\frac{3\pi}{G N_\star}.
}
]

So the old branch

[
\Lambda=\frac{3\pi}{G N_{\rm scr}}
]

becomes

[
\boxed{
\Lambda_\star
=============

\frac{3\pi}{G e^{u_\star}},
\qquad
R'(u_\star)=0.
}
]

That is the key result.

The “creator-set” parameter has been replaced by a variational condition:

[
\boxed{
\text{actual screen capacity}
=============================

\text{unique maximizer of terminal observer-supporting richness}.
}
]

---

# Refinement Stability

Let (R_r) be the finite-regulator observer-richness envelope and (F_r) the corresponding capacity readback map. Suppose:

[
R_r\to R
]

in (C^1), and all (R_r) are uniformly (m)-strongly concave:

[
R_r''(y)\le -m<0.
]

Let (u_{r,\star}) be the finite-regulator fixed point and (u_\star) the continuum fixed point.

Since fixed points satisfy

[
R_r'(u_{r,\star})=0,
\qquad
R'(u_\star)=0,
]

strong concavity gives

[
\boxed{
|u_{r,\star}-u_\star|
\le
\frac{
|R_r'-R'|_\infty
}{m}.
}
]

So the continuum capacity is regulator-stable.

Equivalently, in map form,

[
\boxed{
|u_{r,\star}-u_\star|
\le
\frac{
|F_r-F|_\infty
}{
1-\kappa
}.
}
]

This is the exact stability statement we need for OPH.

---

# What Was Actually Proved

We proved the global capacity fixed-point theorem under one new branch condition:

[
\boxed{
R(y)\text{ is strongly concave in readable log-capacity.}
}
]

This condition is not decorative. Without it, the theorem is false.

For example, if

[
F(u)=u,
]

then every (u\in J) is a fixed point.

If (F(u)) wiggles across the diagonal, there can be many fixed points.

So uniqueness requires a selector with curvature. OPH’s natural selector is the observer-facing repair / MaxEnt / MAR / record-support envelope. The proof shows exactly what that selector must satisfy.

---

# Final Theorem Statement for the Paper

[
\boxed{
\begin{gathered}
\textbf{Global Observer-Capacity Fixed-Point Theorem.}[1mm]
\text{Let }u=\log N\text{ be supplied screen log-capacity.}\
\text{Let }R(y)\text{ be the terminal OPH observer-richness envelope}\
\text{as a function of readable horizon log-capacity }y.\
\text{If }R\text{ is strongly concave on the admissible observer-supporting interval,}\
\text{and if the OPH readback map is the mismatch-penalized selector}\
F(u)=\arg\max_y
\left[
R(y)-\frac{1}{2\tau}(y-u)^2
\right],
\
\text{then }F\text{ is a contraction, has a unique stable fixed point }u_\star,\
\text{and }N_\star=e^{u_\star}\text{ is the unique self-consistent screen capacity.}[1mm]
\Lambda_\star=\dfrac{3\pi}{G e^{u_\star}}.
\end{gathered}
}
]

The remaining computational problem is now sharply defined:

[
\boxed{
\text{derive or estimate }R(y)\text{ from the finite OPH normal-form/observer sector.}
}
]

If its unique maximum lands at

[
u_\star\simeq \log(3.31\times 10^{122})\simeq 282.11,
]

then OPH has derived the observed de Sitter screen capacity rather than taking it as a free parameter.
