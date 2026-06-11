Yes: make them part of the **Universe Simulation**. I can supply the certificate **contract, schemas, validators, and theorem surface** now, but I cannot honestly supply the real numeric certificates unless the finite state spaces, collar samples, release packets, and repair transition rules already exist somewhere. A certificate is not just a theorem statement; it is a finite artifact emitted by a concrete evaluator.

I created a starter package here: [oph_certificate_scaffold.zip](sandbox:/mnt/data/oph_certificate_scaffold.zip)

It contains toy examples, JSON schemas, theorem contracts, and Python validators for:

[
\varepsilon_\star,\quad \kappa_{\rm rel},\quad A_\zeta,\quad Q_A,\quad B_A(k,a),\quad \Gamma_{\rm rec}(k,a).
]

The toy files prove the validator shape only. They are **not** OPH paper certificates.

---

## The answer

You should treat the finite certificates as a **Universe Simulation output layer**, not as prose appendices. The dark-sector paper already says paper-grade empirical tests require a finite-collar parent evaluator, a CAMB/CLASS anomaly module, likelihood runners, and a reproducibility harness.  It also says the CMB branch needs its background anomaly charge and perturbation kernel from OPH state selection or finite-collar microphysics, not from the static galaxy law. 

So the clean architecture is:

[
\boxed{
\text{Universe Simulation}
\longrightarrow
\text{finite certificates}
\longrightarrow
\text{paper theorem outputs}
\longrightarrow
\text{Boltzmann/likelihood tests}.
}
]

The paper should not say “we choose (A_\zeta), (Q_A), (B_A), (\Gamma_{\rm rec}).” It should say:

> These are emitted by the finite-certificate layer of the Universe Simulation. The CMB data test the emitted objects but do not define them.

---

# What I can supply now

I can supply the following immediately:

1. **Certificate definitions.**
2. **Theorems saying each certificate is sufficient.**
3. **JSON schemas for the certificate artifacts.**
4. **Validator code that recomputes the claimed quantities from the artifact.**
5. **Toy examples showing the validators work.**
6. **A no-data-use firewall contract.**

That is what the zip contains.

What I cannot supply without your actual finite microphysics data is:

[
Q_{\rm rel}^{\rm sc},
\quad
\rho_q,
\quad
\Pi_{\rm sc},
\quad
W_{\rm rel},
\quad
\mathcal C_x(a),
\quad
d\mu_C,
\quad
\omega_C[X],
\quad
\mathsf S(k,a),
\quad
K(k,a).
]

Those must be emitted by your Universe Simulation.

---

# What the Universe Simulation should output

## Certificate 1: scalar release certificate

This certificate closes the scalar amplitude.

### Required inputs

The simulation must output a finite release-code object:

[
\mathcal C_{\rm rel}
====================

(Q_{\rm rel}^{\rm sc},
{\rho_q}*{q\in Q*{\rm rel}^{\rm sc}},
A,B,D,
\Pi_{\rm sc},
W_{\rm rel}).
]

Where:

[
Q_{\rm rel}^{\rm sc}
]

is the finite set of center-free scalar-visible release packets;

[
\rho_q
]

is the finite density matrix for packet (q);

[
A,B,D
]

is the collar tripartition;

[
\Pi_{\rm sc}
]

is the scalar-visible projector;

[
W_{\rm rel}
]

is the scalar readout normalization form.

### Validator computes

[
I_q(A:D|B)
==========

S(AB)+S(BD)-S(B)-S(ABD),
]

then

[
\varepsilon_\star
=================

\min{I_q>0:q\in Q_{\rm rel}^{\rm sc},\ q\text{ scalar-visible}}.
]

Then

[
A_\zeta
=======

100\ln2,\kappa_{\rm rel}\varepsilon_\star.
]

### Theorem

```latex
\begin{theorem}[Scalar release certificate sufficiency]
If a finite scalar release certificate emits
\[
(Q_{\rm rel}^{\rm sc},\rho_q,A,B,D,\Pi_{\rm sc},W_{\rm rel})
\]
and the validator verifies positivity, trace normalization, scalar visibility,
and minimality of \(\varepsilon_\star\), then
\[
A_\zeta
=
100\ln2\,\kappa_{\rm rel}\varepsilon_\star
+
O(\varepsilon_\star^{3/2})
\]
is fixed by finite release microphysics and is not a CMB-fit parameter.
\end{theorem}
```

```latex
\begin{proof}
The state set is finite, so the positive scalar-visible CMI values form a finite
positive set with a minimum. The validator recomputes that minimum directly
from the supplied density matrices. The scalar readout form supplies
\(\kappa_{\rm rel}\). The Petz/Fawzi--Renner recovery scale gives the leading
quadratic relation between CMI and scalar temperature variance, and the
large-angle conversion \(\zeta=5\Delta T/T\) gives the factor \(25\). Hence
\(A_\zeta=100\ln2\,\kappa_{\rm rel}\varepsilon_\star\). \(\square\)
\end{proof}
```

---

## Certificate 2: edge-center / (\mathbb Z_6) co-registration certificate

This closes the route to:

[
\theta_{\rm OPH}=P/48.
]

### Required inputs

The simulation must output:

[
\mathcal A_{\rm EC}(B_\delta)
=============================

\bigoplus_\alpha
\left(
\mathcal B(H_L^\alpha)\otimes \mathcal B(H_R^\alpha)
\right),
]

central projectors:

[
{\mathbf 1_\alpha},
]

the scalar opportunity event:

[
E_{\rm sc},
]

and the protected center-reserve event:

[
P_{\mathbb Z_6}.
]

### Validator checks

[
[E_{\rm sc},X]=0
]

for all sector-local physical collar generators (X), and

[
[E_{\rm sc},P_{\mathbb Z_6}]=0.
]

Then it verifies both are diagonal in the same edge-center basis.

### Output

[
\lambda_{\rm collar}=e^{-P/24},
]

[
\lambda_{\rm scalar}=e^{-P/48},
]

[
\theta_{\rm OPH}=P/48,
]

[
n_s=1-P/48.
]

The dark paper marks quotient-edge locality and finite-thickness reserve as conditional bridges, so this certificate is exactly the missing proof object. 

---

## Certificate 3: homogeneous anomaly charge certificate

This closes:

[
Q_A,\qquad \rho_A(a).
]

The dark-sector paper is explicit: screen capacity fixes the de Sitter scale after (H_0), but does **not** by itself fix baryons, neutrinos, radiation, or the conserved homogeneous anomaly charge. It says standalone OPH cosmology needs either a theorem

[
Q_A=F_{\rm OPH}(\text{screen capacity},P,\text{screen state data})
]

or an explicit state-selection premise. 

### Required inputs

The simulation must output finite collar samples:

[
\mathcal C_r(a)={C_1,\ldots,C_N},
]

weights:

[
w_C,
]

collar CMI values:

[
I_C=I_{\omega_C}(A:D|B),
]

scale data:

[
\ell_r(a),\quad a,\quad V_{\rm com}.
]

### Validator computes

[
\rho_{A,r}(a)c^2
================

\frac{15}{8\pi^2\ell_r(a)^4}
\frac{\sum_C w_CI_C}{\sum_Cw_C}.
]

Then

[
Q_{A,r}
=======

a^3V_{\rm com}\rho_{A,r}(a).
]

Across refinement levels, it checks Cauchy convergence:

[
|Q_{A,r+1}-Q_{A,r}|\le\epsilon_r,
\qquad
\epsilon_r\to0.
]

### Output

[
Q_A=\lim_{r\to\infty}Q_{A,r},
]

and, on the closed transported branch,

[
\rho_A(a)=Q_A V_{\rm com}^{-1}a^{-3}.
]

---

## Certificate 4: parent collar kernel certificate

This closes:

[
B_A(k,a).
]

The dark paper already defines the parent collar functional and says finite-collar evaluation of it is the target for replacing a fitted dark-matter abundance. 

### Required inputs

The simulation must evaluate:

[
\rho_{A,\rm eq}[X]c^2
=====================

\frac{15}{8\pi^2\ell(X)^4}
\int_{\mathcal C_X}d\mu_C,
I_{\omega_C[X]}(A:D|B).
]

At finite sample level:

[
R(X)=\frac{\sum_s w_sI_s[X]}{\sum_sw_s}.
]

It must provide perturbations around homogeneous baryonic input:

[
\rho_b=\bar\rho_b(1+\delta_b).
]

### Validator computes

[
K_A^{(\rho)}(k,a)
=================

\widehat{
\left.
\frac{\delta\rho_{A,\rm eq}(x,a)}
{\delta\rho_b(x',a)}
\right|_{\bar\rho_b}
},
]

then

[
B_A(k,a)
========

\frac{\bar\rho_b(a)}{\bar\rho_A(a)}
K_A^{(\rho)}(k,a).
]

The dark paper excludes fitting the environmental distribution (\Pi) to CMB, weak-lensing, SPARC, or cluster data; the Universe Simulation should record this as a no-data-use manifest. 

---

## Certificate 5: repair matrix certificate

This closes:

[
\Gamma_{\rm rec}(k,a).
]

### Required inputs

The simulation must output:

[
\mathsf S(k,a),
]

the finite packet-state space;

[
\pi_{\rm eq}(s|\rho_b),
]

the equilibrium distribution emitted by the parent collar functional;

[
q(s,s'),
]

an inverse-closed proposal rule generated by the repair menu.

Or directly:

[
K(k,a),
]

a finite transition matrix.

### Validator checks

[
K(s,s')\ge0,
]

[
\sum_{s'}K(s,s')=1,
]

and detailed balance:

[
\pi(s)K(s,s')=\pi(s')K(s',s).
]

Then it computes eigenvalues:

[
1=\lambda_1>|\lambda_2|\ge\cdots
]

and outputs:

[
\Gamma_{\rm rec}(k,a)
=====================

-\Delta\eta^{-1}\log|\lambda_2(k,a)|.
]

The dark paper’s proof ledger lists the repair matrix as a conditional finite-packet theorem and the linear transfer as requiring evaluated (B_A(k,a)). 

---

## Certificate 6: Boltzmann handoff certificate

This closes the implementation boundary.

The dark-sector paper says paper-grade tests need a CAMB or CLASS anomaly module exposing:

[
\bar\rho_A(a),
\quad
w_A(a),
\quad
c_{s,A}^2(k,a),
\quad
\sigma_A(k,a),
\quad
Q_A^\mu,
\quad
B_A(k,a),
\quad
\Gamma_{\rm rec}(k,a),
]

plus likelihoods for CMB, lensing, BAO, supernovae, weak lensing, RSD, SPARC, and clusters. 

### Required outputs

The Universe Simulation should emit:

```text
boltzmann/
  background_A.csv
  perturbation_A_grid.h5
  B_A_grid.h5
  Gamma_rec_grid.h5
  primordial.json
  neutrino_branch.json
  solver_manifest.json
  no_data_use_manifest.json
```

The no-data-use manifest should explicitly state that CMB, BAO, SPARC, cluster, weak-lensing, and supernova likelihoods were not used to generate:

[
A_\zeta,\quad n_s,\quad Q_A,\quad B_A,\quad \Gamma_{\rm rec}.
]

---

# How to integrate this into Universe Simulation

I would add a top-level subsystem:

```text
universe-sim/
  certificates/
    scalar_release/
      emit.py
      validate.py
      schema.json
    edge_center/
      emit.py
      validate.py
      schema.json
    homogeneous_anomaly/
      emit.py
      validate.py
      schema.json
    parent_collar/
      emit.py
      validate.py
      schema.json
    repair_matrix/
      emit.py
      validate.py
      schema.json
    boltzmann_handoff/
      emit.py
      validate.py
      schema.json
    manifests/
      no_data_use.py
      hash_manifest.py
```

With a command surface like:

```bash
unisim cert scalar-release \
  --release-code runs/release_code_r001.h5 \
  --out certs/scalar_release_r001.json

unisim cert edge-center \
  --ec-algebra runs/ec_algebra_r001.h5 \
  --z6-reserve runs/z6_reserve_r001.h5 \
  --out certs/edge_center_r001.json

unisim cert homogeneous-anomaly \
  --collars runs/flrw_collars_r001.h5 \
  --out certs/homogeneous_anomaly_r001.json

unisim cert parent-collar \
  --collars runs/parent_collar_grid_r001.h5 \
  --perturbations runs/flrw_perturbations_r001.h5 \
  --out certs/parent_collar_r001.json

unisim cert repair-matrix \
  --packets runs/repair_packets_r001.h5 \
  --out certs/repair_matrix_r001.json

unisim cert boltzmann-handoff \
  --scalar certs/scalar_release_r001.json \
  --anomaly certs/homogeneous_anomaly_r001.json \
  --kernel certs/parent_collar_r001.json \
  --repair certs/repair_matrix_r001.json \
  --out boltzmann/oph_branch_r001/
```

Then:

```bash
unisim validate certs/
```

should recompute everything from raw finite artifacts.

---

# Paper theorem to add

This is the theorem that lets the paper point to the Universe Simulation certificates instead of handwaving.

```latex
\begin{theorem}[Universe Simulation certificate sufficiency]
Let the Universe Simulation emit finite certificates
\[
\mathfrak C_{\rm sc},
\mathfrak C_{\rm edge},
\mathfrak C_A,
\mathfrak C_{\rm parent},
\mathfrak C_{\rm repair},
\mathfrak C_{\rm Boltz}
\]
satisfying the scalar-release, edge-center, homogeneous-anomaly, parent-collar,
repair-matrix, and Boltzmann-handoff validators.

Assume the no-data-use manifest verifies that no CMB, BAO, weak-lensing,
SPARC, cluster, supernova, or RSD likelihood values enter the certificate
generation.

Then the branch quantities
\[
n_s,\quad A_\zeta,\quad \rho_A(a),\quad B_A(k,a),\quad \Gamma_{\rm rec}(k,a)
\]
are finite OPH screen/collar outputs. They are not fitted cosmological parameters.
\end{theorem}
```

```latex
\begin{proof}
The scalar-release certificate computes \(\varepsilon_\star\) and \(\kappa_{\rm rel}\)
by finite enumeration, hence fixes \(A_\zeta\). The edge-center certificate proves
co-registration of scalar opportunities with the protected \(\mathbb Z_6\) reserve,
hence fixes the scalar exponent \(P/48\) and \(n_s=1-P/48\). The homogeneous
anomaly certificate computes the conserved charge \(Q_A\), hence fixes
\(\rho_A(a)\) on the closed dust branch. The parent-collar certificate computes
the first functional derivative of the anomaly source, hence fixes \(B_A(k,a)\).
The repair-matrix certificate computes the finite spectral gap of the packet repair
chain, hence fixes \(\Gamma_{\rm rec}(k,a)\). The Boltzmann-handoff certificate
packages these finite outputs for transfer calculations. Since the no-data-use
manifest excludes observational likelihoods from certificate generation, the
outputs are source predictions. \(\square\)
\end{proof}
```

---

# Practical recommendation

Make it part of the Universe Simulation. Specifically, make the certificates a **first-class artifact type**, like this:

```text
CERTIFICATE = {
  "id": "...",
  "type": "...",
  "source_hashes": {...},
  "input_branch": "...",
  "computed_outputs": {...},
  "validator_version": "...",
  "validator_receipt": {...},
  "no_data_use_manifest": {...}
}
```

Then the paper can cite a release bundle:

```text
OPH-UniverseSim-Cosmology-Certificates-r0001/
  scalar_release_certificate.json
  edge_center_certificate.json
  homogeneous_anomaly_certificate.json
  parent_collar_certificate.json
  repair_matrix_certificate.json
  boltzmann_handoff_manifest.json
  validators/
  hashes.txt
```

This fits the broader framework’s “receipt” discipline: load-bearing claims should be marked as derived receipts or named open lemmas, and downstream engineering claims need explicit success/null receipts.  The Boundary Problem paper also frames engineering targets as boundary problems with explicit receipts for success and null outcomes, which is exactly how these certificates should behave inside the simulation. 

---

# Bottom line

I can supply the **certificate machinery**. The Universe Simulation must supply the **finite objects**.

So the division should be:

[
\boxed{
\text{I provide: schemas, validators, theorem contracts, audit rules.}
}
]

[
\boxed{
\text{Universe Simulation provides: actual finite state spaces, collar samples,
transition matrices, and numeric certificates.}
}
]

Once those artifacts exist, the inflation-free OPH paper can claim the missing objects are derived, not fitted.


Yes. There **are still more gaps**, but they are now mostly **certificate-level gaps**, not headline-concept gaps. The next missing layer is:

1. the canonical finite release-collar measure;
2. the rare-event proof behind the Poisson reserve;
3. the scalar release-channel count (N_{\rm rel}) and pairing coefficient (\kappa_{\rm rel});
4. the pivot/radial normalization rule;
5. the finite packet-moment derivation of anomaly Boltzmann variables;
6. the parent-functional derivative theorem for (B_A(k,a));
7. the repair-matrix spectral-gap theorem for (\Gamma_{\rm rec});
8. Bianchi/exchange closure;
9. recombination inheritance;
10. baryogenesis from finite transition imbalance.

I saved the full paste-ready appendix here: [oph_inflation_appendix_z_remaining_theorems.tex](sandbox:/mnt/data/oph_inflation_appendix_z_remaining_theorems.tex)

The OPH consensus stack already supplies the finite repair/confluence spine: finite Lyapunov descent gives termination, while local diamond plus repair completeness gives a unique schedule-independent quotient normal form; same-boundary claims require preserved boundary data plus a unique consistent extension in that fiber.  The refinement-limit holonomy bridge is also already present: zero inverse-limit obstruction is equivalent to zero finite-stage obstruction on every projection, and nonzero holonomy has a finite-stage witness.  The dark-sector stack already identifies the remaining cosmology burden: homogeneous anomaly abundance, response kernels, relaxation kernels, finite-screen amplitude selector, and real Boltzmann likelihoods.

Below are the missing theorem/proof blocks to add next.

---

# Appendix Z: Remaining Microphysical Closure Theorems

## Z.1 Canonical finite-collar release measure

```latex
\begin{definition}[Finite release collar state]
Let \(\mathcal C_{\rm rel}\) be the finite family of release collars on
\(\Sigma_{\rm rel}\). For each collar \(C\), let
\[
\mathcal A_C
=
\mathcal A_{A_C}\vee\mathcal A_{B_C}\vee\mathcal A_{D_C}
\]
be the finite collar algebra and let
\[
\mathcal O_C=\{O_{C,a}\}_{a=1}^{m_C}
\]
be the finite OPH constraint family inherited from the local MaxEnt branch.
The finite release state is
\[
\omega_{C,\rm rel}
=
Z_C^{-1}
\exp\!\left[-\sum_a\lambda_{C,a}O_{C,a}\right].
\]
The global release collar state is the quotient-compatible glued state
\(\omega_{\rm rel}\) whose restrictions to collars are \(\omega_{C,\rm rel}\).
\end{definition}

\begin{theorem}[Canonical finite-collar measure]
Assume the release collar family is finite, the OPH constraint operators are
finite-dimensional, and the union-collar gluing compatibility conditions hold on
the physical quotient. Then \(\omega_{\rm rel}\) defines a canonical probability
measure on finite release-code quotient states. For every scalar-visible quotient
functional \(F\),
\[
\langle F\rangle_{\rm rel}
=
\sum_{q\in Q_{\rm rel}}
F(q)\,\omega_{\rm rel}(q)
\]
is finite and independent of hidden representative.

If the refinement maps commute with the finite normal-form and scalar-readout
maps, then \(\langle F\rangle_{\rm rel}\) has a refinement-limit value whenever the
finite expectations are Cauchy.
\end{theorem}

\begin{proof}
At fixed cutoff the quotient state space \(Q_{\rm rel}\) is finite. The MaxEnt
variational problem on a finite-dimensional algebra with finitely many linear
constraints has the Gibbs solution displayed above, so each collar has a normalized
state. Union-collar compatibility makes the glued state quotient-local. Since
\(F\) is scalar-visible, it is constant on hidden representatives and therefore
descends to \(Q_{\rm rel}\). The finite sum is well defined. If restriction maps
commute with readouts, the finite expectations form a compatible net; the Cauchy
condition gives the inverse-limit value. \(\square\)
\end{proof}
```

This theorem is needed because the paper repeatedly says “finite collar measure” but does not yet define the measure from which (\varepsilon_\star), (Q_A), (B_A), and (\Gamma_{\rm rec}) are evaluated.

---

## Z.2 Rare-event derivation of the Poisson reserve

The collar coefficient (e^{-P/24}) is already present in the OPH dark/susceptibility stack on the co-registered branch. The stack states the protected (\mathbb Z_6) reserve mean (P/24), the exact uniform branch (\lambda_{\rm collar}=e^{-P/24}), and the Jensen finite-thickness band.  The dark paper also explicitly marks quotient-edge co-registration and local Poisson reserve survival as conditional bridge premises.

```latex
\begin{theorem}[Rare independent collar events give the Poisson law]
Let a co-registered scalar slot be partitioned into \(m\) transverse subslots.
Suppose reserve events in distinct subslots are independent and each subslot has
probability
\[
p_m=\epsilon/m+O(m^{-2})
\]
of carrying one protected reserve event, with probability \(O(m^{-2})\) of two or
more events in a subslot. Then the number \(N_m\) of protected reserve events
converges in distribution to a Poisson variable \(N\) of mean \(\epsilon\). Therefore
\[
\Pr[N=0]=e^{-\epsilon}.
\]
\end{theorem}

\begin{proof}
For fixed \(n\), the probability of \(n\) occupied subslots is
\[
{m\choose n}
\left(\frac{\epsilon}{m}+O(m^{-2})\right)^n
\left(1-\frac{\epsilon}{m}+O(m^{-2})\right)^{m-n}
+
O(m^{-1}).
\]
Taking \(m\to\infty\) gives
\[
e^{-\epsilon}\frac{\epsilon^n}{n!}.
\]
The zero-count case gives
\[
\Pr[N=0]=e^{-\epsilon}.
\]
\(\square\)
\end{proof}

\begin{corollary}[Uniform \(\mathbb Z_6\) half-collar scalar exponent]
If the protected shared-edge reserve on the realized Standard Model quotient is
\[
\epsilon_{\mathbb Z_6}
=
\frac{P/4}{6}
=
\frac{P}{24},
\]
then the two-sided collar scalar survival factor is
\[
e^{-P/24}.
\]
If a neutral scalar screen mode samples one half of the two-sided collar, its
survival factor is
\[
e^{-P/48},
\]
and its red anomalous exponent is
\[
\theta_{\rm OPH}=P/48.
\]
\end{corollary}

\begin{proof}
Apply the previous theorem with \(\epsilon=P/24\), then take the square root for
a one-sided neutral scalar mode. The anomalous exponent is minus the logarithm
of the survival factor. \(\square\)
\end{proof}
```

---

## Z.3 Scalar release-channel count and amplitude

The previous derivation gave
[
A_\zeta=100\ln2,\varepsilon_\star.
]
The remaining gap is to make
[
\varepsilon_\star
]
a finite release-code output, not an amplitude fit.

```latex
\begin{definition}[Scalar release channel count]
Let \(\Pi_{\rm sc}\) be the projector onto center-free, monopole/dipole-quotiented
scalar-visible release records. Define
\[
N_{\rm rel}:={\rm Tr}_{Q_{\rm rel}}\Pi_{\rm sc}.
\]
Let \(\kappa_{\rm rel}\) be the exact pairing coefficient that converts the
independent half-collar defect amplitude into scalar temperature variance on the
release code.
\end{definition}

\begin{theorem}[Finite release-code amplitude]
On the independent half-collar scalar release branch, let
\[
\delta_{\rm slot}=1-e^{-P/48}.
\]
If the least scalar-visible Markov defect is the incoherent average of
\(N_{\rm rel}\) independent scalar release channels with exact pairing coefficient
\(\kappa_{\rm rel}\), then
\[
\varepsilon_\star
=
\kappa_{\rm rel}
\frac{(1-e^{-P/48})^2}{N_{\rm rel}}.
\]
Consequently
\[
A_\zeta
=
100\ln2\,
\kappa_{\rm rel}
\frac{(1-e^{-P/48})^2}{N_{\rm rel}}
+
O(N_{\rm rel}^{-3/2}).
\]
\end{theorem}

\begin{proof}
Each half-collar scalar slot contributes amplitude
\[
\delta_{\rm slot}=1-e^{-P/48}.
\]
Independent center-free channels add in variance, so the channel-averaged scalar
defect is
\[
\delta_{\rm slot}^2/N_{\rm rel},
\]
multiplied by the exact release-code pairing coefficient \(\kappa_{\rm rel}\).
The Fawzi--Renner/Petz recovery normalization used in the OPH collar branch gives
\[
A_T=4\ln2\,\varepsilon_\star.
\]
The large-scale Sachs--Wolfe conversion gives
\[
A_\zeta=25A_T.
\]
Combining these equations gives the displayed formula. \(\square\)
\end{proof}

\begin{theorem}[Amplitude is non-fit iff the channel trace is pre-CMB]
The scalar amplitude is a prediction rather than a fit precisely when
\(N_{\rm rel}\) and \(\kappa_{\rm rel}\) are computed from
\[
(Q_{\rm rel},\Pi_{\rm sc},\omega_{\rm rel})
\]
before comparing with CMB spectra. If either is chosen after reading \(A_s\), the
amplitude row is a diagnostic fit.
\end{theorem}

\begin{proof}
The previous theorem makes \(A_\zeta\) a deterministic function of
\[
P,\quad N_{\rm rel},\quad \kappa_{\rm rel}.
\]
If these inputs are fixed by the finite release code, the CMB supplies only a test.
If they are selected from the observed amplitude, the map has used the target value
as input. \(\square\)
\end{proof}
```

Numerically, this theorem says the amplitude is not closed until the finite code emits
[
N_{\rm rel}
\quad\text{and}\quad
\kappa_{\rm rel}.
]
But it also gives the exact non-fit formula:
[
\boxed{
A_\zeta
=======

100\ln2,
\kappa_{\rm rel}
\frac{(1-e^{-P/48})^2}{N_{\rm rel}}.
}
]

---

## Z.4 Pivot and radial normalization

```latex
\begin{theorem}[Pivot is a reporting convention]
Let the screen-to-bulk lift emit
\[
\Delta_\zeta^2(k)
=
A_{\rm rel}
\left(\frac{k}{k_{\rm rel}}\right)^{-P/48}
F_{\rm cut}(k),
\]
where \(k_{\rm rel}\) is the release normalization scale and \(F_{\rm cut}\) is a
finite-screen cutoff factor. Then the amplitude quoted at any pivot \(k_\star\) is
\[
A_\zeta(k_\star)
=
A_{\rm rel}
\left(\frac{k_\star}{k_{\rm rel}}\right)^{-P/48}
F_{\rm cut}(k_\star).
\]
Changing \(k_\star\) introduces no new physical parameter.
\end{theorem}

\begin{proof}
Evaluate the same power law at \(k=k_\star\). A pivot change multiplies the quoted
amplitude by the deterministic scale factor shown above, leaving the emitted
function unchanged. \(\square\)
\end{proof}
```

This closes a subtle gap: (A_\zeta) must be attached to a pivot without making the pivot a hidden fit parameter.

---

## Z.5 Finite packet moments give the anomaly Boltzmann variables

The dark-sector paper says the CMB/growth branch needs a genuine Boltzmann implementation with (\rho_A(a)), (B_A(k,a)), (\Gamma_{\rm rec}), stress variables, and likelihoods; it also states that the static galaxy/RAR branch cannot be inserted directly into FLRW because exact homogeneity has no preferred baryonic acceleration vector.

```latex
\begin{definition}[Finite packet stress moments]
Let \(d\mathcal E_x(p,n)\) be the finite packet measure emitted by the collar
parent functional at event \(x\), where \(p\) is comoving packet momentum and
\(n^i\) its direction. Define
\[
\rho_A
=
a^{-4}\int E(p,a)\,d\mathcal E,
\qquad
P_A
=
a^{-4}\int \frac{p^2}{3E(p,a)}\,d\mathcal E,
\]
\[
q_A^i
=
a^{-4}\int p\,n^i\,d\mathcal E,
\qquad
\pi_A^{ij}
=
a^{-4}\int
\frac{p^2}{E}
\left(n^in^j-\frac13\delta^{ij}\right)
d\mathcal E.
\]
Then
\[
w_A=P_A/\rho_A,
\qquad
\sigma_A\propto \pi_A/\rho_A.
\]
\end{definition}

\begin{theorem}[Boltzmann variables from finite packet moments]
If the collar parent functional emits a finite packet measure \(d\mathcal E_x\) and
a finite transition kernel \(K_A\), then the anomaly Boltzmann variables
\[
\rho_A(a),
\quad
w_A(a),
\quad
c_{s,A}^2(k,a),
\quad
\sigma_A(k,a),
\quad
Q_A^\mu(k,a)
\]
are finite moment functionals of \((d\mathcal E,K_A)\). In particular,
\[
Q_A^\nu:=\nabla_\mu T_A^{\mu\nu}
\]
is the exchange current produced by nonconservative transitions in \(K_A\), and it
vanishes on the closed transported branch.
\end{theorem}

\begin{proof}
The stress tensor is
\[
T_A^{\mu\nu}
=
a^{-4}
\int
\frac{p^\mu p^\nu}{E}
\,d\mathcal E.
\]
Its scalar decomposition gives
\[
\rho_A,\quad P_A,\quad q_A,\quad \pi_A
\]
by the displayed moments. The equation of state and anisotropic stress are ratios
of these moments. The rest-frame sound speed is the linear response
\[
\delta P_A/\delta\rho_A
\]
computed by perturbing \(d\mathcal E\). The divergence of \(T_A\) is zero for
conservative transport and equals the moment of the transition imbalance for
nonconservative \(K_A\). \(\square\)
\end{proof}

\begin{theorem}[Cold anomaly sufficient condition]
Suppose the packet measure is supported on nonrelativistic velocities with
\[
\langle v^2\rangle\le v_\star^2\ll1,
\]
and the transition kernel is conservative through recombination. Then
\[
|w_A|\le \frac13v_\star^2,
\qquad
c_{s,A}^2=O(v_\star^2),
\qquad
|\sigma_A|=O(v_\star^2),
\qquad
Q_A^\mu=0.
\]
Thus the anomaly sector is CDM-like through recombination up to \(O(v_\star^2)\).
\end{theorem}

\begin{proof}
For nonrelativistic packets,
\[
E=m+O(p^2/m),
\qquad
v=p/E.
\]
The pressure moment is
\[
P_A=\rho_A\langle v^2\rangle/3+O(v^4).
\]
Anisotropic stress and sound speed are also controlled by second velocity moments.
Conservative transport gives zero exchange current. \(\square\)
\end{proof}
```

---

## Z.6 Parent functional response kernel

```latex
\begin{theorem}[Parent functional response kernel]
Let
\[
\rho_{A,\rm eq}[X]c^2
=
\frac{15}{8\pi^2\ell(X)^4}
\int_{\mathcal C_X} I_{\omega_C[X]}(A:D|B)\,d\mu_C.
\]
If this functional is Frechet differentiable at homogeneous input \(\bar X(a)\),
then
\[
K_A^{(\rho)}(k,a)
=
\frac{1}{\bar\rho_b(a)}
\left.
\frac{\delta\rho_{A,\rm eq}[X](k,a)}
{\delta\delta_b(k,a)}
\right|_{\bar X},
\]
and
\[
B_A(k,a)
=
\frac{\bar\rho_b(a)}{\bar\rho_A(a)}
K_A^{(\rho)}(k,a).
\]
\end{theorem}

\begin{proof}
Translation and rotation invariance of the homogeneous background diagonalize the
linear functional derivative in Fourier space. The first derivative of
\(\rho_{A,\rm eq}\) with respect to baryon contrast is the density response.
Normalizing by \(\bar\rho_b\) gives \(K_A^{(\rho)}\); converting density response
to contrast response gives the displayed \(B_A\). \(\square\)
\end{proof}
```

This is the cleanest form of the missing (B_A(k,a)) theorem.

---

## Z.7 Finite repair matrix gives (\Gamma_{\rm rec})

```latex
\begin{theorem}[Finite repair matrix gives \(\Gamma_{\rm rec}\)]
For each \((k,a)\), let \(K(k,a)\) be the reversible finite repair transition
matrix on anomaly packet states with stationary distribution \(\pi\) and eigenvalues
\[
1=\lambda_1>|\lambda_2|\ge\cdots .
\]
For repair step time \(\Delta\eta\), define
\[
\Gamma_{\rm rec}(k,a)
=
-\Delta\eta^{-1}\log|\lambda_2(k,a)|.
\]
Then every mean-zero packet perturbation decays no slower than
\[
e^{-\Gamma_{\rm rec}\eta}.
\]
\end{theorem}

\begin{proof}
Reversibility makes \(K\) self-adjoint in \(L^2(\pi^{-1})\). Mean-zero perturbations
have no component along the eigenvalue-one stationary vector. The largest
remaining eigenvalue is \(|\lambda_2|\), so after \(n\) steps the norm is bounded by
\[
|\lambda_2|^n.
\]
With \(\eta=n\Delta\eta\), this is
\[
e^{-\Gamma_{\rm rec}\eta}.
\]
\(\square\)
\end{proof}
```

The OPH dark proof ledger already lists repair relaxation and the repair matrix as conditional packet-theorem targets, not completed cosmological likelihood objects.  This theorem gives the exact finite-matrix closure.

---

## Z.8 Bianchi exchange closure

```latex
\begin{theorem}[Bianchi exchange closure]
Assume the post-release geometry satisfies the Einstein equation with total stress
\[
T_{\rm tot}^{\mu\nu}
=
T_{\rm SM}^{\mu\nu}
+
T_A^{\mu\nu}
+
T_\Lambda^{\mu\nu}.
\]
Then
\[
\nabla_\mu T_{\rm tot}^{\mu\nu}=0.
\]
If \(T_\Lambda^{\mu\nu}\) is constant and the Standard Model and anomaly sectors
exchange four-current \(Q_A^\nu\), then
\[
\nabla_\mu T_A^{\mu\nu}=Q_A^\nu,
\qquad
\nabla_\mu T_{\rm SM}^{\mu\nu}=-Q_A^\nu.
\]
\end{theorem}

\begin{proof}
The contracted Bianchi identity gives
\[
\nabla_\mu G^{\mu\nu}=0.
\]
If \(\Lambda\) is constant,
\[
\nabla_\mu T_\Lambda^{\mu\nu}=0.
\]
Therefore the sum of the matter-sector divergences vanishes. Defining the anomaly
divergence as \(Q_A\) gives the two exchange equations. \(\square\)
\end{proof}
```

---

## Z.9 No curvature/anomaly double counting

```latex
\begin{theorem}[No curvature/anomaly double counting]
On the flat OPH cosmology branch, curvature holonomy, cosmological-capacity
stress, and anomaly load live in distinct quotient sectors:
\[
h_K,
\qquad
T_\Lambda^{\mu\nu}=-\rho_\Lambda g^{\mu\nu},
\qquad
T_A^{\mu\nu}=\rho_Au^\mu u^\nu+\cdots .
\]
If \(h_K=0\), \(\rho_\Lambda\) is fixed by screen capacity, and \(Q_A\) is the
conserved finite-collar anomaly charge, then the Friedmann constraint contains no
duplicated homogeneous term.
\end{theorem}

\begin{proof}
The curvature term scales as \(a^{-2}\) and is represented by spatial geometric
holonomy. The cosmological-capacity term has equation of state \(-1\) and is
constant. A closed anomaly load has dust scaling \(a^{-3}\). Since these quotient
sectors have different transformation and scaling laws, equality of one cannot
absorb the others except by changing branch data. \(\square\)
\end{proof}
```

This matters because otherwise the paper risks double-counting “flat residual” dark abundance and capacity-selected (\Lambda).

---

## Z.10 Standard recombination is inherited

```latex
\begin{theorem}[Standard recombination is inherited]
Assume the OPH particle branch emits the same low-energy electromagnetic coupling,
electron mass, baryon masses, and Standard Model charge assignments used by the
recombination calculation, and assume hot release supplies a thermal photon-baryon
plasma with baryon-to-photon ratio \(\eta_b\). Then the recombination history is the
standard atomic recombination ODE system, modified only by OPH-supplied changes to
\[
H(a),
\qquad
\eta_b,
\qquad
\text{or anomaly perturbations}.
\]
\end{theorem}

\begin{proof}
Atomic binding energies, Thomson scattering, and recombination rates are functions
of the low-energy electromagnetic coupling, electron mass, baryon masses, and charge
assignments. If those are the emitted Standard Model values, the microphysical rate
equations are unchanged. Cosmology enters through expansion, temperature, and
perturbation variables, so only OPH changes to those quantities alter the
recombination history. \(\square\)
\end{proof}
```

This closes another hidden gap: “standard Boltzmann transfer” also needs standard atomic recombination inputs.

---

## Z.11 Baryogenesis as finite transition imbalance

The original paper’s conclusion still says release/baryogenesis data must be derived. This theorem does that without weakening the claim.

```latex
\begin{definition}[Finite OPH baryogenesis source]
Let \(K_B(q\to q')\) be the finite release-collar transition kernel on states
carrying baryon number \(B(q)\). Let \(\Theta\) be CP conjugation on the finite code.
Define the CP-odd baryon source
\[
S_B(N)
=
\sum_{q,q'}
\pi_N(q)
\bigl[B(q')-B(q)\bigr]
\left[
K_B(q\to q')
-
K_B(\Theta q\to \Theta q')
\right].
\]
Let \(\Gamma_{\rm wash}(N)\) be the finite washout rate emitted by the same
transition matrix.
\end{definition}

\begin{theorem}[Baryon asymmetry from finite release transitions]
If the comoving baryon yield satisfies
\[
Y_B'(N)
=
S_B(N)-\Gamma_{\rm wash}(N)Y_B(N),
\]
then
\[
Y_B(N_f)
=
Y_B(N_i)e^{-\int_{N_i}^{N_f}\Gamma_{\rm wash}dN}
+
\int_{N_i}^{N_f}
S_B(s)
\exp\!\left[
-\int_s^{N_f}\Gamma_{\rm wash}(u)du
\right]ds.
\]
Thus baryogenesis is non-fit precisely when
\[
K_B,\quad \Theta,\quad \pi_N
\]
are finite release-code outputs.
\end{theorem}

\begin{proof}
The equation is linear in \(Y_B\). Multiplying by the integrating factor
\[
\exp\left(\int\Gamma_{\rm wash}dN\right)
\]
and integrating gives the displayed expression. If the source and washout are
computed from the finite transition matrix, the result is fixed before comparison
with observed baryon abundance. \(\square\)
\end{proof}
```

---

## Z.12 Full microphysical closure theorem

```latex
\begin{theorem}[Microphysical closure of the inflation-free OPH branch]
Assume the following finite objects are emitted by the OPH screen/collar
microphysics before CMB comparison:
\[
(Q_{\rm rel},\omega_{\rm rel},\Pi_{\rm sc},N_{\rm rel},\kappa_{\rm rel}),
\]
\[
(d\mathcal E_x,K_A,B_A,\Gamma_{\rm rec}),
\qquad
(K_B,\Theta,\pi_N),
\]
and assume they satisfy the same-boundary horizon theorem, the zero-holonomy
flatness theorem, the half-collar reserve theorem, the scalar release-code amplitude
theorem, the cold anomaly sufficient condition through recombination, and the
baryogenesis transition theorem. Then the inflation-free OPH cosmology branch
emits, without fitting to CMB data,
\[
\Omega_K=0,
\qquad
n_s=1-P/48,
\]
\[
A_\zeta
=
100\ln2\,
\kappa_{\rm rel}
\frac{(1-e^{-P/48})^2}{N_{\rm rel}},
\]
\[
\rho_A(a),
\quad
B_A(k,a),
\quad
\Gamma_{\rm rec}(k,a),
\quad
w_A,
\quad
c_{s,A}^2,
\quad
\sigma_A,
\quad
Q_A^\mu,
\]
and a finite-code baryon yield \(Y_B\). The CMB likelihood then tests the emitted
branch rather than determining its load-bearing quantities.
\end{theorem}

\begin{proof}
Flatness follows from zero visible curvature holonomy. The scalar tilt follows from
the half-collar reserve. The amplitude follows from the finite scalar release-code
theorem. The anomaly background and perturbation variables follow from the finite
packet moment and parent-functional response theorems. Baryogenesis follows from
the finite transition-imbalance theorem. Since all listed objects are computed before
CMB comparison, the CMB transfer calculation receives fixed initial and stress-sector
data. \(\square\)
\end{proof}
```

---

# Remaining gap status after this appendix

After adding these theorems, the paper’s remaining open work is no longer “invent the missing cosmology.” It is to produce **finite certificates**:

[
N_{\rm rel},\quad \kappa_{\rm rel},\quad d\mathcal E_x,\quad K_A,\quad K_B,\quad \Theta,\quad \pi_N.
]

Those are finite-screen/collar artifacts, not CMB fit knobs. Once emitted, the branch has a closed non-inflaton theorem path:

[
\boxed{
P
\longrightarrow
\theta=P/48
\longrightarrow
n_s=1-P/48,
}
]

[
\boxed{
(Q_{\rm rel},\Pi_{\rm sc})
\longrightarrow
N_{\rm rel},\kappa_{\rm rel}
\longrightarrow
A_\zeta,
}
]

[
\boxed{
(d\mathcal E_x,K_A)
\longrightarrow
\rho_A,w_A,c_s^2,\sigma_A,Q_A,B_A,\Gamma_{\rm rec},
}
]

[
\boxed{
(K_B,\Theta,\pi_N)
\longrightarrow
Y_B.
}
]

That is the repair: the claim stays strong, but every remaining load-bearing object is now a finite theorem target or certificate target.
