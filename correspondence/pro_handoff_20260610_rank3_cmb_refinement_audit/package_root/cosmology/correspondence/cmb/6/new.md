Done — I turned the “conditional theorem ledger” into an explicit theorem/proof package.

[Download the theorem package](sandbox:/mnt/data/OPH-CMB-missing-theorems-and-proofs-v1.2.zip)
[Open the main theorem/proof note](sandbox:/mnt/data/oph_cmb_missing_theorems_v1_2/math/OPH-CMB-Missing-Theorems-and-Proofs-v1.2.md)
[Open the theorem status table](sandbox:/mnt/data/oph_cmb_missing_theorems_v1_2/data/theorem_status_v1_2.csv)

The key point is this: the **screen covariance math is now theorem-level**, but the **exact numerical branch**

[
\eta_R=e\alpha\sqrt\pi,\qquad q_{\rm IR}=\frac14,\qquad \ell_{\rm IR}=32
]

still requires three explicit selector lemmas from the finite-patch carrier. I did not hide that by calling them “proved from OPH” without the missing premises. The uploaded OPH-CMB starting note already says the correct architecture is an OPH screen/freezeout effective theory first, not a direct derivation of the whole physical CMB from the current OPH-FPE run.  It also explicitly separates what is mostly mathematical from what needs simulation, including finite observer-patch derivations of (\eta), (q_{\rm IR}), effective capacity, parity amplitude, and off-diagonal covariance. 

## What is now closed

| Claim                                         | Status                                            |
| --------------------------------------------- | ------------------------------------------------- |
| Schedule-independent freezeout readout        | closed under OPH confluence assumptions           |
| Finite graph MaxEnt covariance                | closed                                            |
| (C_\ell\propto1/[\ell(\ell+1)])               | closed under graph-to-sphere spectral convergence |
| (n_s=1-\eta_R)                                | closed                                            |
| Finite-capacity/cellulation window            | closed                                            |
| Heat-kernel low-(\ell) envelope               | closed                                            |
| Parity envelope                               | closed under antipodal cap-pair channel           |
| BipoSH/off-diagonal covariance representation | closed                                            |
| Isotropic screen-to-primordial bridge         | closed in thin-screen approximation               |
| Acoustic peak preservation bound              | closed as analytic bound                          |

The theorem pack gives full statements and proofs for each.

## The important no-free theorem

The central missing-math result is actually a **no-free numeric kernel theorem**:

[
\boxed{
\text{OPH confluence + local MaxEnt fixes the covariance class, but not }q_{\rm IR},\ell_{\rm IR}.
}
]

Proof idea: once the baseline covariance is positive,

[
C_\ell^{(0)}\propto \frac{1}{[\ell(\ell+1)]^{1+\eta_R/2}},
]

then for any (0\le q_{\rm IR}\le1) and any (\ell_{\rm IR}>0),

[
F_{\rm IR}(\ell)
================

1-q_{\rm IR}
\exp!\left[
-\frac{\ell(\ell+1)}{\ell_{\rm IR}(\ell_{\rm IR}+1)}
\right]
]

is a valid positive isotropic spectral multiplier. So the previous assumptions allow a family of IR kernels. That is why (q_{\rm IR}) and (\ell_{\rm IR}) need a selector theorem, not just a fit.

This is consistent with the earlier CMB project note: the mathematical tier derives the allowed screen covariance and natural low-(\ell)/parity forms, while finite observer-patch simulations are needed to derive (\eta), (q_{\rm IR}), (N_{\rm eff}), parity strength, and off-diagonal covariance rather than fitting them. 

## The three missing selector theorems

I made the three exact numerical claims into explicit theorem targets.

### 1. Pixel-detuning anomalous-dimension theorem

If the freezeout repair RG has one relevant red anomalous dimension and OPH’s pixel detuning transfers into that dimension by the Euler repair-time normalization,

[
\eta_R=e\Delta_P,
\qquad
\Delta_P=\alpha(0)\sqrt\pi,
]

then

[
\boxed{
\eta_R=e\alpha(0)\sqrt\pi,
\qquad
n_s=1-e\alpha(0)\sqrt\pi.
}
]

Proof: substitute (\Delta_P=\alpha\sqrt\pi) and use the already-proved relation

[
n_s=1-\eta_R.
]

What remains: prove from finite-patch repair RG that the anomalous dimension is exactly (e\Delta_P).

### 2. Quarter-reserve theorem for (q_{\rm IR}=1/4)

If the global repair reserve decomposes into four equipotent observer-facing sectors,

[
R_{\rm glob}
============

R_0\oplus R_1\oplus R_2\oplus R_3,
\qquad
\dim R_a=d,
]

and freezeout withholds exactly one protected sector, then

[
\boxed{
q_{\rm IR}=\frac{d}{4d}=\frac14.
}
]

Proof: direct dimension counting.

What remains: prove the four-sector decomposition from the finite-patch carrier. A plausible OPH route is a (3+1) repair split: three spatial collar-orientation classes plus one record/clock closure class, with equipotence forced by support-visible Lorentz normalization before freezeout.

### 3. Pentagonal binary-collar theorem for (\ell_{\rm IR}=32)

If the relevant global repair collar is controlled by five independent binary orientation registers,

[
B_{\rm pent}\cong(\mathbb Z_2)^5,
]

and if the coherent global repair angular degree is the character count of that register, then

[
\boxed{
\ell_{\rm IR}=|\widehat B_{\rm pent}|=2^5=32.
}
]

Proof: ((\mathbb Z_2)^5) has (2^5=32) characters.

What remains: prove that the OPH global repair collar is exactly this five-bit pentagonal orientation register, and prove that the angular repair degree is the character count. This should be visible as a mode cluster near (\ell\simeq32) in finite-patch simulations.

## Exact kernel theorem under those selectors

Once the quarter-reserve and pentagonal-collar selectors are accepted, the exact scalar OPH IR kernel follows immediately:

[
\boxed{
F_{\rm IR}^{\rm exact}(\ell)
============================

1-\frac14
\exp!\left[
-\frac{\ell(\ell+1)}{32\cdot33}
\right].
}
]

This is now a theorem **conditional only on the two selector lemmas**, not a fitted ansatz.

## Why this is the right claim boundary

The OPH-CMB note says the first analytic paper should prove MaxEnt screen synchronization, near scale invariance, finite-capacity corrections, low-(\ell) repair terms, and parity/off-diagonal covariance terms; it also says Planck likelihoods, Monte Carlo anomaly significance, acoustic transfer, and finite-patch parameter derivation need computation.  The OPH synthesis also treats the regulated screen-side architecture as a finite patch-carrier habitat where local observables, overlap observables, records, and synchronization maps can be written without claiming a unique final UV completion. 

So the current theorem status is:

[
\boxed{
\text{local OPH screen covariance theorem: closed;}
}
]

[
\boxed{
\text{exact OPH numeric CMB kernel: reduced to three selector lemmas;}
}
]

[
\boxed{
\text{official Planck likelihood: software/data execution gate, not a theorem gate.}
}
]

The next mathematical task is therefore very specific: prove or falsify the three selector lemmas from finite-patch microphysics.
