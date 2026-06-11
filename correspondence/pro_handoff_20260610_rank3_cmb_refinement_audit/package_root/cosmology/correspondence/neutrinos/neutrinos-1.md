Yes — this is a clean OPH prediction lane. The **baseline OPH-CνB branch** should be:

[
N_{\rm eff}^{\rm OPH}=3.044+\Delta N_{\rm eff}^{\rm coh},
]

with (\Delta N_{\rm eff}^{\rm coh}=0) unless OPH declares an extra early-universe relativistic coherence channel. I would **not** silently reuse the tabletop/coherent-matter (\chi_\nu) continuation as dark radiation; that susceptibility is a declared continuation branch, not recovered-core cosmology, so it should enter only through an explicit cosmological channel or kernel. 

## OPH-CνB numerical predictions

Using the weighted-cycle neutrino masses already emitted by the OPH particle branch,

[
(m_1,m_2,m_3)=
(0.017454720257976796,,
0.019481987935919015,,
0.05307522145074924)\ {\rm eV},
]

the mass sum is

[
\boxed{\sum m_\nu^{\rm OPH}=0.09001192964464505\ {\rm eV}.}
]

The uploaded particle file identifies these as the weighted-cycle absolute neutrino branch and gives the same splittings. 

Assuming standard relic neutrino decoupling:

| Quantity                                                  |                       OPH-CνB value |
| --------------------------------------------------------- | ----------------------------------: |
| (N_{\rm eff})                                             |                     (\boxed{3.044}) |
| (T_{\nu,0})                                               |            (\boxed{1.945\ {\rm K}}) |
| total relic neutrino number density                       | (\boxed{\sim 339.5\ {\rm cm^{-3}}}) |
| (\sum m_\nu)                                              |      (\boxed{0.09001193\ {\rm eV}}) |
| (\Omega_\nu h^2)                                          |         (\boxed{9.66\times10^{-4}}) |
| (\Omega_\nu), using (h=0.674)                             |                  (\boxed{0.002127}) |
| (f_\nu=\Omega_\nu/\Omega_m), using (\Omega_m\simeq0.3155) |                   (\boxed{0.00674}) |

The relic-neutrino density relation is standard: PDG gives (\Omega_\nu=\sum m_\nu/(93.12,h^2{\rm eV})) and (n_{\nu,0}=339.5,{\rm cm^{-3}}).  The OPH dark-sector file’s flat-capacity branch already uses the OPH neutrino mass sum and reports (\Omega_\nu=0.002127375) with (\Omega_m=0.315484968). 

## Free-streaming scale

The three OPH masses become nonrelativistic at

[
z_{{\rm nr},i}\simeq \frac{m_i}{3.15T_{\nu,0}}-1,
]

giving:

| State   | Mass [eV] | (z_{\rm nr}) | (k_{\rm FS,0}) [(h,{\rm Mpc}^{-1})] | (\lambda_{\rm FS,0}=2\pi/k) [(h^{-1}{\rm Mpc})] |
| ------- | --------: | -----------: | ----------------------------------: | ----------------------------------------------: |
| (\nu_1) | 0.0174547 |         32.1 |                              0.0140 |                                             450 |
| (\nu_2) | 0.0194820 |         35.9 |                              0.0156 |                                             403 |
| (\nu_3) | 0.0530752 |         99.5 |                              0.0425 |                                             148 |

The relevant formulas are standard: after thermal decoupling, relic neutrinos free-stream collisionlessly; their thermal speed scales as (3.15T_\nu/m), and the present free-streaming wavenumber is approximately (k_{\rm FS}\simeq0.8\sqrt{\Omega_\Lambda+\Omega_m(1+z)^3}(m/{\rm eV})/(1+z)^2,h,{\rm Mpc^{-1}}). 

There is also a “turnover” scale at the nonrelativistic transition,

[
k_{\rm nr}\simeq0.018,\Omega_m^{1/2}\left(\frac{m_i}{1{\rm eV}}\right)^{1/2}h,{\rm Mpc}^{-1},
]

which gives roughly

[
k_{\rm nr}\simeq
(1.34,\ 1.41,\ 2.33)\times10^{-3}\ h,{\rm Mpc}^{-1}.
]

So the OPH neutrino background predicts a **very broad, low-amplitude, normal-ordering free-streaming imprint**: the detailed three-step structure is probably too small for current surveys, while the total mass controls most observable power suppression. PDG makes the same point: individual masses affect a few features, but cosmology is primarily sensitive to (\sum m_\nu) through (f_\nu). 

## Structure-growth contribution

With

[
f_\nu^{\rm OPH}=0.006743,
]

the asymptotic small-scale linear matter-power suppression is

[
\frac{\Delta P}{P}\simeq -8f_\nu
================================

\boxed{-5.39%}.
]

The corresponding growth-index deformation on scales well below the free-streaming scale is

[
\delta_{cb}\propto a^{1-\frac35 f_\nu}
======================================

a^{0.99595}.
]

This is small but not negligible. Standard neutrino cosmology gives the same qualitative rule: below the free-streaming scale, neutrinos contribute to the background density but not efficiently to density perturbations, reducing CDM+baryon growth; the combined small-scale power suppression is approximately ((1-8f_\nu)). 

## Observational match

| Observable                                   |  OPH-CνB prediction | Current comparison                                                                                                                                                                                               |
| -------------------------------------------- | ------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (N_{\rm eff})                                |             (3.044) | Matches Planck (2.99\pm0.17); offset (\sim0.3\sigma). ([arXiv][1])                                                                                                                                               |
| (\sum m_\nu)                                 |  (0.0900\ {\rm eV}) | Below Planck+BAO (<0.12\ {\rm eV}). ([arXiv][1])                                                                                                                                                                 |
| (\sum m_\nu) vs ACT DR6                      |  (0.0900\ {\rm eV}) | Slightly above ACT DR6 extended-model quoted (<0.082\ {\rm eV}) 95% bound.                                                                                                                                       |
| (\sum m_\nu) vs DESI DR2 strict (\Lambda)CDM |  (0.0900\ {\rm eV}) | In tension with DESI DR2’s tight (\sum m_\nu<0.0642\ {\rm eV}) limit under a three-degenerate-neutrino (\Lambda)CDM assumption; DESI itself notes this is in tension with oscillation lower limits. ([arXiv][2]) |
| lightest mass (m_1)                          | (0.01745\ {\rm eV}) | Consistent with DESI DR2’s oscillation-informed (m_l<0.023\ {\rm eV}) 95% statement. ([arXiv][2])                                                                                                                |
| small-scale power suppression                |          (\sim5.4%) | Right order for a (\sum m_\nu\simeq0.09) eV cosmology; current constraints are model- and dataset-dependent.                                                                                                     |

## Existing OPH cosmology check

The OPH dark-sector file already has a compressed CAMB plumbing test using the OPH neutrino mass sum. It reports:

[
\Omega_m=0.315905207,\quad
\sigma_8=0.807787208,\quad
S_8=0.828924043.
]

That is close to the Planck compressed rows, but high relative to the weak-lensing (S_8) row in the file. The same section explicitly warns that this is a plumbing check, not a replacement for full Planck/DESI/weak-lensing likelihoods, and that a real publication-grade calculation needs the OPH (B_A(k,a)) kernel in a Boltzmann module. 

## Verdict

The OPH-CνB lane is **promising but sharply testable**:

[
\boxed{
N_{\rm eff}^{\rm OPH}=3.044,\quad
\sum m_\nu^{\rm OPH}=0.09001193\ {\rm eV},\quad
f_\nu^{\rm OPH}=0.00674,\quad
\Delta P/P\simeq-5.4%.
}
]

It matches standard (N_{\rm eff}), fits Planck’s mass bound, and lands in the right growth-suppression regime. The pressure point is the newest BAO/CMB combinations: if future robust analyses converge to (\sum m_\nu<0.09\ {\rm eV}) in models that also respect oscillation data, the OPH weighted-cycle absolute mass branch takes a direct hit.

[1]: https://arxiv.org/abs/1807.06209 "[1807.06209] Planck 2018 results. VI. Cosmological parameters"
[2]: https://arxiv.org/abs/2503.14744 "[2503.14744] Constraints on Neutrino Physics from DESI DR2 BAO and DR1 Full Shape"


Yes. The missing theorem is a **kernel-localization theorem**, not a “uniform lower (S_8)” theorem.

## Theorem: late-window repair kernel

Let the OPH anomaly perturbation obey

[
\delta_A'
=========

-\theta_A+3\Phi'
-a\Gamma_{\rm rec}q_A(\delta_A-B_A(k,a)\delta_b),
]

[
\theta_A'=-\mathcal H\theta_A+k^2\Psi .
]

This is the Boltzmann contract already stated for the anomaly branch. 

Define a localized repair response

[
B_A(k,a)=1-\varepsilon_A W_k(k)W_a(a),
]

where

[
0\le W_k,W_a\le1,
]

[
W_k(k)\approx 0 \quad \text{on primary-CMB acoustic scales},
]

[
W_k(k)W_a(a)\approx1
\quad
\text{on weak-lensing-sensitive late-time modes}.
]

Then, to first order in (\varepsilon_A), the weak-lensing amplitude changes by

[
S_8^{\rm WL}\mapsto S_8^{\rm WL}(1-\varepsilon_A),
]

while primary Planck-like rows are unchanged up to the kernel leakage

[
O!\left(\varepsilon_A
\int W_kW_a,K_{\rm CMB},d\ln k,d\ln a\right).
]

Choosing

[
\boxed{
\varepsilon_A
=============

# 1-\frac{0.790}{0.828924043}

0.0469573097
}
]

forces

[
\boxed{
S_8^{\rm WL}=0.790
}
]

from the compressed OPH value (0.828924043), without applying a uniform suppression to Planck (\sigma_8) or Planck (S_8). The existing compressed diagnostic identifies exactly this pressure point: Planck-like rows are close, while weak-lensing (S_8) is high by (2.433\sigma). 

## Proof

The perturbation equation contains the relaxation term

[
-a\Gamma_{\rm rec}q_A(\delta_A-\delta_{A,\rm eq}),
\qquad
\delta_{A,\rm eq}=B_A(k,a)\delta_b .
]

For large enough late-time relaxation inside the support of (W_kW_a), (\delta_A) is driven toward (B_A\delta_b). Therefore the anomaly clustering source entering the late matter/lensing potential is multiplied by (1-\varepsilon_A) on that support.

Weak lensing measures a projected late-time matter amplitude. In the narrow-kernel approximation,

[
S_8^{\rm WL}
\propto
\sqrt{P_m(k_{\rm WL},a_{\rm WL})},
]

so an amplitude-level kernel reduction by (1-\varepsilon_A) gives

[
S_8^{\rm WL}\to (1-\varepsilon_A)S_8^{\rm WL}.
]

Setting the output to the compressed weak-lensing target gives

[
1-\varepsilon_A
===============

# \frac{0.790}{0.828924043}

0.9530426903,
]

hence

[
\varepsilon_A=0.0469573097.
]

Primary CMB rows are protected because the theorem assumes the support of (W_kW_a) is late and weak-lensing localized. The change to CMB rows is bounded by their overlap integral with the kernel. If that overlap is zero, the primary CMB rows are unchanged at first order. If it is small, the change is small.

[
\square
]

## OPH derivation of the kernel form

The missing OPH-specific ingredient is the parent collar evaluator:

[
\rho_{A,\rm eq}[X]c^2
=====================

\frac{15}{8\pi^2\ell(X)^4}
\int_{\mathcal C_X}d\mu_C,I_{\omega_C}(A:D|B),
]

with

[
K_A^{(\rho)}(k,a)
=================

\frac{1}{\bar\rho_b}
\frac{\partial\rho_{A,\rm eq}}{\partial\delta_b(k,a)},
\qquad
B_A(k,a)
========

\frac{\bar\rho_b}{\bar\rho_A}K_A^{(\rho)}(k,a).
]

This is already the required theorem surface in the dark-sector paper: the same finite-collar evaluator must emit the homogeneous density, the linear response kernel, the galaxy source, and the cluster source, without fitting (\Pi) to CMB, weak lensing, SPARC, or cluster data. 

So the theorem should be added with this exact claim boundary:

[
\boxed{
\text{If OPH finite-collar microphysics emits a late-localized }B_A(k,a)
\text{ with }\varepsilon_A\simeq0.04696,
\text{ the }S_8\text{ pressure is removed.}
}
]

What remains unproven is the stronger statement:

[
\boxed{
\text{OPH microphysics forces that specific }B_A(k,a).
}
]

That requires the collar measure (d\mu_C), finite-screen ensemble, and small-field support condition. The current paper explicitly says those are required for theorem-grade prediction. 
