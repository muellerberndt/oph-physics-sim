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
---

Here is the drop-in **Research Track 9: Cosmological Neutrino Background** package.

# 9. Cosmological Neutrino Background

## 9.1 Claim boundary

The clean baseline is:

[
\boxed{
\text{OPH-C}\nu\text{B}_{0}
===========================

\text{standard thermal relic neutrino background}
+
\text{OPH weighted-cycle neutrino masses}.
}
]

That means the baseline branch predicts

[
\boxed{
N_{\rm eff}^{\rm OPH}=3.044
}
]

with **no extra relativistic OPH coherence radiation** unless a separate early-universe coherence channel is declared.

This boundary matters. The (\chi_\nu) coherent-matter susceptibility branch explicitly says the recovered OPH core leaves (\chi_\nu) unfixed and that a zero coherent-matter continuation is allowed unless an additional continuation branch is declared. It then declares a separate quotient-edge continuation where (\chi_\nu^{\rm can}) is forced into the band (0.9343006394893864\ldots\le\chi_\nu^{\rm can}\le1).  So the CνB track should **not** silently convert tabletop coherent-matter susceptibility into cosmological dark radiation.

The extended branch is therefore explicitly parameterized:

[
\boxed{
\text{OPH-C}\nu\text{B}_{+}
===========================

\text{OPH-C}\nu\text{B}*{0}
+
\Delta N*{\rm eff}^{\rm coh}
+
\Gamma_{\nu{\rm coh}}(k,a)
+
Q^\mu_{\nu A}
+
B_{\nu A}(k,a).
}
]

Baseline values:

[
\Delta N_{\rm eff}^{\rm coh}=0,
\qquad
\Gamma_{\nu{\rm coh}}=0,
\qquad
Q^\mu_{\nu A}=0,
\qquad
B_{\nu A}=0.
]

Any nonzero term is a **new declared OPH cosmology branch**, not part of the baseline.

---

## 9.2 Baseline numerical prediction

Use the OPH weighted-cycle absolute neutrino branch as the cosmological mass-eigenstate vector:

[
(m_1,m_2,m_3)
=============

(0.017454720257976796,,
0.019481987935919015,,
0.05307522145074924)\ {\rm eV}.
]

The same file gives the emitted splittings

[
\Delta m_{21}^2=7.488059465106851\times10^{-5}\ {\rm eV}^2,
]

[
\Delta m_{31}^2=2.5123118727618473\times10^{-3}\ {\rm eV}^2,
]

[
\Delta m_{32}^2=2.4374312781107786\times10^{-3}\ {\rm eV}^2.
]



Therefore

[
\boxed{
\sum m_\nu^{\rm OPH}=0.09001192964464505\ {\rm eV}.
}
]

Using standard relic-neutrino thermodynamics:

| Quantity                                           |                   OPH-CνB baseline |
| -------------------------------------------------- | ---------------------------------: |
| (N_{\rm eff})                                      |                    (\boxed{3.044}) |
| (T_{\nu,0})                                        |         (\boxed{1.94537\ {\rm K}}) |
| total relic neutrino number density                | (\boxed{\sim339.5\ {\rm cm^{-3}}}) |
| (\sum m_\nu)                                       |     (\boxed{0.09001193\ {\rm eV}}) |
| (\Omega_\nu h^2)                                   |       (\boxed{9.666\times10^{-4}}) |
| (\Omega_\nu), for (h=0.674)                        |                 (\boxed{0.002128}) |
| (f_\nu=\Omega_\nu/\Omega_m), for (\Omega_m=0.3155) |                 (\boxed{0.006744}) |

PDG’s 2025 neutrino-cosmology review gives the standard relic-neutrino framework, including modern (N_{\rm eff}=3.044), the neutrino density relation (\Omega_\nu=\sum m_\nu/(93.12h^2{\rm eV})), and total relic number density (n_{\nu,0}\simeq339.5,{\rm cm^{-3}}). ([Kek][1])

---

## 9.3 Free-streaming prediction

For each mass eigenstate,

[
z_{{\rm nr},i}\simeq \frac{m_i}{3.151,T_{\nu,0}}-1,
]

and the present free-streaming scale can be summarized by

[
k_{{\rm FS},0}\simeq0.8\left(\frac{m_i}{1,{\rm eV}}\right)
h,{\rm Mpc}^{-1}.
]

| State   |  Mass [eV] | (z_{\rm nr}) | (v_{\nu,0}) [km/s] | (k_{{\rm FS},0}) [(h,{\rm Mpc}^{-1})] | (\lambda_{{\rm FS},0}=2\pi/k) [(h^{-1}{\rm Mpc})] |
| ------- | ---------: | -----------: | -----------------: | ------------------------------------: | ------------------------------------------------: |
| (\nu_1) | 0.01745472 |        32.04 |              9,073 |                               0.01396 |                                             450.0 |
| (\nu_2) | 0.01948199 |        35.88 |              8,129 |                               0.01559 |                                             403.1 |
| (\nu_3) | 0.05307522 |        99.48 |              2,984 |                               0.04246 |                                             148.0 |

The turnover scale at nonrelativistic transition,

[
k_{\rm nr}\simeq
0.018,\Omega_m^{1/2}
\left(\frac{m_i}{1{\rm eV}}\right)^{1/2}
h,{\rm Mpc}^{-1},
]

gives

[
k_{\rm nr}\simeq
(1.336,\ 1.411,\ 2.329)\times10^{-3}
h,{\rm Mpc}^{-1}.
]

So the OPH-CνB signature is a **normal-ordering, three-step free-streaming imprint**. In practice, current cosmology is mostly sensitive to (\sum m_\nu), but the full OPH branch predicts the non-degenerate triplet, not just the sum.

---

## 9.4 Structure-growth prediction

The neutrino matter fraction is

[
f_\nu^{\rm OPH}=0.006744.
]

The asymptotic small-scale linear matter-power suppression is approximately

[
\frac{\Delta P}{P}\simeq -8f_\nu,
]

so

[
\boxed{
\frac{\Delta P}{P}\simeq -5.40%.
}
]

The corresponding CDM+baryon growth exponent deformation is

[
\delta_{cb}\propto a^{1-\frac35f_\nu}
=====================================

a^{0.99595}.
]

This gives a compact OPH prediction:

[
\boxed{
\text{OPH-C}\nu\text{B suppresses small-scale matter power by about }5.4%.
}
]

That is large enough to matter for weak lensing, CMB lensing, RSD, and (S_8), but small enough that degeneracies with dark energy, anomaly stress, optical depth, and baryonic feedback must be handled in a full likelihood.

---

## 9.5 CMB phase-shift and anisotropic-stress signature

Baseline OPH-CνB predicts standard free-streaming relativistic neutrinos:

[
\boxed{
c_{\rm eff}^2=\frac13,
\qquad
c_{\rm vis}^2=\frac13,
\qquad
\Gamma_{\nu{\rm int}}=0.
}
]

This is a key observable. Free-streaming relativistic neutrino perturbations produce a characteristic CMB acoustic phase shift; Bashinsky and Seljak identify that phase shift as a unique effect of neutrino free streaming under adiabatic initial conditions. ([arXiv][2])

Therefore OPH gets a sharp fork:

[
\boxed{
\text{Baseline OPH predicts the standard free-streaming CMB phase shift.}
}
]

A nonstandard phase shift, reduced viscosity, neutrino self-interaction signal, or dark-radiation phase pattern can only be claimed by OPH if the extended branch emits

[
\Gamma_{\nu{\rm coh}}(k,a)
\quad\text{or}\quad
\Delta N_{\rm eff}^{\rm coh}
]

from OPH microphysics.

---

## 9.6 BBN and thermal-history closure

The baseline thermal-history ledger is:

[
T_{\nu,0}
=========

\left(\frac{4}{11}\right)^{1/3}T_{\gamma,0},
]

[
N_{\rm eff}=3.044,
]

[
n_{\nu,0}\simeq339.5\ {\rm cm^{-3}}.
]

So baseline OPH inherits standard BBN consistency. The BBN row should be carried as a required check:

[
\boxed{
\left(
N_{\rm eff}^{\rm OPH},
\eta_b,
Y_p,
{\rm D/H}
\right)
}
]

with

[
N_{\rm eff}^{\rm OPH}=3.044
]

unless a new OPH relativistic coherence channel is explicitly declared. Any nonzero (\Delta N_{\rm eff}^{\rm coh}) must survive both CMB and BBN, not only CMB.

---

## 9.7 Laboratory bridge

The OPH neutrino track also gives two laboratory observables.

Using the OPH weighted-cycle angles

[
\theta_{12}=34.225904631810025^\circ,
\qquad
\theta_{13}=8.686355527700156^\circ,
]

and the emitted Majorana pair

[
\alpha_{21}=153.6185177794357^\circ,
\qquad
\alpha_{31}=257.0032408220805^\circ,
]

the electron-flavor weights are

[
|U_{e1}|^2=0.6680488646,
]

[
|U_{e2}|^2=0.3091424594,
]

[
|U_{e3}|^2=0.0228086760.
]

The OPH particle file gives the PMNS angles, (\delta_{\rm PMNS}), and the physical Majorana pair on the weighted-cycle branch. 

### Beta-decay endpoint

[
m_\beta
=======

\left(
\sum_i |U_{ei}|^2m_i^2
\right)^{1/2}
]

gives

[
\boxed{
m_\beta^{\rm OPH}=0.0196244\ {\rm eV}.
}
]

KATRIN’s 2025 direct limit is still far above this scale:

[
m_\nu<0.45\ {\rm eV}
\quad
(90%{\rm \ CL}).
]

([Science][3])

### Neutrinoless double beta

[
m_{\beta\beta}
==============

\left|
c_{12}^2c_{13}^2m_1
+
s_{12}^2c_{13}^2m_2e^{i\alpha_{21}}
+
s_{13}^2m_3e^{i(\alpha_{31}-2\delta)}
\right|.
]

Depending on the PMNS/Majorana phase convention used for the (U_{e3}^2) term, the OPH branch gives roughly

[
\boxed{
m_{\beta\beta}^{\rm OPH}
\simeq
6.18\text{--}7.98\ {\rm meV}.
}
]

That is below present experimental reach but is useful because it ties the cosmological mass sum, PMNS branch, and Majorana branch into one consistency target.

---

## 9.8 Direct CνB detection pattern

For a PTOLEMY-style relic-neutrino capture experiment on tritium, OPH predicts three electron-capture line weights:

| State   |    Offset controlled by | Relative electron-flavor weight |
| ------- | ----------------------: | ------------------------------: |
| (\nu_1) | (m_1=17.455\ {\rm meV}) |                          0.6680 |
| (\nu_2) | (m_2=19.482\ {\rm meV}) |                          0.3091 |
| (\nu_3) | (m_3=53.075\ {\rm meV}) |                          0.0228 |

So the ultimate direct-detection receipt is:

[
\boxed{
\text{three capture features with OPH mass offsets and OPH electron-flavor weights.}
}
]

The exact detectability depends on energy resolution, target mass, local clustering, and neutrino nature. The key OPH point is that the branch predicts a **line pattern**, not just a total CνB density.

---

## 9.9 OPH anomaly-sector coupling

The dark-sector paper already says the static galaxy law cannot simply be inserted into FLRW perturbation theory; the cosmological branch must declare a homogeneous anomaly abundance and a perturbation kernel, because an exactly homogeneous background has no preferred baryonic acceleration vector. 

So the CνB extension should be written as a clean interaction ansatz:

[
\nabla_\mu T_\nu^{\mu\nu}
=========================

Q^\nu_{\nu A},
]

[
\nabla_\mu T_A^{\mu\nu}
=======================

-Q^\nu_{\nu A}.
]

Baseline:

[
\boxed{
Q^\nu_{\nu A}=0.
}
]

Extended OPH:

[
Q^\nu_{\nu A}
=============

\mathcal F_{\rm OPH}
\left(
S_{\rm coh},
B_A(k,a),
\Gamma_{\rm rec}(k,a),
m_i
\right).
]

The dark-sector paper’s Boltzmann contract already requires a module exposing background density, equation of state, sound speed, anisotropic stress, exchange current, response kernel (B_A(k,a)), and relaxation kernel (\Gamma_{\rm rec}(k,a)), with the OPH neutrino mass sum in the same run. 

For the CνB paper, add the neutrino extension variables:

[
\boxed{
\Delta N_{\rm eff}^{\rm coh},
\quad
\Gamma_{\nu{\rm coh}}(k,a),
\quad
Q^\mu_{\nu A},
\quad
B_{\nu A}(k,a).
}
]

---

## 9.10 Boltzmann-module contract

The paper-grade implementation should expose two switches.

### Baseline switch

```text
oph_cnb_baseline = true
```

with

[
N_{\rm eff}=3.044,
]

[
m_i=
(0.017454720257976796,,
0.019481987935919015,,
0.05307522145074924)\ {\rm eV},
]

[
c_{\rm eff}^2=c_{\rm vis}^2=\frac13.
]

### Extension switch

```text
oph_cnb_coherence_extension = optional
```

with

[
\Delta N_{\rm eff}^{\rm coh},
\qquad
\Gamma_{\nu{\rm coh}}(k,a),
\qquad
Q^\mu_{\nu A},
\qquad
B_{\nu A}(k,a).
]

The module must output:

[
C_\ell^{TT},\ C_\ell^{TE},\ C_\ell^{EE},
\quad
C_L^{\phi\phi},
\quad
P(k,z),
\quad
f\sigma_8(z),
\quad
S_8,
\quad
H(z),
\quad
D_A(z),
\quad
r_d.
]

The dark-sector paper already calls for a CAMB or CLASS anomaly module, Planck/ACT spectra, CMB lensing, BAO, supernovae, weak lensing, RSD, SPARC, and cluster map likelihoods under the same nuisance treatment used for (\Lambda{\rm CDM}) and (w_0w_a). 

---

## 9.11 Observational scorecard

| Test                                         |     OPH-CνB prediction | Current status                                                                                                                                                                |
| -------------------------------------------- | ---------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (N_{\rm eff})                                |                (3.044) | Consistent with Planck+BAO (N_{\rm eff}=2.99\pm0.17). ([arXiv][4])                                                                                                            |
| (\sum m_\nu)                                 | (0.09001193\ {\rm eV}) | Below Planck+BAO (\sum m_\nu<0.12\ {\rm eV}). ([arXiv][4])                                                                                                                    |
| (\sum m_\nu) vs ACT DR6                      | (0.09001193\ {\rm eV}) | Slight pressure: ACT DR6 extended-model analysis quotes (\sum m_\nu<0.082\ {\rm eV}) at 95% CL, while also finding no evidence for nonstandard neutrino physics. ([arXiv][5]) |
| (\sum m_\nu) vs DESI DR2 strict (\Lambda)CDM | (0.09001193\ {\rm eV}) | Tension: DESI DR2 reports (\sum m_\nu<0.0642\ {\rm eV}) under (\Lambda)CDM with three degenerate neutrino states. ([arXiv][6])                                                |
| (\sum m_\nu) vs DESI DR2 (w_0w_a)CDM         | (0.09001193\ {\rm eV}) | Allowed: DESI DR2 reports (\sum m_\nu<0.163\ {\rm eV}) in (w_0w_a)CDM. ([arXiv][6])                                                                                           |
| lightest mass (m_1)                          |    (0.01745\ {\rm eV}) | Within DESI DR2’s oscillation-informed (m_l<0.023\ {\rm eV}) statement. ([arXiv][6])                                                                                          |
| small-scale suppression                      |             (\sim5.4%) | Directly testable in CMB lensing, weak lensing, RSD, and galaxy clustering.                                                                                                   |

CMB-S4 is an especially important future test because its science program targets light relics, neutrino mass, CMB lensing, and dark-sector measurements. ([cmb-s4.org][7]) Euclid is also important because it maps the large-scale structure of the Universe over billions of galaxies and 10 billion years of cosmic history. ([science.esa.int][8])

---

## 9.12 Falsifier matrix

| Observable                 |         Baseline OPH-CνB prediction | Failure mode                                                               |
| -------------------------- | ----------------------------------: | -------------------------------------------------------------------------- |
| (N_{\rm eff})              |                             (3.044) | robust (N_{\rm eff}\neq3.044) with no declared OPH light channel           |
| (\sum m_\nu)               |              (0.09001193\ {\rm eV}) | robust bound below (0.09\ {\rm eV}) in a model respecting oscillation data |
| mass ordering              |              normal, non-degenerate | confirmed inverted ordering or quasi-degenerate spectrum                   |
| free-streaming phase shift |             standard free-streaming | CMB phase shift incompatible with (c_{\rm vis}^2=1/3)                      |
| neutrino self-interaction  |                                none | robust nonzero (\Gamma_{\nu{\rm int}}) without OPH extension               |
| small-scale (P(k))         |              (\sim5.4%) suppression | absent suppression or wrong scale dependence                               |
| (m_\beta)                  |                  (19.62\ {\rm meV}) | beta endpoint measurement incompatible with OPH mass vector                |
| (m_{\beta\beta})           |      (6.18\text{--}7.98\ {\rm meV}) | 0νββ signal outside OPH mass/phase band, if Majorana                       |
| direct CνB capture         | three-line pattern with OPH weights | wrong capture-line offsets or weights after direct detection               |
| anomaly coupling           |          (Q^\mu_{\nu A}=0) baseline | nonzero coupling required by data but not derivable from OPH microphysics  |

---

## 9.13 Research output to add now

The final track can be summarized as:

[
\boxed{
N_{\rm eff}^{\rm OPH}=3.044,
\quad
\sum m_\nu^{\rm OPH}=0.0900119296\ {\rm eV},
\quad
f_\nu^{\rm OPH}=0.006744,
\quad
\Delta P/P\simeq-5.40%.
}
]

Add this as a named lane:

[
\boxed{
\textbf{OPH-C}\nu\textbf{B: thermal relic background with weighted-cycle masses.}
}
]

And add the extension rule:

[
\boxed{
\Delta N_{\rm eff}^{\rm coh},
\Gamma_{\nu{\rm coh}},
Q^\mu_{\nu A},
B_{\nu A}
\text{ are zero unless separately derived.}
}
]

This gives OPH a clean, high-value cosmological test: it is compatible with standard (N_{\rm eff}), makes a sharp absolute mass prediction, predicts a specific free-streaming/growth signature, and can be tested by Planck/ACT/DESI/CMB-S4/Euclid-class likelihoods without inventing a new fit parameter.

[1]: https://ccwww.kek.jp/pdg/2025/reviews/rpp2025-rev-neutrinos-in-cosmology.pdf?utm_source=chatgpt.com "26. Neutrinos in Cosmology"
[2]: https://arxiv.org/abs/astro-ph/0310198 "[astro-ph/0310198] Signatures of Relativistic Neutrinos in CMB Anisotropy and Matter Clustering"
[3]: https://www.science.org/doi/10.1126/science.adq9592?utm_source=chatgpt.com "Direct neutrino-mass measurement based on 259 days of ..."
[4]: https://arxiv.org/abs/1807.06209?utm_source=chatgpt.com "Planck 2018 results. VI. Cosmological parameters"
[5]: https://arxiv.org/abs/2503.14454?utm_source=chatgpt.com "The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models"
[6]: https://arxiv.org/abs/2503.14744?utm_source=chatgpt.com "Constraints on Neutrino Physics from DESI DR2 BAO and DR1 Full Shape"
[7]: https://cmb-s4.org/science/the-dark-universe/?utm_source=chatgpt.com "the dark universe"
[8]: https://www.science.esa.int/web/cm25/euclid?utm_source=chatgpt.com "Euclid - ESA Science Programme - European Space Agency"


Yes. The next useful move is to turn the OPH-CνB lane into a **closed mini-Boltzmann model** with fixed neutrino masses and only clearly declared optional OPH channels.

The clean baseline is:

[
\boxed{
\text{OPH-C}\nu\text{B}*0:
\quad
N*{\rm eff}=3.044,\qquad
\Gamma_{\nu,\rm OPH}=0,\qquad
\Delta N_{\rm eff}^{\rm coh}=0,
}
]

with the OPH particle branch supplying only the absolute neutrino mass spectrum:

[
(m_1,m_2,m_3)
=============

(0.017454720257976796,,
0.019481987935919015,,
0.05307522145074924)\ {\rm eV}.
]

That mass triple and the splittings are explicitly emitted by the weighted-cycle neutrino branch.  The same corpus marks the neutrino row as theorem-grade only on that declared weighted-cycle branch, with bridge/adapter sidecars not feeding back into the theorem state. 

## 1. Background CνB distribution

Use comoving momentum

[
q\equiv \frac{p}{T_{\nu,0}/a},
]

and the zero-chemical-potential relic Fermi-Dirac distribution

[
f_0(q)=\frac{1}{e^q+1}.
]

For each mass eigenstate (i),

[
\epsilon_i(q,a)=\sqrt{q^2+y_i(a)^2},
\qquad
y_i(a)=\frac{a m_i}{T_{\nu,0}}.
]

Then

[
\rho_{\nu_i}(a)
===============

\frac{g_\nu T_{\nu,0}^4}{2\pi^2a^4}
\int_0^\infty dq,q^2,\epsilon_i(q,a),f_0(q),
]

[
P_{\nu_i}(a)
============

\frac{g_\nu T_{\nu,0}^4}{6\pi^2a^4}
\int_0^\infty dq,\frac{q^4}{\epsilon_i(q,a)},f_0(q),
]

with (g_\nu=2) for neutrino plus antineutrino of one mass eigenstate.

The total neutrino background is

[
\rho_\nu(a)=\sum_{i=1}^3\rho_{\nu_i}(a),
\qquad
P_\nu(a)=\sum_{i=1}^3P_{\nu_i}(a),
\qquad
w_\nu(a)=\frac{P_\nu(a)}{\rho_\nu(a)}.
]

The usual radiation bookkeeping is

[
\rho_r
======

\rho_\gamma
\left[
1+\frac78\left(\frac{4}{11}\right)^{4/3}N_{\rm eff}
\right],
]

with the Standard Model prediction (N_{\rm eff}\simeq 3.044). PDG summarizes the same cosmological-neutrino bookkeeping and notes that most massive-neutrino effects depend mainly on (\sum m_\nu), while a few free-streaming features depend on individual (m_i). ([Kek][1])

For an OPH extra relativistic coherence channel, the honest parameterization is

[
\boxed{
N_{\rm eff}^{\rm OPH}
=====================

3.044+\Delta N_{\rm eff}^{\rm coh}
}
]

where

[
\Delta N_{\rm eff}^{\rm coh}
============================

# \frac{\rho_{\rm coh,rel}}{\rho_{\nu,1}}

\frac87
\left(\frac{11}{4}\right)^{4/3}
\frac{\rho_{\rm coh,rel}}{\rho_\gamma}
\simeq
4.4032,\frac{\rho_{\rm coh,rel}}{\rho_\gamma}.
]

Baseline OPH-CνB sets (\rho_{\rm coh,rel}=0). That matters because the (\chi_\nu) coherent-matter susceptibility in the OPH stack is a declared continuation channel, not automatically an early-universe dark-radiation species; the (\chi_\nu) file says the continuation couples only to the OPH-native coherent scalar, not undifferentiated rest mass or heat. 

## 2. OPH mass sum and density fraction

The mass sum is

[
\boxed{
\sum_i m_i
==========

0.09001192964464505\ {\rm eV}.
}
]

Using the standard late-time relation

[
\Omega_\nu h^2
\simeq
\frac{\sum_i m_i}{93.14\ {\rm eV}},
]

we get

[
\boxed{
\Omega_\nu h^2
==============

9.6641539\times 10^{-4}.
}
]

For (h=0.674),

[
\boxed{
\Omega_\nu
==========

0.002127375.
}
]

Using the OPH dark-sector compressed cosmology row

[
\Omega_m=0.315905207,
]

the neutrino matter fraction is

[
\boxed{
f_\nu
=====

# \frac{\Omega_\nu}{\Omega_m}

0.00673422.
}
]

The dark-sector paper’s local Boltzmann plumbing test explicitly uses CAMB, the OPH neutrino mass sum, and compressed (\Omega_m,\sigma_8,S_8) rows, while warning that a publication-grade calculation needs the OPH (B_A(k,a)) kernel and full covariances. 

Mass-eigenstate fractions are:

[
f_{\nu_i}=f_\nu\frac{m_i}{\sum_jm_j},
]

so

[
(f_{\nu_1},f_{\nu_2},f_{\nu_3})
===============================

(0.00130587,\ 0.00145754,\ 0.00397081).
]

The heavy state carries about (59%) of the total neutrino mass density.

## 3. Nonrelativistic transition redshifts

A useful analytic marker is

[
\langle p\rangle = 3.151,T_\nu,
]

so the transition occurs near

[
3.151,T_{\nu,0}(1+z_{{\rm nr},i})=m_i.
]

With

[
T_{\nu,0}
=========

\left(\frac{4}{11}\right)^{1/3}T_{\gamma,0}
\simeq
1.94537\ {\rm K}
\simeq
1.67639\times10^{-4}\ {\rm eV},
]

[
3.151T_{\nu,0}\simeq5.28230\times10^{-4}\ {\rm eV}.
]

Therefore

[
z_{{\rm nr},i}
==============

\frac{m_i}{3.151T_{\nu,0}}-1.
]

Numerically:

| state   | (m_i) [eV] | (z_{\rm nr}) |        present thermal speed |
| ------- | ---------: | -----------: | ---------------------------: |
| (\nu_1) |  0.0174547 |        32.04 | (9.07\times10^3\ {\rm km/s}) |
| (\nu_2) |  0.0194820 |        35.88 | (8.13\times10^3\ {\rm km/s}) |
| (\nu_3) |  0.0530752 |        99.48 | (2.98\times10^3\ {\rm km/s}) |

So all three OPH neutrinos are still relativistic at recombination, but become nonrelativistic before today. This makes the prediction observationally clean: **early CMB mostly sees (N_{\rm eff}); late growth and lensing see (\sum m_\nu) and the free-streaming kernel.**

## 4. Free-streaming scale

For nonrelativistic thermal neutrinos,

[
v_{{\rm th},i}(z)
\simeq
151(1+z)
\left(\frac{1\ {\rm eV}}{m_i}\right)
{\rm km/s}.
]

The comoving free-streaming wavenumber is approximately

[
k_{{\rm FS},i}(z)
\simeq
0.810,
\frac{E(z)}{(1+z)^2}
\left(\frac{m_i}{1\ {\rm eV}}\right)
h,{\rm Mpc}^{-1},
]

where

[
E(z)=\frac{H(z)}{H_0}
=====================

\left[
\Omega_\Lambda+\Omega_m(1+z)^3+\Omega_r(1+z)^4
\right]^{1/2}.
]

This is the scale below which neutrino clustering is erased by thermal motion; Lesgourgues’ review emphasizes that the matter power spectrum gets a scale-dependent imprint from massive-neutrino free-streaming. ([arXiv][2])

At (z=0), using the same (\Omega_m) value as above:

| state   | (k_{{\rm FS},0}) ([h,{\rm Mpc}^{-1}]) | (\lambda_{{\rm FS},0}=2\pi/k) ([h^{-1}{\rm Mpc}]) |
| ------- | ------------------------------------: | ------------------------------------------------: |
| (\nu_1) |                               0.01414 |                                               444 |
| (\nu_2) |                               0.01578 |                                               398 |
| (\nu_3) |                               0.04299 |                                               146 |

The nonrelativistic-turnover wavenumber is approximately

[
k_{{\rm nr},i}
\simeq
0.018,\Omega_m^{1/2}
\left(\frac{m_i}{1\ {\rm eV}}\right)^{1/2}
h,{\rm Mpc}^{-1},
]

giving

[
(k_{{\rm nr},1},k_{{\rm nr},2},k_{{\rm nr},3})
\simeq
(0.00134,\ 0.00141,\ 0.00233)
h,{\rm Mpc}^{-1}.
]

That means the OPH spectrum predicts a broad, low-amplitude normal-hierarchy free-streaming feature, not a sharp single cutoff.

## 5. Three-step suppression kernel

A useful analytic approximation is to treat each mass eigenstate as contributing suppression only for modes with

[
k\gtrsim k_{{\rm FS},i}.
]

A smooth version is

[
W_i(k,a)
========

\frac{k^2}{k^2+k_{{\rm FS},i}^2(a)}.
]

Then a first-order OPH-CνB matter-power suppression model is

[
\boxed{
\frac{\Delta P_m(k,a)}{P_m(k,a)}
\simeq
-8\sum_{i=1}^3 f_{\nu_i}W_i(k,a).
}
]

At (z=0), this gives the limiting steps:

| (k) range                      | free-streaming states | approximate (\Delta P/P) |
| ------------------------------ | --------------------- | -----------------------: |
| (k\ll0.014\ h{\rm Mpc}^{-1})   | none                  |                     (0%) |
| (0.014\lesssim k\lesssim0.016) | (\nu_1)               |                 (-1.04%) |
| (0.016\lesssim k\lesssim0.043) | (\nu_1,\nu_2)         |                 (-2.21%) |
| (k\gg0.043)                    | all three             |         (\boxed{-5.39%}) |

This is not a substitute for CAMB/CLASS, but it is the right analytic “fingerprint” to look for. The exact calculation must integrate the massive-neutrino Boltzmann hierarchy.

## 6. Growth equation

On scales well below the relevant free-streaming scale, neutrino perturbations are suppressed, so CDM+baryon growth approximately obeys

[
\frac{d^2\delta_{cb}}{d(\ln a)^2}
+
\frac12\frac{d\delta_{cb}}{d\ln a}
----------------------------------

# \frac32(1-f_\nu)\delta_{cb}

0
]

during matter domination.

Try

[
\delta_{cb}\propto a^p.
]

Then

[
p^2+\frac12p-\frac32(1-f_\nu)=0,
]

so

[
p
=

\frac{\sqrt{25-24f_\nu}-1}{4}.
]

With

[
f_\nu=0.00673422,
]

[
\boxed{
p=0.995953.
}
]

Equivalently,

[
\delta_{cb}\propto a^{0.995953}
]

instead of (a^1). The exponent shift is

[
1-p=0.004047.
]

The corresponding small-scale power suppression is the standard

[
\boxed{
\Delta P/P\simeq -8f_\nu=-5.39%,
}
]

and the rough amplitude suppression is

[
\boxed{
\Delta\sigma_8/\sigma_8\simeq -4f_\nu=-2.69%.
}
]

## 7. Full Boltzmann form

For publication-grade work, evolve the distribution perturbation

[
f_i(\mathbf k,q,\mu,\tau)
=========================

f_0(q)\left[1+\Psi_i(\mathbf k,q,\mu,\tau)\right],
]

with

[
\Psi_i
======

\sum_{\ell=0}^{\infty}
(-i)^\ell(2\ell+1)\Psi_{i\ell}P_\ell(\mu).
]

In conformal Newtonian gauge,

[
ds^2=a^2(\tau)
\left[
-(1+2\psi)d\tau^2+(1-2\phi)d\mathbf x^2
\right].
]

The massive-neutrino Boltzmann equation is

[
\Psi_i'
+
i\frac{q}{\epsilon_i}k\mu,\Psi_i
+
\frac{d\ln f_0}{d\ln q}
\left[
\phi'
-----

i\frac{\epsilon_i}{q}k\mu,\psi
\right]
=0.
]

The multipoles obey

[
\Psi_{i0}'
==========

## -\frac{qk}{\epsilon_i}\Psi_{i1}

\phi'
\frac{d\ln f_0}{d\ln q},
]

[
\Psi_{i1}'
==========

\frac{qk}{3\epsilon_i}
\left(
\Psi_{i0}-2\Psi_{i2}
\right)
-------

\frac{\epsilon_i k}{3q}
\psi
\frac{d\ln f_0}{d\ln q},
]

[
\Psi_{i\ell}'
=============

\frac{qk}{(2\ell+1)\epsilon_i}
\left[
\ell\Psi_{i,\ell-1}
-------------------

(\ell+1)\Psi_{i,\ell+1}
\right],
\qquad
\ell\ge2.
]

The density perturbation is

[
\delta\rho_{\nu_i}
==================

\frac{g_\nu T_{\nu,0}^4}{2\pi^2a^4}
\int dq,q^2,\epsilon_i f_0(q)\Psi_{i0}.
]

The pressure perturbation is

[
\delta P_{\nu_i}
================

\frac{g_\nu T_{\nu,0}^4}{6\pi^2a^4}
\int dq,\frac{q^4}{\epsilon_i} f_0(q)\Psi_{i0}.
]

This is the exact place to insert the OPH mass triple and test the CνB prediction against CMB lensing, BAO, full-shape galaxy power, weak lensing, and RSD.

## 8. OPH dark-sector coupling surface

The dark-sector paper already says the cosmology branch needs a Boltzmann module exposing

[
\bar\rho_A(a),\
\bar\rho_{A,\rm eq}(a),\
w_A(a),\
c_{s,A}^2(k,a),\
\sigma_A(k,a),\
Q_A^\mu,\
B_A(k,a),\
\Gamma_{\rm rec}(k,a),
]

with the OPH neutrino mass sum in the same run. 

The anomaly branch has two background possibilities:

[
\rho_A'+3\mathcal H\rho_A=0,
\qquad
\rho_A(a)=\rho_{A0}a^{-3},
]

or

[
\rho_A'+3\mathcal H\rho_A
=========================

-a\Gamma_{\rm rec}
(\rho_A-\rho_{A,\rm eq}).
]

The perturbative repair branch defines

[
q_A=\frac{\rho_{A,\rm eq}}{\rho_A},
\qquad
\delta_{A,\rm eq}(k,a)=B_A(k,a)\delta_b(k,a),
]

and evolves

[
\delta_A'
=========

-\theta_A+3\Phi'
-a\Gamma_{\rm rec}q_A
(\delta_A-\delta_{A,\rm eq}),
]

[
\theta_A'
=========

-\mathcal H\theta_A+k^2\Psi.
]

Those equations are already in the OPH dark-sector cosmology contract. 

So the combined OPH cosmology calculation should be:

[
\boxed{
\text{Einstein-Boltzmann}
+
\text{fixed OPH }m_i
+
\text{optional }A\text{-sector }(\rho_A,B_A,\Gamma_{\rm rec})
+
\text{optional }\Delta N_{\rm eff}^{\rm coh}.
}
]

## 9. Minimal falsifiable OPH-CνB scorecard

The clean research track is:

[
\boxed{
N_{\rm eff}=3.044,
\quad
\sum m_\nu=0.09001193\ {\rm eV},
\quad
\Omega_\nu h^2=9.664\times10^{-4},
\quad
f_\nu=0.006734,
}
]

[
\boxed{
z_{\rm nr}=(32.04,\ 35.88,\ 99.48),
}
]

[
\boxed{
k_{{\rm FS},0}
==============

(0.01414,\ 0.01578,\ 0.04299)
,h,{\rm Mpc}^{-1},
}
]

[
\boxed{
\Delta P/P\big|*{k\gg k*{\rm FS}}
\simeq
-5.39%.
}
]

The key mathematical win is that OPH’s neutrino contribution is **not a fit parameter** once the weighted-cycle branch is accepted. It becomes a fixed vector in the cosmology likelihood.

The next concrete computation is to run two likelihood variants:

[
\mathcal M_0:
\quad
{N_{\rm eff}=3.044,\ m_i=m_i^{\rm OPH},\ B_A=0,\ \Gamma_{\rm rec}=0},
]

and

[
\mathcal M_A:
\quad
{N_{\rm eff}=3.044,\ m_i=m_i^{\rm OPH},\ B_A(k,a),\ \Gamma_{\rm rec}(k,a)\ \text{from OPH dark-sector closure}}.
]

Then compare

[
\Delta\chi^2
============

## \chi^2(\mathcal M_{\rm OPH})

\chi^2(\Lambda{\rm CDM}+\sum m_\nu\ {\rm free})
]

under the same data vector and covariance. The existing OPH dark-sector document is explicit that the compressed CAMB row is only plumbing and that full likelihoods need the OPH (B_A(k,a)) kernel and covariances. 

[1]: https://ccwww.kek.jp/pdg/2025/reviews/rpp2025-rev-neutrinos-in-cosmology.pdf?utm_source=chatgpt.com "26. Neutrinos in Cosmology"
[2]: https://arxiv.org/pdf/1212.6154?utm_source=chatgpt.com "Neutrino mass from Cosmology"
---

Yes. I evaluated the **first concrete proxy computation**: fixed OPH neutrino masses inside a compressed cosmology scorecard, plus the amount of extra OPH anomaly response needed to relieve the weak-lensing (S_8) pressure.

This is **not** a full Planck/ACT/DESI likelihood. It is the honest next step before that: a diagonal compressed likelihood and kernel-target calculation.

# 1. Fixed OPH-CνB input

The OPH neutrino branch gives

[
(m_1,m_2,m_3)=
(0.017454720257976796,,
0.019481987935919015,,
0.05307522145074924)\ {\rm eV},
]

so

[
\boxed{
\sum m_\nu^{\rm OPH}=0.09001192964464505\ {\rm eV}.
}
]

The same OPH particle file marks this as the weighted-cycle absolute neutrino branch and gives the corresponding mass splittings. 

Using the standard relic-neutrino density relation,

[
\Omega_\nu =
\frac{\sum m_\nu}{93.12,h^2{\rm eV}},
]

and (h=0.674), I get

[
\boxed{
\Omega_\nu h^2 = 9.664\times10^{-4},
}
]

[
\boxed{
\Omega_\nu =0.00212737,
}
]

[
\boxed{
f_\nu=\frac{\Omega_\nu}{\Omega_m}=0.00673422
}
]

for (\Omega_m=0.315905207). PDG gives the same density formula and (n_{\nu,0}=339.5,{\rm cm^{-3}}). 

The nonrelativistic transition redshifts are

[
z_{{\rm nr},i}
==============

\frac{m_i}{3.151T_{\nu,0}}-1,
]

giving

[
\boxed{
z_{\rm nr}=(32.04,\ 35.88,\ 99.48).
}
]

PDG uses the same criterion, (\langle p\rangle=3.15T), for the nonrelativistic transition. 

# 2. Boltzmann-lite free-streaming kernel

For the present-day free-streaming scale I used

[
k_{{\rm FS},0,i}
\simeq
0.810
\left(\frac{m_i}{1{\rm eV}}\right)
h,{\rm Mpc}^{-1}.
]

This gives

| state   | (m_i) [eV] | (k_{{\rm FS},0}) ([h,{\rm Mpc}^{-1}]) | (\lambda_{{\rm FS},0}=2\pi/k) ([h^{-1}{\rm Mpc}]) |
| ------- | ---------: | ------------------------------------: | ------------------------------------------------: |
| (\nu_1) |  0.0174547 |                               0.01414 |                                               444 |
| (\nu_2) |  0.0194820 |                               0.01578 |                                               398 |
| (\nu_3) |  0.0530752 |                               0.04299 |                                               146 |

The matter-suppression approximation was

[
\frac{\Delta P(k)}{P(k)}
\simeq
-8\sum_i f_{\nu_i}
\frac{k^2}{k^2+k_{{\rm FS},i}^2}.
]

Numerically:

| (k) ([h,{\rm Mpc}^{-1}]) | (\Delta P/P) |
| -----------------------: | -----------: |
|                    0.005 |     (-0.26%) |
|                    0.010 |     (-0.85%) |
|                    0.014 |     (-1.34%) |
|                    0.016 |     (-1.56%) |
|                    0.020 |     (-1.98%) |
|                    0.030 |     (-2.81%) |
|                    0.050 |     (-3.85%) |
|                    0.100 |     (-4.84%) |
|                    0.200 |     (-5.23%) |
|                    1.000 |     (-5.38%) |

So the asymptotic OPH neutrino suppression remains

[
\boxed{
\Delta P/P \simeq -8f_\nu = -5.39%.
}
]

This is the right analytic fingerprint: a broad, normal-ordering, three-step suppression that mostly behaves like a total-mass effect. PDG similarly notes that neutrino free streaming smooths neutrino perturbations on small scales and yields an approximate small-scale suppression ((1-8f_\nu)). 

# 3. Compressed OPH dark-sector scorecard

The OPH dark-sector paper already has a compressed CAMB plumbing row using the OPH neutrino mass sum. It reports that the diagonal compressed value is

[
\chi^2_{\rm diag}=11.463
]

for six rows and explicitly says this is a plumbing check, not a replacement for Planck/DESI/weak-lensing likelihoods; the same section says the publication-grade run needs an OPH (B_A(k,a)) kernel in a custom Boltzmann module and full covariances. 

I recomputed that diagonal score from the printed rows:

[
\chi^2_{\rm diag}
=================

\sum_j
\left(
\frac{x_j^{\rm OPH}-x_j^{\rm target}}{\sigma_j}
\right)^2
=========

\boxed{11.463261435}.
]

Breakdown:

| row                                  | OPH prediction |                 target |   pull | (\chi^2) contribution |
| ------------------------------------ | -------------: | ---------------------: | -----: | --------------------: |
| Planck (\Omega_m)                    |    0.315905207 |        (0.315\pm0.007) | +0.129 |                0.0167 |
| Planck (\sigma_8)                    |    0.807787208 |        (0.811\pm0.006) | -0.535 |                0.2867 |
| Planck (S_8)                         |    0.828924043 | (0.831027707\pm0.0111) | -0.190 |                0.0359 |
| DESI DR1 BAO (\Omega_m)              |    0.315905207 |        (0.295\pm0.015) | +1.394 |                1.9423 |
| DESI DR1 BAO/BBN/(\theta_\ast) (H_0) |           67.4 |         (68.52\pm0.62) | -1.806 |                3.2633 |
| weak-lensing (S_8)                   |    0.828924043 |        (0.790\pm0.016) | +2.433 |                5.9183 |

The pressure points are therefore clear:

[
\boxed{
\chi^2_{S_8,{\rm WL}}=5.918,
\qquad
\chi^2_{H_0}=3.263,
\qquad
\chi^2_{\Omega_m,{\rm DESI}}=1.942.
}
]

The Planck-like rows are fine. The weak-lensing (S_8) row is the main problem.

# 4. What the OPH anomaly kernel must do

The diagnostic OPH value is

[
S_8^{\rm OPH}=0.828924043.
]

The weak-lensing compressed target is

[
S_8^{\rm WL}=0.790\pm0.016.
]

So the required amplitude response is

[
R_{\rm WL}
==========

# \frac{0.790}{0.828924043}

\boxed{0.9530426903}.
]

That means the OPH anomaly kernel must reduce the weak-lensing amplitude by

[
\boxed{
1-R_{\rm WL}=4.70%.
}
]

Since power scales roughly like amplitude squared,

[
R_{\rm WL}^2-1
==============

\boxed{-9.17%}
]

as a matter-power-level target on the weak-lensing-sensitive scales and redshifts.

This is the most concrete result of the computation:

[
\boxed{
B_A(k,a)\ \text{must supply roughly a }4.7%\text{ late-time amplitude reduction,}
}
]

or equivalently a roughly

[
\boxed{
9.2%\text{ power-level reduction}
}
]

on the weak-lensing kernel, while not ruining the Planck-like CMB rows.

# 5. Uniform suppression fails

I tested a one-parameter uniform growth rescaling,

[
\sigma_8 \mapsto r\sigma_8,
\qquad
S_8 \mapsto rS_8.
]

The best uniform fit is

[
\boxed{
r_{\rm best}=0.9984919412,
}
]

which barely changes the score:

[
\chi^2_{\rm diag}: 11.4633 \to 11.4033.
]

So a single scale-independent amplitude dial does **not** solve the pressure.

If we instead force the weak-lensing row to match exactly,

[
r=0.9530426903,
]

then the Planck growth rows blow up:

| row                | forced-(S_8^{\rm WL}) prediction |          pull |
| ------------------ | -------------------------------: | ------------: |
| Planck (\sigma_8)  |                         0.769856 | (-6.86\sigma) |
| Planck (S_8)       |                         0.790000 | (-3.70\sigma) |
| weak-lensing (S_8) |                         0.790000 |     (0\sigma) |

The total diagonal score becomes

[
\boxed{
\chi^2_{\rm diag}=65.91,
}
]

which is terrible.

So the anomaly kernel cannot be a uniform suppression of all growth observables. It has to be a **redshift-, scale-, or observable-kernel-dependent response**:

[
\boxed{
B_A(k,a)\neq \text{constant}.
}
]

The target shape is something like:

[
B_A(k,a)\approx
1-\epsilon_A,W_{\rm WL}(k,a),
]

with

[
\epsilon_A\simeq0.047
]

at the amplitude level on weak-lensing scales, while keeping the CMB-primary and CMB-lensing rows close to their existing values.

# 6. Neutrino-mass constraint score

The OPH mass sum is

[
\sum m_\nu^{\rm OPH}=0.09001193\ {\rm eV}.
]

Comparison with current constraints:

| dataset/model                                         |                                                        published constraint | OPH status                                       |
| ----------------------------------------------------- | --------------------------------------------------------------------------: | ------------------------------------------------ |
| Planck 2018 + BAO (\Lambda)CDM                        |                                                 (\sum m_\nu<0.12\ {\rm eV}) | allowed                                          |
| ACT DR6 extended-model summary                        |                                                (\sum m_\nu<0.082\ {\rm eV}) | mild pressure                                    |
| DESI DR2 strict (\Lambda)CDM, three degenerate states | (\sum m_\nu<0.0642\ {\rm eV}), quoted marginalized (\sigma=0.020\ {\rm eV}) | strong pressure in that parameterization         |
| DESI DR2 oscillation-informed lightest mass           |                                                       (m_l<0.023\ {\rm eV}) | allowed, since (m_1^{\rm OPH}=0.01745\ {\rm eV}) |
| DESI DR2 (w_0w_a)CDM                                  |                                                (\sum m_\nu<0.163\ {\rm eV}) | allowed                                          |

Planck gives (N_{\rm eff}=2.99\pm0.17) and (\sum m_\nu<0.12) eV when combined with BAO, while ACT DR6 reports (N_{\rm eff}=2.86\pm0.13), (N_{\rm eff}=2.89\pm0.11) with BBN, and (\sum m_\nu<0.082) eV at 95% CL. ([arXiv][1]) DESI DR2 reports the strict (\Lambda)CDM degenerate-neutrino bound (\sum m_\nu<0.0642) eV, but also says its oscillation-informed analysis gives (m_l<0.023) eV and that (w_0w_a)CDM relaxes the mass bound to (\sum m_\nu<0.163) eV. ([arXiv][2]) PDG’s 2025 review emphasizes that cosmological neutrino constraints are obtained within (\Lambda)CDM or simple extensions and warns that discrepant data combinations should be treated cautiously. 

A useful proxy table:

| comparison                                                |       proxy pull/penalty |
| --------------------------------------------------------- | -----------------------: |
| (N_{\rm eff}^{\rm OPH}=3.044) vs Planck+BAO (2.99\pm0.17) |            (+0.32\sigma) |
| (N_{\rm eff}^{\rm OPH}=3.044) vs ACT DR6 (2.86\pm0.13)    |            (+1.42\sigma) |
| (\sum m_\nu^{\rm OPH}=0.0900) vs Planck+BAO (<0.12)       |               inside 95% |
| (\sum m_\nu^{\rm OPH}=0.0900) vs ACT (<0.082)             | slightly above 95% bound |
| (\sum m_\nu^{\rm OPH}=0.0900) vs DESI strict (<0.0642)    |     above by (0.0258) eV |
| (m_1^{\rm OPH}=0.01745) vs DESI (m_l<0.023)               |               inside 95% |

# 7. Result of the computation

The concrete score is:

[
\boxed{
\chi^2_{\rm diag}({\rm OPH\ flat\ anomaly+C}\nu{\rm B})
=======================================================

11.4633
\quad
\text{for six compressed rows.}
}
]

The fixed OPH neutrino branch itself predicts

[
\boxed{
\sum m_\nu=0.09001193\ {\rm eV},
\quad
f_\nu=0.006734,
\quad
\Delta P/P|_{\rm small\ scale}\simeq-5.39%.
}
]

The anomaly-sector target is now numerically sharp:

[
\boxed{
R_{\rm WL}=0.9530\pm0.0193
}
]

or

[
\boxed{
\Delta P_A/P_A\simeq -9.2%\pm3.7%
}
]

on weak-lensing-sensitive modes, **without** applying the same uniform suppression to Planck’s inferred growth rows.

So the next theorem target is not vague anymore:

[
\boxed{
\text{Derive an OPH }B_A(k,a)\text{ that suppresses late weak-lensing }S_8
\text{ by about }4.7%\text{ in amplitude,}
}
]

while preserving

[
\boxed{
\Omega_m\simeq0.316,
\quad
H_0\simeq67.4,
\quad
\sigma_8^{\rm CMB}\simeq0.808,
\quad
N_{\rm eff}=3.044,
\quad
\sum m_\nu=0.0900\ {\rm eV}.
}
]

That is the clean mathematical pressure point for OPH cosmology.

[1]: https://arxiv.org/abs/1807.06209 "[1807.06209] Planck 2018 results. VI. Cosmological parameters"
[2]: https://arxiv.org/abs/2503.14744 "[2503.14744] Constraints on Neutrino Physics from DESI DR2 BAO and DR1 Full Shape"
