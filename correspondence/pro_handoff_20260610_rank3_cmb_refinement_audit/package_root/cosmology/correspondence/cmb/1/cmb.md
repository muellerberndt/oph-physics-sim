The clean target is **not “predict the CMB directly from the current OPH-FPE run.”** The better target is:

[
\boxed{
\text{OPH screen/freezeout covariance}
;\rightarrow;
P_{\mathcal R}^{\rm OPH}(k)
;\rightarrow;
\text{Boltzmann transfer}
;\rightarrow;
TT,TE,EE,BB,\phi\phi,\text{LSS}.
}
]

That matches the boundary in your draft: build an OPH screen/freezeout effective theory first, then progressively replace phenomenological kernels with simulator-derived kernels.  The OPH basis for starting on (S^2) is strong: the screen is the observer-facing chart over finite patch carriers, caps/collars encode support-visible cuts, and (\mathrm{Conf}(S^2)\simeq SO^+(3,1)) supplies the kinematic bridge to emergent (3+1)D spacetime. 

## 1. The list of CMB properties OPH should predict

Here is the prediction ledger I would use.

| Layer                          | What to predict                                                                                                                             | OPH object                                                                         | Data product                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Screen covariance**          | (C_\ell^\chi), (D_\ell^\chi), tilt, finite-capacity window, low-(\ell) suppression, parity envelope                                         | freezeout synchronization / repair-load field (\chi(\hat n))                       | analytic (C_\ell^\chi), simulator-derived parameters |
| **Primordial scalar spectrum** | (A_s), (n_s), running (\alpha_s), IR cutoff, low-(k) modulation                                                                             | (P_{\mathcal R}^{\rm OPH}(k))                                                      | input to CAMB/CLASS                                  |
| **Primary temperature**        | (C_\ell^{TT}), acoustic peak positions, peak heights, damping tail, low-(\ell) behavior                                                     | OPH initial field + photon-baryon transfer                                         | Planck/ACT TT spectra                                |
| **Polarization**               | (C_\ell^{TE}), (C_\ell^{EE}), reionization bump, low-(\ell) parity/correlation tests                                                        | same scalar initial field + Thomson scattering transfer                            | Planck low-(\ell) EE, high-(\ell) TE/EE              |
| **B-modes**                    | lensing (BB); primordial tensor (BB) only if OPH derives a tensor screen mode                                                               | tensor/edge repair channel, if present                                             | (C_\ell^{BB}), tensor-to-scalar (r) constraint       |
| **CMB lensing**                | (C_\ell^{\phi\phi}), lensing smoothing of peaks, (A_L)-like diagnostics                                                                     | OPH anomaly/dark stress and matter growth                                          | Planck/ACT lensing                                   |
| **Large-angle anomalies**      | quadrupole, octopole, (S_{1/2}), odd/even parity, hemispherical asymmetry, quadrupole-octopole alignment                                    | global repair modes / off-diagonal covariance                                      | masked-sky Monte Carlo                               |
| **Statistical isotropy**       | BipoSH coefficients (A_{\ell\ell'}^{LM}), preferred axes, anisotropic covariance                                                            | residual cap-pair / holonomy / repair defects                                      | Planck component-separated maps                      |
| **Non-Gaussianity**            | (f_{\rm NL}^{\rm local,equil,orth}), defect bispectra, trispectrum/covariance excess                                                        | rare repair defects, non-linear freezeout                                          | Planck bispectrum/trispectrum                        |
| **Background cosmology**       | (\Omega_b h^2), (\Omega_A h^2) or (\Omega_c h^2), (H_0), (\Omega_\Lambda), (\Omega_K), (N_{\rm eff}^{\rm rel}), (\sum m_\nu), (Y_p), (\tau) | OPH matter/radiation/anomaly state selection + standard recombination/reionization | Planck + BAO + SNe                                   |
| **Late structure**             | (P(k)), (\sigma_8), (S_8), (f\sigma_8), BAO distances, weak-lensing amplitudes                                                              | OPH anomaly-stress Boltzmann module                                                | ACT/DESI/weak lensing/RSD                            |
| **Secondary CMB**              | ISW, tSZ, kSZ, Rees–Sciama, spectral distortions (\mu/y) if energy injection exists                                                         | late-time stress, reionization, heating                                            | ACT/SPT/PIXIE-like targets                           |

Two notation fixes matter immediately:

1. **Do not call finite freezeout capacity (N_{\rm eff}).** In CMB cosmology (N_{\rm eff}) already means effective relativistic species; Planck gives (N_{\rm eff}=2.99\pm0.17). Use (N_{\rm frz}), (N_{\rm cap}^{\rm eff}), or (N_{\rm cell}^{\rm frz}) for OPH freezeout capacity. ([arXiv][1])

2. **Fix the tilt sign.** Your draft says positive (\eta) gives (n_s<1), but the written exponent
   [
   C_\ell^\chi\propto [\ell(\ell+1)]^{-1+\eta/2}
   ]
   actually gives (D_\ell^\chi\propto \ell^{+\eta}), a blue tilt. For a red scalar tilt, define
   [
   \eta_R:=1-n_s>0
   ]
   and write
   [
   \boxed{
   C_\ell^\chi
   =
   \frac{A_\chi}
   {[\ell(\ell+1)+\mu^2]^{1+\eta_R/2}}
   \times \text{corrections}.
   }
   ]
   Then (D_\ell^\chi\propto \ell^{-\eta_R}), matching (n_s<1). Planck’s benchmark (n_s=0.965\pm0.004) gives the first OPH numerical target:
   [
   \boxed{
   \eta_R^{\rm OPH}=1-n_s\simeq 0.035\pm0.004.
   }
   ]
   ([arXiv][1])

## 2. Minimal OPH screen model to start predicting

Define the freezeout observer-consensus scalar:

[
\chi(\hat n)
============

\text{freezeout synchronization / repair-load / record-phase field on }S^2,
]

[
\chi(\hat n)=\sum_{\ell m}a_{\ell m}^{\chi}Y_{\ell m}(\hat n),
\qquad
C_\ell^\chi=
\frac{1}{2\ell+1}\sum_m|a_{\ell m}^{\chi}|^2.
]

Your draft’s MaxEnt result is the right first theorem: a local repair-cost action

[
S[\chi]=
\frac{1}{2\sigma^2}
\int_{S^2}d\Omega,|\nabla_{S^2}\chi|^2
]

gives the inverse spherical Laplacian,

[
\langle a_{\ell m}a^*_{\ell'm'}\rangle
======================================

\frac{A_\chi}{\ell(\ell+1)}
\delta_{\ell\ell'}\delta_{mm'},
]

so

[
D_\ell^\chi
\equiv
\frac{\ell(\ell+1)}{2\pi}C_\ell^\chi
\simeq \text{constant}.
]

That is the OPH screen analogue of scale invariance. 

The corrected working model should be:

[
\boxed{
C_\ell^\chi
===========

\frac{A_\chi}
{[\ell(\ell+1)+\mu^2]^{1+\eta_R/2}}
,W_\ell^2
,F_{\rm IR}(\ell)
,F_P(\ell)
+
N_\ell^{\rm cap}
}
]

with

[
W_\ell=
\exp!\left[-\frac{\ell(\ell+1)}{2\ell_{\rm cap}^2}\right],
]

[
F_{\rm IR}(\ell)
================

1-q_{\rm IR}
\exp!\left[
-\frac{\ell(\ell+1)}
{\ell_{\rm IR}(\ell_{\rm IR}+1)}
\right],
]

[
F_P(\ell)
=========

1+\epsilon_P(-1)^\ell
\exp(-\ell/\ell_P).
]

Parameter meanings:

| Parameter                               | Meaning                                  | Expected CMB imprint                     |
| --------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| (A_\chi)                                | screen-field amplitude                   | sets (A_s) after bridge normalization    |
| (\eta_R)                                | red anomalous repair dimension           | scalar tilt (n_s=1-\eta_R)               |
| (\mu)                                   | IR repair/capacity regulator             | lowest-(\ell) rollover                   |
| (\ell_{\rm cap})                        | effective freezeout cell/window scale    | UV smoothing or cutoff                   |
| (N_{\rm cap}^{\rm eff}) / (N_{\rm frz}) | effective freezeout capacity             | sampling noise, covariance floor         |
| (q_{\rm IR})                            | global repair suppression strength       | low quadrupole / large-angle suppression |
| (\ell_{\rm IR})                         | angular range of global repair mode      | which low multipoles are affected        |
| (\epsilon_P)                            | parity/cap-pair residual                 | odd/even power asymmetry                 |
| (\ell_P)                                | parity envelope width, not Planck length | low-(\ell)-localized parity effect       |
| (A_{\ell\ell'}^{LM})                    | off-diagonal BipoSH covariance           | alignment / hemispherical asymmetry      |
| (g_{\rm defect})                        | non-Gaussian repair-defect strength      | bispectrum/trispectrum signatures        |

The OPH consensus basis for treating repair as a real mathematical object is already in the stack: finite observer patches compare overlap data and local repair moves reconcile mismatch; under the stated local-fit, local-diamond, and repair-completeness clauses, the system reaches a schedule-independent quotient normal form. 

## 3. Bridge from screen covariance to physical CMB

The screen theory should not be compared directly to observed (TT). It should first be mapped to a primordial curvature spectrum:

[
\boxed{
P_{\mathcal R}^{\rm OPH}(k)
===========================

A_s
\left(\frac{k}{k_0}\right)^{-\eta_R}
F_{\rm IR}(k)
F_{\rm cap}(k)
F_{\rm defect}(k)
}
]

with the first approximation

[
\ell\simeq kD_*,
]

where (D_*) is the comoving distance to last scattering. This is exactly the architecture in your draft: screen covariance (\rightarrow) primordial bridge (\rightarrow) CAMB/CLASS (\rightarrow) (TT,TE,EE,\phi\phi,P(k)). 

For parity and BipoSH terms, be careful: a pure (k)-space scalar power spectrum cannot capture all angular repair defects. Those belong in the angular covariance:

[
\langle a_{\ell m}a^*_{\ell'm'}\rangle
======================================

C_\ell\delta_{\ell\ell'}\delta_{mm'}
+
\Delta_{\ell m,\ell'm'}^{\rm repair}.
]

A standard way to express (\Delta) is BipoSH:

[
\Delta_{\ell m,\ell'm'}
=======================

\sum_{LM}
A_{\ell\ell'}^{LM}
(-1)^{m'}
C^{LM}_{\ell m,\ell' -m'}.
]

That becomes the OPH signature for structured large-angle anisotropy rather than arbitrary mode mixing.

## 4. First concrete predictions

### Prediction 1 — scale-invariant screen baseline

[
\boxed{
D_\ell^\chi\simeq \text{constant}
}
]

before photon-baryon transfer. This is the clean analytic OPH theorem target.

Status: mostly mathematical.

Failure mode: finite patch simulations do not approach inverse-Laplacian covariance under MaxEnt repair.

### Prediction 2 — scalar tilt from repair anomalous dimension

[
\boxed{
n_s=1-\eta_R,\qquad \eta_R\simeq0.035\pm0.004.
}
]

This is the first numerical CMB target. Planck 2018 gives (n_s=0.965\pm0.004). ([arXiv][1])

Status: analytic sign and parameterization now; simulator should derive (\eta_R), not fit it.

### Prediction 3 — low-(\ell) suppression from global repair modes

[
\boxed{
F_{\rm IR}(\ell)
================

1-q_{\rm IR}
e^{-\ell(\ell+1)/[\ell_{\rm IR}(\ell_{\rm IR}+1)]},
\quad q_{\rm IR}>0.
}
]

Expected imprint:

[
C_2,C_3,C_4,\ldots
\text{ suppressed relative to the transfer baseline,}
]

with high-(\ell) spectra essentially unchanged.

Status: analytic form now; significance needs masked-sky Monte Carlo.

### Prediction 4 — parity asymmetry has a smooth low-(\ell) envelope

[
\boxed{
C_\ell
======

C_\ell^{(0)}
\left[
1+\epsilon_P(-1)^\ell e^{-\ell/\ell_P}
\right].
}
]

This predicts that any OPH parity effect should be **coherent and low-(\ell)-localized**, not random alternating high-(\ell) noise. Your draft already frames this as residual orientation / antipodal / cap-pair synchronization defect. 

Status: analytic form now; significance needs Planck/WMAP masks and component-separated map checks.

### Prediction 5 — off-diagonal covariance is structured

OPH should not predict arbitrary covariance. It should predict low-rank or low-(L) repair covariance:

[
A_{\ell\ell'}^{LM}\neq0
\quad\text{mainly for small }L,\ell,\ell'.
]

Expected signatures:

[
\text{quadrupole-octopole alignment, hemispherical power asymmetry, preferred-axis BipoSH modes.}
]

Status: requires Monte Carlo and Planck map analysis.

### Prediction 6 — acoustic peaks come from transfer physics, not direct screen settling

OPH supplies the initial curvature/repair field; standard photon-baryon plasma physics supplies acoustic oscillations:

[
\text{OPH screen field}
\rightarrow
P_{\mathcal R}^{\rm OPH}(k)
\rightarrow
\Theta_\ell(k)
\rightarrow
C_\ell^{TT,TE,EE}.
]

This is a strength, not a weakness. The draft already correctly draws this boundary. 

Data target: match Planck’s base-(\Lambda)CDM acoustic constraints, including (100\theta_*=1.0411\pm0.0003), (\Omega_bh^2=0.0224\pm0.0001), (\Omega_ch^2=0.120\pm0.001), (H_0=67.4\pm0.5), (\Omega_m=0.315\pm0.007), and (\sigma_8=0.811\pm0.006). ([arXiv][1])

### Prediction 7 — lensing/growth must be handled by OPH anomaly stress

The OPH dark-sector paper already separates the static galaxy branch from cosmological perturbation theory. In that framework, the dark source is a transported modular/collar information-defect remainder: stress-first like dark matter, baryon-linked only on the settled galaxy branch. 

The cosmological Boltzmann module needs:

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
\Gamma_{\rm rec}(k,a).
]

The current OPH dark-sector diagnostic already has a plumbing check with CAMB-like compressed rows, including (S_8=0.828924043) against a Planck target near (0.831), but the file explicitly says this does **not** replace full Planck/DESI/weak-lensing likelihoods and needs a custom (B_A(k,a)) kernel plus full covariances. 

Independent lensing target: ACT DR6 reconstructs a lensing map over 9400 square degrees, and ACT+Planck lensing gives (S_8=0.831\pm0.023). ([arXiv][2])

### Prediction 8 — BAO/background geometry is a hard cross-check

DESI DR2 reports BAO measurements from more than 14 million galaxies and quasars over three years; it finds BAO consistency with earlier BAO/SNe over the same redshift range, but also reports mild tension with CMB-inferred parameters and combined-data preference for evolving dark energy in some combinations. ([arXiv][3])

OPH must therefore predict CMB while also not breaking:

[
D_M(z)/r_d,\quad
D_H(z)/r_d,\quad
D_V(z)/r_d,\quad
H_0,\quad
\Omega_m,\quad
w_0,w_a\text{ diagnostics}.
]

This is where OPH’s (N_{\rm scr}) branch matters but does not solve everything. The synthesis paper uses (N_{\rm scr}) as de Sitter entropy capacity and gives (N_{\rm scr}\simeq3.31\times10^{122}) for the late-time scale, with (\Lambda=3\pi/(GN_{\rm scr})).  But the dark-sector paper is explicit that screen capacity alone does not fix baryon density, neutrino density, radiation density, or the homogeneous anomaly charge (Q_A). 

## 5. The “must predict” checklist

I would turn the project into this checklist.

### A. OPH-screen predictions

These are closest to OPH first principles:

[
C_\ell^\chi,\quad
D_\ell^\chi,\quad
\eta_R,\quad
\mu,\quad
\ell_{\rm cap},\quad
N_{\rm frz},\quad
q_{\rm IR},\quad
\epsilon_P,\quad
A_{\ell\ell'}^{LM},\quad
f_{\rm NL}^{\rm defect}.
]

Immediate target:

[
\eta_R\simeq0.035.
]

### B. Primary CMB spectra

These require Boltzmann transfer:

[
C_\ell^{TT},\quad
C_\ell^{TE},\quad
C_\ell^{EE},\quad
C_\ell^{BB}.
]

Sub-predictions:

[
\ell_1^{TT},\ell_2^{TT},\ell_3^{TT},
\quad
H_2^{TT},H_3^{TT},
\quad
\theta_*,
\quad
\ell_{\rm damping}.
]

### C. Lensing and structure

[
C_\ell^{\phi\phi},
\quad
A_L,
\quad
P(k),
\quad
\sigma_8,
\quad
S_8,
\quad
f\sigma_8.
]

### D. Low-(\ell) anomaly observables

[
C_2,
\quad
C_3,
\quad
S_{1/2},
\quad
R_{\rm OE},
\quad
\hat q\cdot \hat o,
\quad
A_{\rm hemi},
\quad
A_{\ell\ell'}^{LM}.
]

### E. Background and composition

[
\Omega_bh^2,
\quad
\Omega_Ah^2\text{ or }\Omega_ch^2,
\quad
H_0,
\quad
\Omega_\Lambda,
\quad
\Omega_K,
\quad
N_{\rm eff}^{\rm rel},
\quad
\sum m_\nu,
\quad
Y_p,
\quad
\tau.
]

### F. Cross-dataset tests

[
\text{BAO},
\quad
\text{SNe},
\quad
\text{weak lensing},
\quad
\text{RSD},
\quad
\text{cluster lensing},
\quad
\text{SPARC/RAR consistency}.
]

## 6. What OPH should claim now

A good claim boundary is:

[
\boxed{
\text{OPH predicts an effective screen/freezeout covariance class,}
}
]

not yet:

[
\boxed{
\text{OPH has derived the full observed CMB from first principles.}
}
]

Your draft already says not to claim OPH replaces inflation, not to call the current FPE (C_\ell) the CMB, and not to treat low-(\ell) anomalies as proof.  That boundary is exactly right.

## 7. Immediate next deliverable

The next concrete document should be:

```text
OPH-CMB Prediction Ledger v0.1
```

with four tables:

1. **Screen-theory parameters**: (A_\chi,\eta_R,\mu,\ell_{\rm cap},N_{\rm frz},q_{\rm IR},\ell_{\rm IR},\epsilon_P,A_{\ell\ell'}^{LM}).
2. **CMB observables**: (TT,TE,EE,BB,\phi\phi,\theta_*,A_s,n_s,\alpha_s,\tau,\Omega_bh^2,\Omega_Ah^2,H_0,S_8).
3. **Anomaly statistics**: (C_2,C_3,S_{1/2},R_{\rm OE},A_{\rm hemi},) BipoSH coefficients, quadrupole-octopole alignment.
4. **Claim status**: analytic prediction, Boltzmann-transfer prediction, simulator-derived prediction, phenomenological placeholder, or external astrophysical nuisance.

The simulator should estimate, not assume:

[
\eta_R,\quad
\mu,\quad
N_{\rm frz},\quad
q_{\rm IR},\quad
\epsilon_P,\quad
\text{off-diagonal covariance strength},\quad
\text{defect non-Gaussianity}.
]

That matches your draft’s Tier D simulator target. 

[1]: https://arxiv.org/abs/1807.06209 "[1807.06209] Planck 2018 results. VI. Cosmological parameters"
[2]: https://arxiv.org/abs/2304.05203 "[2304.05203] The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters"
[3]: https://arxiv.org/abs/2503.14738 "[2503.14738] DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints"

----

Done — I created the first working package:

[Download the OPH-CMB Prediction Ledger v0.1](sandbox:/mnt/data/OPH-CMB-Prediction-Ledger-v0.1.md)
[Download the starter screen-model code](sandbox:/mnt/data/oph_cmb_screen_model_v0_1.py)
[Download an example primordial-power table](sandbox:/mnt/data/oph_primordial_power_v0_1.txt)

The ledger turns the project into a concrete prediction program:

[
\text{finite observer-consensus on }S^2
\rightarrow
C_\ell^\chi
\rightarrow
P_{\mathcal R}^{\rm OPH}(k)
\rightarrow
TT,TE,EE,BB,\phi\phi,P(k),\text{BAO/growth}.
]

I kept the claim boundary strict: OPH should first derive a **screen/freezeout effective theory**, not claim that the current FPE screen (C_\ell) is already the physical CMB. That matches the uploaded draft’s own boundary and staged plan. 

The most important correction I made is the tilt convention. Instead of using an ambiguous (\eta), the ledger defines

[
\eta_R := 1-n_s>0,
]

and writes

[
C_\ell^\chi
===========

\frac{A_\chi}
{[\ell(\ell+1)+\mu^2]^{1+\eta_R/2}}
W_\ell^2F_{\rm IR}(\ell)F_{\rm parity}(\ell)+N_\ell^{\rm frz}.
]

That gives

[
D_\ell^\chi\propto \ell^{-\eta_R},
\qquad
n_s=1-\eta_R.
]

So the first numerical OPH target is

[
\boxed{\eta_R^{\rm OPH}\simeq 0.035\pm0.004.}
]

Planck 2018 reports (n_s=0.965\pm0.004), along with (\Omega_bh^2=0.0224\pm0.0001), (\Omega_ch^2=0.120\pm0.001), (\tau=0.054\pm0.007), (100\theta_*=1.0411\pm0.0003), (H_0=67.4\pm0.5), (\Omega_m=0.315\pm0.007), (\sigma_8=0.811\pm0.006), (N_{\rm eff}=2.99\pm0.17), and (\sum m_\nu<0.12) eV with BAO. ([arXiv][1])

The ledger’s prediction list now includes:

| Class            | Predict                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Screen theory    | (C_\ell^\chi), (D_\ell^\chi), (\eta_R), (\mu), (\ell_{\rm cap}), (N_{\rm frz}), (q_{\rm IR}), (\epsilon_P), BipoSH terms |
| Primordial field | (A_s,n_s,\alpha_s,r,f_{\rm NL}), isocurvature checks                                                                     |
| Primary CMB      | (TT,TE,EE,BB), acoustic scale, damping tail, peak heights                                                                |
| Lensing/growth   | (C_\ell^{\phi\phi}), (A_L), (P(k)), (\sigma_8), (S_8), (f\sigma_8)                                                       |
| Anomalies        | (C_2,C_3,S_{1/2}), parity ratio, hemispherical asymmetry, quadrupole-octopole alignment                                  |
| Cross-dataset    | BAO, SNe, weak lensing, RSD, SPARC/RAR/BTFR consistency                                                                  |

The ledger also formalizes the large-angle OPH predictions already sketched in the draft: low-(\ell) suppression should come from global repair modes, parity asymmetry should have a smooth low-(\ell) envelope, and preferred correlations should appear as structured BipoSH coefficients rather than arbitrary covariance. 

I also added the dark-sector cosmology contract. The uploaded OPH dark-matter paper says the publication-grade Boltzmann module needs

[
\bar\rho_A(a),\quad
w_A(a),\quad
c_{s,A}^2(k,a),\quad
\sigma_A(k,a),\quad
Q_A^\mu,\quad
B_A(k,a),\quad
\Gamma_{\rm rec}(k,a),
]

and must run full (TT/TE/EE), CMB lensing, BAO, SNe, weak-lensing, RSD, and (S_8) likelihoods.  ACT DR6 gives a useful independent lensing/growth benchmark: ACT+Planck CMB lensing reports (S_8=0.831\pm0.023). ([arXiv][2])

The starter code implements the v0.1 scalar screen model:

```python
C_ell_oph(ell, params)
D_ell(ell, C_ell)
parity_ratio(ell, C_ell)
F_oph_k(k, screen, D_star)
primordial_power_oph(k, prim, F)
export_primordial_table(path, k, P_R)
```

It deliberately excludes parity and BipoSH from the isotropic (P_{\mathcal R}(k)) export, because those are angular covariance effects and should be tested in (a_{\ell m}) space, not squeezed into a scalar power spectrum.

The next concrete move is to run three diagnostic plots from the starter code: (D_\ell^\chi) for (\eta_R=0,0.035,0.06), low-(\ell) suppression for several (q_{\rm IR}), and parity envelopes for several (\epsilon_P). Then the exported (P_{\mathcal R}^{\rm OPH}(k)) table can be fed into CLASS/CAMB.

[1]: https://arxiv.org/abs/1807.06209?utm_source=chatgpt.com "Planck 2018 results. VI. Cosmological parameters"
[2]: https://arxiv.org/abs/2304.05203?utm_source=chatgpt.com "The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters"



---

Done — I generated the first OPH-CMB diagnostic run.

[Download the clean diagnostics bundle](sandbox:/mnt/data/OPH-CMB-diagnostics-v0.1-clean.zip)
[Open the diagnostics report](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/OPH-CMB-Diagnostics-v0.1.md)
[Download the CLASS/CAMB primordial power table](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/oph_primordial_power_CLASS_v0_1.txt)

I kept the boundary exactly where we wanted it: this is a **screen/freezeout diagnostic plus primordial-table handoff**, not a full physical `TT/TE/EE` fit yet. CAMB and CLASS are not installed in this runtime, so I did not pretend to run Boltzmann transfer. This follows the OPH-CMB scaffold: derive the screen/freezeout effective theory first, then feed its isotropic piece into transfer codes. 

Key outputs:

| Diagnostic                                          | File                                                                                     |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Screen tilt (D_\ell^\chi) for (\eta_R=0,0.035,0.06) | [plot](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/01_screen_Dell_tilt_etaR.png)    |
| Low-(\ell) global-repair suppression                | [plot](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/02_low_l_IR_suppression.png)     |
| Parity/cap-pair envelope                            | [plot](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/03_parity_envelope.png)          |
| Isotropic (F_{\rm OPH}(k)) bridge                   | [plot](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/04_primordial_Fk_correction.png) |
| Exported primordial (P_{\mathcal R}(k)) diagnostic  | [plot](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/05_primordial_power_export.png)  |

The most important numerical sanity check passed:

[
D_\ell^\chi \propto \ell^{-\eta_R}.
]

The fitted slope over (30\le\ell\le1000) came out as:

| (\eta_R) | fitted slope | expected |
| -------: | -----------: | -------: |
|        0 |  (\approx 0) |        0 |
|    0.035 |   (-0.03491) | (-0.035) |
|     0.06 |   (-0.05985) |  (-0.06) |

For the reference low-(\ell) run,

[
q_{\rm IR}=0.15,\qquad \ell_{\rm IR}=6,
]

the suppression is mainly quadrupole/octopole scale:

[
F_{\rm IR}(2)=0.86997,\quad
F_{\rm IR}(3)=0.88728,\quad
F_{\rm IR}(4)=0.90683,\quad
F_{\rm IR}(10)=0.98907.
]

The exported primordial table uses:

[
\eta_R=0.035,\qquad n_s=0.965,
]

[
A_s=2.1\times10^{-9},\qquad k_0=0.05,{\rm Mpc}^{-1},
]

[
q_{\rm IR}=0.15,\qquad \ell_{\rm IR}=6,\qquad D_*=13{,}800,{\rm Mpc}.
]

I also included these handoff files:

[CLASS template](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/class_external_primordial_template.ini)
[CAMB handoff sketch](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/camb_external_primordial_note.py)
[Reproducibility script](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/run_oph_cmb_diagnostics_v0_1.py)
[Machine-readable summary](sandbox:/mnt/data/oph_cmb_diagnostics_v0_1_clean/oph_cmb_diagnostic_summary_v0_1.json)

One important modeling choice is now explicit in the report: **parity and BipoSH terms are not included in the scalar (P_{\mathcal R}(k)) export**, because they are angular covariance effects. They need an (a_{\ell m})-space Monte Carlo next, not a scalar power-spectrum table.

---

Yes. The next move is to turn OPH-CMB into a **number ladder**, where each rung produces quantities that can be compared to public Planck/ACT/DESI data.

The strongest immediate path is:

[
\boxed{
\text{screen parameters}
\rightarrow
\text{low-}\ell\text{ residual fits}
\rightarrow
\text{CLASS/CAMB spectra}
\rightarrow
\text{map-level anomaly Monte Carlo}
\rightarrow
\text{CMB+lensing+BAO+growth likelihoods}.
}
]

Your draft already has the right staging: Phase 2 says the first serious CMB-comparable output is (P_{\mathcal R}^{\rm OPH}(k)\rightarrow TT/TE/EE/)lensing through CAMB/CLASS, Phase 3 targets low-(\ell) anomaly statistics, and Phase 5 connects CMB/LSS to OPH anomaly stress. 

## 1. Immediate numbers we already have

These can be compared without a full Boltzmann run.

| Quantity                                  |                                   Current OPH value | Data comparison target                 |
| ----------------------------------------- | --------------------------------------------------: | -------------------------------------- |
| Red repair dimension                      |                                      (\eta_R=0.035) | (n_s=1-\eta_R=0.965)                   |
| Planck scalar tilt                        |                                                   — | (n_s=0.965\pm0.004)                    |
| OPH IR angular scale                      |                                   (\ell_{\rm IR}=6) | low-(\ell) suppression window          |
| OPH IR wavenumber                         | (k_{\rm IR}\simeq 4.35\times10^{-4},{\rm Mpc}^{-1}) | largest observable CMB modes           |
| Reference IR suppression                  |                                   (q_{\rm IR}=0.15) | Planck low-(\ell) TT residuals         |
| (F_{\rm IR}(2))                           |                                           (0.86997) | quadrupole-scale suppression           |
| (F_{\rm IR}(3))                           |                                           (0.88728) | octopole-scale suppression             |
| (F_{\rm IR}(4))                           |                                           (0.90683) | low-(\ell) tail                        |
| (F_{\rm IR}(10))                          |                                           (0.98907) | should be nearly gone by (\ell\sim10)  |
| Parity envelope example                   |                                   (\epsilon_P=0.05) | odd/even low-(\ell) parity statistic   |
| (R_{\rm OE}^{2-30}) for (\epsilon_P=0.05) |                                            (0.8765) | Planck/WMAP parity estimators          |
| (R_{\rm OE}^{2-30}) for (\epsilon_P=0.10) |                                            (0.8243) | stronger low-(\ell) odd/even asymmetry |

The cleanest first numerical target remains:

[
\boxed{
\eta_R^{\rm OPH}=1-n_s\simeq0.035\pm0.004.
}
]

Planck’s final parameter analysis gives (\Omega_ch^2=0.120\pm0.001), (\Omega_bh^2=0.0224\pm0.0001), (n_s=0.965\pm0.004), (\tau=0.054\pm0.007), (100\theta_*=1.0411\pm0.0003), (H_0=67.4\pm0.5), (\Omega_m=0.315\pm0.007), and (\sigma_8=0.811\pm0.006). ([arXiv][1])

## 2. Fit (q_{\rm IR}) and (\ell_{\rm IR}) directly to Planck low-(\ell) data

This is the fastest way to get **real measured residual numbers**.

Use the Planck public TT spectrum and best-fit (\Lambda)CDM theory spectrum:

[
R_\ell^{TT}
===========

\frac{D_{\ell,\rm obs}^{TT}}
{D_{\ell,\Lambda{\rm CDM}}^{TT}},
\qquad
D_\ell=\frac{\ell(\ell+1)}{2\pi}C_\ell.
]

Then fit:

[
R_\ell^{TT}
\approx
F_{\rm IR}(\ell)
================

1-q_{\rm IR}
\exp!\left[
-\frac{\ell(\ell+1)}
{\ell_{\rm IR}(\ell_{\rm IR}+1)}
\right].
]

This would produce directly comparable numbers:

[
\boxed{
q_{\rm IR}^{\rm best},
\quad
\ell_{\rm IR}^{\rm best},
\quad
\Delta\chi^2,
\quad
{\rm AIC/BIC},
\quad
{\rm PTE}.
}
]

ESA’s Planck PR3 cosmology products publish the full TT/TE/EE spectra, best-fit theory spectra, likelihood code, and likelihood data packages, with TT covering (\ell=2)–2508 and polarization spectra covering (\ell=2)–1996. ([ESDC DOI][2])

A useful sanity table for our current (\ell_{\rm IR}=6):

| Desired quadrupole ratio (R_2) | Required (q_{\rm IR}) |
| -----------------------------: | --------------------: |
|                          (0.8) |               (0.231) |
|                          (0.7) |               (0.346) |
|                          (0.5) |               (0.577) |
|                          (0.3) |               (0.807) |
|                          (0.2) |               (0.923) |

So our current (q_{\rm IR}=0.15) is conservative. It gives mild low-(\ell) suppression, not an aggressive low-quadrupole fit.

## 3. Run a parity fit against measured odd/even power

For OPH, parity should not be a random wiggle. The prediction should be a smooth low-(\ell) envelope:

[
C_\ell
======

C_\ell^{(0)}
\left[
1+\epsilon_P(-1)^\ell e^{-\ell/\ell_P}
\right].
]

The measured statistic can be:

[
R_{\rm OE}(\ell_{\max})
=======================

\frac{
\sum_{\ell=3,5,\ldots}^{\ell_{\max}}
w_\ell D_\ell
}{
\sum_{\ell=2,4,\ldots}^{\ell_{\max}}
w_\ell D_\ell
}.
]

Fit:

[
\boxed{
\epsilon_P^{\rm best},
\quad
\ell_P^{\rm best},
\quad
R_{\rm OE}^{2-20},
\quad
R_{\rm OE}^{2-30},
\quad
R_{\rm OE}^{2-60}.
}
]

Planck’s isotropy/statistics analysis says the 2018 CMB temperature data remain broadly consistent with Gaussian (\Lambda)CDM but confirm several large-angle anomalies; it does **not** claim an unambiguous cosmological anomaly detection in polarization. ([arXiv][3]) That is exactly why OPH should report these as **falsifiable anomaly fits**, not as proof.

## 4. Run CLASS/CAMB and compare (TT,TE,EE,\phi\phi)

This is the first place we get physical CMB spectra:

[
P_{\mathcal R}^{\rm OPH}(k)
===========================

A_s
\left(\frac{k}{k_0}\right)^{-\eta_R}
F_{\rm IR}(k)
F_{\rm cap}(k).
]

Then feed the exported table into CLASS or CAMB and produce:

[
C_\ell^{TT},\quad
C_\ell^{TE},\quad
C_\ell^{EE},\quad
C_\ell^{\phi\phi}.
]

Concrete output numbers:

| Output                           | Compare to                          |
| -------------------------------- | ----------------------------------- |
| (\chi^2_{TT})                    | Planck TT likelihood                |
| (\chi^2_{TE})                    | Planck TE likelihood                |
| (\chi^2_{EE})                    | Planck EE likelihood                |
| (\chi^2_{\phi\phi})              | Planck/ACT lensing                  |
| (\Delta D_\ell^{TT}/D_\ell^{TT}) | Planck residual plot                |
| (\Delta D_\ell^{TE}/D_\ell^{TE}) | Planck TE residual                  |
| (\Delta D_\ell^{EE}/D_\ell^{EE}) | Planck EE residual                  |
| (A_L)-like smoothing shift       | Planck lensing-amplitude diagnostic |
| (S_8)                            | ACT/Planck lensing, weak lensing    |

ACT DR6 is a very useful independent lensing target: it reconstructed a lensing mass map over (9400,{\rm deg}^2), and ACT+Planck lensing gives (S_8=0.831\pm0.023). ([arXiv][4])

## 5. Use the existing OPH dark-sector numbers as a compressed cosmology check

The OPH dark-sector paper already has a numerical diagnostic row:

| Quantity                               |                      OPH diagnostic output |
| -------------------------------------- | -----------------------------------------: |
| (\Omega_A)                             |                              (0.264114401) |
| (\Omega_{\Lambda,\rm OPH})             |                              (0.684423332) |
| (\rho_A/\rho_b)                        |                              (5.363470441) |
| (\gamma_{\rm rec})                     |                              (0.050717471) |
| (a_{0,\rm eff}), (\mathbb Z_6)/Poisson | (1.179018696\times10^{-10},{\rm m,s^{-2}}) |
| CAMB (\Omega_m)                        |                              (0.315905206) |
| CAMB (\sigma_8)                        |                              (0.807787204) |
| CAMB (S_8)                             |                              (0.828924037) |

The same file is clear that this is still a diagnostic scaffold, not a full CMB/BAO/growth likelihood replacement. It says CMB/growth needs a Boltzmann module, an evaluated (B_A(k,a)), and full likelihoods. 

Two quick comparisons are already strong:

[
\frac{\Omega_c h^2}{\Omega_b h^2}
=================================

\frac{0.120}{0.0224}
\simeq
5.357,
]

while the OPH diagnostic gives

[
\frac{\rho_A}{\rho_b}=5.36347.
]

That is a very concrete background-ratio target. Planck gives the underlying (\Omega_ch^2) and (\Omega_bh^2) values. ([arXiv][1])

For lensing/growth:

[
S_8^{\rm OPH}=0.828924
]

compared to

[
S_8^{\rm ACT+Planck,lensing}=0.831\pm0.023.
]

That is essentially on target for CMB lensing, while the same OPH file notes tension against lower weak-lensing rows. 

## 6. Build a low-(\ell) Monte Carlo anomaly machine

This is the proper way to compare OPH anomaly claims.

Generate (a_{\ell m}) skies from:

[
\langle a_{\ell m}a^*_{\ell'm'}\rangle
======================================

C_\ell\delta_{\ell\ell'}\delta_{mm'}
+
\Delta_{\ell m,\ell'm'}^{\rm repair}.
]

Then compute for each simulated sky:

[
C_2,\quad
C_3,\quad
S_{1/2},\quad
R_{\rm OE},\quad
A_{\rm hemi},\quad
\hat q\cdot\hat o,\quad
A_{\ell\ell'}^{LM}.
]

The measured comparison should use Planck SMICA, NILC, Commander, SEVEM, plus WMAP as a cross-check. Your draft already lists exactly this anomaly suite: low quadrupole suppression, odd/even parity asymmetry, large-angle correlation deficit, hemispherical power asymmetry, quadrupole-octopole alignment, and finite-capacity off-diagonal covariance. 

The output numbers should be:

[
\boxed{
{\rm PTE}*{S*{1/2}},
\quad
{\rm PTE}*{R*{\rm OE}},
\quad
{\rm PTE}*{q/o},
\quad
{\rm PTE}*{A_{\rm hemi}},
\quad
{\rm PTE}_{\rm BipoSH}.
}
]

This is where OPH can start saying, quantitatively:

```text
The OPH repair-covariance ensemble makes the observed low-l sky more/less typical than Lambda-CDM.
```

## 7. Add BAO and background-distance checks

Once OPH modifies background or anomaly stress, it must not break BAO distances.

DESI DR2 reports BAO measurements from more than 14 million galaxies and quasars over three years; those DR2 BAO results are consistent with DESI DR1 and SDSS, and their distance-redshift relation matches recent supernova compilations over the same redshift range. ([arXiv][5])

Concrete OPH comparison numbers:

[
D_M(z)/r_d,
\qquad
D_H(z)/r_d,
\qquad
D_V(z)/r_d,
\qquad
r_d,
\qquad
H(z),
\qquad
w_0,w_a.
]

This matters because a model can fit CMB low-(\ell) and still fail geometry.

## 8. The next best executable deliverable

The next package should be:

```text
OPH-CMB Measurement Comparison v0.2
```

with four files:

```text
01_planck_low_l_fit.csv
02_oph_low_l_bestfit.json
03_class_camb_spectra_comparison.csv
04_anomaly_mc_pvalues.csv
```

The first run should fit only four OPH parameters:

[
\boxed{
\eta_R,\quad
q_{\rm IR},\quad
\ell_{\rm IR},\quad
\epsilon_P.
}
]

Hold everything else fixed to Planck background cosmology for now. That gives clean, interpretable numbers before opening the full dark-sector parameter space.

The minimum success criterion is not “beat (\Lambda)CDM.” It is:

[
\boxed{
\text{produce a public table where every OPH parameter maps to a measured statistic.}
}
]

That table is the bridge from OPH theory to real CMB data.

[1]: https://arxiv.org/abs/1807.06209 "[1807.06209] Planck 2018 results. VI. Cosmological parameters"
[2]: https://esdcdoi.esac.esa.int/doi/html/data/astronomy/planck/Cosmology.html "Dataset provided by the European Space Agency"
[3]: https://arxiv.org/abs/1906.02552?utm_source=chatgpt.com "Planck 2018 results. VII. Isotropy and Statistics of the CMB"
[4]: https://arxiv.org/abs/2304.05203?utm_source=chatgpt.com "The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters"
[5]: https://arxiv.org/abs/2503.14738?utm_source=chatgpt.com "DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints"



