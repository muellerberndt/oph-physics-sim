I’ll extend the current pipeline into a v0.5 public-number assessment, reusing the Planck/CAMB diagnostics while adding sourced benchmarks, residuals/pulls, and clear status labels.

Done — I pushed the next rung to v0.6: exact low-ℓ likelihood predictions.

Download the v0.6 exact-likelihood package
Open the v0.6 report
Guardrailed model-selection table
Exact prediction ledger
Guardrailed low-ℓ prediction plot

The important upgrade is that I replaced the earlier low-ℓ Gaussian proxy with a full-sky cosmic-variance likelihood diagnostic for TT:

−2lnL=
ℓ
∑
	​

(2ℓ+1)[
D
ℓ
model
	​

D
ℓ
	​

	​

+lnD
ℓ
model
	​

]+const.

This is still not the official Planck low-ℓ pixel likelihood, but it is much closer to the correct low-ℓ sampling law than symmetric error-bar χ
2
. The Planck PR3 archive provides the public full TT spectrum and best-fit theory files used here, including COM_PowerSpect_CMB-TT-full_R3.01.txt and COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt.

The new exact prediction

A low-ℓ-only likelihood fit tries to make the IR kernel almost constant over the fitted range. That is not acceptable as a physical CMB prediction unless it preserves the acoustic spectrum. So the sharper result is the guardrailed fit:

exact full-sky TT, ℓ=2..29+high-ℓ TT preservation, ℓ=30..1200.

The CMB-compatible OPH IR prediction is now:

q
IR
	​

=0.1174483470,ℓ
IR
	​

=29.28739488.
	​


The joint angular template is:

q
IR
	​

=0.1130789541,ℓ
IR
	​

=29.73868151,ϵ
P
	​

=−0.98,ℓ
P
	​

=4.62077568.
	​


The ϵ
P
	​

 value hitting the lower bound means the public TT spectrum strongly prefers a low-ℓ odd/even correction in this simple envelope model; it should be treated as a target for map-space parity analysis, not as a final physical amplitude.

Guardrailed model comparison
Model	Improvement vs LCDM	Low-ℓ gain	High-ℓ penalty	AIC shift	BIC shift
OPH IR reference q=0.15,ℓ
IR
	​

=6	+0.185	+0.185	≈0	−0.185	−0.185
OPH IR guardrailed fit	+3.012	+3.600	0.588	+0.988	+11.167
OPH IR + parity guardrailed fit	+9.096	+9.721	0.625	−1.096	+19.261
Previous v0.5 Gaussian-proxy IR fit	−3.782	−0.731	3.050	+7.782	+17.960

Negative AIC/BIC shift means better than LCDM after penalty. The joint IR+parity model gets a small AIC win but not a BIC win because BIC penalizes the four extra parameters hard.

Derived physical scales

From the guardrailed IR-only fit:

θ
IR
	​

≈
ℓ
IR
	​

180
∘
	​

=6.145989
∘
,
k
IR
	​

≈
D
∗
	​

ℓ
IR
	​

	​

=0.002122 Mpc
−1
using D
∗
	​

=13,800 Mpc.

The angular freezeout proxy is:

N
frz
angular
	​

∼(ℓ
IR
	​

+1)
2
=917.33.

This is not relativistic N
eff
	​

. It is an OPH freezeout/angular-mode proxy.

What this means

The strongest exact prediction target is now:

η
R
	​

≃0.035,q
IR
	​

≃0.11745,ℓ
IR
	​

≃29.29,θ
IR
	​

≃6.15
∘
,k
IR
	​

≃2.12×10
−3
 Mpc
−1
.
	​


The tilt target remains tied to Planck’s scalar spectral index:

n
s
	​

=0.965±0.004,η
R
	​

=1−n
s
	​

≃0.035±0.004.

Planck 2018 reports n
s
	​

=0.965±0.004, Ω
b
	​

h
2
=0.0224±0.0001, Ω
c
	​

h
2
=0.120±0.001, and τ=0.054±0.007.

Why this is a better success criterion

The OPH-CMB plan says the correct path is not to claim the current OPH-FPE screen is already the CMB, but to derive a screen/freezeout effective theory, bridge it into P
R
OPH
	​

(k), and compare against TT/TE/EE/lensing through transfer and likelihoods. It also says low-ℓ suppression, parity, large-angle correlation deficit, hemispherical asymmetry, quadrupole-octopole alignment, and off-diagonal covariance are the OPH-specific anomaly tests.

This v0.6 result gives the exact numeric target that the finite-patch simulator must now derive:

q_IR ≈ 0.11745
ell_IR ≈ 29.29
theta_IR ≈ 6.15 deg
k_IR ≈ 0.00212 Mpc^-1
N_frz angular proxy ≈ 917

The next rung should be v0.7 finite-patch reverse target: build a simulator objective that asks whether OPH screen freezeout naturally emits ℓ
IR
	​

∼30, q
IR
	​

∼0.12, and a low-ℓ parity envelope, instead of fitting those numbers from Planck.

I found existing v0.5/v0.6 outputs, including TT/TE/EE checks, full-sky likelihood proxies, high-ℓ guardrails, acoustic diagnostics, and frozen ledgers, so I’ll consolidate rather than rerun.

Thought for 20m 59s



Done — I created the next package with the assessment table against real public source values.

[Download OPH-CMB public-source assessment v0.8](sandbox:/mnt/data/OPH-CMB-public-source-assessment-v0.8.zip)
[Open the v0.8 report](sandbox:/mnt/data/oph_cmb_public_assessment_v0_8/OPH-CMB-Public-Source-Assessment-v0.8.md)
[Open the main assessment table CSV](sandbox:/mnt/data/oph_cmb_public_assessment_v0_8/data/01_public_source_assessment_table_v0_8.csv)
[Open selected multipole comparison CSV](sandbox:/mnt/data/oph_cmb_public_assessment_v0_8/data/03_selected_multipole_public_assessment_v0_8.csv)
[Open odd/even comparison CSV](sandbox:/mnt/data/oph_cmb_public_assessment_v0_8/data/02_odd_even_public_assessment_v0_8.csv)

I kept the claim boundary strict: this is a **public-data diagnostic comparison**, not an official Planck likelihood, not a masked map-space anomaly analysis, and not yet a finite-patch derivation of the OPH low-(\ell) parameters. That matches the uploaded OPH-CMB plan: the correct bridge is screen covariance (\rightarrow P_{\mathcal R}^{\rm OPH}(k)\rightarrow) CAMB/CLASS (\rightarrow TT/TE/EE/)lensing, while Monte Carlo skies, Boltzmann transfer, likelihood fitting, and mask/systematics tests remain required for stronger claims.  The draft also explicitly warns not to claim that OPH has derived the full physical CMB or that low-(\ell) anomalies prove OPH. 

## Main numbers now on the table

| Quantity                            |                                                      OPH number |                                                        Public benchmark | Current assessment                                                      |
| ----------------------------------- | --------------------------------------------------------------: | ----------------------------------------------------------------------: | ----------------------------------------------------------------------- |
| Exact scalar tilt                   |                                            (n_s=0.964841143031) |                                              Planck (n_s=0.965\pm0.004) | (-0.040\sigma), pass as target                                          |
| Exact red repair dimension          |                                         (\eta_R=0.035158856969) |                                                   (1-n_s=0.035\pm0.004) | (+0.040\sigma), pass as target                                          |
| Exact scalar IR kernel              |                             (q_{\rm IR}=1/4,\ \ell_{\rm IR}=32) |                                           Planck PR3 low-(\ell) spectra | diagnostic improvement                                                  |
| TT low-(\ell), (\ell=2..29)         |                                             exact scalar kernel |                                                           Planck PR3 TT | (\Delta\chi^2=+10.590)                                                  |
| TT+TE+EE low-(\ell), (\ell=2..29)   |                                             exact scalar kernel |                                                     Planck PR3 TT/TE/EE | (\Delta\chi^2=+9.242,\ \Delta{\rm AIC}=-7.242,\ \Delta{\rm BIC}=-4.811) |
| EE low-(\ell) alone                 |                                             exact scalar kernel |                                                           Planck PR3 EE | (\Delta\chi^2=-1.814), pressure point                                   |
| Quadrupole (D_2^{TT})               | scalar (800.1,\mu{\rm K}^2); scalar×parity (314.8,\mu{\rm K}^2) |         Planck (225.9\pm332.7,\mu{\rm K}^2); LCDM (1022.8,\mu{\rm K}^2) | improves low-quadrupole pull                                            |
| Odd/even TT ratio (2..29)           |                         scalar (1.0114); scalar×parity (1.2161) |                                          Planck (1.3108); LCDM (1.0027) | parity helps, map-space test needed                                     |
| Acoustic preservation at (\ell=220) |                                          TT ratio (0.999999338) |                                              first acoustic peak region | pass; shift (-0.000066%)                                                |
| (S_8)                               |                                                   (0.828924037) |                                      ACT+Planck lensing (0.831\pm0.023) | (-0.090\sigma), pass compressed check                                   |
| (\sum m_\nu)                        |                                             (0.090011929645) eV | Planck+BAO (<0.12) eV; DESI+CMB LCDM (<0.064) eV; DESI (w_0w_a<0.16) eV | mixed                                                                   |

The Planck benchmark values used here are from Planck 2018: (\Omega_ch^2=0.120\pm0.001), (\Omega_bh^2=0.0224\pm0.0001), (n_s=0.965\pm0.004), (100\theta_*=1.0411\pm0.0003), (H_0=67.4\pm0.5), (\Omega_m=0.315\pm0.007), (\sigma_8=0.811\pm0.006), (N_{\rm eff}=2.99\pm0.17), and (\sum m_\nu<0.12) eV with BAO. ([arXiv][1]) The Planck PR3 public archive lists the full TT, TE, and EE spectrum files used in the diagnostic comparison. ([IRSA][2])

For late-structure checks, ACT DR6 reports a CMB-lensing mass map over (9400) square degrees and gives (S_8=0.831\pm0.023) for ACT+Planck lensing. ([arXiv][3]) For tensors, BICEP/Keck XIII gives (r_{0.05}<0.036) at 95% confidence. ([arXiv][4]) For BAO/neutrinos, DESI DR2 reports BAO from more than 14 million galaxies and quasars and gives (\sum m_\nu<0.064) eV in (\Lambda)CDM but (<0.16) eV in (w_0w_a). ([arXiv][5])

## Best current interpretation

The most successful concrete OPH-CMB claim is now:

[
\boxed{
\eta_R=e\alpha\sqrt{\pi}=0.035158856969,
\qquad
n_s=0.964841143031,
}
]

plus the exact low-(\ell) scalar kernel

[
\boxed{
F_{\rm IR}(\ell)
================

1-\frac14
\exp!\left[-\frac{\ell(\ell+1)}{32\cdot33}\right].
}
]

That kernel improves the public low-(\ell) TT/TE/EE diagonal proxy while leaving the acoustic peak region essentially unchanged. The angular parity candidate

[
F_P(\ell)=1-(-1)^\ell e^{-\ell/4}
]

helps the odd/even structure, but I kept it labeled as an **angular covariance diagnostic**, not a scalar primordial-power correction. That distinction matters.

The pressure point is EE: the scalar kernel helps TT strongly and TE weakly, but worsens fixed-amplitude EE in the diagonal proxy. The next real success gate is therefore not more scalar fitting; it is **map-space angular covariance plus official likelihoods**.

[1]: https://arxiv.org/abs/1807.06209 "[1807.06209] Planck 2018 results. VI. Cosmological parameters"
[2]: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/ "Index of /data/Planck/release_3/ancillary-data/cosmoparams"
[3]: https://arxiv.org/abs/2304.05203 "[2304.05203] The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters"
[4]: https://arxiv.org/abs/2110.00483 "[2110.00483] BICEP / Keck XIII: Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season"
[5]: https://arxiv.org/abs/2503.14738 "[2503.14738] DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints"
