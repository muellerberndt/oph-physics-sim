Done — I built the next package and a cumulative bundle.

[Download v0.5 hard-gates package](sandbox:/mnt/data/OPH-CMB-rest-hard-gates-v0.5-final.zip)
[Open the v0.5 report](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/OPH-CMB-Rest-Hard-Gates-v0.5.md)
[Download cumulative v0.1–v0.5 bundle](sandbox:/mnt/data/OPH-CMB-hard-gates-v0.5-combined-final.zip)

This extends the previous success ladder into the remaining gates that can be run from the public Planck PR3 spectra already available locally. The public Planck PR3 archive explicitly provides the full TT, TE, and EE spectra, plus the Planck best-fit theory spectra used as comparison anchors. ([IRSA][1])

## Main result

The OPH IR/global-repair kernel fitted to low-(\ell) TT still succeeds strongly in TT, is nearly neutral in TE, and becomes a mild pressure point in EE.

| Check                                             | OPH best IR result |
| ------------------------------------------------- | -----------------: |
| (TT), (\ell=2..29), (\Delta\chi^2)                |         (+10.7536) |
| (TE), (\ell=2..29), (\Delta\chi^2)                |          (+0.4347) |
| (EE), (\ell=2..29), (\Delta\chi^2)                |          (-1.8257) |
| Combined (TT+TE+EE), (\ell=2..29), (\Delta\chi^2) |          (+9.3626) |
| (TT), (\ell=30..1200), high-(\ell) penalty        |          (-0.3884) |

Positive means improvement over the CAMB (\Lambda)CDM power-law baseline in this diagonal proxy. Negative means worse.

The best-fit kernel is still:

[
q_{\rm IR}=0.2445545068,
\qquad
\ell_{\rm IR}=33.614958.
]

The combined low-(\ell) score remains favorable even after the two-parameter penalty:

[
\Delta{\rm AIC}*{TT+TE+EE}\simeq -5.36,
\qquad
\Delta{\rm BIC}*{TT+TE+EE}\simeq -0.50.
]

So the v0.5 status is:

```text
TT low-l: passed.
TE low-l: weakly directionally OK.
EE low-l: mild pressure point.
High-l acoustic region: preserved.
Map-space anomalies: pipeline emitted, not yet run on full Planck maps.
Official likelihoods: not yet run.
```

## Why the EE result matters

This is a useful failure mode. It says OPH should **not** keep trying to solve everything with only an isotropic scalar (P_{\mathcal R}(k)) correction.

The original OPH-CMB plan already made this separation: OPH should supply a screen/freezeout covariance and then use Boltzmann transfer for (TT,TE,EE), while parity, alignment, hemispherical asymmetry, and BipoSH/off-diagonal covariance require map-level Monte Carlo and masks. 

The new result sharpens that into a concrete constraint:

[
\boxed{
\text{A TT-fitted scalar IR kernel is not enough for low-}\ell EE.
}
]

So the next theoretical object should be an **angular covariance / record-sector freezeout correction**, not just another scalar power-spectrum multiplier.

## Parity and large-angle scalar proxies

The v0.5 package also computes scalar odd/even and (S_{1/2})-like proxies from the public spectra.

Observed low-(\ell) odd/even ratios:

[
R_{\rm OE}^{TT}(2..29)=1.31082,
]

[
R_{\rm OE}^{EE}(2..29)=1.25526,
]

[
R_{\rm OE}^{TE}(2..29)=1.39645
]

using (|TE|) for the TE proxy.

The scalar (S_{1/2})-like proxy for TT gives:

[
S_{1/2,{\rm obs}}^{TT}(\ell\le64)
=================================

6766.896,
]

while the CAMB (\Lambda)CDM scalar proxy is about (5.16\times) the observed value, and the OPH best IR scalar proxy is about (3.17\times) the observed value.

That means the OPH scalar IR suppression moves the large-angle-correlation proxy in the right direction, but does **not** solve the real phase-sensitive (S_{1/2}), quadrupole-octopole, hemispherical, or BipoSH problem. Those require maps, masks, phases, and Monte Carlo. Planck’s own isotropy/statistics paper reports that temperature large-angle anomalies persist, while no unambiguous polarization counterpart is claimed, which is exactly why this has to be treated as a hard map-space gate rather than a claim of discovery. ([arXiv][2])

## What is inside v0.5

Key files:

[TT/TE/EE diagonal proxy table](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/01_TT_TE_EE_diagonal_proxy_chi2_v0_5.csv)
[Combined low-(\ell) TT+TE+EE table](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/02_combined_lowell_TT_TE_EE_proxy_v0_5.csv)
[Low-(\ell) observed/model spectra table](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/03_lowell_TT_TE_EE_observed_vs_models_v0_5.csv)
[Parity and low-power statistics](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/04_parity_lowpower_statistics_TT_TE_EE_v0_5.csv)
[(S_{1/2})-like scalar proxy table](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/05_scalar_S12_correlation_proxy_TT_EE_v0_5.csv)
[Planck map/likelihood gate ledger](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/06_planck_map_and_likelihood_data_gates_v0_5.csv)
[Success/falsifier status ledger](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/data/07_success_falsifier_status_ledger_v0_5.csv)
[Map-space anomaly runner](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/scripts/run_planck_map_anomalies_v0_5.py)
[Map download helper](sandbox:/mnt/data/oph_cmb_rest_hard_gates_v0_5/scripts/download_planck_pr3_maps.sh)

The Planck map-space runner targets the component-separated Commander, NILC, SEVEM, and SMICA maps and common masks. Planck PR3 provides full-mission CMB intensity/polarization maps from those pipelines, plus associated masks; the common-mask page also notes the intensity and polarization masks and their (N_{\rm side}) values. ([IRSA][3])

## Current OPH-CMB status

The strongest current empirical statement is now:

[
\boxed{
\text{OPH has a measured low-}\ell TT\text{ success,}
}
]

[
\boxed{
\text{a mostly harmless TE continuation,}
}
]

[
\boxed{
\text{and a real low-}\ell EE pressure point.}
}
]

That is exactly the kind of falsifiable ladder we wanted. The uploaded OPH plan explicitly says not to claim OPH replaces inflation, has derived the full physical CMB, or that low-(\ell) anomalies prove OPH; the proper claim is an OPH continuation effective theory with falsifiable bridges to Planck/ACT/DESI data. 

## Next hard gate

The next object to build is:

[
\Delta_{\ell m,\ell' m'}^{\rm OPH}
]

rather than another scalar (F_{\rm IR}(k)).

In practical terms:

```text
OPH v0.6 = angular covariance gate
```

with:

[
\langle a_{\ell m}a^*_{\ell'm'}\rangle
======================================

C_\ell\delta_{\ell\ell'}\delta_{mm'}
+
\Delta_{\ell m,\ell'm'}^{\rm repair}.
]

That is where the EE pressure point, parity, quadrupole-octopole alignment, hemispherical asymmetry, and BipoSH structure all belong.

[1]: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/ "Index of /data/Planck/release_3/ancillary-data/cosmoparams"
[2]: https://arxiv.org/abs/1906.02552 "[1906.02552] Planck 2018 results. VII. Isotropy and Statistics of the CMB"
[3]: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/matrix_cmb.html "Planck PR3 CMB Maps"
