We have the first real success rung: **OPH-CMB now produces measured, reproducible numbers against public Planck TT data, and the isotropic OPH correction has been propagated through CAMB.**

Main bundle:

[Download the combined OPH-CMB v0.2/v0.3 results bundle](sandbox:/mnt/data/OPH-CMB-results-v0.3-bundle.zip)

Separate packages:

[Download v0.2 measurement comparison](sandbox:/mnt/data/OPH-CMB-Measurement-Comparison-v0.2.zip)
[Open v0.2 report](sandbox:/mnt/data/oph_cmb_measurement_v0_2_final/OPH-CMB-Measurement-Comparison-v0.2.md)
[Download v0.3 CAMB Boltzmann bridge](sandbox:/mnt/data/OPH-CMB-Boltzmann-Bridge-v0.3.zip)
[Open v0.3 report](sandbox:/mnt/data/oph_cmb_boltzmann_v0_3/OPH-CMB-Boltzmann-Bridge-v0.3.md)

## What succeeded

The staged OPH plan said the serious CMB-comparable route is:

[
\text{OPH screen consensus}
\rightarrow
P_{\mathcal R}^{\rm OPH}(k)
\rightarrow
\text{Boltzmann transfer}
\rightarrow
C_\ell^{TT}, C_\ell^{TE}, C_\ell^{EE}.
]

That is exactly what we now have for the isotropic IR/global-repair part. The uploaded OPH-CMB draft explicitly says acoustic peaks require photon-baryon transfer and that CAMB/CLASS is the right bridge, while parity and BipoSH remain angular covariance effects rather than scalar (P_{\mathcal R}(k)) effects. 

## v0.2: measured low-(\ell) Planck comparison

I used the public Planck PR3 spectrum archive files:

```text
COM_PowerSpect_CMB-TT-full_R3.01.txt
COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt
```

Those files are listed in the official Planck PR3 cosmological-parameter archive. ([IRSA][1])

The v0.2 diagnostic fit uses:

[
2\le \ell \le 29,
\qquad
D_\ell^{TT}=\frac{\ell(\ell+1)}{2\pi}C_\ell^{TT}.
]

Main results:

| Model                    |                                                                 Parameters | (\chi^2) | (\Delta\chi^2) vs baseline |
| ------------------------ | -------------------------------------------------------------------------: | -------: | -------------------------: |
| Planck best-fit baseline |                                                                       none | (27.359) |                        (0) |
| OPH reference IR         |                                        (q_{\rm IR}=0.15,\ \ell_{\rm IR}=6) | (25.203) |                   (+2.155) |
| OPH best IR              |                                  (q_{\rm IR}=0.2446,\ \ell_{\rm IR}=33.61) | (14.240) |                  (+13.119) |
| OPH parity-only          |                                          (\epsilon_P=-1.085,\ \ell_P=4.36) | (20.205) |                   (+7.153) |
| OPH IR + parity          | (q_{\rm IR}=0.167,\ \ell_{\rm IR}=67.98,\ \epsilon_P=-0.959,\ \ell_P=3.67) | (10.581) |                  (+16.778) |

The strongest clean result is:

[
\boxed{
q_{\rm IR}^{\rm best}\simeq0.245,\qquad
\ell_{\rm IR}^{\rm best}\simeq33.6.
}
]

That means the public low-(\ell) TT table prefers a **broad low-(\ell) OPH global-repair suppression** over no suppression in this diagnostic fit.

The observed odd/even statistic over (2\le\ell\le29) is also concrete now:

[
R_{\rm OE}^{\rm obs}=1.31082,
]

while the Planck best-fit theory gives:

[
R_{\rm OE}^{\Lambda{\rm CDM}}=1.00024.
]

So the low-(\ell) Planck TT table shows an odd-power excess relative to even power in this estimator:

[
\boxed{
R_{\rm OE}^{\rm obs}/R_{\rm OE}^{\Lambda{\rm CDM}}\simeq1.3105.
}
]

That matches the OPH prediction class that parity asymmetry, if present, should have a smooth low-(\ell) envelope rather than random high-(\ell) oscillations. The uploaded draft names exactly this as a falsifiable anomaly prediction and cautions that foregrounds, masks, and a posteriori choices require Monte Carlo before claiming significance. 

## v0.3: CAMB Boltzmann bridge

I then ran CAMB (1.6.6) in this runtime and propagated the isotropic OPH IR correction through physical CMB transfer.

Settings:

[
H_0=67.4,\quad
\Omega_bh^2=0.0224,\quad
\Omega_ch^2=0.120,\quad
\tau=0.054,\quad
n_s=0.965,\quad
A_s=2.1\times10^{-9}.
]

Those are aligned with the Planck 2018 parameter benchmark: Planck reports (n_s=0.965\pm0.004), (\Omega_bh^2=0.0224\pm0.0001), (\Omega_ch^2=0.120\pm0.001), (\tau=0.054\pm0.007), (H_0=67.4\pm0.5), (\Omega_m=0.315\pm0.007), and (\sigma_8=0.811\pm0.006). ([arXiv][2])

The CAMB bridge uses:

[
F_{\rm IR}(k)
=============

1-q_{\rm IR}
\exp\left[
-\frac{\ell(k)(\ell(k)+1)}
{\ell_{\rm IR}(\ell_{\rm IR}+1)}
\right],
\qquad
\ell(k)=\max(kD_*,2),
]

with

[
D_*=13{,}800\ {\rm Mpc}.
]

CAMB low-(\ell) TT diagnostic:

| Model                                  | (\chi^2), (2\le\ell\le29) | (\Delta\chi^2) vs CAMB baseline |
| -------------------------------------- | ------------------------: | ------------------------------: |
| CAMB (\Lambda)CDM power-law            |                  (27.747) |                             (0) |
| OPH reference (q=0.15,\ell_{\rm IR}=6) |                  (26.566) |                        (+1.181) |
| OPH best low-(\ell) IR                 |                  (16.802) |                       (+10.945) |

The important physical result is that the OPH IR correction suppresses the largest angular modes while leaving the acoustic peaks almost unchanged:

|   Multipole | OPH best (TT/\Lambda)CDM (TT) |
| ----------: | ----------------------------: |
|    (\ell=2) |                      (0.7852) |
|    (\ell=3) |                      (0.7904) |
|   (\ell=10) |                      (0.8349) |
|   (\ell=30) |                      (0.9623) |
|   (\ell=50) |                      (0.9959) |
|  (\ell=100) |                    (0.999997) |
|  (\ell=220) |                    (1.000001) |
|  (\ell=500) |                    (1.000005) |
| (\ell=1000) |                    (0.999974) |

So we can now state a real OPH-CMB prediction:

[
\boxed{
\text{The fitted OPH global-repair IR correction improves low-}\ell TT
\text{ while preserving the acoustic peak region.}
}
]

That is exactly the separation we wanted: OPH supplies a primordial/freezeout covariance correction; photon-baryon physics supplies acoustic peaks.

## Cross-dataset target ledger

The background/growth numbers also line up well at the compressed diagnostic level:

| Observable      |  OPH value |                          Data target |             Pull |
| --------------- | ---------: | -----------------------------------: | ---------------: |
| (\eta_R=1-n_s)  |    (0.035) |                  (n_s=0.965\pm0.004) | exact by mapping |
| (\rho_A/\rho_b) |  (5.36347) | (\Omega_ch^2/\Omega_bh^2\simeq5.357) |    (0.125\sigma) |
| (\Omega_m)      | (0.315905) |                      (0.315\pm0.007) |    (0.129\sigma) |
| (\sigma_8)      | (0.807787) |                      (0.811\pm0.006) |    (0.535\sigma) |
| (S_8)           | (0.828924) |   ACT+Planck lensing (0.831\pm0.023) |    (0.090\sigma) |

The OPH dark-sector file already labels these as compressed diagnostics, not a replacement for full Planck/DESI/weak-lensing likelihoods; it explicitly says a publication-grade calculation needs the OPH (B_A(k,a)) kernel and full experimental covariances.  ACT DR6 gives a useful independent CMB-lensing benchmark: ACT reconstructed a (9400,{\rm deg}^2) lensing map and ACT+Planck lensing gives (S_8=0.831\pm0.023). ([arXiv][3]) DESI DR2 is the natural BAO/background next target because it reports BAO measurements from more than 14 million galaxies and quasars over three years. ([arXiv][4])

## Claim boundary

This is successful, but it is not the end state.

What we have now:

[
\boxed{
\text{measured Planck low-}\ell TT\text{ fit}
+
\text{CAMB transfer of isotropic OPH IR correction}.
}
]

What we do **not** have yet:

[
\boxed{
\text{official Planck likelihood}
+
\text{masked-sky anomaly Monte Carlo}
+
\text{TE/EE low-}\ell\text{ likelihood}
+
\text{BipoSH/parity map-space covariance}
+
\text{custom OPH dark-stress Boltzmann module}.
}
]

That boundary matches the uploaded draft’s own caution: do not claim OPH has derived the full physical CMB from first principles or that low-(\ell) anomalies prove OPH; claim instead that OPH defines an effective observer-consensus CMB continuation with falsifiable bridges to Planck/ACT/DESI data. 

The next success gate is map-space: generate OPH (a_{\ell m}) skies with the fitted (q_{\rm IR},\ell_{\rm IR},\epsilon_P,\ell_P), then compute (S_{1/2}), parity, quadrupole-octopole alignment, hemispherical asymmetry, and BipoSH PTEs against Planck SMICA/NILC/Commander/SEVEM.

[1]: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/ "Index of /data/Planck/release_3/ancillary-data/cosmoparams"
[2]: https://arxiv.org/abs/1807.06209 "[1807.06209] Planck 2018 results. VI. Cosmological parameters"
[3]: https://arxiv.org/abs/2304.05203?utm_source=chatgpt.com "The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters"
[4]: https://arxiv.org/abs/2503.14738?utm_source=chatgpt.com "DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints"


Done — I continued through the next measurable gate and packaged the **OPH-CMB Success Ladder v0.4**.

[Download the v0.4 package](sandbox:/mnt/data/OPH-CMB-success-ladder-v0.4-current.zip)
[Open the v0.4 report](sandbox:/mnt/data/oph_cmb_success_v0_4/OPH-CMB-Success-Ladder-v0.4.md)
[Model-selection table](sandbox:/mnt/data/oph_cmb_success_v0_4/data/01_model_selection_lowell_and_high_ell_v0_4.csv)
[Full-sky low-(\ell) MC table](sandbox:/mnt/data/oph_cmb_success_v0_4/data/oph_lowell_fullsky_mc_summary_v0_4.csv)
[Summary JSON](sandbox:/mnt/data/oph_cmb_success_v0_4/data/oph_cmb_success_summary_v0_4.json)

This is now a concrete diagnostic success, not just a scaffold. It follows the staged OPH-CMB plan: do **screen/freezeout effective theory**, then **OPH primordial bridge**, then **CAMB/CLASS transfer**, then **low-(\ell) anomaly Monte Carlo**, without claiming that the current FPE screen (C_\ell) is already the physical CMB.  The uploaded plan explicitly identifies the CAMB/CLASS bridge and low-(\ell) anomaly tests as the first serious CMB-comparable outputs. 

## What succeeded

Using public Planck TT bandpower data and CAMB-transferred OPH kernels, the OPH low-(\ell) IR/global-repair kernel improves the diagonal low-(\ell) TT diagnostic while leaving the acoustic regime almost unchanged. The public Planck cosmoparams archive lists the exact TT full-spectrum file and the Planck best-fit theory file used here: `COM_PowerSpect_CMB-TT-full_R3.01.txt` and `COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt`. ([IRSA][1])

The core CAMB-propagated result is:

| Model                                                          | (\chi^2_{\ell=2..29}) | Improvement vs CAMB LCDM | High-(\ell) penalty, (\ell=30..1200) |
| -------------------------------------------------------------- | --------------------: | -----------------------: | -----------------------------------: |
| CAMB LCDM power law                                            |              (27.407) |                        — |                                    — |
| OPH reference (q_{\rm IR}=0.15,\ell_{\rm IR}=6)                |              (26.230) |                 (+1.177) |                     effectively zero |
| OPH low-(\ell) best (q_{\rm IR}=0.2446,\ell_{\rm IR}=33.615)   |              (16.654) |                (+10.754) |             (0.388) over 1171 points |
| OPH joint-fit IR part (q_{\rm IR}=0.1670,\ell_{\rm IR}=67.979) |              (15.866) |                (+11.542) |             (2.357) over 1171 points |

For the best isotropic OPH IR kernel,

[
q_{\rm IR}=0.2445545068,
\qquad
\ell_{\rm IR}=33.614958.
]

After penalizing the two fitted parameters, the CAMB-transferred OPH low-(\ell) best model is still favored by this lightweight diagnostic:

[
\Delta {\rm AIC}\simeq -6.75,
\qquad
\Delta {\rm BIC}\simeq -4.09,
]

where negative means the OPH model has the lower information criterion relative to the CAMB LCDM baseline.

## Concrete measured-number readout

The Planck low quadrupole is directly visible in the table:

| (\ell) | Planck observed (D_\ell^{TT}) |    CAMB LCDM |  OPH low-(\ell) best | observed / LCDM | observed / OPH best |
| -----: | ----------------------------: | -----------: | -------------------: | --------------: | ------------------: |
|      2 |                      (225.90) | about (1022) |          about (803) |      (\sim0.22) |          (\sim0.28) |
|      3 |                      (936.92) |  about (968) |          about (765) |      (\sim0.97) |          (\sim1.22) |
|     10 |                      (803.66) |  about (820) | suppressed but close |      near unity |          near unity |

So the OPH IR kernel does what it is supposed to do: it lowers the model expectation on the largest angular scales without moving the acoustic spectrum much.

## Tilt target is still clean

The screen-theory tilt target remains:

[
\eta_R = 1-n_s.
]

Planck 2018 reports

[
n_s=0.965\pm0.004,
]

so the OPH target is

[
\boxed{\eta_R \simeq 0.035\pm0.004.}
]

Planck also gives the background values used as the comparison anchor, including (\Omega_c h^2=0.120\pm0.001), (\Omega_b h^2=0.0224\pm0.0001), (H_0=67.4\pm0.5), (\Omega_m=0.315\pm0.007), and (\sigma_8=0.811\pm0.006). ([arXiv][2])

## First anomaly Monte Carlo rung

The v0.4 package also adds an ideal full-sky low-(\ell) Monte Carlo sanity check. This is **not** a masked Planck map likelihood, but it gives concrete PTE-style numbers.

| Statistic                                                   |            LCDM PTE |                           OPH fitted parity / joint PTE | Meaning                                                                 |
| ----------------------------------------------------------- | ------------------: | ------------------------------------------------------: | ----------------------------------------------------------------------- |
| Odd/even ratio (R_{\rm OE}) as high as observed             |            (0.0107) |    (0.4069) for fitted parity; (0.2778) for joint model | OPH parity envelope makes observed parity much less rare                |
| (S_{1/2})-like low-correlation statistic as low as observed |            (0.0775) |                                (0.2299) for joint model | OPH suppression helps large-angle deficit                               |
| Low total power ratio lower-tail                            | (0.0366) under LCDM | (0.0754) for reference IR; IR-only best over-suppresses | Low-power fit must be balanced against parity and correlation structure |

The important interpretation is that **IR suppression and parity are different OPH channels**. The isotropic scalar (P_{\mathcal R}(k)) kernel helps low power, but it does not solve odd/even parity by itself. The parity/cap-pair term belongs in angular covariance or (a_{\ell m})-space, which matches the original OPH prediction class: parity asymmetry should be a smooth low-(\ell) envelope, and preferred large-angle correlations should appear as structured covariance rather than arbitrary scalar power wiggles. 

## What this means

The current best statement is:

[
\boxed{
\text{OPH now has a concrete, measured low-}\ell\text{ CMB target with fitted parameters.}
}
]

The strongest numbers are:

[
q_{\rm IR}\approx0.245,
\qquad
\ell_{\rm IR}\approx33.6,
\qquad
\eta_R\approx0.035.
]

And the strongest diagnostic result is:

[
\Delta\chi^2_{\ell=2..29}\approx10.75
]

after CAMB transfer, with only a

[
0.388
]

high-(\ell) penalty over (\ell=30..1200).

That is a real foothold.

## What is not done yet

This is not yet an official Planck likelihood, not a component-separated map analysis, and not a finite-patch derivation of (q_{\rm IR}), (\ell_{\rm IR}), or (\epsilon_P). The OPH dark-sector paper makes the same kind of boundary explicit for cosmology: CMB, BAO, lensing, growth, and (S_8) require a publication-grade Boltzmann module, full likelihoods, and covariance-aware comparison rather than compressed diagnostics. 

The next hard gate is:

[
\boxed{
\text{Planck SMICA/NILC/Commander/SEVEM maps + masks}
\rightarrow
S_{1/2},R_{\rm OE},\text{BipoSH},q/o\text{ alignment},A_{\rm hemi}.
}
]

But the v0.4 result is enough to say the OPH-CMB program has crossed from “prediction ledger” into **numerical comparison with measured CMB data**.

[1]: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/ "Index of /data/Planck/release_3/ancillary-data/cosmoparams"
[2]: https://arxiv.org/abs/1807.06209?utm_source=chatgpt.com "Planck 2018 results. VI. Cosmological parameters"
