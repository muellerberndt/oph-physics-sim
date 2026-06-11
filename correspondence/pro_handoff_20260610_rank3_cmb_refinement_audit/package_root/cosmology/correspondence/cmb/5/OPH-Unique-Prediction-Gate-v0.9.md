# OPH Unique Prediction Gate v0.9

## Answer to the self-check question

> What value can OPH predict that no current SM/GR baseline predicts, and that can be compared to public data now or measured soon?

The best current answer is:

\[
\boxed{n_s^{\rm OPH} = 1 - e\,\alpha(0)\sqrt{\pi} = 0.964841143031}
\]

with

\[
\eta_R^{\rm OPH}=e\,\alpha(0)\sqrt{\pi}=0.035158856969.
\]

Using CODATA/NIST \(\alpha^{-1}(0)=137.035999177\), this lands at Planck's public benchmark \(n_s=0.965\pm0.004\) with pull -0.040 sigma. SM+GR/ΛCDM does not predict this value; it parameterizes the primordial spectrum. Inflationary model classes can produce red tilts, but no current SM/GR calculation fixes this alpha-linked number.

The best **CMB-anomaly** OPH-only number is:

\[
\boxed{q_{\rm IR}=\frac14,\qquad \ell_{\rm IR}=32}
\]

which corresponds to:

\[
\theta_{\rm IR}=5.625^\circ,\qquad
k_{\rm IR}=0.00231884\,{\rm Mpc}^{-1},\qquad
N_{\rm frz,proxy}=(\ell_{\rm IR}+1)^2=1089.
\]

In the v0.8 public-data diagnostic this gives \(\Delta\chi^2=+9.242\) for low-\(\ell\) TT+TE+EE diagonal comparison and keeps the acoustic peak region essentially unchanged. This remains a diagnostic until the official Planck likelihood and map-space covariance tests are run.

The best **near-future lab-only** OPH-only number is:

\[
\boxed{\chi_\nu^{\rm can}=e^{-P/24}=0.934300639489386}
\]

This is not a public CMB-data value; it is a coherent-matter susceptibility target for a sign-reversal balance/torsion experiment.

## Ranked OPH-only predictions

| Rank | ID | OPH-only value | Public comparison now | Status |
|---:|---|---|---|---|
| 1 | U1 | n_s = 1 - e alpha sqrt(pi) = 0.964841143031; eta_R=0.035158856969 | Planck 2018 n_s = 0.965 ± 0.004 — pull = -0.040 sigma | BEST SINGLE NOW-TESTABLE OPH-ONLY NUMBER |
| 2 | U2 | q_IR=1/4; ell_IR=32; theta_IR=5.625 deg; k_IR=0.00231884 Mpc^-1; N_frz_proxy=1089 | Planck PR3 TT/TE/EE low-l spectra, ell=2..29 — v0.8 diagonal proxy: Delta chi2(TT+TE+EE)=+9.242; Delta AIC=-7.242; Delta BIC=-4.811; high-l acoustic shift negligible. | STRONGEST CMB-ANOMALY NUMBER; DIAGNOSTIC NOT OFFICIAL LIKELIHOOD |
| 3 | U3 | F_P(ell)=1-(-1)^ell exp(-ell/4); predicted R_OE^TT(2..29)=1.216064; fitted parity diagnostic=1.278713 | Planck PR3 observed R_OE^TT(2..29)=1.310820; LCDM exact-tilt CAMB=1.002724 — Exact parity envelope moves about 69% of the way from LCDM to observed in this estimator; fitted envelope moves about 90%. | HIGH-UNIQUENESS ANGULAR-COVARIANCE TEST |
| 4 | U4 | m1=0.017454720258 eV; m2=0.019481987936 eV; m3=0.053075221451 eV; sum=0.090011929645 eV; m_beta_proxy=0.01956 eV | Planck+BAO <0.12 eV; DESI+CMB LambdaCDM <0.064 eV; DESI+CMB w0wa <0.16 eV; ACT+Planck lensing <0.13 eV. — Compatible with Planck+BAO and DESI w0wa; in tension with DESI+CMB under strict LambdaCDM. | MOST IMPORTANT NON-CMB NEAR-FUTURE COSMOLOGY FALSIFIER |
| 5 | U5 | chi_nu^can = exp(-P/24) = 0.934300639489386; engineering chi≈9.343e-23 for N_coh=1e22 | No public positive signal; only null conventional controls exist unless a dedicated device runs the gate protocol. — Near-future lab falsifier, not public-data pass/fail yet. | BEST NEAR-FUTURE LAB-ONLY OPH-UNIQUE VALUE |
| 6 | U6 | a0_eff=1.179018696e-10 m/s^2; rho_A/rho_b=5.363470441; S8=0.828924037 | RAR in SPARC; Planck ratio Omega_c h^2/Omega_b h^2≈5.357; ACT+Planck lensing S8=0.831±0.023. — rho_A/rho_b pull=+0.125 sigma vs Planck ratio; S8 pull=-0.090 sigma vs ACT+Planck lensing. | PROMISING BUT NEEDS FULL BOLTZMANN + GALAXY LIKELIHOOD |

## Public assessment table

| ID | Prediction | OPH value | Public value | Comparison | Assessment |
|---|---|---:|---|---|---|
| A1 | Scalar tilt exact relation | 0.964841143031 | 0.965 ± 0.004 | -0.040 sigma | PASS / strongest exact-now target |
| A2 | Red repair dimension | 0.035158856969 | 0.035 ± 0.004 | +0.040 sigma | PASS as mirror of A1 |
| A3 | Exact IR kernel scale | 0.250000, 32 | Delta chi2=+9.242; Delta AIC=-7.242; Delta BIC=-4.811 | model-selection improvement | PASS diagnostic; official likelihood needed |
| A4 | IR angular scale | 5.625000 deg; 0.00231884 Mpc^-1 | ell range 2..29 public spectra | scale compatible with low-l only; acoustic ratios ~1 at ell>=100 | PASS diagnostic |
| A5 | TT quadrupole D2 | scalar=800.088 uK^2; scalar×parity=314.810 uK^2 | 225.895 ± 332.716 uK^2; LCDM=1022.838 | LCDM +2.395σ; scalar +1.726σ; scalar×parity +0.267σ | PASS for low quadrupole |
| A6 | Odd/even TT ratio | scalar=1.011405; scalar×parity=1.216064; fitted=1.278713 | 1.310820; LCDM=1.002724 | moves toward observed parity excess | PASS diagnostic; map-space required |
| A7 | Acoustic preservation | 0.999999338 | first acoustic peak around ell≈220 | -0.000066% ratio shift | PASS |
| A8 | Tensor amplitude on scalar-only branch | 0 | <0.036 at 95% CL | compatible by construction | PASS limit; tensor branch open |
| A9 | Neutrino mass sum | 0.090011929645 eV | <0.12 eV; <0.064 eV LCDM; <0.16 eV w0wa | 0.75× Planck+BAO; 1.41× DESI LCDM; 0.56× DESI w0wa | MIXED / high-value falsifier |
| A10 | Matter/baryon ratio | 5.363470441 | 5.357142857 ± 0.050645345 | +0.125 sigma | PASS compressed check |
| A11 | Late structure amplitude | 0.828924037 | 0.831 ± 0.023 | -0.090 sigma | PASS compressed check |
| A12 | Galaxy acceleration scale | 1.179018696e-10 m/s^2 | tight RAR with 0.057 dex residual scatter in individual-galaxy fits | needs source-level re-fit for exact pull | PROMISING / incomplete |
| A13 | Coherent-matter susceptibility | 0.934300639489386 | near-future dedicated balance experiment | not yet public-data comparable | LAB FALSIFIER |

## Claim boundary

This package is an assessment and prioritization gate. It does not replace official Planck likelihoods, component-separated map analyses, finite-patch derivations, or laboratory evidence bundles. The CMB-specific OPH work remains staged as:

```text
screen/freezeout covariance -> primordial bridge -> CAMB/CLASS -> public spectra -> official likelihood/map-space tests
```

The next technical success criterion is to pre-register U1/U2/U3 and run the official low-l likelihood plus map-space parity/BipoSH tests.
