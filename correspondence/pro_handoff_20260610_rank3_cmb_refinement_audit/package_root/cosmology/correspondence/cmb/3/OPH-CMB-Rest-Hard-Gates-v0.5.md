# OPH-CMB Rest / Hard Gates v0.5

Generated: 2026-06-08

## Claim boundary

This package extends the v0.4 success ladder into the remaining gates that can be executed from the public Planck PR3 spectra already present locally. It is **not** an official Planck likelihood, and it is **not** a completed masked-map anomaly analysis. It adds:

1. TT/TE/EE diagonal public-spectrum proxy checks.
2. Combined low-ell TT+TE+EE proxy checks.
3. TT/TE/EE parity and scalar large-angle correlation proxies.
4. A Planck map-space anomaly script for SMICA/NILC/Commander/SEVEM maps once the large FITS files are available.
5. A hard-gate ledger that separates passed, pressure-point, and not-yet-run tests.

## Main result

The OPH IR kernel fitted to low-ell TT remains successful in TT and mostly harmless in TE, but it mildly worsens the low-ell EE diagonal proxy.

| Check | OPH best IR result |
|---|---:|
| TT delta chi2 improvement, ell=2..29 | 10.753596 |
| TE delta chi2 improvement, ell=2..29 | 0.434728 |
| EE delta chi2 improvement, ell=2..29 | -1.825675 |
| TT high-ell delta chi2, ell=30..1200 | -0.388371 |
| Combined TT+TE+EE delta chi2, ell=2..29 | 9.362649 |

Positive delta chi2 means improvement over the CAMB LCDM power-law baseline. Negative means the OPH model is worse in this diagnostic.

## Interpretation

The v0.5 pressure point is low-ell EE. A scalar IR suppression fitted to TT cannot be assumed to solve polarization. This is useful: it gives OPH a clean falsifier and tells us that the next model step cannot be just another scalar P_R(k) tweak. If OPH is right, the missing piece should live in angular covariance / record-sector freezeout structure, not in the isotropic primordial power table.

## Key files

| File | Role |
|---|---|
| `data/01_TT_TE_EE_diagonal_proxy_chi2_v0_5.csv` | TT/TE/EE diagonal chi2 proxy by spectrum/range/model |
| `data/02_combined_lowell_TT_TE_EE_proxy_v0_5.csv` | Combined low-ell TT+TE+EE proxy |
| `data/03_lowell_TT_TE_EE_observed_vs_models_v0_5.csv` | Low-ell measured/model spectra table |
| `data/04_parity_lowpower_statistics_TT_TE_EE_v0_5.csv` | Odd/even and low-power scalar statistics |
| `data/05_scalar_S12_correlation_proxy_TT_EE_v0_5.csv` | S_1/2-like scalar correlation proxy |
| `data/06_planck_map_and_likelihood_data_gates_v0_5.csv` | Map/likelihood data gate list with URLs |
| `data/07_success_falsifier_status_ledger_v0_5.csv` | Passed/pressure/not-yet-run ledger |
| `scripts/run_planck_map_anomalies_v0_5.py` | Map-space anomaly script for Planck FITS maps |
| `scripts/download_planck_pr3_maps.sh` | Direct map download helper |

## Phase-sensitive map-space gate

The following require the component-separated maps and masks, because they depend on phases and sky cuts rather than only C_ell:

- quadrupole-octopole alignment;
- hemispherical power asymmetry;
- BipoSH/off-diagonal covariance;
- masked-sky S_1/2;
- SMICA/NILC/Commander/SEVEM consistency;
- WMAP cross-check.

The script `run_planck_map_anomalies_v0_5.py` is ready for this gate. The runtime could not fetch the 384 MB to 1.9 GB Planck FITS maps directly, so the package emits the pipeline instead of pretending the map analysis was run.

## Model-status conclusion

Current status:

```text
TT low-ell: passed diagonal proxy.
TE low-ell: weakly directionally OK.
EE low-ell: mild pressure point.
High-ell acoustics: preserved.
Map-space anomalies: pipeline ready, not yet executed on Planck maps.
Official likelihoods: not yet executed.
```
