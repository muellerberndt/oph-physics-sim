# Run highlights: oph_universe_64k_3p1d_reearned (65,536 patches / 1,024 observers)

## Milestones (this run's receipts)

| Milestone | Receipt | Detail |
|---|---|---|
| Observer-local modular time | PASS | 1024 materialized observers |
| 2 pi KMS clock (gauge-covariant, sector replay) | PASS | two_pi_selected=True, degenerate=False |
| Conformal H3 spatial chart | PASS | observer-facing chart, support-visible |
| H3 modular response | open | perturb/resettle candidate with controls |
| Observer-facing 3+1D experience (all four gates) | open | integer 3 spatial + 1 modular time; never a fractional bulk |
| Observer mutual agreement (bulk-as-agreement) | open | pairs None, median defect None, cocycle perfect None, control 0 |
| Emergent-bulk agreement field | PASS | coverage 78.0 pct, pair-certified 16.3 pct, max multiplicity 43 |
| Proto-particle population (organic) | PASS | defects 16, fusion candidates 900, identity fraction 0.23 |
| Strict neutral third-person bulk | open | glued objective bulk; blockers in neutral_3d_bulk_audit_report.md |
| Einstein branch entry (E1-E6) | open | blockers: ['sphere_fold', 'bw_2pi', 'null_stress', 'bounded_interval', 'fixed_cap_entropy', 'small_ball_area', 'remainder_control', 'timelike_coverage', 'stress_closure', 'lambda_closure', 'newton_forbidden_input_audit', 'einstein_residual'] |
| Physical particle / physical CMB | open | fail-closed by contract; screen proxies stay diagnostics |

## Run-derived diagnostics

- Screen C_l proxy vs Planck TT shape (cmb-lite): `None` (shape correlation of the best screen field; a proxy diagnostic, never a CMB prediction)
- Committed fraction at freezeout: `0.961395263671875` (freezeout cycle 106)
- Cosmology product gate: `True` (kms_bw_pass=True)

## Near-fits against public data (paper-side lanes, curated)

| Quantity | OPH | Measured | Pull |
|---|---|---|---|
| Scalar tilt n_s = 1 - P/48 | 0.9660215 | 0.9649(42) (Planck 2018) | +0.27 sigma |
| Higgs-top criticality relation m_H(m_t) | 125.72 | 125.13(11) GeV (PDG) | +0.47 percent |
| Electroweak chart pair M_W / M_Z | 80.330 / 91.119 GeV | 80.3692(133) / 91.1880(20) | -0.05 / -0.08 percent (chart, pole packet open) |
| Strong coupling alpha_s(M_Z) | 0.11834 | 0.1179(9) | +0.5 sigma |
| QCD scale Lambda_QCD(3) | 334.8 [319, 350] MeV | 338(12) MeV (FLAG-class) | inside interval |
| Capacity -> Lambda (EW branch) | Lambda l_P^2 = 2.668e-122 | 2.845e-122 (Planck display) | +6.6 percent over 122 orders of magnitude |
| CMB low-l IR filter scale | corpus l_IR = 32 | unbinned PR3 TT optimum l_IR ~ 26 (flat 24-40), q ~ 0.16 | corpus scale 0.2 chi2 units from the data optimum (net -3.9) |
| Cosmic birefringence candidate alpha_U/(2 pi) | 0.37501 deg | 0.342 +0.094/-0.091 deg (Planck+WMAP EB) | +0.35 sigma (S1 coincidence, 4 trials declared) |
| Clock-free ratio m_t/m_W | 2.149 | 2.1476 | +0.07 percent |
| SPARC RAR fixed-branch RMS | 0.13283 dex (Z6) | 0.13281 dex same-data optimum | 2e-5 dex from optimal (retrospective) |

## Banked negatives (kept visible)

- Cassini universal continuation excluded (19.2 sigma raw): the settled-galaxy scope survives, the universal extension died
- Weighted-cycle neutrino candidate rejected by NuFIT 6.1
- Screen C_l proxy anti-correlates with Planck TT: the raw screen is not the CMB

Claim boundary: Milestone receipts are this run's own gate values. Near-fit rows are curated conditional or diagnostic comparisons from the paper-side lanes (tracker section 0a); none is a frozen physical-prediction receipt. Banked negatives stay visible because the gates bite.
