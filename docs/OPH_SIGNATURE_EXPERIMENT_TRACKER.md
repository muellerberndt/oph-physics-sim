# OPH Signature and Falsification Experiment Tracker

Date: 2026-07-13, relocated 2026-07-14. Location: `oph-physics-sim/docs/`
(moved from `oph-meta/epic-wins/`; a pointer stub stays there). This is the working tracker
for every procedure that computes an OPH-specific quantity and compares it
against public measurement data, a classical (von Neumann) computation, or a
quantum computation. Paths note after the move: artifact paths starting
with `epic-wins/`, `particle_crisis/`, or `audit-bundles/` live in the
`oph-meta` repo one level up; paths starting with `runs/`, `oph_fpe/`,
`configs/`, or `docs/` live in this repo. It merges four inputs:

1. Pro audit round 1 (`audit-bundles/Pro1.md`, `OPH_MEASUREMENT_AUDIT_PRO_20260713.md`).
2. Pro audit round 2 + extended bundle (`audit-bundles/Pro2.md`,
   `audit-bundles/ROUND2_EXTENDED/` with the 51-row candidate catalog,
   dependency graph, primary-source ledger, receipt template).
3. The falsification catalog in `oph-meta/particle_crisis/PUBLIC_DATA_FALSIFICATION_CATALOG.md`.
4. Two fresh corpus sweeps (2026-07-13): a 16-question paper survey over
   `reverse-engineering-reality/{paper,extra,cosmology}` and a 10-question
   survey of `oph-physics-sim` machinery and run artifacts, plus new
   numerical spot-checks (receipts in the session scratchpad,
   `signature_checks.py`).

## 0a. At a glance (updated 2026-07-14, day)

Done rows first, compared against measurement where a public datum
exists. Pulls follow the discipline rules of section 0 (measurement-only
pulls; claim tiers inherited from each lane). Details in section 1b;
full board in section 1.

### Done: OPH result vs measurement

| Experiment | OPH result | Measurement | Pull / verdict |
|---|---|---|---|
| Scalar tilt n_s = 1 - P/48 | 0.9660215 | Planck 2018: 0.9649(42) | +0.27 sigma |
| Capacity -> Lambda (EW branch, provenance clean) | Lambda l_P^2 = 2.668e-122 | Planck display 2.845e-122 | +6.6 percent gap; l_star/l_P = 0.969 if exact |
| Higgs-top criticality relation | m_H(curve at measured m_t) = 125.72 GeV | PDG 125.13(11) | +0.47 percent |
| Electroweak pair (chart) | M_W 80.330, M_Z 91.119 GeV | PDG 80.3692(133), 91.1880(20) | -0.05 / -0.08 percent (chart, pole packet open) |
| Strong coupling | alpha_s(M_Z) = 0.11834 | world 0.1179(9) | +0.5 sigma |
| QCD scale | Lambda_QCD(3) = 334.8 [319, 350] MeV | FLAG-class 338(12) MeV | inside interval |
| Clock-free ratio scoreboard | m_t/m_W 2.1490; m_H/m_t 0.72856; m_p/m_e 1818.4 | 2.1476; 0.72497; 1836.15 | +0.07 / +0.50 / -0.97 percent |
| Charged-lepton ratio (MCPR lane) | m_mu/m_e 206.7683 | CODATA 206.76828 | 0.09 ppm (declared-model lane) |
| DK-05 CMB low-l IR filter (executed 07-14) | corpus filter, scale l_IR = 32 | Planck PR3 unbinned TT prefers q 0.13-0.16 at l_IR 24-40, net Delta chi2 -3.9 | corpus scale sits 0.2 chi2 units from the data optimum |
| Birefringence candidate alpha_U/(2 pi) | 0.37501 deg | Planck+WMAP EB: 0.342 +0.094/-0.091 deg | +0.35 sigma (S1 coincidence, 4 trials declared) |
| MM-01 ringdown area quantum (naive bridge) | alpha = 4 predicts f_line = 204 Hz (GW150914) | measured f220 = 251 Hz -> alpha_implied 16.0 +/- 2.0 | -5.9 sigma under the naive bridge; 4-nat bundles (alpha_eff 16) match; bridge theorem constrained |
| DK-01 capacity gap vs DESI thawing (mock) | induced drift (w0, wa) = (-1.07, +0.33), freezing | DESI DR2 prefers (-0.72, -1.0), thawing | OPPOSITE direction: gap cannot mimic DESI; (w0,wa)=(-1,0) stance fully exposed |
| VN-04 screen chirality | turning z: +0.73, -0.17, +0.74, -0.80, -1.48 across runs | null hypothesis 0 | consistent with zero; signs scatter; chirality banned from scalar/holonomy sectors by theorem |
| Sigma m_nu floor (rank lane, to be stated) | 0.0588 eV | DESI DR2+CMB bound ~0.064 eV | window 0.005 eV: one release cycle decides |
| Cassini universal continuation (banked kill) | Q2 = 3.62e-26 s^-2 | ephemeris (1.6 +/- 1.8)e-27 | 19.2 sigma excluded; scoped law survives |

### Mismatch and high-sigma rows: candidate explanations (declared, not assumed)

| Row | Size of miss | Candidate explanations, most plausible first |
|---|---|---|
| MM-01 ringdown (naive bridge) | -5.9 sigma on GW150914 | (1) The bridge assumption is the weak link: the 220 ringdown mode is a classical resonance, not the quantum transition line; the OPH line would then be a SECONDARY spectral feature at r(a) x f220 (the defined search), and no pull applies to f220 itself. (2) Multi-quantum emission: transitions in 4-nat bundles give alpha_eff = 16, matching the implied 16.0 +/- 2.0; the theorem question becomes why the twelve-port record emits in bundles. (3) The area-entropy normalization (4 l_P^2 per nat) could carry a factor from the port structure; any such factor must come from a theorem, never from fitting this number. |
| Cassini universal continuation | 19.2 sigma (banked kill) | Established, in-corpus: the universal static extrapolation of the settled-galaxy law is wrong; the law's domain predicate (old, settled, low-acceleration systems) excludes the Solar System. The OPEN part is deriving that predicate from screen coarse-graining instead of declaring it (DK-08); until then the kill bounds any future applicability law (>= 85.6 percent suppression at Saturn). |
| Capacity -> Lambda | +6.6 percent | (1) Two capacity branches exist (N_CRC_EW = 3.53e122 from the EW hierarchy; N_star ~ 3.31e122 as the de Sitter display) and their 6.7 percent gap IS the Lambda gap; a reconciliation theorem for the two countings would close or expose it. (2) Equivalently l_star = 0.969 l_P: a 3 percent length-calibration question. (3) NOT the nats/bits convention: that would be a 44 percent shift. (4) The measured Lambda display is itself fit-model conditional (evolving-DE fits move it by percent). |
| W/Z chart | -0.05 / -0.08 percent | Established category: chart coordinates, never poles; the radiative packet (Delta r, Delta kappa, Delta rho) is the declared missing map (CL-03), and its expected size (~0.03 in Delta r) matches the gap scale. Pro round 2: the one-scale repair lands at 80.3905, between the chart and the poles. |
| m_p/m_e (clock-free) | -0.97 percent | The hadron lane enters through an external lattice ratio and the Ward-projected rho_had chain; the miss is attributed to the open hadronic packet (declared QCD truncation), the same object blocking alpha at CODATA precision (C4). A source-side QCD spectral backend (CL-04) is the single fix for both. |
| m_H/m_t relation | +0.50 percent | Inside the declared band: the 2-loop RG truncation carries +/- 2.1 GeV and the matching scheme +/- 1.9 GeV on m_H; the 3-loop packet is the registered discriminator that shrinks the band and re-tests. |
| MCPR mu/e ratio | 0.09 ppm fractional, but +3.9 sigma against CODATA precision | The measurement is so precise that even a ppm-scale miss is many sigma: the lane needs its source-side NLO correction (the corrected kinetic-whitening determinant changes the endpoint at exactly this order), and MCPR is a declared-model lane, never yet a source theorem. |
| DK-01 mock (opposite direction) | structural | Working as designed: a constant-Lambda shift stays inside the LCDM family, so no display bias can fake thawing; if DESI DR3 confirms evolving w, the fixed-capacity branch is falsified outright, and the corpus would need a horizon-growing N law stated BEFORE that release to survive as a prediction. |

### Done: sim-internal receipts (no external datum)

| Experiment | Verdict | Number that matters |
|---|---|---|
| 3+1D observer experience | RE-EARNED under covariant contract | 4k: all 4 gates TRUE, no overrides; time gate TRUE at 4k/16k/64k |
| Observer mutual agreement | TRUE at every scale | 4k..1M: pair defect 0.0, cocycle 100 percent, control ~0.80 |
| 2 pi KMS clock | GENUINELY CERTIFIED post replay fix | two_pi_selected true, non-degenerate (was fail-closed) |
| Proto-particles | RECEIPTED | organic population + worldlines TRUE; 2-defect assays TRUE |

### In flight / pending

| Experiment | Status | Next gate |
|---|---|---|
| At-scale full 3+1D receipt | 128k + 1M reruns in tails | all four gates at 128k/1M |
| Effective-4D direction check | 256k rung in tail | four-rung ladder audit: rank-3 stability, drift <= 0.10 |
| Particle promotion | producers named | covariant fusion transport (same playbook as the clock fix) |
| Einstein bridge E1-E6 | red, bw_2pi resolved | sphere_fold, null_stress, bounded_interval |
| Freeze registry FZ-01 | pending | hash-pin the 20 rows (1 day) |

## 0. Discipline (applies to every entry)

- **Claim classes** (Pro round 1, section 14): S0 identity/imported, S1
  internal arithmetic candidate, S2 calibrated/retrospective, S3 frozen
  conditional prediction, S4 source-closed prospective prediction. Only
  S3/S4 count as predictions.
- **Measurement-only pulls.** A sigma quoted without an OPH-side covariance
  is a measurement-only pull, never a model significance.
- **Count once.** Correlated descendants of one theorem or one dataset are
  one evidence item. The dependency graph lives at
  `audit-bundles/ROUND2_EXTENDED/oph_round2_dependency_graph.csv`.
- **Trials ledger.** Every retrospective coincidence entry declares how many
  candidate expressions were tried before the hit.
- **Data pins.** External numbers below are orientation values from named
  releases; re-pin the exact file and hash at execution time using
  `audit-bundles/ROUND2_EXTENDED/oph_round2_primary_sources.md` and the
  receipt template `OPH_PREDICTION_RECEIPT_TEMPLATE.md`.
- **Status vocabulary.** LIVE (number exists today), BUILD (defined
  procedure, days of effort), GATED (a named theorem or receipt blocks it),
  EXPLORATORY (mechanism plausible, computation to be defined),
  PASSED-NULL (data agrees with an exact-zero structural claim),
  BANKED-KILL (a branch died and the record is kept), THEOREM-TARGET
  (the corpus is silent; stating the theorem creates the test).

---

## 1. Status board

| ID | Test | Class | Status | Priority | Effort | Next action |
|---|---|---|---|---|---|---|
| DK-01 | Fixed capacity vs DESI evolving dark energy | S3 after theorem | GATED (w law) | A | days | state fixed-N theorem; run mock w0-wa fit |
| DK-02 | Neutrino floor 0.0588 eV vs cosmological bound | S3 after lane | GATED (rank lane) | A | days | state NU-2c rank lane as conditional |
| DK-03 | Hubble tension side (67.4) | S2 diagnostic | LIVE | B | hours | register dark-siren holdout |
| DK-04 | S8 from repair-stress parent | S3 after parent | GATED (parent sector) | B | months | build transported repair-stress branch |
| DK-05 | CMB low-ell deficit via IR filter (l_IR ~ 32) | S1 -> S3 | BUILD | A | days | fit F_IR(l) on Planck low-ell likelihood |
| DK-06 | Cosmic birefringence alpha_U/(2 pi) = 0.375 deg | S1 coincidence | THEOREM-TARGET | A | days + theorem | tomography check; hunt emission theorem |
| DK-07 | Parity-odd LSS from screen chirality | S1 | BUILD (sim side) | B | days | signed S3-holonomy statistic on runs |
| DK-08 | Wide binaries + dwarfs + Cassini EFE triangulation | S3 after predicate | GATED (applicability law) | A | weeks | derive settled-domain predicate; freeze verdict |
| DK-09 | Muon g-2 adjudication via one spectral backend | S3 after backend | GATED (backend precision) | B | weeks | requadrature of rho_EM with g-2 kernel |
| DK-10 | W-mass split: OPH sides against CDF | S2 today | LIVE (chart level) | B | with CL-03 | close Delta r/kappa/rho packet |
| DK-11 | Quasar dipole excess | none predicted | EXPLORATORY | C | days (sim) | measure reconstruction dipole in sim |
| DK-12 | JWST early massive galaxies | J0 diagnostic | BUILD (stub exists) | C | days | populate JWST lane from public catalogs |
| DK-13 | UHECR: Amaterasu, Auger dipole | S4 candidate | GATED (real emission + map) | A | weeks | VN-08 then forward map |
| DK-14 | LHAASO GRB 221009A transparency | none | EXPLORATORY | C | undefined | park until a transport object exists |
| DK-15 | Neutron lifetime beam-bottle | S0 structural | LIVE | C | hours | register no-dark-channel stance |
| DK-16 | X17 / ATOMKI | S0 structural | LIVE | C | hours | register null stance |
| DK-17 | R(D*) lepton universality | S0 structural | LIVE | C | hours | register fade stance |
| DK-18 | Gallium/BEST deficit vs exactly-3 | S0 + stub lane | BUILD | C | days | populate `data/gallium/` lane |
| DK-19 | Cluster lensing offset pattern (Bullet-type) | stated mechanism | BUILD | B | weeks | quantify predicted offset law |
| MM-01 | Ringdown area-quantum comb (alpha = 4 nat) | S3 after bridge | GATED (Phase-III bridge) | A | days for scan | Foit-Kleban scan on GWTC-3 ringdowns |
| MM-02 | Repair-noise decoupling, 48-order data wall | structural | LIVE (bound) | B | done/theorem | state exact-decoupling theorem |
| MM-03 | Holographic transverse noise (Holometer) | null, conditional | PASSED-NULL | C | none | park; amplitude law absent |
| MM-04 | Gravitational entanglement stance (BMV) | THEOREM-TARGET | GATED | B | theorem | derive LOCC vs quantum channel verdict |
| MM-05 | Clock-network holonomy loops | S3 after map | GATED (forward map) | C | weeks | define closed-loop observable |
| MM-06 | Flyby anomaly replay (12 certificates) | diagnostic | BUILD | B | weeks | run OD replay; test K = 2 omega R/c |
| MM-07 | LIV / GW-speed / EP / photon-mass block | S0 block | PASSED-NULL | B | hours | bank as one recovered-structure block |
| CL-01 | RAR-BTFR closure C = r_B + 2 r_R = 0 | decisive | BUILD | A | 1-2 weeks | matched-galaxy hierarchical pipeline |
| CL-02 | Rotation-lensing no-slip Delta Sigma(R) | decisive | BUILD | A | weeks | predict KiDS/DES vectors, no halo refit |
| CL-03 | Electroweak Delta r / Delta kappa / Delta rho packet | decisive | GATED (pole packet) | A | months | one loop-complete self-energy map |
| CL-04 | Single QCD spectral backend (alpha(q), HVP, moments) | decisive | GATED | A | months | one rho_EM(s) for five observables |
| CL-05 | Capacity two-branch reconciliation + Lambda | decisive | BUILD | A | days | wire N_CRC_EW into sim; kill circular default |
| CL-06 | Clock-free dimensionless ratio scoreboard | LIVE | LIVE | B | done | freeze the four ratios |
| CL-07 | Common-basis flavor packet (Y_e, M_nu, Y_u, Y_d) | decisive | GATED | B | months | one scheme, one scale, holdout release |
| VN-01 | Dimension-drop decisive run (8-12 -> ~3?) | decisive | BUILD | A | days | locality-preserving features + estimator |
| VN-02 | W5 coefficient measurement vs frozen locus 1.8890 | decisive | BUILD | A | weeks | conditioned-ensemble sim measurement |
| VN-03 | Gauge-covariant kernel -> CMB TT re-diagnostic | BUILD | BUILD | A | weeks | rerun chi2/bin after K1 fix |
| VN-04 | Signed parity statistic on freezeout screens | BUILD | BUILD | B | days | oriented S3 correlator + shuffle controls |
| VN-05 | CMB anomaly report population + icosahedral axes | BUILD | BUILD | B | days | fill stubs; new axes test on Planck alms |
| VN-06 | Reconstruction source-count dipole | EXPLORATORY | BUILD | C | days | count statistic on reconstructed bulks |
| VN-07 | In-sim capacity growth N(t) | EXPLORATORY | BUILD | C | days | does Lambda_eff drift in-universe? |
| VN-08 | Real UHE coefficient emission (replace demo) | prerequisite | BUILD | A | days | emit from source runs, then forward map |
| QC-00 | IBM cloud heat-kernel lane (exists) | anchor | LIVE | C | done | extend beyond Z3 sectors |
| QC-01 | Icosa12 frustrated model ED/VQE (A5 x Z2) | verification | BUILD | B | days | exact spectrum + hardware VQE cross-check |
| QC-02 | Transfer-gap spectra at larger volumes | verification | BUILD | C | weeks | reversible-projection operator on QPU |
| QC-03 | Circuit-fidelity structural null | S0 bound | LIVE | C | hours | register no-intrinsic-floor stance |
| QC-04 | Bell / Tsirelson / I3 battery | S0 block | PASSED-NULL | C | hours | bank once |
| NT-01..11 | Null-test battery | S0/structural | mixed | B | hours each | see section 6 |
| LB-01 | chi_nu force experiment | S3 lab | BUILD | B | months | fix drive/detector/controls per audit |
| LB-02 | SHA-256d beta(k) + public-chain u = H/T | decisive lab | BUILD | B | weeks | blinded protocol; chain-data test is free |
| LB-03 | Cross-species clock ratios from one Cs freeze | decisive | GATED (clock chain) | B | months | freeze Cs, predict Sr/Yb/Al+ |
| LB-04 | Low-acceleration EFE in the lab / saddle points | long horizon | EXPLORATORY | C | years | depends on DK-08 predicate |
| FZ-01 | Freeze-registry artifact | protocol | BUILD | A | 1 day | hash-pin the section 8 table |

---

## 1b. Results log (running)

Newest first. Every entry names the artifact that reproduces it.

- **2026-07-14 (day) · FIRST FULL SPACETIME-CONSENSUS RECEIPT AT SCALE.**
  `oph_universe_128k_3p1d_earned` (131,072 patches, 32,000 observers, the
  1M production configuration one step down, zero overrides, covariant
  probe): `observer_facing_3p1d_h3_experience_receipt: TRUE` with all
  four component gates (modular time, gauge-covariant 2 pi KMS clock at
  score 0.304 non-degenerate, conformal H3 chart, H3 modular response),
  AND `OBSERVER_SPACETIME_CONSENSUS_RECEIPT: TRUE` with an EMPTY blocker
  list: 600/600 agreement pairs at defect 0.0, 300/300 cocycle-exact
  triples, shuffled control 0.804. Operational reading, exactly at the
  claim boundary: observers experience integer 3+1 charts and their
  charts glue exactly; a shared spacetime in the OPH sense holds at this
  scale over the shared committed record. The follow-on physics gate is
  unchanged: genuinely independent per-observer commit histories, then
  strict neutral bulk. Emergent-bulk field: pair-certified multiplicity
  up to 169. Package #2:
  `runs/oph_universe_128k_3p1d_earned_visualizer_bundle.zip` (202.9 MB)
  with the milestone ladder reading 7 PASS / 4 open.
- **2026-07-14 (day) · THE 3+1D EXPERIENCE RECEIPT IS RE-EARNED UNDER THE
  COVARIANT CONTRACT.** Root cause of the loss, from the deep audit: the
  e460f34 covariance upgrade correctly invalidated the gauge-blind 2 pi
  KMS clock certification (the time half of 3+1), and the H3 space gates
  were healthy all along in the e5 lineage. Fix: production sector-repair
  replay implemented inside BOTH perturb-remeasure probe copies
  (`modular_probe.py` and the duplicated look-alike in
  `transition_selection.py`, the exact anti-pattern the
  `repair_production_sector_links` docstring warns against): per-source
  local gauge, covariant discrepancy feeding the sector-link mutation,
  production coin-side endpoint repair; fail-closed kept for a missing
  replay config; 4 new tests. Decisive 4k rerun: `two_pi_selected: true`
  (score 0.691, non-degenerate), `kms_bw_pass: true`,
  `observer_facing_3p1d_h3_experience_receipt: TRUE` with all four
  component gates true, no overrides. The receipt certifies MORE than
  the 07-11 version: gauge covariance plus faithful production-law
  replay. One of the four named Einstein-branch-entry blockers (bw_2pi)
  falls with it, pending reruns.
- **2026-07-14 (day) · Effective-4D-GR direction check defined and
  running.** The 1M run carries the full neutral-bulk and
  Einstein-bridge suite: bridge manifest TRUE, dependency discharge
  TRUE, branch-entry gates E1-E6 false with named blockers (sphere_fold,
  bw_2pi, null_stress, bounded_interval), strict neutral bulk false with
  an EXECUTABLE blocker list: the 4k/16k/64k/256k refinement ladder
  (integer rank-3 stability, drift <= 0.10), an independent rank-3
  selector, and measured cross-observer overlap (the agreement
  certificate's territory). Ladder rungs 16k and 256k are the only
  missing runs; both launched/queued today. The audit's own claim
  boundary language is the right one: rank-3 candidates stay diagnostic
  until independent selection, negative controls, directional H3 gates,
  and refinement stability all pass.
- **2026-07-14 · THE 1M RUN: mutual agreement is scale-stable to one
  million patches.** `oph_universe_1m_night1`: 1,048,576 patches, 64,000
  materialized observers, 128 cycles, committed_fraction 1.0 at
  freezeout cycle 107, ~2.5 h wall-clock on a 10-core laptop, 1.8 GB
  artifacts. Agreement certificate (2048-observer deterministic cohort):
  800/800 pairs re-gauge at defect 0.0, 400/400 observer triples
  cocycle-exact, shuffled control 0.801. Across the night ladder
  (4k, 64k, 128k, 1M) the certificate reads identically: truncated
  observer charts over the committed record admit one mutual re-gauging
  with exact Cech consistency at every scale reached, against a
  chance-level control near 0.8. The consensus receipt stays gated by
  the covariant-probe blocker (KMS sector replay, queued fix), so the
  claim stands as chart-gluing consistency, never as certified 3+1
  consensus. Worldline turning at 1M: z = -1.48, null, power-limited.
  Viz ZIP #3: 247.3 MB
  (`runs/oph_universe_1m_night1_visualizer_bundle.zip`), including the
  agreement-graph consensus scene data. Ladder ZIPs: 64k 115.1 MB,
  128k 200.4 MB, 1M 247.3 MB, each under the 256 MB cap with the
  web-agent README.

- **2026-07-14 · DK-05 DECISIVE PASS: the unbinned low-l data prefers a
  shallow IR filter at the corpus scale.** Downloaded the full PR3 TT
  table (l = 2..2508, IRSA, sha256 ccf31136...) and ran the filter fit
  with the amplitude frozen on l >= 100 and side-dependent diagonal
  errors (declared approximation of the non-Gaussian Commander
  posterior). Joint l = 2..300 net scan: optimum q = 0.16 at l_IR = 26,
  net Delta chi2 = -3.9 for two fitted parameters; the surface is flat
  over l_IR in [24, 40], and the corpus diagnostic l_IR = 32 sits 0.2
  chi2 units from optimal (best q there 0.13). Reading: a ~15 percent
  deep, l_IR ~ 30 suppression is mildly preferred by the data
  (~1.6 sigma equivalent), and the tail penalty confines the scale to
  exactly the corpus's window. Freeze target defined: the source theorem
  must emit (q_IR, t_IR); emission near (0.15, l_IR = 32) lands on the
  measured optimum, emission far from it is a kill. Artifacts:
  `epic-wins/analysis/dk05_ir_filter_lowell.py`, pinned data at
  `oph-physics-sim/data/measurements/planck2018/COM_PowerSpect_CMB-TT-full_R3.01.txt`.
- **2026-07-14 · MM-01 harness built and sharpened into a spin-dependent
  line prediction.** Kerr-corrected transition line
  2 pi f = 2 Omega_H + (alpha/8 pi) kappa/c. GW150914 pinned
  (M_f = 63.1, z = 0.09, a_f = 0.69, f220 = 251 Hz): implied
  alpha = 16.0 +/- 2.0 under the declared ringdown-equals-transition-line
  assumption, disfavoring both the naive alpha = 4 single-nat line
  (-5.9 sigma) and Bekenstein 8 pi (-4.5 sigma) on one event. Analytic
  sharpening with the Berti 220 QNM fit: under the naive bridge,
  alpha_implied is a strong function of remnant spin alone (24.5 at
  a = 0.40, 16.6 at 0.66, 6.5 at 0.90, 4.0 at a* = 0.939), so NO constant
  area quantum fits a spin-diverse population through that bridge. The
  OPH-specific registered form: with alpha = 4 fixed, the horizon
  emission line sits at a predictable fraction of the ringdown
  fundamental, f_line/f220 = 0.56 (a = 0.4), 0.79 (0.66), 0.82 (0.69),
  0.91 (0.80), crossing unity at a* = 0.939. Defined experiment: stacked
  search for a secondary spectral feature at r(a_f) x f220 in public LVK
  ringdown residuals; a feature tracking the predicted r(a) curve is an
  OPH-specific discovery channel, absence at forecast sensitivity
  constrains the bridge multiplicity. Placeholder event rows stay
  excluded until GWOSC pinning. Artifact:
  `epic-wins/analysis/mm01_ringdown_area_quantum_harness.py`.
- **2026-07-14 · DK-05 EXECUTED (first pass): binned-TT wedge on the IR
  filter.** CAMB spectrum at n_s = 0.9660215 with the corpus filter
  F_IR(l) = 1 - q exp(-l(l+1)/(l_IR(l_IR+1))), profiled against the 83
  binned PR3 TT points (l >= 48, diagonal, one amplitude): best fit
  q_IR = 0 at every l_IR; constraint wedge: l_IR = 16 unconstrained,
  l_IR = 24 gives q <= 0.41, l_IR = 32 gives q <= 0.070 (deficit-sized
  q = 0.3 costs Delta chi2 = +4.5), l_IR = 40 gives q <= 0.030. The
  corpus diagnostic l_IR = 32 sits exactly where the surviving tail
  starts to push back, so the unbinned l = 2..29 likelihood (queued,
  needs the public low-l table) is a genuine two-sided test: the deficit
  pulls q up, the tail caps it. Artifact:
  `epic-wins/analysis/dk05_ir_filter_binned_tt.py` (+ scan block in the
  session log).
- **2026-07-14 · KMS receipt bisect: the covariant probe fails closed, and
  the older 3+1 experience receipts were certified gauge-blind.** Night
  runs on the current tree lose `kms_bw_pass`; bisect (4k dense, fixed
  seed) lands on commit e460f34, whose gauge-covariant perturb-remeasure
  probe declines to certify while production sector repair is enabled
  (`proof_blockers: production_sector_repair_not_replayed_by_response_probe`).
  Two consequences. First, the 07-11 runs whose
  `observer_facing_3p1d_h3_experience_receipt` reads TRUE were certified
  by the pre-covariant probe; under the stricter contract those receipts
  are unproven and must be re-earned. Second, the queued fix is
  probe-side sector-repair replay, filed with a fixture-test plan at
  `oph-physics-sim/docs/KMS_PROBE_SECTOR_REPLAY_TODO.md`. Night ladder
  runs keep sector repair enabled and declare a product-gating override
  (`require_kms_bw_pass: false` with reason string); no receipt value is
  altered.
- **2026-07-14 · DK-01 mock EXECUTED: the capacity gap cannot mimic DESI
  thawing (banked negative for the escape hatch, sharper registered
  prediction).** Truth = flat LCDM with Lambda at the OPH capacity value
  (0.938 x Planck display, h_true = 0.659), analyst = w0waCDM fit to
  DESI-like BAO (1 percent) + offset-marginalized SNe under the Planck H0
  display prior. Induced drift: (w0, wa) = (-1.07, +0.33) with the
  67.36(54) prior, (-1.12, +0.61) with h hard-fixed: the FREEZING
  direction, opposite to DESI DR2 thawing (-0.72, -1.0), and weak
  (Delta chi2 = 1.5). Consequences: (a) OPH gains no partial credit for
  the DESI w0wa preference through display bias; (b) the fixed-N stance
  (w0, wa) = (-1, 0) carries full exposure: a confirmed DR3+ thawing
  detection falsifies the fixed-capacity branch with no capacity-gap
  refuge; (c) sanity fit recovers truth exactly. Artifact:
  `epic-wins/analysis/dk01_capacity_gap_w0wa_mock.py`.

- **2026-07-14 · VN-04 CORRECTED, then re-executed as a dynamical
  statistic.** Correction first: the original cross-gradient pseudo-scalar
  mean over a CLOSED screen vanishes identically by Stokes
  (integral of da ^ db over S2 is exact), so its measured values were
  sampling noise by construction and test nothing about dynamics; the
  entry below supersedes it (the field-pair reports stay as estimator
  diagnostics). Two structural facts survive and sharpen DK-06/DK-07:
  S3 ambivalence kills holonomy-class chirality, and quadratic
  scalar-harmonic invariants on S2 are parity-even, so screen chirality
  must be DYNAMICAL (transport, worldlines, polarization-like data).
  VN-04 v2 = defect-worldline turning handedness: mean signed geodesic
  turning per step with exact mirror antisymmetry and worldline sign-flip
  null. Executed on four runs: z = +0.73 (64k night1), -0.17 (k1 rerun),
  +0.74 (seed 20260753), -0.80 (128k pilot): no chirality, signs
  scatter, though statistical power is low (4-18 usable worldlines per
  run from the capped timeline events). Power fix queued: feed the full
  cluster worldline catalogs (~1500 per run). Artifacts:
  `oph_fpe/cosmology/screen_parity.py`
  (`defect_worldline_turning_report`),
  `defect_worldline_turning_report.json` in four run dirs.
- **2026-07-14 · 128k/32,000-observer pilot COMPLETE (the 1M
  configuration, one step down).** Full commitment (committed_fraction
  1.0 over 131,072 patches), products emitted under the declared KMS
  override, agreement certificate at scale: 600/600 pairs defect 0.0
  (2048-observer deterministic cohort), 300/300 triples cocycle-exact
  after the triangle-first upgrade, shuffled control 0.804. Viz ZIP #2:
  200.4 MB (`runs/oph_universe_128k_obs32k_night1_visualizer_bundle.zip`).
  The freezeout npz of the bounded profile carries constant fields
  (known audit quirk: the live story is in the evolution trace), which
  is why the field-pair parity diagnostics read degenerate there.
- **2026-07-14 · Scoping theorem for DK-06/DK-07: S3 ambivalence.** Every
  element of S3 is conjugate to its inverse, so every gauge-invariant
  function of S3 holonomies is parity-even. The screen's S3 gauge sector
  cannot source birefringence-type chirality through holonomy statistics
  at any scale. Combined with the executed VN-04 null on the scalar
  sector, a cosmological parity signal in OPH must come from the U(1),
  modular-transport, or worldline-dynamics side. This sharpens DK-06: the
  alpha_U/(2 pi) coincidence lives exactly on the U(1) side.
- **2026-07-14 · Observer mutual-agreement certificate built and
  validated (mission A1).** New `oph_fpe/bulk/observer_agreement.py` +
  CLI `observer-agreement-report`: per-pair chart re-gauging recovery on
  overlaps (S3 lattice-gauge gluing), section-uniqueness detection, Cech
  cocycle check on triples, shuffled-view controls, integer-only
  experienced-chart block, `bulk_dimension_claim: null` enforced by
  schema. K1-fusion 64k rerun: 400/400 pairs defect 0.0 (399
  section-unique), shuffled control 0.788, 15/15 triples cocycle-exact,
  one shared modular grid. `MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT: true`;
  `OBSERVER_SPACETIME_CONSENSUS_RECEIPT: false` with the faithful blocker
  (that run's 3+1 experience receipt is false). Continuous dimension
  estimators demoted to `claim_level: internal_diagnostic_only`
  (`bulk/dimensions.py`). Policy: fractional bulk dimensions are a
  category error; the bulk is observer agreement.

---

## 2. Anomaly docket

Recent measurements in tension with the SM, GR, or LambdaCDM, paired with
the OPH object that bears on each. Every stance here is registrable in
advance of the next data release; that is the point of the docket.

### DK-01. DESI evolving dark energy vs fixed capacity
- Anomaly: DESI DR2 BAO + CMB + SNe prefer a thawing dark-energy track
  (w0 > -1, wa < 0) over LambdaCDM at 2.8-4.2 sigma depending on the SNe
  compilation. This is the largest live crack in LambdaCDM.
- OPH object: `Lambda l_star^2 = 3 pi / N_star` (boltzmann_transport:84)
  with `l_star == l_P` by construction. If capacity is a fixed constant of
  the theory, Lambda is exactly constant: (w0, wa) = (-1, 0), and the DESI
  preference must fade. The cosmology program paper defers the w law
  ("A full cosmology paper should derive the equation of state, possible
  time evolution, and any deviation from w = -1", cmb_program:325). A
  horizon-growing N gives a specific thawing track instead. Either branch,
  once stated, is a registered prediction; the fixed-point language of the
  corpus leans to fixed N.
- Computation A (theorem): state which N enters the closure and whether it
  is time-dependent. Computation B (mock discriminator): generate mock
  BAO+SNe+CMB data with constant Lambda at the capacity value
  `Lambda l_P^2 = 2.668e-122` (6.2 percent below the Planck display) and
  fit w0waCDM: does the recovered (w0, wa) move toward the DESI direction?
  If yes, OPH explains the DESI signal as a capacity-gap artifact without
  any evolving field. Sign resolution comes from the fit, deliberately
  unguessed here.
- Data: DESI DR2 BAO likelihoods, Planck PR4, DESY5/Pantheon+ SNe.
- Kill: DESI DR3 + external data confirm evolving w at > 5 sigma with the
  fixed-N theorem on file.

### DK-02. Cosmological neutrino-mass tension vs the oscillation floor
- Anomaly: DESI DR2 + CMB bounds push Sigma m_nu below ~0.064 eV (95
  percent), and extended fits prefer a negative effective mass; the
  oscillation floor for normal ordering is 0.0588 eV. Cosmology and
  oscillations are close to mutual exclusion inside LambdaCDM.
- OPH objects: (a) a proposed rank lane (lightest mass exactly zero, normal
  ordering) giving Sigma m_nu = 0.0588 eV, m_beta ~ 8.8 meV, m_bb in
  [1.5, 3.7] meV. Corpus status: the existing weighted-cycle tuple
  (Sigma = 0.0900 eV) is rejected by NuFIT 6.1 and no rank statement
  exists; this lane must be stated as a conditional before it counts
  (task NU-2c in `particle_crisis/FULL_SPECTRUM_SOURCE_CLOSURE_TODO.md`).
  (b) The creative half: the cosmological bound assumes GR+CDM growth. The
  OPH transported repair-stress sector (dark_matter:2052+, designed, never
  delivered) changes growth, so OPH can hold the floor value while
  attributing the sub-floor cosmological preference to growth
  misattribution. That converts today's "vanishing neutrino" tension into
  an OPH consistency check with one release-cycle resolution.
- Computation: state the rank lane; deliver the repair-stress growth
  kernel; refit the Sigma m_nu bound inside the OPH growth model.
- Data: DESI DR2/DR3 + Planck PR4 chains (public), NuFIT 6.1, KATRIN.
- Kill: cosmological bound confirmed below 0.058 eV inside the OPH growth
  model itself; or ordering resolves inverted.

### DK-03. Hubble tension
- OPH side: every OPH-side cosmology display sits on the Planck side
  (H0 = 67.4 via DESI BAO+BBN in the dark paper; compressed diagnostic
  legacy table). The asymptotic de Sitter rate from the EW capacity is
  `H_Lambda = 53.98 km/s/Mpc` (a different object than H0; do not conflate).
- Stance to register: local-ladder systematics; standard-siren H0 lands at
  67-68. Data: LVK O5 dark sirens, JWST Cepheid/TRGB crosschecks.
- Kill: siren H0 at 73 +/- 1.

### DK-04. S8 / weak lensing
- Anomaly: legacy diagnostic showed +2.43 sigma vs a weak-lensing S8
  target 0.790(16); KiDS-Legacy 2025 softened the lensing side.
- OPH side: gated on the transported repair-stress parent (same object as
  DK-02b). No number today; the parent must jointly emit CMB lensing, BAO,
  f sigma8, shear (Pro-14). Park under CL-05/DK-02 build.

### DK-05. CMB large-angle anomalies through the finite-capacity filter
- Anomaly: low-ell power deficit (ell <~ 30), quadrupole-octopole
  alignment, hemispherical asymmetry A ~ 0.07(2), parity asymmetry, and
  the PR3 lensing-amplitude excess A_L = 1.18(6): each 2-3 sigma, jointly
  awkward for LambdaCDM.
- OPH object, present in the corpus: the IR suppression filter
  `F_IR(ell) = 1 - q_IR exp(-t_IR ell(ell+1))` with a diagnostic scale
  `ell_IR = 32` (cmb_program:617-626, flagged "not a source theorem").
  The measured deficit scale and the diagnostic ell_IR coincide. The
  filter has two parameters the source must eventually emit; sampling
  variance alone at ell <= 64 gives only ~0.02 dipolar modulation, a
  factor 3 short of A = 0.07, so any OPH explanation of the asymmetry
  needs a capacity gradient, a computable sim question.
- Computation: (1) fit (q_IR, t_IR) against the public Planck low-ell
  likelihood and report Delta chi2 vs LambdaCDM; (2) populate the sim
  `cmb_anomaly_report` stubs (machinery at
  `oph_fpe/cosmology/cmb_anomaly.py` computes low-ell power, parity,
  S_1/2; alignment axes are missing by declaration); (3) new: the
  icosahedral-axes test, VN-05.
- Data: Planck PR4 low-ell likelihood and SMICA alms (public).
- Class: S1 today; S3 once the source emits (q_IR, t_IR).

### DK-06. Cosmic birefringence
- Anomaly: Planck+WMAP EB analyses give a uniform polarization rotation
  beta = 0.342 +0.094/-0.091 deg (~3.6 sigma), parity violation with no SM
  mechanism.
- OPH coincidence, found 2026-07-13: `alpha_U / (2 pi) = 0.37501 deg`,
  a +0.35 sigma match. Trials ledger: four expressions tried
  (P/48 rad +17.1 sigma; alpha_U rad +21.4 sigma; alpha_U/(4 pi) -1.6
  sigma; alpha_U/(2 pi) +0.35 sigma), so the look-elsewhere factor is 4.
  Class S1, retrospective.
- Corpus status: cosmological parity violation is SILENT; the orientation
  doubling of the compact proof is an internal register statement
  (compact_proof:497), no emission theorem connects it to photon
  polarization transport. The natural shape of the missing theorem: one
  U(1) holonomy quantum of the screen per full phase turn rotates the
  polarization plane by alpha_U/(2 pi).
- Discriminators even before the theorem: (1) tomography, a recombination
  era rotation predicts the specific EB(ell) shape fit by
  Eskilt-Komatsu, a washed-out ISW-era rotation differs; (2) frequency
  independence (screen holonomy is achromatic; Faraday rotation is not);
  (3) sign consistency with DK-07 chirality.
- Data: Planck PR4 EB tomography (public), future LiteBIRD.
- Kill: beta collapses to zero with better calibration, or the theorem
  lands with the wrong sign/magnitude.

### DK-07. Parity-odd large-scale structure
- Anomaly: BOSS 4-point parity-odd claims at ~3 sigma (Philcox;
  Hou-Slepian-Cahn), weakened by later covariance reanalyses; galaxy-spin
  chirality claims exist. Unsettled, watch-status.
- OPH object: the sim stores oriented-triangle S3 holononomy classes per
  run (`oph_fpe/defects/array_s3_holonomy.py`; 64k run counts: identity
  1952, transposition 6060, threecycle 3988) plus defect worldlines with
  class labels and an `s3_class_density` field. Transpositions are the
  odd class: a signed, orientation-weighted correlator over the freezeout
  screen is directly constructible from existing artifacts (VN-04). If
  the dynamics generates a nonzero chirality, its sign and scale
  dependence become the registered object, and DK-06 and DK-07 must share
  it (one chirality parameter, three datasets).
- Data: BOSS/DESI 4PCF public measurements; galaxy-spin catalogs.
- Class: EXPLORATORY until VN-04 reports; the shuffle controls decide
  whether the screen chirality is real or a sampling artifact.

### DK-08. Wide binaries, dwarf spheroidals, Cassini: the applicability triangulation
- Anomaly: Gaia DR3 wide-binary analyses split (Chae: ~1.4 velocity boost
  at a < a0, high claimed significance; Banik: Newtonian at ~16 sigma).
  Dwarf-galaxy analyses claim external-field-effect detections. Cassini
  excludes the universal static continuation at 19.22 sigma (banked kill;
  85.6 percent suppression floor).
- OPH object: the settled-galaxy response with domain predicate "old,
  settled, low-acceleration galaxies" (dark_matter:59; cold-isotropic
  monokinetic at :1208). Wide binaries and dwarfs are unnamed in the
  corpus: the predicate underdetermines exactly the systems where the data
  disagrees. One frozen applicability law must simultaneously produce full
  response in galaxy outskirts, >= 85.6 percent suppression at Saturn, and
  a definite wide-binary verdict. This is the sharpest place where OPH is
  forced to make a novel call: MOND-type interpolation predicts one
  wide-binary answer, screen coarse-graining by binding state may predict
  the other.
- Computation: derive the predicate from screen microphysics (binding
  state vs acceleration threshold); freeze the wide-binary velocity-boost
  profile BEFORE Gaia DR4; evaluate the same kernel on dwarfs, open
  clusters, ephemerides (Pro-9/Pro-10 "new decisive").
- Data: Gaia DR3/DR4 wide-binary samples, SPARC dwarfs, DE440 ephemerides.
- Kill: no predicate exists that fits all three regimes: the settled-galaxy
  branch dies as physics and survives only as phenomenology.

### DK-09. Muon g-2: adjudicate the SM's internal split
- Anomaly: the experiment is final (FNAL 2025, 127 ppb) and the anomaly
  moved inside the SM: data-driven e+e- HVP vs lattice (BMW) disagree, and
  CMD-3 vs KLOE disagree within e+e-.
- OPH object: the declared-empirical spectral object behind
  `Delta alpha_had^(5)(M_Z^2) = 0.02676(75)` (1.1 sigma low vs KNT19).
  Requadrature with the g-2 kernel gives `a_mu^HVP-LO` with central value
  213e-11 BELOW the data-driven number, uncertainty +/- ~194e-11. The BMW
  lattice sits ABOVE data-driven by a comparable amount, so today's OPH
  central value points the opposite way from the lattice, i.e. toward a
  larger, real g-2 anomaly. At 2.8 percent backend precision this
  adjudicates nothing; at sub-1 percent it picks a side in the sharpest
  metrology fight in particle physics.
- Computation: one `rho_EM(s)` object jointly predicting Delta alpha_had,
  a_mu^HVP, a_e^HVP, e+e- moments, lattice time moments (CL-04, Pro-35).
- Data: FNAL final a_mu, CMD-3/KLOE/BaBar spectra, BMW/RBC lattice.
- Kill: none for OPH per se; the entry is an adjudication service that
  becomes a falsifier once the backend is source-closed.

### DK-10. W-mass split
- Anomaly: CDF 2022 (80.4335(94)) vs ATLAS/CMS/LHCb (80.360-80.367),
  mutually exclusive at ~4 sigma.
- OPH object: chart pair (80.330, 91.119) with the one-scale repair
  `MW_common_scale = 80.39048` (Pro round 2). Pulls: CDF -4.58, ATLAS24
  +1.51, CMS24 +3.06, LHCb +1.14. The chart level therefore sides against
  CDF. The registered statement waits for CL-03 (Delta r/kappa/rho packet)
  which must emit the physical pole with a band; until then this is a
  chart-level lean, class S2.
- Kill: packet closes on the CDF side of 80.40.

### DK-11. Quasar dipole excess
- Anomaly: CatWISE quasar number-count dipole exceeds the kinematic
  expectation at ~4.9 sigma (Secrest et al.), an FLRW isotropy stress.
- Corpus status: opposite lean. Monopole and dipole are projected out by
  construction, and isotropy is a MaxEnt theorem
  (data_likelihood_contracts:172; OBSERVERS_TECHNICAL_SUPPLEMENT:76). OPH
  currently predicts NO intrinsic dipole: if the excess is real and
  cosmological, that is anti-OPH evidence at the level of that theorem's
  premises. The constructive move: measure whether observer-centered
  reconstruction of sim bulks induces any count dipole for off-center
  observers (VN-06). EXPLORATORY.

### DK-12. JWST early massive galaxies
- Anomaly: overly massive quiescent candidates at z > 10 strain LambdaCDM
  structure-formation timelines.
- OPH object: J0 diagnostic stubs exist (`oph_fpe/jwst/contracts.py`,
  frozen-catalog likelihood unevaluated). If the repair-stress parent
  strengthens early galaxy-scale growth, OPH predicts an earlier massive
  tail; unquantified until DK-04's parent exists. BUILD the stub
  population; class J0 diagnostic.

### DK-13. UHECR: Amaterasu and the dipole
- Anomaly: 244 EeV event (TA 2023) back-tracks to the Local Void; Auger
  dipole 6.6 percent above 8 EeV.
- OPH object, corrected status: the celebrated "frozen UHE coefficients"
  `(0.25, 0.35, -0.20, 0.10, 0.05)` are a planted-recovery demo fixture
  (`runs/uhe/coefficient_emission/`: `SYNTHETIC_DEMO_FIXTURE`,
  `physical_claim: false`), exponential-family tilts on five abstract
  source features with the propagation/detector map explicitly declared
  absent. Two builds stand between OPH and a real registered UHE test:
  VN-08 (emit real coefficients from source runs under the existing
  `NO_UHE_DATA_USE` gate) and the forward map to channel flux ratios.
  Done in that order this is the suite's cleanest path to an S4 object;
  claimed early it is vapor. Priority A.
- Data: IceCube diffuse flux, Auger spectrum/composition/dipole, LHAASO.

### DK-14. LHAASO GRB 221009A transparency
- Anomaly: >= 13 TeV photons traversed an EBL depth that should have
  absorbed them; axion-photon or LIV explanations circulate.
- OPH status: no transport object bears on pair-production opacity. PARK,
  revisit if the UHE forward map produces a photon-transport kernel.

### DK-15. Neutron lifetime beam-bottle split (~4 sigma)
- OPH stance: exact SM field content, no dark decay channel: the bottle
  value is correct and the beam discrepancy is systematic. Registered
  stance, class S0-structural. Kill: a confirmed dark branching ratio.

### DK-16. X17 / ATOMKI
- OPH stance: no 17 MeV state exists in the recovered spectrum; MEG-II
  and PADME resolve it as a systematic. Kill: confirmed boson.

### DK-17. R(D*), R(D)
- OPH stance: lepton universality is exact at gauge level (Yukawa
  differences only); the ~3.3 sigma HFLAV excess fades. Kill: confirmed
  LFU violation at 5 sigma.

### DK-18. Gallium/BEST deficit
- OPH stance: exactly three families (MAR + matter package,
  reality:3035); eV-scale steriles are excluded on that branch, the
  deficit is nuclear/cross-section systematics. The sim carries a gallium
  lane stub (`data/gallium/`, `oph_fpe/gallium/ga71_td.py`): populate it.
  Kill: confirmed sterile oscillation.

### DK-19. Cluster lensing offsets (Bullet-type)
- Corpus mechanism: the dark paper predicts lensing-potential/gas offsets
  through a relaxation law "incompatible with old settled" systems
  (dark_matter:466, 3880): merging clusters are maximally unsettled, so
  the response law there must differ from the galaxy law in a specific,
  computable way. Quantify the predicted offset pattern vs the standard
  collisionless-DM offset; public weak-lensing maps of the Bullet and
  El Gordo systems decide. BUILD, weeks.

### Parking lot (one line each)
- Lithium-7 BBN deficit: corpus silent; no OPH object.
- EDGES 21-cm depth: contested by SARAS-3; no OPH transport object.
- ARCADE-2 radio excess: no OPH object.
- NANOGrav SGWB index (3.2 vs 13/3): no OPH stochastic-background law.
- "Cosmic glitch" Gpc-scale gravity claims: contested; fold into DK-04
  parent when it exists.
- Etherington duality tests: OPH inherits photon-number conservation;
  expected null, no unique signature.

---

## 3. Micro-to-macro signature computations

Where the patch/repair microphysics could surface at laboratory or
astrophysical scales.

### MM-01. Ringdown area-quantum comb
- OPH statement, present in corpus: `S_dS = A/(4 l_P^2) = N_scr`
  (string_selector:467, 2905) and `l_star == l_P` by construction. One nat
  of record capacity therefore costs `Delta A = 4 l_P^2`: the horizon area
  quantum is alpha = 4 in Planck units (nat convention; the bit
  convention gives 4 ln 2 = 2.773; declare the fork).
- Consequence (Bekenstein-Mukhanov/Foit-Kleban form): quantized ringdown
  emission with frequency spacing `delta_f = alpha c^3 / (64 pi^2 G M)`.
  Computed 2026-07-13: for a 62 Msun remnant delta_f = 20.7 Hz (8.3
  percent of the 250 Hz fundamental); 150 Msun gives 8.6 Hz. This is a
  macroscopically observable spacing, and existing Foit-Kleban-style
  analyses of public LVK data constrain alpha weakly; O4/O5 stacking
  sharpens it.
- Gate: the black-hole paper lists horizon discreteness and
  spectroscopy/ringdown bridges as Phase-III continuations (compact:349),
  so today this is S1 with the area quantum inherited from stated
  capacity relations; the bridge theorem (which transitions dominate,
  selection rules) promotes it to S3.
- Data: GWTC-3 ringdown posteriors (public), O4/O5 events.
- Kill: ringdown combs excluded at alpha = 4 once the bridge fixes the
  emission pattern.

### MM-02. Repair noise: the 48-order decoupling wall
- Corpus status: SILENT on repair-induced diffusion/heating/collapse
  (seed noise and repair jitter are declared implementation artifacts,
  compact:1044). The data forces the strong form: if repair events
  coupled to bulk matter as position localization at length l_P, each
  event deposits ~1.3e28 J; Earth's 44 TW heat budget then bounds the
  matter-coupled rate at lambda < 9.7e-67 per nucleon-second, while one
  repair per nucleon per Hubble time is 2.3e-18 per second: 48 orders of
  magnitude of mandatory decoupling.
- Tracker action: state the exact-decoupling theorem (recovered unitary
  dynamics carries zero repair back-action on matter). The theorem is
  data-forced, cheap to state, and instantly banks a passed null. For any
  future finite-length proposal, the mapping onto the CSL (lambda, r_c)
  exclusion plane is defined: X-ray spontaneous-emission bounds give
  lambda < ~2e-13 per second at r_c = 1e-7 m; LISA Pathfinder and
  macromolecule interferometry cover other slices.
- Sim handle: the finite repair transition clock computes a
  continuous rate `gamma = -log(lambda_2)/repair_step_time`
  (`finite_repair_transition_clock.py:565`) with per-run
  `mismatch_trace.csv`: the internal object to normalize if any coupling
  theorem ever emits a physical rate.

### MM-03. Holographic transverse noise
- The Fermilab Holometer excluded Hogan-type transverse jitter. OPH
  recovers exact SO+(3,1) in the refinement limit (PAPER:1736) and emits
  no jitter amplitude law (corpus SILENT): the null is consistent, and
  no OPH bound exists to register (Pro-39 concurs). PASSED-NULL,
  conditional; park.

### MM-04. Can consensus gravity entangle? (BMV stance)
- Corpus status: gravity is recovered as an entanglement-equilibrium
  branch sourcing the metric from the expectation value,
  `G_ab + Lambda g_ab = 8 pi G <T_ab>` (PAPER:1924, Thm 4.3g). BMV-type
  gravitationally-induced entanglement is unaddressed.
- Why it matters: tabletop BMV experiments (two 1e-14 kg masses in
  superposition) target the 2030s. If the OPH consensus channel between
  patches is LOCC-classical, gravity cannot generate entanglement and a
  positive BMV result falsifies the recovered branch's mechanism; if the
  record dynamics transports quantum phase, OPH predicts the standard
  positive result. Either theorem is a registered prediction years ahead
  of the data. THEOREM-TARGET, priority B, zero data cost today.

### MM-05. Clock-network holonomy
- Pro-40: closed-loop clock transport (Sagnac + gravitational redshift
  loops) over GNSS/optical-clock networks as a test of the common causal
  clock map. Needs the OPH forward map for loop holonomy; GATED.

### MM-06. Flyby anomaly replay
- Sim carries 12 flyby certificates, all `OD_REPLAY_PENDING` (Cassini,
  EPOXI I/II/III, Galileo I/II, Juno, MESSENGER, NEAR, Rosetta I/II/III).
  Target structure worth testing in the replay: Anderson's empirical
  constant K = 3.099e-6 equals `2 omega_E R_E / c = 3.0993e-6` to 0.01
  percent, a rotating-observer geometric combination. If the OD replay
  reproduces the anomalous Delta-v pattern with a rotating-screen term of
  that form, this becomes a distinctive OPH lane; if the replay kills the
  anomalies as mismodeling, bank the kill. BUILD, weeks.

### MM-07. Exact-symmetry block (count once)
- Recovered exact structures imply: no LIV dispersion (Fermi-LAT,
  LHAASO), GW speed = c (GW170817), exact EP (MICROSCOPE eta ~ 1e-15),
  photon mass zero (PDG < 1e-18 eV). All pass. The corpus itself notes
  the GW bound is consistency, no theory-side certificate (PAPER:3419).
  One evidence block, class S0.

---

## 4. Cross-domain closures (the decisive comparisons)

### CL-01. RAR-BTFR closure `C = r_B + 2 r_R = 0` [Pro round 2 flagship]
- One deep-limit law forces the closure; global distance, inclination,
  M/L, and a0 shifts cancel in C. Today's aggregate extractions disagree:
  `a0_BTFR / a0_RAR = 1.3837` (0.141 dex). Either the finite-radius/disk
  geometry correction (computable from the source PDE) explains 0.14 dex,
  or the galaxy branch is internally broken regardless of how well each
  fit looks separately.
- Build: matched-galaxy hierarchical pipeline on SPARC with the exact
  disk solver; per-galaxy `C_i`; test intercept 0, slope -2, and the
  source-predicted scatter. 1-2 weeks. The single most decisive
  galaxy-scale computation available.

### CL-02. Rotation-lensing no-slip closure
- The same potential that fits rotation curves must predict
  galaxy-galaxy lensing `Delta Sigma(R)` with no halo refit:
  `rho(r) = (1/4 pi G r^2) d/dr [r^2 g(r)]`, project, compare to
  KiDS/DES/Euclid stacked lensing vectors with full covariance. A
  no-slip violation or a required halo refit kills the response
  interpretation. BUILD, weeks.

### CL-03. Electroweak radiative packet
- One loop-complete `(Delta r, Delta kappa, Delta rho)` map from OPH
  two-point functions connecting chart values to poles, widths, line
  shape, asymmetries, low-Q weak angle, G_F. Numerical hooks from Pro
  round 2: on-shell shape `s2_OS = 0.22279` vs coordinate `0.23074`
  needs `kappa = 1.0357`; chart closure gives `Delta r ~ 0.0333` vs
  representative pole-level `~0.036`: radiative-completion sized, no
  tens-of-sigma category errors. Resolves DK-10. GATED, months.

### CL-04. Single QCD electromagnetic spectral backend
- One `rho_EM(s)` predicting jointly: Delta alpha_had(q), a_mu^HVP,
  a_e^HVP, e+e- moments, Euclidean/lattice moments, and the Thomson
  endpoint in one declared scheme. Feeds DK-09 and the alpha program
  (source root 136.9948 vs no-hadron 137.03596 vs measured 137.035999).
  GATED, months; highest-leverage particle-side object.

### CL-05. Capacity reconciliation and the Lambda inversion
- Finding (2026-07-13 sim survey): the sim's live `DEFAULT_N_CRC` is
  CIRCULAR, built from the observed de Sitter radius
  (`oph_constants.py:12-14`, N ~ 3.31e122). The non-circular EW value
  `N_CRC_EW = 3.5323546226929906e122` derives from P and alpha_U alone
  (observers:607-620 explicitly avoids G, Lambda, H0) and sits unwired in
  imported ledger metadata. The two differ by 6.7 percent, which IS the
  +6.6 percent Lambda gap.
- Actions: (1) wire N_CRC_EW into the sim constants and label the
  de Sitter readout branch circular-diagnostic; (2) declare nats and the
  horizon convention (Pro round 2 section 7); (3) register
  `Lambda l_P^2 = 3 pi / N_CRC_EW = 2.668e-122` vs the Planck display
  2.845e-122 as the standing +6.6 percent test; (4) the reconciliation
  theorem for the 6.7 percent branch gap is a first-class research
  object: closing it makes Lambda a parameter-free EW-side prediction
  spanning 122 orders of magnitude. Days for (1)-(3).

### CL-06. Clock-free dimensionless ratio scoreboard
- Ratios of independently derived lanes need no clock: m_t/m_W = 2.1490
  (+0.07 percent), m_H/m_t = 0.72856 (+0.50 percent), m_p/m_e = 1818.4
  (-0.97 percent), m_W/m_p = 86.46 (+0.94 percent). LIVE; freeze the
  four ratios with their lane gates inherited.

### CL-07. Common-basis flavor packet
- `Y_e, M_nu, Y_u, Y_d` in one scheme at one scale, diagonalized
  together: PMNS via `U_eL^dagger U_nu` (the round-1 correction), CKM +
  Jarlskog, then a future-release holdout (FLAG update, NuFIT 7).
  GATED on the source Yukawa emission; months.

---

## 5. Computational experiments

### Von Neumann (classical)

- **VN-01. Dimension drop.** Replace hash-token record features with
  locality-preserving packet features, add the real perturb-resettle
  channel, rerun the spectral/volume-growth/correlation estimators
  (`bulk/dimensions.py`, walk lengths 1-12). Current runs store a stub
  (`dimension_report.json: not_computed_for_bw_primary_path`). The
  decisive question: does the measured dimension move from 8-12 to ~3?
  A yes opens the 3D-bulk path; a stable no at increasing scale is
  evidence against 3D emergence in this dynamics. Highest-information
  computational experiment in the program.
- **VN-02. W5 coefficients from dynamics.** Umbrella-sample the
  twelve-port defect record, fit the five invariant coefficients, feed
  the frozen decision harness (locus witness ratio 1.8831 vs MCPR target
  1.88901). Closes or kills the conditioned lepton-shape lane (C3).
- **VN-03. Gauge-covariant kernel CMB rerun.** After the K1 mismatch fix
  (label equality ignores gauge, `bw_array.py:313`), recompute the
  finite-clock TT diagnostic (today chi2/bin 1.417 vs CAMB 0.944,
  disfavored) and the cmb-lite shape correlation (today -0.754). Whether
  the disfavored verdict was an artifact of the gauge-blind kernel is an
  open, answerable question.
- **VN-04. Signed parity statistic.** Oriented S3-holonomy correlator
  (transposition = odd class) over freezeout screens with orientation
  shuffles as controls; existing artifacts suffice (holonomy reports,
  worldline catalogs, `s3_class_density`). Output feeds DK-06/DK-07.
- **VN-05. Anomaly report population + icosahedral axes.** Fill the
  `cmb_anomaly_report` stubs from current runs (low-ell, parity, S_1/2);
  add the missing axes statistics; new public-data test: fit the best
  icosahedral frame (six C5 axes, pairwise 63.435 deg) to the Planck
  ell = 2..5 multipole axes and compare against isotropic Monte Carlo.
  The A5 orbit geometry is load-bearing in the lepton sector; finding
  (or bounding) icosahedral alignment in the sky is a free afternoon
  with public SMICA alms.
- **VN-06. Reconstruction dipole.** Source-count dipole of observer
  reconstructions for off-center observers (DK-11).
- **VN-07. In-sim capacity growth.** Track N(t) and Lambda_eff in long
  runs: does the sim's own capacity stay fixed (DK-01 discriminator)?
- **VN-08. Real UHE emission.** Emit the five MaxEnt coefficients from
  actual source runs under `NO_UHE_DATA_USE` (replacing the planted demo
  fixture), then build the propagation/detector forward map (DK-13).

### Quantum hardware

- **QC-00. Existing anchor.** The corpus has a working IBM cloud lane:
  heat-kernel eigenvalue readout on `ibm_marrakesh` / `ibm_fez`, Z3
  sectors at lambda = 3 (`extra/IBM_QUANTUM_CLOUD.md:214-256`). Extend
  rather than invent.
- **QC-01. Icosa12 model.** The sim's `build_icosa12_universe` is a
  12-spin frustrated Z2 model on the icosahedral graph (A5 x Z2
  automorphisms) with exact enumeration in code. Exact-diagonalize the
  quantum (transverse-field) extension classically, then VQE the same
  Hamiltonian on hardware: an independent verification channel for the
  port-model ground-state structure the W5 lepton lane conditions on.
- **QC-02. Transfer-gap spectra.** The Yang-Mills lane computes a finite
  transfer-gap proxy on 2^4 Wilson lattices (`yang_mills_gap.py:354`,
  reversible projection gives a self-adjoint operator). Estimate the gap
  at larger volumes on quantum hardware (QPE/variational); connect to
  the glueball ratio packet (Pro-37).
- **QC-03. Structural null.** Exact consensus recovery implies no
  intrinsic device-independent gate-error floor beyond hardware physics.
  Register the stance; any confirmed intrinsic floor at accessible depth
  falsifies exact recovery. Class S0.
- **QC-04. Bell/Tsirelson/I3 battery.** CHSH = 2 sqrt 2, third-order
  interference I3 = 0, no-signaling: inherited exact theorems, testable
  to arbitrary precision on hardware, count once as one block.

---

## 6. Null-test battery

| # | Null | Data (passing today) | OPH status | Teeth |
|---|---|---|---|---|
| NT-01 | No LIV dispersion | Fermi-LAT GRBs, LHAASO | recovered SO+(3,1), refinement-limit | confirmed LIV kills the core |
| NT-02 | alpha-dot = G-dot = mu-dot = 0 | Oklo, optical clocks, ESPRESSO (Webb dipole unconfirmed) | SILENT: constants fixed from P implies zero drift; theorem unstated | state it; three datasets bank immediately; confirmed drift kills pixel closure |
| NT-03 | Photon mass = 0 | PDG < 1e-18 eV | unbroken U(1) exact | massive photon kills quotient |
| NT-04 | Equivalence principle | MICROSCOPE eta = (-1.5 +/- 2.7)e-15 | geodesic inheritance | confirmed violation kills recovery |
| NT-05 | GW speed = c, 2 modes | GW170817; LVK polarization | consistency, uncertified | anomalous speed/polarization kills |
| NT-06 | Proton decay | Super-K > 2.4e34 yr (e+ pi0) | corpus: `tau_p^(gauge) = infinity`, no leptoquarks, dim-5 forbidden (compact:11576) | ANY confirmed decay kills the gauge branch; Hyper-K null is the registered outcome |
| NT-07 | No particle dark matter | LZ/XENONnT/PandaX nulls; ADMX slices | conditional on response branch; cosmological abundance selector fork DISCLOSED (dark paper imports Omega_c baseline) | confirmed WIMP/axion halo detection kills the response interpretation of galaxy phenomenology |
| NT-08 | No 0nubb at current sensitivity | KamLAND-Zen/GERDA m_bb < 28-122 meV | conditional on proposed rank lane (m_bb 1.5-3.7 meV) | discovery at LEGEND-1000 sensitivity kills the rank lane (and NO with m1 = 0) |
| NT-09 | No 4th family, steriles, exotics, extra U(1) | LEP N_nu = 2.9840(82) (+1.95 sigma for 3), Higgs fits, BEST tension | MAR + matter package theorem; string selector adds no-exotics | any fourth chiral family or light exotic kills the branch |
| NT-10 | No GW echoes / horizon reflectivity | LVK echo searches (null) | inherited GR horizon at leading order | confirmed echoes exceed the recovered structure |
| NT-11 | Neutron EDM / theta-bar | d_n < 1.8e-26 e cm | corpus explicitly does NOT derive theta-bar (deriving:3678); THEOREM-TARGET | a theta-bar = 0 theorem banks the null; a nonzero prediction meets a 10x-tighter bound this decade |

---

## 7. Laboratory and hardware lanes

- **LB-01. chi_nu coherent-matter force.** `chi_nu^can = exp(-P/24) =
  0.9343006395`; engineering window `chi_nu^eng in [9.34e-23, 1.00e-22]`
  at N_coh = 1e22 (falsifiability ledger:139, 265). Pro round-1
  engineering audit stands: fix the 43.5 kHz drive bug, replace the
  passive Schottky detector, add power-matched shuffled controls, use a
  geometry capable of lift. Default hypothesis zero force.
- **LB-02. SHA-256d.** Two-sided: (a) blinded photonic enrichment
  `beta(k)` with predeclared stopping (hardware, weeks); (b) free
  public-data test: accepted-block hashes u = H/T conditional-on-target
  uniformity from any full node (Pro-47), a clean statistical null OPH
  must pass.
- **LB-03. Cross-species clock ratios.** Freeze the Cs-based scale once,
  predict Sr/Yb/Al+/Rb ratios (Pro-46): converts the G checksum
  (6.674299996e-11, display-exact) into a test and probes the CODATA G
  scatter question.
- **LB-04. Low-acceleration laboratory EFE.** Saddle-point missions or
  shielded interferometry test the DK-08 predicate directly; years-scale;
  parked until the predicate theorem exists.

---

## 8. Freeze registry shortlist

The suite holds zero frozen prospective predictions (all four audits
agree). These are the rows to hash-pin in one registry artifact BEFORE the
next releases (PDG 2026, DESI DR3, O5, Hyper-K first light, LiteBIRD).
Rows marked (cond.) freeze together with their named condition.

| Row | Value | Compare against | Trigger |
|---|---|---|---|
| n_s = 1 - P/48 | 0.9660214 | SPT-3G, ACT DR6+, CMB-S4 combined tilt | next combined-likelihood release |
| W/Z chart pair + s2_OS | 80.330 / 91.119 / 0.22279 | named W releases separately | CL-03 closure |
| m_H(m_t) criticality relation | 125.72 GeV at measured m_t | PDG m_H | PDG update |
| alpha_s(M_Z) | 0.11834 | world average | PDG update |
| Lambda_QCD(3) | 334.8 [319,350] MeV | FLAG | FLAG update |
| Ratio scoreboard (4 ratios) | CL-06 values | PDG/CODATA | PDG update |
| Glueball 0++ | 1.41-1.61 GeV | BESIII PWA f0(1500)/f0(1710) | next BESIII PWA |
| Capacity Lambda (EW branch) | Lambda l_P^2 = 2.668e-122 | Planck/DESI Lambda display | DESI DR3 |
| (w0, wa) (cond. fixed-N theorem) | (-1, 0) | DESI DR3 + SNe | DR3 release |
| Sigma m_nu (cond. rank lane) | 0.0588 eV, NO | DESI DR3 + CMB | DR3 release |
| m_bb (cond. rank lane) | 1.5-3.7 meV | LEGEND-1000/nEXO | first results |
| Proton decay | no signal (tau_gauge = infinity) | Hyper-K | first exposure years |
| Ringdown area quantum (cond. bridge) | alpha = 4 nat (2.773 bit) | O4/O5 stacked ringdowns | O5 catalog |
| Birefringence (cond. emission theorem) | 0.37501 deg | LiteBIRD, PR4 reanalyses | theorem, then release |
| Dark-siren H0 | 67.4 +/- 1 | LVK O5 dark sirens | O5 catalog |
| IR filter (cond. source emission) | (q_IR, t_IR), ell_IR ~ 32 | Planck low-ell likelihood | emission theorem |
| UHE coefficients (after VN-08) | real emission values | IceCube/Auger/LHAASO | VN-08 completion |
| Neutron-bottle stance | bottle correct | next beam remeasurement | publication |
| Exactly-3 block | N_nu = 3, no steriles/exotics | LHC, BEST resolution, N_eff | ongoing |
| No particle DM (cond. response branch) | all direct searches null | LZ full exposure, XLZD | ongoing |

Registry mechanics: one JSON + markdown artifact in
`reverse-engineering-reality/`, SHA-256 of inputs and code, named
acceptance band per row, scored on release, hits and kills recorded in
place. Pattern to copy: the boundary-scale candidate registry and the
`NO_UHE_DATA_USE` receipt.

---

## 9. Pro catalog integration

- The 51-row machine-readable catalog:
  `audit-bundles/ROUND2_EXTENDED/oph_round2_candidate_test_catalog.csv`
  (domains: galaxy dynamics/lensing, external field, cosmology,
  electroweak, Higgs/top, leptons, neutrinos, quarks, gauge structure,
  fine structure, QCD/HVP, Yang-Mills, gravity, synchronization,
  foundations, string selector, absolute scale, SHA-256d, thinking,
  paradise). Rows flagged "new decisive" map here to CL-01, CL-02,
  CL-03, CL-04, DK-08, CL-07, QC-02, LB-02, LB-03.
- Dependency rules: `oph_round2_dependency_graph.csv`; count correlated
  descendants once (three lepton ratios = one phase; SM quotient
  consequences = one selection; Born/Tsirelson/I3 = one theorem).
- Primary sources and release discipline:
  `oph_round2_primary_sources.md`; receipt template:
  `OPH_PREDICTION_RECEIPT_TEMPLATE.md`.
- Pro's two structural verdicts adopted tracker-wide: cross-domain
  closures beat isolated decimal proximity, and every retrospective
  patch is charged as description length.

---

## 10. Execution order

Next 90 days, ordered by information per unit effort:

1. **FZ-01 registry artifact** (1 day): hash-pin section 8 as it stands,
   conditions declared. Converts the whole program from scoreboard to
   experiment.
2. **CL-05 capacity wiring** (hours-days): N_CRC_EW into the sim, circular
   default demoted to diagnostic, nats convention declared.
3. **VN-04 + VN-05** (days): parity statistic and anomaly population from
   existing artifacts; icosahedral-axes test on public Planck alms.
4. **MM-01 ringdown scan** (days): Foit-Kleban alpha = 4 comb against
   GWTC-3 public posteriors.
5. **DK-01 mock w0-wa fit** (days): does the capacity gap mimic thawing?
6. **VN-08 UHE real emission** (days): the demo fixture is the single
   most misleading object in the suite if left as is; replace it, then
   build the forward map (DK-13).
7. **VN-01 dimension-drop run** (days-weeks, couples to the sim fix
   backlog): the decisive 3D-emergence experiment.
8. **CL-01 RAR-BTFR matched pipeline** (1-2 weeks): the decisive
   galaxy-branch experiment.
9. **Theorem queue** (standing): fixed-N w-law (DK-01), rank lane
   (DK-02), applicability predicate (DK-08), exact-decoupling (MM-02),
   BMV channel (MM-04), alpha-dot zero (NT-02), theta-bar (NT-11),
   area-quantum bridge (MM-01), parity emission (DK-06/07), IR-filter
   emission (DK-05).
10. **CL-04 backend + CL-03 packet** (months, parallel): the two objects
    that turn the particle sector from charts into observables.

---

## Appendix A. Session-verified numbers (2026-07-13)

Receipts: `signature_checks.py` in the session scratchpad; agent survey
reports; Pro bundle recompute JSONs.

- alpha_U/(2 pi) = 0.37501 deg; Eskilt-Komatsu beta = 0.342 +0.094/-0.091
  deg; pull +0.35 sigma; 4 expressions tried.
- Ringdown comb: delta_f = alpha c^3/(64 pi^2 G M); alpha = 4 nat;
  62 Msun -> 20.7 Hz (8.3 percent of f220); 150 Msun -> 8.6 Hz.
- Repair wall: Planck-kick 1.27e28 J; Earth 44 TW -> lambda < 9.7e-67 /s
  per nucleon; one-per-Hubble-time = 2.3e-18 /s.
- Capacity: Lambda l_P^2 (EW) = 2.668e-122 vs display 2.845e-122 (+6.6
  percent); H_Lambda(EW) = 53.98, H_Lambda(dS display) = 55.76,
  Planck-implied 55.74 km/s/Mpc.
- Sigma m_nu floor (NO) = 0.0588 eV; m_bb(m1 = 0) = [1.50, 3.70] meV;
  m_beta ~ 8.8 meV.
- Flyby: 2 omega_E R_E/c = 3.0993e-6 vs Anderson K = 3.099e-6.
- W repair 80.39048 pulls: CDF -4.58, ATLAS24 +1.51, CMS24 +3.06,
  LHCb +1.14, PDG(no CDF) +1.60 (measurement-only).
- g-2: OPH backend -3.1 percent vs KNT19 -> a_mu^HVP shift -213e-11 +/-
  ~194e-11.
- LEP N_nu = 2.9840(82): exact 3 at +1.95 sigma.
- Hemispherical sampling floor sqrt(2/N_modes(l <= 64)) = 0.022 vs
  measured A ~ 0.07(2).

## Appendix B. Corrections to prior internal claims (log)

1. **UHE coefficients are a planted demo.** `runs/uhe/coefficient_emission/`
   carries `SYNTHETIC_DEMO_FIXTURE`, `physical_claim: false`,
   `PLANTED_COEFFICIENT_RECOVERY_ONLY`; the falsification catalog's B1
   description ("cleanest prospective object") overstated it. Two builds
   (VN-08 + forward map) precede any registered UHE test.
2. **Sim live N_CRC is circular.** `oph_constants.DEFAULT_N_CRC =
   pi (R_dS/l_P)^2` uses the observed de Sitter radius; the non-circular
   `N_CRC_EW` sits unwired in `data/oph_cross_repo_current/imported_oph_artifacts/particle_derivation_gap_ledger.json:82`.
3. **Flyby count is 12**, all `OD_REPLAY_PENDING` (earlier notes said 9).
4. **Unification-scale fork.** The papers carry `M_U = 1.20665e16 GeV`
   (string selector) and `M_U(P) = E_P e^{-2 pi} P^{1/6}` (fine
   structure), which evaluates to 2.47e16 GeV on the E_star display; the
   particle-code lane uses 2.474e16. Reconcile the two displays before
   any quantitative proton-decay statement beyond `tau_gauge = infinity`.
5. **Neutrino rank lane is proposed, unstated.** The corpus's only tuple
   (Sigma = 0.0900 eV) is rejected by NuFIT 6.1; "lightest mass = 0"
   appears nowhere in the papers. Section 2 DK-02 and NT-08 rows are
   conditional on stating that lane.
6. **Two comparators were undocumented** in prior inventories: gallium
   (`data/gallium/`) and FRB/compact-transient receipts
   (`oph_fpe/compact_transients/receipts.py`).
