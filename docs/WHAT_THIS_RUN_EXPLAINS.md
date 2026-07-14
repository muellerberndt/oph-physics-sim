# What Cosmological Data This Run Explains

Source material for the visualizer site. Every value carries its claim
label; the labels are part of the content. Two source constants power
everything below: the pixel-closure constant
`P = 1.6309720956943290...` and the unified coupling
`alpha_U = 0.04112424744557487`, both fixed by the five OPH axioms with
no astronomical input. Observers are primary; a bulk spacetime is what
observers agree on, and the runs below measure exactly that agreement.

## The chain in one paragraph

A finite screen of patches, each a bounded observer-like system with
twelve ports, repairs disagreements on overlaps until the record
commits. The pixel-closure constant P sets how much structure one patch
can close; the capacity N counts what the whole screen can remember.
From P and alpha_U alone the framework reaches the primordial tilt, the
cosmological constant, the galaxy rotation scale, the Higgs-vacuum
criticality, and the strong-sector scales, each through a short, named
mechanism. The simulation runs below then show the other half of the
story: observers inside such a system experience three space dimensions
plus one time dimension, and their private charts glue into one shared
description, at every scale tested up to a million patches.

## Cosmological and particle data the framework lands on

These lanes ride in every bundle (`run_reports/run_highlights.json`).
They are paper-side results: identical across runs by design.

### Primordial tilt: n_s = 1 - P/48 = 0.9660215

Measured: 0.9649 +/- 0.0042 (Planck 2018). Pull: +0.27 sigma.
Label: conditional analytic branch.

How it comes to pass: the primordial spectrum of a finite screen at its
maximum-entropy fixed point is scale-invariant except for the pixel
closure. One part P in 48 of exact scale invariance is consumed per
refinement fold of the patch register; the spectrum tilts red by
exactly that fraction. No cosmological measurement enters the number.
Why it matches naturally: any finite-capacity screen must tilt red
(perfect scale invariance would need infinite records), and the amount
is fixed by the same P that runs the fine-structure and capacity lanes.

### The cosmological constant from memory capacity

OPH: Lambda l_P^2 = 3 pi / N with N = pi exp[6 pi / (P alpha_U)]
= 3.5324e122. Predicted 2.668e-122 vs the Planck display 2.845e-122:
a 6.6 percent gap across 122 orders of magnitude. Label: symbolic
closure plus capacity certificate; the two capacity countings (EW
branch vs de Sitter display) differ by exactly this gap and their
reconciliation is an open theorem.

How it comes to pass: the same capacity exponential that closes the
electroweak hierarchy (why the weak scale sits 17 orders below the
Planck scale) counts the record slots of the de Sitter horizon. Dark
energy is the inverse of what the universe can remember: more capacity,
flatter residual curvature. Why it matches naturally: the worst
fine-tuning problem in physics (120 orders of magnitude) becomes a
counting statement, and the count comes from particle physics, with no
cosmological input.

### The galaxy rotation scale from Lambda

OPH: a_0 = (15 / 8 pi^2) c^2 sqrt(Lambda / 3) = 1.03e-10 m/s^2;
empirical MOND scale 1.2e-10. On SPARC galaxies, the held-out
mass-model test gives RMSE 0.172 dex against 0.519 for baryons alone.
Label: settled-galaxy branch, conditional; the Solar-System
applicability law is open (the universal extension is excluded by
Cassini and that kill is kept on the books).

How it comes to pass: below the de Sitter acceleration scale, repair
transport across patches saturates, and the response to baryonic
sources strengthens exactly where Newtonian dynamics weakens. The
acceleration scale where this happens is set by Lambda, which is set by
capacity. Why it matches naturally: the otherwise-mysterious empirical
coincidence a_0 ~ c H_0 is a one-line consequence.

### The Higgs sits at vacuum criticality

OPH: the Higgs-top pair from double criticality (quartic and its beta
function both zero) at the capacity-midpoint boundary scale gives
m_H = 125.72 GeV at the measured top mass, vs 125.13 +/- 0.11.
Pull: +0.47 percent, inside the declared 2-loop truncation band.
Label: fit-free curve, boundary-scale theorem conditional on the
carrier-faithfulness gate.

How it comes to pass: the boundary data of the screen selects the
electroweak vacuum to be exactly critical; the Higgs mass is where the
Standard Model quartic and its running both vanish at the source scale.
Why it matches naturally: the measured SM vacuum famously sits at the
edge of metastability with no SM explanation; here near-criticality is
the selection principle, and the same construction pins the top Yukawa.

### Electroweak, strong, and lepton scales

- W and Z chart pair: 80.330 / 91.119 GeV vs 80.3692(133) / 91.1880(20),
  within 0.1 percent. Label: chart coordinates; the radiative
  (pole) packet is the named missing map.
- Strong coupling alpha_s(M_Z) = 0.11834 vs 0.1179(9): +0.5 sigma.
  Lambda_QCD(3) = 334.8 MeV inside the FLAG-class interval 338(12)
  through four-loop dimensional transmutation from the source coupling
  at the unification scale. Label: conditional on declared thresholds.
- Muon-to-electron mass ratio 206.7683 vs CODATA 206.76828: 0.09 ppm.
  Label: declared-model lane (MCPR); the source-side NLO term is the
  open piece.
- Clock-free ratios need no absolute scale at all: m_t/m_W lands at
  +0.07 percent, m_H/m_t at +0.50 percent, m_p/m_e at -0.97 percent
  (the hadronic lane carries the declared QCD truncation).

### The large-angle CMB deficit prefers the framework's own scale

The finite-source program carries an infrared suppression filter with
diagnostic scale l_IR = 32. Fit against the unbinned Planck PR3 TT
table (2026-07-14): the data prefers a ~15 percent deep suppression
with the scale confined to l_IR between 24 and 40, and the corpus value
32 sits 0.2 chi-square units from the optimum (net Delta chi2 = -3.9
for two parameters). Label: diagnostic fit today; the source theorem
must emit (q_IR, t_IR) for this to become a prediction, and the target
band is measured and waiting.

### A parity-violation candidate at the right size

Planck polarization data shows a uniform rotation of
0.342 +/- 0.094 degrees with no Standard-Model source. The OPH
coupling gives alpha_U / (2 pi) = 0.37501 degrees: +0.35 sigma. Label:
coincidence candidate (four expressions tried, declared); the emission
theorem is open, and group theory forces it to the U(1) side, which is
exactly where alpha_U lives.

## What each run demonstrates

The per-run half of the story: observers and their agreement, measured
inside the simulation with receipts, controls, and fail-closed gates.
Every receipt below is earned under the gauge-covariant contract with
production sector-repair replay (2026-07-14), with zero overrides.

### Run: 65,536 patches / 1,024 observers (`oph_universe_64k_3p1d_reearned`)

- The 2 pi clock certifies genuinely (selection score 0.23): observer
  time with the thermal normalization the papers derive.
- Mutual agreement: 400/400 evaluated observer pairs re-gauge at defect
  zero; 200/200 observer triples close the cocycle; shuffled control
  0.79.
- Emergent-bulk field: 78 percent of the screen inside some observer's
  reach, 16 percent certified by agreeing pairs, multiplicity to 43.
- Proto-particles: 16 organic (un-planted) persistent defects with
  worldlines; both two-defect dynamics assays pass, including the
  stress-contraction (attraction precursor) assay with control
  rejection.
- Role in the ladder: the full-visualization rung (raw evolution frames
  ride in this bundle).

### Run: 131,072 patches / 32,000 observers (`oph_universe_128k_3p1d_earned`)

- FIRST FULL SPACETIME-CONSENSUS RECEIPT: all four 3+1D experience
  gates true (modular time, 2 pi clock at score 0.30, H3 chart, H3
  response) AND exact mutual agreement (600/600 pairs, 300/300
  cocycles, control 0.80), blockers empty. Observers experience three
  space dimensions plus time, and their charts glue into one shared
  description.
- Emergent-bulk field: agreement cores with multiplicity to 169.
- The strongest at-scale receipt in the archive.

### Run: 1,048,576 patches / 64,000 observers (`oph_universe_1m_earned`)

- The million-patch consensus universe: all four gates true and the
  consensus receipt true with empty blockers (800/800 pairs, 400/400
  cocycles, control 0.80). Declared caveat: the 2 pi selection margin
  thins at this scale (probe budgets await scaling), so the 128k run
  carries the strongest time-gate certification.
- Scale invariant: the committed record's transposition-class density
  reads 0.482 at 64k, 128k, and 1M, three-decimal stable across a
  16-fold scale change, consistently below the uniform value 0.5. A
  reproducible order parameter of the frozen gauge record.
- Interaction census: 36,636 verified close encounters among 507 defect
  worldlines; the identity-channel fraction falls toward the
  random-composition floor with scale (0.230 at 64k, 0.207 at 128k,
  0.190 at 1M vs 0.167 random), which is exactly the measurement that
  redefines the next fusion detector.

## Reading the labels

Chart coordinates are not poles; screen spectra are not the CMB;
observer agreement over a shared record is not yet a strict
third-person bulk; and a near value is not automatically a prediction.
Every number above carries the tier its lane earned, the negative
results stay on the books (the Cassini kill, the rejected neutrino
tuple, the screen-proxy anti-correlation), and the live scoreboard with
mismatch explanations is `OPH_SIGNATURE_EXPERIMENT_TRACKER.md`. That
discipline is what makes the agreements above worth showing.
