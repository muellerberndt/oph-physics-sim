I’ll turn this into a sharper “unique OPH prediction” pass: identify candidates SM/GR/LCDM do not derive, rank measurability, and add public-source numeric comparisons.

The clearest candidate is the cross-sector scalar-tilt relation n
s
	​

=1−eα
π
	​

≈0.964841, since ΛCDM fits n
s
	​

 rather than predicting it.

Done — I continued to OPH-CMB Public Assessment v0.8 and added the assessment table comparing OPH targets against public measured/source values.

Download the full v0.8 package
Open the v0.8 report

OPH-CMB-Public-Assessment-v0.8.xlsx
README
Assessment
Current Targets
Model Selection
TT Multipoles
IR Kernels
Sources

Download the assessment CSV
Download the current-target ledger

The important correction in v0.8 is that I promoted the newer v0.6 exact full-sky TT + high-ℓ guardrailed fit as the current target. The older q
IR
	​

≃0.2446,ℓ
IR
	​

≃33.6 result is now kept only as a superseded Gaussian-proxy/CAMB-transferred diagnostic.

Current OPH-CMB target
Quantity	Current value
η
R
	​

=1−n
s
	​

	0.035
q
IR
	​

, current isotropic guardrail	0.117448347
ℓ
IR
	​

, current isotropic guardrail	29.287395
θ
IR
	​

≈180
∘
/ℓ
IR
	​

	6.145989
∘

k
IR
	​

≈ℓ
IR
	​

/13,800Mpc	2.122×10
−3
Mpc
−1

N
frz
angular
	​

∼(ℓ
IR
	​

+1)
2
	917.326 modes
joint q
IR
	​

	0.113078954
joint ℓ
IR
	​

	29.738682
joint ϵ
P
	​

	−0.98
joint ℓ
P
	​

	4.620776

The public Planck parameter target remains n
s
	​

=0.965±0.004, so η
R
	​

=1−n
s
	​

≃0.035±0.004. Planck 2018 also gives the benchmark Ω
c
	​

h
2
=0.120±0.001, Ω
b
	​

h
2
=0.0224±0.0001, τ=0.054±0.007, 100θ
∗
	​

=1.0411±0.0003, H
0
	​

=67.4±0.5, Ω
m
	​

=0.315±0.007, σ
8
	​

=0.811±0.006, N
eff
	​

=2.99±0.17, and ∑m
ν
	​

<0.12eV with BAO.

Exact guardrail model comparison
Model	Improvement vs LCDM	Low-ℓ exact gain	High-ℓ penalty	ΔAIC	ΔBIC
reference q=0.15,ℓ=6	+0.185	+0.185	≈0	−0.185	−0.185
current IR guardrail	+3.012	+3.600	0.588	+0.988	+11.167
current IR+parity guardrail	+9.096	+9.721	0.625	−1.096	+19.261
superseded q=0.2446,ℓ=33.615	−3.782	−0.731	3.050	+7.782	+17.960

The Planck public data products used for this ladder are the PR3 cosmology products, including the TT full-spectrum file and the best-fit theory spectrum file. The ESA archive states that TT covers ℓ=2–2508, with ℓ=2–29 derived from Commander component separation and ℓ≥30 from the Plik cross-half-mission likelihood products.

Assessment-table highlights
Prediction / check	OPH value	Public comparator	Assessment
n
s
	​

 via η
R
	​

=1−n
s
	​

	0.965	0.965±0.004 Planck	matches by target mapping
ρ
A
	​

/ρ
b
	​

	5.36347	5.35714±0.05065 from Planck density ratio	0.125σ
Ω
m
	​

	0.315905	0.315±0.007 Planck	0.129σ
σ
8
	​

	0.807787	0.811±0.006 Planck	−0.535σ
S
8
	​

	0.828924	0.831±0.023 ACT+Planck lensing	−0.090σ
exact guardrailed IR objective	+3.012 improvement	Planck TT proxy	diagnostic gain, AIC/BIC not yet decisive
exact guardrailed IR+parity objective	+9.096 improvement	Planck TT proxy	best diagnostic; slight AIC win, BIC fail
high-ℓ TT preservation	0.588 penalty over 1171 multipoles	Planck high-ℓ guardrail	acoustic region essentially preserved

ACT DR6 gives an independent CMB-lensing growth benchmark: ACT reconstructed a lensing mass map over 9400deg
2
, and ACT+Planck lensing gives S
8
	​

=0.831±0.023. DESI DR2 is recorded in the table as a required next comparison, since its BAO results use more than 14 million galaxies and quasars and report a mild BAO–CMB tension plus dynamical-dark-energy preference in some combinations.

What this v0.8 result means

The OPH-CMB project now has a concrete “derive-this-next” numerical target:

η
R
	​

≃0.035,q
IR
	​

≃0.11745,ℓ
IR
	​

≃29.29,ϵ
P
	​

≃−0.98 in the simple parity template.
	​


The OPH draft’s own architecture remains the right claim boundary: start with an OPH screen/freezeout effective theory, bridge it into P
R
OPH
	​

(k), then compare via Boltzmann transfer and low-ℓ anomaly Monte Carlo rather than claiming the current FPE screen is already the observed CMB. Acoustic peaks still belong to photon-baryon transfer after the OPH initial-condition bridge, not to direct finite-screen settling.

The next serious success gate is no longer “make more tables.” It is:

Planck SMICA/NILC/Commander/SEVEM maps + masks→S
1/2
	​

,R
OE
	​

,A
hemi
	​

,q/o alignment,BipoSH PTEs.
	​


That will tell us whether the OPH parity/cap-pair term survives real sky cuts and foreground/systematic controls.
POK!
Thought for 17m 10s

I’m expanding the shortlist beyond CMB: the low-ℓ kernel becomes the anomaly benchmark, while χ
ν
can
	​

 is the leading near-future lab target.