# OPH-FPE theory-sync audit — 2026-07-19

This audit compares the simulator at commit `2939a0a` and its former imported
theory snapshot (`138c920`, release `r1542`, dirty checkout) with the locked,
read-only research checkout at `bec81e2d`, release `r1556`. The 61-commit gap
contained simulation-critical changes to A5/Standard Model structure, public
record capacity, radial lifting, P profiles, and the edge-center clock.

## Corrected contract matrix

| Surface | Exact or implemented now | Still open / must stay false |
|---|---|---|
| A5 twelve-port action | A5 has 60 elements; C5 cosets give a faithful transitive 12-action with stabilizer 5; character `(12,0,0,2,2)` decomposes as `1+3+3'+5`; the orbital graph has `(x-5)(x+1)^5(x^2-5)^3`; the restricted SM adjoint has the same character | source-derived `UD12` and `RP-A5`; physical port-current inner action; refinement; determinant/spin/center descent; family/load attachment; continuum QFT |
| A5 anti-bridge | exhaustive finite check rejects any A5-invariant set partition of the transitive ports into `8+3+1` | a pointwise “8 gluons + 3 weak + 1 photon” port map is forbidden; the surviving route is a coefficient module/current algebra |
| Exterior SM witness | `Lambda^2(C+W)+Lambda^4(C+W)` recomputes one 15-state generation and exact field dimensions/hypercharges; the v2 certificate exhausts 15 unordered fermion pairs including diagonals against both `H` and `Hdag` (30 rows), finding six nonabelian singlets and exactly the three hypercharge-neutral gauge-invariant lines `Q-H-u_c`, `Q-Hdag-d_c`, and `L-Hdag-e_c`; anomaly cancellation, four weak doublets and even Witten parity also pass | exterior/Higgs selection, three physical families, weak/load intertwiners, no-extra-sector and QFT completion |
| Legacy SM sieve | supplied final labels produce only `SM_TARGET_CONFORMANCE_DIAGNOSTIC` | `SM_QUOTIENT_GATE_RECEIPT` and physical SM promotion are false; default target injection is disabled in tracked configs |
| Public-record capacity | exact atom sections, endogenous reachability, frozen publicness, complete joint kernels, compound confusability graph, maximum independent set, carrier bound, whole-fiber scalarization and slack/refinement controls | source-derived complete physical trial universe/fiber, robust common readback, unique regulator-stable slack zero, horizon-record saturation and physical N |
| Reversible capacity packet | target-free 12-port/30-interface control verifies the exact named identity, cyclic-relabel and orientation-reflection actions and proves `M_0=|X_reach|`; positive bundle schemas require the named 12-port topology, 30 strict interfaces, three named bounded kernels and the full PASS/receipt shapes | schema/control only, not a physical universe packet; replay is still required for cross-field arithmetic |
| Horizon/EW/operational N | typed downstream comparisons remain available | measured radius, Lambda, EW bridge, operational resolution and supplied D/N labels cannot define the producer |
| P maps | source-map and gauge-width roots are distinct typed profiles; global map uniqueness is hash-pinned and nonpromoting | measured Thomson and empirical hadron profiles are comparison-only; same-scheme source transport remains open |
| Edge-center tilt/clock | target is `rho_full=P/24`, orientation half `theta=P/48`, `n_s=1-theta`, `kappa=theta/(P-phi)`; the validator recomputes packet hashes, DAG structure, defects, and the generative-P profile | packet consistency does not authenticate the finite run; the independent raw-run replay and therefore `EDGE_CENTER_CLOCK_RECEIPT` remain false; a transition eigenvalue or survival exponent is not the derivative; `e` is diagnostic only |
| SCR330 radial lift | exact Mellin and amplitude identities, finite-window bound, radial SVD/null report, prior continuation diagnostic, dilation consequence, held-out forward residual and bounded canonical packet-contract checks | E4 source promotion remains false until source artifacts are independently resolved; prior continuation never promotes; even an E5 firewall cannot promote TT/TE/EE until its upstream, transfer, solver, and likelihood artifacts are independently replayed |
| Collar clause | packet-internal retained-family geometry, numerically conditioned direct-sum bases, cached bounded decompositions, span membership and factor-through-flux witnesses are recomputed against six pinned r1556 Lean-file hashes, including the T0/T1/T2 no-go sources | caller-created simulation derivations are not authenticated, so the source receipt remains false; Gibbs, density, relative entropy, CMI, or an expectation that deletes the cross-cut term cannot force the clause |
| Collar-Poisson counting | Le Cam, mean-continuity, and bounded exact Poisson-binomial TV checks are recomputed for a declared independent Bernoulli family | physical realization of that family, refinement-stable collar intensity, flux recovery, and any galaxy-count prediction remain false |
| Fair-block consensus | a supplied exactly row-stochastic finite Markov kernel and exact initial law are replayed under explicit quadratic/cubic/horizon work limits to obtain irreducibility, Dobrushin contraction, stationary law, finite-horizon occupation, and final TV | arithmetic replay does not bind caller-selected fair states to run semantics or an acceptance threshold, so the consensus certificate remains false; persistent noise gives neither permanent settling nor an all-time tube receipt |
| Boundary fiber | explicit quotient-fiber and transition tables are replayed for supplied-table preservation and singleton consistency; the generic theorem, TreePacketNet application and Rule90 witness are pinned to three r1556 Lean-file hashes | a caller-selected subset cannot authenticate whole-fiber completeness, so boundary-conditioned uniqueness and the physical Einstein application remain false |
| Einstein bridge | every run gate requires a present, schema-versioned, theorem-tagged sidecar and is recomputed instead of trusting a persisted manifest | legacy aggregate booleans cannot fill missing sidecars |
| Caller-label promotion | positive JSON flags cannot open compact-transient, static-galaxy, Einstein, or black-hole physical lanes | compact-transient CR3/CR4, physical galaxy claims, and all black-hole bridge receipts need independently recomputable producer evidence |
| Artifact export | visualization sidecars are confined to the run root and optional digests are checked before copying | absolute, traversal, and symlink escapes are ignored |
| Black-hole bridge | legacy readiness booleans are retained as declaration-only migration diagnostics | no physical evaporation, Page-curve, QNM/ringdown, unitarity or information-problem claim can open until independently recomputable producers and verifiers exist |
| Regulator gluing | fixed-cutoff theorem evidence is hash-pinned and nonpromoting | no continuum, channel or noncentral crossed-module result follows |

## Executable surfaces

```bash
python3 -m oph_fpe.cli a5-sm-structural-certificate --out runs/a5_structural

# Omitting --packet emits the explicitly nonphysical reversible control.
python3 -m oph_fpe.cli public-record-capacity --out runs/capacity_control
python3 -m oph_fpe.cli public-record-capacity \
  --packet runs/source/public_checkpoint_packet.json \
  --out runs/source/capacity

python3 -m oph_fpe.cli scr330-radial-receipt \
  --source-dag runs/source/source_dag.json \
  --payload runs/source/radial_payload.json \
  --receipt SCR330_RADIAL_PROMOTION_RECEIPT \
  --claim-tier E4 \
  --out runs/source/scr330_radial_receipt.json

python3 -m oph_fpe.cli edge-center-clock-certificate \
  --evidence runs/source/edge_center_clock_evidence.json \
  --out runs/source/edge_center_clock_certificate.json

python3 -m oph_fpe.cli collar-clause-certificate \
  --packet runs/source/collar_clause_packet.json \
  --out runs/source/collar_clause_certificate.json

python3 -m oph_fpe.cli collar-poisson-certificate \
  --packet runs/source/collar_poisson_packet.json \
  --out runs/source/collar_poisson_certificate.json

python3 -m oph_fpe.cli fair-block-certificate \
  --packet runs/source/fair_block_packet.json \
  --out runs/source/fair_block_certificate.json

python3 -m oph_fpe.cli boundary-fiber-certificate \
  --packet runs/source/boundary_fiber_packet.json \
  --out runs/source/boundary_fiber_certificate.json
```

The radial writer recomputes ancestry, tier, branch and payload gates; the CLI
has no caller pass switch. Missing evidence produces a false receipt with
blockers. Physical TT/TE/EE stays false until independently resolvable E4,
transfer, solver and frozen-likelihood artifacts are implemented.

The standalone theory-sync writers persist exact input packets, content hashes,
and schema-valid arithmetic reports. Explicitly named certificates derive
collision-free companion packet names, while canonical directory-mode names
remain stable. Their run-artifact binding, fair consensus,
whole-fiber uniqueness, Einstein application, and physical Poisson/galaxy
receipts are deliberately false; they are replay primitives, not substitutes
for a source-bound simulation campaign.

The JSON schemas enforce artifact shape and pin every open physical gate to
literal `false`; they do not prove equality between duplicated summary fields
or between a report and its input digest. Trusting code must replay the
persisted input packet through the corresponding evaluator and compare the
content hash. Schema validation of a report by itself is never promotion
evidence.

The public-capacity writer rejects nonfinite/nonstandard JSON and persists both a bundle validated by
`public_record_capacity_bundle.schema.json` and, for the canonical reversible
control, a standalone `public_record_capacity_reference_certificate.json`
validated by the stricter receipt schema. A positive bundle must expose the
named 12-port topology, 30 structured interfaces, three named checkpoint
kernels, the full PASS evaluation and the full strict receipt; schema checks
still cannot replace arithmetic replay. Terminal-fiber evaluation also has
aggregate packet-count and byte budgets; per-packet limits cannot be multiplied
into an unbounded sweep.

## Imported evidence and historical bundles

`data/oph_cross_repo_current/` is regenerated from clean `bec81e2d` / `r1556`.
It includes A5 structure, the conditional exterior witness, P global
uniqueness, the reversible capacity control, SCR330, the exact conditional
issue-320 collar-Poisson witness, and fixed-cutoff regulator gluing. Every
imported row remains `simulation_receipt_eligible=false`; the collar witness is
hash-pinned as theorem provenance but cannot bind a simulation run, and a dirty
source checkout now fails verification.

Historical bundles containing `SM_QUOTIENT_GATE_RECEIPT=true`, an observed
radius as an N source, or `kappa=e` as the selected target are superseded. They
remain immutable evidence records, but consumers must reject or regenerate
them. Existing finite patch counts remain regulator sizes. The legacy numeric
`P_STAR` and `DEFAULT_N_CRC` aliases remain for configuration replay and
comparison displays; new source producers must use a typed P profile and the
exact public-record-capacity lane. The observed-horizon and electroweak N values
are both downstream comparisons, not competing capacity producers.

An internally hash-consistent edge-clock packet can pass
`EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT`, but the physical clock receipt
remains false until raw finite-run evidence is independently resolved and
replayed. The same boundary applies downstream: no CMB lane may promote the
packet-consistency receipt into a finite-lattice clock certificate.

## Remaining empirical campaign

No 256k/1M run was launched in this update because the required evidence packet
changed. The next earned run should freeze code/config/source hashes and emit:

- the full-collar derivative, orientation-half, 24-slot cadence and refinement packet;
- run-derived `UD12` and `RP-A5` without an icosahedral target;
- physical port-current, global-form and matter/family packets;
- a complete source-derived `PUBLIC_CHECKPOINT_PACKET` fiber over a declared D/regulator table;
- a clean SCR330 embedding/mode basis and held-out residual without fitted `n_s`;
- explicit collar-clause evidence and the state-only/expectation negative controls.

Until those exist, the new A5, exterior, capacity and radial results are exact
finite or conditional certificates—not claims that a simulated screen has
produced the physical Standard Model, cosmological N, or a physical CMB sky.
