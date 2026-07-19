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
| Exterior SM witness | `Lambda^2(C+W)+Lambda^4(C+W)` recomputes one 15-state generation, exact field dimensions/hypercharges, three conditional Higgs invariant lines, anomaly cancellation, four weak doublets and even Witten parity | exterior/Higgs selection, three physical families, weak/load intertwiners, no-extra-sector and QFT completion |
| Legacy SM sieve | supplied final labels produce only `SM_TARGET_CONFORMANCE_DIAGNOSTIC` | `SM_QUOTIENT_GATE_RECEIPT` and physical SM promotion are false; default target injection is disabled in tracked configs |
| Public-record capacity | exact atom sections, endogenous reachability, frozen publicness, complete joint kernels, compound confusability graph, maximum independent set, carrier bound, whole-fiber scalarization and slack/refinement controls | source-derived complete physical trial universe/fiber, robust common readback, unique regulator-stable slack zero, horizon-record saturation and physical N |
| Reversible capacity packet | target-free 12-port/30-interface permutation control proves `M_0=|X_reach|` | schema/control only, not a physical universe packet |
| Horizon/EW/operational N | typed downstream comparisons remain available | measured radius, Lambda, EW bridge, operational resolution and supplied D/N labels cannot define the producer |
| P maps | source-map and gauge-width roots are distinct typed profiles; global map uniqueness is hash-pinned and nonpromoting | measured Thomson and empirical hadron profiles are comparison-only; same-scheme source transport remains open |
| Edge-center tilt/clock | target is `rho_full=P/24`, orientation half `theta=P/48`, `n_s=1-theta`, `kappa=theta/(P-phi)` | a transition eigenvalue or finite survival exponent is not the derivative; clock binding, source DAG, cadence, refinement, and a matching generative-P profile remain required; `e` is diagnostic only |
| SCR330 radial lift | exact Mellin and amplitude identities, finite-window bound, radial SVD/null report, prior continuation diagnostic, dilation consequence, held-out forward residual and ten canonical receipts | E4 needs a clean physical embedding plus one allowed uniqueness branch; prior continuation never promotes; TT/TE/EE and likelihood are E5 |
| Collar clause | an explicit retained-family/factor-through-flux packet is independently checked | Gibbs, density, relative entropy, CMI, or an expectation that deletes the cross-cut term cannot force the clause |
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
  --claim-tier E4 --claimed-pass \
  --out runs/source/scr330_radial_receipt.json

python3 -m oph_fpe.cli edge-center-clock-certificate \
  --evidence runs/source/edge_center_clock_evidence.json \
  --out runs/source/edge_center_clock_certificate.json

python3 -m oph_fpe.cli collar-clause-certificate \
  --packet runs/source/collar_clause_packet.json \
  --out runs/source/collar_clause_certificate.json
```

The radial writer does not trust `--claimed-pass`: it recomputes ancestry,
tier, branch and payload gates. Missing evidence produces a false receipt with
blockers.

## Imported evidence and historical bundles

`data/oph_cross_repo_current/` is regenerated from clean `bec81e2d` / `r1556`.
It includes A5 structure, the conditional exterior witness, P global
uniqueness, the reversible capacity control, SCR330, and fixed-cutoff regulator
gluing. Every imported row remains `simulation_receipt_eligible=false`; a dirty
source checkout now fails verification.

Historical bundles containing `SM_QUOTIENT_GATE_RECEIPT=true`, an observed
radius as an N source, or `kappa=e` as the selected target are superseded. They
remain immutable evidence records, but consumers must reject or regenerate
them. Existing finite patch counts remain regulator sizes. The legacy numeric
`P_STAR` and `DEFAULT_N_CRC` aliases remain for configuration replay and
comparison displays; new source producers must use a typed P profile and the
exact public-record-capacity lane. The observed-horizon and electroweak N values
are both downstream comparisons, not competing capacity producers.

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
