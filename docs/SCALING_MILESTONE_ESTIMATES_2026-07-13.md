# OPH-FPE scaling and milestone estimates — 2026-07-13

## Purpose and status

This note records empirical lower bounds and planning estimates from the archived OPH-FPE runs. It is an experiment-planning document, not a theorem receipt or a physical claim.

Three quantities must remain distinct:

1. **Smallest observed success**: the smallest archived run whose receipt passed under the contract used by that run.
2. **Smallest meaningful next experiment**: a proposed scale at which the next unresolved question can be tested.
3. **Physical or mathematical minimum**: generally unknown. No tested finite size establishes such a minimum.

Older positive receipts predate the 2026-07-13 gauge-covariant mismatch, coupled quotient replay, feature-ancestry, and producer-hardening audit. They are useful historical observations, but they are not automatically grandfathered as proofs under the hardened contracts.

## Camera, chart, and bulk terminology

- An **H3 chart** is a finite chart construction or response fit with an H3 spatial target.
- **Observer-local modular time** is a first-person record/modular-time ordering for an observer-like self-reading patch.
- An **observer-facing 3+1D camera** combines an H3-valued spatial view with observer time. This is still a local or support-visible construction.
- A **glued objective bulk** requires chart-blind cross-observer correspondence, compatible transition maps, quotient descent and invariance, negative controls, and regulator refinement. An observer camera is not by itself a glued spacetime.
- A **physical particle** requires more than a persistent screen defect. It also requires bulk localization, transport, gauge-covariant common-basepoint fusion, reproducible scattering, and repeated-seed/refinement evidence.
- A **physical CMB prediction** requires more than a screen angular-power proxy. It requires a physical source/transfer pipeline and the relevant bulk and clock gates.

## Empirical lower bounds and planning estimates

| Milestone | Smallest archived success | Smallest meaningful next experiment | Current interpretation |
|---|---:|---:|---|
| H3 chart plumbing | 4,096 cells / 64 analyzed observers | 4k is already sufficient for plumbing and control tests | The hardened 4k run retains `CHART_LORENTZ_H3_RECEIPT=true` and `H3_RESPONSE_CANDIDATE_RECEIPT=true`. This does not establish an emergent neutral bulk. |
| Observer-local modular time | 4,096 / 64 historically | 4k with at least 64–128 observers; use more observers for coverage and stability | The hardened 4k run also has `observer_modular_time_receipt=true`. Basic existence is not currently cell-limited. |
| Observer-facing H3 + time camera | 4,096 / 64 historically | 4k–16k with a clean clock calibration | The hardened 4k run has the H3 and modular-time components, but its aggregate 3+1D receipt is blocked by the BW/KMS branch-replay prerequisite. |
| Glued objective 4D spacetime | No success at any scanned scale | Complete the 4k/16k/64k/256k ladder; use at least 8,192 analyzed observers at the larger sizes and roughly 32k–64k materialized observers | The strict refinement contract explicitly requires the four regulator sizes. A 262,144-cell endpoint is therefore the smallest current proof-oriented programme, not a guaranteed emergence threshold. |
| Physical particle | No success at any scanned scale | Only after a neutral bulk is available; first serious search at or above 262k cells with at least 8,192 analyzed observers and a topology-complete or bias-certified detector | The hardened 4k run contains persistent screen-defect precursors but no transportable, fusion-qualified, scattering-qualified, or particle-like worldline. Scale cannot substitute for the missing producers. |
| Physical CMB | No success at any scanned scale | A 1,048,576-cell / 64,000-observer run after physical source closure, calibrated `k` and `a` evolution, and bulk/clock gates pass | Current `C_l` output is a measurement-facing screen proxy only. The 1M scale is a proposed first physical attempt, not an empirical minimum. |
| Full early-universe model with spacetime and particles | No success | First integrated engineering probe at 1M / 64k after all prerequisite producers pass, followed by multiple seeds and regulator refinement | No defensible minimum can yet be inferred. Larger scales, including 10^8 cells, should not be funded until the 128k/256k feature and bulk ablations pass. |

The planned 131,072-cell / 32,000-materialized-observer run, with 8,192 analyzed observers and a 2,048-observer dense-distance cohort, is a bridge experiment. Its main value is to test whether the hardened geometry and clock diagnostics trend correctly before the 256k refinement endpoint. It cannot alone certify objective bulk, particles, or physical cosmology.

## Evidence behind the estimates

The archived run scan found:

- 17 H3 chart successes, with the smallest at 4,096 cells and 64 observers.
- 17 observer-local modular-time successes, with the smallest at 4,096/64.
- 12 observer-facing 3+1D/H3 successes, with the smallest at 4,096/64.
- No strict-neutral-bulk success.
- No object-bulk-population success in the canonical receipt ladder.
- No positive particle-matter receipt or positive `particle_like_count`.
- No physical CMB prediction receipt.

Representative historical observer-camera success: `runs/k1_population_transfer_4k_dense_20260712`. Representative 64k success: `runs/oph_universe_64k_final_audited_20260711`, which reported an observer-facing 3+1D/H3 receipt for 1,024 analyzed observers. Those reports explicitly describe observer-facing/support-visible experience rather than chart-blind strict neutral bulk.

The hardened 4k science run (`runs/accuracy_audit_4k_final_20260713`) currently shows:

- gauge-covariant mismatch falls from 20,205 to zero;
- the first zero-mismatch cycle is 95;
- all 4,096 records are committed by cycle 106;
- the run ends at cycle 127 after 128 cycles;
- the H3 chart and H3 perturb/resettle response candidate pass;
- observer-local modular time passes;
- the aggregate observer-facing 3+1D receipt is blocked by BW/KMS branch replay;
- finite settling passes, but finite consensus fails because the exact coupled quotient replay finds endpoint-branch nonconfluence;
- the neutral bulk gate remains false;
- 463 persistent screen-defect worldlines are present, but `particle_like_count=0`;
- the screen CMB proxy gate is allowed, while physical CMB and matter-power predictions remain false.

## Repair-cycle and history requirements

Cell and observer counts are not sufficient parameters. Repair duration, repair budget, snapshot cadence, and history length materially affect every downstream receipt.

For the hardened 4k run:

- mismatch first reaches zero at cycle 95;
- all records commit at cycle 106;
- 128 cycles provide only a 22-cycle post-commit tail;
- the total endpoint-repair selections are 21,068;
- the total gauge-sector link mutations are 1,661;
- mismatch never increases between recorded cycle endpoints.

Consequences:

- Fewer than roughly 106 cycles would have truncated the observed settle-and-commit process in this seed.
- A history window of at least 32 remains necessary for the current modular-response and perturb/resettle instruments.
- With `repairs_per_cycle = patch_count / 16` and bounded degree, the settling-cycle count may remain of order 10^2 as the carrier grows, but this is an empirical hypothesis and must be measured.
- Looking only at a final fixed point can erase the transient information needed for clocks, response kernels, defect motion, interactions, and particles.
- Too few perturb/resettle repair steps measure an unrelaxed impulse; too many can wash the response into the same terminal state.

Recommended temporal refinement:

1. Run 64, 128, and 256-cycle variants with otherwise identical configuration.
2. Keep `history_window >= 32`.
3. Report first-zero cycle, first-full-commit cycle, post-commit tail length, response-kernel stability, defect survival, and clock-selection stability.
4. Require the selected geometry/clock conclusions to survive the temporal ladder before interpreting a spatial scale-up.

## Hardening-related reporting recommendation

The current aggregate observer-facing receipt combines two scientifically different claims. Future reports should expose both explicitly:

1. `OBSERVER_KINEMATIC_H3_RECORD_TIME_CAMERA_RECEIPT`: H3 chart + observer-local record/modular time + nondegenerate controlled H3 response.
2. `OBSERVER_LORENTZ_BW_CALIBRATED_3P1D_RECEIPT`: the first receipt plus the finite BW/KMS or endogenous modular-clock calibration.

This split would preserve valid observer-camera evidence when BW calibration fails, without weakening the Lorentz/BW claim or misreporting a local camera as a glued objective spacetime.

## Decision rule for larger runs

Proceed to the next expensive scale only if the preceding run has:

- complete gauge-covariant production and replay receipts;
- a nondegenerate, control-sensitive locality/overlap/perturb-resettle instrument;
- stable conclusions across the temporal ladder;
- no unbounded or silently truncated detector output;
- explicit separation of observer-facing charts, neutral glued bulk, particles, and physical cosmology.

If the 128k and 256k runs do not move the chart-blind neutral geometry toward a stable H3/3D result after ancestry and control issues are removed, increasing to 1M or 10^8 cells is not justified as a bulk-discovery strategy.


## Measured actuals (2026-07-14)

Single 10-core laptop (24 GB), monolithic runs on the covariant-probe
tree:

| Run | Patches / observers | Wall-clock | Artifacts |
|---|---|---|---|
| `oph_universe_64k_night1_20260713` | 65,536 / 1,024 | ~35 min | full viz |
| `oph_universe_128k_obs32k_night1` | 131,072 / 32,000 | ~1.5-2 h | 1.4 GB |
| `oph_universe_1m_night1` | 1,048,576 / 64,000 | ~2.5 h | 1.8 GB |

The observer-report tail dominates above 64k. Receipt state per run sits
in the tracker results log; the 3+1D receipts earned before the
sector-replay fix are superseded per the hardened-contract rule above.
