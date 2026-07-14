# Best available public-data comparisons

Status 2026-07-14: the cross-repo experiment scoreboard, including rows
this suite feeds, lives in `OPH_SIGNATURE_EXPERIMENT_TRACKER.md`
(section 0a). This document stays the contract for the generated
comparison bundle itself.

The comparison suite turns the strongest currently available OPH-FPE measurement surfaces into one provenance-bound scoreboard. It selects a primary run before looking at public metrics, binds every CMB report and sidecar to that exact run directory, and keeps evidence classes separate. It never emits a combined “OPH score.”

The generated bundle contains:

- `best_of_public_data_comparisons.json`: canonical machine-readable report;
- `best_of_public_data_comparisons.md`: reader-facing scoreboard;
- `best_of_public_data_metrics.csv`: one metric per row.

Validate the JSON with `schemas/cosmology/best_of_public_data_comparisons.schema.json`.

## Current result (as of the 2026-07-11 audited primary run)

Update 2026-07-14: earned-receipt runs exist at 64k/128k/1M
(`oph_universe_{64k_3p1d_reearned,128k_3p1d_earned,1m_earned}`); rerun
the suite with one of them as primary to refresh this table. Live
verdicts: `OPH_SIGNATURE_EXPERIMENT_TRACKER.md` section 0a.

The current audited primary run is `runs/oph_universe_64k_final_audited_20260711`. Its strongest available comparisons are mixed:

| Evidence | Result | Interpretation |
|---|---:|---|
| Planck PR3 binned TT | OPH diagonal chi2/bin `1.41749`; LambdaCDM `0.94450`; OPH-minus-LambdaCDM total diagnostic delta chi2 `+39.26` over 83 bins | LambdaCDM is better. One OPH amplitude was fitted on the same bins; this is not the official Planck likelihood. |
| Conditional analytic `n_s=1-P/48` | `n_s=0.9660215`, `0.27 sigma` from Planck's scalar-tilt summary; CAMB diagonal chi2/bin `0.95450` versus `0.94450`, total delta chi2 `+0.83` over 83 bins | Encouraging math-only diagnostic. It is distinct from the simulation-derived `n_s=0.97893`; the source/radial-lift/amplitude/official-likelihood gates remain open. |
| Planck TT profiled residual RMS | `1.19058 sigma` | Recomputed from the amplitude-profiled curve. The older exported `1.27608 sigma` value used the unscaled curve. |
| Cassini Solar-System external field | OPH conditional-Z6 branch `Q2 = 3.62018e-26 s^-2` versus Cassini `(1.6 +/- 1.8)e-27 s^-2`; raw fixed-input pull `19.22 sigma`; about `10.7 sigma` after a linearized Gaia-external-field uncertainty only | Strongly excludes the natural universal/full-source QUMOND extension. It does not by itself falsify recovered OPH core because the paper scopes the static law to settled galaxies; instead it exposes a missing quantitative Solar-System applicability/screening/transport gate. |
| SPARC RAR calibration | `0.13281 dex` over 2,693 aggregate RAR points | A good same-data calibration, not a prediction. Only `a0/lambda_collar^2` is identifiable. |
| Fixed OPH RAR branches | Z6 aggregate RMS `0.13283 dex`; unit branch `0.13421 dex`; same-data effective-scale optimum `0.13295 dex` | Positive retrospective formula check. The coefficient is benchmark-derived/conditional and shared galaxy nuisances are not profiled, so this is not a blind prediction or a formal significance. |
| SPARC galaxy holdout | acceleration RMSE `0.17239 dex`; velocity RMSE `22.69 km/s` versus `59.18 km/s` baryon-only | Positive galaxy-level holdout, but the velocity diagonal chi2 proxy remains large at `33.89` per point. |
| SPARC BTFR check | error-aware observed slope `3.84565 +/- 0.08582`; OPH slope `4.0` is `1.80 sigma` high; at 100 km/s the Z6 fixed normalization is `0.13476 dex` high (`6.47 sigma` statistical-only) | Mixed. The slope is compatible and within the published systematic range; normalization is under pressure for the table's fixed mass-to-light convention, but global galaxy systematics are not marginalized. This table overlaps the SPARC sample and is not a wholly independent galaxy sample. |
| Compressed cosmology reference | archived OPH `11.4633`; fixed-H0 grid `9.5504`; free compressed point `6.2852` | Invalidated and blocked: hard-coded constants, no attached public covariance, correlated compressed variables, and a rejected neutrino input. It is excluded from evidence. |

There are currently zero frozen physical-prediction receipts. The Planck promotion ledger stops at `SPECTRUM_DIAGNOSTIC`; the first blocked parent gate is the finite quotient source law.

The Cassini row is deliberately not promoted to a frozen OPH prediction. Its OPH `a0` is a calibrated benchmark, the exact `lambda=exp(-P/24)` coefficient is conditional on an unclosed uniform-product-thickening gate, and no current source law says that the settled-galaxy PDE applies to the Milky-Way-plus-Sun source. The `19.22 sigma` value is therefore labeled a raw central-input residual, not a nuisance-marginalized exclusion significance. Even so, the unit-lambda endpoint is still about `18.01 sigma` raw, so the current Jensen lambda band does not remove the problem under universal applicability.

The dark-sector benchmark also has an unresolved capacity provenance fork. The existing calculator uses the observed rounded `N_scr=3.31e122`; using the exact source-side electroweak capacity `N_EW=3.5323546e122` changes `a0` to `9.96267e-11 m/s^2` and the Z6 effective scale to `1.14131e-10 m/s^2`. This does not rescue Cassini (about `18.94 sigma` raw) and slightly worsens the BTFR normalization. Any capacity-derived acceleration comparison must freeze and disclose one branch.

## Why other attractive signatures are not in the scoreboard yet

- Emergent spacetime and gravity-as-repair currently emit finite geometry, overlap, BW/KMS, reconstruction, and Einstein-branch readiness receipts. They do not yet emit a frozen detector-space observable with a physical scale map, continuum error, and public likelihood. Those results are theorem or simulator tests, not observational wins.
- The Boltzmann transport surface still lacks a promoted source-derived `B_A(k,a)` kernel. The suite records paired-response and kernel readiness separately instead of treating a CAMB run with conventional background inputs as an OPH prediction.
- The black-hole workspace names excellent future targets—EHT shadows, LVK ringdown/QNMs, Hawking channels, and Page-style records—but its current CBH8 reports explicitly keep the physical QNM, Hawking, and continuum bridges false. Raw repair eigenvalues must not be compared directly with observed ringdown frequencies.

These omissions are useful: they identify exactly which mathematical proof or bridge must be closed before another public-data row can be added without relabeling an internal finite-model receipt as a physical signature.

Generate the current suite with:

```bash
cd /Users/muellerberndt/Projects/oph-meta/oph-physics-sim

python3 tools/best_of_public_data.py \
  --primary-run runs/oph_universe_64k_final_audited_20260711 \
  --baseline-run runs/oph_universe_64k_final_audited_20260711 \
  --out runs/best_of_public_data/current
```

Use `--strict` in automation to fail if a primary/baseline report, public table, manifest, config, or repair-readiness sidecar is missing or malformed. Use `--require-frozen-prediction` only at a release gate that genuinely requires a likelihood-evaluated physical prediction.

## Prepare the large run and preflight its scale contract

The current bounded monolithic preparer makes `1,048,576` carrier patches and `64,000` materialized observer-like self-reading systems. The expensive cross-observer analyses use a deterministic `8,192`-observer subset. These observer-like systems have bounded local state, ports or boundaries, readback, records, and feedback/repair moves; they are not merely generic graph samples. This is not a one-million-materialized-observer run.

Prepare an ignored, machine-local config:

```bash
python3 tools/prepare_million_patch_config.py \
  --base configs/e4_shared_observer_bulk_256k_observers4096_theorem.yml \
  --out configs/local/e5_shared_observer_bulk_1m_bounded_visualizer.yml \
  --seed 20260753
```

Record the planned scale beside the current evidence before spending compute:

```bash
python3 tools/best_of_public_data.py \
  --primary-run runs/oph_universe_64k_final_audited_20260711 \
  --baseline-run runs/oph_universe_64k_final_audited_20260711 \
  --planned-config configs/local/e5_shared_observer_bulk_1m_bounded_visualizer.yml \
  --out runs/best_of_public_data/preflight_1m_patch
```

The preflight deliberately records carrier-patch count and materialized-observer count as different fields. It also warns that the bounded raw screen export has `ell_max=24`, below the first bundled Planck TT bin at approximately `ell=47.7`. A larger patch count therefore cannot automatically improve or promote the Planck lane.

## Run protocol

Freeze the exact worktree, config, seed, primary-run designation, and baseline before evaluating public metrics. The current development worktree is dirty, so a remote run from a clean checkout of the old commit will not reproduce it unless the relevant changes are committed or the exact worktree is transferred with its provenance hash.

The prepared monolithic command for a 64-vCPU, 256-GiB machine is:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
OPH_FPE_CPUS=64 \
python3 -m oph_fpe.cli run-oph-universe \
  --config configs/local/e5_shared_observer_bulk_1m_bounded_visualizer.yml \
  --out-dir runs \
  --run-id oph_universe_1m_patch_seed20260753 \
  --inner-jobs 32 \
  --skip-visualizations
```

`base_progress.json` is a progress/ETA record, not a mid-dynamics checkpoint. Once `manifest.json` exists, failed post-processing can be resumed without repeating the base dynamics:

```bash
python3 -m oph_fpe.cli run-oph-universe \
  --config configs/local/e5_shared_observer_bulk_1m_bounded_visualizer.yml \
  --out-dir runs \
  --source-run-dir runs/oph_universe_1m_patch_seed20260753 \
  --skip-base-run \
  --skip-visualizations
```

For remote execution, check for existing instances before creating a cost-incurring machine, copy back the complete run directory and logs, then stop or delete the machine after retrieval. One seed is a scale point, not an uncertainty estimate; predeclare a second independent seed after peak memory and runtime are known.

## Compare the returned run

Keep the 64k run as the declared baseline and the large run as the new primary:

```bash
python3 tools/best_of_public_data.py \
  --primary-run runs/oph_universe_1m_patch_seed20260753 \
  --history-run runs/oph_universe_64k_final_audited_20260711 \
  --baseline-run runs/oph_universe_64k_final_audited_20260711 \
  --planned-config configs/local/e5_shared_observer_bulk_1m_bounded_visualizer.yml \
  --out runs/best_of_public_data/primary_1m_patch_vs_64k \
  --strict
```

The report will show:

- the primary and baseline Planck diagnostics, their artifact hashes, patch ratio, and delta chi2;
- the run-derived finite-repair-clock and selector inputs used by CAMB;
- paired-response and `B_A` kernel readiness changes without treating them as public-data fits;
- the unchanged SPARC continuation and compressed reference as run-independent lanes;
- a false prediction receipt unless the same run bundle passes the frozen source, solver, likelihood, physical-input, and promotion ledger gates.

The suite may order explicitly supplied Planck diagnostics by their diagonal statistic, but a better historical score never replaces the predeclared primary run. A larger run also does not “win” because it has more patches.

## Optional screen post-processing

The bounded run saves `freezeout_fields.npz`, so a predeclared higher-multipole screen diagnostic can be computed later without repeating the dynamics:

```bash
python3 -m oph_fpe.cli cl-from-freezeout-npz \
  --run-dir runs/oph_universe_1m_patch_seed20260753 \
  --out runs/oph_universe_1m_patch_seed20260753/cl_post_ell96 \
  --ell-max 96 \
  --harmonic-batch-size 8192 \
  --n-jobs 32 \
  --fields record_signature,cumulative_repair_load \
  --benchmark data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt
```

This remains a screen-proxy diagnostic. It does not create a physical multipole map, a physical scale bridge, or an official CMB likelihood.

## What the large run can and cannot improve

The large run can materially test cross-scale convergence of the finite repair clock, transition matrices, `B_A` response controls, H3 reconstruction, defect persistence, and theorem/readiness receipts. These are the useful scale-sensitive signatures.

It cannot improve the present SPARC result by itself because the SPARC continuation refits its effective acceleration scale directly from public galaxy data. It also cannot change the compressed cosmology row, which is hard-coded as a regression reference. Those lanes become simulation predictions only if a future source-side run freezes the relevant parameters before the public tables are opened.
