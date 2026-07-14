# Residualized Overlap Graph Geometry Sweep

- sweep receipt: `false`
- case count: `144`
- residual graph receipts: `38`
- spatial-3D candidates: `0`
- strict-H3 candidates: `0`
- rank-3 selectors: `0`
- nontrivial rank-3 selectors: `0`
- spatial-H3 plus independent rank-3 coincidences: `0`
- spatial-H3 plus nontrivial rank-3 coincidences: `0`
- raw rank-1 cases: `60`

## Best Case

- source run: `runs/oph_universe_128k_3p1d_earned`
- seed / max points / k / remove modes: `7` / `768` / `8` / `3`
- spatial-3D candidate: `false`
- strict-H3 candidate: `false`
- rank-3 selector: `false`
- nontrivial rank-3 selector: `false`
- raw largest-gap rank: `1`
- residual largest-gap rank: `2`
- nontrivial largest-gap rank: `1`
- median dimension: `2.639523659146536`
- selected model: `H3`
- blockers: `['overlap_residual_graph_independent_rank3_selector_false', 'overlap_residual_graph_not_spatial_3d_candidate', 'overlap_residual_graph_not_strict_h3_candidate']`

## Closest Strict-Gate Rows

- `runs/oph_universe_128k_3p1d_earned` seed `11` max `512` k `8` remove `2` score `4` dim `3.079238081863876` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `768` k `8` remove `3` score `4` dim `2.639523659146536` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `768` k `8` remove `2` score `4` dim `2.576028858381486` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `768` k `8` remove `4` score `4` dim `2.4411014827901028` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `12` remove `4` score `4` dim `2.413896505382913` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `11` max `1024` k `8` remove `4` score `4` dim `2.2052126827061738` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `8` remove `4` score `4` dim `2.180117900547125` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `8` remove `3` score `4` dim `2.050451525952449` model `H3` missing `['spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`

## Blockers

- `overlap_residual_graph_sweep_no_strict_h3_candidate`
- `overlap_residual_graph_sweep_no_independent_rank3_selector`
- `overlap_residual_graph_sweep_no_spatial_3d_candidate`

## Claim Boundary

Residualized observer-overlap graph parameter sweep. It tests whether target-rank-free common-mode removal exposes a stable 3D/H3 sector, but each row remains diagnostic. It does not promote strict neutral bulk unless independent-rank, H3/leakage, refinement, and frontier gates all pass.
