# Residualized Overlap Graph Geometry Sweep

- sweep receipt: `false`
- case count: `1`
- residual graph receipts: `0`
- spatial-3D candidates: `0`
- strict-H3 candidates: `0`
- rank-3 selectors: `0`
- nontrivial rank-3 selectors: `0`
- spatial-H3 plus independent rank-3 coincidences: `0`
- spatial-H3 plus nontrivial rank-3 coincidences: `0`
- raw rank-1 cases: `1`

## Best Case

- source run: `runs/oph_universe_64k_3p1d_reearned`
- seed / max points / k / remove modes: `7` / `384` / `12` / `1`
- spatial-3D candidate: `false`
- strict-H3 candidate: `false`
- rank-3 selector: `false`
- nontrivial rank-3 selector: `false`
- raw largest-gap rank: `1`
- residual largest-gap rank: `2`
- nontrivial largest-gap rank: `1`
- median dimension: `1.6461713734472294`
- selected model: `H2`
- blockers: `['overlap_residual_graph_degenerate_or_disconnected', 'overlap_residual_graph_independent_rank3_selector_false', 'overlap_residual_graph_not_spatial_3d_candidate', 'overlap_residual_graph_not_strict_h3_candidate']`

## Closest Strict-Gate Rows

- `runs/oph_universe_64k_3p1d_reearned` seed `7` max `384` k `12` remove `1` score `1` dim `1.6461713734472294` model `H2` missing `['graph_receipt', 'spatial_3d_candidate', 'h3_model', 'h3_beats_h2_h4', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`

## Blockers

- `overlap_residual_graph_sweep_no_strict_h3_candidate`
- `overlap_residual_graph_sweep_no_independent_rank3_selector`
- `overlap_residual_graph_sweep_no_spatial_3d_candidate`

## Claim Boundary

Residualized observer-overlap graph parameter sweep. It tests whether target-rank-free common-mode removal exposes a stable 3D/H3 sector, but each row remains diagnostic. It does not promote strict neutral bulk unless independent-rank, H3/leakage, refinement, and frontier gates all pass.
