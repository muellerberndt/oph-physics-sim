# Overlap-Native Graph Geometry Sweep

- sweep receipt: `false`
- case count: `36`
- graph receipts: `0`
- spatial-3D candidates: `0`
- strict-H3 candidates: `0`
- rank-3 selectors: `0`
- nontrivial rank-3 selectors: `0`
- spatial-H3 plus independent rank-3 coincidences: `0`
- spatial-H3 plus nontrivial rank-3 coincidences: `0`

## Best Case

- source run: `runs/oph_universe_128k_3p1d_earned`
- seed / max points / k: `17` / `512` / `12`
- spatial-3D candidate: `false`
- strict-H3 candidate: `false`
- rank-3 selector: `false`
- nontrivial rank-3 selector: `false`
- nontrivial largest-gap rank: `1`
- median dimension: `1.8231561231826507`
- selected model: `H3`
- blockers: `['overlap_graph_degenerate_or_disconnected', 'overlap_graph_negative_controls_did_not_all_fail', 'overlap_graph_independent_rank3_selector_false', 'overlap_graph_not_spatial_3d_candidate', 'overlap_graph_not_strict_h3_candidate']`

## Closest Strict-Gate Rows

- `runs/oph_universe_128k_3p1d_earned` seed `17` max `512` k `12` score `3` dim `1.8231561231826507` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `17` max `512` k `16` score `3` dim `1.8231561231826507` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `17` max `512` k `24` score `3` dim `1.8231561231826507` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `17` max `512` k `8` score `3` dim `1.823056662972872` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `16` score `3` dim `1.4955336956320018` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `24` score `3` dim `1.495520243551566` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `12` score `3` dim `1.494350292240847` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`
- `runs/oph_universe_128k_3p1d_earned` seed `7` max `1024` k `8` score `3` dim `1.4791276373106148` model `H3` missing `['graph_receipt', 'spatial_3d_candidate', 's2_leakage_pass', 'independent_rank3_selector', 'strict_h3_candidate']`

## Blockers

- `overlap_graph_sweep_no_strict_h3_candidate`
- `overlap_graph_sweep_no_independent_rank3_selector`
- `overlap_graph_sweep_no_spatial_3d_candidate`

## Claim Boundary

Observer-overlap graph parameter sweep. It broadens the search for strict-H3/rank-3 neutral geometry, but each row remains diagnostic. It does not promote strict neutral bulk unless the independent-rank, H3/leakage, refinement, and frontier gates all pass.
