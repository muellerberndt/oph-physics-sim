# Overlap-Native Neutral Control

- overlap-native negative-control receipt: `false`
- sampled observers: `1024` / `8192`
- overlap spatial-3D candidate: `false`
- overlap strict-H3 candidate: `false`

## Controls

- `degree_preserving_overlap_graph_rewire`: expected failure `true`, distance corr `0.001711927900612254`, delta `0.0032629883144445772`
- `overlap_edge_weight_permutation`: expected failure `false`, distance corr `0.7393249508000087`, delta `0.0011554049973387943`
- `columnwise_histogram_null`: expected failure `true`, distance corr `0.002131283854672306`, delta `0.005447229721117392`

## Blockers

- `overlap_native_negative_controls_did_not_all_fail`
- `overlap_native_distance_not_spatial_3d_candidate`
- `overlap_native_distance_not_strict_h3_candidate`

## Rank Obstruction

- unavailable

## Claim Boundary

This is the observer-overlap substrate audit. It can certify that the neutral overlap distance is nondegenerate and control-sensitive. Strict neutral bulk still additionally requires rank selection, H3/dimension/leakage gates, refinement across regulator sizes, and promotion by the neutral 3D audit.
