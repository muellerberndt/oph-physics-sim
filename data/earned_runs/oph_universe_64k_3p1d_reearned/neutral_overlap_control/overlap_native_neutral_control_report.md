# Overlap-Native Neutral Control

- overlap-native negative-control receipt: `false`
- sampled observers: `384` / `1024`
- overlap spatial-3D candidate: `false`
- overlap strict-H3 candidate: `false`

## Controls

- `degree_preserving_overlap_graph_rewire`: expected failure `true`, distance corr `-0.002223360301483388`, delta `0.003273084184595509`
- `overlap_edge_weight_permutation`: expected failure `false`, distance corr `0.6824607728360552`, delta `0.001263962805712252`
- `columnwise_histogram_null`: expected failure `true`, distance corr `0.0010266116297741811`, delta `0.0067618028246162285`

## Blockers

- `overlap_native_negative_controls_did_not_all_fail`
- `overlap_native_distance_not_spatial_3d_candidate`
- `overlap_native_distance_not_strict_h3_candidate`

## Rank Obstruction

- unavailable

## Claim Boundary

This is the observer-overlap substrate audit. It can certify that the neutral overlap distance is nondegenerate and control-sensitive. Strict neutral bulk still additionally requires rank selection, H3/dimension/leakage gates, refinement across regulator sizes, and promotion by the neutral 3D audit.
