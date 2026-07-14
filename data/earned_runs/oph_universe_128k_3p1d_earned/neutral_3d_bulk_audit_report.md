# Neutral 3D Bulk Audit

- strict neutral bulk ready: `false`
- control-residualized rank-3 refinement candidate: `false`
- independent rank-3 selector all: `false`
- directional strict-ready total: `0`
- control-quotient candidate count: `1`
- overlap-native negative-control receipts: `0` / `1`
- overlap-native spatial-3D candidates: `0`

## Blockers

- `required_4k_16k_64k_256k_refinement_ladder_incomplete:4096,16384,65536,262144`
- `control_quotient_rank3_candidate_not_stable_across_refinement`
- `independent_svd_rank3_selector_not_stable_or_false`
- `control_quotient_lane_is_not_a_negative_control`
- `directional_h3_strict_rank_gate_not_passed`
- `measured_cross_observer_overlap_refinement_gate_not_passed`
- `no_directional_rank_passes_strict_h3_model_and_leakage_gates`
- `requires_refinement_stability_across_regulator_sizes`
- `requires_independent_rank_selection_rule_before_physical_interpretation`
- `strict_neutral_bulk_refinement_receipt_false`
- `overlap_native_negative_control_receipt_false`

## Claim Boundary

Audit for strict neutral 3D bulk promotion. Control-residualized rank-3 rows are treated as diagnostic candidates only. Strict neutral bulk remains false until independent rank selection, proper negative/null controls, directional H3 gates, and refinement all pass.
