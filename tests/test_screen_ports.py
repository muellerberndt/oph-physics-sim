from __future__ import annotations

import numpy as np

from oph_fpe.core.icosahedral import (
    geodesic_icosahedral_patch_arrays,
    icosahedral_a5_port_permutations,
)
from oph_fpe.core.screen_ports import (
    ReferenceDiagonalA5Intertwiner,
    assign_echosahedral_ports,
    canonicalize_echosahedral_patch_state,
    echosahedral_patch_architecture_report,
    echosahedral_patch_record_signature,
    echosahedral_patch_state_report,
    echosahedral_port_names,
    initialize_echosahedral_patch_state,
    reference_diagonal_a5_intertwiner,
    reference_diagonal_a5_intertwiner_report,
    sync_routed_echosahedral_patch_state,
)
from oph_fpe.gauge.covariant_overlap import (
    group_multiply_indices,
    transform_local_frames,
)


def test_echosahedral_port_names_are_explicit():
    assert echosahedral_port_names(12) == [f"P{index}" for index in range(12)]


def test_assign_echosahedral_ports_reports_overflow():
    left = np.array([0, 0, 0, 0])
    right = np.array([1, 2, 3, 4])
    port_map = assign_echosahedral_ports(left, right, patch_count=5, ports_per_patch=3)
    report = port_map.as_jsonable(sample_edges=2)

    assert port_map.left_port.tolist() == [0, 1, 2, 0]
    assert port_map.overflow_count == 1
    assert report["port_names"] == ["P0", "P1", "P2"]
    assert report["sample_edge_ports"][0]["left_port"] == "P0"


def test_assign_echosahedral_ports_matches_sequential_endpoint_order():
    left = np.array([0, 0, 1, 2, 0, 2, 1], dtype=np.int64)
    right = np.array([1, 2, 2, 3, 3, 0, 3], dtype=np.int64)
    ports = 2
    port_map = assign_echosahedral_ports(left, right, patch_count=4, ports_per_patch=ports)

    counters = np.zeros(4, dtype=np.int64)
    expected_left: list[int] = []
    expected_right: list[int] = []
    overflow = 0
    for a, b in zip(left, right, strict=False):
        left_count = counters[int(a)]
        right_count = counters[int(b)]
        overflow += int(left_count >= ports) + int(right_count >= ports)
        expected_left.append(int(left_count % ports))
        expected_right.append(int(right_count % ports))
        counters[int(a)] += 1
        counters[int(b)] += 1

    assert port_map.left_port.tolist() == expected_left
    assert port_map.right_port.tolist() == expected_right
    assert port_map.overflow_count == overflow
    assert port_map.node_degree.tolist() == [4, 3, 4, 3]


def test_every_cell_uses_one_exact_twelve_port_echosahedral_template():
    report = echosahedral_patch_architecture_report()

    assert report["ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT"] is True
    assert report["port_names"] == [f"P{index}" for index in range(12)]
    assert len(report["port_direction_template"]) == 12
    assert len(report["local_edges"]) == 30
    assert len(report["local_faces"]) == 20
    assert len(report["antipodal_pairs"]) == 6
    assert report["proper_rotation_group_order"] == 60
    assert report["shared_template_instantiated_per_cell"] is True
    assert report["ECHOSAHEDRAL_CARRIER_CONFORMANCE_RECEIPT"] is True
    assert report["LOCAL_ICOSAHEDRAL_INCIDENCE_RECEIPT"] is True
    incidence = report["incidence_audit"]
    assert incidence["vertex_degree_profile"] == {"5": 12}
    assert incidence["adjacency_irrep_multiplicities"] == [1, 3, 3, 5]
    assert incidence["all_a5_actions_preserve_edges"] is True
    assert incidence["all_a5_actions_preserve_oriented_faces"] is True
    assert incidence["all_a5_actions_commute_with_antipode"] is True
    assert report["hidden_coordinates_eligible_for_promoted_geometry"] is False
    assert report["GLOBAL_TWELVE_UNIT_SCREEN_SIEVE_RECEIPT"] is False
    assert report["LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT"] is False


def test_icosahedral_cell_federation_routes_edges_to_distinct_local_ports():
    points, left, right = geodesic_icosahedral_patch_arrays(1, patch_basis="cells")
    port_map = assign_echosahedral_ports(
        left,
        right,
        patch_count=points.shape[0],
        ports_per_patch=12,
        points=points,
    )
    report = port_map.as_jsonable()

    assert port_map.routing_mode == "icosahedral_directional_assignment"
    assert port_map.overflow_count == 0
    assert set(port_map.node_degree.tolist()) == {3}
    for node in range(points.shape[0]):
        ports = [
            int(port_map.left_port[index])
            for index in np.flatnonzero(left == node)
        ] + [
            int(port_map.right_port[index])
            for index in np.flatnonzero(right == node)
        ]
        assert len(ports) == 3
        assert len(set(ports)) == 3
    assert report["ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT"] is True
    assert report["GEOMETRIC_LOCAL_PORT_ROUTING_RECEIPT"] is True
    assert report["REFERENCE_SINGLETON_FEDERATION_SEWING_RECEIPT"] is True
    assert report["FEDERATION_SEWING_RECEIPT"] is False
    assert report["unused_port_slots"] == points.shape[0] * 9
    reference = report["reference_diagonal_a5_intertwiner"]
    assert reference["REFERENCE_DIAGONAL_A5_INTERTWINER_RECEIPT"] is True
    assert reference["LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT"] is False
    assert report["LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT"] is False
    assert report["PHYSICAL_GLOBAL_SCREEN_SIEVE_RECEIPT"] is False


def test_reference_diagonal_maps_intertwine_all_sixty_a5_actions():
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    intertwiner = reference_diagonal_a5_intertwiner(
        4,
        cell_weights=weights,
    )
    rng = np.random.default_rng(20260720)
    local_values = rng.normal(size=(4, 12, 3))
    global_values = rng.normal(size=(12, 3))

    assert isinstance(intertwiner, ReferenceDiagonalA5Intertwiner)
    assert np.allclose(intertwiner.project(intertwiner.embed(global_values)), global_values)
    expected = intertwiner.conditional_expectation(local_values)
    assert np.allclose(intertwiner.conditional_expectation(expected), expected)
    for element_index in range(60):
        assert np.allclose(
            intertwiner.project(
                intertwiner.act_local(local_values, element_index)
            ),
            intertwiner.act_global(
                intertwiner.project(local_values), element_index
            ),
        )
        assert np.allclose(
            intertwiner.act_local(
                intertwiner.embed(global_values), element_index
            ),
            intertwiner.embed(
                intertwiner.act_global(global_values, element_index)
            ),
        )
        assert np.allclose(
            intertwiner.conditional_expectation(
                intertwiner.act_local(local_values, element_index)
            ),
            intertwiner.act_local(expected, element_index),
        )


def test_reference_intertwiner_hashes_contract_but_never_promotes_physics():
    report = reference_diagonal_a5_intertwiner_report(
        4,
        cell_weights=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )

    assert report["REFERENCE_DIAGONAL_A5_INTERTWINER_RECEIPT"] is True
    assert report["REFERENCE_LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT"] is True
    assert report["LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT"] is False
    assert report["A5_EQUIVARIANT_CURRENT_CARRIER_INTERTWINER_RECEIPT"] is False
    assert report["GLOBAL_TWELVE_UNIT_SCREEN_SIEVE_RECEIPT"] is False
    assert report["group_element_count"] == 60
    assert report["unique_action_count"] == 60
    assert report["base_port_orbit_size"] == 12
    assert report["base_port_stabilizer_size"] == 5
    assert report["element_order_profile"] == {"1": 1, "2": 15, "3": 20, "5": 24}
    assert len(report["action_rows"]) == 60
    assert all(
        row["embedding_intertwining_exact"]
        and row["weighted_projection_intertwining_exact"]
        for row in report["action_rows"]
    )
    assert report["contract_sha256"] == (
        "sha256:7faa672b1a257bb2db313c227a714bcde50f96f4ddabf02d9f048fa60d43c13b"
    )
    assert report["source_derived_common_trivialization"] is False
    assert report["physical_receipt_eligible"] is False
    assert "common_local_port_frame_is_declared_not_source_derived" in report[
        "physical_promotion_blockers"
    ]


def test_reference_intertwiner_audit_rejects_mutated_action_table():
    valid = reference_diagonal_a5_intertwiner(3)
    mutated_permutations = list(valid.port_permutations)
    mutated_permutations[-1] = mutated_permutations[0]
    mutated = ReferenceDiagonalA5Intertwiner(
        cell_weights=valid.cell_weights,
        port_permutations=tuple(mutated_permutations),
    )

    report = mutated.audit()
    assert report["unique_action_count"] == 59
    assert report["REFERENCE_DIAGONAL_A5_INTERTWINER_RECEIPT"] is False
    assert report["LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT"] is False


def test_reference_intertwiner_fails_closed_without_canonical_federation():
    empty = reference_diagonal_a5_intertwiner_report(0)
    wrong_ports = reference_diagonal_a5_intertwiner_report(5, ports_per_patch=8)

    assert empty["REFERENCE_DIAGONAL_A5_INTERTWINER_RECEIPT"] is False
    assert "positive_patch_federation_missing" in empty["blockers"]
    assert wrong_ports["REFERENCE_DIAGONAL_A5_INTERTWINER_RECEIPT"] is False
    assert "canonical_twelve_port_local_template_missing" in wrong_ports["blockers"]


def test_reference_intertwiner_requires_faithful_normalized_weights():
    with np.testing.assert_raises_regex(ValueError, "strictly positive"):
        reference_diagonal_a5_intertwiner(
            3,
            cell_weights=np.asarray([0.5, 0.5, 0.0]),
        )
    with np.testing.assert_raises_regex(ValueError, "normalized to one"):
        reference_diagonal_a5_intertwiner(
            3,
            cell_weights=np.asarray([0.2, 0.2, 0.2]),
        )


def test_full_twelve_port_federation_state_is_synchronized_and_record_bound():
    points, left, right = geodesic_icosahedral_patch_arrays(0, patch_basis="cells")
    port_map = assign_echosahedral_ports(
        left,
        right,
        patch_count=points.shape[0],
        ports_per_patch=12,
        points=points,
    )
    rng = np.random.default_rng(73)
    routed_left = rng.integers(0, 6, size=left.size, dtype=np.int16)
    routed_right = rng.integers(0, 6, size=right.size, dtype=np.int16)
    state = initialize_echosahedral_patch_state(
        patch_count=points.shape[0],
        ports_per_patch=12,
        group_order=6,
        edge_left=left,
        edge_right=right,
        port_map=port_map,
        routed_left_state=routed_left,
        routed_right_state=routed_right,
        rng=rng,
    )

    routed_left[0] = np.int16((int(routed_left[0]) + 1) % 6)
    routed_right[-1] = np.int16((int(routed_right[-1]) + 1) % 6)
    sync_routed_echosahedral_patch_state(
        state,
        edge_left=left,
        edge_right=right,
        port_map=port_map,
        routed_left_state=routed_left,
        routed_right_state=routed_right,
    )
    report = echosahedral_patch_state_report(
        state,
        edge_left=left,
        edge_right=right,
        port_map=port_map,
        routed_left_state=routed_left,
        routed_right_state=routed_right,
        record_signature_bound=True,
    )

    assert state.shape == (20, 12)
    assert report["ECHOSAHEDRAL_PATCH_STATE_INSTANTIATION_RECEIPT"] is True
    assert report["PATCH_ALL_PORT_READBACK_RECEIPT"] is True
    assert report["RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT"] is True
    assert report["PHYSICAL_A5_PORT_EMERGENCE_RECEIPT"] is False

    routed_signature = np.arange(points.shape[0], dtype=np.int64)
    baseline = echosahedral_patch_record_signature(routed_signature, state)
    for port_index in range(12):
        perturbed = state.copy()
        perturbed[0, port_index] = np.int16((int(perturbed[0, port_index]) + 1) % 6)
        changed = echosahedral_patch_record_signature(routed_signature, perturbed)
        assert int(changed[0]) != int(baseline[0])
        assert np.array_equal(changed[1:], baseline[1:])


def test_full_patch_record_input_is_invariant_under_all_local_a5_reorientations():
    rng = np.random.default_rng(20260721)
    state = rng.integers(0, 6, size=(17, 12), dtype=np.int16)
    routed_signature = rng.integers(0, 2**31, size=17, dtype=np.int64)
    baseline = echosahedral_patch_record_signature(routed_signature, state)

    for permutation in icosahedral_a5_port_permutations():
        transformed = np.empty_like(state)
        transformed[:, np.asarray(permutation, dtype=np.int64)] = state
        observed = echosahedral_patch_record_signature(
            routed_signature,
            transformed,
        )
        assert np.array_equal(observed, baseline)


def test_full_patch_record_input_is_invariant_under_local_s3_frame_changes():
    points, left, right = geodesic_icosahedral_patch_arrays(0, patch_basis="cells")
    port_map = assign_echosahedral_ports(
        left,
        right,
        patch_count=points.shape[0],
        ports_per_patch=12,
        points=points,
    )
    rng = np.random.default_rng(79)
    routed_left = rng.integers(0, 6, size=left.size, dtype=np.int16)
    routed_right = rng.integers(0, 6, size=right.size, dtype=np.int16)
    gauge = rng.integers(0, 6, size=left.size, dtype=np.int16)
    state = initialize_echosahedral_patch_state(
        patch_count=points.shape[0],
        ports_per_patch=12,
        group_order=6,
        edge_left=left,
        edge_right=right,
        port_map=port_map,
        routed_left_state=routed_left,
        routed_right_state=routed_right,
        rng=rng,
    )
    canonical = canonicalize_echosahedral_patch_state(
        state,
        edge_left=left,
        edge_right=right,
        routed_left_state=routed_left,
        routed_right_state=routed_right,
        group_name="S3",
        group_order=6,
    )

    frames = rng.integers(0, 6, size=points.shape[0], dtype=np.int16)
    transformed_left, transformed_right, _ = transform_local_frames(
        routed_left,
        routed_right,
        gauge,
        left,
        right,
        frames,
        group_name="S3",
        group_order=6,
    )
    transformed_state = group_multiply_indices(
        frames[:, None],
        state,
        group_name="S3",
        group_order=6,
    )
    transformed_canonical = canonicalize_echosahedral_patch_state(
        transformed_state,
        edge_left=left,
        edge_right=right,
        routed_left_state=transformed_left,
        routed_right_state=transformed_right,
        group_name="S3",
        group_order=6,
    )

    assert np.array_equal(canonical, transformed_canonical)
