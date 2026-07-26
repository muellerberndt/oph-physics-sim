from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import networkx as nx
import numpy as np

from oph_fpe.core.echosahedral_federation import (
    CARRIER_FORBIDDEN_PROMOTION_FIELDS,
    EchosahedralFederation,
    ExternalBoundaryBundle,
    InterfaceAlgebraBinding,
    ObserverSupport,
    SeamBundle,
    carrier_quotient_invariance_report,
    carrier_refinement_naturality_report,
    controlled_oriented_s2_limit_report,
    echosahedral_carrier_conformance_report,
    echosahedral_federation_receipt,
    finite_matrix_interface_algebra_schema,
    federation_sewing_report,
    interface_algebra_sha256,
    main,
    presentation_firewall_report,
    reference_echosahedral_carrier,
    reference_federation_instrument_bundle,
    reference_incidence_nerve_federation,
    relabel_federation_ports,
    screen_port_map_carrier_bridge_report,
    verify_reference_federation_instrument_bundle,
)
from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations
from oph_fpe.core.screen_ports import assign_echosahedral_ports


RECORD_HASH = interface_algebra_sha256({"algebra": "record", "version": 1})
CHECKPOINT_HASH = interface_algebra_sha256({"algebra": "checkpoint", "version": 1})


def _binding(matrix_size: int = 1) -> InterfaceAlgebraBinding:
    interface_id = "I_overlap_v1"
    interface_hash = interface_algebra_sha256(
        finite_matrix_interface_algebra_schema(interface_id, matrix_size)
    )
    return InterfaceAlgebraBinding(
        interface_algebra_id=interface_id,
        interface_algebra_sha256=interface_hash,
        left_interface_algebra_sha256=interface_hash,
        right_interface_algebra_sha256=interface_hash,
    )


def _external_components(carrier, used_ports: set[int], prefix: str):
    remaining = set(range(12)) - set(used_ports)
    graph = nx.Graph()
    graph.add_nodes_from(remaining)
    graph.add_edges_from(
        (left, right)
        for left, right in carrier.edges
        if left in remaining and right in remaining
    )
    return tuple(
        ExternalBoundaryBundle(
            boundary_id=f"{prefix}-{index}",
            carrier_id=carrier.carrier_id,
            ports=tuple(sorted(component)),
            boundary_condition="open_external",
            boundary_algebra_sha256=_binding().interface_algebra_sha256,
        )
        for index, component in enumerate(nx.connected_components(graph))
    )


def _two_carrier_federation() -> EchosahedralFederation:
    left = reference_echosahedral_carrier("c0")
    right = reference_echosahedral_carrier("c1")
    seam = SeamBundle(
        seam_id="s01",
        left_carrier_id="c0",
        right_carrier_id="c1",
        left_ports=(0,),
        right_ports=(0,),
        left_to_right_ports=(0,),
        right_to_left_ports=(0,),
        left_to_right_orientation=(-1,),
        right_to_left_orientation=(-1,),
        collar_kind="single_port",
        interface_algebra=_binding(),
    )
    boundaries = _external_components(left, {0}, "b0") + _external_components(
        right, {0}, "b1"
    )
    observer = ObserverSupport(
        observer_token="observer-01",
        carrier_ids=frozenset({"c0", "c1"}),
        visible_seam_ids=frozenset({"s01"}),
        record_algebra_sha256=RECORD_HASH,
        checkpoint_cut_sha256=CHECKPOINT_HASH,
    )
    return EchosahedralFederation(
        federation_id="fed-two",
        carriers=(left, right),
        seams=(seam,),
        external_boundaries=boundaries,
        observer_supports=(observer,),
    )


def _chain_federation(size: int) -> EchosahedralFederation:
    carriers = tuple(
        reference_echosahedral_carrier(f"c{index}") for index in range(size)
    )
    seams = tuple(
        SeamBundle(
            seam_id=f"s{index}-{index + 1}",
            left_carrier_id=f"c{index}",
            right_carrier_id=f"c{index + 1}",
            left_ports=(1,),
            right_ports=(0,),
            left_to_right_ports=(0,),
            right_to_left_ports=(1,),
            left_to_right_orientation=(-1,),
            right_to_left_orientation=(-1,),
            collar_kind="single_port",
            interface_algebra=_binding(),
        )
        for index in range(size - 1)
    )
    boundaries = []
    for index, carrier in enumerate(carriers):
        used_ports = set()
        if index:
            used_ports.add(0)
        if index + 1 < size:
            used_ports.add(1)
        boundaries.extend(_external_components(carrier, used_ports, f"b{index}"))
    return EchosahedralFederation(
        federation_id=f"chain-{size}",
        carriers=carriers,
        seams=seams,
        external_boundaries=tuple(boundaries),
    )


def test_reference_carrier_exactly_certifies_12_30_20_antipode_and_a5() -> None:
    carrier = reference_echosahedral_carrier("cell")
    report = echosahedral_carrier_conformance_report(carrier)

    assert report["port_count"] == 12
    assert report["edge_count"] == 30
    assert report["face_count"] == 20
    assert report["vertex_degree_profile"] == {"5": 12}
    assert report["antipode_fixed_point_count"] == 0
    assert report["closed_euler_two_surface"] is True
    assert report["a5_action"]["registered_action_count"] == 60
    assert report["a5_action"]["all_actions_preserve_edges"] is True
    assert report["a5_action"]["all_actions_preserve_oriented_faces"] is True
    assert report["a5_action"]["all_actions_commute_with_antipode"] is True
    assert report["declared_response_sector_dimensions"] == [1, 3, 3, 5]
    assert report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is True
    assert report["hidden_coordinates_eligible_for_promoted_geometry"] is False


def test_edge_and_face_mutations_fail_local_conformance() -> None:
    carrier = reference_echosahedral_carrier("cell")
    mutated_edges = list(carrier.edges)
    mutated_edges[0] = (0, carrier.antipode[0])
    edge_report = echosahedral_carrier_conformance_report(
        replace(carrier, edges=tuple(mutated_edges))
    )
    assert edge_report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is False

    first, second, third = carrier.faces[0]
    mutated_faces = list(carrier.faces)
    mutated_faces[0] = (first, third, second)
    face_report = echosahedral_carrier_conformance_report(
        replace(carrier, faces=tuple(mutated_faces))
    )
    assert face_report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is False
    assert "faces_are_not_consistently_outward_oriented" in face_report["blockers"]

    malformed_antipode = echosahedral_carrier_conformance_report(
        replace(carrier, antipode=(99,) * 12)
    )
    assert malformed_antipode["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is False

    malformed_actions = list(carrier.a5_actions)
    malformed_actions[-1] = malformed_actions[0]
    malformed_a5 = echosahedral_carrier_conformance_report(
        replace(carrier, a5_actions=tuple(malformed_actions))
    )
    assert malformed_a5["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is False


def test_complete_typed_sewing_and_external_boundary_declarations_pass() -> None:
    federation = _two_carrier_federation()
    report = federation_sewing_report(federation)

    assert report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is True
    assert report["FEDERATION_SEWING_RECEIPT"] is True
    assert report["undeclared_dangling_ports"] == []
    assert report["seams"][0]["gluing_map_bijective"] is True
    assert report["seams"][0]["forward_reverse_composition_identity"] is True
    assert report["seams"][0]["orientation_reversing"] is True
    assert (
        report["seams"][0]["forward_reverse_orientation_composition_identity"] is True
    )
    assert report["seams"][0]["endpoint_interface_algebra_hashes_agree"] is True
    assert report["carrier_count_is_support_chart_cell_count"] is False
    assert report["carrier_count_is_screen_entropy_capacity_N_star"] is False


def test_antipodal_edge_and_face_collar_types_are_explicitly_checked() -> None:
    left = reference_echosahedral_carrier("left")
    right = reference_echosahedral_carrier("right")
    cases = [
        (
            "antipodal_pair",
            (0, left.antipode[0]),
            (0, right.antipode[0]),
        ),
        ("edge_bundle", tuple(left.edges[0]), tuple(right.edges[0])),
        ("face_collar", tuple(left.faces[0]), tuple(right.faces[0])),
    ]
    for index, (kind, left_ports, right_ports) in enumerate(cases):
        seam = SeamBundle(
            seam_id=f"s-{index}",
            left_carrier_id="left",
            right_carrier_id="right",
            left_ports=left_ports,
            right_ports=right_ports,
            left_to_right_ports=right_ports,
            right_to_left_ports=left_ports,
            left_to_right_orientation=(-1,) * len(left_ports),
            right_to_left_orientation=(-1,) * len(right_ports),
            collar_kind=kind,
            interface_algebra=_binding(len(left_ports)),
        )
        federation = EchosahedralFederation(
            federation_id=f"typed-{index}",
            carriers=(left, right),
            seams=(seam,),
            external_boundaries=(
                _external_components(left, set(left_ports), f"l-{index}")
                + _external_components(right, set(right_ports), f"r-{index}")
            ),
        )
        report = federation_sewing_report(federation)
        assert report["seams"][0]["collar_kind"] == kind
        assert report["seams"][0]["SEAM_BUNDLE_RECEIPT"] is True
        assert report["FEDERATION_SEWING_RECEIPT"] is True


def test_invalid_bijection_orientation_hash_and_disconnected_bundle_fail() -> None:
    federation = _two_carrier_federation()
    seam = federation.seams[0]
    invalid_inverse = replace(seam, right_to_left_ports=(1,))
    report = federation_sewing_report(replace(federation, seams=(invalid_inverse,)))
    assert report["FEDERATION_SEWING_RECEIPT"] is False
    assert report["seams"][0]["forward_reverse_composition_identity"] is False

    invalid_orientation = replace(seam, left_to_right_orientation=(1,))
    report = federation_sewing_report(replace(federation, seams=(invalid_orientation,)))
    assert report["seams"][0]["orientation_reversing"] is False

    invalid_binding = replace(
        seam.interface_algebra,
        right_interface_algebra_sha256=interface_algebra_sha256({"different": True}),
    )
    report = federation_sewing_report(
        replace(federation, seams=(replace(seam, interface_algebra=invalid_binding),))
    )
    assert report["seams"][0]["endpoint_interface_algebra_hashes_agree"] is False

    left, right = federation.carriers
    disconnected_left = (0, left.antipode[0])
    disconnected_right = (0, right.antipode[0])
    disconnected = replace(
        seam,
        left_ports=disconnected_left,
        right_ports=disconnected_right,
        left_to_right_ports=disconnected_right,
        right_to_left_ports=disconnected_left,
        left_to_right_orientation=(-1, -1),
        right_to_left_orientation=(-1, -1),
        collar_kind="connected_bundle",
    )
    report = federation_sewing_report(
        replace(
            federation,
            seams=(disconnected,),
            external_boundaries=(
                _external_components(left, set(disconnected_left), "dl")
                + _external_components(right, set(disconnected_right), "dr")
            ),
        )
    )
    assert report["seams"][0]["left_bundle_connected"] is False
    assert report["FEDERATION_SEWING_RECEIPT"] is False


def test_dangling_or_reused_ports_fail_sewing() -> None:
    federation = _two_carrier_federation()
    missing_boundary = replace(
        federation, external_boundaries=federation.external_boundaries[:-1]
    )
    report = federation_sewing_report(missing_boundary)
    assert report["FEDERATION_SEWING_RECEIPT"] is False
    assert report["undeclared_dangling_ports"]

    reused = replace(federation.seams[0], seam_id="s-reused")
    report = federation_sewing_report(
        replace(federation, seams=federation.seams + (reused,))
    )
    assert report["FEDERATION_SEWING_RECEIPT"] is False
    assert any(
        "local_port_used_by_more_than_one_seam" in item for item in report["blockers"]
    )


def test_one_carrier_observer_allowed_but_disconnected_support_fails() -> None:
    single = reference_echosahedral_carrier("single")
    single_support = ObserverSupport(
        "one-carrier",
        frozenset({"single"}),
        frozenset(),
        RECORD_HASH,
        CHECKPOINT_HASH,
    )
    single_federation = EchosahedralFederation(
        "single-federation",
        (single,),
        (),
        _external_components(single, set(), "single-boundary"),
        (single_support,),
    )
    assert (
        federation_sewing_report(single_federation)["FEDERATION_SEWING_RECEIPT"] is True
    )

    c0 = reference_echosahedral_carrier("c0")
    c1 = reference_echosahedral_carrier("c1")
    c2 = reference_echosahedral_carrier("c2")
    s01 = replace(
        _two_carrier_federation().seams[0],
        seam_id="s01",
        left_carrier_id="c0",
        right_carrier_id="c1",
    )
    s12 = replace(
        s01,
        seam_id="s12",
        left_carrier_id="c1",
        right_carrier_id="c2",
        left_ports=(1,),
        right_ports=(0,),
        left_to_right_ports=(0,),
        right_to_left_ports=(1,),
    )
    disconnected_support = ObserverSupport(
        "disconnected-observer",
        frozenset({"c0", "c2"}),
        frozenset(),
        RECORD_HASH,
        CHECKPOINT_HASH,
    )
    boundaries = (
        _external_components(c0, {0}, "c0b")
        + _external_components(c1, {0, 1}, "c1b")
        + _external_components(c2, {0}, "c2b")
    )
    report = federation_sewing_report(
        EchosahedralFederation(
            "chain",
            (c0, c1, c2),
            (s01, s12),
            boundaries,
            (disconnected_support,),
        )
    )
    assert report["observer_supports"][0]["connected"] is False
    assert report["FEDERATION_SEWING_RECEIPT"] is False


def test_arbitrary_port_renaming_and_a5_reorientation_are_exact_invariances() -> None:
    federation = _two_carrier_federation()
    arbitrary = tuple(reversed(range(12)))
    a5_reorientation = federation.carriers[1].a5_actions[17]
    permutations = {"c0": arbitrary, "c1": a5_reorientation}
    transformed = relabel_federation_ports(federation, permutations)

    assert all(
        echosahedral_carrier_conformance_report(carrier)[
            "ECHOSAHEDRAL_CARRIER_CONFORMANCE"
        ]
        for carrier in transformed.carriers
    )
    assert federation_sewing_report(transformed)["FEDERATION_SEWING_RECEIPT"] is True
    invariance = carrier_quotient_invariance_report(
        federation, transformed, permutations
    )
    assert invariance["exact_cotransformation_verified"] is True
    assert invariance["quotient_visible_contract_exports_equal"] is True
    assert invariance["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is True

    # Keeping the old A5 rows after renaming the rest is not a valid presentation change.
    broken_carrier = replace(
        transformed.carriers[0], a5_actions=federation.carriers[0].a5_actions
    )
    assert (
        echosahedral_carrier_conformance_report(broken_carrier)[
            "ECHOSAHEDRAL_CARRIER_CONFORMANCE"
        ]
        is False
    )

    broken_seam = replace(
        transformed.seams[0],
        left_ports=federation.seams[0].left_ports,
    )
    broken = replace(transformed, seams=(broken_seam,))
    assert (
        carrier_quotient_invariance_report(federation, broken, permutations)[
            "CARRIER_QUOTIENT_INVARIANCE_RECEIPT"
        ]
        is False
    )


def test_hidden_coordinates_names_and_carrier_to_h3_aliases_are_rejected() -> None:
    federation = _two_carrier_federation()
    safe = presentation_firewall_report(
        federation,
        {"port_response": [0.1, 0.2], "semantic_record": {"accepted": True}},
    )
    assert safe["CARRIER_PRESENTATION_FIREWALL_RECEIPT"] is True

    coordinates = federation.carriers[0].port_coordinates
    leaked_by_value = presentation_firewall_report(
        federation, {"innocent_name": coordinates}
    )
    assert leaked_by_value["CARRIER_PRESENTATION_FIREWALL_RECEIPT"] is False
    assert leaked_by_value["hidden_coordinate_value_paths"]

    leaked_names = presentation_firewall_report(
        federation, {"labels": list(federation.carriers[0].port_names)}
    )
    assert leaked_names["CARRIER_PRESENTATION_FIREWALL_RECEIPT"] is False

    forbidden = presentation_firewall_report(
        federation, {"geometry": {"h3_point": [0.0, 0.0, 0.0]}}
    )
    assert "h3_point" in CARRIER_FORBIDDEN_PROMOTION_FIELDS
    assert forbidden["forbidden_keys_present"] == ["h3_point"]
    assert forbidden["CARRIER_PRESENTATION_FIREWALL_RECEIPT"] is False


def test_parent_receipts_compute_refinement_and_stop_before_h3_event_and_bw() -> None:
    federation = _two_carrier_federation()
    permutations = {
        "c0": tuple(reversed(range(12))),
        "c1": federation.carriers[1].a5_actions[3],
    }
    transformed = relabel_federation_ports(federation, permutations)
    report = echosahedral_federation_receipt(
        federation,
        promoted_payload={"port_response": [1, 2, 3]},
        equivalent_presentation=transformed,
        presentation_port_permutations=permutations,
    )

    assert report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is True
    assert report["FEDERATION_SEWING_RECEIPT"] is True
    assert report["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is True
    assert report["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is True
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is True
    assert report["INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT"] is True
    assert report["HIGHER_OVERLAP_COCYCLE_RECEIPT"] is True
    assert report["NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"] is False
    assert report["FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT"] is True
    assert report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is False
    assert report["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False
    assert report["refinement_naturality"]["blockers"] == []
    assert report["carrier_to_support_chart_realization"] == (
        "verified_finite_reference_tower_levels_0_2"
    )
    # The cardinality firewall stays intact even with computed receipts.
    assert report["support_chart_cell_count"] is None
    assert report["support_regulator_count"] is None
    assert report["S2_SUPPORT_CHART_EMERGENCE_RECEIPT"] is False
    assert report["H3_FRAME_EMERGENCE_RECEIPT"] is False
    assert report["EVENT_MANIFOLD_RECEIPT"] is False
    assert report["BW_KMS_CLOCK_RECEIPT"] is False
    assert report["PHYSICAL_H3_KMS_EMERGENCE_RECEIPT"] is False

    no_witness = echosahedral_federation_receipt(federation)
    assert no_witness["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert no_witness["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False


def test_doctored_embedding_fails_refinement_naturality_with_named_blockers() -> None:
    baseline = carrier_refinement_naturality_report()
    assert baseline["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is True
    assert baseline["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is True
    assert baseline["blockers"] == []

    doctored = list(baseline["embedding_port_to_defect_vertex"])
    doctored[0], doctored[1] = doctored[1], doctored[0]
    report = carrier_refinement_naturality_report(embedding_override=doctored)

    assert report["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is False
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False
    assert report["carrier_embedding_map_supplied"] is False
    assert (
        "declared_embedding_differs_from_recomputed_nearest_vertex_matching"
        in report["blockers"]
    )


def test_broken_port_permutation_fails_seam_law_naturality() -> None:
    actions = list(icosahedral_a5_port_permutations())
    transposition = list(range(12))
    transposition[0], transposition[1] = 1, 0
    actions[-1] = tuple(transposition)
    report = carrier_refinement_naturality_report(
        port_permutation_override=actions
    )

    assert report["seam_law_naturality_verified"] is False
    assert report["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is False
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False
    assert any(
        blocker.startswith("seam_law_naturality_violated_for_")
        for blocker in report["blockers"]
    )
    assert report["seam_law_violation_examples"]
    assert report["seam_law_violation_examples"][0]["failure"] == (
        "port_permutation_has_no_proper_rotation_realization"
    )


def test_compact_json_bundle_reuses_one_template_and_remains_fail_closed() -> None:
    federation = _two_carrier_federation()
    bundle = reference_federation_instrument_bundle(
        federation, promoted_payload={"port_response": [0.1, 0.2]}
    )

    assert bundle["local_carrier_template"] == (
        "regular_icosahedron_12_30_20_antipode_a5_v1"
    )
    assert "port_coordinates" not in bundle
    report = verify_reference_federation_instrument_bundle(bundle)
    assert report["INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"] is True
    assert report["shared_template_encoded_once"] is True
    assert report["per_carrier_coordinate_tables_embedded"] is False
    assert report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is True
    assert report["FEDERATION_SEWING_RECEIPT"] is True
    assert report["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert report["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is True
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is True
    assert report["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False

    corrupted = copy.deepcopy(bundle)
    corrupted["seams"][0]["interface_algebra"]["right_interface_algebra_sha256"] = (
        interface_algebra_sha256({"corrupted": True})
    )
    failed = verify_reference_federation_instrument_bundle(corrupted)
    assert failed["INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"] is True
    assert failed["FEDERATION_SEWING_RECEIPT"] is False


def test_existing_screen_port_map_bridge_never_promotes_current_engine() -> None:
    port_map = assign_echosahedral_ports(
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        2,
    )
    current_report = port_map.as_jsonable()
    bridge = screen_port_map_carrier_bridge_report(current_report)

    assert bridge["REFERENCE_SCREEN_PORT_MAP_CARRIER_BRIDGE_RECEIPT"] is True
    assert bridge["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is True
    assert bridge["FEDERATION_SEWING_RECEIPT"] is False
    assert bridge["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert bridge["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is True
    assert bridge["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is True
    assert bridge["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False
    assert bridge["S2_SUPPORT_CHART_EMERGENCE_RECEIPT"] is False
    assert bridge["H3_FRAME_EMERGENCE_RECEIPT"] is False
    assert bridge["EVENT_MANIFOLD_RECEIPT"] is False
    assert bridge["BW_KMS_CLOCK_RECEIPT"] is False


def test_bundle_cli_can_require_sewing_but_source_tier_stays_failed(
    tmp_path: Path, capsys
) -> None:
    bundle = reference_federation_instrument_bundle(_two_carrier_federation())
    path = tmp_path / "federation.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    assert main([str(path), "--require", "sewing"]) == 0
    sewing_output = json.loads(capsys.readouterr().out)
    assert sewing_output["FEDERATION_SEWING_RECEIPT"] is True

    assert main([str(path), "--require", "source"]) == 1
    source_output = json.loads(capsys.readouterr().out)
    assert source_output["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False


def test_reference_carriers_share_one_immutable_template_and_one_local_audit() -> None:
    left = reference_echosahedral_carrier("left-shared")
    right = reference_echosahedral_carrier("right-shared")

    assert left is not right
    assert left.port_names is right.port_names
    assert left.port_coordinates is right.port_coordinates
    assert left.edges is right.edges
    assert left.faces is right.faces
    assert left.antipode is right.antipode
    assert left.a5_actions is right.a5_actions

    federation = _chain_federation(130)
    report = federation_sewing_report(federation)
    summary = report["carrier_conformance_summary"]

    assert report["FEDERATION_SEWING_RECEIPT"] is True
    assert report["carrier_count"] == 130
    assert report["exact_source_carrier_count"] == 130
    assert report["support_regulator_count"] is None
    assert report["carrier_count_is_support_regulator_count"] is False
    assert summary["unique_structural_presentation_count"] == 1
    assert summary["local_conformance_audit_count"] == 1
    assert summary["shared_reference_template_carrier_count"] == 130
    assert summary["shared_reference_template_conformance_verified_once"] is True


def test_large_report_is_bounded_but_hashes_every_verified_row() -> None:
    federation = _chain_federation(130)
    report = federation_sewing_report(federation)

    assert report["report_detail_limit"] == 64
    assert len(report["carrier_conformance"]) == 64
    assert report["carrier_conformance_summary"]["carrier_reports_truncated"] is True
    assert len(report["seams"]) == 64
    assert report["seam_rows_reported_count"] == 64
    assert report["seam_rows_truncated"] is True
    assert report["seam_count"] == 129
    assert report["seam_rows_sha256"].startswith("sha256:")
    assert len(report["external_boundaries"]) == 64
    assert report["external_boundary_rows_truncated"] is True

    late_seams = list(federation.seams)
    late_binding = replace(
        late_seams[-1].interface_algebra,
        right_interface_algebra_sha256=interface_algebra_sha256(
            {"late_unsampled_corruption": True}
        ),
    )
    late_seams[-1] = replace(late_seams[-1], interface_algebra=late_binding)
    failed = federation_sewing_report(replace(federation, seams=tuple(late_seams)))

    assert failed["seam_rows_sha256"] != report["seam_rows_sha256"]
    assert failed["FEDERATION_SEWING_RECEIPT"] is False
    assert failed["seam_failure_examples"]
    assert failed["seam_failure_examples"][0]["seam_id"] == late_seams[-1].seam_id


def test_late_carrier_defect_and_dangling_port_cannot_hide_beyond_report_limit() -> (
    None
):
    federation = _chain_federation(130)
    carriers = list(federation.carriers)
    broken_edges = list(carriers[-1].edges)
    broken_edges[0] = (0, carriers[-1].antipode[0])
    carriers[-1] = replace(carriers[-1], edges=tuple(broken_edges))
    malformed = federation_sewing_report(replace(federation, carriers=tuple(carriers)))

    assert malformed["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is False
    assert (
        malformed["carrier_conformance_summary"]["unique_structural_presentation_count"]
        == 2
    )
    assert any(
        row["conforming"] is False
        for row in malformed["carrier_conformance_summary"]["structural_class_examples"]
    )

    missing_late_boundary = federation_sewing_report(
        replace(federation, external_boundaries=federation.external_boundaries[:-1])
    )
    assert missing_late_boundary["FEDERATION_SEWING_RECEIPT"] is False
    assert missing_late_boundary["undeclared_dangling_port_count"] > 0
    assert missing_late_boundary["undeclared_dangling_ports"]


def test_interface_algebra_and_cocycle_receipts_are_computed_from_checks() -> None:
    report = federation_sewing_report(_two_carrier_federation())
    seam = report["seams"][0]

    assert seam["interface_schema_hashes_agree"] is True
    assert seam["INTERFACE_SCHEMA_HASH_BINDING_RECEIPT"] is True
    assert seam["INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT"] is True
    assert seam["FULL_INTERFACE_ALGEBRA_SEAM_RECEIPT"] is True
    assert seam["interface_algebra_map"]["basis_product_law_exact"] is True
    assert seam["interface_algebra_map"]["products_preserved_exact"] is True
    assert report["INTERFACE_SCHEMA_HASH_BINDING_RECEIPT"] is True
    assert report["INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT"] is True
    assert report["HIGHER_OVERLAP_COCYCLE_RECEIPT"] is True
    assert report["HIGHER_OVERLAP_COCYCLE_CONDITION_RECEIPT"] is True
    assert report["NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"] is False
    assert report["higher_overlap_cocycle"]["identity_violation_count"] == 0
    assert report["FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT"] is True
    assert report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is False
    # The sewing report never asserts the separate chart realization.
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False


def test_equal_opaque_hashes_do_not_bind_the_checked_interface_algebra() -> None:
    federation = _two_carrier_federation()
    opaque = "sha256:" + "0" * 64
    binding = replace(
        federation.seams[0].interface_algebra,
        interface_algebra_sha256=opaque,
        left_interface_algebra_sha256=opaque,
        right_interface_algebra_sha256=opaque,
    )
    report = federation_sewing_report(
        replace(
            federation,
            seams=(replace(federation.seams[0], interface_algebra=binding),),
        )
    )
    seam = report["seams"][0]

    assert seam["interface_schema_hashes_agree"] is True
    assert seam["interface_schema_hash_binds_checked_algebra"] is False
    assert seam["FULL_INTERFACE_ALGEBRA_SEAM_RECEIPT"] is False
    assert (
        "interface_algebra_hash_does_not_bind_checked_matrix_algebra"
        in seam["blockers"]
    )
    assert report["FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT"] is False


def test_non_bijective_gluing_fails_interface_algebra_homomorphism() -> None:
    federation = _two_carrier_federation()
    left = federation.carriers[0]
    right = federation.carriers[1]
    edge_left = tuple(left.edges[0])
    edge_right = tuple(right.edges[0])
    collapsed = SeamBundle(
        seam_id="collapsed",
        left_carrier_id="c0",
        right_carrier_id="c1",
        left_ports=edge_left,
        right_ports=edge_right,
        left_to_right_ports=(edge_right[0], edge_right[0]),
        right_to_left_ports=(edge_left[0], edge_left[0]),
        left_to_right_orientation=(-1, -1),
        right_to_left_orientation=(-1, -1),
        collar_kind="edge_bundle",
        interface_algebra=_binding(2),
    )
    report = federation_sewing_report(replace(federation, seams=(collapsed,)))
    seam = report["seams"][0]

    assert seam["INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT"] is False
    assert seam["interface_algebra_map"]["receipt"] is False
    assert (
        "induced_position_map_is_not_injective"
        in seam["interface_algebra_map"]["blockers"]
    )
    assert report["INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT"] is False
    assert report["FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT"] is False
    assert report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is False


def test_triangle_loop_that_misses_identity_fails_higher_overlap_cocycle() -> None:
    carriers = tuple(
        reference_echosahedral_carrier(f"t{index}") for index in range(3)
    )

    def _seam(seam_id, left, right, left_port, right_port):
        return SeamBundle(
            seam_id=seam_id,
            left_carrier_id=left,
            right_carrier_id=right,
            left_ports=(left_port,),
            right_ports=(right_port,),
            left_to_right_ports=(right_port,),
            right_to_left_ports=(left_port,),
            left_to_right_orientation=(-1,),
            right_to_left_orientation=(-1,),
            collar_kind="single_port",
            interface_algebra=_binding(),
        )

    # Around the triangle the composed maps send t0 port 0 to t0 port 1.
    seams = (
        _seam("s01", "t0", "t1", 0, 0),
        _seam("s12", "t1", "t2", 0, 0),
        _seam("s20", "t2", "t0", 0, 1),
    )
    federation = EchosahedralFederation(
        federation_id="broken-triangle",
        carriers=carriers,
        seams=seams,
        external_boundaries=(),
    )
    report = federation_sewing_report(federation)
    cocycle = report["higher_overlap_cocycle"]

    assert cocycle["triangle_count"] == 1
    assert cocycle["composable_port_loop_count"] >= 1
    assert cocycle["identity_violation_count"] >= 1
    assert cocycle["identity_violations"][0]["start_port"] == 0
    assert cocycle["identity_violations"][0]["returned_port"] == 1
    assert report["HIGHER_OVERLAP_COCYCLE_RECEIPT"] is False
    assert report["FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT"] is False
    assert report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is False


def test_coherent_triangle_supplies_a_nonvacuous_cocycle_witness() -> None:
    carriers = tuple(
        reference_echosahedral_carrier(f"w{index}") for index in range(3)
    )

    def _seam(seam_id: str, left: str, right: str) -> SeamBundle:
        return SeamBundle(
            seam_id=seam_id,
            left_carrier_id=left,
            right_carrier_id=right,
            left_ports=(0,),
            right_ports=(0,),
            left_to_right_ports=(0,),
            right_to_left_ports=(0,),
            left_to_right_orientation=(-1,),
            right_to_left_orientation=(-1,),
            collar_kind="single_port",
            interface_algebra=_binding(),
        )

    federation = EchosahedralFederation(
        federation_id="coherent-triangle-control",
        carriers=carriers,
        seams=(
            _seam("w01", "w0", "w1"),
            _seam("w12", "w1", "w2"),
            _seam("w20", "w2", "w0"),
        ),
        # Port reuse deliberately makes this only a cocycle control, not a
        # positive physical federation-sewing fixture.
        external_boundaries=(),
    )
    report = federation_sewing_report(federation)
    cocycle = report["higher_overlap_cocycle"]

    assert cocycle["triangle_count"] == 1
    assert cocycle["composable_port_loop_count"] >= 1
    assert cocycle["identity_violation_count"] == 0
    assert cocycle["HIGHER_OVERLAP_COCYCLE_CONDITION_RECEIPT"] is True
    # Graph adjacency plus matching raw labels is not itself a declaration
    # that the three collars have a common nonempty algebra restriction.
    assert cocycle["NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"] is False
    assert report["FEDERATION_SEWING_RECEIPT"] is False


def test_source_incidence_nerve_is_nonvacuous_and_respects_port_ownership() -> None:
    federation = reference_incidence_nerve_federation()
    report = federation_sewing_report(federation)
    cocycle = report["higher_overlap_cocycle"]

    assert len(federation.carriers) == 12
    assert len(federation.seams) == 30
    assert len(federation.triple_overlaps) == 20
    endpoint_owners = [
        (seam.left_carrier_id, seam.left_ports[0])
        for seam in federation.seams
    ] + [
        (seam.right_carrier_id, seam.right_ports[0])
        for seam in federation.seams
    ]
    assert len(endpoint_owners) == len(set(endpoint_owners))
    assert report["FEDERATION_SEWING_RECEIPT"] is True
    assert report["CONNECTED_OBSERVER_SUPPORT_WITNESS"] is True
    assert report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is True
    assert cocycle["triangle_count"] == 20
    assert cocycle["declared_nonempty_triple_overlap_count"] == 20
    assert cocycle["valid_declared_triple_overlap_count"] == 20
    assert cocycle["composable_triple_restriction_algebra_loop_count"] == 20
    assert cocycle["identity_violation_count"] == 0
    assert cocycle["HIGHER_OVERLAP_COCYCLE_CONDITION_RECEIPT"] is True
    assert cocycle["NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"] is True
    assert all(
        row["matrix_unit_identity_composition_exact"]
        for row in report["triple_overlaps"]
    )


def test_incidence_nerve_has_controlled_oriented_s2_limit_on_same_tower() -> None:
    federation = reference_incidence_nerve_federation()
    support = controlled_oriented_s2_limit_report(federation, max_level=3)

    assert support["carrier_to_base_vertex_bijection"] is True
    assert support["source_edge_nerve_exact"] is True
    assert support["source_oriented_face_nerve_exact"] is True
    assert support["nerve_simplex_counts"] == {
        "vertices": 12,
        "edges": 30,
        "oriented_faces": 20,
    }
    assert support["analytic_contraction_factor"] < 1.0
    assert support["analytic_all_levels_contraction_proved"] is True
    assert support["mesh_bound_tends_to_zero"] is True
    assert support["computed_prefix_mesh_bounds_verified"] is True
    assert support["computed_prefix_outward_oriented"] is True
    assert support["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"] is True

    permutations = {
        carrier.carrier_id: tuple(reversed(range(12)))
        for carrier in federation.carriers
    }
    equivalent = relabel_federation_ports(federation, permutations)
    parent = echosahedral_federation_receipt(
        federation,
        equivalent_presentation=equivalent,
        presentation_port_permutations=permutations,
    )
    assert parent["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is True
    assert parent["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is True
    assert parent["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"] is True
    assert parent["S2_SUPPORT_CHART_EMERGENCE_RECEIPT"] is True
    assert parent["H3_FRAME_EMERGENCE_RECEIPT"] is False
    assert parent["BW_KMS_CLOCK_RECEIPT"] is False


def test_incidence_nerve_negative_controls_fail_closed() -> None:
    federation = reference_incidence_nerve_federation()

    missing_face = replace(
        federation, triple_overlaps=federation.triple_overlaps[:-1]
    )
    support = controlled_oriented_s2_limit_report(missing_face)
    parent = echosahedral_federation_receipt(missing_face)
    assert support["source_oriented_face_nerve_exact"] is False
    assert support["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"] is False
    assert parent["S2_SUPPORT_CHART_EMERGENCE_RECEIPT"] is False
    assert parent["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False

    first = federation.triple_overlaps[0]
    corrupted_hash = replace(
        first,
        restriction_algebra_sha256=interface_algebra_sha256(
            {"corrupted": "triple restriction"}
        ),
    )
    corrupted = replace(
        federation,
        triple_overlaps=(corrupted_hash, *federation.triple_overlaps[1:]),
    )
    corrupted_report = federation_sewing_report(corrupted)
    assert corrupted_report["FEDERATION_SEWING_RECEIPT"] is False
    assert (
        corrupted_report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"]
        is False
    )
    assert (
        "triple_restriction_hash_does_not_bind_canonical_M1_algebra"
        in corrupted_report["triple_overlap_failure_examples"][0]["blockers"]
    )

    wrong_seams = replace(
        first,
        oriented_seam_ids=(
            federation.triple_overlaps[1].oriented_seam_ids[0],
            first.oriented_seam_ids[1],
            first.oriented_seam_ids[2],
        ),
    )
    misidentified = replace(
        federation,
        triple_overlaps=(wrong_seams, *federation.triple_overlaps[1:]),
    )
    misidentified_report = federation_sewing_report(misidentified)
    assert misidentified_report["FEDERATION_SEWING_RECEIPT"] is False
    assert misidentified_report[
        "NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"
    ] is False


def test_incidence_nerve_bundle_round_trip_binds_the_common_tower() -> None:
    bundle = reference_federation_instrument_bundle(
        reference_incidence_nerve_federation()
    )
    report = verify_reference_federation_instrument_bundle(bundle)

    assert bundle["presentation_relabeling_witness"] == (
        "reverse_every_local_port_order_v1"
    )
    assert report["INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"] is True
    assert report["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is True
    assert report["NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"] is True
    assert report["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is True
    assert report["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"] is True
    assert report["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is True
    assert report["S2_SUPPORT_CHART_EMERGENCE_RECEIPT"] is True

    corrupted = copy.deepcopy(bundle)
    corrupted["support_tower_binding"]["carrier_to_base_vertex"][0][1] = 1
    failed = verify_reference_federation_instrument_bundle(corrupted)
    assert failed["INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"] is True
    assert failed["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"] is False
    assert failed["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False


def test_compact_bundle_rejects_count_type_and_embedded_template_forgery() -> None:
    bundle = reference_federation_instrument_bundle(_two_carrier_federation())

    wrong_count = copy.deepcopy(bundle)
    wrong_count["exact_source_carrier_count"] += 1
    assert (
        verify_reference_federation_instrument_bundle(wrong_count)[
            "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"
        ]
        is False
    )

    coerced_id = copy.deepcopy(bundle)
    coerced_id["carrier_ids"][0] = 0
    assert (
        verify_reference_federation_instrument_bundle(coerced_id)[
            "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"
        ]
        is False
    )

    coerced_endpoint = copy.deepcopy(bundle)
    coerced_endpoint["seams"][0]["left_carrier_id"] = 0
    assert (
        verify_reference_federation_instrument_bundle(coerced_endpoint)[
            "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"
        ]
        is False
    )

    embedded_coordinates = copy.deepcopy(bundle)
    embedded_coordinates["carriers"] = [{"port_coordinates": bundle["carrier_ids"]}]
    failed = verify_reference_federation_instrument_bundle(embedded_coordinates)
    assert failed["INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"] is False
    assert "per-carrier local template fields are forbidden" in failed["parse_error"]


def test_typed_runtime_corruption_fails_closed_without_bool_or_string_coercion() -> (
    None
):
    carrier = reference_echosahedral_carrier("malformed")
    malformed_edges = list(carrier.edges)
    malformed_edges[0] = (0, "not-an-integer")
    local = echosahedral_carrier_conformance_report(
        replace(carrier, edges=tuple(malformed_edges))
    )
    assert local["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] is False
    assert any(
        blocker.startswith("malformed_carrier_presentation:")
        for blocker in local["blockers"]
    )

    federation = _two_carrier_federation()
    bool_port = replace(
        federation.seams[0],
        left_ports=(True,),
        right_to_left_ports=(True,),
    )
    report = federation_sewing_report(replace(federation, seams=(bool_port,)))
    assert report["FEDERATION_SEWING_RECEIPT"] is False
    assert (
        "seam_port_and_orientation_arrays_must_be_exact_integer_tuples"
        in report["seams"][0]["blockers"]
    )

    bool_boundary = replace(federation.external_boundaries[0], ports=(False,))
    report = federation_sewing_report(
        replace(
            federation,
            external_boundaries=(bool_boundary,) + federation.external_boundaries[1:],
        )
    )
    assert report["FEDERATION_SEWING_RECEIPT"] is False
    assert (
        "external_boundary_ports_must_be_an_exact_integer_tuple"
        in report["external_boundaries"][0]["blockers"]
    )
