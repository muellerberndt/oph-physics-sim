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
    echosahedral_carrier_conformance_report,
    echosahedral_federation_receipt,
    federation_sewing_report,
    interface_algebra_sha256,
    main,
    presentation_firewall_report,
    reference_echosahedral_carrier,
    reference_federation_instrument_bundle,
    relabel_federation_ports,
    screen_port_map_carrier_bridge_report,
    verify_reference_federation_instrument_bundle,
)
from oph_fpe.core.screen_ports import assign_echosahedral_ports


INTERFACE_HASH = interface_algebra_sha256(
    {"algebra": "finite_overlap_visible_matrix_star_algebra", "dimension": 4}
)
RECORD_HASH = interface_algebra_sha256({"algebra": "record", "version": 1})
CHECKPOINT_HASH = interface_algebra_sha256({"algebra": "checkpoint", "version": 1})


def _binding() -> InterfaceAlgebraBinding:
    return InterfaceAlgebraBinding(
        interface_algebra_id="I_overlap_v1",
        interface_algebra_sha256=INTERFACE_HASH,
        left_interface_algebra_sha256=INTERFACE_HASH,
        right_interface_algebra_sha256=INTERFACE_HASH,
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
            boundary_algebra_sha256=INTERFACE_HASH,
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
            interface_algebra=_binding(),
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


def test_parent_receipts_stop_before_support_refinement_s2_h3_event_and_bw() -> None:
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
    assert report["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is False
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False
    assert report["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"] is False
    assert report["support_chart_cell_count"] is None
    assert report["S2_SUPPORT_CHART_EMERGENCE_RECEIPT"] is False
    assert report["H3_FRAME_EMERGENCE_RECEIPT"] is False
    assert report["EVENT_MANIFOLD_RECEIPT"] is False
    assert report["BW_KMS_CLOCK_RECEIPT"] is False
    assert report["PHYSICAL_H3_KMS_EMERGENCE_RECEIPT"] is False

    no_witness = echosahedral_federation_receipt(federation)
    assert no_witness["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is False


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
    assert report["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is False
    assert report["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False
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
    assert bridge["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is False
    assert bridge["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False
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
