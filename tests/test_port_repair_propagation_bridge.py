from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json

import pytest
import sympy as sp

from oph_fpe.dynamics import port_repair_propagation_bridge as bridge
from oph_fpe.dynamics import verify_port_repair_propagation_bridge_independent as independent


REPORT = bridge.produce_bridge_receipt()


@pytest.fixture(scope="module")
def vertex_packet() -> dict:
    return bridge.make_candidate_source_packet(bridge.ORBIT_VERTEX)


def _independent_orbit_betas() -> dict[str, Fraction]:
    """Recompute the sixth-order residual at a normalized vertex with SymPy.

    This implementation neither imports the producer's Q(sqrt(5)) polynomial
    helpers nor consumes its orbit rows.  The vertex residual defines I6=1 at
    this reference direction, so ``sum (u dot v)^6 - N/7`` is beta directly.
    """

    phi = (1 + sp.sqrt(5)) / 2
    vertices: list[sp.Matrix] = []
    for left in (sp.Integer(1), sp.Integer(-1)):
        for right in (phi, -phi):
            vertices.append(sp.Matrix((0, left, right)))
    for left in (sp.Integer(1), sp.Integer(-1)):
        for right in (phi, -phi):
            vertices.append(sp.Matrix((left, right, 0)))
    for left in (sp.Integer(1), sp.Integer(-1)):
        for right in (phi, -phi):
            vertices.append(sp.Matrix((right, 0, left)))

    def dot(left: sp.Matrix, right: sp.Matrix) -> sp.Expr:
        return sp.expand((left.T * right)[0])

    def adjacent(left: int, right: int) -> bool:
        return sp.simplify(dot(vertices[left], vertices[right]) - phi) == 0

    edges = [
        (left, right)
        for left in range(12)
        for right in range(left + 1, 12)
        if adjacent(left, right)
    ]
    faces = [
        (first, second, third)
        for first in range(12)
        for second in range(first + 1, 12)
        for third in range(second + 1, 12)
        if adjacent(first, second)
        and adjacent(first, third)
        and adjacent(second, third)
    ]
    assert len(edges) == 30
    assert len(faces) == 20

    orbits = {
        bridge.ORBIT_VERTEX: vertices,
        bridge.ORBIT_FACE: [
            vertices[first] + vertices[second] + vertices[third]
            for first, second, third in faces
        ],
        bridge.ORBIT_EDGE: [
            vertices[left] + vertices[right] for left, right in edges
        ],
    }
    reference = vertices[0]
    reference_norm = dot(reference, reference)
    result: dict[str, Fraction] = {}
    for orbit, directions in orbits.items():
        moment = sp.Integer(0)
        for direction in directions:
            direction_norm = dot(direction, direction)
            moment += dot(direction, reference) ** 6 / (
                direction_norm**3 * reference_norm**3
            )
        beta = sp.simplify(moment - sp.Rational(len(directions), 7))
        assert beta.is_Rational
        result[orbit] = Fraction(int(beta.p), int(beta.q))
    return result


def test_current_source_receipt_fails_closed_without_translation() -> None:
    verification = bridge.verify_bridge_receipt(REPORT)

    assert verification["receipt"] is True
    assert REPORT["status"] == (
        bridge.BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
    )
    assert REPORT[bridge.INTERNAL_SEAM_REPAIR_CERTIFIED] is True
    assert REPORT[bridge.SPATIAL_PORT_HOP_SOURCE_RECEIPT] is False
    assert REPORT[bridge.SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT] is False
    assert REPORT[bridge.FZ11_FORCED_EXCLUSIVE_RECEIPT] is False
    assert REPORT["classification"]["blockers"] == [
        "NO_TWELVE_PORT_ORBIT_TRANSLATION_BINDING"
    ]
    assert REPORT["classification"][
        "internal_support_count_used_as_spatial_selection"
    ] is False
    assert REPORT["source_packet"]["internal_seam_repair"][
        "spatial_translation_identification"
    ] is False
    assert REPORT["epistemic_boundary"]["comparison_data_read"] is False
    assert REPORT["epistemic_boundary"]["physical_prediction_unsealed"] is False
    acceptance = REPORT["live_issue_acceptance_audit"]
    assert acceptance["machine_checked_exact_orbit_implications"] is True
    assert acceptance["mutation_controls"] is True
    assert acceptance["comparison_data_consumed"] is False
    assert acceptance["independent_orbit_ray_recomputation"] is True
    assert acceptance["independent_current_exit_implementation"] is True
    assert acceptance["independent_full_bridge_implementation"] is False
    assert acceptance["lean_full_bridge_implication_chain"] is False
    assert acceptance["forced_exclusive_exit_supported"] is False
    assert acceptance["defensible_exit"] == (
        bridge.BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
    )
    dependency = REPORT["dependency_audit"]
    assert dependency[
        "closed_issue_62_supplies_accepted_physical_repair_law"
    ] is False
    assert dependency["geometry_and_orbit_moments_available"] is True
    assert dependency["geometry_implies_physical_translation_operator"] is False
    assert dependency["indexed_tracked_serialized_data_census_complete"] is True
    assert dependency["semantic_scan_limited_to_current_canonical_json"] is True
    assert dependency[
        "legacy_imported_external_payloads_semantically_scanned"
    ] is False
    assert dependency["local_spatial_or_kinetic_operators_exist"] is True
    assert dependency[
        "registered_accepted_vertex12_translation_to_physical_readout_chain_on_scanned_surface_exists"
    ] is False
    inventory = REPORT["source_operator_ancestry_inventory"]
    assert inventory == REPORT["source_packet"][
        "source_operator_ancestry_inventory"
    ]
    assert inventory[
        "registered_packet_count_excluding_recursive_outputs"
    ] == 0
    assert inventory[
        "accepted_bridge_count_excluding_recursive_outputs"
    ] == 0
    assert inventory["recursive_parent_bridge_receipt_exclusion"] == {
        "path": "data/repair_closure/port_repair_propagation_bridge_receipt.json",
        "reason": (
            "parent output embeds the current negative source packet and is "
            "excluded to avoid recursive custody"
        ),
        "packet_count_included_in_scan": False,
    }
    assert inventory["local_spatial_or_kinetic_operators_exist"] is True
    assert inventory["claim_that_no_spatial_operator_exists"] is False
    assert inventory[
        "producer_code_or_sibling_repository_absence_claimed"
    ] is False
    assert inventory["unregistered_equivalent_semantics_ruled_out"] is False


def test_orbit_rays_are_exact_and_independently_recomputed() -> None:
    rows = REPORT["exact_orbit_ray_table"]["rows"]
    expected = {
        bridge.ORBIT_VERTEX: {
            "support_size": 12,
            "beta": Fraction(64, 175),
            "B6": "2/7875",
            "B6_over_C4_squared": "32/315",
            "B6_over_B0": "16/75",
        },
        bridge.ORBIT_FACE: {
            "support_size": 20,
            "beta": Fraction(-64, 189),
            "B6": "-2/14175",
            "B6_over_C4_squared": "-32/567",
            "B6_over_B0": "-16/135",
        },
        bridge.ORBIT_EDGE: {
            "support_size": 30,
            "beta": Fraction(-2, 7),
            "B6": "-1/12600",
            "B6_over_C4_squared": "-2/63",
            "B6_over_B0": "-1/15",
        },
    }
    independent_betas = _independent_orbit_betas()
    for orbit, values in expected.items():
        row = rows[orbit]
        assert row["support_size"] == values["support_size"]
        assert row["antipodal_pair_count"] * 2 == row["support_size"]
        assert len(row["directions"]) == row["support_size"]
        assert independent_betas[orbit] == values["beta"]
        assert row["ray"]["C4"] == "-1/20"
        assert row["ray"]["B0"] == "1/840"
        assert row["ray"]["B0_over_C4_squared"] == "10/21"
        assert row["ray"]["B6"] == values["B6"]
        assert row["ray"]["B6_over_C4_squared"] == values[
            "B6_over_C4_squared"
        ]
        assert row["ray"]["B6_over_B0"] == values["B6_over_B0"]
        assert row["moment_identities"][
            "identities_checked_coefficientwise_over_Q_sqrt5"
        ] is True
    assert REPORT["exact_orbit_ray_table"]["pairwise_distinct"] is True
    assert REPORT["exact_orbit_ray_table"][
        "intrinsic_angular_ranks_one_through_five"
    ] == "zero"
    assert REPORT["exact_orbit_ray_table"][
        "binary_refinement_of_rank_six_coefficient"
    ] == "B6(a/2) = B6(a)/16"
    assert REPORT["exact_orbit_ray_table"]["physicalization_claimed"] is False


def test_all_three_classifier_exits_are_reachable_and_typed() -> None:
    vertex_source = bridge.make_candidate_source_packet(bridge.ORBIT_VERTEX)
    vertex_core = vertex_source["spatial_hop_operator"]["operator_core"]
    assert vertex_core["symbol_class"] == "real_reciprocal_finite_range_cosine"
    assert vertex_core["direction_coordinates"] == (
        "unnormalized_exact_carrier_rays"
    )
    assert vertex_core["unit_direction_rule"] == "u = d/sqrt(d dot d)"
    assert vertex_core["continuum_normalization"] == "6/12"
    vertex = bridge.classify_source_packet(vertex_source)
    face = bridge.classify_source_packet(
        bridge.make_candidate_source_packet(bridge.ORBIT_FACE)
    )
    edge = bridge.classify_source_packet(
        bridge.make_candidate_source_packet(bridge.ORBIT_EDGE)
    )

    assert vertex["operator_classifier_exit"] == (
        bridge.PORT_INTERFACE_REPAIR_FORCES_FZ11
    )
    assert vertex["issue_certified_exit"] == (
        bridge.ADDITIONAL_PHYSICAL_PREMISE_REQUIRED_ALTERNATIVES_CERTIFIED
    )
    assert vertex["selected_ray"]["B6_over_B0"] == "16/75"
    assert edge["operator_classifier_exit"] == (
        bridge.SEAM_REPAIR_SELECTS_EDGE30_NOT_FZ11
    )
    assert edge["issue_certified_exit"] == (
        bridge.ADDITIONAL_PHYSICAL_PREMISE_REQUIRED_ALTERNATIVES_CERTIFIED
    )
    assert edge["selected_ray"]["B6_over_B0"] == "-1/15"
    assert face["operator_classifier_exit"] == (
        bridge.NO_SOURCE_NATIVE_TRANSLATION_BRIDGE
    )
    assert face["issue_certified_exit"] == (
        bridge.ADDITIONAL_PHYSICAL_PREMISE_REQUIRED_ALTERNATIVES_CERTIFIED
    )
    assert face["blockers"] == [
        "FACE20_ORBIT_IS_A_DISTINCT_NON_FZ11_ALTERNATIVE"
    ]
    assert all(
        result["physicalization_assumed"] is False
        for result in (vertex, face, edge)
    )


def test_missing_readout_never_promotes_a_spatial_operator() -> None:
    packet = bridge.make_candidate_source_packet(
        bridge.ORBIT_VERTEX,
        attach_physical_readout=False,
    )
    result = bridge.classify_source_packet(packet)

    assert result["operator_classifier_exit"] == (
        bridge.NO_SOURCE_NATIVE_TRANSLATION_BRIDGE
    )
    assert result["issue_certified_exit"] == (
        bridge.BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
    )
    assert result["spatial_hop_source_certified"] is True
    assert result["same_operator_physical_readout_certified"] is False
    assert result["blockers"] == ["PHYSICAL_READOUT_ABSENT"]


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("extra_support", "SPATIAL_GRAMMAR"),
        ("missing_support", "SPATIAL_GRAMMAR"),
        ("unequal_weight", "SPATIAL_GRAMMAR"),
        ("onsite_term", "SPATIAL_GRAMMAR"),
        ("radius_two", "SPATIAL_GRAMMAR"),
        ("extra_directional", "SPATIAL_GRAMMAR"),
        ("extra_isotropic", "SPATIAL_GRAMMAR"),
        ("operator_digest", "SPATIAL_DIGEST"),
        ("source_history", "SOURCE_HISTORY"),
        ("readout_operator", "PHYSICAL_READOUT_BINDING"),
        ("readout_digest", "PHYSICAL_READOUT_DIGEST"),
        ("internal_domain", "INTERNAL_REPAIR_BINDING"),
        ("domain_collapse", "FORBIDDEN_DOMAIN_COLLAPSE"),
    ],
)
def test_operator_and_domain_mutations_fail_closed(
    vertex_packet: dict,
    mutation: str,
    blocker: str,
) -> None:
    packet = copy.deepcopy(vertex_packet)
    spatial = packet["spatial_hop_operator"]
    core = spatial["operator_core"]
    readout = packet["physical_readout"]

    if mutation == "extra_support":
        core["directions"].append(["1", "0", "0"])
    elif mutation == "missing_support":
        core["directions"].pop()
    elif mutation == "unequal_weight":
        core["terms"][0]["weight"] = "2"
    elif mutation == "onsite_term":
        core["independent_onsite_term"] = "1"
    elif mutation == "radius_two":
        core["radius_two_terms"] = [{"direction_index": 0, "weight": "1"}]
    elif mutation == "extra_directional":
        core["additional_directional_terms"] = [{"monomial": "kx^6"}]
    elif mutation == "extra_isotropic":
        core["additional_isotropic_terms_through_k6"] = ["r^6"]
    elif mutation == "operator_digest":
        spatial["operator_sha256"] = "sha256:" + "0" * 64
    elif mutation == "source_history":
        spatial["source_history_replay"][0]["weight"] = "2"
    elif mutation == "readout_operator":
        readout["readout_core"]["readback_operator_sha256"] = "sha256:" + "1" * 64
    elif mutation == "readout_digest":
        readout["readout_sha256"] = "sha256:" + "2" * 64
    elif mutation == "internal_domain":
        packet["internal_seam_repair"]["support_kind"] = "spatial_edge_hops"
    elif mutation == "domain_collapse":
        packet["scope_boundary"]["internal_seams_are_spatial_hops"] = True
    else:  # pragma: no cover - guards the parametrization itself
        raise AssertionError(mutation)

    result = bridge.classify_source_packet(packet)
    assert result["operator_classifier_exit"] == (
        bridge.NO_SOURCE_NATIVE_TRANSLATION_BRIDGE
    )
    assert result["issue_certified_exit"] == (
        bridge.BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
    )
    assert blocker in result["blockers"]


def test_receipt_verifier_rejects_status_and_physical_promotion_mutations() -> None:
    mutations = []

    wrong_status = copy.deepcopy(REPORT)
    wrong_status["status"] = bridge.PORT_INTERFACE_REPAIR_FORCES_FZ11
    mutations.append(wrong_status)

    promoted = copy.deepcopy(REPORT)
    promoted[bridge.FZ11_FORCED_EXCLUSIVE_RECEIPT] = True
    mutations.append(promoted)

    changed_ray = copy.deepcopy(REPORT)
    changed_ray["exact_orbit_ray_table"]["rows"][bridge.ORBIT_VERTEX]["ray"][
        "B6"
    ] = "0"
    mutations.append(changed_ray)

    changed_pin = copy.deepcopy(REPORT)
    changed_pin["source_packet"]["carrier_manifest_pin"]["sha256"] = (
        "sha256:" + "0" * 64
    )
    mutations.append(changed_pin)

    for mutation in mutations:
        verification = bridge.verify_bridge_receipt(mutation)
        assert verification["receipt"] is False
        assert verification["status"] == "FAIL"


def _independently_rehash(report: dict) -> None:
    payload = {key: value for key, value in report.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_independent_current_exit_verifier_passes_without_producer_import() -> None:
    result = independent.verify_independently(REPORT)

    assert result["receipt"] is True
    assert result["independent_implementation"] is True
    assert result["imports_bridge_producer"] is False
    assert result["verified_exit"] == (
        bridge.BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
    )
    assert result["positive_physical_bridge_verified"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "forced_status",
        "spatial_operator",
        "physical_readout",
        "internal_physical_promotion",
        "classification_promotion",
        "source_pin",
        "comparison_consumed",
        "full_independence_promotion",
        "top_inventory_summary",
        "dependency_reason",
        "epistemic_result",
        "issue",
        "pin_path",
        "scope_extra_field",
    ],
)
def test_independent_verifier_rejects_rehashed_semantic_mutations(
    mutation: str,
) -> None:
    report = copy.deepcopy(REPORT)
    if mutation == "forced_status":
        report["status"] = bridge.FORCED_EXCLUSIVE_PRIMITIVE_PORT_PROPAGATION_BRANCH
    elif mutation == "spatial_operator":
        report["source_packet"]["spatial_hop_operator"] = {}
    elif mutation == "physical_readout":
        report["source_packet"]["physical_readout"] = {}
    elif mutation == "internal_physical_promotion":
        report["source_packet"]["internal_seam_repair"][
            "physical_repair_law_receipt"
        ] = True
    elif mutation == "classification_promotion":
        report["classification"]["operator_classifier_exit"] = (
            bridge.PORT_INTERFACE_REPAIR_FORCES_FZ11
        )
    elif mutation == "source_pin":
        report["source_packet"]["carrier_manifest_pin"]["sha256"] = (
            "sha256:" + "0" * 64
        )
    elif mutation == "comparison_consumed":
        report["live_issue_acceptance_audit"]["comparison_data_consumed"] = True
    elif mutation == "full_independence_promotion":
        report["live_issue_acceptance_audit"][
            "independent_full_bridge_implementation"
        ] = True
    elif mutation == "top_inventory_summary":
        report["source_operator_ancestry_inventory"][
            "registered_packet_count_excluding_recursive_outputs"
        ] = 1
    elif mutation == "dependency_reason":
        report["dependency_audit"]["reason"] = "changed"
    elif mutation == "epistemic_result":
        report["epistemic_boundary"]["current_result"] = "changed"
    elif mutation == "issue":
        report["issue"] = 0
    elif mutation == "pin_path":
        report["source_packet"]["carrier_manifest_pin"][
            "repository_relative_path"
        ] = "oph_fpe/dynamics/canonical_seam_repair.py"
    elif mutation == "scope_extra_field":
        report["source_packet"]["scope_boundary"]["extra"] = False
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _independently_rehash(report)

    result = independent.verify_independently(report)
    assert result["receipt"] is False
    assert result["status"] == "FAIL"


def test_canonical_receipt_is_byte_exact() -> None:
    canonical = json.loads(bridge.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert canonical == REPORT
    assert bridge.verify_bridge_receipt(canonical)["receipt"] is True
