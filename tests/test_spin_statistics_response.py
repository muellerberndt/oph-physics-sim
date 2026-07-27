"""Tests for the spin statistics semantic artifact producer (#314)."""

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from oph_fpe.core.charged_response import ChargedResponseError, Q5
from oph_fpe.core.spin_statistics_response import (
    QUAT_MINUS_ONE,
    QUAT_ONE,
    Quat,
    produce_spin_statistics_artifact,
    q5_sqrt,
    quat_rotation_matrix,
    quaternion_lift,
)

FIXTURE = Path(__file__).parent / "fixtures" / "echosahedral_federation_reference.json"
PINNED_CARRIER_SHA256 = (
    "cee9f912d2a2b9a66ac9866e3b04666a7acf74355771389bec73d7a04fbb8280"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact(manifest: dict) -> dict:
    return produce_spin_statistics_artifact(manifest)


def test_artifact_binds_the_pinned_certified_carrier(artifact: dict) -> None:
    assert artifact["schema"] == "oph.spin_statistics_semantic_artifact.v1"
    assert artifact["issue"] == 314
    binding = artifact["carrier_binding"]
    assert binding["carrier_manifest_sha256"] == PINNED_CARRIER_SHA256
    assert binding["incidence_edge_count"] == 30
    assert binding["oriented_face_count"] == 20


def test_target_firewall_and_self_hash(artifact: dict) -> None:
    assert "standard_model" not in json.dumps(artifact).lower().replace(
        "no_standard_model_label", ""
    )
    from oph_fpe.core.charged_response import canonical_sha256

    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    assert artifact["artifact_sha256"] == "sha256:" + canonical_sha256(body)


def test_measured_lift_group_is_binary_icosahedral(artifact: dict) -> None:
    lift = artifact["lift_measurement"]
    assert lift["lift_group_order"] == 120
    assert lift["order_profile"] == {
        "1": 1,
        "2": 1,
        "3": 20,
        "4": 30,
        "5": 24,
        "6": 20,
        "10": 24,
    }
    assert lift["unique_nontrivial_involution"] == "-1"
    assert lift["centre_order"] == 2
    assert lift["centre_elements"] == ["+1", "-1"]


def test_klein_four_section_obstruction_is_exhaustive(artifact: dict) -> None:
    obstruction = artifact["section_obstruction"]
    assert obstruction["deck_involutions"] == 15
    assert obstruction["klein_four_subgroups"] == 5
    assert obstruction["no_section_over_any_klein_four_subgroup"] is True
    tables = obstruction["section_exhaustion_per_subgroup"]
    assert len(tables) == 5
    for table in tables:
        assert table["sign_assignments_tested"] == 8
        assert table["sections_found"] == 0
    assert len(artifact["canonical_klein_four_lift_table"]) == 3
    for row in artifact["canonical_klein_four_lift_table"]:
        assert row["lift_square"] == "-1"


def test_support_homology_and_unique_spin_structure(artifact: dict) -> None:
    homology = artifact["support_homology"]
    assert homology["betti_numbers"] == [1, 0, 1]
    assert homology["euler_characteristic"] == 2
    assert homology["integral_h2_torsion"] == []
    assert homology["spin_structure_count"] == 1


def test_orientation_and_refinement_measurements(artifact: dict) -> None:
    deck = artifact["deck_measurement"]
    assert deck["incidence_automorphism_group_order"] == 120
    assert deck["orientation_preserving_rotations"] == 60
    assert deck["antipode_is_orientation_reversing"] is True
    refinement = artifact["refinement_equivariance"]
    assert refinement["levels_measured"] == 3
    assert refinement["persistence_map"] == "identity_on_defect_ports"
    assert all(row["defect_ports"] == 12 for row in refinement["per_level"])


def test_gate_scope_separates_open_lanes(artifact: dict) -> None:
    gate = artifact["physical_source_gate"]
    assert gate["passed"] is True
    assert gate["laboratory_exchange_measurement"] is False
    assert gate["continuum_spin_statistics_theorem"] is False
    assert gate["no_section_over_any_klein_four_deck_subgroup"] is True
    assert gate["unique_spin_structure_on_oriented_support"] is True


def test_deterministic_replay(manifest: dict, artifact: dict) -> None:
    replay = produce_spin_statistics_artifact(manifest)
    assert json.dumps(replay, sort_keys=True) == json.dumps(artifact, sort_keys=True)


def test_exact_sqrt_helper() -> None:
    value = Q5(Fraction(3, 8), Fraction(1, 8))
    root = q5_sqrt(value)
    assert root is not None and (root * root - value).is_zero()
    value = Q5(Fraction(3, 8), Fraction(-1, 8))
    root = q5_sqrt(value)
    assert root is not None and (root * root - value).is_zero() and root.sign() > 0
    assert q5_sqrt(Q5(Fraction(-1), Fraction(0))) is None
    assert q5_sqrt(Q5(Fraction(2), Fraction(0))) is None


def test_reflection_has_no_quaternion_lift() -> None:
    from oph_fpe.core.charged_response import ONE, ZERO

    reflection = [
        [-ONE, ZERO, ZERO],
        [ZERO, ONE, ZERO],
        [ZERO, ZERO, ONE],
    ]
    with pytest.raises(ChargedResponseError, match="LIFT_DETERMINANT"):
        quaternion_lift(reflection)


def test_quaternion_round_trip_pi_rotation() -> None:
    from oph_fpe.core.charged_response import ONE, ZERO

    z_pi = [
        [-ONE, ZERO, ZERO],
        [ZERO, -ONE, ZERO],
        [ZERO, ZERO, ONE],
    ]
    lift = quaternion_lift(z_pi)
    assert lift.mul(lift).key() == QUAT_MINUS_ONE.key()
    assert all(
        (quat_rotation_matrix(lift)[i][j] - z_pi[i][j]).is_zero()
        for i in range(3)
        for j in range(3)
    )
    assert quaternion_lift(quat_rotation_matrix(QUAT_ONE)).key() == QUAT_ONE.key()


def test_single_reversed_face_fails_orientation(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    face = mutated["carrier"]["oriented_faces"][0]
    mutated["carrier"]["oriented_faces"][0] = [face[0], face[2], face[1]]
    with pytest.raises(ChargedResponseError, match="FRAME_ORIENTATION"):
        produce_spin_statistics_artifact(mutated)


def test_edge_rewire_fails_closed(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    edges = mutated["carrier"]["edges"]
    (a, b), (c, d) = edges[0], edges[7]
    assert len({a, b, c, d}) == 4
    edges[0], edges[7] = [a, c], [b, d]
    with pytest.raises(ChargedResponseError):
        produce_spin_statistics_artifact(mutated)


def test_wrong_regularity_fails_closed(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    mutated["carrier"]["edges"] = [
        [f"p{i:02d}", f"p{(i + 1) % 12:02d}"] for i in range(12)
    ]
    with pytest.raises(ChargedResponseError, match="CARRIER_EDGES|CARRIER_REGULARITY"):
        produce_spin_statistics_artifact(mutated)


def test_doctored_manifest_moves_the_binding_hash(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    mutated["carrier"]["central_port_atoms"][0]["primitive"] = False
    artifact = produce_spin_statistics_artifact(mutated)
    assert (
        artifact["carrier_binding"]["carrier_manifest_sha256"]
        != PINNED_CARRIER_SHA256
    )


def test_split_lift_injection_is_impossible_in_measurement() -> None:
    # A doctored "split" lift table would need an involution lift squaring to
    # +1; the measured quaternion algebra rejects it: every unit quaternion
    # squaring to +1 is +-1, and only -1 lifts a nonidentity involution.
    from oph_fpe.core.charged_response import ONE, ZERO

    candidate = Quat(ZERO, ONE, ZERO, ZERO)
    assert candidate.mul(candidate).key() == QUAT_MINUS_ONE.key()
