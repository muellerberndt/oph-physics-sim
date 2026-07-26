"""Tests for the charged response semantic artifact producer (#599)."""

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import oph_fpe.core.charged_response as response_module
from oph_fpe.core.charged_response import (
    ChargedResponseError,
    Q5,
    isotypic_projectors,
    load_carrier,
    negative_antipode_response,
    produce_charged_response_artifact,
    refinement_persistence,
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
    return produce_charged_response_artifact(manifest)


def test_artifact_binds_the_pinned_certified_carrier(artifact: dict) -> None:
    assert artifact["schema"] == "oph.charged_response_semantic_artifact.v3"
    binding = artifact["carrier_binding"]
    assert binding["carrier_manifest_sha256"] == PINNED_CARRIER_SHA256
    assert binding["port_order"][0] == "p00"
    assert len(binding["axis_representatives"]) == 6


def test_derived_construction_and_signed_scales(artifact: dict) -> None:
    derived = artifact["derived"]
    assert derived["construction"] == "charged_double_triplet"
    assert (
        derived["construction_provenance"]
        == "canonical_equivariant_compact_lift_from_the_impulse_readback_"
        "derived_antipode_response"
    )
    assert derived["response_band_scales"] == {
        "unit_band": "-1",
        "quintet_band": "-1",
        "frame_band": "1",
        "kernel_band": "1",
    }


def test_sector_structure_and_galois_pairing(artifact: dict) -> None:
    basis = artifact["response_basis"]
    assert basis["sector_dimensions"] == {
        "unit_band": 1,
        "quintet_band": 5,
        "frame_band": 3,
        "kernel_band": 3,
    }
    assert basis["adjacency_channel_values"]["frame_band"] == "0 + 1*sqrt(5)"
    assert basis["adjacency_channel_values"]["kernel_band"] == "0 + -1*sqrt(5)"
    assert basis["galois_pairing"]["frame_and_kernel_swapped_by_conjugation"] is True


def test_incidence_derived_response(artifact: dict) -> None:
    response = artifact["source_response"]
    assert response["operator"] == "negative_graph_antipode_involution"
    assert response["source"] == "target_blind_maximal_distance_impulse_readback"
    assert response["unique_nonidentity_central_involution"] is True
    assert response["central_involution_count_including_identity"] == 2
    assert response["commutes_with_propagation_generator"] is True
    assert response["self_adjoint_unitary_involution"] is True
    assert response["impulse_readback_response_executed"] is True
    assert response["physical_perturb_readback_source_bound"] is True
    assert (
        response["antipode_polynomial_identity"]
        == "J = (A^3 - 4*A^2 - 5*A + 10*I)/10"
    )
    protocol = response["impulse_readback_protocol"]
    assert protocol["status"] == "source_bound_operational_producer"
    assert protocol["homogeneous_filter_coefficients"] == [
        "1",
        "-1/2",
        "-2/5",
        "1/10",
    ]
    assert protocol["unique_solution_rank"] == 4
    assert protocol["target_labels_used"] is False
    assert response["sector_eigenvalues"] == {
        "unit_band": -1,
        "quintet_band": -1,
        "frame_band": 1,
        "kernel_band": 1,
    }
    audits = artifact["structural_audits"]
    assert (
        audits["frame_normalization"]["unit_channel_constant"]
        == "10 + 2*sqrt(5)"
    )
    assert (
        audits["rotation_automorphisms"]["incidence_automorphism_group_order"]
        == 120
    )
    assert (
        audits["rotation_automorphisms"]["commuting_incidence_automorphisms"]
        == 60
    )


def test_runtime_binding_reaches_the_propagated_dynamics(artifact: dict) -> None:
    runtime = artifact["provenance"]["runtime_binding"]
    assert runtime["spectrum_multiplicities"] == [1, 3, 3, 5]
    assert runtime["equivariance_receipt"] is True
    assert runtime["charged_response_operator_receipt"] is True
    assert sorted(runtime["manifest_to_runtime_port_map"]) == list(range(12))


def test_physical_refinement_maps_are_defect_port_persistence(artifact: dict) -> None:
    maps = artifact["physical_refinement_maps"]["port_persistence_maps"]
    assert len(maps) == 2
    for row in maps:
        assert sorted(row["port_map"]) == list(range(12))
        assert row["origin"] == "defect_port_persistence_in_geodesic_tower"
        assert row["map_hash"].startswith("sha256:")


def test_artifact_is_deterministic(manifest: dict) -> None:
    first = produce_charged_response_artifact(manifest)
    second = produce_charged_response_artifact(manifest)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_wrong_regularity_fails_closed(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    mutated["carrier"]["edges"] = [
        [f"p{i:02d}", f"p{(i + 1) % 12:02d}"] for i in range(12)
    ]
    with pytest.raises(ChargedResponseError, match="CARRIER_EDGES|CARRIER_REGULARITY"):
        produce_charged_response_artifact(mutated)


def test_bipartite_five_regular_spectrum_fails_closed() -> None:
    # K_{6,6} minus a perfect matching is five-regular with a rational
    # spectrum, so no Galois-paired charged sector structure exists.
    adjacency = [[0] * 12 for _ in range(12)]
    for left in range(6):
        for right in range(6):
            if left != right:
                adjacency[left][6 + right] = adjacency[6 + right][left] = 1
    with pytest.raises(ChargedResponseError, match="SECTOR_PROJECTOR|SECTOR_DIMENSION"):
        isotypic_projectors(adjacency)


def test_single_reversed_face_fails_orientation(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    face = mutated["carrier"]["oriented_faces"][0]
    mutated["carrier"]["oriented_faces"][0] = [face[0], face[2], face[1]]
    with pytest.raises(ChargedResponseError, match="FRAME_ORIENTATION"):
        produce_charged_response_artifact(mutated)


def test_doctored_antipode_response_mapping_fails_closed(manifest: dict) -> None:
    carrier = load_carrier(manifest)
    projectors = isotypic_projectors(carrier["adjacency"])
    carrier["antipode"] = list(carrier["antipode"])
    carrier["antipode"][0] = 0
    with pytest.raises(ChargedResponseError, match="IMPULSE_READBACK"):
        negative_antipode_response(carrier, projectors)


def test_runtime_response_law_mismatch_fails_closed(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        response_module,
        "reference_negative_antipode_response",
        lambda: np.eye(12, dtype=np.complex128),
    )
    with pytest.raises(ChargedResponseError, match="RUNTIME_RESPONSE"):
        produce_charged_response_artifact(manifest)


def test_edge_rewire_fails_closed(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    edges = mutated["carrier"]["edges"]
    (a, b), (c, d) = edges[0], edges[7]
    assert len({a, b, c, d}) == 4
    edges[0], edges[7] = [a, c], [b, d]
    with pytest.raises(ChargedResponseError):
        produce_charged_response_artifact(mutated)


def test_doctored_atoms_do_not_change_the_binding_scope(manifest: dict) -> None:
    # The binding hash covers the whole manifest, so any edit moves the hash.
    mutated = copy.deepcopy(manifest)
    mutated["carrier"]["central_port_atoms"][0]["primitive"] = False
    artifact = produce_charged_response_artifact(mutated)
    assert (
        artifact["carrier_binding"]["carrier_manifest_sha256"]
        != PINNED_CARRIER_SHA256
    )


def test_q5_sign_and_render() -> None:
    assert Q5.of(Fraction(-1), Fraction(1)).sign() == 1
    assert Q5.of(Fraction(3), Fraction(-1)).sign() == 1
    assert Q5.of(Fraction(-3), Fraction(1)).sign() == -1
    assert Q5.of(5, 1).render() == "5 + 1*sqrt(5)"
    assert Q5.of(Fraction(1, 4)).render() == "1/4"


def test_refinement_persistence_is_measured_not_assumed() -> None:
    result = refinement_persistence()
    assert result["per_level_defect_port_count"] == 12
    assert [row["target_level"] for row in result["port_persistence_maps"]] == [1, 2]
