"""Tests for the global form semantic artifact producer (#567)."""

import copy
import json
from pathlib import Path

import pytest

from oph_fpe.core.charged_response import ChargedResponseError
from oph_fpe.core.global_form_response import (
    _boundary_two,
    _edge_list,
    flux_tube_witness,
    produce_global_form_artifact,
    single_puncture_impossibility,
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
    return produce_global_form_artifact(manifest)


def test_artifact_binds_the_pinned_certified_carrier(artifact: dict) -> None:
    assert artifact["schema"] == "oph.global_form_semantic_artifact.v1"
    assert artifact["issue"] == 567
    binding = artifact["carrier_binding"]
    assert binding["carrier_manifest_sha256"] == PINNED_CARRIER_SHA256


def test_self_hash_recomputes(artifact: dict) -> None:
    from oph_fpe.core.charged_response import canonical_sha256

    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    assert artifact["artifact_sha256"] == "sha256:" + canonical_sha256(body)


def test_six_axis_class_group_is_measured_order_six(artifact: dict) -> None:
    axis = artifact["six_axis_class_measurement"]
    assert axis["axis_count"] == 6
    assert axis["smith_invariants"] == [1, 1, 1, 1, 1, 6]
    assert axis["class_group_order"] == 6
    assert axis["rotation_action_faithful_order"] == 60
    assert axis["rotation_action_transitive"] is True
    assert axis["antipode_reverses_every_oriented_axis"] is True


def test_federation_deck_action_upgrades_identity_only(artifact: dict) -> None:
    deck = artifact["federation_deck_action"]
    assert deck["charts"] == 12
    assert deck["seams"] == 30
    assert deck["triple_overlaps"] == 20
    assert deck["deck_group_order"] == 120
    assert deck["orientation_preserving_deck_elements"] == 60
    assert deck["orientation_reversing_deck_elements"] == 60
    assert len(deck["carrier_to_chart_isomorphism"]) == 12
    assert deck["source_incidence_sha256"].startswith("sha256:")


def test_reference_federation_is_vacuum_sector_with_verifier_binding(artifact: dict) -> None:
    sector = artifact["federation_sector_class"]
    assert sector["measured_sector_class"] == 0
    assert sector["face_holonomies_all_zero"] is True
    binding = sector["federation_verifier_binding"]
    assert binding["higher_overlap_cocycle_condition"] is True
    assert binding["nonvacuous_witness"] is True
    assert binding["bundle_sha256"].startswith("sha256:")


def test_flux_tube_menu_realizes_all_six_classes(artifact: dict) -> None:
    menu = artifact["sector_menu"]
    assert menu["class_order_source"] == "measured_six_axis_class_group"
    assert menu["realized_flux_menu"] == [0, 1, 2, 3, 4, 5]
    assert menu["puncture_faces"]["antipodal"] is True
    for witness in menu["flux_tube_witnesses"]:
        assert witness["interior_faces_flat"] is True
    impossibility = menu["single_puncture_impossibility"]
    assert impossibility["single_puncture_nonzero_flux_impossible"] is True


def test_subgroup_obstruction_menu_is_exact(artifact: dict) -> None:
    menu = artifact["sector_menu"]["subgroup_obstruction_menu"]
    assert menu["order_1"]["liftable_fluxes"] == [0]
    assert menu["order_2"]["liftable_fluxes"] == [0, 3]
    assert menu["order_3"]["liftable_fluxes"] == [0, 2, 4]
    assert menu["order_6"]["liftable_fluxes"] == [0, 1, 2, 3, 4, 5]
    assert menu["order_2"]["obstructed_fluxes"] == [1, 2, 4, 5]


def test_refined_sector_menu_is_natural(artifact: dict) -> None:
    refined = artifact["refined_sector_menu"]
    assert refined["refined_complex"] == {"vertices": 42, "seams": 120, "faces": 80}
    assert refined["refinement_natural_sector_menu"] is True
    assert refined["realized_flux_menu"] == [0, 1, 2, 3, 4, 5]


def test_gate_scope_separates_open_lanes(artifact: dict) -> None:
    gate = artifact["physical_source_gate"]
    assert gate["passed"] is True
    assert gate["laboratory_line_measurement"] is False
    assert gate["four_dimensional_instanton_attachment"] is False
    assert gate["flux_tube_sector_menu_realized"] is True
    assert gate["single_puncture_impossibility_verified"] is True


def test_deterministic_replay(manifest: dict, artifact: dict) -> None:
    replay = produce_global_form_artifact(manifest)
    assert json.dumps(replay, sort_keys=True) == json.dumps(artifact, sort_keys=True)


def test_flux_tube_witness_rejects_wrong_holonomy(manifest: dict) -> None:
    from oph_fpe.core.charged_response import load_carrier

    carrier = load_carrier(manifest)
    face_rows = carrier["faces"]
    edges = _edge_list(face_rows)
    boundary = _boundary_two(face_rows, edges)
    # Same start and end face: the prescribed holonomy c at the start face and
    # -c at the same face cancel, so a nonzero tube must fail.
    with pytest.raises((ChargedResponseError, KeyError)):
        flux_tube_witness(face_rows, edges, boundary, 0, 0, 1, 6)


def test_single_puncture_impossibility_rejects_incoherent_orientation(manifest: dict) -> None:
    from oph_fpe.core.charged_response import load_carrier

    carrier = load_carrier(manifest)
    face_rows = [list(face) for face in carrier["faces"]]
    face_rows[0] = [face_rows[0][0], face_rows[0][2], face_rows[0][1]]
    edges = _edge_list(face_rows)
    boundary = _boundary_two(face_rows, edges)
    with pytest.raises(ChargedResponseError, match="SECTOR_COMPLEX"):
        single_puncture_impossibility(boundary, len(face_rows), 6)


def test_edge_rewire_fails_closed(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    edges = mutated["carrier"]["edges"]
    (a, b), (c, d) = edges[0], edges[7]
    assert len({a, b, c, d}) == 4
    edges[0], edges[7] = [a, c], [b, d]
    with pytest.raises(ChargedResponseError):
        produce_global_form_artifact(mutated)


def test_single_reversed_face_fails_closed(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    face = mutated["carrier"]["oriented_faces"][0]
    mutated["carrier"]["oriented_faces"][0] = [face[0], face[2], face[1]]
    with pytest.raises(ChargedResponseError, match="FRAME_ORIENTATION|SECTOR_COMPLEX"):
        produce_global_form_artifact(mutated)


def test_doctored_manifest_moves_the_binding_hash(manifest: dict) -> None:
    mutated = copy.deepcopy(manifest)
    mutated["carrier"]["central_port_atoms"][0]["primitive"] = False
    artifact = produce_global_form_artifact(mutated)
    assert (
        artifact["carrier_binding"]["carrier_manifest_sha256"]
        != PINNED_CARRIER_SHA256
    )
