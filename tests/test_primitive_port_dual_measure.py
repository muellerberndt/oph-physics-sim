from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import primitive_port_dual_measure as producer
from oph_fpe.dynamics.verify_primitive_port_dual_measure_independent import (
    IndependentVerificationError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/primitive_port_dual_measure_receipt.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("receipt_sha256", None)
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def _write_mutation(tmp_path: Path, report: dict, name: str = "mutated.json") -> Path:
    _rehash(report)
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def _set_path(report: dict, path: tuple[object, ...], value: object) -> None:
    cursor: object = report
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


@pytest.fixture(scope="module")
def canonical() -> dict:
    return producer.load_receipt_strict(RECEIPT)


def test_canonical_receipt_replays_exactly(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)
    assert producer.verify_receipt(canonical)["receipt"] is True


def test_exact_base_measure_is_one_twelfth(canonical: dict) -> None:
    packet = canonical["exact_base_port_dual_measure"]
    assert packet["partition_type"] == (
        "barycentric_vertex_dual_partition_on_base_face_cells"
    )
    assert packet["disjoint_characteristic_cells_claimed"] is False
    assert packet["voronoi_cell_identity_claimed"] is False
    assert packet["port_orbit_size"] == 12
    assert packet["face_orbit_size"] == 20
    assert packet["edge_orbit_size"] == 30
    assert packet["exact_identity"] == "5*(1/3)*(1/20)=1/12"
    assert packet["exact_normalized_measure_per_port"] == "1/12"
    assert len(packet["port_rows"]) == 12
    assert all(row["incident_face_count"] == 5 for row in packet["port_rows"])
    assert all(row["exact_normalized_measure"] == "1/12" for row in packet["port_rows"])
    assert packet["PORT_DUAL_NORMALIZED_MEASURE_EXACT"] is True


def test_declared_refinement_levels_retain_measure(canonical: dict) -> None:
    refinement = canonical["refinement_naturality"]
    assert refinement["declared_levels"] == [0, 1, 2]
    rows = refinement["level_rows"]
    assert [row["face_count"] for row in rows] == [20, 80, 320]
    assert [row["descendants_per_base_face"] for row in rows] == [1, 4, 16]
    for row in rows:
        assert row["child_to_base_lineage_exact"] is True
        assert row["normalized_area_pushforward_gate_passed"] is True
        assert row["partition_of_unity_gate_passed"] is True
        assert row["all_port_mass_1_over_12_gate_passed"] is True
        assert row["analytic_measure_consequence_if_exact_area_partition"] == "1/12"
        assert row["exact_refined_spherical_areas_machine_proved"] is False
    assert refinement["symbolic_exact_refined_spherical_area_proof_present"] is False
    assert refinement["REFINEMENT_NATURAL_PORT_DUAL_MEASURE_RECEIPT"] is True


def test_all_attachment_controls_are_retained_and_unselected(canonical: dict) -> None:
    controls = canonical["attachment_controls"]
    natural = controls["natural_orbit_controls"]
    assert natural["port_orbit"]["conditional_kappa_geom"] == "3/pi"
    assert natural["face_orbit"]["conditional_kappa_geom"] == "5/pi"
    assert natural["edge_orbit"]["conditional_kappa_geom"] == "15/(2*pi)"
    assert natural["whole_shell_single_cell"]["conditional_kappa_geom"] == "1/(4*pi)"
    assert [
        row["conditional_kappa_geom"]
        for row in controls["refinement_stage_controls"]
    ] == ["5/pi", "20/pi", "80/pi"]
    assert controls["no_attachment_selected"] is True
    assert controls["controls_fail_closed"] is True


def test_physical_and_comparison_boundaries_remain_false(canonical: dict) -> None:
    assert canonical["comparison_data_read"] is False
    assert canonical["target_data_read"] is False
    assert canonical["issue_662_armed"] is False
    attainment = canonical["attainment"]
    for key in (
        "physical_P_pixel_is_primitive_port_sector",
        "support_areal_radius_is_issue_655_translation_hop",
        "terminal_physical_refinement_stage_selected",
        "kappa_geom_source_selected",
        "positive_carrier_lower_bound_promoted",
        "physical_prediction_promoted",
        "comparison_permitted",
        "issue_664_closure_supported",
        "issue_662_armed",
    ):
        assert attainment[key] is False
    boundary = canonical["epistemic_boundary"]
    assert boundary["comparison_data_read"] is False
    assert boundary["target_data_read"] is False
    assert boundary["target_data_paths"] == []
    assert boundary["measured_P_value_read"] is False
    assert boundary["physical_pixel_area_value_read"] is False
    assert boundary["shared_geometry_implies_physical_identity"] is False


def test_independent_verifier_reconstructs_packet() -> None:
    result = verify_independent(RECEIPT)
    assert result["receipt"] is True
    assert result["producer_imported"] is False
    assert result["source_federation_implementation_imported"] is False
    assert result["exact_base_rational_measure_independently_reimplemented"] is True
    assert result["symbolic_exact_refined_spherical_area_proof_present"] is False
    assert result["finite_refinement_floating_replay_passed"] is True
    assert result["checked_ports"] == 12
    assert result["checked_proper_actions"] == 60
    assert result["checked_refinement_levels"] == 3
    assert result["comparison_data_read"] is False
    assert result["physical_pixel_identity"] is False
    assert result["physical_hop_identity"] is False
    assert result["issue_662_armed"] is False


def test_independent_verifier_imports_neither_producer_nor_federation() -> None:
    path = (
        ROOT
        / "oph_fpe/dynamics/verify_primitive_port_dual_measure_independent.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.primitive_port_dual_measure" not in imported
    assert "oph_fpe.core.echosahedral_federation" not in imported


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("exact_base_port_dual_measure", "exact_normalized_measure_per_port"), "1/11"),
        (("exact_base_port_dual_measure", "barycentric_weight_per_incident_port"), "1/2"),
        (("exact_base_port_dual_measure", "port_rows", 0, "incident_face_count"), 4),
        (("refinement_naturality", "level_rows", 2, "analytic_measure_consequence_if_exact_area_partition"), "1/11"),
        (("attachment_controls", "natural_orbit_controls", "face_orbit", "conditional_kappa_geom"), "3/pi"),
        (("attachment_controls", "natural_orbit_controls", "edge_orbit", "conditional_kappa_geom"), "3/pi"),
        (("attachment_controls", "natural_orbit_controls", "whole_shell_single_cell", "conditional_kappa_geom"), "3/pi"),
        (("attachment_controls", "refinement_stage_controls", 1, "orbit_cardinality"), 12),
        (("attainment", "physical_P_pixel_is_primitive_port_sector"), True),
        (("attainment", "support_areal_radius_is_issue_655_translation_hop"), True),
        (("attainment", "kappa_geom_source_selected"), True),
        (("attainment", "issue_662_armed"), True),
        (("epistemic_boundary", "comparison_data_read"), True),
        (("epistemic_boundary", "target_data_paths"), ["forbidden-target-path"]),
        (("source_scope", "source_geometry_hash"), "0" * 64),
    ],
)
def test_semantic_and_numeric_mutations_fail_closed(
    tmp_path: Path,
    canonical: dict,
    path: tuple[object, ...],
    value: object,
) -> None:
    changed = copy.deepcopy(canonical)
    _set_path(changed, path, value)
    mutated = _write_mutation(tmp_path, changed)
    with pytest.raises(IndependentVerificationError):
        verify_independent(mutated)
    with pytest.raises(producer.PrimitivePortDualMeasureError):
        producer.load_receipt_strict(mutated)


def test_unregistered_target_field_fails_closed(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["observed_target"] = 1.0
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_implementation_pin_removal_fails_closed(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["implementation_pins"].pop()
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_duplicate_json_key_fails_both_strict_loaders(
    tmp_path: Path, canonical: dict
) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    duplicated = rendered[:-1] + ', "issue": 662}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicated, encoding="utf-8")
    with pytest.raises(IndependentVerificationError, match="duplicate JSON key"):
        verify_independent(path)
    with pytest.raises(producer.PrimitivePortDualMeasureError, match="duplicate JSON key"):
        producer.load_receipt_strict(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_fail_both_strict_loaders(
    tmp_path: Path,
    canonical: dict,
    constant: str,
) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    malformed = rendered[:-1] + f', "nonfinite_control": {constant}}}'
    path = tmp_path / f"nonfinite-{constant.replace('-', 'minus')}.json"
    path.write_text(malformed, encoding="utf-8")
    with pytest.raises(IndependentVerificationError, match="non-finite JSON constant"):
        verify_independent(path)
    with pytest.raises(
        producer.PrimitivePortDualMeasureError,
        match="non-finite JSON constant",
    ):
        producer.load_receipt_strict(path)


def test_independent_verifier_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_primitive_port_dual_measure_independent",
            str(RECEIPT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["receipt"] is True
