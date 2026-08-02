from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import port_gram_equivariant_action as producer
from oph_fpe.dynamics.verify_port_gram_equivariant_action_independent import (
    IndependentEquivariantActionError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/port_gram_equivariant_action_receipt.json"


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


def _write_mutation(tmp_path: Path, report: dict, name: str) -> Path:
    _rehash(report)
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def canonical() -> dict:
    return producer.load_json_strict(RECEIPT)


def test_canonical_receipt_replays_exactly(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)
    assert producer.verify_receipt(canonical) == {
        "receipt": True,
        "faithful_completion_action": True,
        "finite_recharting_cocycle": True,
        "cofinal_gluing": False,
        "physical_action": False,
        "comparison_permitted": False,
    }


def test_proper_carrier_action_is_exact_and_faithful(canonical: dict) -> None:
    action = canonical["exact_proper_carrier_action"]
    assert action["full_incidence_automorphism_count"] == 120
    assert action["oriented_proper_automorphism_count"] == 60
    assert action["element_order_histogram"] == {
        "1": 1,
        "2": 15,
        "3": 20,
        "5": 24,
    }
    assert action["presentation"] == "<s,t | s^2=t^3=(s*t)^5=1>"
    assert action["product_order"] == 5
    assert action["generated_subgroup_order"] == 60
    assert action["abstract_group_identification"] == (
        "A5 (proper icosahedral rotations)"
    )
    assert action["orientation_field_is_load_bearing"] is True
    assert action["removing_orientation_retains_full_order_120_group"] is True

    completion = canonical["exact_completion_action"]
    assert completion["all_proper_maps_preserve_selected_Gram"] is True
    assert completion["all_proper_maps_commute_with_repair_incidence"] is True
    assert completion["quotient_action_kernel_size"] == 1
    assert completion["quotient_action_faithful"] is True
    assert completion["all_proper_maps_preserve_antipodal_relations"] is True
    assert completion["signed_integral_action_count"] == 60
    assert completion["signed_integral_action_faithful"] is True
    assert completion["signed_integral_determinant_values"] == [1]
    assert completion["signed_action_composition_exact"] is True
    assert (
        completion[
            "dense_module_isometries_extend_uniquely_to_metric_completion"
        ]
        is True
    )
    assert completion["completion_action_faithful"] is True


def test_declared_tower_is_a_finite_recharting_cocycle(canonical: dict) -> None:
    packet = canonical["declared_finite_tower_cocycle"]
    assert len(packet["map_rows"]) == 3
    assert all(row["proper_carrier_action_member"] for row in packet["map_rows"])
    assert all(row["selected_Gram_intertwined_exactly"] for row in packet["map_rows"])
    assert all(row["repair_incidence_intertwined_exactly"] for row in packet["map_rows"])
    assert packet["direct_r0_r2_equals_r1_r2_after_r0_r1"] is True
    assert packet["signed_module_cocycle_exact"] is True
    assert packet["completion_isometry_cocycle_exact"] is True
    assert packet["finite_recharting_naturality_attained"] is True
    assert packet["maps_add_new_carrier_degrees_of_freedom"] is False
    assert packet["scale_refinement_semigroup_proved"] is False
    assert packet["cofinal_refinement_family_proved"] is False
    assert packet["overlap_atlas_gluing_proved"] is False
    assert packet["global_carrier_gluing_proved"] is False


def test_source_physical_and_comparison_boundaries_remain_closed(
    canonical: dict,
) -> None:
    assert canonical["target_data_read"] is False
    assert canonical["comparison_data_read"] is False
    completion = canonical["exact_completion_action"]
    assert completion["extension_is_conditional_on_parent_completion_premises"] is True
    assert completion["source_native_physical_action_promoted"] is False
    attainment = canonical["attainment"]
    assert attainment["conditional_faithful_A5_completion_action_certified"] is True
    assert attainment["finite_declared_tower_recharting_cocycle_certified"] is True
    for key in (
        "parent_completion_premises_discharged",
        "canonical_signed_record_source_selected",
        "A2_operational_position_topology_selected",
        "scale_refinement_naturality_proved",
        "cofinal_overlap_refinement_gluing_proved",
        "global_space_promoted",
        "physical_action_promoted",
        "physical_prediction_promoted",
        "comparison_permitted",
    ):
        assert attainment[key] is False
    assert "finite recharting naturality" in canonical["claim_boundary"]
    assert "rather than scale refinement" in canonical["claim_boundary"]
    assert "No global or physical space" in canonical["claim_boundary"]


def test_independent_verifier_reconstructs_action_and_cocycle() -> None:
    assert verify_independent(RECEIPT) == {
        "receipt": True,
        "producer_imported": False,
        "automorphisms_independently_enumerated": True,
        "faithful_completion_action": True,
        "finite_recharting_cocycle": True,
        "cofinal_gluing": False,
        "physical_action": False,
        "comparison_permitted": False,
    }


def test_independent_verifier_does_not_import_producer() -> None:
    tree = ast.parse(
        (
            ROOT
            / "oph_fpe/dynamics/verify_port_gram_equivariant_action_independent.py"
        ).read_text(encoding="utf-8")
    )
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.port_gram_equivariant_action" not in imported


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("exact_completion_action", "quotient_action_kernel_size"), 2),
        (("exact_completion_action", "source_native_physical_action_promoted"), True),
        (("declared_finite_tower_cocycle", "direct_r0_r2_equals_r1_r2_after_r0_r1"), False),
        (("declared_finite_tower_cocycle", "cofinal_refinement_family_proved"), True),
        (("attainment", "physical_action_promoted"), True),
        (("attainment", "comparison_permitted"), True),
    ],
)
def test_independent_verifier_rejects_rehashed_semantic_mutations(
    tmp_path: Path,
    canonical: dict,
    path: tuple[str, str],
    value: object,
) -> None:
    report = copy.deepcopy(canonical)
    report[path[0]][path[1]] = value
    mutated = _write_mutation(tmp_path, report, f"{path[1]}.json")
    with pytest.raises(IndependentEquivariantActionError):
        verify_independent(mutated)


def test_independent_verifier_rejects_mutated_group_generator(
    tmp_path: Path, canonical: dict
) -> None:
    report = copy.deepcopy(canonical)
    report["exact_proper_carrier_action"]["order_two_generator"] = list(range(12))
    mutated = _write_mutation(tmp_path, report, "bad-generator.json")
    with pytest.raises(IndependentEquivariantActionError):
        verify_independent(mutated)


def test_independent_verifier_rejects_mutated_tower_row(
    tmp_path: Path, canonical: dict
) -> None:
    report = copy.deepcopy(canonical)
    row = report["declared_finite_tower_cocycle"]["map_rows"][0]
    row["port_permutation"][0], row["port_permutation"][1] = (
        row["port_permutation"][1],
        row["port_permutation"][0],
    )
    mutated = _write_mutation(tmp_path, report, "bad-tower.json")
    with pytest.raises(IndependentEquivariantActionError):
        verify_independent(mutated)


def test_producer_and_independent_cli_validate() -> None:
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    producer_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.port_gram_equivariant_action",
            "--verify",
            str(RECEIPT),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PORT_GRAM_EQUIVARIANT_ACTION_VALID" in producer_run.stdout
    verifier_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_port_gram_equivariant_action_independent",
            str(RECEIPT),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PORT_GRAM_EQUIVARIANT_ACTION_INDEPENDENT_VALID" in verifier_run.stdout
