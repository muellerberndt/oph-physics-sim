from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics import fz11_3d_translation_bridge as producer
from oph_fpe.dynamics.verify_fz11_3d_translation_bridge_independent import (
    IndependentVerificationError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("receipt_sha256", None)
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def _write_mutation(tmp_path: Path, report: dict) -> Path:
    _rehash(report)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def canonical() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_receipt_replays_byte_exactly(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)


def test_producer_verifier_passes(canonical: dict) -> None:
    result = producer.verify_receipt(canonical)
    assert result["receipt"] is True
    assert result["status"] == "PASS"


def test_independent_verifier_reconstructs_bridge() -> None:
    result = verify_independent(RECEIPT)
    assert result["receipt"] is True
    assert result["producer_imported"] is False
    assert result["exact_Qsqrt5_frame_independently_reimplemented"] is True
    assert result["checked_scaled_gram_entries"] == 144
    assert result["checked_A5_actions"] == 60
    assert result["checked_source_events"] == 12
    assert result["checked_impulse_sites"] == 13
    assert result["checked_plane_wave_samples"] == 3
    assert result["comparison_data_read"] is False
    assert result["issue_662_armed"] is False


def test_scope_is_conditional_and_non_lattice(canonical: dict) -> None:
    adapter = canonical["conditional_R3_translation_adapter"]
    assert adapter["integer_module_map_is_injective"] is True
    assert adapter["integer_injection_certificate"]["determinant"] == "-8"
    assert adapter["image_density_certificate"]["raw_image_index_in_Z_phi_cubed"] == 8
    assert adapter["image_is_dense_index_8_submodule_of_scaled_Z_phi_cubed"] is True
    assert adapter["image_is_locally_finite_spatial_lattice_or_quasicrystal"] is False
    assert adapter["observer_modulo_three_quotient_factors_this_R3_map"] is False
    assert adapter["continuous_R3_field_is_an_auxiliary_input"] is True
    attainment = canonical["attainment"]
    assert attainment["exact_FZ11_cosine_symbol"] is True
    assert attainment["canonical_source_selection"] is False
    assert attainment["time_evolution_derived"] is False
    assert attainment["photon_sector_selected"] is False
    assert attainment["finite_scale_selected"] is False
    assert attainment["carrier_frame_selected"] is False
    assert attainment["boost_law_derived"] is False
    assert attainment["exclusivity_proved"] is False
    assert attainment["physical_prediction_promoted"] is False
    assert attainment["issue_662_armed"] is False


def test_source_parent_pin_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["parent_pins"]["constructive_source_receipt"]["raw_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_frozen_prediction_pin_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["parent_pins"]["frozen_FZ11_prediction"]["exact_prediction_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_relabel_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    row = changed["exact_port_frame_and_relabel"]["source_to_RER_PortFrameGram_label"]
    row[0], row[1] = row[1], row[0]
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_exact_gram_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["exact_port_frame_and_relabel"]["source_scaled_gram_5G_qsqrt5_integer_pairs"][0][1] = [0, -1]
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_A5_conjugacy_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["exact_port_frame_and_relabel"]["RER_conjugated_A5_action_sha256"] = "sha256:" + "f" * 64
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_source_event_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["conditional_R3_translation_adapter"]["port_rows"][0]["source_event_id"] = "sha256:" + "1" * 64
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_source_direction_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["conditional_R3_translation_adapter"]["port_rows"][0]["source_Z6_direction"][0] = 0
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize("key", ["determinant", "rank"])
def test_integer_injection_mutation_is_rejected(
    tmp_path: Path, canonical: dict, key: str
) -> None:
    changed = copy.deepcopy(canonical)
    certificate = changed["conditional_R3_translation_adapter"]["integer_injection_certificate"]
    certificate[key] = "0" if key == "determinant" else 5
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_dense_image_certificate_mutation_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["conditional_R3_translation_adapter"]["image_density_certificate"][
        "raw_image_index_in_Z_phi_cubed"
    ] = 4
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize(
    "key",
    [
        "image_is_locally_finite_spatial_lattice_or_quasicrystal",
        "observer_modulo_three_quotient_factors_this_R3_map",
        "adapter_selected_by_canonical_repair_dynamics",
    ],
)
def test_adapter_promotion_is_rejected(
    tmp_path: Path, canonical: dict, key: str
) -> None:
    changed = copy.deepcopy(canonical)
    changed["conditional_R3_translation_adapter"][key] = True
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("domain", "physical spacetime lattice"),
        ("codomain", "the photon field"),
        ("map", "target-calibrated translation"),
        ("quotient_nonfactorization_reason", "the finite quotient is faithful"),
    ],
)
def test_adapter_semantic_mutation_is_rejected(
    tmp_path: Path, canonical: dict, key: str, value: str
) -> None:
    changed = copy.deepcopy(canonical)
    changed["conditional_R3_translation_adapter"][key] = value
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_operator_normalization_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["operator_contract"]["resulting_weight_per_port"] = "1/a^2"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_operator_semantic_string_mutation_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["operator_contract"]["cosine_symbol"] = "omega^2=k^2"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_coefficient_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["exact_expansion_certificate"]["coefficients"]["C4_over_a2"] = "1/20"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_group_velocity_scope_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["exact_expansion_certificate"]["group_velocity_scope"] = "full vector gradient"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_full_vector_boundary_mutation_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["exact_expansion_certificate"]["full_vector_group_velocity_boundary"] = (
        "the radial derivative is the complete vector velocity"
    )
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_impulse_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["exact_impulse_replay"]["output_support_at_a_equals_1"][0]["coefficient_at_a_equals_1"] = "5"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_plane_wave_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["target_free_plane_wave_replay"]["sample_rows"][0]["cosine_symbol_10dp"] = "0.0000000000"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_unregistered_target_field_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["operator_contract"]["observed_target"] = 137.036
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_boundary", "This is a physical photon prediction."),
        ("open_premises", ["none"]),
    ],
)
def test_semantic_boundary_mutation_is_rejected(
    tmp_path: Path, canonical: dict, field: str, value: object
) -> None:
    changed = copy.deepcopy(canonical)
    changed[field] = value
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_duplicate_json_key_is_rejected(tmp_path: Path, canonical: dict) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    duplicate = rendered[:-1] + ', "issue_662_armed": true}'
    path = tmp_path / "duplicate-key.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(IndependentVerificationError, match="duplicate JSON key"):
        verify_independent(path)


@pytest.mark.parametrize(
    "key",
    [
        "canonical_source_selection",
        "canonical_A1_A2_A3_derivation",
        "observer_quotient_spatial_readout",
        "faithful_finite_Q_translation_action",
        "spatial_site_lattice_or_quasicrystal",
        "time_evolution_derived",
        "finite_scale_selected",
        "carrier_frame_selected",
        "photon_sector_selected",
        "exclusivity_proved",
        "physical_sector_selected",
        "boost_law_derived",
        "canonical_physical_readout",
        "physical_prediction_promoted",
        "comparison_permitted",
        "issue_655_closure_supported",
        "issue_662_armed",
    ],
)
def test_attainment_promotion_is_rejected(
    tmp_path: Path, canonical: dict, key: str
) -> None:
    changed = copy.deepcopy(canonical)
    changed["attainment"][key] = True
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_top_level_comparison_and_arming_mutations_are_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    for key in ("comparison_data_read", "issue_662_armed"):
        changed = copy.deepcopy(canonical)
        changed[key] = True
        with pytest.raises(IndependentVerificationError):
            verify_independent(_write_mutation(tmp_path, changed))


def test_implementation_pin_omission_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["implementation_pins"].pop()
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))
