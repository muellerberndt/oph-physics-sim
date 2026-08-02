from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics import vertex12_constructive_source_law as producer
from oph_fpe.dynamics.verify_vertex12_constructive_source_law_independent import (
    IndependentVerificationError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/vertex12_constructive_source_law_receipt.json"


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
    report["receipt_sha256"] = (
        "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    )


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


def test_independent_verifier_reconstructs_complete_packet() -> None:
    result = verify_independent(RECEIPT)
    assert result["receipt"] is True
    assert result["producer_imported"] is False
    assert result["source_engine_independently_reimplemented"] is False
    assert result["quotient_algebra_independently_reimplemented"] is True
    assert result["checked_meaning_states"] == 729
    assert result["checked_source_events"] == 12
    assert result["checked_descent_squares"] == 12
    assert result["checked_inverse_rows"] == 6
    assert result["checked_endpoint_diamond_rows"] == 15
    assert result["checked_A5_covariance_rows"] == 720


def test_claim_boundary_keeps_packet_as_control(canonical: dict) -> None:
    attainment = canonical["attainment"]
    assert attainment["accepted_surjective_quotient"] is True
    assert attainment["fifteen_complete_all_state_endpoint_diamonds"] is True
    assert attainment["same_Q_A5_covariance"] is True
    assert attainment["canonical_source_selection"] is False
    assert attainment["spatial_translation"] is False
    assert attainment["physical_readout"] is False
    assert attainment["physical_prediction"] is False
    disposition = canonical["issue_655_disposition"]
    assert disposition["classification"] == "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ONLY"
    assert disposition["advances_canonical_source_bridge"] is False
    assert disposition["advances_physical_bridge"] is False
    assert disposition["issue_closure_supported"] is False


def test_capture_root_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["source_capture"][
        "source_capture_root_sha256"
    ] = "sha256:" + "0" * 64
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_post_capture_event_assignment_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    event = changed["constructive_source_law"]["a1_complete_event_alphabet"][
        "event_rows"
    ][0]
    event["emission_phase"] = "post_capture_assignment"
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_event_direction_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["a1_complete_event_alphabet"]["event_rows"][0][
        "raw_direction_in_Z_power_6"
    ][0] = 0
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_surjectivity_witness_tamper_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["a2_interpretation"]["surjectivity_witnesses"][
        1
    ]["raw_integer_representative"] = [0] * 6
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_raw_step_direction_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["raw_step_rows"][0]["direction"][0] = 2
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_meaning_step_table_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    table = changed["constructive_source_law"]["meaning_step_rows"][0]["permutation"]
    table[0], table[1] = table[1], table[0]
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_descent_square_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    endpoints = changed["constructive_source_law"]["a2_descent_rows"][0][
        "complete_residue_endpoint_indices"
    ]
    endpoints[0], endpoints[1] = endpoints[1], endpoints[0]
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_inverse_table_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["antipodal_inverse_rows"][0][
        "forward_then_reverse_endpoints"
    ][0] = 1
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_endpoint_diamond_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["positive_axis_endpoint_diamond_rows"][0][
        "right_after_left_endpoints"
    ][0] += 1
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_A5_action_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    row = changed["constructive_source_law"]["same_Q_A5_action"]["group_rows"][0][
        "meaning_permutation"
    ]
    row[0], row[1] = row[1], row[0]
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_A5_covariance_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["same_Q_A5_action"]["covariance_rows"][0][
        "exact"
    ] = False
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_A3_reference_tamper_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["a3_counting_reference"]["weight_per_event"] = (
        "1/11"
    )
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_implementation_pin_omission_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["implementation_pins"].pop()
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_capture_contract_tamper_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    capture = changed["constructive_source_law"]["source_capture"]
    capture["payload"]["a2_contract"] = "arbitrary quotient"
    capture["source_capture_root_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(capture["payload"])
    ).hexdigest()
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_unregistered_target_field_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["constructive_source_law"]["a2_interpretation"][
        "target_value"
    ] = 137.036
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("attainment", "canonical_source_selection"),
        ("attainment", "spatial_translation"),
        ("attainment", "physical_readout"),
        ("attainment", "physical_prediction"),
        ("provenance_boundary", "canonical_A1_A2_A3_derivation_claimed"),
        ("provenance_boundary", "full_canonical_A1_typed_object_instantiated"),
        (
            "provenance_boundary",
            "full_A2_observer_federation_functor_instantiated",
        ),
        (
            "provenance_boundary",
            "canonical_A3_maximum_entropy_selection_instantiated",
        ),
        ("provenance_boundary", "physical_attachment_claimed"),
        ("provenance_boundary", "comparison_or_target_data_used"),
    ],
)
def test_unsupported_promotion_is_rejected(
    tmp_path: Path, canonical: dict, section: str, key: str
) -> None:
    changed = copy.deepcopy(canonical)
    changed[section][key] = True
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))
