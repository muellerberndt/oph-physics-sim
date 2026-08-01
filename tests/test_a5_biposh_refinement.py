from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.cosmology.a5_biposh_refinement import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_RECEIPT,
    build_a5_biposh_dual_operator_packet,
)
from oph_fpe.cosmology.verify_a5_biposh_refinement_independent import (
    VerificationError,
    verify_packet,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_mutation(
    tmp_path: Path,
    receipt: dict,
    coefficients: dict,
) -> tuple[Path, Path]:
    coefficient_path = tmp_path / "coefficients.json"
    receipt_path = tmp_path / "receipt.json"
    coefficient_payload = _canonical_bytes(coefficients)
    receipt["full_coefficient_bundle"]["path"] = coefficient_path.name
    receipt["full_coefficient_bundle"]["bytes"] = len(coefficient_payload)
    receipt["full_coefficient_bundle"]["sha256"] = (
        "sha256:" + hashlib.sha256(coefficient_payload).hexdigest()
    )
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = (
        "sha256:" + hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    )
    coefficient_path.write_bytes(coefficient_payload)
    receipt_path.write_bytes(_canonical_bytes(receipt))
    return receipt_path, coefficient_path


@pytest.fixture(scope="module")
def canonical() -> tuple[dict, dict]:
    return (
        json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8")),
        json.loads(DEFAULT_COEFFICIENTS.read_text(encoding="utf-8")),
    )


def test_canonical_packet_passes_independent_replay() -> None:
    result = verify_packet()
    assert result["status"] == "PASS"
    assert result["checked_operator_cases"] == 8
    assert result["checked_full_biposh_coefficients"] == 8 * 5929


def test_rank_controls_prevent_silent_ell8_truncation(canonical: tuple[dict, dict]) -> None:
    receipt, _ = canonical
    rows = receipt["level_rows"]
    assert [row["sampled_harmonic_design_rank"] for row in rows[:2]] == [12, 42]
    assert all(row["calculation_performed"] is False for row in rows[:2])
    assert all(row["calculation_performed"] is True for row in rows[2:])
    assert rows[2]["sampled_harmonic_design_rank"] == 81


def test_primary_operator_fingerprint_is_frozen_without_physical_promotion(
    canonical: tuple[dict, dict],
) -> None:
    receipt, _ = canonical
    by_operator = {
        row["operator_id"]: row for row in receipt["refinement_diagnostics"]
    }
    graph = [row["value"] for row in by_operator[
        "equal_seam_raw_graph_laplacian"
    ]["primary_values"]]
    cotangent = [row["value"] for row in by_operator[
        "geometric_cotangent_control"
    ]["primary_values"]]
    assert graph == pytest.approx(
        [
            0.031459170981830153,
            0.03312791041666894,
            0.033406204659135574,
            0.03346810635836158,
        ],
        abs=2.0e-12,
    )
    assert cotangent == pytest.approx(
        [
            0.0015640997923285105,
            0.000325384756368263,
            0.00007767664600210185,
            0.00001919596731704102,
        ],
        abs=2.0e-12,
    )
    assert all(
        cotangent[index + 1] < cotangent[index] / 3.0
        for index in range(len(cotangent) - 1)
    )
    assert receipt["selection_decision"] == {
        "base_equal_seam_operator_bounded_reconstructed": True,
        "continuum_residual_decided": False,
        "equal_seam_operator_source_selected": False,
        "global_frame_quotient_visible": False,
        "physical_covariance_selected": False,
        "physical_prediction": False,
        "physical_repair_law_selected": False,
        "physical_release_ensemble_selected": False,
        "promotion_allowed": False,
        "refinement_tower_equal_seam_extension_source_selected": False,
        "screen_to_sky_readout_selected": False,
    }
    assert "not a covariance" in receipt["claim_boundary"]
    assert "no analytic tail bound" in receipt["claim_boundary"]


def test_complete_coefficient_surface_and_a5_forbidden_rank_controls(
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = canonical
    assert len(coefficients["cases"]) == 8
    assert all(len(case["rows"]) == 5929 for case in coefficients["cases"])
    allowed = {0, 6, 10, 12, 15, 16}
    for level in receipt["level_rows"][2:]:
        for operator in level["operator_cases"]:
            norms = {
                int(rank): float(value)
                for rank, value in operator["biposh_summary"][
                    "total_L_frobenius_norms"
                ].items()
            }
            forbidden = [value for rank, value in norms.items() if rank not in allowed]
            assert max(forbidden) < 1.0e-11
            assert norms[6] > 0.0
            leakage = operator["biposh_summary"]["a5_selection_leakage"]
            assert leakage["gate_passed"] is True
            assert leakage["maximum_forbidden_total_L_norm"] < 1.0e-11


def test_spectral_rows_are_basis_invariant_projector_certificates(
    canonical: tuple[dict, dict],
) -> None:
    receipt, _ = canonical
    for level in receipt["level_rows"][2:]:
        for operator in level["operator_cases"]:
            spectral = operator["spectral_projector_diagnostics"]
            rows = spectral["rows"]
            assert [row["ell"] for row in rows] == list(range(2, 9))
            assert sorted(
                index
                for row in rows
                for index in row["assigned_eigenvalue_indices"]
            ) == list(range(77))
            for row in rows:
                if row["davis_kahan_hypothesis_2r_lt_gap"]:
                    assert row["bound_contains_measured_angle"] is True
                    assert (
                        row["maximum_principal_angle_sine"]
                        <= row["davis_kahan_sine_upper_bound"] + 2.0e-12
                    )
                else:
                    assert row["davis_kahan_sine_upper_bound"] is None


def test_rebuilt_packet_matches_committed_primary_values(canonical: tuple[dict, dict]) -> None:
    committed, _ = canonical
    rebuilt, rebuilt_coefficients = build_a5_biposh_dual_operator_packet()
    assert rebuilt["status"] == committed["status"]
    assert rebuilt_coefficients["case_count"] == 8
    for committed_level, rebuilt_level in zip(
        committed["level_rows"][2:], rebuilt["level_rows"][2:], strict=True
    ):
        for committed_case, rebuilt_case in zip(
            committed_level["operator_cases"],
            rebuilt_level["operator_cases"],
            strict=True,
        ):
            assert rebuilt_case["operator_id"] == committed_case["operator_id"]
            assert rebuilt_case["biposh_summary"][
                "primary_amplitude_free_statistic"
            ] == pytest.approx(
                committed_case["biposh_summary"][
                    "primary_amplitude_free_statistic"
                ],
                abs=2.0e-9,
            )


def test_coefficient_serialization_contract_is_explicit_and_satisfied(
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = canonical
    assert receipt["full_coefficient_bundle"][
        "coefficient_significant_decimal_digits"
    ] == 12
    assert coefficients["serialization_contract"][
        "coefficient_real_and_imaginary_significant_decimal_digits"
    ] == 12
    for case in coefficients["cases"]:
        for row in case["rows"]:
            for value in row[4:]:
                assert float(f"{float(value):.11e}") == float(value)
    for level in receipt["level_rows"][2:]:
        for operator in level["operator_cases"]:
            error = operator["coefficient_serialization_error"]
            assert (
                error["maximum_error_divided_by_max_one_abs_raw"]
                <= error["normalized_error_gate"]
                == 5.1e-12
            )


def test_bounded_repair_mean_generator_is_connected_only_on_base_carrier(
    canonical: tuple[dict, dict],
) -> None:
    receipt, _ = canonical
    bridge = receipt["bounded_repair_generator_bridge"]
    assert bridge[
        "base_carrier_operator_matches_bounded_reconstructed_one_atom_mean_generator_up_to_scale"
    ] is True
    assert bridge["conditional_mean_identity"] == (
        "E[X_next | X=x] = (I - L_icosahedron/60) x"
    )
    assert bridge["one_atom_generator"] == "-L_icosahedron/60"
    assert bridge["parent_status"] == (
        "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_"
        "FROZEN_ADVERSARIAL_SUITE"
    )
    assert bridge["parent_certificate_payload_sha256"] == (
        "sha256:9e87c5e4abfb3baed80058ffc832a6dbd3412f386eb383d68fee4ebee10c00d5"
    )
    assert bridge["base_carrier_vertex_count"] == 12
    assert bridge["base_carrier_edge_count"] == 30
    assert bridge["parent_exact_oriented_face_sha256"] == (
        "sha256:772cceb28efb46a2f322dd1cd87eac61f8bde0e54bbf387163efb13a6df9ad1c"
    )
    assert bridge["base_carrier_oriented_face_sha256"] == (
        "sha256:772cceb28efb46a2f322dd1cd87eac61f8bde0e54bbf387163efb13a6df9ad1c"
    )
    assert bridge["base_carrier_labelled_face_presentation_matches_parent"] is True
    assert bridge["base_carrier_edge_set_matches_face_presentation"] is True
    assert bridge["refinement_tower_extension_source_selected"] is False
    assert bridge["global_a1_a3_policy_uniqueness_receipt"] is False
    assert bridge["physical_repair_law_receipt"] is False
    assert bridge["physical_time_scale_selected"] is False


def test_rehashed_base_face_binding_mutation_is_rejected(
    tmp_path: Path,
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = copy.deepcopy(canonical)
    receipt["bounded_repair_generator_bridge"][
        "base_carrier_oriented_face_sha256"
    ] = "sha256:" + "0" * 64
    receipt_path, coefficient_path = _write_mutation(
        tmp_path,
        receipt,
        coefficients,
    )
    with pytest.raises(VerificationError, match="bounded repair generator bridge"):
        verify_packet(receipt_path, coefficient_path)


def test_rehashed_coefficient_mutation_is_rejected(
    tmp_path: Path,
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = copy.deepcopy(canonical)
    coefficients["cases"][0]["rows"][0][4] += 1.0e-4
    receipt_path, coefficient_path = _write_mutation(
        tmp_path,
        receipt,
        coefficients,
    )
    with pytest.raises(VerificationError, match="case hash mismatch|real"):
        verify_packet(receipt_path, coefficient_path)


def test_rehashed_operator_identity_swap_is_rejected(
    tmp_path: Path,
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = copy.deepcopy(canonical)
    coefficients["cases"][0]["operator_id"] = "geometric_cotangent_control"
    receipt_path, coefficient_path = _write_mutation(
        tmp_path,
        receipt,
        coefficients,
    )
    with pytest.raises(VerificationError, match="duplicate or missing coefficient cases"):
        verify_packet(receipt_path, coefficient_path)


def test_receipt_promotion_mutation_is_rejected(
    tmp_path: Path,
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = copy.deepcopy(canonical)
    receipt["selection_decision"]["physical_covariance_selected"] = True
    receipt_path, coefficient_path = _write_mutation(
        tmp_path,
        receipt,
        coefficients,
    )
    with pytest.raises(VerificationError, match="forbidden physical promotion"):
        verify_packet(receipt_path, coefficient_path)


def test_source_pin_mutation_is_rejected(
    tmp_path: Path,
    canonical: tuple[dict, dict],
) -> None:
    receipt, coefficients = copy.deepcopy(canonical)
    receipt["source_pins"][0]["sha256"] = "sha256:" + "0" * 64
    receipt_path, coefficient_path = _write_mutation(
        tmp_path,
        receipt,
        coefficients,
    )
    with pytest.raises(VerificationError, match="source pin mismatch"):
        verify_packet(receipt_path, coefficient_path)
