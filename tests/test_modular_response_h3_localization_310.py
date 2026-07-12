from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.bulk.modular_response_h3_localization import (
    AMBIGUOUS,
    CERTIFIED,
    modular_response_h3_localization_report,
)
from oph_fpe.bulk.theorem_contract import finite_oph_theorem_contract_report
from oph_fpe.claims import MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT


def _tetra_dirs() -> list[list[float]]:
    scale = 1.0 / math.sqrt(3.0)
    return [
        [scale, scale, scale],
        [scale, -scale, -scale],
        [-scale, scale, -scale],
        [-scale, -scale, scale],
    ]


def _normals() -> list[list[float]]:
    return [[1.0 / math.sqrt(2.0), *(math.sqrt(3.0 / 2.0) * np.asarray(u)).tolist()] for u in _tetra_dirs()]


def _eta(x: list[float], y: list[float]) -> float:
    return -x[0] * y[0] + sum(x[i] * y[i] for i in range(1, 4))


def _features(point: list[float], normals: list[list[float]]) -> list[float]:
    return [_eta(point, normal) for normal in normals]


def _payload(*, ambiguous: bool = False) -> dict:
    normals = _normals()
    true_point = [math.cosh(0.3), math.sinh(0.3), 0.0, 0.0]
    other_point = [math.cosh(0.55), 0.0, math.sinh(0.55), 0.0]
    far_point = [math.cosh(0.8), 0.0, 0.0, math.sinh(0.8)]
    lower = [0.0, 0.0, 0.0] if ambiguous else [0.0, 0.93, 1.38]
    upper = [0.95, 1.0, 1.5] if ambiguous else [0.001, 0.95, 1.40]
    return {
        "curvature_radius": 1.0,
        "kernel": {"type": "signed_linear", "declared": True},
        "normals": normals,
        "domain": {
            "type": "ball",
            "center": [1.0, 0.0, 0.0, 0.0],
            "D": 2.0,
            "epsilon": 0.01,
            "epsilon_net_certificate": {
                "certificate_id": "synthetic-net-v1",
                "method": "analytic_cover_witness",
                "coverage_verified": True,
                "domain_center": [1.0, 0.0, 0.0, 0.0],
                "domain_radius": 2.0,
                "epsilon": 0.01,
                "candidate_count": 3,
                "max_covering_radius": 0.01,
            },
        },
        "epsilon": 0.01,
        "tau": 0.0,
        "L": 8.0,
        "point_source": {"passed": True, "held_out_residual_passed": True},
        "chart_naturality": {"passed": True},
        "refinement": {"passed": True},
        "negative_controls": {
            "shuffled_cap_normals": True,
            "complement_orientation_swap": True,
            "shuffled_token_labels": True,
            "wrong_R_H": True,
            "wrong_2pi": True,
            "duplicate_timestamps_not_rank": True,
            "two_source_mixture": True,
            "extended_source": True,
            "refinement_replay": True,
        },
        "observations": [
            {
                "token_id": "rec-1",
                "responses": _features(true_point, normals),
                "sigma": 0.001,
                "candidate_points": [true_point, other_point, far_point],
                "residual_lower_bounds": lower,
                "residual_upper_bounds": upper,
                "residual_interval_certificate": {
                    "certificate_id": "synthetic-directed-rounding-v1",
                    "method": "directed_rounding_interval_arithmetic",
                    "bounds_certified": True,
                    "candidate_count": 3,
                },
            }
        ],
    }


def test_issue310_modular_response_localization_certifies_gap_and_ball() -> None:
    report = modular_response_h3_localization_report(_payload())

    assert report["terminal_status"] == CERTIFIED
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is True
    assert report["H3LOC"] is True
    assert report["frame"]["rank"] == 4
    assert report["frame"]["sigma_min"] > 0
    assert report["tokens"][0]["Delta_loc"] > 0
    assert report["tokens"][0]["Delta_loc"] == pytest.approx(0.93 - 0.001)
    assert report["tokens"][0]["localization_radius"] > 0
    assert report["mandatory_nonclaims"]["PARTICLE_SPECIES_DERIVED"] is False
    assert report["mandatory_nonclaims"]["NEUTRAL_CHART_BLIND_BULK"] is False


def test_issue310_noisy_finite_output_is_ambiguous_without_positive_gap() -> None:
    report = modular_response_h3_localization_report(_payload(ambiguous=True))

    assert report["terminal_status"] == AMBIGUOUS
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["tokens"][0]["Delta_loc"] <= 0
    assert report["tokens"][0]["error_receipt"] is True
    assert report["tokens"][0]["gap_receipt"] is False


def test_theorem_contract_surfaces_issue310_source_file(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "modular_response_h3_localization_source.json").write_text(
        json.dumps(_payload()),
        encoding="utf-8",
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["issue_310_modular_response_h3_localization"]["receipt"] is True
    assert report["issue_310_modular_response_h3_localization"]["terminal_status"] == CERTIFIED
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is True
    assert report["h3_modular_response_localization_receipt"] is True


def test_issue310_rejects_candidates_outside_declared_ball() -> None:
    payload = _payload()
    payload["domain"]["D"] = 0.01
    payload["domain"]["epsilon_net_certificate"]["domain_radius"] = 0.01

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["tokens"][0]["inside_declared_domain"] is False
    assert "OUTSIDE_DECLARED_DOMAIN" in report["tokens"][0]["blockers"]


def test_issue310_rejects_indefinite_weight_matrix_and_inconsistent_bounds() -> None:
    payload = _payload()
    payload["weights"] = [1.0, 1.0, 1.0, -1.0]
    payload["alpha"] = 1.0e9
    payload["L"] = 1.0e-6

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_WEIGHT_MATRIX_PSD_RECEIPT"] is False
    assert report["H3_OBSERVABILITY_BOUNDS_RECEIPT"] is False
    assert report["weights"]["minimum_eigenvalue"] == -1.0


def test_issue310_rejects_fabricated_residual_intervals() -> None:
    payload = _payload()
    payload["observations"][0]["residual_lower_bounds"] = [0.0, 999.0, 999.0]
    payload["observations"][0]["residual_upper_bounds"] = [-999.0, 0.0, 0.0]

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["tokens"][0]["residual_interval_receipt"] is False
    assert report["tokens"][0]["Delta_loc"] == float("-inf")


def test_issue310_requires_explicit_finite_total_error_sigma() -> None:
    payload = _payload()
    payload["observations"][0].pop("sigma")

    report = modular_response_h3_localization_report(payload)

    token = report["tokens"][0]
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_TOTAL_ERROR_SIGMA_RECEIPT"] is False
    assert report["H3_LOCALIZATION_ERROR_RECEIPT"] is False
    assert token["sigma"] is None
    assert token["sigma_input_valid"] is False
    assert token["error_receipt"] is False
    assert token["gap_receipt"] is False
    assert token["worldline_handoff_receipt"] is False


@pytest.mark.parametrize("sigma", [True, "0.001", float("nan"), float("inf"), -0.1])
def test_issue310_rejects_non_numeric_or_nonfinite_total_error_sigma(sigma) -> None:
    payload = _payload()
    payload["observations"][0]["sigma"] = sigma

    report = modular_response_h3_localization_report(payload)

    token = report["tokens"][0]
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_TOTAL_ERROR_SIGMA_RECEIPT"] is False
    assert token["sigma_input_valid"] is False
    assert token["gap_receipt"] is False
    assert token["worldline_handoff_receipt"] is False


def test_issue310_raw_second_best_gap_is_diagnostic_without_certified_intervals() -> None:
    payload = _payload()
    token_source = payload["observations"][0]
    token_source.pop("residual_lower_bounds")
    token_source.pop("residual_upper_bounds")
    token_source.pop("residual_interval_certificate")

    report = modular_response_h3_localization_report(payload)

    token = report["tokens"][0]
    assert token["diagnostic_residual_gap"] > 0.0
    assert token["certified_residual_gap"] == float("-inf")
    assert token["Delta_loc"] == float("-inf")
    assert token["residual_interval_receipt"] is False
    assert token["gap_receipt"] is False
    assert token["worldline_handoff_receipt"] is False
    assert report["H3_RESIDUAL_INTERVAL_CERTIFICATE_RECEIPT"] is False
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False


@pytest.mark.parametrize(
    "certificate_patch",
    [
        {"bounds_certified": "true"},
        {"method": "some_numeric_method"},
        {"candidate_count": True},
        {"certificate_id": "   "},
    ],
)
def test_issue310_residual_interval_certificate_is_typed_and_method_limited(certificate_patch) -> None:
    payload = _payload()
    payload["observations"][0]["residual_interval_certificate"].update(certificate_patch)

    report = modular_response_h3_localization_report(payload)

    token = report["tokens"][0]
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert token["residual_interval_receipt"] is False
    assert token["gap_receipt"] is False
    assert token["worldline_handoff_receipt"] is False


@pytest.mark.parametrize(
    "kernel",
    [
        "signed_linear",
        {"type": "signed_linear"},
        {"type": "signed_linear", "declared": "true"},
        {"type": "signed_linear", "declared": 1},
    ],
)
def test_issue310_kernel_receipt_requires_literal_explicit_declaration(kernel) -> None:
    payload = _payload()
    payload["kernel"] = kernel

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_MODULAR_RESPONSE_KERNEL_RECEIPT"] is False


def test_issue310_paired_hinge_uses_two_channels_per_cap() -> None:
    payload = _payload()
    payload["kernel"] = {"type": "paired_hinge", "declared": True}
    signed = np.asarray(payload["observations"][0]["responses"], dtype=float)
    payload["observations"][0]["responses"] = np.concatenate(
        [np.maximum(signed, 0.0), np.maximum(-signed, 0.0)]
    ).tolist()

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is True
    assert report["tokens"][0]["residuals"][0] == 0.0
    assert report["observability"]["kernel_alpha_upper_bound"] == pytest.approx(
        report["frame"]["sigma_min"] / math.sqrt(2.0)
    )
    assert report["observability"]["alpha"] == pytest.approx(
        report["frame"]["sigma_min"] / math.sqrt(2.0)
    )


def test_issue310_derives_global_lipschitz_bound_on_compact_h3_domain() -> None:
    payload = _payload()
    payload.pop("L")

    report = modular_response_h3_localization_report(payload)

    certificate = report["observability"]["compact_domain_lipschitz_certificate"]
    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is True
    assert certificate["certified"] is True
    assert report["observability"]["L_source"] == "derived_compact_domain_global_bound"
    assert report["observability"]["L"] == certificate["global_L_bound"]
    assert report["observability"]["L"] > report["frame"]["sigma_max"]


def test_issue310_rejects_frame_sigma_max_as_global_lipschitz_shortcut() -> None:
    payload = _payload()
    baseline = modular_response_h3_localization_report(payload)
    payload["L"] = baseline["frame"]["sigma_max"]

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_OBSERVABILITY_BOUNDS_RECEIPT"] is False
    assert report["observability"]["L_consistent_with_frame"] is False


def test_issue310_requires_positive_definite_not_merely_semidefinite_weights() -> None:
    payload = _payload()
    payload["weights"] = [1.0, 1.0, 1.0, 0.0]

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_WEIGHT_MATRIX_PSD_RECEIPT"] is True
    assert report["H3_WEIGHT_MATRIX_POSITIVE_DEFINITE_RECEIPT"] is False
    assert report["weights"]["valid_positive_definite"] is False


def test_issue310_epsilon_net_coverage_requires_literal_boolean() -> None:
    payload = _payload()
    payload["domain"]["epsilon_net_certificate"]["coverage_verified"] = "true"

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_EPSILON_NET_COVERAGE_RECEIPT"] is False


def test_issue310_visualization_assumptions_do_not_promote_computed_receipt() -> None:
    payload = _payload()
    payload["alpha"] = 1.0e9
    payload["observability_assumption"] = {
        "enabled": True,
        "scope": "visualization_only",
        "bridge": "alpha_and_lipschitz_bounds",
        "assumption_id": "assume-observability-v1",
    }

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["simulation_assumption_status"]["observability_bounds_assumed"] is True
    assert report["H3_OBSERVABILITY_BOUNDS_RECEIPT"] is False


def test_issue310_does_not_treat_truthy_strings_or_objects_as_pass_booleans() -> None:
    payload = _payload()
    payload["chart_naturality"]["passed"] = "false"
    payload["refinement"]["passed"] = {"claimed": True}
    payload["negative_controls"]["wrong_2pi"] = "true"

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["H3_CHART_NATURALITY_RECEIPT"] is False
    assert report["H3_REFINEMENT_LOCALIZATION_RECEIPT"] is False
    assert report["H3_NEGATIVE_CONTROLS_RECEIPT"] is False


def test_issue310_rejects_invalid_declared_constants_instead_of_using_fallbacks() -> None:
    payload = _payload()
    payload["curvature_radius"] = -1.0
    payload["alpha"] = -1.0
    payload["L"] = float("nan")
    payload["epsilon"] = -0.1

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["curvature_radius_input_valid"] is False
    assert report["observability"]["alpha_input_valid"] is False
    assert report["observability"]["L_input_valid"] is False
    assert report["observability"]["global_error_inputs_valid"] is False


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), "not-a-radius"])
def test_issue310_rejects_invalid_curvature_radius_without_crashing(radius) -> None:
    payload = _payload()
    payload["curvature_radius"] = radius

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["curvature_radius_input_valid"] is False
    assert report["H3_CURVATURE_RADIUS_VALID_RECEIPT"] is False


def test_issue310_rejects_mixed_malformed_normals_instead_of_dropping_rows() -> None:
    payload = _payload()
    payload["normals"].append([1.0, 2.0, 3.0])

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["normals_input_valid"] is False
    assert report["H3_PRIMITIVE_INPUT_SHAPE_RECEIPT"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_points", [[1.0, 0.0, 0.0, float("nan")]]),
        ("responses", [0.0, 0.0, float("nan"), 0.0]),
    ],
)
def test_issue310_rejects_malformed_token_primitives_without_filtering(field, value) -> None:
    payload = _payload()
    payload["observations"][0][field] = value

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["tokens"][0]["primitive_inputs_valid"] is False
    assert "malformed_or_nonfinite_responses_or_candidates" in report["tokens"][0]["blockers"]


@pytest.mark.parametrize(
    ("field", "value", "validity_key"),
    [
        ("alpha", 0.0, "alpha_input_valid"),
        ("L", -1.0, "L_input_valid"),
        ("epsilon", -1.0, "epsilon_input_valid"),
        ("tau", float("nan"), "tau_input_valid"),
    ],
)
def test_issue310_explicit_invalid_constants_block_without_fallback(field, value, validity_key) -> None:
    payload = _payload()
    payload[field] = value

    report = modular_response_h3_localization_report(payload)

    assert report[MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT] is False
    assert report["observability"][validity_key] is False
    if field == "alpha":
        assert report["observability"]["alpha_source"] == "invalid_declared"
        assert report["observability"]["alpha"] == 0.0
    if field == "L":
        assert report["observability"]["L_source"] == "invalid_declared"
        assert report["observability"]["L"] == 0.0
