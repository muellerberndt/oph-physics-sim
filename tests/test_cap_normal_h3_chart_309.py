from __future__ import annotations

import json
import math

import pytest

from oph_fpe.bulk.cap_normal_h3_chart import (
    APPROXIMATE,
    CERTIFIED,
    cap_normal_h3_chart_report,
)
from oph_fpe.bulk.theorem_contract import finite_oph_theorem_contract_report
from oph_fpe.claims import CAP_NORMAL_H3_CHART_RECEIPT


def _primitive_chart(source_type: str = "analytic_round_cap") -> dict:
    alpha = math.pi / 3.0
    boundary_x = math.sqrt(1.0 - math.cos(alpha) ** 2)
    return {
        "radius_provenance": "UNIT_CONVENTION",
        "sky_points": [
            [boundary_x, 0.0, math.cos(alpha)],
            [-boundary_x, 0.0, math.cos(alpha)],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        "caps": [
            {
                "source_type": source_type,
                "center": [0.0, 0.0, 1.0],
                "alpha": alpha,
                "boundary_points": [
                    [boundary_x, 0.0, math.cos(alpha)],
                    [-boundary_x, 0.0, math.cos(alpha)],
                ],
                "interior_points": [[0.0, 0.0, 1.0]],
                "exterior_points": [[0.0, 0.0, -1.0]],
            }
        ],
        "transitions": [
            {
                "lambda": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "cap_images": [
                    {
                        "cap_index": 0,
                        "center": [0.0, 0.0, 1.0],
                        "alpha": alpha,
                    }
                ],
            }
        ],
        "h3_points": [
            [1.0, 0.0, 0.0, 0.0],
            [math.cosh(0.4), math.sinh(0.4), 0.0, 0.0],
        ],
    }


def test_issue309_cap_normal_h3_chart_certifies_analytic_primitive_fields() -> None:
    report = cap_normal_h3_chart_report(_primitive_chart())

    assert report["terminal_status"] == CERTIFIED
    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is True
    assert report["residuals"]["r_cap_normal"] < 1.0e-12
    assert report["residuals"]["r_boundary_C"] < 1.0e-12
    assert report["residuals"]["m_in"] > 0
    assert report["residuals"]["m_out"] > 0
    assert report["mandatory_nonclaims"]["POPULATED_BULK"] is False
    assert report["mandatory_nonclaims"]["NEUTRAL_CHART_BLIND_BULK"] is False
    assert report["mandatory_nonclaims"]["EINSTEIN_BRANCH_ENTRY"] is False


def test_issue309_fitted_chart_remains_approximate_without_global_certificate() -> None:
    report = cap_normal_h3_chart_report(_primitive_chart(source_type="fitted_round_cap"))

    assert report["terminal_status"] == APPROXIMATE
    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["h3_chart_approximate"] is True


def test_issue309_truthy_roundness_string_does_not_certify_fitted_caps() -> None:
    payload = _primitive_chart(source_type="fitted_round_cap")
    payload["global_roundness_certificate"] = "true"

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["terminal_status"] == APPROXIMATE


def test_issue309_strict_structured_roundness_certificate_is_accepted() -> None:
    payload = _primitive_chart(source_type="fitted_round_cap")
    payload["global_roundness_certificate"] = {
        "schema": "oph_global_round_cap_certificate_v1",
        "scope": "all_caps",
        "passed": True,
        "cap_count": 1,
    }

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is True


def test_theorem_contract_surfaces_issue309_cap_normal_h3_receipt(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "cap_normal_h3_chart_source.json").write_text(
        json.dumps(_primitive_chart()),
        encoding="utf-8",
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["issue_309_cap_normal_h3_chart"]["receipt"] is True
    assert report["issue_309_cap_normal_h3_chart"]["terminal_status"] == CERTIFIED
    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is True
    assert report["cap_normal_h3_chart_receipt"] is True


def test_issue309_rejects_nonunit_center_instead_of_normalizing_it() -> None:
    payload = _primitive_chart()
    payload["caps"][0]["center"] = [0.0, 0.0, 99.0]

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["caps"][0]["status"] == "nonunit_center"
    assert report["caps"][0]["center_norm_residual"] == 98.0


def test_issue309_missing_source_type_is_not_assumed_analytic() -> None:
    payload = _primitive_chart()
    payload["caps"][0].pop("source_type")

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["terminal_status"] == APPROXIMATE
    assert report["caps"][0]["source_type"] == "unspecified"


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), "not-a-radius"])
def test_issue309_rejects_invalid_curvature_radius_without_crashing(radius) -> None:
    payload = _primitive_chart()
    payload["curvature_radius"] = radius

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["lorentz_convention"]["curvature_radius_input_valid"] is False
    assert "invalid_curvature_radius" in report["blockers"]


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("sky_points", [[1.0, 2.0]], "malformed_or_nonfinite_sky_points"),
        ("h3_points", [[float("nan"), 0.0, 0.0, 0.0]], "malformed_or_nonfinite_h3_points"),
    ],
)
def test_issue309_rejects_malformed_or_nonfinite_point_arrays(field, value, blocker) -> None:
    payload = _primitive_chart()
    payload[field] = value

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert blocker in report["blockers"]


def test_issue309_rejects_malformed_or_nonfinite_lorentz_matrix_cleanly() -> None:
    payload = _primitive_chart()
    payload["transitions"][0]["lambda"] = [[float("nan")]]

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["transitions"][0]["status"] == "missing_lorentz_matrix"


def test_issue309_requires_unit_s2_cap_samples() -> None:
    payload = _primitive_chart()
    payload["caps"][0]["interior_points"] = [[0.0, 0.0, 2.0]]

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["caps"][0]["status"] == "cap_sample_not_unit_s2"
    assert report["caps"][0]["sample_unit_residuals"]["interior"] == 3.0


def test_issue309_requires_cap_equivariance_sample_for_every_transition() -> None:
    payload = _primitive_chart()
    payload["transitions"].append(
        {
            "lambda": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "cap_images": [],
        }
    )

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["transitions"][0]["cap_equivariance_samples"] == 1
    assert report["transitions"][1]["cap_equivariance_samples"] == 0
    assert "missing_cap_equivariance_samples_per_transition" in report["blockers"]


def test_issue309_cap_equivariance_uses_original_cap_index_mapping() -> None:
    payload = _primitive_chart()
    invalid = dict(payload["caps"][0])
    invalid["center"] = [0.0, 0.0, 99.0]
    payload["caps"] = [invalid, payload["caps"][0]]
    payload["transitions"][0]["cap_images"][0]["cap_index"] = 1

    report = cap_normal_h3_chart_report(payload)

    assert report[CAP_NORMAL_H3_CHART_RECEIPT] is False
    assert report["transitions"][0]["cap_equivariance_samples"] == 1
    assert report["transitions"][0]["cap_equivariance_residual"] < 1.0e-12
