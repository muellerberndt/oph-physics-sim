from __future__ import annotations

import json
import math

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
