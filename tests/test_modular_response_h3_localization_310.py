from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

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
    lower = [0.0, 0.03, 0.04] if ambiguous else [0.0, 0.35, 0.45]
    upper = [0.05, 0.04, 0.05] if ambiguous else [0.001, 0.50, 0.60]
    return {
        "curvature_radius": 1.0,
        "kernel": "signed_linear",
        "normals": normals,
        "domain": {"type": "ball", "center": [1.0, 0.0, 0.0, 0.0], "D": 2.0, "epsilon": 0.01},
        "epsilon": 0.01,
        "tau": 0.0,
        "L": 3.0,
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
