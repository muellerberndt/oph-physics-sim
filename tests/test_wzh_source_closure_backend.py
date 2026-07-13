from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.bosons.brst_blocks import block_structure_receipt
from oph_fpe.bosons.pipeline import build_wzh_campaign_report, load_wzh_config, write_wzh_campaign_bundle
from oph_fpe.bosons.pole_enclosure import pole_enclosure_receipt
from oph_fpe.bosons.rg_transport import piecewise_affine_rg_receipt
from oph_fpe.bosons.source_clock import PLANCK_H_J_S, source_clock_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bosons" / "wzh_source_closure_diagnostic_v1.yml"


def test_source_clock_computes_gap_but_fails_closed_without_source_packet() -> None:
    receipt = source_clock_receipt(
        np.diag([0.0, 2.0, 5.0]),
        frequency_hz=10.0,
        perturbation_norm_bound=0.1,
        source_packet_verified=False,
        no_target_ancestry=False,
    )

    assert receipt["dimensionless_gap"] == 2.0
    assert receipt["gap_interval"] == [1.8, 2.2]
    assert receipt["E_star_joule"] == PLANCK_H_J_S * 5.0
    assert receipt["E_star_GeV"] is not None
    assert receipt["source_clock_receipt"] is False
    assert "source_clock_packet_not_verified" in receipt["blockers"]


def test_piecewise_rg_matches_scalar_affine_solution() -> None:
    receipt = piecewise_affine_rg_receipt(
        [2.0],
        initial_scale=math.e,
        segments=[
            {
                "end_scale": 1.0,
                "matrix": [[1.0]],
                "offset": [0.0],
                "matching_matrix": [[1.0]],
                "matching_offset": [0.0],
            }
        ],
        source_packet_verified=True,
        same_branch=True,
        scheme="test",
        loop_order="exact_affine",
    )

    assert receipt["final_state"][0] == pytest.approx(2.0 / math.e, abs=1.0e-10)
    assert receipt["rg_matching_receipt"] is True


def test_neutral_block_factors_photon_and_massive_control_root() -> None:
    block = block_structure_receipt(
        [
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-3.0, 0.2]]],
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
        ],
        block_id="Z",
        block_kind="neutral",
    )

    assert block["ward_photon_factor_at_s_zero"] is True
    assert block["brst_block_receipt"] is False
    pole = pole_enclosure_receipt(
        block["determinant_coefficients"],
        contour_center=(3.0, -0.2),
        contour_radius=0.25,
    )
    assert pole["numerical_zero_control_receipt"] is True
    assert pole["physical_pole_receipt"] is False


def test_default_campaign_is_hash_pinned_fail_closed_diagnostic(tmp_path: Path) -> None:
    config = load_wzh_config(CONFIG)
    report = build_wzh_campaign_report(config, repository_root=ROOT, generated_utc="2026-07-12T00:00:00Z")

    assert report["overall_status"] == "diagnostic_backend_source_packets_incomplete"
    assert report["promotion_allowed"] is False
    assert report["promotion_gates"]["certificate_candidate_claim"] is False
    assert report["promotion_gates"]["paper_inputs_hash_pinned"] is False
    assert all(
        item["numerical_zero_control_receipt"] is True
        and item["physical_pole_receipt"] is False
        and item["physical_readout_attached"] is False
        and item["M_GeV"] is None
        for item in report["pole_enclosures"].values()
    )

    output = tmp_path / "bundle"
    written = write_wzh_campaign_bundle(CONFIG, output, repository_root=ROOT)
    assert written["promotion_allowed"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["promotion_allowed"] is False
