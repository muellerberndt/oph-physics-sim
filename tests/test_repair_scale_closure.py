from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.repair_scale_closure import (
    capacity_from_contraction,
    effective_repair_round_depth,
    local_repair_contraction_from_p,
    repair_scale_closure_report,
    write_repair_scale_closure_report,
)


def test_repair_scale_closure_matches_maarten_numeric_bridge():
    report = repair_scale_closure_report(regulator_patch_counts=(4096, 65536, 262144, 1048576))
    outputs = report["closure_outputs"]

    assert math.isclose(local_repair_contraction_from_p(P_STAR), 0.0027873406516833745)
    assert math.isclose(outputs["n_s_p_over_2m"], 1.0 - P_STAR / 48.0)
    assert math.isclose(outputs["capacity_predicted_from_local_P"], 4.274424586583862e122)
    assert outputs["relative_error_gprime_vs_N_CRC_closure"] < 0.01
    assert report["readiness_gates"]["scale_closure_numeric_match_within_1_percent"] is True
    assert report["readiness_gates"]["twenty_four_round_hypothesis_derived_from_finite_selector"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["bulk_3d_established"] is False


def test_effective_round_depth_explains_small_regulators():
    contraction = local_repair_contraction_from_p(P_STAR)

    assert math.isclose(effective_repair_round_depth(4096, contraction), 0.7069723417630998)
    assert math.isclose(effective_repair_round_depth(1_048_576, contraction), 1.178287236271833)
    assert math.isclose(capacity_from_contraction(contraction), 4.274424586583862e122)


def test_repair_scale_closure_writes_outputs_and_aggregates(tmp_path: Path):
    out = tmp_path / "repair-scale"
    report = write_repair_scale_closure_report(
        out,
        regulator_patch_counts=(4096, 1048576),
    )

    assert (out / "repair_scale_closure_report.json").exists()
    assert (out / "repair_scale_closure_report.md").exists()
    assert (out / "repair_scale_round_depth.csv").exists()

    loaded = json.loads((out / "repair_scale_closure_report.json").read_text())
    assert loaded["mode"] == "oph_repair_scale_closure_hypothesis_v0"
    assert loaded["closure_outputs"]["n_s_p_over_2m"] == report["closure_outputs"]["n_s_p_over_2m"]

    comparable = comparable_data_report([out])
    lane = comparable["measurement_lanes"]["oph_repair_scale_closure"]

    assert lane["run_count"] == 1
    assert lane["numeric_match_within_1_percent_count"] == 1
    assert lane["twenty_four_rounds_declared_count"] == 1
    assert lane["twenty_four_rounds_derived_count"] == 0
    assert lane["finite_lattice_derived_eta_R_count"] == 0
    assert math.isclose(lane["mean_n_s"], 1.0 - P_STAR / 48.0)
    assert math.isclose(lane["mean_1m_effective_round_depth"], 1.178287236271833)
