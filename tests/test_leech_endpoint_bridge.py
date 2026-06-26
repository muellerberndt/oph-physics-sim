from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import (
    ALPHA_INV_ENDPOINT_CALIBRATED,
    ALPHA_INV_SOURCE_CANDIDATE,
)
from oph_fpe.cosmology.leech_endpoint_bridge import (
    C_Q_TARGET,
    DELTA_H_FP_CALIBRATED,
    LeechEndpointBridgeInputs,
    leech_endpoint_bridge_report,
    write_leech_endpoint_bridge_report,
)


def test_default_leech_endpoint_bridge_is_quarantined_without_artifact():
    report = leech_endpoint_bridge_report()

    assert report["mode"] == "leech_endpoint_bridge_quarantine_v0"
    assert report["LEECH_ENDPOINT_BRIDGE_QUARANTINE_RECEIPT"] is True
    assert report["LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT"] is False
    assert report["SAME_SCHEME_HADRONIC_ENDPOINT_FUNCTIONAL_RECEIPT"] is False
    assert report["FINE_STRUCTURE_ALPHA_ENDPOINT_PROMOTION_RECEIPT"] is False
    assert math.isclose(
        report["calibrated_endpoint_reference"]["Delta_H_cal_fp"],
        ALPHA_INV_ENDPOINT_CALIBRATED - ALPHA_INV_SOURCE_CANDIDATE,
    )
    assert "source_artifact_missing" in report["blockers"]


def test_source_only_same_scheme_candidate_passes_bridge_but_not_endpoint_promotion(tmp_path: Path):
    artifact = tmp_path / "candidate.json"
    artifact.write_text(
        json.dumps(
            {
                "outputs": {
                    "delta_H_fp": DELTA_H_FP_CALIBRATED,
                    "c_Q": C_Q_TARGET,
                },
                "endpoint_convention": "oph_same_scheme_hadronic_endpoint",
                "same_scheme_endpoint_transport": True,
                "hadronic_em_spectral_transport": True,
                "source_only_map_receipt": True,
                "full_endpoint_map_receipt": True,
                "oph_string_branch_descent_receipt": True,
                "fixed_point_loop_reproduced": True,
                "interval_uniqueness_receipt": True,
            }
        ),
        encoding="utf-8",
    )

    report = write_leech_endpoint_bridge_report(
        tmp_path / "out",
        LeechEndpointBridgeInputs(source_artifact=artifact),
    )

    assert report["LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT"] is True
    assert report["SAME_SCHEME_HADRONIC_ENDPOINT_FUNCTIONAL_RECEIPT"] is True
    assert report["FINE_STRUCTURE_ALPHA_ENDPOINT_PROMOTION_RECEIPT"] is False
    assert report["readiness_gates"]["target_leakage_free"] is True
    assert report["blockers"] == ["paper_review_promotion_gate_closed"]
    assert (tmp_path / "out" / "leech_endpoint_bridge_report.json").exists()
    assert (tmp_path / "out" / "leech_endpoint_bridge_report.md").exists()


def test_endpoint_decimal_only_candidate_is_rejected(tmp_path: Path):
    artifact = tmp_path / "alpha_only.json"
    artifact.write_text(
        json.dumps(
            {
                "alpha_inverse_endpoint": ALPHA_INV_ENDPOINT_CALIBRATED,
                "same_scheme_endpoint_transport": True,
                "hadronic_em_spectral_transport": True,
                "source_only_map_receipt": True,
                "full_endpoint_map_receipt": True,
                "oph_string_branch_descent_receipt": True,
                "fixed_point_loop_reproduced": True,
                "interval_uniqueness_receipt": True,
            }
        ),
        encoding="utf-8",
    )

    report = leech_endpoint_bridge_report(LeechEndpointBridgeInputs(source_artifact=artifact))

    assert report["LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT"] is False
    assert report["candidate_outputs"]["alpha_endpoint_only"] is True
    assert "alpha_endpoint_decimal_without_endpoint_functional" in report["blockers"]
    assert "delta_H_fp_or_c_Q_not_emitted" in report["blockers"]


def test_target_leakage_fails_even_when_delta_matches(tmp_path: Path):
    artifact = tmp_path / "leaky.json"
    artifact.write_text(
        json.dumps(
            {
                "delta_H_fp": DELTA_H_FP_CALIBRATED,
                "same_scheme_endpoint_transport": True,
                "hadronic_em_spectral_transport": True,
                "source_only_map_receipt": True,
                "full_endpoint_map_receipt": True,
                "oph_string_branch_descent_receipt": True,
                "fixed_point_loop_reproduced": True,
                "interval_uniqueness_receipt": True,
                "uses_codata_alpha": True,
            }
        ),
        encoding="utf-8",
    )

    report = leech_endpoint_bridge_report(LeechEndpointBridgeInputs(source_artifact=artifact))

    assert report["readiness_gates"]["delta_H_fp_target_match"] is True
    assert report["readiness_gates"]["target_leakage_free"] is False
    assert report["LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT"] is False
    assert "target_leakage_or_posthoc_fit_not_excluded" in report["blockers"]


def test_false_leakage_receipt_does_not_count_as_target_use(tmp_path: Path):
    artifact = tmp_path / "candidate_with_false_leakage_flags.json"
    artifact.write_text(
        json.dumps(
            {
                "delta_H_fp": DELTA_H_FP_CALIBRATED,
                "endpoint_convention": "OPH_A_Z_same_scheme",
                "hadronic_em_spectral_transport": True,
                "source_only_map_receipt": True,
                "full_endpoint_map_receipt": True,
                "oph_string_branch_descent_receipt": True,
                "fixed_point_loop_reproduced": True,
                "interval_uniqueness_receipt": True,
                "uses_codata_alpha": False,
                "uses_calibrated_gap": False,
            }
        ),
        encoding="utf-8",
    )

    report = leech_endpoint_bridge_report(LeechEndpointBridgeInputs(source_artifact=artifact))

    assert report["readiness_gates"]["same_scheme_endpoint_transport"] is True
    assert report["readiness_gates"]["target_leakage_free"] is True
    assert report["LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT"] is True


def test_standard_hadronic_running_control_is_not_used_as_inverse_alpha_correction():
    report = leech_endpoint_bridge_report()
    control = report["convention_mixing_control"]

    assert control["rejected_as_same_scheme_inverse_alpha_correction"] is True
    assert control["invalid_additive_inverse_alpha_insertion"] != ALPHA_INV_ENDPOINT_CALIBRATED
    assert report["readiness_gates"]["standard_hadronic_running_used_as_inverse_alpha_correction"] is False


def test_leech_endpoint_bridge_is_available_from_lazy_cosmology_exports():
    from oph_fpe.cosmology import LeechEndpointBridgeInputs as ExportedInputs
    from oph_fpe.cosmology import leech_endpoint_bridge_report as exported_report

    report = exported_report(ExportedInputs())

    assert report["LEECH_ENDPOINT_BRIDGE_QUARANTINE_RECEIPT"] is True


def test_measurement_pack_copies_leech_endpoint_bridge_report(tmp_path: Path):
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    out = tmp_path / "pack"
    write_leech_endpoint_bridge_report(run)

    pack = export_measurement_pack([run], out)

    assert (out / "leech_endpoint_bridge_report.json").exists()
    assert (out / "leech_endpoint_bridge_report.md").exists()
    assert pack["claims"]["leech_endpoint_bridge_written"] is True
    assert pack["claims"]["leech_endpoint_bridge_candidate_receipt"] is False
    assert pack["claims"]["leech_alpha_endpoint_promotion_receipt"] is False
