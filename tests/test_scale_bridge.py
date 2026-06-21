from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from oph_fpe.cosmology.scale_bridge import (
    C_SI,
    EPSILON_CS_SELECTED,
    HBAR_SI,
    NoGClockBridgeInputs,
    ScaleBridgeInputs,
    no_g_clock_bridge_report,
    dimensionless_pn_invariants,
    scale_bridge_report,
    write_no_g_clock_bridge_report,
    write_scale_bridge_report,
)
from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC


DIRECT_B_ELL = 3.6078739146803216e70


def test_pn_invariants_do_not_set_dimensionful_scale():
    report = scale_bridge_report(ScaleBridgeInputs(B_ell_m2_inverse=None))

    assert report["dimensionless_invariants"]["P_N_determine_dimensionful_scale"] is False
    assert report["dimensionless_invariants"]["P_cancels_from_local_gravity_readout"] is True
    assert math.isclose(report["dimensionless_invariants"]["B_ell_ell_star_squared"], 3.0 * math.pi)
    assert report["scale_bridge"]["B_ell_m2_inverse"] is None
    assert report["scale_bridge"]["ell_star_squared_m2"] is None
    assert report["scale_bridge"]["G_SI"] is None
    assert report["readiness_gates"]["independent_scale_bridge_supplied"] is False
    assert report["readiness_gates"]["dimensionful_G_SI_eligible"] is False
    assert report["readiness_gates"]["finite_simulator_derived_G_SI"] is False


def test_direct_b_ell_bridge_recovers_si_gravity_scale():
    report = scale_bridge_report(ScaleBridgeInputs(B_ell_m2_inverse=DIRECT_B_ELL))
    ell2 = 3.0 * math.pi / DIRECT_B_ELL
    expected_g = ell2 * C_SI**3 / HBAR_SI

    assert report["scale_bridge"]["source_input_mode"] == "direct_B_ell"
    assert report["scale_bridge"]["independent_scale_bridge_supplied"] is True
    assert math.isclose(report["scale_bridge"]["ell_star_squared_m2"], ell2)
    assert math.isclose(report["scale_bridge"]["G_SI"], expected_g)
    assert math.isclose(report["scale_bridge"]["G_SI"], 6.6743e-11, rel_tol=5.0e-5)
    assert report["scale_bridge"]["finite_simulator_derived_G_SI"] is False


def test_lambda_times_n_bridge_matches_direct_b_ell():
    lambda_star = DIRECT_B_ELL / DEFAULT_N_CRC
    report = scale_bridge_report(
        ScaleBridgeInputs(
            N_star=DEFAULT_N_CRC,
            Lambda_star_m2_inverse=lambda_star,
        )
    )

    assert report["scale_bridge"]["source_input_mode"] == "Lambda_star_times_N_star"
    assert math.isclose(report["scale_bridge"]["B_ell_m2_inverse"], DIRECT_B_ELL)
    assert math.isclose(report["scale_bridge"]["ell_star_squared_m2"], 3.0 * math.pi / DIRECT_B_ELL)


def test_dimensionless_invariants_rescale_with_n_only_dimensionlessly():
    n_small = 1.0e6
    invariants = dimensionless_pn_invariants(n_star=n_small)

    assert math.isclose(invariants["Lambda_star_ell_star_squared"], 3.0 * math.pi / n_small)
    assert invariants["P_N_determine_dimensionful_scale"] is False


def test_scale_bridge_rejects_ambiguous_or_incomplete_inputs():
    with pytest.raises(ValueError, match="either B_ell"):
        scale_bridge_report(ScaleBridgeInputs(B_ell_m2_inverse=1.0, Lambda_star_m2_inverse=1.0))

    with pytest.raises(ValueError, match="requires N_star"):
        scale_bridge_report(ScaleBridgeInputs(N_star=None, Lambda_star_m2_inverse=1.0))


def test_scale_bridge_report_writes_json_and_markdown(tmp_path: Path):
    report = write_scale_bridge_report(tmp_path, ScaleBridgeInputs(B_ell_m2_inverse=DIRECT_B_ELL))
    loaded = json.loads((tmp_path / "oph_scale_bridge_report.json").read_text(encoding="utf-8"))

    assert (tmp_path / "oph_scale_bridge_report.md").exists()
    assert loaded["mode"] == "oph_pn_scale_bridge_v1"
    assert loaded["scale_bridge"]["G_SI"] == report["scale_bridge"]["G_SI"]


def test_no_g_clock_bridge_emits_checksum_but_blocks_without_proof_artifacts():
    report = no_g_clock_bridge_report()

    assert report["NO_G_CLOCK_BRIDGE_RECEIPT"] is False
    assert report["source_predictive_G_SI"] is False
    assert report["clock_bridge"]["source_predictive_G_SI"] is False
    assert math.isclose(report["clock_bridge"]["epsilon_Cs"], EPSILON_CS_SELECTED)
    assert math.isclose(report["clock_bridge"]["G_SI"], 6.6743e-11, rel_tol=5.0e-5)
    assert "public_dependency_graph_missing" in report["blockers"]
    assert report["readiness_gates"]["no_forbidden_dependency_paths"] is False


def test_no_g_clock_bridge_receipt_requires_clean_dependency_graph():
    graph = {
        "gamma_star": ["R_gamma"],
        "epsilon_Cs": ["R_gamma"],
        "R_gamma": ["R_U", "R_alpha", "R_e_abs", "R_QCD_nuc_133Cs", "R_atom_133Cs"],
    }
    report = no_g_clock_bridge_report(
        NoGClockBridgeInputs(
            dependency_graph=graph,
            public_dependency_graph=True,
            source_readback_map_emitted=True,
            contraction_certificate=True,
            residual_certificate=True,
        )
    )

    assert report["NO_G_CLOCK_BRIDGE_RECEIPT"] is True
    assert report["source_predictive_G_SI"] is True
    assert report["clock_bridge"]["dimensionful_G_SI_eligible"] is True
    assert report["readiness_gates"]["forbidden_dependency_path_count"] == 0


def test_no_g_clock_bridge_rejects_forbidden_dependency_path():
    graph = {
        "gamma_star": ["R_gamma"],
        "R_gamma": ["R_U", "measured_G"],
    }
    report = no_g_clock_bridge_report(
        NoGClockBridgeInputs(
            dependency_graph=graph,
            public_dependency_graph=True,
            source_readback_map_emitted=True,
            contraction_certificate=True,
            residual_certificate=True,
        )
    )

    assert report["NO_G_CLOCK_BRIDGE_RECEIPT"] is False
    assert report["readiness_gates"]["forbidden_dependency_path_count"] == 1
    assert report["clock_bridge"]["forbidden_dependency_paths"][0][-1] == "measured_G"


def test_no_g_clock_bridge_report_writes_json_and_markdown(tmp_path: Path):
    report = write_no_g_clock_bridge_report(tmp_path)
    loaded = json.loads((tmp_path / "no_g_clock_bridge_report.json").read_text(encoding="utf-8"))

    assert (tmp_path / "no_g_clock_bridge_report.md").exists()
    assert loaded["mode"] == "oph_no_g_clock_bridge_v0"
    assert loaded["clock_bridge"]["G_SI"] == report["clock_bridge"]["G_SI"]


def test_scale_bridge_is_available_from_lazy_cosmology_exports():
    from oph_fpe.cosmology import ScaleBridgeInputs as ExportedInputs
    from oph_fpe.cosmology import NoGClockBridgeInputs as ExportedClockInputs
    from oph_fpe.cosmology import no_g_clock_bridge_report as exported_clock_report
    from oph_fpe.cosmology import scale_bridge_report as exported_report

    report = exported_report(ExportedInputs(B_ell_m2_inverse=DIRECT_B_ELL))
    clock_report = exported_clock_report(ExportedClockInputs())

    assert report["scale_bridge"]["independent_scale_bridge_supplied"] is True
    assert clock_report["clock_bridge"]["calibration_checksum_available"] is True
