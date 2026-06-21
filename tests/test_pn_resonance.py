from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.cosmology.pn_resonance import (
    ALPHA_U_P_STAR,
    PNResonanceInputs,
    ew_bridge_capacity_from_p_alpha,
    pn_resonance_report,
    write_pn_resonance_report,
)
from oph_fpe.cosmology.scale_bridge import C_SI, HBAR_SI
from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC


DIRECT_B_ELL = 3.6078739146803216e70


def test_default_pn_resonance_replays_exact_paper_bridge_without_theorem_grade_promotion():
    report = pn_resonance_report()
    bridge = report["paper_bridge_relation"]
    gates = report["readiness_gates"]

    assert report["mode"] == "oph_pn_resonance_v0"
    assert report["branch_status"] == "paper_declared_pn_resonance_replay_diagnostic"
    assert report["PN_RESONANCE_NUMERIC_REPLAY"] is True
    assert report["PN_RESONANCE_RECEIPT"] is False
    assert math.isclose(bridge["selected_N_star"], ew_bridge_capacity_from_p_alpha())
    assert abs(bridge["log_residual_selected_minus_target"]) < 1.0e-10
    assert gates["scale_compressed_pn_resonance_replay_eligible"] is True
    assert gates["P_alpha_U_source_certificate_implemented"] is False
    assert gates["N_CRC_fixed_point_solved_from_finite_simulator"] is False
    assert gates["finite_regulator_patch_count_used_as_cosmic_capacity"] is False
    assert gates["theorem_grade_pn_resonance"] is False


def test_screen_capacity_default_is_reported_as_off_exact_pn_bridge_relation():
    report = pn_resonance_report(PNResonanceInputs(N_source="screen-capacity-default"))
    bridge = report["paper_bridge_relation"]
    observed = report["observed_branch_sidecar"]

    assert report["branch_status"] == "off_pn_bridge_relation_diagnostic"
    assert report["PN_RESONANCE_NUMERIC_REPLAY"] is False
    assert bridge["selected_N_star"] == DEFAULT_N_CRC
    assert bridge["paper_bridge_relation_exact"] is False
    assert observed["selected_N_over_default_N_CRC"] == 1.0
    assert observed["observed_display_compatible"] is True


def test_direct_n_can_match_ew_bridge_but_still_needs_source_and_capacity_proofs():
    n_ew = ew_bridge_capacity_from_p_alpha(alpha_u=ALPHA_U_P_STAR)
    report = pn_resonance_report(PNResonanceInputs(N_source="direct", N_star=n_ew))

    assert report["branch_status"] == "direct_pn_resonance_replay_diagnostic"
    assert report["PN_RESONANCE_NUMERIC_REPLAY"] is True
    assert report["PN_RESONANCE_RECEIPT"] is False
    assert report["readiness_gates"]["F_N_readback_map_implemented"] is False


def test_direct_scale_bridge_unlocks_dimensionful_checksum_but_not_finite_derivation():
    report = pn_resonance_report(PNResonanceInputs(B_ell_m2_inverse=DIRECT_B_ELL))
    scale_bridge = report["component_reports"]["oph_scale_bridge_report"]["scale_bridge"]
    expected_ell2 = 3.0 * math.pi / DIRECT_B_ELL

    assert math.isclose(scale_bridge["ell_star_squared_m2"], expected_ell2)
    assert math.isclose(scale_bridge["G_SI"], expected_ell2 * C_SI**3 / HBAR_SI)
    assert report["readiness_gates"]["dimensionful_G_SI_eligible"] is True
    assert report["readiness_gates"]["finite_simulator_derived_G_SI"] is False
    assert report["PN_RESONANCE_RECEIPT"] is False


def test_pn_resonance_report_writes_json_and_markdown(tmp_path: Path):
    report = write_pn_resonance_report(tmp_path)
    loaded = json.loads((tmp_path / "pn_resonance_report.json").read_text(encoding="utf-8"))

    assert (tmp_path / "pn_resonance_report.md").exists()
    assert loaded["mode"] == "oph_pn_resonance_v0"
    assert loaded["paper_bridge_relation"]["selected_N_star"] == report["paper_bridge_relation"]["selected_N_star"]


def test_pn_resonance_is_available_from_lazy_cosmology_exports():
    from oph_fpe.cosmology import PNResonanceInputs as ExportedInputs
    from oph_fpe.cosmology import pn_resonance_report as exported_report

    report = exported_report(ExportedInputs())

    assert report["PN_RESONANCE_NUMERIC_REPLAY"] is True


def test_measurement_pack_copies_pn_resonance_receipt(tmp_path: Path):
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    out = tmp_path / "pack"
    write_pn_resonance_report(run)

    pack = export_measurement_pack([run], out)

    assert (out / "pn_resonance_report.json").exists()
    assert (out / "pn_resonance_report.md").exists()
    assert pack["claims"]["pn_resonance_written"] is True
    assert pack["claims"]["pn_resonance_numeric_replay"] is True
    assert pack["claims"]["pn_resonance_receipt"] is False
