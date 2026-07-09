from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.physics_problem_outputs import (
    cmb_simulation_promotion_contract,
    compact_transient_contract,
    e8_triality_certificate_contract,
    fractional_quantum_hall_k_matrix_readout,
    fusion_ledger_readout,
    gamma_morphology_contract,
    high_tc_gate_readout,
    jwst_compact_object_contract,
    ltas_source_only_readout,
    uhe_coefficient_emission_contract,
    write_physics_problem_outputs_report,
)


def test_ltas_source_only_values_are_frozen_from_problem_note():
    report = ltas_source_only_readout()

    assert report["receipts"]["LTAS_SOURCE_OUTPUT_RECEIPT"] is True
    assert report["receipts"]["MEAN_FREE_PATH_READOUT_RECEIPT"] is True
    assert report["receipts"]["MATERIAL_SPECIFIC_BRANCH_AUDIT_RECEIPT"] is False
    assert math.isclose(report["collarSurvival"], 0.9343004881992495)
    assert math.isclose(report["lambdaOverEll"], 3.244098917358505e-3)
    assert math.isclose(report["acousticTunnelingStrengthCStar"], 3.2869594215959025e-4)
    assert math.isclose(report["acousticPlateauQ0Inverse"], 5.163143785766721e-4)
    assert math.isclose(report["dominantThermalLambdaOverEll"], 4.108699276994878e-3)


def test_fractional_quantum_hall_laughlin_k_matrix_readout():
    report = fractional_quantum_hall_k_matrix_readout([[3]], [1], quasiparticle_vectors=[[1]])

    assert report["receipts"]["FQH_ABELIAN_K_MATRIX_READOUT_RECEIPT"] is True
    assert report["fillingFractionNu"]["text"] == "1/3"
    assert report["sigmaXYUnitsE2OverH"]["text"] == "1/3"
    assert report["quasiparticleGroupOrderAbsDetK"] == 3
    qp = report["representativeQuasiparticles"][0]
    assert qp["chargeFractionQOverE"]["text"] == "1/3"
    assert qp["selfStatisticsThetaOverPi"]["text"] == "1/3"


def test_fractional_quantum_hall_jain_two_fifths_readout():
    report = fractional_quantum_hall_k_matrix_readout([[3, 2], [2, 3]], [1, 1])

    assert report["fillingFractionNu"]["text"] == "2/5"
    assert report["quasiparticleGroupOrderAbsDetK"] == 5
    assert report["representativeQuasiparticles"][0]["chargeFractionQOverE"]["text"] == "1/5"
    assert report["representativeQuasiparticles"][0]["selfStatisticsThetaOverPi"]["text"] == "3/5"
    assert report["receipts"]["FQH_NONCENTRAL_5_2_SELECTOR_RECEIPT"] is False


def test_high_tc_gate_readout_uses_weakest_gate_and_penalties():
    report = high_tc_gate_readout(
        {
            "T_amp": 160.0,
            "T_phase": 120.0,
            "T_hol": 140.0,
            "lambda_inst": 2.0,
            "P_inst": 3.0,
        }
    )

    assert report["receipts"]["HIGH_TC_GATE_OUTPUT_RECEIPT"] is True
    assert report["TcOPH"] == 120.0
    assert report["bottleneckGates"] == ["T_phase"]
    assert report["penaltyTotal"] == 6.0
    assert report["TcPredAfterPenalties"] == 114.0
    assert report["receipts"]["HIGH_TC_MATERIAL_RECEIPT"] is False


def test_compact_transient_contract_is_conditional_cr2():
    report = compact_transient_contract()

    assert report["computed"] is True
    assert report["claim"] == "CR2_CONDITIONAL_PHENOMENOLOGY"
    assert report["promotionAllowed"] is False
    assert report["firstBlockedGate"] == "CONTROLS"
    assert report["receipts"]["DETECTION_THINNING_RECEIPT"] is True
    assert report["receipts"]["CONTROL_MODEL_RECEIPT"] is False


def test_jwst_gamma_cmb_and_e8_contracts_are_registered():
    jwst = jwst_compact_object_contract()
    gamma = gamma_morphology_contract()
    cmb = cmb_simulation_promotion_contract()
    e8 = e8_triality_certificate_contract()

    assert jwst["strongestAllowedClaim"] == "J0_DIAGNOSTIC_PROXY"
    assert jwst["firstBlockedGate"] == "CATALOG_INGESTION_RECEIPT"
    assert gamma["strongestAllowedClaim"] == "DIAGNOSTIC_GAMMA_MAP"
    assert gamma["milestone"] == "GAMMA_MORPHOLOGY_AUDIT"
    assert gamma["receipts"]["OPH_GAMMA_MORPHOLOGY_PREDICTION_RECEIPT"] is False
    assert cmb["currentClaimTier"] == "UNSTARTED_OR_INVALIDATED"
    assert cmb["receipts"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert e8["receipts"]["E8_TRIALITY_CERTIFICATE_STATEMENT_RECEIPT"] is True
    assert e8["receipts"]["E8_TRIALITY_PUBLIC_RAW_BUNDLE_RECEIPT"] is False


def test_uhe_coefficient_emission_contract_is_source_only_fixture():
    report = uhe_coefficient_emission_contract()

    assert report["computed"] is True
    assert report["claimTier"] == "SOURCE_ONLY"
    assert report["strongestAllowedClaim"] == "SOURCE_ONLY_COEFFICIENT_EMITTED"
    assert report["physicalClaim"] is False
    assert report["receipts"]["NO_UHE_DATA_USE"] is True
    assert report["receipts"]["COMMON_SOURCE_LOCK"] is True


def test_fusion_ledger_separates_plasma_gain_from_net_plant_promotion():
    report = fusion_ledger_readout(
        {
            "P_fus": 500.0,
            "P_aux": 50.0,
            "P_loss": 120.0,
            "P_ch": 90.0,
            "W": 240.0,
            "E_load": 1000.0,
            "E_all_inputs": 200.0,
            "Delta_E_stored": 10.0,
            "E_startup": 20.0,
            "E_shutdown": 5.0,
            "E_aux": 50.0,
            "E_consumables": 15.0,
            "E_maintenance": 25.0,
            "E_waste": 5.0,
            "u_L": 100.0,
        }
    )

    assert report["receipts"]["FUSION_LEDGER_OUTPUT_RECEIPT"] is True
    assert report["plasmaGainQ"] == 10.0
    assert report["tauE"] == 2.0
    assert report["fusionRecordSurvivalGate"] is True
    assert report["plantLedger"]["LPlant"] == 670.0
    assert report["plantLedger"]["netPlantPromotionReceipt"] is True
    assert report["receipts"]["FUSION_NET_PLANT_PROMOTION_RECEIPT"] is True


def test_physics_problem_outputs_writer_emits_json_and_markdown(tmp_path: Path):
    report = write_physics_problem_outputs_report(tmp_path)

    assert report["schema"] == "oph_physics_problem_outputs_v1"
    assert (tmp_path / "physics_problem_outputs_report.json").exists()
    assert (tmp_path / "physics_problem_outputs_report.md").exists()
    parsed = json.loads((tmp_path / "physics_problem_outputs_report.json").read_text(encoding="utf-8"))
    assert parsed["outputs"]["low_temperature_amorphous_universality"]["receipts"][
        "LTAS_SOURCE_OUTPUT_RECEIPT"
    ] is True
    assert parsed["outputs"]["fractional_quantum_hall"]["computed"] is False
    assert parsed["outputs"]["fractional_quantum_hall"]["canonicalFormulaExamples"]["laughlin_1_3"][
        "fillingFractionNu"
    ]["text"] == "1/3"
    assert parsed["outputs"]["fractional_quantum_hall"]["canonicalFormulaExamples"]["jain_2_5"][
        "fillingFractionNu"
    ]["text"] == "2/5"
    assert parsed["outputs"]["compact_record_transients"]["claim"] == "CR2_CONDITIONAL_PHENOMENOLOGY"
    assert parsed["outputs"]["jwst_compact_object_source_release"]["strongestAllowedClaim"] == "J0_DIAGNOSTIC_PROXY"
    assert parsed["outputs"]["gamma_ray_morphology_claims"]["strongestAllowedClaim"] == "DIAGNOSTIC_GAMMA_MAP"
    assert parsed["outputs"]["high_energy_messenger_coefficients"]["claimTier"] == "SOURCE_ONLY"
    assert parsed["outputs"]["cmb_simulation_promotion"]["currentClaimTier"] == "UNSTARTED_OR_INVALIDATED"
    assert parsed["outputs"]["e8_spin8_triality_alt9_certificate"]["repositoryReceiptStatus"] == "pending_raw_bundle"
