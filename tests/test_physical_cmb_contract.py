from __future__ import annotations

import numpy as np

from oph_fpe.cosmology.physical_cmb_contract import (
    PhysicalCMBInputContract,
    contract_from_reports,
    validate_physical_cmb_contract,
)


def test_physical_cmb_contract_blocks_selector_constants():
    contract = _valid_contract()
    contract.eta_R_source = "selector_constant"
    contract.B_A_source = "diagnostic_proxy"

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "eta_R_not_finite_derived" in validation["blockers"]
    assert "B_A_k_a_missing_or_not_finite" in validation["blockers"]


def test_physical_cmb_requires_BA_and_Gamma():
    contract = _valid_contract()
    contract.B_A_k_a = None
    contract.Gamma_rec_k_a = None

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "B_A_k_a_missing_or_not_finite" in validation["blockers"]
    assert "Gamma_rec_k_a_missing_or_not_finite" in validation["blockers"]


def test_cdm_limit_regression_and_likelihood_must_pass():
    contract = _valid_contract()
    contract.cdm_limit_regression_passed = False
    contract.official_likelihood_ready = False

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "cdm_limit_regression_not_passed" in validation["blockers"]
    assert "official_likelihood_not_ready" in validation["blockers"]


def test_valid_physical_cmb_contract_passes_only_with_finite_sources():
    validation = validate_physical_cmb_contract(_valid_contract())

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is True
    assert validation["blockers"] == []


def test_physical_cmb_contract_requires_primordial_lift_receipt():
    contract = _valid_contract()
    contract.screen_to_primordial_lift_receipt = False

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "screen_to_primordial_lift_receipt_missing" in validation["blockers"]


def test_physical_cmb_contract_requires_source_provenance_receipts():
    contract = _valid_contract()
    contract.source_provenance_receipt = False
    contract.pooled_source_reducer_receipt = False
    contract.contradiction_free_provenance_receipt = False
    contract.N_CRC_consensus_invariant_receipt = False
    contract.global_likelihood_reduction_receipt = False

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "source_provenance_receipt_missing" in validation["blockers"]
    assert "pooled_source_reducer_receipt_missing" in validation["blockers"]
    assert "source_provenance_contradiction_check_failed" in validation["blockers"]
    assert "N_CRC_consensus_invariant_receipt_missing" in validation["blockers"]
    assert "global_likelihood_reduction_receipt_missing" in validation["blockers"]


def test_physical_cmb_contract_requires_finite_covariant_parent_and_frozen_hashes():
    contract = _valid_contract()
    contract.finite_covariant_parent_receipt = False
    contract.stress_energy_closure_receipt = False
    contract.frozen_likelihood_protocol_receipt = False
    contract.frozen_source_hash = None

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "finite_covariant_parent_receipt_missing" in validation["blockers"]
    assert "stress_energy_closure_not_certified" in validation["blockers"]
    assert "frozen_likelihood_protocol_not_certified" in validation["blockers"]
    assert "frozen_source_hash_missing" in validation["blockers"]


def test_physical_cmb_contract_requires_recipient_stress_for_nonzero_Gamma_rec():
    contract = _valid_contract()
    contract.explicit_recipient_stress_receipt = False

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "recipient_stress_missing_for_nonzero_Gamma_rec" in validation["blockers"]


def test_contract_from_reports_does_not_promote_diagnostic_rows():
    contract = contract_from_reports(
        no_data_use_receipt=True,
        repair_clock_report={
            "source": "diagnostic_proxy",
            "eta_R": 0.035,
            "Gamma_rec_grid": [[0.1, 0.2]],
        },
        ba_kernel_report={"source": "diagnostic_proxy", "B_A_grid": [[1.0]]},
        scalar_release_report={"source": "finite_lattice", "A_zeta": 2.1e-9},
        freezeout_report={
            "source": "finite_lattice",
            "q_IR": 0.25,
            "ell_IR": 32,
            "freezeout_surface": {"cycle": 24},
        },
        background_report={
            "source": "finite_lattice",
            "rho_A_grid": [[0.01, 0.02]],
        },
        likelihood_report={
            "official_likelihood_ready": True,
            "cdm_limit_regression_passed": True,
        },
    )

    validation = validate_physical_cmb_contract(contract)

    assert validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "eta_R_not_finite_derived" in validation["blockers"]
    assert "B_A_k_a_missing_or_not_finite" in validation["blockers"]
    assert "screen_to_primordial_lift_receipt_missing" in validation["blockers"]


def _valid_contract() -> PhysicalCMBInputContract:
    finite = "finite_lattice"
    return PhysicalCMBInputContract(
        no_data_use_receipt=True,
        P_source="OPH_pixel_branch_predeclared",
        N_source="OPH_screen_capacity_branch_predeclared",
        eta_R_source="finite_repair_transition_clock",
        eta_R_value=0.035,
        A_zeta_source=finite,
        A_zeta_value=2.1e-9,
        q_IR_source=finite,
        q_IR_value=0.25,
        ell_IR_source=finite,
        ell_IR_value=32.0,
        B_A_source="parent_collar_finite_difference",
        B_A_k_a=np.ones((3, 3)),
        Gamma_rec_source="finite_repair_transition_clock",
        Gamma_rec_k_a=np.ones((3, 3)),
        rho_A_source=finite,
        rho_A_a=np.ones((3, 2)),
        freezeout_source="neutral_bulk_freezeout",
        freezeout_surface={"cycle": 24},
        official_likelihood_ready=True,
        cdm_limit_regression_passed=True,
        screen_to_primordial_lift_receipt=True,
        finite_covariant_parent_receipt=True,
        stress_energy_closure_receipt=True,
        gauge_independence_receipt=True,
        causal_response_receipt=True,
        refinement_convergence_receipt=True,
        explicit_recipient_stress_receipt=True,
        source_provenance_receipt=True,
        pooled_source_reducer_receipt=True,
        contradiction_free_provenance_receipt=True,
        N_CRC_consensus_invariant_receipt=True,
        global_likelihood_reduction_receipt=True,
        frozen_likelihood_protocol_receipt=True,
        frozen_source_hash="sha256:source",
        frozen_solver_hash="sha256:solver",
        frozen_likelihood_hash="sha256:likelihood",
    )
