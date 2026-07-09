from __future__ import annotations

from typing import Any


CLAIM_TIERS = (
    "CR0_VOCABULARY_ONLY",
    "CR1_QUOTIENT_DIAGNOSTIC",
    "CR2_CONDITIONAL_PHENOMENOLOGY",
    "CR3_FROZEN_PHYSICAL_PREDICTION",
    "CR4_SOURCE_ONLY_OPH_PREDICTION",
)

GATE_RECEIPTS = {
    "QUOTIENT": "COMPACT_QUOTIENT_RECEIPT",
    "SOURCE": "COMPACT_SOURCE_LAW_RECEIPT",
    "KERNEL": "PACKETIZED_KERNEL_RECEIPT",
    "CLOCK": "PHYSICAL_CLOCK_RECEIPT",
    "PACKET": "FINITE_PACKET_PARENT_RECEIPT",
    "PROP": "PROPAGATION_RECEIPT",
    "DETECTOR": "DETECTION_THINNING_RECEIPT",
    "POINTPROCESS": "POINT_PROCESS_LIKELIHOOD_RECEIPT",
    "CENSORING": "CENSORING_AND_UPPER_LIMIT_RECEIPT",
    "CONTROLS": "CONTROL_MODEL_RECEIPT",
    "REFINEMENT": "REFINEMENT_STABILITY_RECEIPT",
    "FREEZE": "FROZEN_HASHES_RECEIPT",
    "LIKE": "HELDOUT_LIKELIHOOD_RECEIPT",
}

CR2_GATES = (
    "QUOTIENT",
    "SOURCE",
    "KERNEL",
    "CLOCK",
    "PACKET",
    "PROP",
    "DETECTOR",
    "POINTPROCESS",
    "CENSORING",
    "LIKE",
)

CR3_GATES = CR2_GATES + ("CONTROLS", "REFINEMENT", "FREEZE")

CR4_RECEIPTS = (
    "COMPACT_SOURCE_ACTION_DERIVED_RECEIPT",
    "EMISSION_MICROPHYSICS_DERIVED_RECEIPT",
    "PHYSICAL_CLOCK_DERIVED_RECEIPT",
    "OLD_HOST_FRB_SOURCE_THEOREM_RECEIPT",
    "BH_GENEALOGY_PRIOR_THEOREM_RECEIPT",
)

FAIL_CLOSED_RULES = (
    "quotient_schema_missing",
    "canonicalizer_nondeterministic",
    "likelihood_reads_representative_labels",
    "source_law_missing",
    "source_law_fitted_but_labeled_source_only",
    "packetized_kernel_missing",
    "physical_clock_missing",
    "packet_parent_missing",
    "conservation_residual_exceeds_bound",
    "detector_thinning_missing",
    "censoring_upper_limit_model_missing",
    "point_process_compensator_missing",
    "controls_missing",
    "refinement_stability_missing",
    "frozen_hashes_missing",
    "frb_source_identity_missing",
    "frb_cadence_exposure_missing",
    "bh_genealogy_dag_missing",
    "generation_prior_uses_ringdown_residual",
    "waveform_template_tuned_after_residual_inspection",
)

NONCLAIMS = (
    "compact record surface is not an FRB rate law",
    "normal form is not a burst timing law",
    "repair eigenvalue is not a physical duration without a clock map",
    "packet label is not emitted fluence, gamma energy, or GW strain",
    "old host is not an OPH source theorem",
    "hierarchical-generation tag must not read ringdown residuals",
    "detected catalog rows are not a full likelihood without exposure and censoring",
)


def conditional_cr2_receipts() -> dict[str, bool]:
    receipts = {name: False for name in GATE_RECEIPTS.values()}
    for gate in CR2_GATES:
        receipts[GATE_RECEIPTS[gate]] = True
    receipts.update(
        {
            "COMPACT_HISTORY_RECEIPT": True,
            "PACKET_CONSERVATION_RECEIPT": True,
            "REPEATER_HISTORY_LIKELIHOOD_RECEIPT": True,
            "FRB_SOURCE_IDENTITY_RECEIPT": True,
            "FRB_CADENCE_EXPOSURE_RECEIPT": True,
            "BH_GENEALOGY_DAG_RECEIPT": True,
            "NO_GENERATION_LEAKAGE_RECEIPT": True,
            "SIMULATOR_ACCURACY_RECEIPT": False,
            "PROMOTION_AUDIT_RECEIPT": True,
        }
    )
    for name in CR4_RECEIPTS:
        receipts[name] = False
    return receipts


def claim_tier(receipts: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    normalized = {str(key): bool(value) for key, value in receipts.items()}
    if not normalized.get(GATE_RECEIPTS["QUOTIENT"], False):
        return CLAIM_TIERS[0], "QUOTIENT", ["QUOTIENT"]

    missing_cr2 = [gate for gate in CR2_GATES if not normalized.get(GATE_RECEIPTS[gate], False)]
    if missing_cr2:
        return CLAIM_TIERS[1], missing_cr2[0], missing_cr2

    missing_cr3 = [gate for gate in CR3_GATES if not normalized.get(GATE_RECEIPTS[gate], False)]
    if missing_cr3:
        return CLAIM_TIERS[2], missing_cr3[0], missing_cr3

    missing_cr4 = [name for name in CR4_RECEIPTS if not normalized.get(name, False)]
    if missing_cr4:
        return CLAIM_TIERS[3], missing_cr4[0], missing_cr4

    return CLAIM_TIERS[4], None, []


def promotion_audit(receipts: dict[str, Any]) -> dict[str, Any]:
    tier, first_blocked, missing = claim_tier(receipts)
    return {
        "receipt_type": "PROMOTION_AUDIT_RECEIPT",
        "CR_READY": tier in {"CR3_FROZEN_PHYSICAL_PREDICTION", "CR4_SOURCE_ONLY_OPH_PREDICTION"},
        "first_blocked_gate": first_blocked,
        "allowed_claim_label": tier,
        "missing_for_next_tier": missing,
        "nonclaims": list(NONCLAIMS),
    }


class PromotionGate:
    """CR0-CR4 promotion helper for compact-transient receipt dictionaries."""

    def claim_tier(self, receipts: dict[str, Any]) -> str:
        return claim_tier(receipts)[0]

    def first_blocked_gate(self, receipts: dict[str, Any]) -> str | None:
        return claim_tier(receipts)[1]

    def audit(self, receipts: dict[str, Any]) -> dict[str, Any]:
        return promotion_audit(receipts)


def receipt_status_payload(
    receipt_type: str,
    *,
    passed: bool | None,
    **fields: Any,
) -> dict[str, Any]:
    if passed is True:
        status = "PASS"
    elif passed is False:
        status = "FAIL"
    else:
        status = "OPEN_GATE"
    return {
        "receipt_type": receipt_type,
        **fields,
        "status": status,
    }


def default_receipt_payloads(receipts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = {str(key): bool(value) for key, value in receipts.items()}
    payloads = {
        "COMPACT_HISTORY_RECEIPT": receipt_status_payload(
            "COMPACT_HISTORY_RECEIPT",
            passed=normalized.get("COMPACT_HISTORY_RECEIPT", False),
            history_schema_hash="declared_in_compact_record_transients_note",
            source_identity_policy_hash="source_id_required_for_repeaters",
            genealogy_policy_hash="genealogy_dag_required_for_bh_recycling",
            packet_ledger_hash="packet_ledger_required_for_emissions",
        ),
        "DETECTION_THINNING_RECEIPT": receipt_status_payload(
            "DETECTION_THINNING_RECEIPT",
            passed=normalized.get("DETECTION_THINNING_RECEIPT", False),
            selection_function_hash="detector_selection_function_required",
            obs_window_hash="obs_window_required",
            threshold_policy_hash="threshold_policy_required",
            cadence_hash="cadence_required",
        ),
        "CENSORING_AND_UPPER_LIMIT_RECEIPT": receipt_status_payload(
            "CENSORING_AND_UPPER_LIMIT_RECEIPT",
            passed=normalized.get("CENSORING_AND_UPPER_LIMIT_RECEIPT", False),
            nondetection_model_hash="nondetection_model_required",
            upper_limit_policy_hash="upper_limit_policy_required",
            multiwavelength_veto_hash="multiwavelength_veto_policy_required",
        ),
        "CONTROL_MODEL_RECEIPT": receipt_status_payload(
            "CONTROL_MODEL_RECEIPT",
            passed=normalized.get("CONTROL_MODEL_RECEIPT", False),
            control_family_hash="frozen_control_family_required",
            oph_model_hash="frozen_oph_model_required",
            comparison_threshold="predeclared_delta_min_required",
            heldout_split_hash="heldout_split_required",
        ),
        "REFINEMENT_STABILITY_RECEIPT": receipt_status_payload(
            "REFINEMENT_STABILITY_RECEIPT",
            passed=normalized.get("REFINEMENT_STABILITY_RECEIPT", False),
            coarse_run_hash="coarse_run_required",
            fine_run_hash="fine_run_required",
            coarse_map_hash="coarse_map_required",
            event_law_distance=None,
        ),
        "SIMULATOR_ACCURACY_RECEIPT": receipt_status_payload(
            "SIMULATOR_ACCURACY_RECEIPT",
            passed=normalized.get("SIMULATOR_ACCURACY_RECEIPT", None),
            epsilon_mu=None,
            epsilon_K=None,
            expected_path_length=None,
            epsilon_E=None,
            epsilon_prop=None,
            epsilon_detector=None,
            epsilon_canon=None,
            epsilon_clock=None,
            epsilon_mc=None,
            tv_bound=None,
        ),
        "PROMOTION_AUDIT_RECEIPT": {
            **promotion_audit(normalized),
            "status": "PASS" if normalized.get("PROMOTION_AUDIT_RECEIPT", False) else "OPEN_GATE",
        },
    }
    generic_contracts = {
        "COMPACT_QUOTIENT_RECEIPT": {
            "quotient_schema_hash": "Q_r_CR_schema_required",
            "canonicalizer_hash": "deterministic_canonicalizer_required",
            "representative_label_access": "forbidden_in_likelihood",
        },
        "COMPACT_SOURCE_LAW_RECEIPT": {
            "source_law_hash": "mu_r_CR_and_K_Gamma_hist_required",
            "source_only_claim_allowed": False,
        },
        "PACKETIZED_KERNEL_RECEIPT": {
            "kernel_hash": "K_Gamma_r_dq_dPi_dell_dtau_required",
            "factorization_shortcut_allowed": False,
        },
        "PHYSICAL_CLOCK_RECEIPT": {
            "clock_map_hash": "physical_compact_repair_clock_required",
            "repair_eigenvalue_is_duration": False,
        },
        "FINITE_PACKET_PARENT_RECEIPT": {
            "packet_parent_hash": "finite_packet_parent_required",
            "required_sectors": ["radio", "gamma", "GW", "optical", "neutrino", "environmental", "recipient"],
        },
        "PACKET_CONSERVATION_RECEIPT": {
            "conservation_residual_bound": "predeclared_bound_required",
        },
        "PROPAGATION_RECEIPT": {
            "propagation_map_hash": "P_prop_required",
        },
        "POINT_PROCESS_LIKELIHOOD_RECEIPT": {
            "compensator_hash": "marked_catalog_compensator_required",
            "catalog_rows_are_full_likelihood": False,
        },
        "REPEATER_HISTORY_LIKELIHOOD_RECEIPT": {
            "source_identity_hash": "source_identity_required",
            "history_likelihood_hash": "repeater_history_likelihood_required",
        },
        "FRB_SOURCE_IDENTITY_RECEIPT": {
            "identity_policy_hash": "old_host_repeater_identity_required",
        },
        "FRB_CADENCE_EXPOSURE_RECEIPT": {
            "cadence_exposure_hash": "cadence_and_exposure_correction_required",
        },
        "BH_GENEALOGY_DAG_RECEIPT": {
            "genealogy_dag_hash": "bh_genealogy_dag_required",
        },
        "NO_GENERATION_LEAKAGE_RECEIPT": {
            "forbidden_inputs": [
                "ringdown_residual",
                "postfit_repair_tail_amplitude",
                "echo_score",
                "waveform_template_tuned_after_residual_inspection",
            ],
        },
        "FROZEN_HASHES_RECEIPT": {
            "model_hash": "frozen_model_hash_required",
            "split_hash": "frozen_split_hash_required",
        },
        "HELDOUT_LIKELIHOOD_RECEIPT": {
            "heldout_split_hash": "heldout_split_required",
            "delta_min": "predeclared_delta_min_required",
        },
        "COMPACT_SOURCE_ACTION_DERIVED_RECEIPT": {
            "derived_from_oph_source_action": False,
        },
        "EMISSION_MICROPHYSICS_DERIVED_RECEIPT": {
            "derived_from_oph_emission_microphysics": False,
        },
        "PHYSICAL_CLOCK_DERIVED_RECEIPT": {
            "derived_from_oph_clock_theorem": False,
        },
        "OLD_HOST_FRB_SOURCE_THEOREM_RECEIPT": {
            "old_host_frb_source_law_theorem": False,
        },
        "BH_GENEALOGY_PRIOR_THEOREM_RECEIPT": {
            "bh_generation_prior_theorem": False,
        },
    }
    for name in sorted(normalized):
        if name not in payloads:
            payloads[name] = receipt_status_payload(
                name,
                passed=normalized.get(name, None),
                **generic_contracts.get(name, {"contract": "declared_in_compact_record_transients_note"}),
            )
    return payloads
