from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.finite_covariant_parent import (
    ACTIVE_FIBER_RECEIPT,
    COMMON_PARENT_RESPONSE_POLE_RECEIPT,
    CONSERVED_SECTOR_DECOMPOSITION_RECEIPT,
    EXPLICIT_RECIPIENT_STRESS_RECEIPT,
    FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT,
    GAUGE_INDEPENDENCE_RECEIPT,
    PARENT_RECEIPT,
    PHYSICAL_CLOCK_RECEIPT,
    STRESS_CLOSURE_RECEIPT,
    finite_covariant_collar_packet_parent_report,
    write_finite_covariant_collar_packet_parent_report,
)


def test_finite_covariant_parent_report_passes_closed_packet_artifact(tmp_path: Path):
    artifact = _artifact()

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is True
    assert report[STRESS_CLOSURE_RECEIPT] is True
    assert report[EXPLICIT_RECIPIENT_STRESS_RECEIPT] is True
    assert report[GAUGE_INDEPENDENCE_RECEIPT] is True
    assert report["PACKET_MASS_SHELL_RECEIPT"] is True
    assert report["CHANNEL_FOUR_MOMENTUM_RECEIPT"] is True
    assert report["RETARDED_RESPONSE_RECEIPT"] is True
    assert report["RESPONSE_STABILITY_RECEIPT"] is True
    assert report["FINITE_PACKET_STRESS_READOUT_RECEIPT"] is True
    assert report["VARIATIONAL_MOMENT_STRESS_AGREEMENT_RECEIPT"] is True
    assert report["LOCAL_FRAME_COVARIANCE_RECEIPT"] is True
    assert report["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"] is True
    assert report["COSMOLOGICAL_GAUGE_INVARIANCE_RECEIPT"] is True
    assert report["FINITE_DOMAIN_OF_DEPENDENCE_RECEIPT"] is True
    assert report["SUBLUMINAL_CHARACTERISTICS_RECEIPT"] is True
    assert report["EXCHANGE_CURRENT_CLOSURE_RECEIPT"] is True
    assert report["DETAILED_BALANCE_RECEIPT"] is True
    assert report[PHYSICAL_CLOCK_RECEIPT] is True
    assert report[ACTIVE_FIBER_RECEIPT] is True
    assert report[CONSERVED_SECTOR_DECOMPOSITION_RECEIPT] is True
    assert report[COMMON_PARENT_RESPONSE_POLE_RECEIPT] is True
    assert report["Gamma_rec_status"] == "PHYSICAL_KERNEL"
    assert report["SOURCE_ROUTE_RECEIPT"] is True
    assert report["ENTROPY_UNIT_RECEIPT"] is True
    assert report["MODULAR_NONADDITIVITY_IDENTITY_RECEIPT"] is True
    assert report["CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT"] is True
    assert report["BW_BALL_NORMALIZATION_RECEIPT"] is True
    assert report["STRESS_TOMOGRAPHY_RECEIPT"] is True
    assert report["SM_CURRENT_NULL_RECEIPT"] is True
    assert report[FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT] is False
    assert report["physical_cmb_prediction"] is False
    assert report["blockers"] == []

    source = tmp_path / "parent.json"
    out = tmp_path / "finite_covariant_collar_packet_parent_report.json"
    source.write_text(json.dumps(artifact), encoding="utf-8")

    written = write_finite_covariant_collar_packet_parent_report(source, out)

    assert written[PARENT_RECEIPT] is True
    assert out.exists()


def test_finite_covariant_parent_does_not_require_likelihood_freeze_for_source_parent():
    artifact = _artifact()
    artifact["frozen_run"] = {"source_hash": _hash("0"), "mutable_source_artifacts": False}

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is True
    assert report[FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT] is False
    assert report["parent_blockers"] == []


def test_finite_covariant_parent_requires_recipient_stress_for_nonzero_gamma():
    artifact = _artifact()
    artifact["packets"]["states"] = [artifact["packets"]["states"][0]]

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report[EXPLICIT_RECIPIENT_STRESS_RECEIPT] is False
    assert "explicit_recipient_stress_missing_for_nonzero_Gamma_rec" in report["blockers"]


def test_finite_covariant_parent_requires_Gamma_rec_promotion_receipts():
    artifact = _artifact()
    artifact["repair"].pop("PHYSICAL_CLOCK_RECEIPT")
    artifact["repair"].pop("ACTIVE_FIBER_RECEIPT")
    artifact["repair"].pop("CONSERVED_SECTOR_DECOMPOSITION_RECEIPT")
    artifact["causal_response"].pop("COMMON_PARENT_RESPONSE_POLE_RECEIPT")

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report[PHYSICAL_CLOCK_RECEIPT] is False
    assert report[ACTIVE_FIBER_RECEIPT] is False
    assert report[CONSERVED_SECTOR_DECOMPOSITION_RECEIPT] is False
    assert report[COMMON_PARENT_RESPONSE_POLE_RECEIPT] is False
    assert report["Gamma_rec_status"] == "UNPROMOTED_REPAIR_STEP_DIAGNOSTIC"
    assert "Gamma_rec_promotion_not_certified" in report["blockers"]


def test_finite_covariant_parent_rejects_recipient_label_without_stress_residuals():
    artifact = _artifact()
    artifact["stress"].pop("recipient_stress_residual")
    artifact["stress"]["EXPLICIT_RECIPIENT_STRESS_RECEIPT"] = True

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report[EXPLICIT_RECIPIENT_STRESS_RECEIPT] is False
    assert "explicit_recipient_stress_missing_for_nonzero_Gamma_rec" in report["blockers"]


def test_finite_covariant_parent_rejects_boolean_only_cdm_limit():
    artifact = _artifact()
    artifact["cdm_limit"] = {"cdm_limit_regression_passed": True}

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report["CDM_LIMIT_RECOVERY_RECEIPT"] is False
    assert "cdm_limit_recovery_not_certified" in report["blockers"]


def test_finite_covariant_parent_rejects_raw_gauge_match_without_invariant_variables():
    artifact = _artifact()
    artifact["gauge"] = {
        "gauge_consistency_residual": 0.0,
        "independent_gauge_presentation_count": 2,
        "raw_newtonian_synchronous_match": True,
    }

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report[GAUGE_INDEPENDENCE_RECEIPT] is False
    assert "cosmological_gauge_invariance_not_certified" in report["blockers"]


def test_finite_covariant_parent_rejects_nonphysical_boolean_only_parent():
    artifact = _artifact()
    artifact["packets"]["states"][0]["u_mu"] = [2.0, 0.0, 0.0, 0.0]
    artifact["packets"]["states"][1]["u_mu"] = [99.0, 0.0, 0.0, 0.0]
    artifact["packets"]["states"][0]["momentum_local"] = [2.0, 0.0, 0.0, 0.0]
    artifact["packets"]["states"][1]["momentum_local"] = [99.0, 0.0, 0.0, 0.0]
    artifact["reaction_channels"]["channels"][0]["transported_delta_p"] = [1.0, 0.0, 0.0, 0.0]
    artifact["causal_response"] = {
        "characteristic_speed_bound": 0.0,
        "kinetic_matrix": [[0.0]],
        "damping_matrix": [[0.0]],
        "propagation_matrix": [[0.0]],
        "source_matrix": [[0.0]],
        "output_matrix": [[0.0]],
        "response_stability_residual": None,
        "retarded_support_residual": None,
    }

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert "packet_0_four_velocity_not_unit_timelike" in report["blockers"]
    assert "packet_1_four_velocity_not_unit_timelike" in report["blockers"]
    assert "packet_0_mass_shell_residual_too_large" in report["blockers"]
    assert "channel_four_momentum_not_certified" in report["blockers"]
    assert "causal_response_not_certified" in report["blockers"]


def test_finite_covariant_parent_rejects_negative_occupation():
    artifact = _artifact()
    artifact["packets"]["states"][0]["occupation"] = -1.0

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert "packet_0_occupation_invalid" in report["blockers"]


def test_finite_covariant_parent_requires_declared_source_route():
    artifact = _artifact()
    artifact["source_localization"].pop("source_route")

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report["SOURCE_ROUTE_RECEIPT"] is False
    assert "source_route_missing" in report["blockers"]


def test_finite_covariant_parent_rejects_raw_cmi_without_matching_theorem():
    artifact = _artifact()
    artifact["source_localization"].pop("matching_theorem_hash")
    artifact["source_localization"].pop("matching_residual")

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report["CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT"] is False
    assert "cmi_to_modular_source_matching_missing" in report["blockers"]


def test_finite_covariant_parent_rejects_missing_entropy_units():
    artifact = _artifact()
    artifact["source_localization"].pop("hbar")

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report["ENTROPY_UNIT_RECEIPT"] is False
    assert "entropy_or_physical_units_missing" in report["blockers"]


def _artifact() -> dict:
    return {
        "manifest": {
            "source_hash": _hash("0"),
            "regulator_id": "N64_eps0",
            "parent_theorem_version": "fccpp-v0",
        },
        "geometry": {
            "metric": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "transport": {"parallel_transport_maps": [{"matrix": _identity4()}]},
        "packets": {
            "states": [
                {
                    "label": "anomaly",
                    "rho": 0.2,
                    "occupation": 1.0,
                    "invariant_weight": 1.0,
                    "mass": 1.0,
                    "momentum_local": [1.0, 0.0, 0.0, 0.0],
                    "u_mu": [1.0, 0.0, 0.0, 0.0],
                    "SM_charges": {"EM": 0.0, "color": [0.0, 0.0, 0.0], "weak": 0.0},
                },
                {
                    "label": "recipient",
                    "rho": 0.1,
                    "occupation": 1.0,
                    "invariant_weight": 1.0,
                    "mass": 1.0,
                    "momentum_local": [1.0, 0.0, 0.0, 0.0],
                    "u_mu": [1.0, 0.0, 0.0, 0.0],
                },
            ]
        },
        "reaction_channels": {"channels": [{"channel_id": "A_to_R", "transported_delta_p": [0.0, 0.0, 0.0, 0.0]}]},
        "moments": {
            "stress_moment_residual": 0.0,
            "finite_divergence_operator_residual": 0.0,
            "stress_volume_weight_residual": 0.0,
            "variational_moment_agreement_residual": 0.0,
        },
        "source_localization": {
            "source_route": "NONLINEAR_CMI_STRESS",
            "entropy_base": "nats",
            "hbar": 1.0,
            "c": 1.0,
            "length_unit": "code_length",
            "proper_diamond_radius_ell": 1.0,
            "uv_scale": 0.01,
            "collar_width": 0.1,
            "curvature_scale_bound": 100.0,
            "gradient_scale_bound": 100.0,
            "cmi_kind": "QUANTUM_EXACT",
            "S_AB": 1.4,
            "S_BD": 1.3,
            "S_B": 1.0,
            "S_ABD": 1.5,
            "CMI": 0.2,
            "DeltaK_expectation": 0.2,
            "first_variation_classification": "MARKOV_BRANCH_QUADRATIC",
            "matching_theorem_hash": _hash("3"),
            "matching_residual": 0.0,
            "modular_source_charge_nats": 0.2,
            "source_localization_residual_nats": 0.0,
            "bw_ball_normalization_residual": 0.0,
            "ell4_scaling_plateau_residual": 0.0,
            "cover_independence_residual": 0.0,
            "timelike_probe_rank": 10,
            "heldout_quadraticity_residual": 0.0,
        },
        "repair": {
            "Gamma_rec": 0.05,
            "detailed_balance_residual": 0.0,
            "PHYSICAL_CLOCK_RECEIPT": True,
            "ACTIVE_FIBER_RECEIPT": True,
            "CONSERVED_SECTOR_DECOMPOSITION_RECEIPT": True,
        },
        "stress": {
            "total_stress_divergence_residual": 0.0,
            "exchange_current_residual": 0.0,
            "recipient_stress_residual": 0.0,
            "recipient_exchange_residual": 0.0,
            "local_frame_covariance_residual": 0.0,
            "carrier_quotient_invariance_residual": 0.0,
        },
        "causal_response": {
            "characteristic_speed_bound": 0.5,
            "kinetic_matrix": [[1.0]],
            "damping_matrix": [[1.0]],
            "propagation_matrix": [[0.5]],
            "source_matrix": [[1.0]],
            "output_matrix": [[1.0]],
            "response_stability_residual": 0.0,
            "retarded_support_residual": 0.0,
            "finite_domain_residual": 0.0,
            "COMMON_PARENT_RESPONSE_POLE_RECEIPT": True,
        },
        "gauge": {
            "gauge_consistency_residual": 0.0,
            "gauge_invariant_variable_residual": 0.0,
            "independent_gauge_presentation_count": 2,
        },
        "refinement": {"convergence_residual": 0.0, "regulator_level_count": 3},
        "cdm_limit": {"cdm_limit_residual": 0.0, "cdm_operator_residual": 0.0},
        "frozen_run": {
            "source_hash": _hash("0"),
            "solver_hash": _hash("1"),
            "likelihood_hash": _hash("2"),
            "mutable_source_artifacts": False,
        },
    }


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
