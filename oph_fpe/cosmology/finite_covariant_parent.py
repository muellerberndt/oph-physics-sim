from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PARENT_RECEIPT = "FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT"
ISSUE_319_CONDITIONAL_SOURCE_RECEIPT = "ISSUE_319_CONDITIONAL_SOURCE_RECEIPT"
SOURCE_ROUTE_RECEIPT = "SOURCE_ROUTE_RECEIPT"
ENTROPY_UNIT_RECEIPT = "ENTROPY_UNIT_RECEIPT"
MODULAR_NONADDITIVITY_IDENTITY_RECEIPT = "MODULAR_NONADDITIVITY_IDENTITY_RECEIPT"
CMI_FIRST_VARIATION_CLASSIFICATION_RECEIPT = "CMI_FIRST_VARIATION_CLASSIFICATION_RECEIPT"
CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT = "CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT"
FIXED_REFERENCE_MODULAR_ENERGY_RECEIPT = "FIXED_REFERENCE_MODULAR_ENERGY_RECEIPT"
SOURCE_LOCALIZATION_SATURATION_RECEIPT = "SOURCE_LOCALIZATION_SATURATION_RECEIPT"
MODULAR_SOURCE_CHARGE_RECEIPT = "MODULAR_SOURCE_CHARGE_RECEIPT"
BW_BALL_NORMALIZATION_RECEIPT = "BW_BALL_NORMALIZATION_RECEIPT"
PHYSICAL_DIAMOND_SCALE_RECEIPT = "PHYSICAL_DIAMOND_SCALE_RECEIPT"
ELL4_SCALING_PLATEAU_RECEIPT = "ELL4_SCALING_PLATEAU_RECEIPT"
COVER_INDEPENDENCE_RECEIPT = "COVER_INDEPENDENCE_RECEIPT"
STRESS_TOMOGRAPHY_RECEIPT = "STRESS_TOMOGRAPHY_RECEIPT"
STRESS_CLOSURE_RECEIPT = "STRESS_ENERGY_CLOSURE_RECEIPT"
GAUGE_INDEPENDENCE_RECEIPT = "GAUGE_INDEPENDENCE_RECEIPT"
EXPLICIT_RECIPIENT_STRESS_RECEIPT = "EXPLICIT_RECIPIENT_STRESS_RECEIPT"
CAUSAL_RESPONSE_RECEIPT = "CAUSAL_RESPONSE_RECEIPT"
REFINEMENT_CONVERGENCE_RECEIPT = "REFINEMENT_CONVERGENCE_RECEIPT"
FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT = "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT"
FINITE_PACKET_KINEMATICS_RECEIPT = "FINITE_PACKET_KINEMATICS_RECEIPT"
PACKET_KINEMATICS_RECEIPT = FINITE_PACKET_KINEMATICS_RECEIPT
PACKET_MASS_SHELL_RECEIPT = "PACKET_MASS_SHELL_RECEIPT"
TRANSPORT_COVARIANCE_RECEIPT = "TRANSPORT_COVARIANCE_RECEIPT"
CHANNEL_FOUR_MOMENTUM_RECEIPT = "CHANNEL_FOUR_MOMENTUM_RECEIPT"
TOTAL_STRESS_CLOSURE_RECEIPT = "TOTAL_STRESS_CLOSURE_RECEIPT"
FINITE_PACKET_STRESS_READOUT_RECEIPT = "FINITE_PACKET_STRESS_READOUT_RECEIPT"
VARIATIONAL_MOMENT_STRESS_AGREEMENT_RECEIPT = "VARIATIONAL_MOMENT_STRESS_AGREEMENT_RECEIPT"
LOCAL_FRAME_COVARIANCE_RECEIPT = "LOCAL_FRAME_COVARIANCE_RECEIPT"
CARRIER_QUOTIENT_INVARIANCE_RECEIPT = "CARRIER_QUOTIENT_INVARIANCE_RECEIPT"
COSMOLOGICAL_GAUGE_INVARIANCE_RECEIPT = "COSMOLOGICAL_GAUGE_INVARIANCE_RECEIPT"
FINITE_DOMAIN_OF_DEPENDENCE_RECEIPT = "FINITE_DOMAIN_OF_DEPENDENCE_RECEIPT"
SUBLUMINAL_CHARACTERISTICS_RECEIPT = "SUBLUMINAL_CHARACTERISTICS_RECEIPT"
RETARDED_RESPONSE_RECEIPT = "RETARDED_RESPONSE_RECEIPT"
RESPONSE_STABILITY_RECEIPT = "RESPONSE_STABILITY_RECEIPT"
SM_CURRENT_NULL_RECEIPT = "SM_CURRENT_NULL_RECEIPT"
REPAIR_EXCHANGE_OFF_RECEIPT = "REPAIR_EXCHANGE_OFF_RECEIPT"
EXCHANGE_CURRENT_CLOSURE_RECEIPT = "EXCHANGE_CURRENT_CLOSURE_RECEIPT"
DETAILED_BALANCE_RECEIPT = "DETAILED_BALANCE_RECEIPT"
DECLARED_NONEQUILIBRIUM_EXCHANGE_RECEIPT = "DECLARED_NONEQUILIBRIUM_EXCHANGE_RECEIPT"
PHYSICAL_CLOCK_RECEIPT = "PHYSICAL_CLOCK_RECEIPT"
ACTIVE_FIBER_RECEIPT = "ACTIVE_FIBER_RECEIPT"
CONSERVED_SECTOR_DECOMPOSITION_RECEIPT = "CONSERVED_SECTOR_DECOMPOSITION_RECEIPT"
COMMON_PARENT_RESPONSE_POLE_RECEIPT = "COMMON_PARENT_RESPONSE_POLE_RECEIPT"
RECIPIENT_PACKET_LABELS = {"recipient", "R", "repair_recipient"}
ANOMALY_PACKET_LABELS = {"anomaly", "A"}
SOURCE_ROUTES = {"FIXED_REFERENCE_MODULAR_ENERGY", "NONLINEAR_CMI_STRESS"}
EXACT_CMI_KINDS = {"QUANTUM_EXACT", "CLASSICAL_EXACT"}


def finite_covariant_collar_packet_parent_report(
    artifact: dict[str, Any],
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Validate the finite covariant collar-packet parent contract.

    This is a source-artifact validator, not a Boltzmann solver. It checks the
    pieces needed before scalar dark/anomaly tables can be promoted to physical
    CMB or matter-transfer inputs.
    """

    manifest = _section(artifact, "manifest")
    geometry = _section(artifact, "geometry")
    packets = _section(artifact, "packets")
    transport = _section(artifact, "transport")
    reactions = _section(artifact, "reaction_channels")
    moments = _section(artifact, "moments")
    repair = _section(artifact, "repair")
    causal = _section(artifact, "causal_response")
    stress = _section(artifact, "stress")
    source_localization = _section(artifact, "source_localization")
    refinement = _section(artifact, "refinement")
    cdm_limit = _section(artifact, "cdm_limit")
    gauge = _section(artifact, "gauge")
    frozen = _section(artifact, "frozen_run")

    blockers: list[str] = []
    if bool(artifact.get("synthetic_placeholder", False)) or bool(manifest.get("synthetic_placeholder", False)):
        blockers.append("synthetic_parent_placeholder_not_promotable")
    if artifact.get("promotion_allowed") is False or manifest.get("promotion_allowed") is False:
        blockers.append("parent_promotion_not_allowed")
    for key in ("source_hash", "regulator_id", "parent_theorem_version"):
        if not manifest.get(key):
            blockers.append(f"manifest_{key}_missing")
    if manifest.get("source_hash") and not _valid_sha256_hash(str(manifest.get("source_hash"))):
        blockers.append("manifest_source_hash_invalid")

    metric = _metric(geometry)
    if metric is None:
        blockers.append("metric_missing_or_invalid")
    packet_rows = packets.get("states") if isinstance(packets.get("states"), list) else []
    packet_check = _packet_state_check(packet_rows, metric=metric, tolerance=float(tolerance))
    blockers.extend(packet_check["blockers"])

    transport_receipt = _transport_receipt(transport)
    if not transport_receipt:
        blockers.append("transport_covariance_not_certified")

    channel_receipt, channel_residual = _channel_four_momentum_receipt(reactions, tolerance=float(tolerance))
    if not channel_receipt:
        blockers.append("channel_four_momentum_not_certified")

    moment_report = _moment_receipt(moments, tolerance=float(tolerance))
    moment_receipt = moment_report["stress_readout_receipt"]
    if not moment_receipt:
        blockers.append("finite_packet_stress_readout_not_certified")
    variational_moment_receipt = moment_report["variational_moment_receipt"]
    if not variational_moment_receipt:
        blockers.append("variational_moment_stress_agreement_not_certified")
    local_frame_receipt = _residual_receipt(
        stress,
        ("local_frame_covariance_residual", "tetrad_lorentz_covariance_residual"),
        tolerance=float(tolerance),
    )
    if not local_frame_receipt:
        blockers.append("local_frame_covariance_not_certified")
    quotient_receipt = _residual_receipt(
        stress,
        ("carrier_quotient_invariance_residual", "readout_quotient_invariance_residual"),
        tolerance=float(tolerance),
    )
    if not quotient_receipt:
        blockers.append("carrier_quotient_invariance_not_certified")

    stress_residual = _float(stress.get("total_stress_divergence_residual"))
    exchange_residual = _float(stress.get("exchange_current_residual", stress.get("Q_A_plus_Q_R_residual")))
    source_residual = _float(source_localization.get("source_localization_residual_nats"))
    modular_charge = _float(source_localization.get("modular_source_charge_nats"))
    source_check = _source_contract_check(
        source_localization,
        manifest=manifest,
        geometry=geometry,
        modular_charge=modular_charge,
        source_residual=source_residual,
        tolerance=float(tolerance),
    )
    blockers.extend(source_check["blockers"])
    source_receipt = bool(source_check["source_receipt"])
    if not source_receipt:
        blockers.append("issue_319_source_localization_saturation_not_certified")

    stress_closed = bool(
        moment_receipt
        and variational_moment_receipt
        and packet_check["mass_shell_receipt"]
        and transport_receipt
        and channel_receipt
        and stress_residual is not None
        and stress_residual <= float(tolerance)
        and exchange_residual is not None
        and exchange_residual <= float(tolerance)
    )
    if not stress_closed:
        blockers.append("stress_energy_closure_not_certified")

    gamma_rec = _float(repair.get("Gamma_rec"))
    repair_step_gamma = _float(
        repair.get(
            "gamma_repair_step",
            repair.get("gamma_rec", repair.get("spectral_gap")),
        )
    )
    gamma_nonzero = bool(gamma_rec is not None and gamma_rec > float(tolerance))
    recipient_states = [row for row in packet_rows if str(row.get("label")) in RECIPIENT_PACKET_LABELS]
    exchange_off_receipt = bool(
        not gamma_nonzero
        and _residual_receipt(repair, ("exchange_off_residual", "Q_A_residual"), tolerance=float(tolerance))
    )
    recipient_receipt = bool(
        gamma_nonzero
        and recipient_states
        and _residual_receipt(stress, ("recipient_stress_residual", "T_R_residual"), tolerance=float(tolerance))
        and stress_closed
    )
    exchange_current_receipt = bool(
        (exchange_residual is not None and exchange_residual <= float(tolerance))
        and _residual_receipt(stress, ("recipient_exchange_residual", "Q_A_plus_Q_R_residual"), tolerance=float(tolerance))
    )
    if gamma_nonzero and not recipient_receipt:
        blockers.append("explicit_recipient_stress_missing_for_nonzero_Gamma_rec")
    if gamma_nonzero and not exchange_current_receipt:
        blockers.append("exchange_current_closure_not_certified")
    if gamma_rec is not None and gamma_rec < -float(tolerance):
        blockers.append("Gamma_rec_negative")

    physical_clock_receipt = bool(
        repair.get(PHYSICAL_CLOCK_RECEIPT, False)
        or repair.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
        or causal.get(PHYSICAL_CLOCK_RECEIPT, False)
        or causal.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
    )
    active_fiber_receipt = bool(
        repair.get(ACTIVE_FIBER_RECEIPT, False)
        or repair.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
        or causal.get(ACTIVE_FIBER_RECEIPT, False)
        or causal.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
    )
    conserved_sector_receipt = bool(
        repair.get(CONSERVED_SECTOR_DECOMPOSITION_RECEIPT, False)
        or stress.get(CONSERVED_SECTOR_DECOMPOSITION_RECEIPT, False)
    )
    common_parent_response_receipt = bool(
        causal.get(COMMON_PARENT_RESPONSE_POLE_RECEIPT, False)
        or causal.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
        or repair.get(COMMON_PARENT_RESPONSE_POLE_RECEIPT, False)
        or repair.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
    )
    if gamma_nonzero and not physical_clock_receipt:
        blockers.append("physical_clock_missing_for_promoted_Gamma_rec")
    if gamma_nonzero and not active_fiber_receipt:
        blockers.append("active_fiber_missing_for_promoted_Gamma_rec")
    if gamma_nonzero and not conserved_sector_receipt:
        blockers.append("conserved_sector_decomposition_missing_for_promoted_Gamma_rec")
    if gamma_nonzero and not common_parent_response_receipt:
        blockers.append("common_parent_response_pole_missing_for_promoted_Gamma_rec")

    detailed_balance_residual = _float(repair.get("detailed_balance_residual"))
    detailed_balance = bool(detailed_balance_residual is not None and detailed_balance_residual <= float(tolerance))
    nonequilibrium_residual = _float(repair.get("nonequilibrium_entropy_production_residual"))
    nonequilibrium_receipt = bool(
        repair.get("declared_nonequilibrium_exchange", False)
        and nonequilibrium_residual is not None
        and nonequilibrium_residual <= float(tolerance)
        and _nonempty(repair.get("affinity_source"))
    )
    thermodynamics_receipt = bool(exchange_off_receipt or detailed_balance or nonequilibrium_receipt)
    if gamma_nonzero and not thermodynamics_receipt:
        blockers.append("reaction_thermodynamics_not_certified")

    characteristic_speed = _float(causal.get("characteristic_speed_bound"))
    response_stability_residual = _float(
        causal.get("response_stability_residual", causal.get("lyapunov_residual"))
    )
    retarded_support_residual = _float(causal.get("retarded_support_residual"))
    causal_matrices = _matrices_present(
        causal,
        ("kinetic_matrix", "damping_matrix", "propagation_matrix", "source_matrix", "output_matrix"),
    )
    response_stability_receipt = bool(
        response_stability_residual is not None and response_stability_residual <= float(tolerance)
    )
    retarded_response_receipt = bool(
        retarded_support_residual is not None and retarded_support_residual <= float(tolerance)
    )
    finite_domain_receipt = _residual_receipt(
        causal,
        ("finite_domain_residual", "causal_impulse_support_residual"),
        tolerance=float(tolerance),
    )
    subluminal_receipt = bool(characteristic_speed is not None and characteristic_speed <= 1.0 + float(tolerance))
    causal_receipt = bool(
        characteristic_speed is not None
        and characteristic_speed <= 1.0 + float(tolerance)
        and causal_matrices
        and response_stability_receipt
        and retarded_response_receipt
        and finite_domain_receipt
    )
    if not causal_receipt:
        blockers.append("causal_response_not_certified")

    gauge_residual = _float(gauge.get("gauge_consistency_residual"))
    gauge_presentations = _int(
        gauge.get("independent_gauge_presentation_count", gauge.get("independent_gauge_presentations"))
    )
    cosmological_gauge_receipt = bool(
        gauge_residual is not None
        and gauge_residual <= float(tolerance)
        and gauge_presentations is not None
        and gauge_presentations >= 2
        and _residual_receipt(
            gauge,
            ("gauge_invariant_variable_residual", "rest_frame_density_residual"),
            tolerance=float(tolerance),
        )
    )
    if not cosmological_gauge_receipt:
        blockers.append("cosmological_gauge_invariance_not_certified")

    refinement_residual = _float(refinement.get("convergence_residual", refinement.get("stress_response_convergence_residual")))
    regulator_levels = _int(refinement.get("regulator_level_count", refinement.get("regulator_levels")))
    refinement_receipt = bool(
        refinement_residual is not None
        and refinement_residual <= float(tolerance)
        and regulator_levels is not None
        and regulator_levels >= 3
    )
    if not refinement_receipt:
        blockers.append("refinement_convergence_not_certified")

    cdm_residual = _float(cdm_limit.get("cdm_limit_residual"))
    cdm_operator_residual = _float(cdm_limit.get("cdm_operator_residual"))
    cdm_limit_receipt = bool(
        cdm_residual is not None
        and cdm_residual <= float(tolerance)
        and cdm_operator_residual is not None
        and cdm_operator_residual <= float(tolerance)
    )
    if not cdm_limit_receipt:
        blockers.append("cdm_limit_recovery_not_certified")

    source_hash = str(frozen.get("source_hash") or manifest.get("source_hash") or "")
    frozen_receipt = False

    exchange_branch_receipt = bool(exchange_off_receipt or (recipient_receipt and exchange_current_receipt and thermodynamics_receipt))
    gamma_promotion_receipt = bool(
        (not gamma_nonzero)
        or (
            physical_clock_receipt
            and active_fiber_receipt
            and conserved_sector_receipt
            and common_parent_response_receipt
        )
    )
    if not exchange_branch_receipt:
        blockers.append("exchange_branch_not_certified")
    if not gamma_promotion_receipt:
        blockers.append("Gamma_rec_promotion_not_certified")
    parent_blockers = [blocker for blocker in blockers if blocker != "frozen_likelihood_protocol_not_certified"]
    parent_receipt = bool(
        not parent_blockers
        and packet_check["kinematics_receipt"]
        and moment_receipt
        and variational_moment_receipt
        and local_frame_receipt
        and quotient_receipt
        and stress_closed
        and finite_domain_receipt
        and subluminal_receipt
        and retarded_response_receipt
        and refinement_receipt
        and cdm_limit_receipt
        and exchange_branch_receipt
        and gamma_promotion_receipt
    )
    return {
        "mode": "finite_covariant_collar_packet_parent_v0",
        PARENT_RECEIPT: parent_receipt,
        ISSUE_319_CONDITIONAL_SOURCE_RECEIPT: source_receipt,
        SOURCE_ROUTE_RECEIPT: source_check["source_route_receipt"],
        ENTROPY_UNIT_RECEIPT: source_check["entropy_unit_receipt"],
        MODULAR_NONADDITIVITY_IDENTITY_RECEIPT: source_check["modular_identity_receipt"],
        CMI_FIRST_VARIATION_CLASSIFICATION_RECEIPT: source_check["first_variation_receipt"],
        CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT: source_check["cmi_matching_receipt"],
        FIXED_REFERENCE_MODULAR_ENERGY_RECEIPT: source_check["fixed_reference_receipt"],
        SOURCE_LOCALIZATION_SATURATION_RECEIPT: source_check["source_localization_receipt"],
        BW_BALL_NORMALIZATION_RECEIPT: source_check["bw_normalization_receipt"],
        PHYSICAL_DIAMOND_SCALE_RECEIPT: source_check["physical_scale_receipt"],
        ELL4_SCALING_PLATEAU_RECEIPT: source_check["ell4_scaling_receipt"],
        COVER_INDEPENDENCE_RECEIPT: source_check["cover_independence_receipt"],
        STRESS_TOMOGRAPHY_RECEIPT: source_check["stress_tomography_receipt"],
        MODULAR_SOURCE_CHARGE_RECEIPT: modular_charge is not None,
        FINITE_PACKET_KINEMATICS_RECEIPT: packet_check["kinematics_receipt"],
        "PACKET_KINEMATICS_RECEIPT": packet_check["kinematics_receipt"],
        PACKET_MASS_SHELL_RECEIPT: packet_check["mass_shell_receipt"],
        SM_CURRENT_NULL_RECEIPT: packet_check["sm_current_null_receipt"],
        TRANSPORT_COVARIANCE_RECEIPT: transport_receipt,
        CHANNEL_FOUR_MOMENTUM_RECEIPT: channel_receipt,
        FINITE_PACKET_STRESS_READOUT_RECEIPT: moment_receipt,
        VARIATIONAL_MOMENT_STRESS_AGREEMENT_RECEIPT: variational_moment_receipt,
        LOCAL_FRAME_COVARIANCE_RECEIPT: local_frame_receipt,
        CARRIER_QUOTIENT_INVARIANCE_RECEIPT: quotient_receipt,
        TOTAL_STRESS_CLOSURE_RECEIPT: stress_closed,
        STRESS_CLOSURE_RECEIPT: stress_closed,
        GAUGE_INDEPENDENCE_RECEIPT: cosmological_gauge_receipt,
        COSMOLOGICAL_GAUGE_INVARIANCE_RECEIPT: cosmological_gauge_receipt,
        EXPLICIT_RECIPIENT_STRESS_RECEIPT: recipient_receipt,
        EXCHANGE_CURRENT_CLOSURE_RECEIPT: exchange_current_receipt,
        REPAIR_EXCHANGE_OFF_RECEIPT: exchange_off_receipt,
        DETAILED_BALANCE_RECEIPT: detailed_balance,
        DECLARED_NONEQUILIBRIUM_EXCHANGE_RECEIPT: nonequilibrium_receipt,
        PHYSICAL_CLOCK_RECEIPT: physical_clock_receipt,
        ACTIVE_FIBER_RECEIPT: active_fiber_receipt,
        CONSERVED_SECTOR_DECOMPOSITION_RECEIPT: conserved_sector_receipt,
        COMMON_PARENT_RESPONSE_POLE_RECEIPT: common_parent_response_receipt,
        CAUSAL_RESPONSE_RECEIPT: causal_receipt,
        FINITE_DOMAIN_OF_DEPENDENCE_RECEIPT: finite_domain_receipt,
        SUBLUMINAL_CHARACTERISTICS_RECEIPT: subluminal_receipt,
        RETARDED_RESPONSE_RECEIPT: retarded_response_receipt,
        RESPONSE_STABILITY_RECEIPT: response_stability_receipt,
        REFINEMENT_CONVERGENCE_RECEIPT: refinement_receipt,
        "CDM_LIMIT_RECOVERY_RECEIPT": cdm_limit_receipt,
        FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT: frozen_receipt,
        "Gamma_rec_nonzero": gamma_nonzero,
        "Gamma_rec_status": "PHYSICAL_KERNEL" if gamma_promotion_receipt and gamma_nonzero else (
            "UNPROMOTED_REPAIR_STEP_DIAGNOSTIC" if repair_step_gamma is not None else (
                "REPAIR_EXCHANGE_OFF" if not gamma_nonzero else "UNPROMOTED_REPAIR_STEP_DIAGNOSTIC"
            )
        ),
        "repair_step_gamma_diagnostic": repair_step_gamma,
        "relaxation_rate_artifact_types": {
            "DISCRETE_STEP_EIGENVALUE": repair.get("discrete_step_eigenvalue"),
            "DISCRETE_STEP_GAP": repair.get("discrete_step_gap"),
            "LOGARITHMIC_STEP_DECAY": repair.get("logarithmic_step_decay", repair_step_gamma),
            "CONTINUOUS_GENERATOR_EIGENVALUE": repair.get("continuous_generator_eigenvalue"),
            "PHYSICAL_RELAXATION_RATE": gamma_rec,
        },
        "source_hash": source_hash or None,
        "solver_hash": None,
        "likelihood_hash": None,
        "packet_state_count": len(packet_rows),
        "anomaly_state_count": packet_check["anomaly_state_count"],
        "recipient_state_count": len(recipient_states),
        "packet_mass_shell_max_abs_residual": packet_check["mass_shell_max_abs_residual"],
        "channel_four_momentum_max_abs_residual": channel_residual,
        "stress_energy_closure_residual": stress_residual,
        "exchange_current_residual": exchange_residual,
        "source_localization_residual_nats": source_residual,
        "modular_source_charge_nats": modular_charge,
        "source_route": source_check["source_route"],
        "entropy_base": source_check["entropy_base"],
        "proper_diamond_radius_ell": source_check["proper_diamond_radius_ell"],
        "modular_identity_residual": source_check["modular_identity_residual"],
        "matching_residual": source_check["matching_residual"],
        "heldout_quadraticity_residual": source_check["heldout_quadraticity_residual"],
        "detailed_balance_residual": detailed_balance_residual,
        "nonequilibrium_entropy_production_residual": nonequilibrium_residual,
        "characteristic_speed_bound": characteristic_speed,
        "response_stability_residual": response_stability_residual,
        "retarded_support_residual": retarded_support_residual,
        "gauge_consistency_residual": gauge_residual,
        "independent_gauge_presentation_count": gauge_presentations,
        "refinement_convergence_residual": refinement_residual,
        "regulator_level_count": regulator_levels,
        "cdm_operator_residual": cdm_operator_residual,
        "parent_blockers": parent_blockers,
        "blockers": blockers,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite covariant collar-packet parent source contract. Passing this report licenses source "
            "functions for a Boltzmann handoff; it is not itself a CMB likelihood result, and the "
            "frozen likelihood receipt is a later physical-CMB promotion gate."
        ),
    }


def write_finite_covariant_collar_packet_parent_report(source: Path, out: Path) -> dict[str, Any]:
    artifact = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("finite parent artifact must be a JSON object")
    report = finite_covariant_collar_packet_parent_report(artifact)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _metric(data: dict[str, Any]) -> np.ndarray | None:
    for key in ("metric", "g", "metric_matrix"):
        if key not in data:
            continue
        try:
            metric = np.asarray(data[key], dtype=float)
        except (TypeError, ValueError):
            return None
        if metric.shape != (4, 4) or not np.all(np.isfinite(metric)):
            return None
        if abs(float(np.linalg.det(metric))) <= 1.0e-18:
            return None
        return metric
    return None


def _metric_quadratic(metric: np.ndarray, vector: np.ndarray) -> float:
    return float(vector @ metric @ vector)


def _packet_state_check(
    rows: list[dict[str, Any]],
    *,
    metric: np.ndarray | None,
    tolerance: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not rows:
        blockers.append("packet_states_missing")
    anomaly_count = 0
    mass_shell_residuals: list[float] = []
    kinematics_ok = bool(metric is not None)
    sm_current_null = bool(rows)
    for index, row in enumerate(rows):
        density = _float(row.get("rho", row.get("density", row.get("background_weight"))))
        if density is None or density <= 0.0:
            blockers.append(f"packet_{index}_density_invalid")
        occupation = _float(row.get("occupation", row.get("f")))
        if occupation is None or occupation < 0.0:
            blockers.append(f"packet_{index}_occupation_invalid")
            kinematics_ok = False
        invariant_weight = _float(row.get("invariant_weight", row.get("weight", row.get("w_z"))))
        if invariant_weight is None or invariant_weight <= 0.0:
            blockers.append(f"packet_{index}_invariant_weight_invalid")
            kinematics_ok = False
        if str(row.get("label")) in ANOMALY_PACKET_LABELS:
            anomaly_count += 1
            charges = row.get("SM_charges", row.get("standard_model_charges"))
            if not _neutral_sm_charges(charges):
                blockers.append(f"packet_{index}_standard_model_charges_not_null")
                sm_current_null = False
        velocity = row.get("u_mu", row.get("four_velocity"))
        if velocity is not None:
            velocity_array = np.asarray(velocity, dtype=float)
            if velocity_array.shape != (4,) or not np.all(np.isfinite(velocity_array)) or velocity_array[0] <= 0.0:
                blockers.append(f"packet_{index}_four_velocity_not_future_directed")
                kinematics_ok = False
            else:
                norm = (
                    _metric_quadratic(metric, velocity_array)
                    if metric is not None
                    else -velocity_array[0] ** 2 + float(np.dot(velocity_array[1:], velocity_array[1:]))
                )
                if abs(norm + 1.0) > 1.0e-6:
                    blockers.append(f"packet_{index}_four_velocity_not_unit_timelike")
                    kinematics_ok = False
        momentum = row.get("momentum_local", row.get("p_mu", row.get("four_momentum")))
        mass = _float(row.get("mass", row.get("m")))
        if momentum is None or mass is None or metric is None:
            blockers.append(f"packet_{index}_mass_shell_data_missing")
            kinematics_ok = False
            continue
        try:
            momentum_array = np.asarray(momentum, dtype=float)
        except (TypeError, ValueError):
            blockers.append(f"packet_{index}_momentum_invalid")
            kinematics_ok = False
            continue
        if momentum_array.shape != (4,) or not np.all(np.isfinite(momentum_array)) or momentum_array[0] <= 0.0:
            blockers.append(f"packet_{index}_momentum_not_future_directed")
            kinematics_ok = False
            continue
        if mass < 0.0:
            blockers.append(f"packet_{index}_mass_negative")
            kinematics_ok = False
            continue
        residual = abs(_metric_quadratic(metric, momentum_array) + mass * mass)
        mass_shell_residuals.append(residual)
        if residual > tolerance:
            blockers.append(f"packet_{index}_mass_shell_residual_too_large")
            kinematics_ok = False
    if anomaly_count <= 0:
        blockers.append("anomaly_packet_states_missing")
    mass_shell_receipt = bool(rows and metric is not None and mass_shell_residuals and max(mass_shell_residuals) <= tolerance)
    return {
        "blockers": blockers,
        "anomaly_state_count": anomaly_count,
        "kinematics_receipt": bool(kinematics_ok and rows),
        "mass_shell_receipt": mass_shell_receipt,
        "sm_current_null_receipt": bool(sm_current_null and anomaly_count > 0),
        "mass_shell_max_abs_residual": max(mass_shell_residuals) if mass_shell_residuals else None,
    }


def _source_contract_check(
    data: dict[str, Any],
    *,
    manifest: dict[str, Any],
    geometry: dict[str, Any],
    modular_charge: float | None,
    source_residual: float | None,
    tolerance: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    route = _source_route(data, manifest)
    route_receipt = bool(route in SOURCE_ROUTES)
    if route is None:
        blockers.append("source_route_missing")
    elif not route_receipt:
        blockers.append("source_route_invalid_or_automatic")

    entropy_base = str(data.get("entropy_base", manifest.get("entropy_base", ""))).strip().lower()
    hbar = _float(data.get("hbar", manifest.get("hbar")))
    c_value = _float(data.get("c", manifest.get("c")))
    length_unit = str(data.get("length_unit", manifest.get("length_unit", ""))).strip()
    proper_ell = _float(
        data.get(
            "proper_diamond_radius_ell",
            geometry.get("proper_diamond_radius_ell", geometry.get("ell")),
        )
    )
    bits_factor = data.get("bits_to_nats_factor", data.get("entropy_to_nats_factor", data.get("log_base_conversion")))
    entropy_units = bool(
        entropy_base in {"nats", "bits"}
        and hbar is not None
        and hbar > 0.0
        and c_value is not None
        and c_value > 0.0
        and length_unit
        and proper_ell is not None
        and proper_ell > 0.0
        and (entropy_base != "bits" or _ln2_declared(bits_factor))
    )
    if not entropy_units:
        blockers.append("entropy_or_physical_units_missing")

    modular_identity_residual = _modular_identity_residual(data)
    cmi_kind = str(data.get("cmi_kind", data.get("CMI_kind", ""))).strip()
    cmi_value = _float(data.get("CMI", data.get("cmi", data.get("R_C_info_nats"))))
    modular_identity_receipt = bool(
        cmi_kind in EXACT_CMI_KINDS
        and cmi_value is not None
        and cmi_value >= -tolerance
        and modular_identity_residual is not None
        and modular_identity_residual <= tolerance
    )
    if not modular_identity_receipt:
        blockers.append("modular_nonadditivity_identity_not_certified")

    first_variation = str(data.get("first_variation_classification", "")).strip()
    first_variation_receipt = first_variation in {"MARKOV_BRANCH_QUADRATIC", "NON_MARKOV_BACKGROUND_DECLARED"}
    if not first_variation_receipt:
        blockers.append("cmi_first_variation_classification_missing")

    matching_residual = _float(data.get("matching_residual", data.get("cmi_to_modular_matching_residual")))
    matching_hash = data.get("matching_theorem_hash", data.get("matching_hypothesis_hash"))
    cmi_matching_receipt = bool(
        route == "NONLINEAR_CMI_STRESS"
        and _nonempty(matching_hash)
        and matching_residual is not None
        and matching_residual <= tolerance
    )
    fixed_reference_hash = data.get("fixed_reference_state_hash", manifest.get("reference_state_hash"))
    anomalous_modular_energy = _float(
        data.get("anomalous_modular_energy_variation", data.get("M_C_anom_nats", modular_charge))
    )
    fixed_reference_receipt = bool(
        route == "FIXED_REFERENCE_MODULAR_ENERGY"
        and _nonempty(fixed_reference_hash)
        and anomalous_modular_energy is not None
    )
    if route == "NONLINEAR_CMI_STRESS" and not cmi_matching_receipt:
        blockers.append("cmi_to_modular_source_matching_missing")
    if route == "FIXED_REFERENCE_MODULAR_ENERGY" and not fixed_reference_receipt:
        blockers.append("fixed_reference_modular_energy_missing")

    source_localization_receipt = bool(
        modular_charge is not None
        and modular_charge >= -tolerance
        and source_residual is not None
        and source_residual <= tolerance
    )
    if not source_localization_receipt:
        blockers.append("source_localization_residual_not_certified")

    bw_residual = _float(data.get("bw_ball_normalization_residual"))
    bw_receipt = bool(bw_residual is not None and bw_residual <= tolerance)
    if not bw_receipt:
        blockers.append("bw_ball_normalization_not_certified")

    scale_receipt = bool(
        proper_ell is not None
        and proper_ell > 0.0
        and _float(data.get("uv_scale", geometry.get("uv_scale"))) is not None
        and _float(data.get("collar_width", geometry.get("collar_width"))) is not None
        and _float(data.get("curvature_scale_bound", geometry.get("curvature_scale_bound"))) is not None
        and _float(data.get("gradient_scale_bound", geometry.get("gradient_scale_bound"))) is not None
    )
    if not scale_receipt:
        blockers.append("physical_diamond_scale_hierarchy_missing")

    ell4_residual = _float(data.get("ell4_scaling_plateau_residual"))
    ell4_receipt = bool(ell4_residual is not None and ell4_residual <= tolerance)
    if not ell4_receipt:
        blockers.append("ell4_scaling_plateau_not_certified")

    cover_residual = _float(data.get("cover_independence_residual"))
    cover_receipt = bool(cover_residual is not None and cover_residual <= tolerance)
    if not cover_receipt:
        blockers.append("cover_independence_not_certified")

    probe_rank = _int(data.get("timelike_probe_rank", data.get("stress_tomography_rank")))
    heldout_residual = _float(data.get("heldout_quadraticity_residual", data.get("stress_tomography_residual")))
    tomography_receipt = bool(
        probe_rank is not None
        and probe_rank >= 10
        and heldout_residual is not None
        and heldout_residual <= tolerance
    )
    if not tomography_receipt:
        blockers.append("stress_tomography_not_certified")

    route_specific = bool(
        (route == "FIXED_REFERENCE_MODULAR_ENERGY" and fixed_reference_receipt)
        or (route == "NONLINEAR_CMI_STRESS" and cmi_matching_receipt)
    )
    source_receipt = bool(
        route_receipt
        and entropy_units
        and modular_identity_receipt
        and first_variation_receipt
        and route_specific
        and source_localization_receipt
        and bw_receipt
        and scale_receipt
        and ell4_receipt
        and cover_receipt
        and tomography_receipt
    )
    return {
        "blockers": blockers,
        "source_receipt": source_receipt,
        "source_route_receipt": route_receipt,
        "entropy_unit_receipt": entropy_units,
        "modular_identity_receipt": modular_identity_receipt,
        "first_variation_receipt": first_variation_receipt,
        "cmi_matching_receipt": cmi_matching_receipt,
        "fixed_reference_receipt": fixed_reference_receipt,
        "source_localization_receipt": source_localization_receipt,
        "bw_normalization_receipt": bw_receipt,
        "physical_scale_receipt": scale_receipt,
        "ell4_scaling_receipt": ell4_receipt,
        "cover_independence_receipt": cover_receipt,
        "stress_tomography_receipt": tomography_receipt,
        "source_route": route,
        "entropy_base": entropy_base or None,
        "proper_diamond_radius_ell": proper_ell,
        "modular_identity_residual": modular_identity_residual,
        "matching_residual": matching_residual,
        "heldout_quadraticity_residual": heldout_residual,
    }


def _transport_receipt(data: dict[str, Any]) -> bool:
    maps = data.get("parallel_transport_maps", data.get("transport_maps"))
    if not isinstance(maps, list) or not maps:
        return False
    for item in maps:
        matrix = item.get("matrix") if isinstance(item, dict) else item
        try:
            array = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError):
            return False
        if array.shape != (4, 4) or not np.all(np.isfinite(array)):
            return False
    return True


def _channel_four_momentum_receipt(data: dict[str, Any], *, tolerance: float) -> tuple[bool, float | None]:
    rows = data.get("channels")
    if not isinstance(rows, list) or not rows:
        return False, None
    residuals: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return False, None
        value = row.get("transported_delta_p", row.get("delta_p", row.get("four_momentum_residual")))
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return False, None
        if array.shape != (4,) or not np.all(np.isfinite(array)):
            return False, None
        residuals.append(float(np.max(np.abs(array))))
    max_residual = max(residuals) if residuals else None
    return bool(max_residual is not None and max_residual <= tolerance), max_residual


def _source_route(data: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    values = [
        str(value).strip()
        for value in (data.get("source_route"), manifest.get("source_route"))
        if str(value or "").strip()
    ]
    if not values:
        return None
    unique = sorted(set(values))
    if len(unique) != 1:
        return "__AMBIGUOUS__"
    return unique[0]


def _modular_identity_residual(data: dict[str, Any]) -> float | None:
    supplied = _float(data.get("modular_identity_residual", data.get("DeltaK_identity_residual")))
    if supplied is not None:
        return abs(supplied)
    s_ab = _float(data.get("S_AB"))
    s_bd = _float(data.get("S_BD"))
    s_b = _float(data.get("S_B"))
    s_abd = _float(data.get("S_ABD"))
    cmi = _float(data.get("CMI", data.get("cmi", data.get("R_C_info_nats"))))
    delta_k = _float(data.get("DeltaK_expectation", data.get("deltaK_expectation")))
    residuals: list[float] = []
    if None not in (s_ab, s_bd, s_b, s_abd, cmi):
        expected = float(s_ab) + float(s_bd) - float(s_b) - float(s_abd)
        residuals.append(abs(expected - float(cmi)))
    if cmi is not None and delta_k is not None:
        residuals.append(abs(float(delta_k) - float(cmi)))
    return max(residuals) if residuals else None


def _ln2_declared(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"ln2", "log(2)", "natural_log_2"}
    parsed = _float(value)
    return bool(parsed is not None and abs(parsed - float(np.log(2.0))) <= 1.0e-12)


def _neutral_sm_charges(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    aliases = {
        "EM": ("EM", "em", "q_EM", "electric", "electric_charge"),
        "color": ("color", "q_color", "color_charge"),
        "weak": ("weak", "q_weak", "weak_charge"),
    }
    for keys in aliases.values():
        found = False
        for key in keys:
            if key not in value:
                continue
            found = True
            charge = value.get(key)
            try:
                array = np.asarray(charge, dtype=float)
            except (TypeError, ValueError):
                return False
            if array.size == 0 or not np.all(np.isfinite(array)) or float(np.max(np.abs(array))) > 1.0e-12:
                return False
            break
        if not found:
            return False
    return True


def _moment_receipt(data: dict[str, Any], *, tolerance: float) -> dict[str, bool]:
    stress_residual = _float(data.get("stress_moment_residual", data.get("moment_reconstruction_residual")))
    divergence_residual = _float(data.get("finite_divergence_operator_residual"))
    volume_residual = _float(data.get("stress_volume_weight_residual"))
    variational_residual = _float(
        data.get("variational_moment_agreement_residual", data.get("metric_moment_stress_residual"))
    )
    stress_readout = bool(
        stress_residual is not None
        and stress_residual <= tolerance
        and divergence_residual is not None
        and divergence_residual <= tolerance
        and volume_residual is not None
        and volume_residual <= tolerance
    )
    variational = bool(variational_residual is not None and variational_residual <= tolerance)
    return {
        "stress_readout_receipt": stress_readout,
        "variational_moment_receipt": variational,
    }


def _residual_receipt(data: dict[str, Any], keys: tuple[str, ...], *, tolerance: float) -> bool:
    for key in keys:
        residual = _float(data.get(key))
        if residual is not None:
            return residual <= tolerance
    return False


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _matrices_present(data: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key not in data:
            return False
        try:
            array = np.asarray(data[key], dtype=float)
        except (TypeError, ValueError):
            return False
        if array.size == 0 or not np.all(np.isfinite(array)):
            return False
    return True


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_sha256_hash(value: Any) -> bool:
    text = str(value or "").strip()
    if not text.startswith("sha256:"):
        return False
    digest = text.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest)
