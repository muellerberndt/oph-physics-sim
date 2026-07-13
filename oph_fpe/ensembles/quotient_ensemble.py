from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from oph_fpe.evidence.hashes import stable_json_hash


CLAIM_TIERS: dict[str, str] = {
    "E0": "seed noise, proposal noise, repair jitter",
    "E1": "conventional reference ensemble",
    "E2": "OPH-native quotient ensemble",
    "E3": "OPH vacuum",
    "E4": "OPH primordial field",
    "E5": "observable cosmological prediction",
}

NONPHYSICAL_REPRESENTATIVE_KEYS = frozenset(
    {
        "gauge_representative",
        "port_label",
        "port_labels",
        "mesh_label",
        "mesh_labels",
        "shard_id",
        "worker_id",
        "queue_order",
        "repair_schedule_id",
        "repair_iteration_id",
        "retry_counter",
        "timestamp",
        "wall_clock_timestamp",
        "hidden_carrier_coordinate",
        "hidden_carrier_coordinates",
        "inert_ancilla_label",
        "inert_ancillary_label",
    }
)

RECEIPT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "E2": (
        "TRACIALLY_POINTED_QUOTIENT_RECEIPT",
        "REFERENCE_STATE_REFINEMENT_RECEIPT",
        "SOURCE_ONLY_CONSTRAINT_LEDGER_RECEIPT",
        "REPRESENTATIVE_LIFT_FIREWALL_RECEIPT",
        "QUOTIENT_LUMPABILITY_RECEIPT",
        "SAMPLER_CORRECTNESS_RECEIPT",
        "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT",
    ),
    "E3": (
        "SOURCE_EUCLIDEAN_SLAB_RECEIPT",
        "TRANSFER_VACUUM_RECEIPT",
        "TRANSFER_REFINEMENT_RECEIPT",
    ),
    "E4": (
        "TOTAL_STRESS_CLOSURE_RECEIPT",
        "SINGLE_CLOCK_NORMAL_FORM_RECEIPT",
        "ENTROPY_REPAIR_GAP_RECEIPT",
        "CURVATURE_EVOLUTION_RECEIPT",
        "ADIABATIC_MODE_RECEIPT",
        "SCREEN_TO_RADIAL_LIFT_RECEIPT",
        "RADIAL_NULL_SPACE_REPORT_RECEIPT",
        "FORWARD_PROJECTION_RESIDUAL_RECEIPT",
    ),
    "E5": (
        "SOURCE_PROVENANCE_RECEIPT",
        "POOLED_SOURCE_REDUCER_RECEIPT",
        "BOLTZMANN_TRANSFER_RECEIPT",
        "NO_DATA_USE_RECEIPT",
        "FROZEN_LIKELIHOOD_RECEIPT",
    ),
}


@dataclass(frozen=True)
class QuotientEnsembleManifest:
    ensemble_id: str
    claim_tier: str
    regulator_id: str
    representative_schema_hash: str
    gauge_action_hash: str
    quotient_canonicalizer_hash: str
    base_measure_definition: str
    action_definition: str
    action_coefficients_hash: str
    coarse_map_hashes: tuple[str, ...]
    zero_mode_projector_hash: str
    amplitude_convention: str
    sampler_kernel_hash: str
    partition_invariant_random_event_schema: str
    smoothing_policy_hash: str
    source_provenance_hash: str
    explicit_nonclaims: tuple[str, ...]

    def as_jsonable(self) -> dict[str, Any]:
        checked_claim_tier(self.claim_tier)
        data = asdict(self)
        data["claim_tier_meaning"] = CLAIM_TIERS[self.claim_tier]
        data["ensemble_manifest_hash"] = stable_hash(data)
        return data


def checked_claim_tier(value: str) -> str:
    tier = str(value)
    if tier not in CLAIM_TIERS:
        raise ValueError(f"unknown ensemble claim tier: {value}")
    return tier


def claim_tier_gate(
    *,
    claim_tier: str,
    receipts: Mapping[str, bool] | None = None,
    explicit_nonclaims: Sequence[str] = (),
) -> dict[str, Any]:
    tier = checked_claim_tier(claim_tier)
    receipt_map = {str(key): bool(value) for key, value in (receipts or {}).items()}
    required = tuple(_requirements_through(tier))
    blockers = [name for name in required if not receipt_map.get(name, False)]
    promotion_allowed = bool(required) and not blockers and tier not in {"E0", "E1"}
    if tier in {"E0", "E1"}:
        blockers.append(f"{tier}_is_not_a_physical_promotion_tier")
    return {
        "claim_tier": tier,
        "claim_tier_meaning": CLAIM_TIERS[tier],
        "promotion_allowed": False if tier in {"E0", "E1"} else promotion_allowed,
        "required_receipts": list(required),
        "promotion_blockers": blockers,
        "explicit_nonclaims": list(explicit_nonclaims),
    }


def canonicalize_representative(
    representative: Mapping[str, Any],
    *,
    dropped_keys: set[str] | None = None,
) -> dict[str, Any]:
    drop = NONPHYSICAL_REPRESENTATIVE_KEYS | set(dropped_keys or set())
    canonical = _canonical_value(representative, drop)
    return {
        "quotient_key": stable_hash(canonical),
        "canonical_payload": canonical,
        "canonicalizer_hash": stable_hash({"dropped_keys": sorted(drop), "version": "quotient-canonicalizer-v1"}),
    }


def canonicalizer_receipt(
    representative: Mapping[str, Any],
    *,
    orbit_size: int = 0,
    stabilizer_size: int = 0,
    collision_check: bool = True,
    dropped_keys: set[str] | None = None,
) -> dict[str, Any]:
    canonical = canonicalize_representative(representative, dropped_keys=dropped_keys)
    return {
        "TRACIALLY_POINTED_QUOTIENT_RECEIPT": False,
        "representative_hash": stable_hash(representative),
        "quotient_key": canonical["quotient_key"],
        "canonicalizer_hash": canonical["canonicalizer_hash"],
        "orbit_size": int(orbit_size),
        "stabilizer_size": int(stabilizer_size),
        "collision_check": "pass" if collision_check else "fail",
        "nonphysical_keys_removed": sorted((NONPHYSICAL_REPRESENTATIVE_KEYS | set(dropped_keys or set()))),
    }


def representative_lift_firewall_receipt(
    *,
    uniform_representative_sampling_declared: bool,
    representative_counting_declared_as_physical: bool,
    orbit_size_correction_applied: bool,
    max_orbit_bias: float,
) -> dict[str, Any]:
    pass_receipt = (
        (not uniform_representative_sampling_declared or representative_counting_declared_as_physical)
        and (representative_counting_declared_as_physical or orbit_size_correction_applied)
        and float(max_orbit_bias) <= 0.0
    )
    return {
        "REPRESENTATIVE_LIFT_FIREWALL_RECEIPT": bool(pass_receipt),
        "representative_lift_check": "pass" if pass_receipt else "fail",
        "max_orbit_bias": float(max_orbit_bias),
        "uniform_representative_sampling_declared": bool(uniform_representative_sampling_declared),
        "representative_counting_declared_as_physical": bool(representative_counting_declared_as_physical),
        "orbit_size_correction_applied": bool(orbit_size_correction_applied),
    }


def quotient_lumpability_receipt(
    *,
    max_lumpability_residual: float,
    representative_kernel_hash: str,
    quotient_kernel_hash: str,
) -> dict[str, Any]:
    residual = float(max_lumpability_residual)
    passed = residual <= 0.0
    return {
        "QUOTIENT_LUMPABILITY_RECEIPT": bool(passed),
        "quotient_lumpability_check": "pass" if passed else "fail",
        "max_lumpability_residual": residual,
        "representative_kernel_hash": str(representative_kernel_hash),
        "quotient_kernel_hash": str(quotient_kernel_hash),
    }


def sampler_correctness_receipt(
    *,
    sampler_type: str,
    target_weight_hash: str,
    proposal_kernel_hash: str,
    acceptance_rule_hash: str,
    hastings_asymmetry_included: bool,
    detailed_balance_max_error: float,
    stationarity_tv_error: float,
    irreducible: bool,
    aperiodic: bool,
    spectral_gap_estimate: float | None = None,
    autocorrelation_time: float | None = None,
    effective_sample_size: int | None = None,
) -> dict[str, Any]:
    detailed_balance = float(detailed_balance_max_error)
    stationarity = float(stationarity_tv_error)
    passed = (
        bool(hastings_asymmetry_included)
        and detailed_balance <= 0.0
        and stationarity <= 0.0
        and bool(irreducible)
        and bool(aperiodic)
    )
    return {
        "SAMPLER_CORRECTNESS_RECEIPT": bool(passed),
        "sampler_type": str(sampler_type),
        "target_weight_hash": str(target_weight_hash),
        "proposal_kernel_hash": str(proposal_kernel_hash),
        "acceptance_rule_hash": str(acceptance_rule_hash),
        "hastings_asymmetry_included": bool(hastings_asymmetry_included),
        "detailed_balance_max_error": detailed_balance,
        "stationarity_tv_error": stationarity,
        "irreducible": bool(irreducible),
        "aperiodic": bool(aperiodic),
        "spectral_gap_estimate": None if spectral_gap_estimate is None else float(spectral_gap_estimate),
        "autocorrelation_time": None if autocorrelation_time is None else float(autocorrelation_time),
        "effective_sample_size": None if effective_sample_size is None else int(effective_sample_size),
    }


def rg_exponential_family_closure_receipt(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the RG MaxEnt receipt from primitive one-edge evidence.

    A producer's top-level boolean is intentionally ignored. Only a simulator-native
    finite-matrix I-projection report with a valid evidence hash, cutoff-independent
    constraint count, unique moment match, vanishing closure defect, and Pinsker check
    can pass this gate.
    """

    blockers: list[str] = []
    if not isinstance(evidence, Mapping):
        return {
            "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT": False,
            "promotion_blockers": ["finite_matrix_refinement_evidence_missing"],
            "claim_boundary": (
                "No MaxEnt closure promotion without simulator-native fine/coarse matrices."
            ),
        }

    if evidence.get("schema_version") != 1:
        blockers.append("unsupported_refinement_evidence_schema")
    if evidence.get("evidence_kind") != "finite_matrix_refinement_closure":
        blockers.append("wrong_refinement_evidence_kind")
    if evidence.get("algorithm") != "oph-maxent-i-projection-v1":
        blockers.append("wrong_refinement_algorithm")
    if not evidence.get("coarse_graining_applied", False):
        blockers.append("coarse_graining_not_applied")
    evidence_hash = str(evidence.get("evidence_input_sha256", ""))
    if len(evidence_hash) != 64 or any(char not in "0123456789abcdef" for char in evidence_hash):
        blockers.append("invalid_refinement_evidence_hash")

    tolerances = evidence.get("tolerances")
    if not isinstance(tolerances, Mapping):
        blockers.append("refinement_tolerances_missing")
        tolerances = {}

    def finite_number(key: str, source: Mapping[str, Any] = evidence) -> float | None:
        try:
            value = float(source[key])
        except (KeyError, TypeError, ValueError):
            blockers.append(f"{key}_missing_or_non_numeric")
            return None
        if not math.isfinite(value):
            blockers.append(f"{key}_non_finite")
            return None
        return value

    moment = finite_number("moment_matching_residual_linf")
    hessian = finite_number("duhamel_hessian_min_eigenvalue")
    defect = finite_number("closure_defect_nats")
    trace_residual = finite_number("trace_norm_residual")
    pinsker = finite_number("pinsker_residual_bound")
    moment_tol = finite_number("moment", tolerances)
    defect_tol = finite_number("closure_defect_nats", tolerances)
    hessian_tol = finite_number("hessian_floor", tolerances)
    numerical_tol = finite_number("numerical", tolerances)

    if not evidence.get("input_state_validated", False):
        blockers.append("fine_state_not_validated")
    if not evidence.get("coarse_state_validated", False):
        blockers.append("coarse_state_not_validated")
    if not str(evidence.get("refinement_channel_id", "")).strip():
        blockers.append("refinement_channel_id_missing")

    constraint_counts: list[int] = []
    for key in (
        "fine_constraint_count",
        "coarse_constraint_count",
        "fine_independent_constraint_count",
        "coarse_independent_constraint_count",
    ):
        try:
            value = int(evidence[key])
        except (KeyError, TypeError, ValueError):
            blockers.append(f"{key}_missing_or_non_integer")
            continue
        constraint_counts.append(value)
    if (
        len(constraint_counts) != 4
        or constraint_counts[0] <= 0
        or len(set(constraint_counts)) != 1
    ):
        blockers.append("constraint_dimension_not_cutoff_independent")
    if not evidence.get("projection_converged", False):
        blockers.append("i_projection_not_converged")

    numeric_values = (
        moment,
        hessian,
        defect,
        trace_residual,
        pinsker,
        moment_tol,
        defect_tol,
        hessian_tol,
        numerical_tol,
    )
    if all(value is not None for value in numeric_values):
        assert moment is not None
        assert hessian is not None
        assert defect is not None
        assert trace_residual is not None
        assert pinsker is not None
        assert moment_tol is not None
        assert defect_tol is not None
        assert hessian_tol is not None
        assert numerical_tol is not None
        if min(moment_tol, defect_tol, hessian_tol, numerical_tol) < 0.0:
            blockers.append("negative_refinement_tolerance")
        if moment > moment_tol:
            blockers.append("moment_matching_residual_above_tolerance")
        if hessian <= hessian_tol:
            blockers.append("i_projection_not_unique")
        if defect < -numerical_tol or defect > defect_tol:
            blockers.append("exponential_family_closure_defect_nonzero")
        expected_pinsker = math.sqrt(2.0 * max(defect, 0.0))
        if abs(pinsker - expected_pinsker) > numerical_tol:
            blockers.append("pinsker_bound_not_recomputed_from_defect")
        if trace_residual > pinsker + numerical_tol:
            blockers.append("pinsker_residual_bound_violated")

    passed = not blockers
    return {
        "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT": bool(passed),
        "promotion_blockers": blockers,
        "evidence_input_sha256": evidence_hash or None,
        "refinement_channel_id": evidence.get("refinement_channel_id"),
        "closure_defect_nats": defect,
        "trace_norm_residual": trace_residual,
        "pinsker_residual_bound": pinsker,
        "claim_boundary": (
            "Finite-dimensional one-edge closure receipt only; regulator-uniform and "
            "continuum closure require a separately audited refinement family."
        ),
    }


def fail_closed_promotion_receipts(
    *,
    claim_tier: str,
    baseline_kind: str,
    rg_closure_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tier = checked_claim_tier(claim_tier)
    rg_gate = rg_exponential_family_closure_receipt(rg_closure_evidence)
    receipts = {
        "TRACIALLY_POINTED_QUOTIENT_RECEIPT": False,
        "REFERENCE_STATE_REFINEMENT_RECEIPT": False,
        "SOURCE_ONLY_CONSTRAINT_LEDGER_RECEIPT": False,
        "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT": bool(
            rg_gate["RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT"]
        ),
        "FIBER_ORBIT_HOMOGENEITY_RECEIPT": False,
        "LOCAL_REPAIR_GAP_FIELD_RECEIPT": False,
        "GLOBAL_POINCARE_CONSTANT_RECEIPT": False,
        "SOURCE_EUCLIDEAN_SLAB_RECEIPT": False,
        "TRANSFER_REFINEMENT_RECEIPT": False,
        "TRANSFER_VACUUM_RECEIPT": False,
        "STATIONARY_LAW_SCHEDULE_INVARIANCE_RECEIPT": False,
        "DETAILED_BALANCE_OF_AGGREGATE_KERNEL_RECEIPT": False,
        "PATHWISE_PARTITION_INVARIANCE_RECEIPT": False,
        "PRIMORDIAL_BRIDGE_RECEIPT": False,
        "FROZEN_LIKELIHOOD_RECEIPT": False,
    }
    return {
        "baseline_kind": str(baseline_kind),
        "claim_tier": tier,
        "receipts": receipts,
        "rg_exponential_family_closure_evidence": rg_gate,
        "claim_tier_gate": claim_tier_gate(claim_tier=tier, receipts=receipts),
        "OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT": False,
        "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": False,
        "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": False,
        "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
    }


def stable_hash(payload: Any) -> str:
    return stable_json_hash(payload).removeprefix("sha256:")


def _requirements_through(claim_tier: str) -> list[str]:
    tier_order = ("E2", "E3", "E4", "E5")
    if claim_tier not in tier_order:
        return []
    out: list[str] = []
    for tier in tier_order:
        out.extend(RECEIPT_REQUIREMENTS[tier])
        if tier == claim_tier:
            break
    return out


def _canonical_value(value: Any, dropped_keys: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(subvalue, dropped_keys)
            for key, subvalue in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in dropped_keys
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, dropped_keys) for item in value]
    return value
