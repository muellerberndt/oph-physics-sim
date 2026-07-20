from __future__ import annotations

from copy import deepcopy

from oph_fpe.bulk.physical_h3_kms_prerun import (
    ALLOWED_CHECKER_IDS,
    ALLOWED_PRODUCER_IDS,
    CELL_CONFIG_SCHEMA,
    PLAN_SCHEMA,
    REGISTERED_HISTORICAL_16K_SOURCE_SEED,
    REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256,
    SCHEMA_VERSION,
    SOURCE_INPUT_FIELDS,
    canonical_sha256,
    physical_h3_kms_prerun_report,
    physical_h3_kms_source_inputs,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import _normalize_config


def _hash(label: str) -> str:
    return canonical_sha256({"label": label})


def _bundle() -> dict:
    rungs = [4_096, 16_384, 65_536, 262_144]
    seeds = [101, 202, 303]
    support_counts = {str(rung): int(1.25 * rung) for rung in rungs}
    observer_counts = dict(zip(map(str, rungs), [32, 64, 128, 256], strict=True))
    producer_registry = {
        role: {
            "producer_id": next(iter(values)),
            "source_code_sha256": _hash(f"producer:{role}"),
        }
        for role, values in ALLOWED_PRODUCER_IDS.items()
    }
    checker_registry = {
        role: {
            "checker_id": next(iter(values)),
            "checker_code_sha256": _hash(f"checker:{role}"),
        }
        for role, values in ALLOWED_CHECKER_IDS.items()
    }
    scaling = {
        "carrier_count_law": "exact_rung_cardinality_v1",
        "support_regulator_law": "first_icosahedral_cell_count_at_or_above_rung_v1",
        "support_counts_by_rung": support_counts,
        "observer_scaling": {
            "law_id": "power_law_ceil_v1",
            "coefficient": 0.5,
            "exponent": 0.5,
            "minimum": 32,
            "maximum": 4096,
            "counts_by_rung": observer_counts,
        },
        "cycles": 160,
        "repair_fraction_per_cycle": 0.0625,
        "record_commit_cycles": 12,
    }

    def cell_config(seed: int, rung: int) -> dict:
        level = {4_096: 4, 16_384: 5, 65_536: 6, 262_144: 7}[rung]
        return {
            "schema": CELL_CONFIG_SCHEMA,
            "cell": {"seed": seed, "rung": rung, "replicate_id": "primary"},
            "source_federation": {
                "family": "federated_echosahedral_carriers",
                "carrier_count": rung,
                "ports_per_carrier": 12,
                "local_template": "regular_icosahedron_12_30_20_antipode_a5_v1",
                **producer_registry["source_federation"],
            },
            "support_regulator": {
                "family": "nested_geodesic_icosahedral",
                "patch_basis": "cells",
                "refinement_level": level,
                "patch_count": support_counts[str(rung)],
                "drives_source_seams": False,
                "drives_source_repairs": False,
                **producer_registry["support_regulator"],
            },
            "source_generator": {
                **producer_registry["source_dynamics"],
                "state_space": "normalized_complex_amplitude_in_C12",
                "rng_family": "numpy_generator_pcg64_v1",
                "initialization_distribution": "normalized_complex_gaussian_v1",
                "intrinsic_phase_distribution": "uniform_unit_interval_v1",
                "propagation_steps": 4,
                "intrinsic_step": 0.137,
                "coupling_strength": 1.0,
                "geometry_sample_count": 4,
            },
            "repair_dynamics": {
                **producer_registry["source_dynamics"],
                "cycles": scaling["cycles"],
                "repair_fraction_per_cycle": scaling["repair_fraction_per_cycle"],
                "record_commit_cycles": scaling["record_commit_cycles"],
                "seam_update_rule": (
                    "disjoint_single_port_endpoint_arithmetic_mean_v1"
                ),
            },
            "observer_capture": {
                **producer_registry["observer_capture"],
                "observer_count": observer_counts[str(rung)],
                "support_size": 2,
                "samples_per_observer": 6,
                "prediction_control": "semantic_hash_shuffle_v1",
                "feedback_enabled": True,
                "checkpoint_interval": 8,
            },
        }

    current = {"seed": seeds[0], "rung": 16_384, "replicate_id": "primary"}
    rows = []
    requested_config = None
    for seed in seeds:
        for rung in rungs:
            config = cell_config(seed, rung)
            rows.append(
                {
                    "cell": config["cell"],
                    "cell_config": config,
                    "config_sha256": canonical_sha256(config),
                    "status": "NOT_EVALUATED",
                }
            )
            if config["cell"] == current:
                requested_config = config
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": "physical-h3-kms-frozen-family-002",
        "instrument_version": "physical-h3-kms-v2",
        "instrument_commit_sha256": _hash("instrument"),
        "seeds": seeds,
        "rungs": rungs,
        "replicate_ids": ["primary"],
        "clock_candidates": ["1x", "pi", "2pi", "4pi"],
        "geometry_models": ["H3", "S2", "E3", "E4"],
        "thresholds": {
            "clock_absolute_residual_max": 0.2,
            "clock_win_margin_min": 0.1,
            "geometry_win_margin_min": 0.05,
            "curvature_minimum_power": 0.9,
        },
        "calibrations": {
            "clock_calibration_sha256": _hash("clock-calibration"),
            "geometry_calibration_sha256": _hash("geometry-calibration"),
            "curvature_calibration_sha256": _hash("curvature-calibration"),
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "physical_threshold_calibration_receipt": False,
        },
        "split_contract": {
            "algorithm_id": "semantic_hash_split_v1",
            "assignment_salt_sha256": _hash("semantic-split-salt"),
            "holdout_fraction": 0.25,
            "derivation": "semantic_event_id_hash_threshold_v1",
            "heldout_ids_materialized_before_capture": False,
        },
        "scaling_contract": scaling,
        "archive_boundary": {
            "frozen_before_source_capture": True,
            "retune_after_freeze": False,
            "archived_16k_failure_preserved": True,
            "archived_outcomes_used_for_threshold_selection": False,
            "historical_receipt_byte_sha256": (
                REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256
            ),
            "historical_campaign_sha256": REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
            "historical_16k_source_seed": REGISTERED_HISTORICAL_16K_SOURCE_SEED,
            "historical_16k_rung": 16_384,
            "historical_16k_joint_independent_receipt": False,
            "historical_stable_branch_failure_established": False,
        },
        "producer_registry": producer_registry,
        "checker_registry": checker_registry,
        "run_matrix": rows,
        "current_cell": current,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    assert requested_config is not None
    return {
        "schema": SCHEMA_VERSION,
        "config": deepcopy(requested_config),
        "plan": plan,
    }


def _recommit(bundle: dict) -> None:
    plan = bundle["plan"]
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def test_strict_plan_allows_capture_but_science_is_not_evaluated() -> None:
    bundle = _bundle()
    report = physical_h3_kms_prerun_report(bundle)
    assert report["admission_status"] == "VALID_PASS"
    assert report["SOURCE_CAPTURE_ALLOWED"] is True
    assert report["scientific_status"] == "NOT_EVALUATED"
    assert report["scientific_failures"] == []
    assert report["stages"]["TARGET_FIREWALL"]["evidence"][
        "scanned_entire_runtime_config"
    ] is True
    expected_source_inputs = {
        "carrier_count": 16_384,
        "seed": 101,
        "rung": 16_384,
        "replicate_id": "primary",
        "preregistered_plan_sha256": bundle["plan"]["plan_sha256"],
        "propagation_steps": 4,
        "intrinsic_step": 0.137,
        "coupling_strength": 1.0,
        "state_space": "normalized_complex_amplitude_in_C12",
        "rng_family": "numpy_generator_pcg64_v1",
        "initialization_distribution": "normalized_complex_gaussian_v1",
        "intrinsic_phase_distribution": "uniform_unit_interval_v1",
        "seam_update_rule": (
            "disjoint_single_port_endpoint_arithmetic_mean_v1"
        ),
        "cycles": 160,
        "repair_fraction_per_cycle": 0.0625,
        "record_commit_cycles": 12,
        "observer_count": 64,
        "observer_support_size": 2,
        "observer_samples": 6,
        "prediction_control": "semantic_hash_shuffle_v1",
        "feedback_enabled": True,
        "checkpoint_interval": 8,
        "support_refinement_level": 5,
        "geometry_sample_count": 4,
    }
    assert set(report["source_inputs"]) == SOURCE_INPUT_FIELDS
    assert report["source_inputs"] == expected_source_inputs
    assert report["source_inputs_sha256"] == canonical_sha256(expected_source_inputs)
    assert physical_h3_kms_source_inputs(bundle) == expected_source_inputs
    assert _normalize_config(report["source_inputs"]) == expected_source_inputs


def test_any_plan_mutation_breaks_full_plan_commitment() -> None:
    bundle = _bundle()
    bundle["plan"]["thresholds"]["clock_win_margin_min"] = 0.11
    report = physical_h3_kms_prerun_report(bundle)
    assert report["admission_status"] == "INSTRUMENT_INVALID"
    assert "plan:plan_sha256_mismatch" in report["invalidators"]


def test_unknown_runtime_field_and_nested_target_are_both_rejected() -> None:
    bundle = _bundle()
    bundle["config"]["source_generator"]["hidden"] = {
        "kms_normalization": 2 * 3.141592653589793
    }
    report = physical_h3_kms_prerun_report(bundle)
    assert report["admission_status"] == "INSTRUMENT_INVALID"
    assert any("unknown_fields:hidden" in value for value in report["invalidators"])
    assert report["stages"]["TARGET_FIREWALL"]["invalidators"]


def test_source_runtime_contract_rejects_legacy_or_missing_declarations() -> None:
    legacy = _bundle()
    generator = legacy["config"]["source_generator"]
    del generator["state_space"]
    generator["state_alphabet_size"] = 6
    generator["initialization_distribution"] = "uniform_local_port_state_v1"
    report = physical_h3_kms_prerun_report(legacy)
    assert report["admission_status"] == "INSTRUMENT_INVALID"
    assert report["SOURCE_CAPTURE_ALLOWED"] is False
    assert report["source_inputs"] == {}
    assert report["source_inputs_sha256"] is None
    assert any("missing_fields:state_space" in item for item in report["blockers"])
    assert any("unknown_fields:state_alphabet_size" in item for item in report["invalidators"])
    assert any("initialization_distribution_mismatch" in item for item in report["invalidators"])

    missing = _bundle()
    del missing["config"]["observer_capture"]["samples_per_observer"]
    report = physical_h3_kms_prerun_report(missing)
    assert report["SOURCE_CAPTURE_ALLOWED"] is False
    assert any("missing_fields:samples_per_observer" in item for item in report["blockers"])


def test_cell_config_hash_counts_and_requested_cell_are_recomputed() -> None:
    bundle = _bundle()
    row = next(
        row
        for row in bundle["plan"]["run_matrix"]
        if row["cell"] == bundle["plan"]["current_cell"]
    )
    row["cell_config"]["observer_capture"]["observer_count"] += 1
    _recommit(bundle)
    report = physical_h3_kms_prerun_report(bundle)
    assert report["admission_status"] == "INSTRUMENT_INVALID"
    assert any("observer_count_mismatches_scaling" in value for value in report["invalidators"])
    assert any("config_sha256_mismatch" in value for value in report["invalidators"])
    assert "plan:requested_config_not_exact_current_cell_config" in report["invalidators"]


def test_missing_seed_is_blocked_and_code_hash_tamper_is_invalid() -> None:
    blocked = _bundle()
    blocked["plan"]["seeds"] = blocked["plan"]["seeds"][:2]
    blocked["plan"]["run_matrix"] = [
        row
        for row in blocked["plan"]["run_matrix"]
        if row["cell"]["seed"] in blocked["plan"]["seeds"]
    ]
    _recommit(blocked)
    report = physical_h3_kms_prerun_report(blocked)
    assert report["admission_status"] == "BLOCKED"
    assert report["scientific_status"] == "NOT_EVALUATED"

    invalid = _bundle()
    invalid["plan"]["checker_registry"]["cell_postflight"][
        "checker_code_sha256"
    ] = "not-a-hash"
    _recommit(invalid)
    report = physical_h3_kms_prerun_report(invalid)
    assert report["admission_status"] == "INSTRUMENT_INVALID"
    assert any("checker_code_sha256_malformed" in value for value in report["invalidators"])


def test_curvature_power_must_be_strictly_positive_and_at_most_one() -> None:
    for invalid_power in (0.0, -0.1, 1.01):
        bundle = _bundle()
        bundle["plan"]["thresholds"]["curvature_minimum_power"] = invalid_power
        _recommit(bundle)
        report = physical_h3_kms_prerun_report(bundle)
        assert report["admission_status"] == "INSTRUMENT_INVALID"
        assert any(
            "curvature_minimum_power_must_lie_in_open_closed_unit_interval" in item
            for item in report["invalidators"]
        )


def test_historical_archive_commitment_cannot_be_replaced_by_well_formed_hashes() -> None:
    bundle = _bundle()
    archive = bundle["plan"]["archive_boundary"]
    archive["historical_receipt_byte_sha256"] = "sha256:" + "0" * 64
    archive["historical_campaign_sha256"] = "f" * 64
    archive["historical_16k_source_seed"] = 123
    _recommit(bundle)

    report = physical_h3_kms_prerun_report(bundle)

    assert report["admission_status"] == "INSTRUMENT_INVALID"
    assert "plan:archive_boundary:historical_receipt_hash_mismatch" in report[
        "invalidators"
    ]
    assert "plan:archive_boundary:historical_campaign_hash_mismatch" in report[
        "invalidators"
    ]
    assert "plan:archive_boundary:historical_16k_source_seed_invalid" in report[
        "invalidators"
    ]
