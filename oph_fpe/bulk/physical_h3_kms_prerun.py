"""Outcome-blind static admission for physical H3/KMS source capture.

This module validates one exact, content-committed campaign plan and the
requested runtime cell configuration.  It deliberately does not ask for
repair outcomes, observers, cap states, clocks, geometry fits, or events that
can exist only after source capture.  A pass authorizes capture only; science
remains ``NOT_EVALUATED`` until registered postflight replay.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "oph.physical-h3-kms.prerun-admission.v1"
CELL_CONFIG_SCHEMA = "oph.source-capture.cell-config.v1"
PLAN_SCHEMA = "oph.physical-h3-kms.frozen-plan.v1"
REQUIRED_RUNGS = (4_096, 16_384, 65_536, 262_144)
REQUIRED_CLOCK_CANDIDATES = ("1x", "pi", "2pi", "4pi")
REQUIRED_GEOMETRY_MODELS = ("H3", "S2", "E3", "E4")
MINIMUM_SOURCE_SEEDS = 3
REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256 = (
    "sha256:cb076b640941928ca603801e7a0123dd0c9745990c7ac309ddf34ddec0ffa1e6"
)
REGISTERED_HISTORICAL_CAMPAIGN_SHA256 = (
    "d1c713b77c4ce91bcd419f53655181ea9f07197ce10a7b46d8ce08a2a937a072"
)
REGISTERED_HISTORICAL_16K_SOURCE_SEED = 20_260_751
SOURCE_INPUT_FIELDS = frozenset(
    {
        "carrier_count",
        "seed",
        "rung",
        "replicate_id",
        "preregistered_plan_sha256",
        "propagation_steps",
        "intrinsic_step",
        "coupling_strength",
        "state_space",
        "rng_family",
        "initialization_distribution",
        "intrinsic_phase_distribution",
        "seam_update_rule",
        "cycles",
        "repair_fraction_per_cycle",
        "record_commit_cycles",
        "observer_count",
        "observer_support_size",
        "observer_samples",
        "prediction_control",
        "feedback_enabled",
        "checkpoint_interval",
        "support_refinement_level",
        "geometry_sample_count",
    }
)

ALLOWED_PRODUCER_IDS = {
    "source_federation": frozenset({"echosahedral_federation_source_v1"}),
    "support_regulator": frozenset({"geodesic_icosahedral_support_v1"}),
    "source_dynamics": frozenset({"proof_carrying_local_repair_source_v1"}),
    "observer_capture": frozenset({"operational_observer_capture_v3"}),
}
ALLOWED_CHECKER_IDS = {
    "source_federation": frozenset({"reference_federation_replayer_v1"}),
    "support_regulator": frozenset({"icosahedral_tower_replayer_v1"}),
    "source_firewall": frozenset({"source_target_firewall_v1"}),
    "repair_replay": frozenset({"transactional_repair_replayer_v1"}),
    "observer_replay": frozenset({"operational_observer_replayer_v3"}),
    "cell_postflight": frozenset({"physical_h3_kms_cell_postflight_v1"}),
}

_TOP_FIELDS = frozenset({"schema", "config", "plan"})
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "campaign_id",
        "instrument_version",
        "instrument_commit_sha256",
        "seeds",
        "rungs",
        "replicate_ids",
        "clock_candidates",
        "geometry_models",
        "thresholds",
        "calibrations",
        "split_contract",
        "scaling_contract",
        "archive_boundary",
        "producer_registry",
        "checker_registry",
        "run_matrix",
        "current_cell",
        "plan_sha256",
    }
)
_CELL_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "cell",
        "source_federation",
        "support_regulator",
        "source_generator",
        "repair_dynamics",
        "observer_capture",
    }
)
_CELL_ID_FIELDS = frozenset({"seed", "rung", "replicate_id"})
_SOURCE_FEDERATION_FIELDS = frozenset(
    {
        "family",
        "carrier_count",
        "ports_per_carrier",
        "local_template",
        "producer_id",
        "source_code_sha256",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "family",
        "patch_basis",
        "refinement_level",
        "patch_count",
        "drives_source_seams",
        "drives_source_repairs",
        "producer_id",
        "source_code_sha256",
    }
)
_SOURCE_GENERATOR_FIELDS = frozenset(
    {
        "producer_id",
        "source_code_sha256",
        "state_space",
        "rng_family",
        "initialization_distribution",
        "intrinsic_phase_distribution",
        "propagation_steps",
        "intrinsic_step",
        "coupling_strength",
        "geometry_sample_count",
    }
)
_REPAIR_FIELDS = frozenset(
    {
        "producer_id",
        "source_code_sha256",
        "cycles",
        "repair_fraction_per_cycle",
        "record_commit_cycles",
        "seam_update_rule",
    }
)
_OBSERVER_FIELDS = frozenset(
    {
        "producer_id",
        "source_code_sha256",
        "observer_count",
        "support_size",
        "samples_per_observer",
        "prediction_control",
        "feedback_enabled",
        "checkpoint_interval",
    }
)
_RUN_ROW_FIELDS = frozenset(
    {"cell", "cell_config", "config_sha256", "status"}
)
_THRESHOLD_FIELDS = frozenset(
    {
        "clock_absolute_residual_max",
        "clock_win_margin_min",
        "geometry_win_margin_min",
        "curvature_minimum_power",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {
        "clock_calibration_sha256",
        "geometry_calibration_sha256",
        "curvature_calibration_sha256",
        "independent_of_campaign_source_seeds",
        "frozen_before_source_capture",
        "physical_threshold_calibration_receipt",
    }
)
_SPLIT_FIELDS = frozenset(
    {
        "algorithm_id",
        "assignment_salt_sha256",
        "holdout_fraction",
        "derivation",
        "heldout_ids_materialized_before_capture",
    }
)
_SCALING_FIELDS = frozenset(
    {
        "carrier_count_law",
        "support_regulator_law",
        "support_counts_by_rung",
        "observer_scaling",
        "cycles",
        "repair_fraction_per_cycle",
        "record_commit_cycles",
    }
)
_OBSERVER_SCALING_FIELDS = frozenset(
    {"law_id", "coefficient", "exponent", "minimum", "maximum", "counts_by_rung"}
)
_ARCHIVE_FIELDS = frozenset(
    {
        "frozen_before_source_capture",
        "retune_after_freeze",
        "archived_16k_failure_preserved",
        "archived_outcomes_used_for_threshold_selection",
        "historical_receipt_byte_sha256",
        "historical_campaign_sha256",
        "historical_16k_source_seed",
        "historical_16k_rung",
        "historical_16k_joint_independent_receipt",
        "historical_stable_branch_failure_established",
    }
)
_REGISTRY_ENTRY_FIELDS = frozenset({"producer_id", "source_code_sha256"})
_CHECKER_ENTRY_FIELDS = frozenset({"checker_id", "checker_code_sha256"})

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPLICATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TARGET_TOKENS = frozenset(
    {
        "candidate",
        "desired",
        "expected",
        "h3",
        "kms",
        "normalization",
        "preferred",
        "target",
        "2pi",
        "4pi",
    }
)


class PrerunStatus(str, Enum):
    VALID_PASS = "VALID_PASS"
    BLOCKED = "BLOCKED"
    INSTRUMENT_INVALID = "INSTRUMENT_INVALID"
    NOT_EVALUATED = "NOT_EVALUATED"


def canonical_sha256(value: Any) -> str:
    """Return the canonical, finite-JSON SHA-256 commitment."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def frozen_campaign_family_sha256(plan: Mapping[str, Any]) -> str:
    """Commit the immutable campaign family independently of cell selection.

    ``current_cell`` chooses which already-frozen row an execution will run;
    it is not a scientific family parameter.  The full per-cell plan hash still
    binds that selector, while this digest must remain identical across all
    seed/rung executions that are later aggregated into one campaign.
    """

    material = {
        key: value
        for key, value in plan.items()
        if key not in {"current_cell", "plan_sha256"}
    }
    return canonical_sha256(material)


def physical_h3_kms_prerun_report(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate an exact frozen plan and requested cell before RNG starts."""

    payload, source_info, load_blockers, load_invalidators = _load_source(source)
    top_blockers = list(load_blockers)
    top_invalidators = list(load_invalidators)
    _exact_fields(payload, _TOP_FIELDS, "bundle", top_blockers, top_invalidators)
    if "schema" in payload and payload.get("schema") != SCHEMA_VERSION:
        top_invalidators.append("bundle:schema_mismatch")

    config = _mapping(payload.get("config"))
    plan = _mapping(payload.get("plan"))
    registry_stage = _validate_registries(plan)
    plan_stage = _validate_plan(plan, config)
    config_stage = _validate_cell_config(
        config,
        producer_registry=_mapping(plan.get("producer_registry")),
        scaling=_mapping(plan.get("scaling_contract")),
        expected_cell=_mapping(plan.get("current_cell")),
        context="config",
    )
    firewall_stage = _validate_target_firewall(config)
    stages = {
        "BUNDLE_SCHEMA": _stage(top_blockers, top_invalidators),
        "STATIC_SOURCE_AND_SUPPORT": config_stage,
        "TARGET_FIREWALL": firewall_stage,
        "FROZEN_CAMPAIGN_PLAN": plan_stage,
        "REGISTERED_PRODUCERS_AND_CHECKERS": registry_stage,
    }
    blockers = list(
        dict.fromkeys(item for stage in stages.values() for item in stage["blockers"])
    )
    invalidators = list(
        dict.fromkeys(item for stage in stages.values() for item in stage["invalidators"])
    )
    if invalidators:
        admission_status = PrerunStatus.INSTRUMENT_INVALID.value
    elif blockers:
        admission_status = PrerunStatus.BLOCKED.value
    else:
        admission_status = PrerunStatus.VALID_PASS.value
    capture_allowed = admission_status == PrerunStatus.VALID_PASS.value
    source_inputs = _source_input_projection(config, plan) if capture_allowed else {}
    return {
        "schema": SCHEMA_VERSION,
        "mode": "outcome_blind_static_source_capture_admission",
        "admission_status": admission_status,
        "scientific_status": PrerunStatus.NOT_EVALUATED.value,
        "SOURCE_CAPTURE_ALLOWED": capture_allowed,
        "source_capture_allowed": capture_allowed,
        "scientific_evaluation_performed": False,
        "scientific_failures": [],
        "blockers": blockers,
        "invalidators": invalidators,
        "stages": stages,
        "source_inputs": source_inputs,
        "source_inputs_sha256": (
            canonical_sha256(source_inputs) if capture_allowed else None
        ),
        "frozen_campaign_family_sha256": (
            frozen_campaign_family_sha256(plan) if capture_allowed else None
        ),
        "input": source_info,
        "status_vocabulary": [status.value for status in PrerunStatus],
        "split_policy": (
            "Held-out semantic event IDs are not listed in the plan. They are derived "
            "after capture from the committed semantic salt, algorithm, and fraction."
        ),
        "claim_boundary": (
            "VALID_PASS authorizes only target-blind source capture under the exact "
            "committed plan. Realized P0-P7 evidence, H3/KMS selection, physical "
            "promotion, and branch retirement remain NOT_EVALUATED."
        ),
    }


def physical_h3_kms_source_inputs(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return the exact admitted runtime projection for source capture.

    The projection deliberately renames observer-plan fields to the runtime
    source-capture names.  It is available only after static admission, so a
    caller cannot accidentally launch from a malformed or uncommitted plan.
    """

    report = physical_h3_kms_prerun_report(source)
    if report["admission_status"] != PrerunStatus.VALID_PASS.value:
        raise ValueError("preregistration does not authorize source capture")
    result = _mapping(report.get("source_inputs"))
    if set(result) != SOURCE_INPUT_FIELDS:
        raise RuntimeError("internal source-input projection field mismatch")
    return result


def _validate_plan(
    plan: Mapping[str, Any], requested_config: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    invalidators: list[str] = []
    _exact_fields(plan, _PLAN_FIELDS, "plan", blockers, invalidators)
    if "schema" in plan and plan.get("schema") != PLAN_SCHEMA:
        invalidators.append("plan:schema_mismatch")
    if not _nonempty_string(plan.get("campaign_id")):
        blockers.append("plan:campaign_id_missing")
    if plan.get("instrument_version") != "physical-h3-kms-v2":
        invalidators.append("plan:instrument_version_mismatch")
    if not _strict_sha256(plan.get("instrument_commit_sha256")):
        blockers.append("plan:instrument_commit_sha256_missing_or_malformed")

    seeds = _strict_int_sequence(plan.get("seeds"))
    rungs = _strict_int_sequence(plan.get("rungs"))
    replicates = _string_sequence(plan.get("replicate_ids"))
    if len(set(seeds)) < MINIMUM_SOURCE_SEEDS:
        blockers.append("plan:at_least_three_unique_source_seeds_required")
    if len(seeds) != len(set(seeds)):
        invalidators.append("plan:source_seeds_must_be_unique")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        invalidators.append("plan:source_seed_outside_runtime_range")
    if tuple(rungs) != REQUIRED_RUNGS:
        invalidators.append("plan:rungs_must_be_exactly_4k_16k_64k_256k")
    if not replicates or len(replicates) != len(set(replicates)):
        blockers.append("plan:nonempty_unique_replicate_ids_required")
    if any(_REPLICATE_ID_RE.fullmatch(value) is None for value in replicates):
        invalidators.append("plan:replicate_id_outside_runtime_contract")
    if tuple(_string_sequence(plan.get("clock_candidates"))) != REQUIRED_CLOCK_CANDIDATES:
        invalidators.append("plan:clock_candidates_must_be_exactly_1x_pi_2pi_4pi")
    if tuple(_string_sequence(plan.get("geometry_models"))) != REQUIRED_GEOMETRY_MODELS:
        invalidators.append("plan:geometry_models_must_be_exactly_h3_s2_e3_e4")

    _validate_thresholds(_mapping(plan.get("thresholds")), blockers, invalidators)
    _validate_calibrations(_mapping(plan.get("calibrations")), blockers, invalidators)
    _validate_split(_mapping(plan.get("split_contract")), blockers, invalidators)
    scaling = _mapping(plan.get("scaling_contract"))
    observer_counts, support_counts = _validate_scaling(
        scaling, rungs, blockers, invalidators
    )
    _validate_archive(_mapping(plan.get("archive_boundary")), blockers, invalidators)

    expected_cells = set(itertools.product(seeds, rungs, replicates))
    observed_cells: set[tuple[int, int, str]] = set()
    current = _mapping(plan.get("current_cell"))
    _exact_fields(current, _CELL_ID_FIELDS, "plan:current_cell", blockers, invalidators)
    current_key = _cell_key(current)
    if current_key is None:
        blockers.append("plan:current_cell_identity_invalid")
    elif current_key not in expected_cells:
        invalidators.append("plan:current_cell_not_in_frozen_matrix")

    producer_registry = _mapping(plan.get("producer_registry"))
    current_row_config: dict[str, Any] | None = None
    rows = _sequence(plan.get("run_matrix"))
    for index, raw in enumerate(rows):
        context = f"plan:run_matrix:{index}"
        if not isinstance(raw, Mapping):
            invalidators.append(f"{context}:row_must_be_object")
            continue
        row = dict(raw)
        _exact_fields(row, _RUN_ROW_FIELDS, context, blockers, invalidators)
        cell = _mapping(row.get("cell"))
        _exact_fields(cell, _CELL_ID_FIELDS, f"{context}:cell", blockers, invalidators)
        key = _cell_key(cell)
        if key is None:
            invalidators.append(f"{context}:cell_identity_invalid")
            continue
        if key in observed_cells:
            invalidators.append(f"{context}:duplicate_cell")
        observed_cells.add(key)
        if row.get("status") != PrerunStatus.NOT_EVALUATED.value:
            invalidators.append(f"{context}:status_must_be_not_evaluated")
        cell_config = _mapping(row.get("cell_config"))
        nested = _validate_cell_config(
            cell_config,
            producer_registry=producer_registry,
            scaling=scaling,
            expected_cell=cell,
            context=f"{context}:cell_config",
            observer_counts=observer_counts,
            support_counts=support_counts,
        )
        blockers.extend(nested["blockers"])
        invalidators.extend(nested["invalidators"])
        declared_config_hash = row.get("config_sha256")
        if not _strict_sha256(declared_config_hash):
            blockers.append(f"{context}:config_sha256_missing_or_malformed")
        else:
            try:
                computed_config_hash = canonical_sha256(cell_config)
            except (TypeError, ValueError):
                invalidators.append(f"{context}:cell_config_not_finite_canonical_json")
            else:
                if declared_config_hash != computed_config_hash:
                    invalidators.append(f"{context}:config_sha256_mismatch")
        if key == current_key:
            current_row_config = cell_config
    if observed_cells != expected_cells:
        blockers.append("plan:run_matrix_must_cover_every_seed_rung_replicate_cell")
    if current_row_config is None:
        blockers.append("plan:current_cell_config_missing_from_run_matrix")
    elif dict(requested_config) != current_row_config:
        invalidators.append("plan:requested_config_not_exact_current_cell_config")

    declared_plan_hash = plan.get("plan_sha256")
    if not _strict_sha256(declared_plan_hash):
        blockers.append("plan:plan_sha256_missing_or_malformed")
    else:
        material = {key: value for key, value in plan.items() if key != "plan_sha256"}
        try:
            computed_plan_hash = canonical_sha256(material)
        except (TypeError, ValueError):
            invalidators.append("plan:not_finite_canonical_json")
        else:
            if declared_plan_hash != computed_plan_hash:
                invalidators.append("plan:plan_sha256_mismatch")
    return _stage(blockers, invalidators)


def _validate_cell_config(
    config: Mapping[str, Any],
    *,
    producer_registry: Mapping[str, Any],
    scaling: Mapping[str, Any],
    expected_cell: Mapping[str, Any],
    context: str,
    observer_counts: Mapping[int, int] | None = None,
    support_counts: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    invalidators: list[str] = []
    _exact_fields(config, _CELL_CONFIG_FIELDS, context, blockers, invalidators)
    if "schema" in config and config.get("schema") != CELL_CONFIG_SCHEMA:
        invalidators.append(f"{context}:schema_mismatch")
    cell = _mapping(config.get("cell"))
    _exact_fields(cell, _CELL_ID_FIELDS, f"{context}:cell", blockers, invalidators)
    key = _cell_key(cell)
    if expected_cell and cell != dict(expected_cell):
        invalidators.append(f"{context}:cell_identity_mismatch")
    rung = key[1] if key is not None else None

    source = _mapping(config.get("source_federation"))
    _exact_fields(
        source, _SOURCE_FEDERATION_FIELDS, f"{context}:source_federation", blockers, invalidators
    )
    if source.get("family") != "federated_echosahedral_carriers":
        blockers.append(f"{context}:source_federation_family_missing")
    if rung is not None and _strict_int(source.get("carrier_count")) != rung:
        invalidators.append(f"{context}:carrier_count_mismatches_rung")
    if _strict_int(source.get("ports_per_carrier")) != 12:
        invalidators.append(f"{context}:ports_per_carrier_must_be_12")
    if source.get("local_template") != "regular_icosahedron_12_30_20_antipode_a5_v1":
        invalidators.append(f"{context}:local_template_mismatch")

    support = _mapping(config.get("support_regulator"))
    _exact_fields(support, _SUPPORT_FIELDS, f"{context}:support_regulator", blockers, invalidators)
    if support.get("family") != "nested_geodesic_icosahedral":
        blockers.append(f"{context}:support_regulator_family_missing")
    if support.get("patch_basis") != "cells":
        invalidators.append(f"{context}:support_patch_basis_must_be_cells")
    level = _strict_int(support.get("refinement_level"))
    support_count = _strict_int(support.get("patch_count"))
    if level is None or level < 0 or support_count is None:
        blockers.append(f"{context}:support_level_or_count_missing")
    elif support_count != 20 * (4**level):
        invalidators.append(f"{context}:support_count_not_exact_icosahedral_count")
    if support.get("drives_source_seams") is not False:
        invalidators.append(f"{context}:support_must_not_drive_source_seams")
    if support.get("drives_source_repairs") is not False:
        invalidators.append(f"{context}:support_must_not_drive_source_repairs")

    generator = _mapping(config.get("source_generator"))
    _exact_fields(generator, _SOURCE_GENERATOR_FIELDS, f"{context}:source_generator", blockers, invalidators)
    if generator.get("state_space") != "normalized_complex_amplitude_in_C12":
        invalidators.append(f"{context}:state_space_must_be_normalized_C12")
    if generator.get("rng_family") != "numpy_generator_pcg64_v1":
        invalidators.append(f"{context}:rng_family_not_allowlisted")
    if (
        generator.get("initialization_distribution")
        != "normalized_complex_gaussian_v1"
    ):
        invalidators.append(f"{context}:initialization_distribution_mismatch")
    if (
        generator.get("intrinsic_phase_distribution")
        != "uniform_unit_interval_v1"
    ):
        invalidators.append(f"{context}:intrinsic_phase_distribution_mismatch")
    propagation_steps = _strict_int(generator.get("propagation_steps"))
    if propagation_steps is None or not 1 <= propagation_steps <= 64:
        blockers.append(f"{context}:propagation_steps_invalid")
    intrinsic_step = _finite_number(generator.get("intrinsic_step"))
    if intrinsic_step is None or intrinsic_step <= 0.0:
        blockers.append(f"{context}:intrinsic_step_must_be_finite_and_positive")
    coupling_strength = _finite_number(generator.get("coupling_strength"))
    if coupling_strength is None or coupling_strength <= 0.0:
        blockers.append(f"{context}:coupling_strength_must_be_finite_and_positive")
    geometry_sample_count = _strict_int(generator.get("geometry_sample_count"))
    if geometry_sample_count is None or not 3 <= geometry_sample_count <= 32:
        blockers.append(f"{context}:geometry_sample_count_invalid")

    repair = _mapping(config.get("repair_dynamics"))
    _exact_fields(repair, _REPAIR_FIELDS, f"{context}:repair_dynamics", blockers, invalidators)
    if (
        repair.get("seam_update_rule")
        != "disjoint_single_port_endpoint_arithmetic_mean_v1"
    ):
        invalidators.append(f"{context}:seam_update_rule_mismatch")
    cycles = _strict_int(repair.get("cycles"))
    if cycles is None or not 1 <= cycles <= 4096:
        blockers.append(f"{context}:cycles_invalid")
    repair_fraction = _finite_number(repair.get("repair_fraction_per_cycle"))
    if repair_fraction is None or not 0.0 < repair_fraction <= 1.0:
        blockers.append(f"{context}:repair_fraction_invalid")
    record_commit_cycles = _strict_int(repair.get("record_commit_cycles"))
    if record_commit_cycles is None or not 1 <= record_commit_cycles <= 4096:
        blockers.append(f"{context}:record_commit_cycles_invalid")
    elif cycles is not None and record_commit_cycles > cycles:
        invalidators.append(f"{context}:record_commit_cycles_exceed_cycles")
    observer = _mapping(config.get("observer_capture"))
    _exact_fields(observer, _OBSERVER_FIELDS, f"{context}:observer_capture", blockers, invalidators)
    if observer.get("prediction_control") != "semantic_hash_shuffle_v1":
        invalidators.append(f"{context}:prediction_control_mismatch")
    if observer.get("feedback_enabled") is not True:
        blockers.append(f"{context}:observer_feedback_not_enabled")
    checkpoint_interval = _strict_int(observer.get("checkpoint_interval"))
    if checkpoint_interval is None or not 1 <= checkpoint_interval <= 1_000_000:
        blockers.append(f"{context}:checkpoint_interval_invalid")
    observer_support_size = _strict_int(observer.get("support_size"))
    carrier_count = _strict_int(source.get("carrier_count"))
    if (
        observer_support_size is None
        or observer_support_size < 1
        or (carrier_count is not None and observer_support_size > carrier_count)
    ):
        blockers.append(f"{context}:observer_support_size_invalid")
    observer_samples = _strict_int(observer.get("samples_per_observer"))
    if observer_samples is None or not 3 <= observer_samples <= 64:
        blockers.append(f"{context}:observer_samples_invalid")

    for role, component in (
        ("source_federation", source),
        ("support_regulator", support),
        ("source_dynamics", generator),
        ("source_dynamics", repair),
        ("observer_capture", observer),
    ):
        registry_entry = _mapping(producer_registry.get(role))
        if component.get("producer_id") != registry_entry.get("producer_id"):
            invalidators.append(f"{context}:producer_id_mismatch:{role}")
        if component.get("source_code_sha256") != registry_entry.get("source_code_sha256"):
            invalidators.append(f"{context}:source_code_sha256_mismatch:{role}")
        if not _strict_sha256(component.get("source_code_sha256")):
            blockers.append(f"{context}:source_code_sha256_malformed:{role}")

    if observer_counts is None:
        observer_counts = _observer_counts(_mapping(scaling.get("observer_scaling")), REQUIRED_RUNGS)
    if support_counts is None:
        support_counts = _int_keyed_mapping(scaling.get("support_counts_by_rung"))
    if rung is not None:
        if support_count != support_counts.get(rung):
            invalidators.append(f"{context}:support_count_mismatches_scaling")
        if _strict_int(observer.get("observer_count")) != (
            observer_counts or {}
        ).get(rung):
            invalidators.append(f"{context}:observer_count_mismatches_scaling")
    observer_count = _strict_int(observer.get("observer_count"))
    if (
        observer_count is None
        or observer_count < 1
        or (carrier_count is not None and observer_count > carrier_count)
    ):
        blockers.append(f"{context}:observer_count_outside_runtime_range")
    for field in ("cycles", "record_commit_cycles"):
        if _strict_int(repair.get(field)) != _strict_int(scaling.get(field)):
            invalidators.append(f"{context}:{field}_mismatches_scaling")
    if _finite_number(repair.get("repair_fraction_per_cycle")) != _finite_number(
        scaling.get("repair_fraction_per_cycle")
    ):
        invalidators.append(f"{context}:repair_fraction_mismatches_scaling")
    return _stage(blockers, invalidators)


def _validate_registries(plan: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    invalidators: list[str] = []
    producers = _mapping(plan.get("producer_registry"))
    checkers = _mapping(plan.get("checker_registry"))
    _exact_fields(producers, frozenset(ALLOWED_PRODUCER_IDS), "producer_registry", blockers, invalidators)
    _exact_fields(checkers, frozenset(ALLOWED_CHECKER_IDS), "checker_registry", blockers, invalidators)
    for role, allowed in ALLOWED_PRODUCER_IDS.items():
        entry = _mapping(producers.get(role))
        _exact_fields(entry, _REGISTRY_ENTRY_FIELDS, f"producer_registry:{role}", blockers, invalidators)
        if entry and entry.get("producer_id") not in allowed:
            invalidators.append(f"producer_registry:{role}:id_not_allowlisted")
        if entry and not _strict_sha256(entry.get("source_code_sha256")):
            invalidators.append(f"producer_registry:{role}:source_code_sha256_malformed")
    for role, allowed in ALLOWED_CHECKER_IDS.items():
        entry = _mapping(checkers.get(role))
        _exact_fields(entry, _CHECKER_ENTRY_FIELDS, f"checker_registry:{role}", blockers, invalidators)
        if entry and entry.get("checker_id") not in allowed:
            invalidators.append(f"checker_registry:{role}:id_not_allowlisted")
        if entry and not _strict_sha256(entry.get("checker_code_sha256")):
            invalidators.append(f"checker_registry:{role}:checker_code_sha256_malformed")
    return _stage(blockers, invalidators)


def _validate_thresholds(
    values: Mapping[str, Any], blockers: list[str], invalidators: list[str]
) -> None:
    _exact_fields(values, _THRESHOLD_FIELDS, "plan:thresholds", blockers, invalidators)
    for field in _THRESHOLD_FIELDS:
        number = _finite_number(values.get(field))
        if number is None or number < 0.0:
            blockers.append(f"plan:thresholds:{field}_invalid")
    power = _finite_number(values.get("curvature_minimum_power"))
    if power is not None and not 0.0 < power <= 1.0:
        invalidators.append(
            "plan:thresholds:curvature_minimum_power_must_lie_in_open_closed_unit_interval"
        )


def _validate_calibrations(
    values: Mapping[str, Any], blockers: list[str], invalidators: list[str]
) -> None:
    _exact_fields(values, _CALIBRATION_FIELDS, "plan:calibrations", blockers, invalidators)
    for field in (
        "clock_calibration_sha256",
        "geometry_calibration_sha256",
        "curvature_calibration_sha256",
    ):
        if not _strict_sha256(values.get(field)):
            blockers.append(f"plan:calibrations:{field}_malformed")
    if values.get("independent_of_campaign_source_seeds") is not True:
        invalidators.append("plan:calibrations:not_independent_of_campaign_seeds")
    if values.get("frozen_before_source_capture") is not True:
        blockers.append("plan:calibrations:not_frozen_before_capture")
    if type(values.get("physical_threshold_calibration_receipt")) is not bool:
        invalidators.append(
            "plan:calibrations:physical_threshold_calibration_receipt_must_be_boolean"
        )


def _validate_split(
    values: Mapping[str, Any], blockers: list[str], invalidators: list[str]
) -> None:
    _exact_fields(values, _SPLIT_FIELDS, "plan:split_contract", blockers, invalidators)
    if values.get("algorithm_id") != "semantic_hash_split_v1":
        invalidators.append("plan:split_contract:algorithm_mismatch")
    if not _strict_sha256(values.get("assignment_salt_sha256")):
        blockers.append("plan:split_contract:salt_hash_malformed")
    if not _open_unit_interval(values.get("holdout_fraction")):
        blockers.append("plan:split_contract:holdout_fraction_invalid")
    if values.get("derivation") != "semantic_event_id_hash_threshold_v1":
        invalidators.append("plan:split_contract:derivation_mismatch")
    if values.get("heldout_ids_materialized_before_capture") is not False:
        invalidators.append("plan:split_contract:preselected_holdout_ids_forbidden")


def _validate_scaling(
    values: Mapping[str, Any],
    rungs: Sequence[int],
    blockers: list[str],
    invalidators: list[str],
) -> tuple[dict[int, int] | None, dict[int, int]]:
    _exact_fields(values, _SCALING_FIELDS, "plan:scaling_contract", blockers, invalidators)
    if values.get("carrier_count_law") != "exact_rung_cardinality_v1":
        invalidators.append("plan:scaling_contract:carrier_count_law_mismatch")
    if values.get("support_regulator_law") != "first_icosahedral_cell_count_at_or_above_rung_v1":
        invalidators.append("plan:scaling_contract:support_law_mismatch")
    support = _int_keyed_mapping(values.get("support_counts_by_rung"))
    expected_support = {rung: _first_icosahedral_count_at_least(rung) for rung in rungs}
    if support != expected_support:
        invalidators.append("plan:scaling_contract:support_counts_mismatch")
    observer_spec = _mapping(values.get("observer_scaling"))
    _exact_fields(
        observer_spec,
        _OBSERVER_SCALING_FIELDS,
        "plan:scaling_contract:observer_scaling",
        blockers,
        invalidators,
    )
    observer = _observer_counts(observer_spec, rungs)
    if observer is None or _int_keyed_mapping(observer_spec.get("counts_by_rung")) != observer:
        invalidators.append("plan:scaling_contract:observer_scaling_mismatch")
    cycles = _strict_int(values.get("cycles"))
    if cycles is None or not 1 <= cycles <= 4096:
        blockers.append("plan:scaling_contract:cycles_invalid")
    fraction = _finite_number(values.get("repair_fraction_per_cycle"))
    if fraction is None or not 0.0 < fraction <= 1.0:
        blockers.append("plan:scaling_contract:repair_fraction_invalid")
    record_commit_cycles = _strict_int(values.get("record_commit_cycles"))
    if record_commit_cycles is None or not 1 <= record_commit_cycles <= 4096:
        blockers.append("plan:scaling_contract:record_commit_cycles_invalid")
    elif (
        cycles is not None
        and record_commit_cycles > cycles
    ):
        invalidators.append(
            "plan:scaling_contract:record_commit_cycles_exceed_cycles"
        )
    return observer, support


def _validate_archive(
    values: Mapping[str, Any], blockers: list[str], invalidators: list[str]
) -> None:
    _exact_fields(values, _ARCHIVE_FIELDS, "plan:archive_boundary", blockers, invalidators)
    if values.get("frozen_before_source_capture") is not True:
        blockers.append("plan:archive_boundary:not_frozen_before_capture")
    if values.get("retune_after_freeze") is not False:
        invalidators.append("plan:archive_boundary:retune_after_freeze_not_false")
    if values.get("archived_16k_failure_preserved") is not True:
        invalidators.append("plan:archive_boundary:archived_16k_failure_not_preserved")
    if values.get("archived_outcomes_used_for_threshold_selection") is not False:
        invalidators.append("plan:archive_boundary:archived_outcome_leak")
    if not _strict_sha256(values.get("historical_receipt_byte_sha256")):
        invalidators.append("plan:archive_boundary:historical_receipt_hash_invalid")
    elif (
        values.get("historical_receipt_byte_sha256")
        != REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256
    ):
        invalidators.append("plan:archive_boundary:historical_receipt_hash_mismatch")
    campaign_sha = values.get("historical_campaign_sha256")
    if not isinstance(campaign_sha, str) or re.fullmatch(r"[0-9a-f]{64}", campaign_sha) is None:
        invalidators.append("plan:archive_boundary:historical_campaign_hash_invalid")
    elif campaign_sha != REGISTERED_HISTORICAL_CAMPAIGN_SHA256:
        invalidators.append("plan:archive_boundary:historical_campaign_hash_mismatch")
    if values.get("historical_16k_source_seed") != REGISTERED_HISTORICAL_16K_SOURCE_SEED:
        invalidators.append("plan:archive_boundary:historical_16k_source_seed_invalid")
    if values.get("historical_16k_rung") != 16_384:
        invalidators.append("plan:archive_boundary:historical_16k_rung_invalid")
    if values.get("historical_16k_joint_independent_receipt") is not False:
        invalidators.append("plan:archive_boundary:historical_16k_result_not_preserved")
    if values.get("historical_stable_branch_failure_established") is not False:
        invalidators.append("plan:archive_boundary:historical_stable_failure_overclaimed")


def _validate_target_firewall(config: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    invalidators: list[str] = []
    hits: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    hits.append(f"{path}:non_string_key")
                    continue
                tokens = _tokens(key)
                if tokens & _TARGET_TOKENS:
                    hits.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str):
            if _tokens(value) & _TARGET_TOKENS:
                hits.append(path)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number) and any(
                math.isclose(number, target, rel_tol=0.0, abs_tol=1.0e-12)
                for target in (math.pi, 2.0 * math.pi, 4.0 * math.pi)
            ):
                hits.append(path)

    visit(config, "config")
    if not config:
        blockers.append("config:target_firewall_scope_missing")
    invalidators.extend(
        f"config:source_target_firewall_hit:{hit}" for hit in sorted(set(hits))
    )
    return _stage(
        blockers,
        invalidators,
        evidence={"scanned_entire_runtime_config": True, "forbidden_paths": sorted(set(hits))},
    )


def _observer_counts(spec: Mapping[str, Any], rungs: Sequence[int]) -> dict[int, int] | None:
    if spec.get("law_id") != "power_law_ceil_v1":
        return None
    coefficient = _finite_number(spec.get("coefficient"))
    exponent = _finite_number(spec.get("exponent"))
    minimum = _strict_int(spec.get("minimum"))
    maximum = _strict_int(spec.get("maximum"))
    if (
        coefficient is None
        or coefficient <= 0.0
        or exponent is None
        or not 0.0 <= exponent <= 1.0
        or minimum is None
        or maximum is None
        or minimum <= 0
        or maximum < minimum
    ):
        return None
    return {
        rung: min(maximum, max(minimum, math.ceil(coefficient * (rung**exponent))))
        for rung in rungs
    }


def _first_icosahedral_count_at_least(rung: int) -> int:
    level = 0
    count = 20
    while count < rung:
        level += 1
        count = 20 * (4**level)
    return count


def _source_input_projection(
    config: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Project one frozen cell onto the source-capture runtime vocabulary."""

    cell = _mapping(config.get("cell"))
    federation = _mapping(config.get("source_federation"))
    generator = _mapping(config.get("source_generator"))
    repair = _mapping(config.get("repair_dynamics"))
    observer = _mapping(config.get("observer_capture"))
    support = _mapping(config.get("support_regulator"))
    result = {
        "carrier_count": federation.get("carrier_count"),
        "seed": cell.get("seed"),
        "rung": cell.get("rung"),
        "replicate_id": cell.get("replicate_id"),
        "preregistered_plan_sha256": plan.get("plan_sha256"),
        "propagation_steps": generator.get("propagation_steps"),
        "intrinsic_step": generator.get("intrinsic_step"),
        "coupling_strength": generator.get("coupling_strength"),
        "state_space": generator.get("state_space"),
        "rng_family": generator.get("rng_family"),
        "initialization_distribution": generator.get(
            "initialization_distribution"
        ),
        "intrinsic_phase_distribution": generator.get(
            "intrinsic_phase_distribution"
        ),
        "seam_update_rule": repair.get("seam_update_rule"),
        "cycles": repair.get("cycles"),
        "repair_fraction_per_cycle": repair.get("repair_fraction_per_cycle"),
        "record_commit_cycles": repair.get("record_commit_cycles"),
        "observer_count": observer.get("observer_count"),
        "observer_support_size": observer.get("support_size"),
        "observer_samples": observer.get("samples_per_observer"),
        "prediction_control": observer.get("prediction_control"),
        "feedback_enabled": observer.get("feedback_enabled"),
        "checkpoint_interval": observer.get("checkpoint_interval"),
        "support_refinement_level": support.get("refinement_level"),
        "geometry_sample_count": generator.get("geometry_sample_count"),
    }
    if set(result) != SOURCE_INPUT_FIELDS:
        raise RuntimeError("internal source-input projection field mismatch")
    return result


def _load_source(
    source: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    if isinstance(source, Mapping):
        try:
            value = dict(source)
            digest = canonical_sha256(value)
        except (TypeError, ValueError):
            return {}, {"source_kind": "mapping"}, [], ["bundle:not_finite_canonical_json"]
        return value, {"source_kind": "mapping", "canonical_sha256": digest}, [], []
    path = Path(source)
    if not path.is_file() or path.is_symlink():
        return {}, {"source_kind": "file", "path": str(path)}, ["bundle:file_missing"], []
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {}, {"source_kind": "file", "path": str(path)}, [], ["bundle:invalid_json"]
    if not isinstance(value, Mapping):
        return {}, {"source_kind": "file", "path": str(path)}, [], ["bundle:root_not_object"]
    return dict(value), {
        "source_kind": "file",
        "path": str(path),
        "raw_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }, [], []


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_key:{key}")
        result[key] = value
    return result


def _exact_fields(
    row: Mapping[str, Any],
    expected: frozenset[str],
    context: str,
    blockers: list[str],
    invalidators: list[str],
) -> None:
    missing = expected - set(row)
    extra = set(row) - expected
    if missing:
        blockers.append(f"{context}:missing_fields:" + ",".join(sorted(missing)))
    if extra:
        invalidators.append(f"{context}:unknown_fields:" + ",".join(sorted(extra)))


def _stage(
    blockers: Sequence[str],
    invalidators: Sequence[str],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blocked = list(dict.fromkeys(blockers))
    invalid = list(dict.fromkeys(invalidators))
    status = (
        PrerunStatus.INSTRUMENT_INVALID.value
        if invalid
        else PrerunStatus.BLOCKED.value
        if blocked
        else PrerunStatus.VALID_PASS.value
    )
    return {
        "status": status,
        "passed": status == PrerunStatus.VALID_PASS.value,
        "blockers": blocked,
        "invalidators": invalid,
        "evidence": dict(evidence or {}),
    }


def _cell_key(value: Mapping[str, Any]) -> tuple[int, int, str] | None:
    seed = _strict_int(value.get("seed"))
    rung = _strict_int(value.get("rung"))
    replicate = value.get("replicate_id")
    if (
        seed is None
        or not 0 <= seed <= 2**63 - 1
        or rung is None
        or not isinstance(replicate, str)
        or _REPLICATE_ID_RE.fullmatch(replicate) is None
    ):
        return None
    return seed, rung, str(replicate)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _strict_int(value: Any) -> int | None:
    return value if type(value) is int else None


def _positive_int(value: Any) -> bool:
    integer = _strict_int(value)
    return integer is not None and integer > 0


def _strict_int_sequence(value: Any) -> list[int]:
    rows = _sequence(value)
    return rows if all(type(row) is int for row in rows) else []


def _string_sequence(value: Any) -> list[str]:
    rows = _sequence(value)
    return rows if all(_nonempty_string(row) for row in rows) else []


def _int_keyed_mapping(value: Any) -> dict[int, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[int, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
            return {}
        integer_value = _strict_int(item)
        if integer_value is None:
            return {}
        result[int(key)] = integer_value
    return result


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _open_unit_interval(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and 0.0 < number < 1.0


def _strict_sha256(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "")))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _tokens(value: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_").split("_"))


__all__ = [
    "ALLOWED_CHECKER_IDS",
    "ALLOWED_PRODUCER_IDS",
    "CELL_CONFIG_SCHEMA",
    "PLAN_SCHEMA",
    "PrerunStatus",
    "REGISTERED_HISTORICAL_16K_SOURCE_SEED",
    "REGISTERED_HISTORICAL_CAMPAIGN_SHA256",
    "REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256",
    "SCHEMA_VERSION",
    "SOURCE_INPUT_FIELDS",
    "canonical_sha256",
    "frozen_campaign_family_sha256",
    "physical_h3_kms_prerun_report",
    "physical_h3_kms_source_inputs",
]
