"""Independent fail-closed validation for the causal self-reading receipt."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from oph_fpe.core.screen_ports import echosahedral_patch_record_signature
from oph_fpe.evidence.hashes import stable_json_hash
from oph_fpe.gauge.covariant_overlap import GAUGE_COVARIANT_OVERLAP_SCHEMA


SCHEMA_VERSION = "oph_source_repair_record_observer_contract_v3"
MODE = "source_dynamics_repair_record_observer_contract"
RUN_BINDING_SCHEMA = "oph_self_reading_same_run_binding_v1"
CAUSAL_EVENT_LOG_SCHEMA = "oph_record_feedback_causal_event_log_v2"
CAUSAL_POLICY_SCHEMA = "oph_committed_record_port_copy_action_v2"
COMMITTED_RECORD_SNAPSHOT_SCHEMA = "oph_committed_local_record_snapshot_v1"
COUNTERFACTUAL_RECORD_SNAPSHOT_SCHEMA = (
    "oph_counterfactual_local_record_snapshot_v1"
)
RECORD_COMMIT_LOG_SCHEMA = "oph_committed_local_record_log_v1"


def validate_self_reading_report_structure(
    report: Mapping[str, Any] | None,
    *,
    materialized_observer_count: int | None = None,
    materialized_observer_id_set_hash: str | None = None,
) -> dict[str, Any]:
    """Validate the positive causal receipt from its nested audit evidence.

    A few top-level booleans are never sufficient. The validator also checks
    the producer schema, nested read/write counts, every causal audit row, its
    hash, and (when supplied) the same-run materialized observer count.
    """

    source = report if isinstance(report, Mapping) else {}
    generic_blockers: list[str] = []
    if source.get("schema_version") != SCHEMA_VERSION:
        generic_blockers.append("source_contract_schema_invalid")
    if source.get("mode") != MODE:
        generic_blockers.append("source_contract_mode_invalid")

    generic_receipts = (
        "PATCH_LOCAL_STATE_RECEIPT",
        "PATCH_PORT_BOUNDARY_RECEIPT",
        "PATCH_READBACK_RECEIPT",
        "PATCH_ALL_PORT_READBACK_RECEIPT",
        "RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT",
        "LOCAL_REPAIR_DYNAMICS_RECEIPT",
        "RECORD_COMMIT_RECEIPT",
        "OBSERVER_SELF_READING_RECORD_LOOP_RECEIPT",
    )
    for key in generic_receipts:
        if source.get(key) is not True:
            generic_blockers.append(f"{key.lower()}_missing_or_false")
    architecture = source.get("source_architecture")
    if not isinstance(architecture, Mapping) or not (
        architecture.get("bounded_patch_system") is True
        and architecture.get("simulation_native_source") is True
        and _positive_int(architecture.get("local_state_factor_count"))
        and _positive_int(architecture.get("boundary_port_count"))
        and architecture.get("all_local_port_readout_maps_materialized") is True
        and architecture.get("all_local_port_states_bound_into_records") is True
    ):
        generic_blockers.append("bounded_observer_architecture_evidence_invalid")
    repair = source.get("repair_dynamics")
    if not isinstance(repair, Mapping) or not (
        repair.get("local_update_rule") is True
        and repair.get("uses_only_local_state_and_ports") is True
        and _positive_int(repair.get("repair_event_count"))
        and repair.get("nonlocal_write_count") == 0
        and _nonempty_string(repair.get("repair_rule_hash"))
        and _nonempty_string(repair.get("repair_event_log_hash"))
    ):
        generic_blockers.append("local_repair_dynamics_evidence_invalid")

    for key, blocker in (
        (
            "OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT",
            "observer_like_self_reading_source_contract",
        ),
        ("RECORD_READ_AFTER_WRITE_RECEIPT", "record_read_after_write_receipt"),
        (
            "OBSERVER_READBACK_FEEDBACK_CAUSAL_LOOP_RECEIPT",
            "observer_readback_feedback_causal_loop_receipt",
        ),
    ):
        if source.get(key) is not True:
            generic_blockers.append(blocker)

    record = source.get("record_observer")
    if not isinstance(record, Mapping):
        generic_blockers.append("record_observer_evidence_missing")
        record = {}
    rows = record.get("record_feedback_audit_rows")
    if not isinstance(rows, list) or not rows:
        generic_blockers.append("record_feedback_audit_rows_missing")
        rows = []

    boundary_port_count = (
        architecture.get("boundary_port_count")
        if isinstance(architecture, Mapping)
        else None
    )
    valid_rows = bool(rows) and all(
        valid_causal_audit_row(row, boundary_port_count=boundary_port_count)
        for row in rows
    )
    event_ids = [
        row.get("event_id")
        for row in rows
        if isinstance(row, Mapping) and _nonempty_string(row.get("event_id"))
    ]
    if len(event_ids) != len(rows) or len(event_ids) != len(set(event_ids)):
        valid_rows = False
    if not valid_rows:
        generic_blockers.append("record_feedback_audit_rows_invalid")
    audited_observer_ids = sorted(
        {
            int(row["observer_id"])
            for row in rows
            if isinstance(row, Mapping) and _nonnegative_int(row.get("observer_id"))
        }
    )
    audited_observer_id_set_hash = observer_id_set_hash(audited_observer_ids)

    readback_count = sum(
        _int_or_zero(row.get("read_count")) for row in rows if isinstance(row, Mapping)
    )
    feedback_event_count = sum(
        _int_or_zero(row.get("write_count")) for row in rows if isinstance(row, Mapping)
    )
    if not _positive_int(record.get("observer_count")):
        generic_blockers.append("record_observer_count_invalid")
    if (
        not _positive_int(record.get("committed_record_count"))
        or record.get("historical_committed_record_count")
        != record.get("committed_record_count")
    ):
        generic_blockers.append("historical_committed_record_count_invalid")
    if not _nonnegative_int(record.get("current_committed_record_count")):
        generic_blockers.append("current_committed_record_count_invalid")
    if record.get("causally_verified_observer_count") != len(audited_observer_ids):
        generic_blockers.append("causally_verified_observer_count_mismatch")
    if record.get("readback_count") != readback_count or readback_count <= 0:
        generic_blockers.append("readback_count_mismatch")
    if record.get("feedback_event_count") != feedback_event_count or feedback_event_count <= 0:
        generic_blockers.append("feedback_event_count_mismatch")
    if record.get("readback_changes_future_local_actions") is not True:
        generic_blockers.append("nested_readback_changes_future_actions_false")
    if record.get("records_causally_bound_to_writes") is not True:
        generic_blockers.append("nested_records_causally_bound_to_writes_false")
    if record.get("orphan_read_count") != 0:
        generic_blockers.append("orphan_read_count_nonzero")
    if record.get("external_cap_refresh_is_observer_feedback") is not False:
        generic_blockers.append("external_cap_refresh_feedback_not_excluded")
    expected_hash = stable_json_hash(rows) if rows else None
    if not rows or record.get("record_readback_feedback_log_hash") != expected_hash:
        generic_blockers.append("record_feedback_audit_hash_mismatch")

    reported_observer_count = _int_or_none(record.get("observer_count"))
    if materialized_observer_count is not None and reported_observer_count != int(
        materialized_observer_count
    ):
        generic_blockers.append("same_run_observer_count_mismatch")
    if (
        materialized_observer_count is not None
        and len(audited_observer_ids) != int(materialized_observer_count)
    ):
        generic_blockers.append("causal_audit_does_not_cover_observer_population")
    if (
        materialized_observer_id_set_hash is not None
        and audited_observer_id_set_hash != materialized_observer_id_set_hash
    ):
        generic_blockers.append("causal_audit_observer_population_mismatch")

    generic_blockers = list(dict.fromkeys(generic_blockers))
    source_qualified_blockers = list(generic_blockers)
    for key in (
        "SOURCE_PATCH_ARCHITECTURE_RECEIPT",
        "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT",
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
        "FEDERATION_SEWING_RECEIPT",
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT",
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT",
        "OPH_SOURCE_QUALIFIED_ATOMIC_SELF_READING_SYSTEM_RECEIPT",
    ):
        if source.get(key) is not True:
            source_qualified_blockers.append(f"{key.lower()}_missing_or_false")
    if not isinstance(architecture, Mapping) or not (
        architecture.get("carrier_is_not_support_chart_cell") is True
        and architecture.get("carrier_is_not_primitive_observer") is True
        and architecture.get("carrier_support_conflation_present") is False
    ):
        source_qualified_blockers.append("source_carrier_separation_evidence_invalid")
    if not isinstance(repair, Mapping) or repair.get("target_free_rule") is not True:
        source_qualified_blockers.append("target_free_repair_rule_invalid")
    if source.get("source_generator_target_free") is not True or source.get(
        "source_forbidden_target_hits"
    ) not in ([], ()):
        source_qualified_blockers.append("source_generator_target_free_receipt_invalid")
    source_qualified_blockers = list(dict.fromkeys(source_qualified_blockers))
    return {
        "passed": not generic_blockers,
        "blockers": generic_blockers,
        "generic_causal_self_reading_passed": not generic_blockers,
        "generic_causal_self_reading_blockers": generic_blockers,
        "source_qualified_atomic_passed": not source_qualified_blockers,
        "source_qualified_atomic_blockers": source_qualified_blockers,
        "schema_version": source.get("schema_version"),
        "mode": source.get("mode"),
        "materialized_observer_count": materialized_observer_count,
        "reported_observer_count": reported_observer_count,
        "causally_verified_observer_count": len(audited_observer_ids),
        "audited_observer_id_set_hash": audited_observer_id_set_hash,
        "readback_count": readback_count,
        "feedback_event_count": feedback_event_count,
        "record_feedback_audit_hash_recomputed": expected_hash,
    }


def validate_self_reading_contract(
    report: Mapping[str, Any] | None,
    *,
    materialized_observer_count: int | None = None,
    materialized_observer_id_set_hash: str | None = None,
) -> dict[str, Any]:
    """Fail closed when only a report, without its source run, is supplied."""

    structural = validate_self_reading_report_structure(
        report,
        materialized_observer_count=materialized_observer_count,
        materialized_observer_id_set_hash=materialized_observer_id_set_hash,
    )
    structural_blockers = list(structural["generic_causal_self_reading_blockers"])
    blockers = [*structural_blockers, "run_bundle_evidence_not_validated"]
    source_blockers = [
        *structural["source_qualified_atomic_blockers"],
        "run_bundle_evidence_not_validated",
    ]
    return {
        **structural,
        "passed": False,
        "blockers": list(dict.fromkeys(blockers)),
        "generic_causal_self_reading_passed": False,
        "generic_causal_self_reading_blockers": list(dict.fromkeys(blockers)),
        "source_qualified_atomic_passed": False,
        "source_qualified_atomic_blockers": list(dict.fromkeys(source_blockers)),
        "structurally_valid": structural["passed"],
        "structural_validation": structural,
    }


def validate_run_self_reading_contract(
    run_dir: Path,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a contract and its same-directory observer-population evidence."""

    root = Path(run_dir)
    source = report
    if source is None:
        source = _read_json(root / "source_dynamics_repair_record_observer_report.json")
    population = validate_observer_population(root)
    contract = validate_self_reading_report_structure(
        source,
        materialized_observer_count=(
            population["materialized_observer_count"]
            if population["passed"]
            else None
        ),
        materialized_observer_id_set_hash=(
            population["observer_id_set_hash"] if population["passed"] else None
        ),
    )
    blockers = list(contract["generic_causal_self_reading_blockers"])
    source_qualified_blockers = list(
        contract["source_qualified_atomic_blockers"]
    )
    if not population["passed"]:
        blockers.append("observer_population_evidence_invalid")
        source_qualified_blockers.append("observer_population_evidence_invalid")
    run_binding = source.get("run_binding") if isinstance(source, Mapping) else None
    binding_validation = _validate_run_binding(root, run_binding, population)
    if not binding_validation["passed"]:
        blockers.append("self_reading_same_run_binding_invalid")
        source_qualified_blockers.append("self_reading_same_run_binding_invalid")
    reported_rows = (
        (source.get("record_observer") or {}).get("record_feedback_audit_rows")
        if isinstance(source, Mapping)
        else None
    )
    causal_artifact_validation = _validate_causal_event_artifact(
        root,
        source.get("causal_event_artifact") if isinstance(source, Mapping) else None,
        reported_rows,
        run_binding,
    )
    if not causal_artifact_validation["passed"]:
        blockers.append("causal_event_artifact_invalid")
        source_qualified_blockers.append("causal_event_artifact_invalid")
    architecture = source.get("source_architecture") if isinstance(source, Mapping) else {}
    architecture = architecture if isinstance(architecture, Mapping) else {}
    manifest = _read_json(root / "manifest.json")
    record_commit_artifact_validation = _validate_record_commit_artifact(
        root,
        source.get("record_commit_artifact") if isinstance(source, Mapping) else None,
        run_binding,
        patch_count=_int_or_none(manifest.get("patch_count")),
        boundary_port_count=_int_or_none(architecture.get("boundary_port_count")),
        group_order=_int_or_none(
            (manifest.get("gauge_coupled_dynamics") or {}).get("group_order")
        ),
    )
    if not record_commit_artifact_validation["passed"]:
        blockers.append("record_commit_artifact_invalid")
        source_qualified_blockers.append("record_commit_artifact_invalid")
    record_observer = source.get("record_observer") if isinstance(source, Mapping) else {}
    record_observer = record_observer if isinstance(record_observer, Mapping) else {}
    if (
        record_observer.get("committed_record_count")
        != record_commit_artifact_validation.get("validated_commit_count")
        or record_observer.get("historical_committed_record_count")
        != record_commit_artifact_validation.get("validated_commit_count")
    ):
        blockers.append("record_commit_artifact_count_mismatch")
        source_qualified_blockers.append("record_commit_artifact_count_mismatch")
    causal_commit_link_validation = _validate_causal_commit_links(
        root,
        source.get("record_commit_artifact") if isinstance(source, Mapping) else None,
        reported_rows,
        config=_read_yaml_mapping(root / "config.yml"),
        patch_count=_int_or_none(manifest.get("patch_count")),
        edge_count=_int_or_none(manifest.get("edge_count")),
        group_order=_int_or_none(
            (manifest.get("gauge_coupled_dynamics") or {}).get("group_order")
        ),
    )
    if not causal_commit_link_validation["passed"]:
        blockers.append("causal_events_not_bound_to_commit_artifact")
        source_qualified_blockers.append(
            "causal_events_not_bound_to_commit_artifact"
        )
    source_bundle_validation = _validate_source_bundle(root, source)
    if not source_bundle_validation["generic_passed"]:
        blockers.append("source_bundle_architecture_or_repair_evidence_invalid")
        source_qualified_blockers.append(
            "source_bundle_architecture_or_repair_evidence_invalid"
        )
    recomputed_current_committed = (
        source_bundle_validation.get("patch_state_artifact_validation") or {}
    ).get("current_committed_record_count")
    if (
        recomputed_current_committed is None
        or record_observer.get("current_committed_record_count")
        != recomputed_current_committed
        or (
            _int_or_none(manifest.get("patch_count")) is not None
            and recomputed_current_committed > int(manifest["patch_count"])
        )
    ):
        blockers.append("current_committed_record_count_mismatch")
        source_qualified_blockers.append(
            "current_committed_record_count_mismatch"
        )
    # No independent verifier for the carrier/transaction/source-qualified
    # conjunction exists yet. Keeping this tier false is an evidence gate, not
    # a claim that another causal self-reading construction is impossible.
    source_qualified_blockers.append(
        "source_qualified_independent_verifier_unimplemented"
    )
    blockers = list(dict.fromkeys(blockers))
    source_qualified_blockers = list(dict.fromkeys(source_qualified_blockers))
    return {
        **contract,
        "passed": not blockers,
        "blockers": blockers,
        "generic_causal_self_reading_passed": not blockers,
        "generic_causal_self_reading_blockers": blockers,
        "source_qualified_atomic_passed": not source_qualified_blockers,
        "source_qualified_atomic_blockers": source_qualified_blockers,
        "observer_population_validation": population,
        "run_binding_validation": binding_validation,
        "causal_event_artifact_validation": causal_artifact_validation,
        "record_commit_artifact_validation": record_commit_artifact_validation,
        "causal_commit_link_validation": causal_commit_link_validation,
        "source_bundle_validation": source_bundle_validation,
    }


def validate_observer_population(run_dir: Path) -> dict[str, Any]:
    """Validate full or compact materialized-observer population custody.

    Compact runs retain only a deterministic enriched subset in JSONL. Their
    full population is independently checked against the hashed NPZ artifact,
    so compact storage is not mistaken for a smaller causal population.
    """

    root = Path(run_dir)
    report = _read_json(root / "observer_population_report.json")
    manifest_patch_count = _int_or_none(_read_json(root / "manifest.json").get("patch_count"))
    blockers: list[str] = []
    if report.get("mode") != "bounded_materialized_observer_population_v1":
        blockers.append("observer_population_schema_invalid")
    materialized = _int_or_none(report.get("materialized_observer_count"))
    if materialized is None or materialized <= 0:
        blockers.append("materialized_observer_count_invalid")
    if manifest_patch_count is None or manifest_patch_count <= 0:
        blockers.append("observer_population_patch_namespace_missing")

    retained_patch_count, retained_cap_count, jsonl_error = _observer_jsonl_counts(
        root / "observer_views.jsonl"
    )
    if jsonl_error:
        blockers.append(jsonl_error)
    if retained_patch_count <= 0:
        blockers.append("retained_patch_observer_rows_missing")
    if report.get("verbose_jsonl_patch_observer_count") != retained_patch_count:
        blockers.append("verbose_jsonl_patch_observer_count_mismatch")
    if report.get("verbose_jsonl_cap_observer_count") != retained_cap_count:
        blockers.append("verbose_jsonl_cap_observer_count_mismatch")
    if report.get("materialized_rows_preserved") is not True:
        blockers.append("materialized_observer_rows_not_preserved")

    compact = report.get("compact_population_artifact")
    population_binding_hash: str | None = None
    population_observer_id_set_hash: str | None = None
    if materialized is not None and retained_patch_count == materialized:
        if report.get("verbose_jsonl_population") != "all_materialized_observers":
            blockers.append("full_observer_population_mode_invalid")
        if compact not in (None, {}):
            blockers.append("unexpected_compact_observer_artifact")
        population_binding_hash, binding_blocker = _jsonl_population_binding_hash(
            root / "observer_views.jsonl"
        )
        if binding_blocker:
            blockers.append(binding_blocker)
        full_pairs, full_pairs_error = _jsonl_patch_identity_pairs(
            root / "observer_views.jsonl"
        )
        if full_pairs_error:
            blockers.append(full_pairs_error)
        else:
            full_ids = [observer_id for observer_id, _ in full_pairs]
            if len(set(full_ids)) != len(full_ids):
                blockers.append("observer_population_ids_not_unique")
            if manifest_patch_count is not None and any(
                observer_id < 0 or observer_id >= manifest_patch_count
                for observer_id in full_ids
            ):
                blockers.append("observer_population_ids_outside_patch_namespace")
            population_observer_id_set_hash = observer_id_set_hash(
                full_ids
            )
    elif materialized is not None and 0 < retained_patch_count < materialized:
        if (
            report.get("verbose_jsonl_population")
            != "deterministic_analysis_subset_plus_cap_observers"
        ):
            blockers.append("compact_observer_population_mode_invalid")
        compact_validation = _validate_compact_population_artifact(
            root,
            compact,
            materialized_observer_count=materialized,
            retained_patch_observer_count=retained_patch_count,
            reported_analyzed_count=report.get("observer_wide_analyzed_count"),
            manifest_patch_count=manifest_patch_count,
        )
        blockers.extend(compact_validation["blockers"])
        population_binding_hash = compact_validation["population_binding_hash"]
        population_observer_id_set_hash = compact_validation[
            "observer_id_set_hash"
        ]
    elif materialized is not None:
        blockers.append("retained_observer_count_exceeds_materialized_population")

    blockers = list(dict.fromkeys(blockers))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "report_path": str(root / "observer_population_report.json"),
        "materialized_observer_count": materialized,
        "retained_patch_observer_count": retained_patch_count,
        "retained_cap_observer_count": retained_cap_count,
        "storage_mode": report.get("verbose_jsonl_population"),
        "observer_population_binding_hash": population_binding_hash,
        "observer_id_set_hash": population_observer_id_set_hash,
    }


def _validate_compact_population_artifact(
    root: Path,
    artifact: Any,
    *,
    materialized_observer_count: int,
    retained_patch_observer_count: int,
    reported_analyzed_count: Any,
    manifest_patch_count: int | None,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {
            "blockers": ["compact_observer_artifact_metadata_missing"],
            "population_binding_hash": None,
            "observer_id_set_hash": None,
        }
    blockers: list[str] = []
    if artifact.get("schema") != "compact_materialized_observer_population_npz_v1":
        blockers.append("compact_observer_artifact_schema_invalid")
    relative_path = artifact.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        return {
            "blockers": [*blockers, "compact_observer_artifact_path_invalid"],
            "population_binding_hash": None,
            "observer_id_set_hash": None,
        }
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.name != relative_path:
        return {
            "blockers": [*blockers, "compact_observer_artifact_path_unsafe"],
            "population_binding_hash": None,
            "observer_id_set_hash": None,
        }
    path = root / candidate
    if not path.is_file():
        return {
            "blockers": [*blockers, "compact_observer_artifact_missing"],
            "population_binding_hash": None,
            "observer_id_set_hash": None,
        }
    payload = path.read_bytes()
    expected_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if artifact.get("sha256") != expected_hash:
        blockers.append("compact_observer_artifact_hash_mismatch")
    if artifact.get("byte_count") != len(payload):
        blockers.append("compact_observer_artifact_byte_count_mismatch")
    if artifact.get("materialized_observer_count") != materialized_observer_count:
        blockers.append("compact_materialized_observer_count_mismatch")
    if artifact.get("analysis_enriched_observer_count") != retained_patch_observer_count:
        blockers.append("compact_analysis_observer_count_mismatch")
    if reported_analyzed_count != retained_patch_observer_count:
        blockers.append("reported_analysis_observer_count_mismatch")
    population_binding_hash: str | None = None
    population_observer_id_set_hash: str | None = None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            observer_ids = np.asarray(arrays["observer_ids"])
            visible_hashes = np.asarray(arrays["visible_readout_hashes"])
            support_offsets = np.asarray(arrays["support_offsets"])
            analysis_mask = np.asarray(arrays["observer_wide_analysis_included"])
            if observer_ids.dtype.kind not in "iu":
                blockers.append("compact_observer_ids_not_integer_typed")
            if observer_ids.ndim != 1 or observer_ids.shape[0] != materialized_observer_count:
                blockers.append("compact_observer_id_population_mismatch")
            elif np.unique(observer_ids).size != materialized_observer_count:
                blockers.append("compact_observer_ids_not_unique")
            elif manifest_patch_count is None or np.any(observer_ids < 0) or np.any(
                observer_ids >= manifest_patch_count
            ):
                blockers.append("compact_observer_ids_outside_patch_namespace")
            if support_offsets.ndim != 1 or support_offsets.shape[0] != materialized_observer_count + 1:
                blockers.append("compact_support_offsets_population_mismatch")
            if analysis_mask.ndim != 1 or analysis_mask.shape[0] != materialized_observer_count:
                blockers.append("compact_analysis_mask_population_mismatch")
            elif int(np.count_nonzero(analysis_mask)) != retained_patch_observer_count:
                blockers.append("compact_analysis_mask_count_mismatch")
            if observer_ids.ndim == 1 and visible_hashes.ndim == 1 and (
                observer_ids.shape[0]
                == visible_hashes.shape[0]
                == materialized_observer_count
            ):
                population_binding_hash = observer_population_binding_hash(
                    [int(value) for value in observer_ids.tolist()],
                    [
                        value.decode("utf-8")
                        if isinstance(value, (bytes, np.bytes_))
                        else str(value)
                        for value in visible_hashes.tolist()
                    ],
                )
                population_observer_id_set_hash = observer_id_set_hash(
                    [int(value) for value in observer_ids.tolist()]
                )
                retained_pairs, retained_error = _jsonl_patch_identity_pairs(
                    root / "observer_views.jsonl"
                )
                if retained_error:
                    blockers.append(retained_error)
                else:
                    selected_pairs = [
                        (int(observer_ids[index]), _decode_hash(visible_hashes[index]))
                        for index in range(materialized_observer_count)
                        if bool(analysis_mask[index])
                    ]
                    if sorted(retained_pairs) != sorted(selected_pairs):
                        blockers.append("compact_retained_jsonl_population_mismatch")
            else:
                blockers.append("compact_visible_readout_population_mismatch")
    except (OSError, OverflowError, TypeError, ValueError, KeyError):
        blockers.append("compact_observer_artifact_unreadable")
    return {
        "blockers": blockers,
        "population_binding_hash": population_binding_hash,
        "observer_id_set_hash": population_observer_id_set_hash,
    }


def observer_population_binding_hash(
    observer_ids: list[int],
    visible_readout_hashes: list[str],
) -> str:
    """Hash identity/readout channels retained losslessly by both stores."""

    return stable_json_hash(
        {
            "schema": "oph_materialized_observer_population_binding_v1",
            "observer_ids": observer_ids,
            "visible_readout_hashes": visible_readout_hashes,
        }
    )


def observer_id_set_hash(observer_ids: list[int]) -> str:
    return stable_json_hash(
        {
            "schema": "oph_materialized_observer_id_set_v1",
            "observer_ids": sorted(observer_ids),
        }
    )


def write_causal_event_artifact(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    run_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _write_jsonl_artifact(
        path,
        rows,
        schema=CAUSAL_EVENT_LOG_SCHEMA,
        run_binding=run_binding,
    )


def write_record_commit_artifact(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    run_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write commit-time snapshots captured before any later feedback event."""

    return _write_jsonl_artifact(
        path,
        rows,
        schema=RECORD_COMMIT_LOG_SCHEMA,
        run_binding=run_binding,
    )


def _write_jsonl_artifact(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    schema: str,
    run_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        for row in rows
    ).encode("utf-8")
    target = Path(path)
    target.write_bytes(serialized)
    metadata = {
        "schema": schema,
        "path": target.name,
        "row_count": len(rows),
        "byte_count": len(serialized),
        "sha256": "sha256:" + hashlib.sha256(serialized).hexdigest(),
        "stable_rows_hash": stable_json_hash(rows),
    }
    if isinstance(run_binding, Mapping):
        metadata["run_binding_hash"] = stable_json_hash(dict(run_binding))
    return metadata


def _validate_causal_event_artifact(
    root: Path,
    metadata: Any,
    reported_rows: Any,
    run_binding: Any,
) -> dict[str, Any]:
    return _validate_jsonl_artifact(
        root,
        metadata,
        reported_rows,
        expected_schema=CAUSAL_EVENT_LOG_SCHEMA,
        artifact_name="causal_event",
        run_binding=run_binding,
    )


def _validate_record_commit_artifact(
    root: Path,
    metadata: Any,
    run_binding: Any,
    *,
    patch_count: int | None,
    boundary_port_count: int | None,
    group_order: int | None,
) -> dict[str, Any]:
    validation = _validate_jsonl_artifact(
        root,
        metadata,
        None,
        expected_schema=RECORD_COMMIT_LOG_SCHEMA,
        artifact_name="record_commit",
        run_binding=run_binding,
        require_reported_rows_match=False,
    )
    rows, row_error = _read_jsonl_rows_from_metadata(root, metadata)
    blockers = list(validation["blockers"])
    if row_error:
        blockers.append(row_error)
    if not rows:
        blockers.append("record_commit_rows_missing")
    if not all(
        valid_committed_record_snapshot(
            row,
            patch_count=patch_count,
            boundary_port_count=boundary_port_count,
            group_order=group_order,
        )
        for row in rows
    ):
        blockers.append("record_commit_rows_invalid")
    commit_ids = [
        row.get("commit_id")
        for row in rows
        if isinstance(row, Mapping) and _nonempty_string(row.get("commit_id"))
    ]
    if len(commit_ids) != len(rows) or len(commit_ids) != len(set(commit_ids)):
        blockers.append("record_commit_ids_not_unique")
    blockers = list(dict.fromkeys(blockers))
    return {
        **validation,
        "passed": not blockers,
        "blockers": blockers,
        "validated_commit_count": len(rows),
    }


def _validate_jsonl_artifact(
    root: Path,
    metadata: Any,
    reported_rows: Any,
    *,
    expected_schema: str,
    artifact_name: str,
    run_binding: Any,
    require_reported_rows_match: bool = True,
) -> dict[str, Any]:
    blockers: list[str] = []
    source = metadata if isinstance(metadata, Mapping) else {}
    if source.get("schema") != expected_schema:
        blockers.append(f"{artifact_name}_artifact_schema_invalid")
    relative_path = source.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        blockers.append(f"{artifact_name}_artifact_path_invalid")
        path = None
    else:
        candidate = Path(relative_path)
        if candidate.is_absolute() or candidate.name != relative_path:
            blockers.append(f"{artifact_name}_artifact_path_unsafe")
            path = None
        else:
            path = Path(root) / candidate
    artifact_rows: list[Any] = []
    payload = b""
    if path is None or not path.is_file():
        blockers.append(f"{artifact_name}_artifact_missing")
    else:
        try:
            payload = path.read_bytes()
            artifact_rows = [
                json.loads(line)
                for line in payload.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"{artifact_name}_artifact_unreadable")
    if source.get("byte_count") != len(payload):
        blockers.append(f"{artifact_name}_artifact_byte_count_mismatch")
    if source.get("sha256") != "sha256:" + hashlib.sha256(payload).hexdigest():
        blockers.append(f"{artifact_name}_artifact_hash_mismatch")
    if source.get("row_count") != len(artifact_rows):
        blockers.append(f"{artifact_name}_artifact_row_count_mismatch")
    if source.get("stable_rows_hash") != stable_json_hash(artifact_rows):
        blockers.append(f"{artifact_name}_artifact_rows_hash_mismatch")
    if require_reported_rows_match and (
        not isinstance(reported_rows, list) or artifact_rows != reported_rows
    ):
        blockers.append(f"{artifact_name}_artifact_report_rows_mismatch")
    expected_run_binding_hash = (
        stable_json_hash(dict(run_binding)) if isinstance(run_binding, Mapping) else None
    )
    if (
        expected_run_binding_hash is None
        or source.get("run_binding_hash") != expected_run_binding_hash
    ):
        blockers.append(f"{artifact_name}_artifact_run_binding_mismatch")
    blockers = list(dict.fromkeys(blockers))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "path": str(path) if path is not None else None,
        "row_count": len(artifact_rows),
    }


def _read_jsonl_rows_from_metadata(
    root: Path,
    metadata: Any,
) -> tuple[list[Any], str | None]:
    source = metadata if isinstance(metadata, Mapping) else {}
    relative_path = source.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        return [], "record_commit_artifact_path_invalid"
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.name != relative_path:
        return [], "record_commit_artifact_path_unsafe"
    try:
        rows = [
            json.loads(line)
            for line in (root / candidate).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], "record_commit_artifact_unreadable"
    return rows, None


def _validate_causal_commit_links(
    root: Path,
    metadata: Any,
    causal_rows: Any,
    *,
    config: Mapping[str, Any],
    patch_count: int | None,
    edge_count: int | None,
    group_order: int | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    commit_rows, commit_error = _read_jsonl_rows_from_metadata(root, metadata)
    if commit_error:
        blockers.append(commit_error)
    commit_map: dict[str, dict[str, Any]] = {}
    for row in commit_rows:
        if not isinstance(row, dict) or not _nonempty_string(row.get("commit_id")):
            blockers.append("record_commit_link_row_invalid")
            continue
        commit_id = str(row["commit_id"])
        if commit_id in commit_map and commit_map[commit_id] != row:
            blockers.append("record_commit_link_id_conflict")
        commit_map[commit_id] = row
    if not isinstance(causal_rows, list) or not causal_rows:
        blockers.append("causal_commit_link_rows_missing")
        causal_rows = []
    try:
        cycle_count = int((config.get("dynamics") or {}).get("cycles", 64))
    except (OverflowError, TypeError, ValueError):
        cycle_count = 0
        blockers.append("causal_event_cycle_domain_invalid")
    if (
        cycle_count <= 0
        or patch_count is None
        or patch_count <= 0
        or edge_count is None
        or edge_count < 0
    ):
        max_event_index = -1
        blockers.append("causal_event_domain_evidence_invalid")
    else:
        max_event_index = cycle_count * (patch_count + edge_count + 1) * 4
    used_commit_ids: set[str] = set()
    for row in causal_rows:
        if not isinstance(row, Mapping):
            blockers.append("causal_commit_link_row_invalid")
            continue
        commit_id = row.get("commit_id")
        embedded = row.get("committed_record_snapshot")
        if not _nonempty_string(commit_id) or not isinstance(embedded, dict):
            blockers.append("causal_commit_link_fields_invalid")
            continue
        committed = commit_map.get(str(commit_id))
        if committed is None or committed != embedded:
            blockers.append("causal_commit_snapshot_mismatch")
        if row.get("committed_record_snapshot_hash") != embedded.get("snapshot_hash"):
            blockers.append("causal_commit_snapshot_hash_mismatch")
        used_commit_ids.add(str(commit_id))
        if group_order is None or row.get("group_order") != group_order:
            blockers.append("causal_event_group_order_mismatch")
        for key in ("commit_cycle", "read_cycle", "write_cycle"):
            value = _int_or_none(row.get(key))
            if value is None or value < 0 or value >= cycle_count:
                blockers.append("causal_event_cycle_outside_run")
        for key in (
            "commit_event_index",
            "read_event_index",
            "write_event_index",
        ):
            value = _int_or_none(row.get(key))
            if value is None or value < 0 or value > max_event_index:
                blockers.append("causal_event_index_outside_run")
    blockers = list(dict.fromkeys(blockers))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "commit_log_row_count": len(commit_rows),
        "linked_causal_event_count": len(used_commit_ids),
        "configured_cycle_count": cycle_count,
        "maximum_event_index": max_event_index,
    }


def _jsonl_population_binding_hash(path: Path) -> tuple[str | None, str | None]:
    pairs, error = _jsonl_patch_identity_pairs(path)
    if error:
        return None, error
    if not pairs:
        return None, "observer_population_binding_rows_missing"
    return observer_population_binding_hash(
        [observer_id for observer_id, _ in pairs],
        [visible_hash for _, visible_hash in pairs],
    ), None


def _jsonl_patch_identity_pairs(
    path: Path,
) -> tuple[list[tuple[int, str]], str | None]:
    pairs: list[tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("view_type") != "patch_observer":
                    continue
                observer_id = _int_or_none(row.get("observer_id"))
                visible_hash = row.get("visible_readout_hash")
                if observer_id is None or not isinstance(visible_hash, str) or not visible_hash:
                    return [], "observer_population_binding_fields_invalid"
                pairs.append((observer_id, visible_hash))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], "observer_population_binding_source_unreadable"
    return pairs, None


def _decode_hash(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _validate_run_binding(
    root: Path,
    binding: Any,
    population: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    source = binding if isinstance(binding, Mapping) else {}
    if source.get("schema") != RUN_BINDING_SCHEMA:
        blockers.append("self_reading_run_binding_schema_invalid")
    seed_material = _read_json(root / "seed_material.json")
    manifest = _read_json(root / "manifest.json")
    config = _read_yaml_mapping(root / "config.yml")
    config_hash = stable_json_hash(config) if config else None
    try:
        config_seed = int(config.get("seed", 1)) if config else None
    except (OverflowError, TypeError, ValueError):
        config_seed = None
        blockers.append("same_run_config_seed_invalid")
    if config_hash is None:
        blockers.append("same_run_config_evidence_missing_or_invalid")
    if seed_material.get("config_hash") != config_hash:
        blockers.append("same_run_seed_material_config_hash_mismatch")
    if seed_material.get("seed") != config_seed:
        blockers.append("same_run_seed_material_seed_mismatch")
    expected = {
        "config_hash": config_hash,
        "seed": config_seed,
        "patch_count": manifest.get("patch_count"),
        "edge_count": manifest.get("edge_count"),
        "observer_population_binding_hash": population.get(
            "observer_population_binding_hash"
        ),
    }
    for key, value in expected.items():
        if value is None:
            blockers.append(f"same_run_{key}_evidence_missing")
        elif source.get(key) != value:
            blockers.append(f"same_run_{key}_mismatch")
    blockers = list(dict.fromkeys(blockers))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "reported": dict(source),
        "recomputed": expected,
    }


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_source_bundle(
    root: Path,
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Cross-check report claims against archived source-run artifacts.

    The existing trace does not retain a replayable write set for every repair.
    Consequently this verifier can establish architecture custody and trace
    arithmetic, while deliberately withholding the causal-local-repair gate.
    """

    source = report if isinstance(report, Mapping) else {}
    blockers: list[str] = []
    manifest = _read_json(root / "manifest.json")
    config = _read_yaml_mapping(root / "config.yml")
    screen_ports = _read_json(root / "screen_ports.json")
    patch_state = _read_json(root / "echosahedral_patch_state_report.json")
    architecture = source.get("source_architecture")
    if not isinstance(architecture, Mapping):
        architecture = {}
        blockers.append("source_architecture_report_missing")
    patch_count = _int_or_none(manifest.get("patch_count"))
    edge_count = _int_or_none(manifest.get("edge_count"))
    boundary_port_count = _int_or_none(architecture.get("boundary_port_count"))
    config_group = str((config.get("group") or {}).get("name", "S3")).upper()
    architecture_checks = bool(
        patch_count is not None
        and patch_count > 0
        and edge_count is not None
        and edge_count > 0
        and boundary_port_count == 12
        and architecture.get("carrier_count") == patch_count
        and architecture.get("group_name") == config_group
        and screen_ports.get("mode") == "explicit_named_echosahedral_ports"
        and screen_ports.get("ports_per_patch") == boundary_port_count
        and screen_ports.get("edge_count") == edge_count
        and screen_ports.get("overflow_count") == 0
        and screen_ports.get("port_names")
        == [f"P{index}" for index in range(boundary_port_count)]
        and patch_state.get("schema")
        == "oph.echosahedral_patch_federation_state.v1"
        and patch_state.get("patch_count") == patch_count
        and patch_state.get("ports_per_patch") == boundary_port_count
        and patch_state.get("materialized_local_port_state_count")
        == patch_count * boundary_port_count
        and patch_state.get("ECHOSAHEDRAL_PATCH_STATE_INSTANTIATION_RECEIPT")
        is True
        and patch_state.get("PATCH_ALL_PORT_READBACK_RECEIPT") is True
        and patch_state.get(
            "RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT"
        )
        is True
        and source.get("patch_federation_state") == patch_state
    )
    patch_artifact_validation = _validate_patch_state_artifact(
        root,
        patch_state,
        patch_count=patch_count,
        edge_count=edge_count,
        boundary_port_count=boundary_port_count,
    )
    if not architecture_checks:
        blockers.append("source_architecture_artifact_mismatch")
    if not patch_artifact_validation["passed"]:
        blockers.append("patch_state_artifact_invalid")

    repair_trace_validation = _validate_repair_trace(root, source, config)
    if not repair_trace_validation["passed"]:
        blockers.append("repair_trace_artifact_invalid")
    blockers.append("local_repair_write_set_independent_verifier_unimplemented")
    blockers = list(dict.fromkeys(blockers))
    return {
        "generic_passed": not blockers,
        "blockers": blockers,
        "architecture_cross_check_passed": architecture_checks,
        "patch_state_artifact_validation": patch_artifact_validation,
        "repair_trace_validation": repair_trace_validation,
        "local_repair_write_set_independently_verified": False,
    }


def _validate_patch_state_artifact(
    root: Path,
    patch_state_report: Mapping[str, Any],
    *,
    patch_count: int | None,
    edge_count: int | None,
    boundary_port_count: int | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    current_committed_record_count: int | None = None
    metadata = patch_state_report.get("artifact")
    if not isinstance(metadata, Mapping) or metadata.get("written") is not True:
        return {"passed": False, "blockers": ["patch_state_artifact_not_written"]}
    relative_path = metadata.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        return {"passed": False, "blockers": ["patch_state_artifact_path_invalid"]}
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.name != relative_path:
        return {"passed": False, "blockers": ["patch_state_artifact_path_unsafe"]}
    path = root / candidate
    if not path.is_file():
        return {"passed": False, "blockers": ["patch_state_artifact_missing"]}
    payload = path.read_bytes()
    if metadata.get("byte_count") != len(payload):
        blockers.append("patch_state_artifact_byte_count_mismatch")
    if metadata.get("file_sha256") != "sha256:" + hashlib.sha256(payload).hexdigest():
        blockers.append("patch_state_artifact_hash_mismatch")
    try:
        with np.load(path, allow_pickle=False) as arrays:
            state = np.asarray(arrays["patch_port_state"])
            canonical = np.asarray(arrays["canonical_record_port_state"])
            left = np.asarray(arrays["edge_left"])
            right = np.asarray(arrays["edge_right"])
            left_port = np.asarray(arrays["left_port"])
            right_port = np.asarray(arrays["right_port"])
            routed_left = np.asarray(arrays["routed_left_state"])
            routed_right = np.asarray(arrays["routed_right_state"])
            signatures = np.asarray(arrays["record_signature"])
            committed = np.asarray(arrays["committed"])
            current_committed_record_count = int(np.count_nonzero(committed))
            names = np.asarray(arrays["port_names"])
            if state.shape != (patch_count, boundary_port_count):
                blockers.append("patch_state_shape_mismatch")
            if canonical.shape != state.shape:
                blockers.append("canonical_patch_state_shape_mismatch")
            if signatures.shape != (patch_count,) or committed.shape != (patch_count,):
                blockers.append("patch_record_channel_shape_mismatch")
            if not (
                left.shape
                == right.shape
                == left_port.shape
                == right_port.shape
                == routed_left.shape
                == routed_right.shape
                == (edge_count,)
            ):
                blockers.append("patch_routing_channel_shape_mismatch")
            elif state.shape == (patch_count, boundary_port_count) and (
                np.any(left < 0)
                or np.any(right < 0)
                or np.any(left >= patch_count)
                or np.any(right >= patch_count)
                or np.any(left_port < 0)
                or np.any(right_port < 0)
                or np.any(left_port >= boundary_port_count)
                or np.any(right_port >= boundary_port_count)
                or not np.array_equal(state[left, left_port], routed_left)
                or not np.array_equal(state[right, right_port], routed_right)
            ):
                blockers.append("patch_routed_state_mismatch")
            if names.tolist() != [f"P{index}" for index in range(12)]:
                blockers.append("patch_port_names_mismatch")
            state_hash = hashlib.sha256(
                np.asarray(state, dtype="<i2").tobytes()
            ).hexdigest()
            if patch_state_report.get("patch_port_state_sha256") != state_hash:
                blockers.append("patch_state_content_hash_mismatch")
    except (KeyError, OSError, OverflowError, TypeError, ValueError):
        blockers.append("patch_state_artifact_unreadable")
    blockers = list(dict.fromkeys(blockers))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "path": str(path),
        "current_committed_record_count": current_committed_record_count,
    }


def _validate_repair_trace(
    root: Path,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, int]] = []
    try:
        with (root / "mismatch_trace.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for raw in csv.DictReader(handle):
                if raw.get("mismatch_definition") != GAUGE_COVARIANT_OVERLAP_SCHEMA:
                    blockers.append("repair_trace_mismatch_schema_invalid")
                rows.append(
                    {
                        "cycle": int(raw["cycle"]),
                        "phi_before": int(raw["phi_before"]),
                        "phi_after": int(raw["phi"]),
                        "chosen_edges": int(raw["chosen_edges"]),
                    }
                )
    except (KeyError, OSError, OverflowError, TypeError, ValueError):
        blockers.append("repair_trace_unreadable")
    if not rows:
        blockers.append("repair_trace_rows_missing")
    repair = report.get("repair_dynamics")
    repair = repair if isinstance(repair, Mapping) else {}
    repair_event_count = sum(row["chosen_edges"] for row in rows)
    strict_descent_count = sum(
        row["chosen_edges"]
        for row in rows
        if row["chosen_edges"] > 0 and row["phi_after"] < row["phi_before"]
    )
    non_descent_cycles = sum(
        1
        for row in rows
        if row["chosen_edges"] > 0 and row["phi_after"] >= row["phi_before"]
    )
    expected_rule_hash = stable_json_hash(
        {
            "mismatch": GAUGE_COVARIANT_OVERLAP_SCHEMA,
            "move": "bounded_covariant_port_pair_repair",
            "group": str((config.get("group") or {}).get("name", "S3")).upper(),
            "dynamics": config.get("dynamics", {}),
        }
    )
    expected_log_hash = stable_json_hash(rows)
    if repair.get("repair_event_count") != repair_event_count or repair_event_count <= 0:
        blockers.append("repair_event_count_mismatch")
    if repair.get("strict_descent_repair_event_count") != strict_descent_count:
        blockers.append("strict_descent_repair_event_count_mismatch")
    if repair.get("non_descent_repair_cycle_count") != non_descent_cycles:
        blockers.append("non_descent_repair_cycle_count_mismatch")
    if repair.get("repair_rule_hash") != expected_rule_hash:
        blockers.append("repair_rule_hash_mismatch")
    if repair.get("repair_event_log_hash") != expected_log_hash:
        blockers.append("repair_event_log_hash_mismatch")
    blockers = list(dict.fromkeys(blockers))
    return {
        "passed": not blockers,
        "blockers": blockers,
        "row_count": len(rows),
        "repair_event_count": repair_event_count,
        "strict_descent_repair_event_count": strict_descent_count,
        "non_descent_repair_cycle_count": non_descent_cycles,
        "repair_rule_hash_recomputed": expected_rule_hash,
        "repair_event_log_hash_recomputed": expected_log_hash,
    }


def _observer_jsonl_counts(path: Path) -> tuple[int, int, str | None]:
    if not path.is_file():
        return 0, 0, "observer_views_jsonl_missing"
    patch_count = 0
    cap_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return 0, 0, "observer_views_jsonl_row_invalid"
                if row.get("view_type") == "patch_observer":
                    patch_count += 1
                elif row.get("view_type") == "cap_observer":
                    cap_count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0, 0, "observer_views_jsonl_unreadable"
    return patch_count, cap_count, None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_causal_audit_row(
    row: Any,
    *,
    boundary_port_count: Any,
) -> bool:
    if not isinstance(row, Mapping):
        return False
    commit_cycle = _int_or_none(row.get("commit_cycle"))
    read_cycle = _int_or_none(row.get("read_cycle"))
    write_cycle = _int_or_none(row.get("write_cycle"))
    commit_event_index = _int_or_none(row.get("commit_event_index"))
    read_event_index = _int_or_none(row.get("read_event_index"))
    write_event_index = _int_or_none(row.get("write_event_index"))
    port_index = _int_or_none(row.get("bounded_port_index"))
    prior_signature = row.get("prior_record_signature")
    read_signature = row.get("readback_record_signature")
    counterfactual_signature = row.get("counterfactual_record_signature")
    record_port_value = row.get("prior_record_port_value")
    counterfactual_record_port_value = row.get(
        "counterfactual_record_port_value"
    )
    before = row.get("port_state_before")
    after = row.get("port_state_after")
    counterfactual_after = row.get("counterfactual_port_state_after")
    group_order = _int_or_none(row.get("group_order"))
    observer_id = _int_or_none(row.get("observer_id"))
    patch_id = _int_or_none(row.get("patch_id"))
    committed_snapshot = row.get("committed_record_snapshot")
    counterfactual_snapshot = row.get("counterfactual_record_snapshot")
    snapshot_pair = _validate_record_snapshot_pair(
        committed_snapshot,
        counterfactual_snapshot,
        commit_id=row.get("commit_id"),
        observer_id=observer_id,
        patch_id=patch_id,
        record_id=row.get("record_id"),
        event_id=row.get("event_id"),
        commit_cycle=commit_cycle,
        commit_event_index=commit_event_index,
        record_coordinate_port_index=row.get("record_coordinate_port_index"),
        boundary_port_count=boundary_port_count,
        group_order=group_order,
    )
    replayable_actions = bool(
        snapshot_pair["passed"]
        and
        _nonnegative_int(record_port_value)
        and _nonnegative_int(counterfactual_record_port_value)
        and group_order is not None
        and group_order > 1
        and int(record_port_value) < group_order
        and int(counterfactual_record_port_value) < group_order
        and record_port_value != counterfactual_record_port_value
        and after == record_port_value
        and counterfactual_after == counterfactual_record_port_value
    )
    return bool(
        row.get("schema") == "oph_record_feedback_causal_event_v2"
        and row.get("policy_schema") == CAUSAL_POLICY_SCHEMA
        and row.get("record_coordinate_source") == "committed_patch_port_state"
        and row.get("committed_record_snapshot_hash")
        == snapshot_pair["committed_record_snapshot_hash"]
        and _nonempty_string(row.get("event_id"))
        and _nonnegative_int(observer_id)
        and patch_id == observer_id
        and _nonempty_string(row.get("record_id"))
        and _int_or_none(prior_signature) is not None
        and read_signature == prior_signature
        and _int_or_none(counterfactual_signature) is not None
        and counterfactual_signature != prior_signature
        and commit_cycle is not None
        and read_cycle is not None
        and write_cycle is not None
        and commit_event_index is not None
        and read_event_index is not None
        and write_event_index is not None
        and commit_cycle >= 0
        and read_cycle >= 0
        and write_cycle >= 0
        and commit_event_index >= 0
        and read_event_index >= 0
        and write_event_index >= 0
        and (commit_cycle, commit_event_index) < (read_cycle, read_event_index)
        and (read_cycle, read_event_index) < (write_cycle, write_event_index)
        and port_index is not None
        and _positive_int(boundary_port_count)
        and 0 <= port_index < int(boundary_port_count)
        and row.get("record_coordinate_port_index") == port_index
        and record_port_value == snapshot_pair["committed_record_port_value"]
        and counterfactual_record_port_value
        == snapshot_pair["counterfactual_record_port_value"]
        and prior_signature == snapshot_pair["committed_record_signature"]
        and counterfactual_signature
        == snapshot_pair["counterfactual_record_signature"]
        and _nonnegative_int(before)
        and _nonnegative_int(after)
        and group_order is not None
        and group_order > 1
        and int(before) < group_order
        and int(after) < group_order
        and _nonnegative_int(counterfactual_after)
        and int(counterfactual_after) < group_order
        and after != counterfactual_after
        and row.get("counterfactual_holds_nonrecord_inputs_fixed") is True
        and replayable_actions
        and row.get("write_is_bounded_local_port") is True
        and row.get("records_causally_bound_to_prior_commits") is True
        and row.get("readback_changes_future_local_actions") is True
        and row.get("read_count") == 1
        and row.get("write_count") == 1
    )


def _validate_record_snapshot_pair(
    committed: Any,
    counterfactual: Any,
    *,
    commit_id: Any,
    observer_id: int | None,
    patch_id: int | None,
    record_id: Any,
    event_id: Any,
    commit_cycle: int | None,
    commit_event_index: int | None,
    record_coordinate_port_index: Any,
    boundary_port_count: Any,
    group_order: int | None,
) -> dict[str, Any]:
    """Recompute a committed record and its one-coordinate counterfactual."""

    failed = {
        "passed": False,
        "committed_record_port_value": None,
        "counterfactual_record_port_value": None,
        "committed_record_signature": None,
        "counterfactual_record_signature": None,
        "committed_record_snapshot_hash": None,
    }
    if not isinstance(committed, Mapping) or not isinstance(counterfactual, Mapping):
        return failed
    if committed.get("schema") != COMMITTED_RECORD_SNAPSHOT_SCHEMA or counterfactual.get(
        "schema"
    ) != COUNTERFACTUAL_RECORD_SNAPSHOT_SCHEMA:
        return failed
    identity = {
        "commit_id": commit_id,
        "observer_id": observer_id,
        "patch_id": patch_id,
        "record_id": record_id,
        "commit_cycle": commit_cycle,
        "commit_event_index": commit_event_index,
    }
    if any(committed.get(key) != value for key, value in identity.items()):
        return failed
    if any(counterfactual.get(key) != value for key, value in identity.items()):
        return failed
    if counterfactual.get("event_id") != event_id:
        return failed
    port_index = _int_or_none(record_coordinate_port_index)
    if (
        not _positive_int(boundary_port_count)
        or int(boundary_port_count) != 12
        or port_index is None
        or not 0 <= port_index < int(boundary_port_count)
        or group_order is None
        or group_order <= 1
    ):
        return failed
    committed_state = _canonical_port_state(
        committed.get("canonical_patch_port_state"),
        boundary_port_count=int(boundary_port_count),
        group_order=group_order,
    )
    counterfactual_state = _canonical_port_state(
        counterfactual.get("canonical_patch_port_state"),
        boundary_port_count=int(boundary_port_count),
        group_order=group_order,
    )
    committed_node_signature = _int_or_none(
        committed.get("routed_node_signature_input")
    )
    counterfactual_node_signature = _int_or_none(
        counterfactual.get("routed_node_signature_input")
    )
    if (
        not valid_committed_record_snapshot(
            committed,
            patch_count=None,
            boundary_port_count=boundary_port_count,
            group_order=group_order,
        )
        or
        committed_state is None
        or counterfactual_state is None
        or committed_node_signature is None
        or counterfactual_node_signature != committed_node_signature
    ):
        return failed
    differing_coordinates = [
        index
        for index, (original, alternate) in enumerate(
            zip(committed_state, counterfactual_state, strict=True)
        )
        if original != alternate
    ]
    if differing_coordinates != [port_index]:
        return failed
    try:
        committed_signature = record_signature_from_snapshot(
            committed_node_signature,
            committed_state,
        )
        counterfactual_signature = record_signature_from_snapshot(
            counterfactual_node_signature,
            counterfactual_state,
        )
    except (OverflowError, TypeError, ValueError):
        return failed
    if (
        committed.get("record_signature") != committed_signature
        or counterfactual.get("record_signature") != counterfactual_signature
        or committed_signature == counterfactual_signature
    ):
        return failed
    return {
        "passed": True,
        "committed_record_port_value": committed_state[port_index],
        "counterfactual_record_port_value": counterfactual_state[port_index],
        "committed_record_signature": committed_signature,
        "counterfactual_record_signature": counterfactual_signature,
        "committed_record_snapshot_hash": committed.get("snapshot_hash"),
    }


def valid_committed_record_snapshot(
    row: Any,
    *,
    patch_count: int | None,
    boundary_port_count: int | None,
    group_order: int | None,
) -> bool:
    """Validate a commit-time row captured at a live false-to-true transition."""

    if not isinstance(row, Mapping) or row.get("schema") != COMMITTED_RECORD_SNAPSHOT_SCHEMA:
        return False
    observer_id = _int_or_none(row.get("observer_id"))
    patch_id = _int_or_none(row.get("patch_id"))
    stable_count = _int_or_none(row.get("stable_count"))
    commit_threshold = _int_or_none(row.get("commit_threshold"))
    incident_mismatch = row.get("incident_mismatch")
    state = _canonical_port_state(
        row.get("canonical_patch_port_state"),
        boundary_port_count=boundary_port_count or 0,
        group_order=group_order or 0,
    )
    routed_signature = _int_or_none(row.get("routed_node_signature_input"))
    if (
        not _nonempty_string(row.get("commit_id"))
        or not _nonempty_string(row.get("record_id"))
        or not _nonnegative_int(observer_id)
        or patch_id != observer_id
        or (
            patch_count is not None
            and (observer_id is None or observer_id >= patch_count)
        )
        or not _nonnegative_int(row.get("commit_cycle"))
        or not _nonnegative_int(row.get("commit_event_index"))
        or stable_count is None
        or commit_threshold is None
        or commit_threshold <= 0
        or stable_count < commit_threshold
        or incident_mismatch != 0
        or not _nonnegative_int(incident_mismatch)
        or row.get("previous_committed") is not False
        or row.get("current_committed") is not True
        or row.get("newly_committed_transition") is not True
        or state is None
        or routed_signature is None
    ):
        return False
    try:
        recomputed_signature = record_signature_from_snapshot(
            routed_signature,
            state,
        )
    except (OverflowError, TypeError, ValueError):
        return False
    if row.get("record_signature") != recomputed_signature:
        return False
    content = dict(row)
    reported_hash = content.pop("snapshot_hash", None)
    return reported_hash == stable_json_hash(content)


def _canonical_port_state(
    value: Any,
    *,
    boundary_port_count: int,
    group_order: int,
) -> list[int] | None:
    if not isinstance(value, list) or len(value) != boundary_port_count:
        return None
    if not all(
        _nonnegative_int(item) and int(item) < group_order and int(item) <= 256
        for item in value
    ):
        return None
    return [int(item) for item in value]


def record_signature_from_snapshot(
    routed_node_signature_input: int,
    canonical_patch_port_state: list[int],
) -> int:
    """Recompute the exact simulator record signature for one local snapshot."""

    signature = echosahedral_patch_record_signature(
        np.asarray([routed_node_signature_input], dtype=np.int64),
        np.asarray([canonical_patch_port_state], dtype=np.int64),
    )
    return int(signature[0])


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _int_or_zero(value: Any) -> int:
    result = _int_or_none(value)
    return result if result is not None else 0


__all__ = [
    "CAUSAL_EVENT_LOG_SCHEMA",
    "CAUSAL_POLICY_SCHEMA",
    "MODE",
    "RUN_BINDING_SCHEMA",
    "SCHEMA_VERSION",
    "observer_id_set_hash",
    "observer_population_binding_hash",
    "validate_observer_population",
    "validate_run_self_reading_contract",
    "validate_self_reading_contract",
    "valid_causal_audit_row",
    "write_causal_event_artifact",
]
