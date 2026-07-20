"""Fail-closed aggregation of replay-verified physical H3/KMS cells.

One campaign-cell bundle contains the complete frozen matrix but may populate
only its selected cell.  This module is the independent family reducer: it
fresh-replays every supplied bundle, derives the cell outcome from P0--P7, and
then fills the frozen seed/rung matrix exactly once.  It never accepts a
caller-authored result mapping or a producer-authored promotion Boolean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from oph_fpe.bulk.physical_h3_kms_preflight import (
    CellStatus,
    GateStatus,
    physical_h3_kms_preflight_report,
)
from oph_fpe.bulk.physical_h3_kms_prerun import (
    REQUIRED_RUNGS,
    canonical_sha256,
    frozen_campaign_family_sha256,
)
from oph_fpe.bulk.physical_h3_kms_replay import (
    replay_physical_h3_kms_bundle,
)


AGGREGATE_SCHEMA = "oph.physical-h3-kms.campaign-family-aggregate.v1"
AGGREGATOR_IMPLEMENTATION = "oph_fpe/bulk/physical_h3_kms_aggregate.py"
REQUIRED_CELL_STAGE_IDS = (
    "P0_source_dynamics_repair_record_observer",
    "P1_nested_refinement_and_expectations",
    "P2_prime_geometric_cap_state",
    "P3_independent_geometric_parameter",
    "P4_native_bw01_bw08",
    "P5_frozen_candidate_interventions",
    "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage",
    "P7_semantic_event_e1_e4_and_frame_fiber_separation",
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VALID_CELL_STATUSES = {status.value for status in CellStatus}


class CampaignAggregationError(RuntimeError):
    """Raised internally when a family cannot be reduced safely."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise CampaignAggregationError("value is not finite canonical JSON") from exc


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _strict_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_exact_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignAggregationError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    return _load_exact_json_bytes(raw, label=label)


def _load_exact_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    """Parse the exact bytes whose digest was checked by the caller."""

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignAggregationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise CampaignAggregationError(f"{label} must be a JSON object")
    if raw != _canonical_bytes(value):
        raise CampaignAggregationError(f"{label} is not exact canonical JSON")
    return dict(value)


def _bound_artifact(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CampaignAggregationError("replay manifest artifacts are missing")
    descriptor = artifacts.get(name)
    if not isinstance(descriptor, Mapping):
        raise CampaignAggregationError(f"replay artifact is missing: {name}")
    relative_raw = descriptor.get("path")
    if not isinstance(relative_raw, str) or not relative_raw:
        raise CampaignAggregationError(f"replay artifact path is missing: {name}")
    relative = PurePosixPath(relative_raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CampaignAggregationError(f"unsafe replay artifact path: {name}")
    root = manifest_path.parent.resolve()
    path = manifest_path.parent.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise CampaignAggregationError(f"replay artifact is not a regular file: {name}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise CampaignAggregationError(f"replay artifact escapes bundle: {name}")
    raw = path.read_bytes()
    if descriptor.get("byte_count") != len(raw):
        raise CampaignAggregationError(f"replay artifact byte count differs: {name}")
    if descriptor.get("byte_sha256") != _sha256_bytes(raw):
        raise CampaignAggregationError(f"replay artifact hash differs: {name}")
    return _load_exact_json_bytes(raw, label=f"artifact {name}")


def _verified_failure_mode(
    scientific_failures: Sequence[str],
) -> tuple[str | None, str | None, list[str]]:
    """Derive a retirement mode only from verifier-produced predicates."""

    def predicate_name(value: str) -> str:
        prefix, separator, remainder = value.partition(":")
        return (
            remainder
            if separator and prefix in REQUIRED_CELL_STAGE_IDS
            else value
        )

    predicates = sorted(
        {
            predicate_name(str(value))
            for value in scientific_failures
            if isinstance(value, str) and value
        }
    )
    if not predicates:
        return None, None, []
    predicate_hash = canonical_sha256(predicates)
    mode = f"verified_predicate_set_v1:{predicate_hash.split(':', 1)[1]}"
    evidence_hash = canonical_sha256(
        {"derivation": "verified_predicate_set_sha256_v1", "predicates": predicates}
    )
    return mode, evidence_hash, predicates


def _cell_key(value: Mapping[str, Any]) -> tuple[int, int, str]:
    seed = value.get("seed")
    rung = value.get("rung")
    replicate_id = value.get("replicate_id")
    if (
        type(seed) is not int
        or type(rung) is not int
        or not isinstance(replicate_id, str)
        or not replicate_id
    ):
        raise CampaignAggregationError("campaign cell identity is malformed")
    return seed, rung, replicate_id


def _expected_cells(plan: Mapping[str, Any]) -> tuple[tuple[int, int, str], ...]:
    seeds = plan.get("seeds")
    rungs = plan.get("rungs")
    replicates = plan.get("replicate_ids")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(type(value) is not int for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise CampaignAggregationError("frozen family seed set is invalid")
    if rungs != list(REQUIRED_RUNGS):
        raise CampaignAggregationError("frozen family rungs are not exact")
    if (
        not isinstance(replicates, list)
        or not replicates
        or any(not isinstance(value, str) or not value for value in replicates)
        or len(set(replicates)) != len(replicates)
    ):
        raise CampaignAggregationError("frozen family replicate set is invalid")
    return tuple(
        (seed, rung, replicate_id)
        for seed in seeds
        for rung in rungs
        for replicate_id in replicates
    )


def _derive_cell_status(
    preflight: Mapping[str, Any],
) -> tuple[str, dict[str, str], list[str], list[str], bool]:
    admission = preflight.get("artifact_replay_admission")
    if not isinstance(admission, Mapping) or admission.get(
        "PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"
    ) is not True:
        raise CampaignAggregationError("cell artifact admission did not replay")
    stages = preflight.get("stages")
    if not isinstance(stages, Mapping):
        raise CampaignAggregationError("cell preflight stages are missing")
    stage_statuses: dict[str, str] = {}
    scientific_failures: list[str] = []
    not_evaluated_reasons: list[str] = []
    instrument_invalid = False
    incomplete = False
    valid_fail = False
    for stage_id in REQUIRED_CELL_STAGE_IDS:
        stage = stages.get(stage_id)
        if not isinstance(stage, Mapping):
            raise CampaignAggregationError(f"cell stage is missing: {stage_id}")
        status = str(stage.get("scientific_status") or "")
        gate_status = str(stage.get("gate_status") or "")
        if status not in _VALID_CELL_STATUSES:
            raise CampaignAggregationError(f"cell stage status is invalid: {stage_id}")
        stage_statuses[stage_id] = status
        evidence = stage.get("evidence")
        evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        failures = evidence.get("scientific_failures")
        if isinstance(failures, list):
            scientific_failures.extend(
                f"{stage_id}:{value}" for value in failures if isinstance(value, str)
            )
        reasons = evidence.get("not_evaluated_reasons")
        if isinstance(reasons, list):
            not_evaluated_reasons.extend(
                f"{stage_id}:{value}" for value in reasons if isinstance(value, str)
            )
        hard_blockers = stage.get("hard_blockers")
        if (
            status == CellStatus.INSTRUMENT_INVALID.value
            or gate_status == GateStatus.INSTRUMENT_INVALID.value
            or (isinstance(hard_blockers, list) and bool(hard_blockers))
        ):
            instrument_invalid = True
        elif (
            status in {CellStatus.NOT_EVALUATED.value, CellStatus.INCOMPLETE.value}
            or gate_status in {GateStatus.NOT_EVALUATED.value, GateStatus.BLOCKED.value}
        ):
            incomplete = True
        elif status == CellStatus.VALID_FAIL.value:
            valid_fail = True
        elif status != CellStatus.VALID_PASS.value:
            incomplete = True

    # P8's producer-authored aggregate outcome is deliberately excluded from
    # the cell physics result.  Its instrument portion is still mandatory:
    # family binding, no-retune, one protocol, and the frozen retirement rule
    # cannot be ignored by a P0--P7 reducer.
    p8 = stages.get("P8_frozen_multiseed_four_rung_campaign")
    p8_evidence = (
        dict(p8.get("evidence"))
        if isinstance(p8, Mapping) and isinstance(p8.get("evidence"), Mapping)
        else {}
    )
    p8_hard_blockers = (
        list(p8.get("hard_blockers", [])) if isinstance(p8, Mapping) else []
    )
    top_hard_blockers = list(preflight.get("hard_blockers", []))
    p8_instrument_contract_valid = bool(
        isinstance(p8, Mapping)
        and p8.get("scientific_status") != CellStatus.INSTRUMENT_INVALID.value
        and p8.get("gate_status") != GateStatus.INSTRUMENT_INVALID.value
        and not p8_hard_blockers
        and not top_hard_blockers
        and not list(p8_evidence.get("diagnostic_hard_blockers", []))
        and not list(p8_evidence.get("diagnostic_blockers", []))
        and p8_evidence.get("single_protocol_hash") is True
        and p8_evidence.get("decisive_rungs") == [16_384, 65_536, 262_144]
        and preflight.get("campaign_status")
        != CellStatus.INSTRUMENT_INVALID.value
    )
    if not p8_instrument_contract_valid:
        instrument_invalid = True
    if instrument_invalid:
        cell_status = CellStatus.INSTRUMENT_INVALID.value
    elif incomplete:
        cell_status = CellStatus.INCOMPLETE.value
    elif valid_fail:
        cell_status = CellStatus.VALID_FAIL.value
    else:
        cell_status = CellStatus.VALID_PASS.value
    if cell_status == CellStatus.VALID_FAIL.value and not scientific_failures:
        raise CampaignAggregationError(
            "valid failed cell has no verifier-derived scientific predicate"
        )
    return (
        cell_status,
        stage_statuses,
        list(dict.fromkeys(scientific_failures)),
        list(dict.fromkeys(not_evaluated_reasons)),
        p8_instrument_contract_valid,
    )


def _load_verified_cell(run_directory: Path) -> dict[str, Any]:
    if run_directory.is_symlink() or not run_directory.is_dir():
        raise CampaignAggregationError("run directory must be a regular directory")
    manifest_path = run_directory / "replay_bundle" / "replay_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CampaignAggregationError("replay manifest is not a regular file")
    manifest_snapshot = manifest_path.read_bytes()
    replay = replay_physical_h3_kms_bundle(manifest_path)
    if replay.get("PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT") is not True:
        blockers = replay.get("blockers")
        raise CampaignAggregationError(f"cell replay failed: {blockers!r}")
    if replay.get("manifest_byte_sha256") != _sha256_bytes(manifest_snapshot):
        raise CampaignAggregationError("replay manifest snapshot hash differs")
    preflight = physical_h3_kms_preflight_report(run_directory)
    if manifest_path.read_bytes() != manifest_snapshot:
        raise CampaignAggregationError(
            "replay manifest changed during family aggregation"
        )
    admission = preflight.get("artifact_replay_admission")
    if not isinstance(admission, Mapping) or admission.get(
        "PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"
    ) is not True:
        raise CampaignAggregationError("cell preflight artifact admission failed")

    manifest = _load_exact_json_bytes(
        manifest_snapshot, label="replay manifest"
    )
    preregistration = _bound_artifact(manifest_path, manifest, "preregistration")
    postrun = _bound_artifact(manifest_path, manifest, "postrun_report")
    runtime = _bound_artifact(manifest_path, manifest, "numerical_runtime")
    plan = preregistration.get("plan")
    campaign = postrun.get("campaign")
    if not isinstance(plan, Mapping) or not isinstance(campaign, Mapping):
        raise CampaignAggregationError("bound plan or campaign manifest is missing")
    plan = dict(plan)
    campaign = dict(campaign)
    expected_cells = _expected_cells(plan)
    current_key = _cell_key(
        dict(plan.get("current_cell"))
        if isinstance(plan.get("current_cell"), Mapping)
        else {}
    )
    if current_key not in expected_cells:
        raise CampaignAggregationError("selected cell is outside the frozen matrix")
    family_plan_hash = frozen_campaign_family_sha256(plan)
    if campaign.get("frozen_campaign_family_sha256") != family_plan_hash:
        raise CampaignAggregationError("cell family-plan commitment differs")
    family_contract = campaign.get("family_contract")
    if not isinstance(family_contract, Mapping):
        raise CampaignAggregationError("cell family contract is missing")
    family_hash = canonical_sha256(family_contract)
    if campaign.get("campaign_family_hash") != family_hash:
        raise CampaignAggregationError("cell family-contract commitment differs")
    runtime_hash = runtime.get("runtime_sha256")
    if not _strict_sha256(runtime_hash):
        raise CampaignAggregationError("cell numerical-runtime hash is missing")

    rows = campaign.get("run_matrix")
    if not isinstance(rows, list):
        raise CampaignAggregationError("cell campaign run matrix is missing")
    current_rows = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("preflight") == "PASS"
        and _cell_key(row) == current_key
    ]
    if len(current_rows) != 1:
        raise CampaignAggregationError("cell campaign did not populate one selected row")
    current_row = current_rows[0]
    (
        derived_status,
        stage_statuses,
        scientific_failures,
        not_evaluated,
        p8_instrument_contract_valid,
    ) = _derive_cell_status(preflight)
    if current_row.get("status") != derived_status:
        raise CampaignAggregationError(
            "replayed postrun cell status differs from P0--P7 derivation"
        )
    protocol_hash = current_row.get("protocol_hash")
    if not _strict_sha256(protocol_hash):
        raise CampaignAggregationError("cell protocol hash is missing")
    result_hash = current_row.get("result_artifact_hash")
    if not _strict_sha256(result_hash):
        raise CampaignAggregationError("cell result artifact hash is missing")
    failure_mode = current_row.get("failure_mode")
    failure_evidence_hash = current_row.get("failure_evidence_hash")
    if derived_status == CellStatus.VALID_FAIL.value:
        expected_mode, expected_producer_evidence, predicates = (
            _verified_failure_mode(scientific_failures)
        )
        if failure_mode != expected_mode:
            raise CampaignAggregationError(
                "producer failure mode differs from verifier-derived predicate set"
            )
        if failure_evidence_hash != expected_producer_evidence:
            raise CampaignAggregationError(
                "producer failure evidence differs from verifier-derived predicates"
            )
    elif failure_mode is not None or failure_evidence_hash is not None:
        raise CampaignAggregationError("nonfailed cell carries failure evidence")
    else:
        predicates = []

    aggregate_failure_evidence_hash = (
        canonical_sha256(
            {
                "derivation": "aggregate_bound_failure_evidence_v1",
                "cell": list(current_key),
                "failure_mode": failure_mode,
                "failure_predicates": predicates,
                "stage_statuses": stage_statuses,
                "source_capture_hash": postrun.get("source_capture_hash"),
                "result_artifact_hash": result_hash,
                "manifest_byte_sha256": replay.get("manifest_byte_sha256"),
                "preflight_sha256": canonical_sha256(preflight),
                "protocol_hash": protocol_hash,
            }
        )
        if derived_status == CellStatus.VALID_FAIL.value
        else None
    )

    return {
        "cell": current_key,
        "expected_cells": expected_cells,
        "seeds": tuple(plan["seeds"]),
        "rungs": tuple(plan["rungs"]),
        "replicate_ids": tuple(plan["replicate_ids"]),
        "campaign_id": campaign.get("campaign_id"),
        "instrument_version": campaign.get("instrument_version"),
        "instrument_commit": family_contract.get("instrument_commit"),
        "campaign_family_hash": family_hash,
        "frozen_campaign_family_sha256": family_plan_hash,
        "family_contract": dict(family_contract),
        "protocol_hash": protocol_hash,
        "runtime_sha256": runtime_hash,
        "cell_status": derived_status,
        "stage_statuses": stage_statuses,
        "p8_instrument_contract_valid": p8_instrument_contract_valid,
        "scientific_failures": scientific_failures,
        "not_evaluated_reasons": not_evaluated,
        "failure_mode": failure_mode,
        "producer_failure_evidence_hash": failure_evidence_hash,
        "failure_evidence_hash": aggregate_failure_evidence_hash,
        "result_artifact_hash": result_hash,
        "source_capture_hash": postrun.get("source_capture_hash"),
        "manifest_byte_sha256": replay.get("manifest_byte_sha256"),
        "preflight_sha256": canonical_sha256(preflight),
        "run_directory": run_directory.resolve().as_posix(),
    }


def _implementation_hash() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _aggregate_verified_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cells:
        raise CampaignAggregationError("at least one replay-verified cell is required")
    first = dict(cells[0])
    family_fields = (
        "campaign_id",
        "instrument_version",
        "instrument_commit",
        "campaign_family_hash",
        "frozen_campaign_family_sha256",
        "family_contract",
        "protocol_hash",
        "runtime_sha256",
        "expected_cells",
        "seeds",
        "rungs",
        "replicate_ids",
    )
    for index, raw in enumerate(cells):
        cell = dict(raw)
        for field in family_fields:
            if cell.get(field) != first.get(field):
                raise CampaignAggregationError(
                    f"cell {index} differs from frozen family field: {field}"
                )
    retirement_rule = (
        dict(first["family_contract"].get("retirement_rule"))
        if isinstance(first.get("family_contract"), Mapping)
        and isinstance(first["family_contract"].get("retirement_rule"), Mapping)
        else {}
    )
    if retirement_rule != {
        "decisive_rungs": [16_384, 65_536, 262_144],
        "same_predeclared_failure_mode_required": True,
        "all_cells_powered_and_complete_required": True,
        "frozen_before_first_run": True,
        "failure_mode_derivation": "verified_predicate_set_sha256_v1",
    }:
        raise CampaignAggregationError("frozen retirement rule is malformed")
    expected = tuple(first["expected_cells"])
    observed_rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in cells:
        cell = dict(raw)
        key = tuple(cell["cell"])
        if key not in expected:
            raise CampaignAggregationError("observed cell is outside frozen family")
        if key in observed_rows:
            raise CampaignAggregationError(f"duplicate campaign cell: {key!r}")
        if cell.get("p8_instrument_contract_valid") is not True:
            raise CampaignAggregationError(
                f"P8 instrument contract is invalid for cell: {key!r}"
            )
        if cell.get("cell_status") == CellStatus.VALID_FAIL.value:
            expected_mode, _, predicates = _verified_failure_mode(
                cell.get("scientific_failures", [])
            )
            if cell.get("failure_mode") != expected_mode:
                raise CampaignAggregationError(
                    f"failure mode is not verifier-derived for cell: {key!r}"
                )
            expected_evidence = canonical_sha256(
                {
                    "derivation": "aggregate_bound_failure_evidence_v1",
                    "cell": list(key),
                    "failure_mode": expected_mode,
                    "failure_predicates": predicates,
                    "stage_statuses": cell.get("stage_statuses"),
                    "source_capture_hash": cell.get("source_capture_hash"),
                    "result_artifact_hash": cell.get("result_artifact_hash"),
                    "manifest_byte_sha256": cell.get("manifest_byte_sha256"),
                    "preflight_sha256": cell.get("preflight_sha256"),
                    "protocol_hash": cell.get("protocol_hash"),
                }
            )
            if cell.get("failure_evidence_hash") != expected_evidence:
                raise CampaignAggregationError(
                    f"failure evidence is not aggregate-bound for cell: {key!r}"
                )
        observed_rows[key] = cell

    missing = [key for key in expected if key not in observed_rows]
    statuses = [str(row["cell_status"]) for row in observed_rows.values()]
    if any(status == CellStatus.INSTRUMENT_INVALID.value for status in statuses):
        campaign_status = CellStatus.INSTRUMENT_INVALID.value
    elif missing or any(
        status in {CellStatus.NOT_EVALUATED.value, CellStatus.INCOMPLETE.value}
        for status in statuses
    ):
        campaign_status = CellStatus.INCOMPLETE.value
    elif statuses and all(status == CellStatus.VALID_PASS.value for status in statuses):
        campaign_status = CellStatus.VALID_PASS.value
    elif statuses and all(
        status in {CellStatus.VALID_PASS.value, CellStatus.VALID_FAIL.value}
        for status in statuses
    ):
        campaign_status = CellStatus.VALID_FAIL.value
    else:
        campaign_status = CellStatus.INCOMPLETE.value

    decisive_rungs = tuple(retirement_rule["decisive_rungs"])
    decisive_keys = [key for key in expected if key[1] in decisive_rungs]
    decisive_rows = [observed_rows[key] for key in decisive_keys if key in observed_rows]
    failure_modes = {
        str(row["failure_mode"])
        for row in decisive_rows
        if isinstance(row.get("failure_mode"), str) and row.get("failure_mode")
    }
    stable_failure = bool(
        campaign_status == CellStatus.VALID_FAIL.value
        and len(decisive_rows) == len(decisive_keys)
        and all(
            row.get("cell_status") == CellStatus.VALID_FAIL.value
            for row in decisive_rows
        )
        and all(_strict_sha256(row.get("failure_evidence_hash")) for row in decisive_rows)
        and len(failure_modes) == 1
    )

    evaluated_statuses = {CellStatus.VALID_PASS.value, CellStatus.VALID_FAIL.value}
    ready_for_rung: dict[str, bool] = {}
    rungs = tuple(first["rungs"])
    for rung_index, rung in enumerate(rungs):
        prerequisite_rungs = set(rungs[:rung_index])
        prerequisite_keys = [key for key in expected if key[1] in prerequisite_rungs]
        ready_for_rung[str(rung)] = bool(
            all(
                key in observed_rows
                and observed_rows[key].get("cell_status") in evaluated_statuses
                for key in prerequisite_keys
            )
        )

    aggregate_rows = []
    for key in expected:
        row = observed_rows.get(key)
        aggregate_rows.append(
            {
                "seed": key[0],
                "rung": key[1],
                "replicate_id": key[2],
                "status": (
                    row.get("cell_status")
                    if row is not None
                    else CellStatus.NOT_EVALUATED.value
                ),
                "stage_statuses": row.get("stage_statuses", {}) if row else {},
                "p8_instrument_contract_valid": (
                    row.get("p8_instrument_contract_valid") if row else None
                ),
                "scientific_failures": row.get("scientific_failures", []) if row else [],
                "not_evaluated_reasons": (
                    row.get("not_evaluated_reasons", [])
                    if row
                    else ["campaign_cell_bundle_missing"]
                ),
                "failure_mode": row.get("failure_mode") if row else None,
                "failure_evidence_hash": (
                    row.get("failure_evidence_hash") if row else None
                ),
                "result_artifact_hash": row.get("result_artifact_hash") if row else None,
                "source_capture_hash": row.get("source_capture_hash") if row else None,
                "manifest_byte_sha256": (
                    row.get("manifest_byte_sha256") if row else None
                ),
                "preflight_sha256": row.get("preflight_sha256") if row else None,
            }
        )

    physical_stage_statuses = [
        status
        for row in observed_rows.values()
        for status in row.get("stage_statuses", {}).values()
    ]
    p8_status = (
        CellStatus.INSTRUMENT_INVALID.value
        if campaign_status == CellStatus.INSTRUMENT_INVALID.value
        else CellStatus.NOT_EVALUATED.value
        if campaign_status == CellStatus.INCOMPLETE.value
        else campaign_status
    )
    return {
        "schema": AGGREGATE_SCHEMA,
        "aggregator_implementation": AGGREGATOR_IMPLEMENTATION,
        "aggregator_code_sha256": _implementation_hash(),
        "aggregation_instrument_status": "VALID_PASS",
        "campaign_id": first["campaign_id"],
        "instrument_version": first["instrument_version"],
        "instrument_commit": first["instrument_commit"],
        "campaign_family_hash": first["campaign_family_hash"],
        "frozen_campaign_family_sha256": first["frozen_campaign_family_sha256"],
        "protocol_hash": first["protocol_hash"],
        "numerical_runtime_sha256": first["runtime_sha256"],
        "family_contract_sha256": canonical_sha256(first["family_contract"]),
        "seeds": list(first["seeds"]),
        "rungs": list(rungs),
        "replicate_ids": list(first["replicate_ids"]),
        "expected_cell_count": len(expected),
        "observed_cell_count": len(observed_rows),
        "missing_cell_count": len(missing),
        "missing_cells": [
            {"seed": key[0], "rung": key[1], "replicate_id": key[2]}
            for key in missing
        ],
        "campaign_status": campaign_status,
        "P8_measurement_status": p8_status,
        "physical_promotion_allowed": campaign_status == CellStatus.VALID_PASS.value,
        "stable_failure_rule_satisfied": stable_failure,
        "branch_retirement_authorized": stable_failure,
        "stable_failure_modes": sorted(failure_modes),
        "ready_for_rung": ready_for_rung,
        "ready_for_64k": ready_for_rung.get("65536", False),
        "larger_run_ready": ready_for_rung.get("65536", False),
        "cell_status_counts": {
            status.value: sum(row["status"] == status.value for row in aggregate_rows)
            for status in CellStatus
        },
        "observed_physical_stage_status_counts": {
            status.value: physical_stage_statuses.count(status.value)
            for status in CellStatus
        },
        "run_matrix": aggregate_rows,
        "claim_boundary": (
            "This aggregate is derived only from fresh-replayed cell bundles. Missing or "
            "unevaluated cells cannot promote or retire the branch. Branch retirement "
            "requires every decisive-rung cell to be a replay-valid scientific failure "
            "with one predeclared failure mode."
        ),
    }


def _invalid_report(blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": AGGREGATE_SCHEMA,
        "aggregator_implementation": AGGREGATOR_IMPLEMENTATION,
        "aggregator_code_sha256": _implementation_hash(),
        "aggregation_instrument_status": "INSTRUMENT_INVALID",
        "campaign_status": CellStatus.INSTRUMENT_INVALID.value,
        "P8_measurement_status": CellStatus.INSTRUMENT_INVALID.value,
        "physical_promotion_allowed": False,
        "stable_failure_rule_satisfied": False,
        "branch_retirement_authorized": False,
        "ready_for_64k": False,
        "larger_run_ready": False,
        "blockers": list(dict.fromkeys(str(value) for value in blockers)),
        "claim_boundary": (
            "Malformed, mixed-family, duplicated, or unreplayable cells are instrument "
            "invalidity, never physical failure evidence."
        ),
    }


def aggregate_physical_h3_kms_family(
    run_directories: Sequence[str | Path],
) -> dict[str, Any]:
    """Fresh-replay run directories and derive one frozen family aggregate."""

    try:
        directories = [Path(value) for value in run_directories]
        if not directories:
            raise CampaignAggregationError("no campaign cell directories supplied")
        cells = [_load_verified_cell(path) for path in directories]
        return _aggregate_verified_cells(cells)
    except (CampaignAggregationError, OSError, TypeError, ValueError) as exc:
        return _invalid_report([f"{type(exc).__name__}:{exc}"])


def write_physical_h3_kms_family_aggregate(
    run_directories: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an append-never aggregate report."""

    report = aggregate_physical_h3_kms_family(run_directories)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(report)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CampaignAggregationError(f"refusing to overwrite aggregate: {path}") from exc
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate replay-verified physical H3/KMS campaign cells."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("run_directories", nargs="+", type=Path)
    args = parser.parse_args(argv)
    report = write_physical_h3_kms_family_aggregate(
        args.run_directories,
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["aggregation_instrument_status"] == "VALID_PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "AGGREGATE_SCHEMA",
    "CampaignAggregationError",
    "aggregate_physical_h3_kms_family",
    "write_physical_h3_kms_family_aggregate",
]
