"""Content-addressed disk replay for the physical source campaign.

This module verifies bytes, code identities, and exact deterministic replay.
Its receipts are software-instrument integrity receipts only.  In particular,
they never promote a clock, geometry, event-manifold, or physics claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from oph_fpe.bulk.physical_h3_kms_postrun import (
    PREREGISTRATION_ENVELOPE_SCHEMA,
    _compute_postrun_reports_from_verified_source,
)
from oph_fpe.bulk.physical_h3_kms_prerun import (
    ALLOWED_CHECKER_IDS,
    ALLOWED_PRODUCER_IDS,
    REGISTERED_HISTORICAL_16K_SOURCE_SEED,
    REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256,
    canonical_sha256,
    physical_h3_kms_prerun_report,
    physical_h3_kms_source_inputs,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import (
    verify_physical_source_capture,
)
from oph_fpe.bulk.physical_h3_kms_runtime import (
    NumericalRuntimeError,
    require_frozen_numerical_runtime,
)


REPLAY_MANIFEST_SCHEMA = "oph.physical-h3-kms.disk-replay-manifest.v2"
REPLAY_REPORT_SCHEMA = "oph.physical-h3-kms.disk-replay-report.v2"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPLAY_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_replay.py"
_CAMPAIGN_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_campaign.py"
_SOURCE_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_source_capture.py"
_PRERUN_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_prerun.py"
_POSTRUN_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_postrun.py"
_RUNTIME_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_runtime.py"
_AGGREGATE_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_aggregate.py"
_PREFLIGHT_IMPLEMENTATION_PATH = "oph_fpe/bulk/physical_h3_kms_preflight.py"
_BW_PREFLIGHT_IMPLEMENTATION_PATH = "oph_fpe/bulk/bw_native_preflight.py"
_BW_CERTIFICATE_IMPLEMENTATION_PATH = "oph_fpe/bulk/bw_certificate_308.py"
_ECHOSAHEDRAL_DYNAMICS_IMPLEMENTATION_PATH = (
    "oph_fpe/core/echosahedral_dynamics.py"
)
_ECHOSAHEDRAL_FEDERATION_IMPLEMENTATION_PATH = (
    "oph_fpe/core/echosahedral_federation.py"
)
_ICOSAHEDRAL_IMPLEMENTATION_PATH = "oph_fpe/core/icosahedral.py"
_SCREEN_PORTS_IMPLEMENTATION_PATH = "oph_fpe/core/screen_ports.py"
_COVARIANT_OVERLAP_IMPLEMENTATION_PATH = "oph_fpe/gauge/covariant_overlap.py"
_FINITE_GROUPS_IMPLEMENTATION_PATH = "oph_fpe/finite_groups.py"
_CLAIMS_IMPLEMENTATION_PATH = "oph_fpe/claims.py"
_BEHAVIOR_BEARING_DEPENDENCY_PATHS = frozenset(
    {
        _PREFLIGHT_IMPLEMENTATION_PATH,
        _RUNTIME_IMPLEMENTATION_PATH,
        _AGGREGATE_IMPLEMENTATION_PATH,
        _BW_PREFLIGHT_IMPLEMENTATION_PATH,
        _BW_CERTIFICATE_IMPLEMENTATION_PATH,
        _ECHOSAHEDRAL_DYNAMICS_IMPLEMENTATION_PATH,
        _ECHOSAHEDRAL_FEDERATION_IMPLEMENTATION_PATH,
        _ICOSAHEDRAL_IMPLEMENTATION_PATH,
        _SCREEN_PORTS_IMPLEMENTATION_PATH,
        _COVARIANT_OVERLAP_IMPLEMENTATION_PATH,
        _FINITE_GROUPS_IMPLEMENTATION_PATH,
        _CLAIMS_IMPLEMENTATION_PATH,
    }
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_NAMES = (
    "preregistration",
    "preregistration_report",
    "freeze_receipt",
    "numerical_runtime",
    "historical_campaign_receipt",
    "source_capture",
    "postrun_report",
    "clock_calibration",
    "geometry_calibration",
    "curvature_calibration",
)
_FREEZE_RECEIPT_SCHEMA = "oph.physical-h3-kms.pre-source-freeze-receipt.v2"
_FREEZE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "frozen_at_utc",
        "plan_sha256",
        "preregistration_sha256",
        "instrument_commit_sha256",
        "artifact_descriptors",
        "source_capture_allowed",
        "scientific_status",
        "retune_after_freeze",
        "archived_16k_failure_preserved",
        "archived_outcomes_used_for_threshold_selection",
        "demo_or_nudge_controls_accepted",
        "claim_boundary",
    }
)
_FROZEN_ARTIFACT_NAMES = (
    "preregistration",
    "preregistration_report",
    "postrun_preregistration",
    "role_code_bindings",
    "instrument_code_bundle",
    "numerical_runtime",
    "clock_calibration",
    "geometry_calibration",
    "curvature_calibration",
    "historical_campaign_receipt",
)
_CALIBRATION_NAMES = (
    "clock_calibration",
    "geometry_calibration",
    "curvature_calibration",
)
_CALIBRATION_KEYS = frozenset(
    {
        "schema",
        "calibration_id",
        "calibration_seeds",
        "independent_of_campaign_source_seeds",
        "frozen_before_source_capture",
        "protocol",
        "thresholds",
    }
)
_CALIBRATION_SCHEMA = {
    "clock_calibration": "oph.physical-h3-kms.clock-calibration.v1",
    "geometry_calibration": "oph.physical-h3-kms.geometry-calibration.v1",
    "curvature_calibration": "oph.physical-h3-kms.curvature-calibration.v1",
}
_CALIBRATION_PLAN_HASH_FIELD = {
    "clock_calibration": "clock_calibration_sha256",
    "geometry_calibration": "geometry_calibration_sha256",
    "curvature_calibration": "curvature_calibration_sha256",
}
_CALIBRATION_THRESHOLD_FIELDS = {
    "clock_calibration": frozenset(
        {"clock_absolute_residual_max", "clock_win_margin_min"}
    ),
    "geometry_calibration": frozenset({"geometry_win_margin_min"}),
    "curvature_calibration": frozenset({"curvature_minimum_power"}),
}
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "artifacts",
        "role_code_bindings",
        "instrument_code_bundle",
        "replay_implementation",
        "claim_boundary",
    }
)
_DESCRIPTOR_KEYS = frozenset({"path", "byte_sha256", "byte_count"})
_COMPATIBILITY_RECEIPTS = (
    "REPLAY_MANIFEST_VERIFICATION_RECEIPT",
    "PRE_SOURCE_FREEZE_REPLAY_RECEIPT",
    "HISTORICAL_16K_ARCHIVE_BYTE_REPLAY_RECEIPT",
    "SOURCE_CAPTURE_REPLAY_RECEIPT",
    "PER_CELL_CONTROL_ARTIFACTS_REPLAY_RECEIPT",
    "PER_CELL_SCIENTIFIC_PREDICATES_RECOMPUTED_RECEIPT",
    "SINGLE_BUNDLE_COMMITMENT_RECEIPT",
    "NUMERICAL_RUNTIME_REPLAY_RECEIPT",
)

_ROLE_IMPLEMENTATION_PATHS = {
    "producer_registry": {
        "source_federation": _SOURCE_IMPLEMENTATION_PATH,
        "support_regulator": _SOURCE_IMPLEMENTATION_PATH,
        "source_dynamics": _SOURCE_IMPLEMENTATION_PATH,
        "observer_capture": _SOURCE_IMPLEMENTATION_PATH,
    },
    "checker_registry": {
        "source_federation": _SOURCE_IMPLEMENTATION_PATH,
        "support_regulator": _SOURCE_IMPLEMENTATION_PATH,
        "source_firewall": _PRERUN_IMPLEMENTATION_PATH,
        "repair_replay": _SOURCE_IMPLEMENTATION_PATH,
        "observer_replay": _SOURCE_IMPLEMENTATION_PATH,
        "cell_postflight": _POSTRUN_IMPLEMENTATION_PATH,
    },
}


class ReplayBundleError(ValueError):
    """Raised by bundle construction for a non-replayable input."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayBundleError("value is not canonical finite JSON") from exc


def _artifact_json_bytes(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayBundleError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> Any:
    raise ReplayBundleError(f"nonfinite JSON constant: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ReplayBundleError("nonfinite JSON number")
    return value


def _strict_json_loads(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except UnicodeDecodeError as exc:
        raise ReplayBundleError(f"{label} is not strict UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ReplayBundleError(f"{label} is not exact JSON") from exc


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayBundleError(f"{label} must be a JSON object")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(expected):
        raise ReplayBundleError(f"{label} field set mismatch")


def _safe_bundle_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ReplayBundleError("artifact path must be a safe POSIX relative path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ReplayBundleError("artifact path must be a safe POSIX relative path")
    root_resolved = root.resolve(strict=True)
    cursor = root_resolved
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReplayBundleError("artifact path must not traverse a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as exc:
        raise ReplayBundleError("artifact path escapes or is missing from bundle") from exc
    if not resolved.is_file():
        raise ReplayBundleError("artifact path is not a regular file")
    return resolved


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return "sha256:" + digest.hexdigest(), byte_count


def _code_file_descriptor(relative: str) -> dict[str, Any]:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReplayBundleError("internal role implementation path is unsafe")
    path = (_REPO_ROOT / pure).resolve(strict=True)
    path.relative_to(_REPO_ROOT.resolve(strict=True))
    if path.is_symlink() or not path.is_file():
        raise ReplayBundleError("registered implementation is not a regular file")
    digest, byte_count = _hash_file(path)
    return {
        "implementation_path": relative,
        "byte_sha256": digest,
        "byte_count": byte_count,
    }


def registered_role_code_bindings() -> dict[str, dict[str, dict[str, Any]]]:
    """Return actual byte hashes for every allowlisted registered role."""

    result: dict[str, dict[str, dict[str, Any]]] = {
        "producer_registry": {},
        "checker_registry": {},
    }
    for registry_name, role_paths in _ROLE_IMPLEMENTATION_PATHS.items():
        allowlist = (
            ALLOWED_PRODUCER_IDS
            if registry_name == "producer_registry"
            else ALLOWED_CHECKER_IDS
        )
        id_key = "producer_id" if registry_name == "producer_registry" else "checker_id"
        for role, relative in role_paths.items():
            allowed_ids = sorted(allowlist[role])
            if len(allowed_ids) != 1:
                raise ReplayBundleError(f"registered role ID is ambiguous: {role}")
            result[registry_name][role] = {
                id_key: allowed_ids[0],
                **_code_file_descriptor(relative),
            }
    return result


def registered_role_code_registries() -> dict[str, dict[str, dict[str, str]]]:
    """Return exact plan-compatible registries bound to current code bytes."""

    bindings = registered_role_code_bindings()
    producers = {
        role: {
            "producer_id": row["producer_id"],
            "source_code_sha256": row["byte_sha256"],
        }
        for role, row in bindings["producer_registry"].items()
    }
    checkers = {
        role: {
            "checker_id": row["checker_id"],
            "checker_code_sha256": row["byte_sha256"],
        }
        for role, row in bindings["checker_registry"].items()
    }
    return {"producer_registry": producers, "checker_registry": checkers}


def registered_instrument_code_bundle() -> dict[str, Any]:
    """Commit the runner, every role implementation, and receipt verifier."""

    relative_paths = sorted(
        {
            _CAMPAIGN_IMPLEMENTATION_PATH,
            _REPLAY_IMPLEMENTATION_PATH,
            *_BEHAVIOR_BEARING_DEPENDENCY_PATHS,
            *(
                path
                for registry in _ROLE_IMPLEMENTATION_PATHS.values()
                for path in registry.values()
            ),
        }
    )
    material = {
        "schema": "oph.physical-h3-kms.instrument-code-bundle.v1",
        "files": [_code_file_descriptor(path) for path in relative_paths],
    }
    return {
        **material,
        "instrument_commit_sha256": canonical_sha256(material),
    }


def _artifact_descriptor(relative: str, data: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "byte_sha256": _sha256_bytes(data),
        "byte_count": len(data),
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise ReplayBundleError(f"refusing to overwrite replay file: {path.name}") from exc


def _postrun_envelope(
    preregistration: Mapping[str, Any], preregistration_report: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": PREREGISTRATION_ENVELOPE_SCHEMA,
        "preregistration": dict(preregistration),
        "preregistration_report": dict(preregistration_report),
        "preregistration_sha256": canonical_sha256(preregistration),
    }


def _verified_historical_campaign_receipt(data: bytes) -> dict[str, Any]:
    """Verify the exact retained legacy receipt and its registered 16k cell."""

    if _sha256_bytes(data) != REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256:
        raise ReplayBundleError("historical campaign receipt byte hash mismatch")
    receipt = _require_mapping(
        _strict_json_loads(data, label="historical_campaign_receipt"),
        "historical_campaign_receipt",
    )
    cells = receipt.get("cells")
    if not isinstance(cells, list):
        raise ReplayBundleError("historical campaign receipt has no cell census")
    selected = [
        row
        for row in cells
        if isinstance(row, Mapping)
        and row.get("source_seed") == REGISTERED_HISTORICAL_16K_SOURCE_SEED
        and row.get("patch_count") == 16_384
    ]
    stable_failure = receipt.get("stable_failure")
    stable_established = (
        stable_failure.get("established")
        if isinstance(stable_failure, Mapping)
        else None
    )
    if not (
        receipt.get("schema") == "oph_physical_h3_kms_campaign_receipt_v2"
        and receipt.get("campaign_sha256")
        == REGISTERED_HISTORICAL_CAMPAIGN_SHA256
        and receipt.get("PHYSICAL_H3_KMS_EMERGENCE_RECEIPT") is False
        and receipt.get("BRANCH_RETIREMENT_RECEIPT") is False
        and receipt.get("retuning_permitted") is False
        and stable_established is False
        and len(selected) == 1
        and selected[0].get("artifact_present") is True
        and selected[0].get("artifact_complete") is True
        and selected[0].get("joint_independent_receipt") is False
        and "independently_derived_2pi_clock_failed"
        in selected[0].get("blockers", [])
        and "h3_receipt_failed" in selected[0].get("blockers", [])
    ):
        raise ReplayBundleError(
            "historical campaign receipt lacks the registered 16k negative cell"
        )
    return receipt


def _assert_freeze_artifact_bindings(
    preregistration: Mapping[str, Any],
    preregistration_report: Mapping[str, Any],
    calibration_artifacts: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    historical_campaign_receipt_bytes: bytes,
    numerical_runtime: Mapping[str, Any],
) -> None:
    """Bind replay admission to the exact pre-source freeze and archive bytes."""

    freeze = _require_mapping(freeze_receipt, "freeze_receipt")
    _require_exact_keys(freeze, _FREEZE_RECEIPT_KEYS, "freeze_receipt")
    plan = _require_mapping(preregistration.get("plan"), "preregistration.plan")
    archive = _require_mapping(plan.get("archive_boundary"), "plan.archive_boundary")
    _verified_historical_campaign_receipt(historical_campaign_receipt_bytes)
    if not (
        freeze.get("schema") == _FREEZE_RECEIPT_SCHEMA
        and isinstance(freeze.get("frozen_at_utc"), str)
        and bool(freeze.get("frozen_at_utc"))
        and freeze.get("plan_sha256") == plan.get("plan_sha256")
        and freeze.get("preregistration_sha256")
        == canonical_sha256(preregistration)
        and freeze.get("instrument_commit_sha256")
        == plan.get("instrument_commit_sha256")
        and freeze.get("source_capture_allowed") is True
        and freeze.get("scientific_status") == "NOT_EVALUATED"
        and freeze.get("retune_after_freeze") is False
        and freeze.get("archived_16k_failure_preserved") is True
        and freeze.get("archived_outcomes_used_for_threshold_selection") is False
        and freeze.get("demo_or_nudge_controls_accepted") is False
    ):
        raise ReplayBundleError("pre-source freeze receipt anchor mismatch")
    if not (
        archive.get("historical_receipt_byte_sha256")
        == REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256
        and archive.get("historical_campaign_sha256")
        == REGISTERED_HISTORICAL_CAMPAIGN_SHA256
        and archive.get("historical_16k_source_seed")
        == REGISTERED_HISTORICAL_16K_SOURCE_SEED
        and archive.get("historical_16k_rung") == 16_384
        and archive.get("historical_16k_joint_independent_receipt") is False
        and archive.get("historical_stable_branch_failure_established") is False
        and archive.get("retune_after_freeze") is False
        and archive.get("archived_outcomes_used_for_threshold_selection") is False
    ):
        raise ReplayBundleError("frozen plan historical archive commitment mismatch")

    frozen_material = {
        "preregistration": _artifact_json_bytes(preregistration),
        "preregistration_report": _artifact_json_bytes(preregistration_report),
        "postrun_preregistration": _artifact_json_bytes(
            _postrun_envelope(preregistration, preregistration_report)
        ),
        "role_code_bindings": _artifact_json_bytes(
            registered_role_code_bindings()
        ),
        "instrument_code_bundle": _artifact_json_bytes(
            registered_instrument_code_bundle()
        ),
        "numerical_runtime": _artifact_json_bytes(numerical_runtime),
        **{
            name: _artifact_json_bytes(calibration_artifacts[name])
            for name in _CALIBRATION_NAMES
        },
        "historical_campaign_receipt": historical_campaign_receipt_bytes,
    }
    descriptors = _require_mapping(
        freeze.get("artifact_descriptors"), "freeze_receipt.artifact_descriptors"
    )
    _require_exact_keys(
        descriptors, set(_FROZEN_ARTIFACT_NAMES), "freeze artifact descriptors"
    )
    for name in _FROZEN_ARTIFACT_NAMES:
        row = _require_mapping(descriptors.get(name), f"freeze descriptor {name}")
        _require_exact_keys(row, _DESCRIPTOR_KEYS, f"freeze descriptor {name}")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative.startswith("freeze/"):
            raise ReplayBundleError(f"freeze descriptor path is invalid: {name}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise ReplayBundleError(f"freeze descriptor path is unsafe: {name}")
        data = frozen_material[name]
        if row.get("byte_sha256") != _sha256_bytes(data) or row.get(
            "byte_count"
        ) != len(data):
            raise ReplayBundleError(f"freeze descriptor byte mismatch: {name}")


def _assert_plan_role_code_bindings(preregistration: Mapping[str, Any]) -> None:
    plan = _require_mapping(preregistration.get("plan"), "preregistration.plan")
    actual = registered_role_code_registries()
    for registry_name in ("producer_registry", "checker_registry"):
        declared = _require_mapping(plan.get(registry_name), f"plan.{registry_name}")
        if _canonical_json_bytes(declared) != _canonical_json_bytes(
            actual[registry_name]
        ):
            raise ReplayBundleError(
                f"plan {registry_name} is not bound to actual registered code bytes"
            )
    actual_instrument = registered_instrument_code_bundle()
    if plan.get("instrument_commit_sha256") != actual_instrument[
        "instrument_commit_sha256"
    ]:
        raise ReplayBundleError(
            "plan instrument_commit_sha256 is not the recomputed code-bundle digest"
        )


def _assert_calibration_artifact_bindings(
    preregistration: Mapping[str, Any],
    calibration_artifacts: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        calibration_artifacts, set(_CALIBRATION_NAMES), "calibration_artifacts"
    )
    plan = _require_mapping(preregistration.get("plan"), "preregistration.plan")
    plan_calibrations = _require_mapping(
        plan.get("calibrations"), "plan.calibrations"
    )
    plan_thresholds = _require_mapping(plan.get("thresholds"), "plan.thresholds")
    campaign_seeds = plan.get("seeds")
    if not isinstance(campaign_seeds, list) or not all(
        type(seed) is int for seed in campaign_seeds
    ):
        raise ReplayBundleError("plan.seeds must be an exact integer list")
    if (
        plan_calibrations.get("independent_of_campaign_source_seeds") is not True
        or plan_calibrations.get("frozen_before_source_capture") is not True
    ):
        raise ReplayBundleError("plan calibration independence/freeze flags are false")
    plan_physical_calibration = plan_calibrations.get(
        "physical_threshold_calibration_receipt"
    )
    if type(plan_physical_calibration) is not bool:
        raise ReplayBundleError(
            "plan physical threshold-calibration receipt must be Boolean"
        )

    seed_sets: dict[str, set[int]] = {}
    calibration_ids: set[str] = set()
    for name in _CALIBRATION_NAMES:
        artifact = _require_mapping(calibration_artifacts.get(name), name)
        _require_exact_keys(artifact, _CALIBRATION_KEYS, name)
        if artifact.get("schema") != _CALIBRATION_SCHEMA[name]:
            raise ReplayBundleError(f"{name} schema mismatch")
        calibration_id = artifact.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ReplayBundleError(f"{name} calibration_id is invalid")
        if calibration_id in calibration_ids:
            raise ReplayBundleError("calibration IDs must be unique")
        calibration_ids.add(calibration_id)
        seeds = artifact.get("calibration_seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or not all(type(seed) is int for seed in seeds)
            or len(seeds) != len(set(seeds))
        ):
            raise ReplayBundleError(f"{name} calibration_seeds are invalid")
        seed_sets[name] = set(seeds)
        if artifact.get("independent_of_campaign_source_seeds") is not True:
            raise ReplayBundleError(f"{name} is not declared source-seed independent")
        if artifact.get("frozen_before_source_capture") is not True:
            raise ReplayBundleError(f"{name} was not frozen before source capture")
        protocol = _require_mapping(artifact.get("protocol"), f"{name}.protocol")
        thresholds = _require_mapping(
            artifact.get("thresholds"), f"{name}.thresholds"
        )
        if not protocol or not thresholds:
            raise ReplayBundleError(f"{name} protocol/thresholds must be nonempty")
        if type(protocol.get("physical_threshold_calibration_receipt")) is not bool:
            raise ReplayBundleError(
                f"{name} physical threshold-calibration receipt must be Boolean"
            )
        if (
            protocol.get("physical_threshold_calibration_receipt")
            is not plan_physical_calibration
        ):
            raise ReplayBundleError(
                f"{name} physical calibration status differs from frozen plan"
            )
        if type(protocol.get("physical_gate_eligible")) is not bool:
            raise ReplayBundleError(f"{name} physical_gate_eligible must be Boolean")
        if protocol.get("physical_gate_eligible") is not plan_physical_calibration:
            raise ReplayBundleError(
                f"{name} physical gate eligibility differs from frozen plan"
            )
        for field in _CALIBRATION_THRESHOLD_FIELDS[name]:
            if field not in thresholds or _canonical_json_bytes(
                thresholds[field]
            ) != _canonical_json_bytes(plan_thresholds.get(field)):
                raise ReplayBundleError(
                    f"{name} threshold does not match frozen plan: {field}"
                )
        hash_field = _CALIBRATION_PLAN_HASH_FIELD[name]
        if plan_calibrations.get(hash_field) != canonical_sha256(artifact):
            raise ReplayBundleError(
                f"{name} canonical content hash differs from plan {hash_field}"
            )
    campaign_seed_set = set(campaign_seeds)
    for name, seeds in seed_sets.items():
        if seeds.intersection(campaign_seed_set):
            raise ReplayBundleError(f"{name} seeds intersect campaign source seeds")
    for index, left_name in enumerate(_CALIBRATION_NAMES):
        for right_name in _CALIBRATION_NAMES[index + 1 :]:
            if seed_sets[left_name].intersection(seed_sets[right_name]):
                raise ReplayBundleError(
                    f"calibration seed sets overlap: {left_name},{right_name}"
                )


def write_physical_h3_kms_replay_bundle(
    output_dir: str | Path,
    preregistration: Mapping[str, Any],
    source_capture: Mapping[str, Any],
    calibration_artifacts: Mapping[str, Any],
    *,
    freeze_receipt: Mapping[str, Any],
    historical_campaign_receipt_bytes: bytes,
    numerical_runtime: Mapping[str, Any],
) -> Path:
    """Write a non-overwriting, content-addressed exact replay bundle."""

    preregistration_value = _require_mapping(preregistration, "preregistration")
    source_value = _require_mapping(source_capture, "source_capture")
    runtime_value = _require_mapping(numerical_runtime, "numerical_runtime")
    try:
        require_frozen_numerical_runtime(runtime_value)
    except NumericalRuntimeError as exc:
        raise ReplayBundleError(str(exc)) from exc
    preregistration_report = physical_h3_kms_prerun_report(preregistration_value)
    if (
        preregistration_report.get("SOURCE_CAPTURE_ALLOWED") is not True
        or preregistration_report.get("scientific_status") != "NOT_EVALUATED"
    ):
        raise ReplayBundleError("preregistration did not authorize source capture")
    expected_inputs = physical_h3_kms_source_inputs(preregistration_value)
    if _canonical_json_bytes(source_value.get("input_config")) != _canonical_json_bytes(
        expected_inputs
    ):
        raise ReplayBundleError("source capture input does not match preregistration")
    source_verification = verify_physical_source_capture(source_value)
    if source_verification.get("SOURCE_CAPTURE_REPLAY_RECEIPT") is not True:
        raise ReplayBundleError("source capture is not an exact deterministic replay")
    _assert_plan_role_code_bindings(preregistration_value)
    calibration_values = _require_mapping(
        calibration_artifacts, "calibration_artifacts"
    )
    _assert_calibration_artifact_bindings(
        preregistration_value, calibration_values
    )
    freeze_value = _require_mapping(freeze_receipt, "freeze_receipt")
    if not isinstance(historical_campaign_receipt_bytes, bytes):
        raise ReplayBundleError("historical_campaign_receipt_bytes must be bytes")
    _assert_freeze_artifact_bindings(
        preregistration_value,
        preregistration_report,
        calibration_values,
        freeze_value,
        historical_campaign_receipt_bytes,
        runtime_value,
    )

    postrun_report = _compute_postrun_reports_from_verified_source(
        source_value,
        _postrun_envelope(preregistration_value, preregistration_report),
    )
    artifact_values = {
        "preregistration": preregistration_value,
        "preregistration_report": preregistration_report,
        "freeze_receipt": freeze_value,
        "numerical_runtime": runtime_value,
        "historical_campaign_receipt": _verified_historical_campaign_receipt(
            historical_campaign_receipt_bytes
        ),
        "source_capture": source_value,
        "postrun_report": postrun_report,
        **{name: calibration_values[name] for name in _CALIBRATION_NAMES},
    }
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ReplayBundleError("output_dir must be a real directory")

    descriptors: dict[str, dict[str, Any]] = {}
    for name in _ARTIFACT_NAMES:
        relative = f"artifacts/{name}.json"
        data = (
            historical_campaign_receipt_bytes
            if name == "historical_campaign_receipt"
            else _artifact_json_bytes(artifact_values[name])
        )
        _write_exclusive(output_root / PurePosixPath(relative), data)
        descriptors[name] = _artifact_descriptor(relative, data)
    manifest = {
        "schema": REPLAY_MANIFEST_SCHEMA,
        "artifacts": descriptors,
        "role_code_bindings": registered_role_code_bindings(),
        "instrument_code_bundle": registered_instrument_code_bundle(),
        "replay_implementation": _code_file_descriptor(
            _REPLAY_IMPLEMENTATION_PATH
        ),
        "claim_boundary": (
            "This manifest commits disk bytes and registered replay code only. "
            "It contains no scientific or physical promotion receipt."
        ),
    }
    manifest_path = output_root / "replay_manifest.json"
    _write_exclusive(manifest_path, _artifact_json_bytes(manifest))
    return manifest_path


def _empty_report() -> dict[str, Any]:
    return {
        "schema": REPLAY_REPORT_SCHEMA,
        "manifest_byte_sha256": None,
        "artifact_byte_receipts": {},
        "PREREGISTRATION_REPORT_EXACT_REPLAY_RECEIPT": False,
        "PREREGISTRATION_SOURCE_CAPTURE_ADMISSION_RECEIPT": False,
        "PRE_SOURCE_FREEZE_ARTIFACT_BINDING_RECEIPT": False,
        "HISTORICAL_16K_ARCHIVE_BYTE_BINDING_RECEIPT": False,
        "REGISTERED_ROLE_CODE_BYTE_BINDING_RECEIPT": False,
        "INSTRUMENT_CODE_BUNDLE_BINDING_RECEIPT": False,
        "CALIBRATION_ARTIFACT_BINDING_RECEIPT": False,
        "NUMERICAL_RUNTIME_ARTIFACT_BINDING_RECEIPT": False,
        "SOURCE_INPUT_BINDING_RECEIPT": False,
        "SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT": False,
        "POSTRUN_REPORT_EXACT_REPLAY_RECEIPT": False,
        "PREFLIGHT_EXPORT_PROJECTION_RECEIPT": False,
        "preflight_export_hashes": {},
        "PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT": False,
        **{name: False for name in _COMPATIBILITY_RECEIPTS},
        "compatibility_receipt_semantics": {
            "REPLAY_MANIFEST_VERIFICATION_RECEIPT": (
                "strict manifest schema, code binding, and artifact-byte verification"
            ),
            "PRE_SOURCE_FREEZE_REPLAY_RECEIPT": (
                "exact frozen-plan, code, calibration, and archive descriptors"
            ),
            "HISTORICAL_16K_ARCHIVE_BYTE_REPLAY_RECEIPT": (
                "registered historical receipt bytes and 16k negative-cell census"
            ),
            "SOURCE_CAPTURE_REPLAY_RECEIPT": (
                "exact deterministic source software replay"
            ),
            "PER_CELL_CONTROL_ARTIFACTS_REPLAY_RECEIPT": (
                "content-bound independent calibration/control artifact replay"
            ),
            "PER_CELL_SCIENTIFIC_PREDICATES_RECOMPUTED_RECEIPT": (
                "postrun predicates recomputed, irrespective of pass or failure"
            ),
            "SINGLE_BUNDLE_COMMITMENT_RECEIPT": (
                "one manifest commits every consumed artifact and implementation"
            ),
            "NUMERICAL_RUNTIME_REPLAY_RECEIPT": (
                "exact interpreter, NumPy/SciPy build, machine ABI, and thread policy"
            ),
            "claim_boundary": (
                "These are instrument/computation compatibility receipts, not "
                "positive scientific predicate results."
            ),
        },
        "scientific_evaluation_performed_by_replay_layer": False,
        "scientific_status": "NOT_EVALUATED_BY_DISK_REPLAY",
        "physical_promotion_allowed": False,
        "scientific_promotion_allowed": False,
        "blockers": [],
        "claim_boundary": (
            "A disk replay receipt proves byte integrity and deterministic software "
            "recomputation only; it never proves a physics stage."
        ),
    }


def _read_artifact(
    bundle_root: Path,
    name: str,
    descriptor: Mapping[str, Any],
    report: dict[str, Any],
) -> Any:
    row = _require_mapping(descriptor, f"artifacts.{name}")
    _require_exact_keys(row, _DESCRIPTOR_KEYS, f"artifacts.{name}")
    declared_hash = row.get("byte_sha256")
    declared_count = row.get("byte_count")
    if not isinstance(declared_hash, str) or not _SHA256_RE.fullmatch(declared_hash):
        raise ReplayBundleError(f"artifacts.{name} byte_sha256 is malformed")
    if type(declared_count) is not int or declared_count < 0:
        raise ReplayBundleError(f"artifacts.{name} byte_count is invalid")
    path = _safe_bundle_file(bundle_root, row.get("path"))
    computed_hash, computed_count = _hash_file(path)
    passed = computed_hash == declared_hash and computed_count == declared_count
    report["artifact_byte_receipts"][name] = {
        "path": row["path"],
        "declared_byte_sha256": declared_hash,
        "computed_byte_sha256": computed_hash,
        "declared_byte_count": declared_count,
        "computed_byte_count": computed_count,
        "BYTE_SHA256_RECEIPT": passed,
    }
    if not passed:
        raise ReplayBundleError(f"artifacts.{name} byte commitment mismatch")
    return _strict_json_loads(path.read_bytes(), label=f"artifacts.{name}")


def _replay_manifest_value(
    manifest: Mapping[str, Any],
    bundle_root: Path,
    report: dict[str, Any],
) -> None:
    value = dict(manifest)
    _require_exact_keys(value, _MANIFEST_KEYS, "manifest")
    if value.get("schema") != REPLAY_MANIFEST_SCHEMA:
        raise ReplayBundleError("manifest schema mismatch")
    artifacts = _require_mapping(value.get("artifacts"), "manifest.artifacts")
    _require_exact_keys(artifacts, set(_ARTIFACT_NAMES), "manifest.artifacts")
    relative_paths = [
        _require_mapping(artifacts[name], f"artifacts.{name}").get("path")
        for name in _ARTIFACT_NAMES
    ]
    if len(set(relative_paths)) != len(relative_paths):
        raise ReplayBundleError("artifact paths must be unique")

    actual_bindings = registered_role_code_bindings()
    if _canonical_json_bytes(value.get("role_code_bindings")) != _canonical_json_bytes(
        actual_bindings
    ):
        raise ReplayBundleError("manifest role code bindings differ from actual bytes")
    actual_instrument = registered_instrument_code_bundle()
    if _canonical_json_bytes(value.get("instrument_code_bundle")) != _canonical_json_bytes(
        actual_instrument
    ):
        raise ReplayBundleError("manifest instrument code bundle differs from actual bytes")
    actual_replay = _code_file_descriptor(_REPLAY_IMPLEMENTATION_PATH)
    if _canonical_json_bytes(value.get("replay_implementation")) != _canonical_json_bytes(
        actual_replay
    ):
        raise ReplayBundleError("manifest replay implementation byte binding mismatch")

    loaded = {
        name: _read_artifact(bundle_root, name, artifacts[name], report)
        for name in _ARTIFACT_NAMES
    }
    preregistration = _require_mapping(
        loaded["preregistration"], "preregistration artifact"
    )
    supplied_prerun = _require_mapping(
        loaded["preregistration_report"], "preregistration report artifact"
    )
    freeze_receipt = _require_mapping(
        loaded["freeze_receipt"], "freeze receipt artifact"
    )
    numerical_runtime = _require_mapping(
        loaded["numerical_runtime"], "numerical runtime artifact"
    )
    source_capture = _require_mapping(loaded["source_capture"], "source capture artifact")
    supplied_postrun = _require_mapping(
        loaded["postrun_report"], "postrun report artifact"
    )
    calibration_artifacts = {
        name: _require_mapping(loaded[name], f"{name} artifact")
        for name in _CALIBRATION_NAMES
    }
    historical_descriptor = _require_mapping(
        artifacts.get("historical_campaign_receipt"),
        "artifacts.historical_campaign_receipt",
    )
    historical_path = _safe_bundle_file(
        bundle_root, historical_descriptor.get("path")
    )
    historical_campaign_receipt_bytes = historical_path.read_bytes()

    try:
        require_frozen_numerical_runtime(numerical_runtime)
    except NumericalRuntimeError as exc:
        raise ReplayBundleError(str(exc)) from exc
    report["NUMERICAL_RUNTIME_ARTIFACT_BINDING_RECEIPT"] = True

    recomputed_prerun = physical_h3_kms_prerun_report(preregistration)
    report["PREREGISTRATION_REPORT_EXACT_REPLAY_RECEIPT"] = bool(
        _canonical_json_bytes(supplied_prerun)
        == _canonical_json_bytes(recomputed_prerun)
    )
    if not report["PREREGISTRATION_REPORT_EXACT_REPLAY_RECEIPT"]:
        raise ReplayBundleError("preregistration report is not an exact replay")
    report["PREREGISTRATION_SOURCE_CAPTURE_ADMISSION_RECEIPT"] = bool(
        recomputed_prerun.get("SOURCE_CAPTURE_ALLOWED") is True
        and recomputed_prerun.get("scientific_status") == "NOT_EVALUATED"
    )
    if not report["PREREGISTRATION_SOURCE_CAPTURE_ADMISSION_RECEIPT"]:
        raise ReplayBundleError("recomputed preregistration did not admit capture")

    _assert_plan_role_code_bindings(preregistration)
    report["REGISTERED_ROLE_CODE_BYTE_BINDING_RECEIPT"] = True
    report["INSTRUMENT_CODE_BUNDLE_BINDING_RECEIPT"] = True
    _assert_calibration_artifact_bindings(preregistration, calibration_artifacts)
    report["CALIBRATION_ARTIFACT_BINDING_RECEIPT"] = True
    _assert_freeze_artifact_bindings(
        preregistration,
        recomputed_prerun,
        calibration_artifacts,
        freeze_receipt,
        historical_campaign_receipt_bytes,
        numerical_runtime,
    )
    report["PRE_SOURCE_FREEZE_ARTIFACT_BINDING_RECEIPT"] = True
    report["HISTORICAL_16K_ARCHIVE_BYTE_BINDING_RECEIPT"] = True
    expected_inputs = physical_h3_kms_source_inputs(preregistration)
    report["SOURCE_INPUT_BINDING_RECEIPT"] = bool(
        _canonical_json_bytes(source_capture.get("input_config"))
        == _canonical_json_bytes(expected_inputs)
    )
    if not report["SOURCE_INPUT_BINDING_RECEIPT"]:
        raise ReplayBundleError("source input differs from admitted projection")
    source_verification = verify_physical_source_capture(source_capture)
    report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"] = bool(
        source_verification.get("SOURCE_CAPTURE_REPLAY_RECEIPT") is True
        and source_verification.get("source_root_sha256")
        == source_capture.get("source_root_sha256")
    )
    if not report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"]:
        raise ReplayBundleError("source capture is not an exact replay")

    recomputed_postrun = _compute_postrun_reports_from_verified_source(
        source_capture,
        _postrun_envelope(preregistration, recomputed_prerun),
    )
    report["POSTRUN_REPORT_EXACT_REPLAY_RECEIPT"] = bool(
        _canonical_json_bytes(supplied_postrun)
        == _canonical_json_bytes(recomputed_postrun)
    )
    if not report["POSTRUN_REPORT_EXACT_REPLAY_RECEIPT"]:
        raise ReplayBundleError("postrun report is not an exact replay")

    source_reports = _require_mapping(
        source_capture.get("reports"), "source_capture.reports"
    )
    preflight_export_values = {
        "config": _require_mapping(source_capture.get("config"), "source_capture.config"),
        "source_observer": _require_mapping(
            source_reports.get("source_observer"), "source_capture.reports.source_observer"
        ),
        "refinement": _require_mapping(
            source_reports.get("refinement"), "source_capture.reports.refinement"
        ),
        "prime_geometric_state": _require_mapping(
            source_reports.get("prime_geometric_state"),
            "source_capture.reports.prime_geometric_state",
        ),
        "independent_geometry": _require_mapping(
            source_reports.get("independent_geometry"),
            "source_capture.reports.independent_geometry",
        ),
        "native_bw": _require_mapping(
            recomputed_postrun.get("native_bw"), "postrun_report.native_bw"
        ),
        "candidate_interventions": _require_mapping(
            recomputed_postrun.get("candidate_interventions"),
            "postrun_report.candidate_interventions",
        ),
        "geometry_controls": _require_mapping(
            recomputed_postrun.get("geometry_controls"),
            "postrun_report.geometry_controls",
        ),
        "semantic_event": _require_mapping(
            recomputed_postrun.get("semantic_event"), "postrun_report.semantic_event"
        ),
        "campaign": _require_mapping(
            recomputed_postrun.get("campaign"), "postrun_report.campaign"
        ),
    }
    report["preflight_export_hashes"] = {
        name: canonical_sha256(export)
        for name, export in preflight_export_values.items()
    }
    report["PREFLIGHT_EXPORT_PROJECTION_RECEIPT"] = True

    report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"] = True
    report["REPLAY_MANIFEST_VERIFICATION_RECEIPT"] = True
    report["PRE_SOURCE_FREEZE_REPLAY_RECEIPT"] = True
    report["HISTORICAL_16K_ARCHIVE_BYTE_REPLAY_RECEIPT"] = True
    report["SOURCE_CAPTURE_REPLAY_RECEIPT"] = True
    report["PER_CELL_CONTROL_ARTIFACTS_REPLAY_RECEIPT"] = True
    report["PER_CELL_SCIENTIFIC_PREDICATES_RECOMPUTED_RECEIPT"] = True
    report["SINGLE_BUNDLE_COMMITMENT_RECEIPT"] = True
    report["NUMERICAL_RUNTIME_REPLAY_RECEIPT"] = True


def replay_physical_h3_kms_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Replay one strict disk bundle and return fail-closed integrity receipts."""

    report = _empty_report()
    try:
        path = Path(manifest_path)
        if path.is_symlink() or not path.is_file():
            raise ReplayBundleError("manifest must be a regular non-symlink file")
        manifest_bytes = path.read_bytes()
        report["manifest_byte_sha256"] = _sha256_bytes(manifest_bytes)
        manifest = _require_mapping(
            _strict_json_loads(manifest_bytes, label="manifest"), "manifest"
        )
        _replay_manifest_value(manifest, path.parent, report)
    except (
        ReplayBundleError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        report["blockers"].append(
            f"replay_failed:{type(exc).__name__}:{str(exc)}"
        )
        report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"] = False
        for receipt_name in _COMPATIBILITY_RECEIPTS:
            report[receipt_name] = False
    return report


__all__ = [
    "REPLAY_MANIFEST_SCHEMA",
    "REPLAY_REPORT_SCHEMA",
    "ReplayBundleError",
    "registered_role_code_bindings",
    "registered_role_code_registries",
    "registered_instrument_code_bundle",
    "replay_physical_h3_kms_bundle",
    "write_physical_h3_kms_replay_bundle",
]
