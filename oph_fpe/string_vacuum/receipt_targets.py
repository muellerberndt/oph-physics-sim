"""Canonical receipt and observable targets for the string-vacuum lane."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from oph_fpe.evidence.hashes import stable_json_hash


REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_TARGETS_PATH = REPO_ROOT / "configs/string_vacuum/receipt_targets_v1.json"
RECEIPT_TARGETS_SCHEMA_PATH = REPO_ROOT / "schemas/string_vacuum_receipt_targets_v1.schema.json"
OBSERVABLE_TARGETS_PATH = REPO_ROOT / "configs/string_vacuum/oph_observable_targets_v1.json"
OBSERVABLE_TARGETS_SCHEMA_PATH = (
    REPO_ROOT / "schemas/string_vacuum_observable_targets_v1.schema.json"
)

CANDIDATE_GATE_IDS = (
    "critical_edge_cft",
    "full_cohomology",
    "operator_safety_realization",
    "superpotential_safety",
    "anomaly_bianchi_tadpole",
    "vacuum_stability",
    "heavy_spectrum",
    "low_energy_continuation",
    "common_scheme_thresholds",
    "complete_target_match",
    "physical_quotient_descent",
    "source_target_separation",
    "augmented_local_isolation",
)

CATALOGUE_PROOF_IDS = {
    "catalogue_enumeration",
    "equivalence_partition",
    "branch_domain_coverage",
    "branch_verdict_replay",
    "unrestricted_reduction",
}

CANONICAL_CANDIDATE_RECEIPT_IDS = {
    "STRING_CRITICAL_EDGE_CFT_RECEIPT",
    "STRING_FULL_MASSLESS_SECTOR_RECEIPT",
    "STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT",
    "STRING_SUPERPOTENTIAL_SAFETY_RECEIPT",
    "STRING_THRESHOLD_SPECTRUM_RECEIPT",
    "STRING_COMPLETED_PHYSICAL_SLICE_RECEIPT",
    "STRING_MODULI_LOCKING_RECEIPT",
    "STRING_CANDIDATE_CONSISTENCY_RECEIPT",
    "LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT",
}

EXPECTED_RECEIPT_TARGET_REGISTRY_SHA256 = (
    "sha256:b5190a5ab0b4c688857c4f9e0a2d82e4f3d586f48ee5d622139541e8151b8c1a"
)
EXPECTED_OBSERVABLE_TARGET_REGISTRY_SHA256 = (
    "sha256:4348065855bdfc2b6ef0447fc67f4582fa0dddcf3e262a019c20659acc235f6e"
)
EXPECTED_RECEIPT_ALIASES = {
    "CRITICAL_EDGE_CFT_RECEIPT": "STRING_CRITICAL_EDGE_CFT_RECEIPT",
    "BD_FULL_COHOMOLOGY_CERTIFICATE_RECEIPT": "STRING_FULL_MASSLESS_SECTOR_RECEIPT",
    "BD_Z4R_COMPACTIFICATION_REALIZATION_RECEIPT": (
        "STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT"
    ),
    "BD_SUPERPOTENTIAL_SAFETY_RECEIPT": "STRING_SUPERPOTENTIAL_SAFETY_RECEIPT",
    "BD_THRESHOLD_SPECTRUM_CERTIFICATE_RECEIPT": "STRING_THRESHOLD_SPECTRUM_RECEIPT",
    "BD_COMPLETED_PHYSICAL_SLICE_RECEIPT": "STRING_COMPLETED_PHYSICAL_SLICE_RECEIPT",
    "BD_MODULI_LOCKING_CERTIFICATE_RECEIPT": "STRING_MODULI_LOCKING_RECEIPT",
}


class ReceiptTargetRegistryError(ValueError):
    """Raised when a canonical string-vacuum target registry drifts."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptTargetRegistryError(f"cannot read target registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptTargetRegistryError(f"target registry is not an object: {path}")
    return payload


def _schema_errors(payload: Mapping[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [
        f"/{'/'.join(str(part) for part in error.absolute_path)}:{error.message}"
        for error in errors
    ]


def validate_receipt_target_registry(payload: Mapping[str, Any]) -> None:
    """Validate schema, identifiers, aliases, dependencies, and scope monotonicity."""

    errors = _schema_errors(payload, RECEIPT_TARGETS_SCHEMA_PATH)
    if errors:
        raise ReceiptTargetRegistryError("receipt target schema failure: " + "; ".join(errors))

    targets = payload["targets"]
    rows: dict[str, Mapping[str, Any]] = {}
    for row in targets:
        receipt_id = row["receipt_id"]
        if receipt_id in rows:
            raise ReceiptTargetRegistryError(f"duplicate receipt target: {receipt_id}")
        rows[receipt_id] = row

    missing_canonical = CANONICAL_CANDIDATE_RECEIPT_IDS - set(rows)
    if missing_canonical:
        raise ReceiptTargetRegistryError(
            f"canonical candidate receipt targets missing: {sorted(missing_canonical)}"
        )

    aliases = payload["receipt_aliases"]
    if aliases != EXPECTED_RECEIPT_ALIASES:
        raise ReceiptTargetRegistryError("receipt compatibility alias registry drift")
    for alias, canonical in aliases.items():
        if alias in rows:
            raise ReceiptTargetRegistryError(f"receipt alias collides with canonical target: {alias}")
        if canonical not in rows:
            raise ReceiptTargetRegistryError(
                f"receipt alias {alias} names unknown canonical target {canonical}"
            )

    scope_rank = {"candidate": 0, "branch": 1, "catalogue": 2}
    for receipt_id, row in rows.items():
        unknown_dependencies = set(row["dependencies"]) - set(rows)
        if unknown_dependencies:
            raise ReceiptTargetRegistryError(
                f"{receipt_id} has unknown dependencies: {sorted(unknown_dependencies)}"
            )
        if receipt_id in row["dependencies"]:
            raise ReceiptTargetRegistryError(f"{receipt_id} depends on itself")
        for dependency in row["dependencies"]:
            if scope_rank[rows[dependency]["report_scope"]] > scope_rank[row["report_scope"]]:
                raise ReceiptTargetRegistryError(
                    f"scope-inverting dependency: {receipt_id} -> {dependency}"
                )
        unknown_gates = set(row["semantic_gate_ids"]) - set(CANDIDATE_GATE_IDS)
        if unknown_gates:
            raise ReceiptTargetRegistryError(
                f"{receipt_id} has unknown semantic gates: {sorted(unknown_gates)}"
            )
        unknown_proofs = set(row["catalogue_proof_ids"]) - CATALOGUE_PROOF_IDS
        if unknown_proofs:
            raise ReceiptTargetRegistryError(
                f"{receipt_id} has unknown catalogue proofs: {sorted(unknown_proofs)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(receipt_id: str) -> None:
        if receipt_id in visiting:
            raise ReceiptTargetRegistryError(f"receipt dependency cycle at {receipt_id}")
        if receipt_id in visited:
            return
        visiting.add(receipt_id)
        for dependency in rows[receipt_id]["dependencies"]:
            visit(dependency)
        visiting.remove(receipt_id)
        visited.add(receipt_id)

    for receipt_id in rows:
        visit(receipt_id)
    if stable_json_hash(payload) != EXPECTED_RECEIPT_TARGET_REGISTRY_SHA256:
        raise ReceiptTargetRegistryError("receipt target registry content hash drift")


def validate_observable_target_registry(payload: Mapping[str, Any]) -> None:
    """Validate the frozen comparison rows and their rank-role partition."""

    errors = _schema_errors(payload, OBSERVABLE_TARGETS_SCHEMA_PATH)
    if errors:
        raise ReceiptTargetRegistryError("observable target schema failure: " + "; ".join(errors))
    rows: dict[str, Mapping[str, Any]] = {}
    for row in payload["rows"]:
        row_id = row["row_id"]
        if row_id in rows:
            raise ReceiptTargetRegistryError(f"duplicate observable target row: {row_id}")
        rows[row_id] = row
    eligible = set(payload["rank_eligible_row_ids"])
    forbidden = set(payload["rank_forbidden_row_ids"])
    if eligible & forbidden:
        raise ReceiptTargetRegistryError("rank-eligible and rank-forbidden target rows overlap")
    if eligible | forbidden != set(rows):
        raise ReceiptTargetRegistryError("observable rank-role lists do not partition all rows")
    if set(payload["compare_only_inputs_not_targets"]) & set(rows):
        raise ReceiptTargetRegistryError("compare-only inputs overlap observable target rows")
    for row_id in eligible:
        if rows[row_id]["role"] != "promoted_oph_target" or not rows[row_id]["rank_eligible"]:
            raise ReceiptTargetRegistryError(f"rank-eligible row has wrong role: {row_id}")
    for row_id in forbidden:
        if rows[row_id]["role"] != "candidate_only" or rows[row_id]["rank_eligible"]:
            raise ReceiptTargetRegistryError(f"rank-forbidden row has wrong role: {row_id}")
    if payload["complete"] or payload["promotion_allowed"]:
        raise ReceiptTargetRegistryError("v1 observable registry must remain fail-closed while OPEN")
    if stable_json_hash(payload) != EXPECTED_OBSERVABLE_TARGET_REGISTRY_SHA256:
        raise ReceiptTargetRegistryError("observable target registry content hash drift")


def receipt_target_registry() -> dict[str, Any]:
    payload = _read_json(RECEIPT_TARGETS_PATH)
    validate_receipt_target_registry(payload)
    return copy.deepcopy(payload)


def observable_target_registry() -> dict[str, Any]:
    payload = _read_json(OBSERVABLE_TARGETS_PATH)
    validate_observable_target_registry(payload)
    return copy.deepcopy(payload)


def receipt_target_registry_sha256() -> str:
    return stable_json_hash(receipt_target_registry())


def observable_target_registry_sha256() -> str:
    return stable_json_hash(observable_target_registry())


def receipt_target(receipt_id: str) -> dict[str, Any]:
    payload = receipt_target_registry()
    canonical = payload["receipt_aliases"].get(receipt_id, receipt_id)
    for row in payload["targets"]:
        if row["receipt_id"] == canonical:
            return copy.deepcopy(row)
    raise KeyError(f"unknown string-vacuum receipt target: {receipt_id}")
