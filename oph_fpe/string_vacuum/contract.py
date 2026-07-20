"""Independent, fail-closed string-vacuum evidence verifier.

Producer classifications and gate Booleans are hints.  This verifier checks the
strict schemas, artifact bytes, coordinate and target registries, and the exact
interval contraction algebra.  It deliberately has no semantic verifiers for
worldsheet, compactification, spectrum, or threshold physics.  Those gates
therefore remain inconclusive until code-owned, replayable primitive verifiers
are implemented and registered here.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

from oph_fpe.evidence.hashes import stable_json_hash
from .receipt_targets import CANDIDATE_GATE_IDS, receipt_target_registry_sha256
from .verified_numerics import verify_interval_contraction


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SCHEMA_PATH = REPO_ROOT / "schemas/string_vacuum_candidate_evidence_v1.schema.json"
CATALOGUE_SCHEMA_PATH = REPO_ROOT / "schemas/string_vacuum_catalogue_evidence_v1.schema.json"

# A positive scientific gate must come from a code-owned primitive verifier.
# The registry is intentionally empty.  Schema-valid producer envelopes and
# hashes alone cannot promote a string vacuum.
SEMANTIC_GATE_VERIFIERS: dict[str, Callable[[dict[str, Any], Path], str]] = {}

# Catalogue completeness, equivalence, branch exclusion, and unrestricted
# reduction are scientific proof obligations too. A content hash proves only
# which bytes were supplied, not what those bytes establish.
CATALOGUE_PROOF_VERIFIERS: dict[
    str, Callable[[dict[str, Any], dict[str, Any] | None, Path], bool]
] = {}

MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _schema_errors(payload: Any, schema_path: Path) -> list[str]:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"canonical_schema_unreadable:{exc}"]
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    rendered: list[str] = []
    for error in errors:
        location = "/" + "/".join(str(part) for part in error.absolute_path)
        rendered.append(f"schema:{location}:{error.message}")
    return rendered


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_regular_file(bundle_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact_path_escape")
    root = bundle_root.resolve()
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("artifact_symlink_rejected")
    resolved = cursor.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact_path_escape") from exc
    if not resolved.is_file():
        raise ValueError("artifact_not_regular_file")
    if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact_exceeds_size_limit")
    return resolved


def _verify_artifacts(
    rows: list[dict[str, Any]],
    bundle_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    blockers: list[str] = []
    seen_paths: set[str] = set()
    for row in rows:
        artifact_id = row["artifact_id"]
        if artifact_id in by_id:
            blockers.append(f"duplicate_artifact_id:{artifact_id}")
            continue
        if row["path"] in seen_paths:
            blockers.append(f"duplicate_artifact_path:{row['path']}")
        seen_paths.add(row["path"])
        by_id[artifact_id] = row
        if bundle_root is None:
            blockers.append(f"artifact_root_not_supplied:{artifact_id}")
            continue
        try:
            path = _resolve_regular_file(bundle_root, row["path"])
        except (OSError, ValueError) as exc:
            blockers.append(f"artifact_invalid:{artifact_id}:{exc}")
            continue
        actual = _sha256_file(path)
        if actual != row["sha256"]:
            blockers.append(f"artifact_hash_mismatch:{artifact_id}")
            continue
        paths[artifact_id] = path
    return by_id, paths, blockers


def _require_artifact(
    artifact_id: str | None,
    *,
    field: str,
    artifacts: Mapping[str, dict[str, Any]],
    verified_paths: Mapping[str, Path],
    blockers: list[str],
) -> bool:
    if not artifact_id:
        blockers.append(f"missing_artifact_reference:{field}")
        return False
    elif artifact_id not in artifacts:
        blockers.append(f"unknown_artifact_reference:{field}:{artifact_id}")
        return False
    elif artifact_id not in verified_paths:
        blockers.append(f"unverified_artifact_reference:{field}:{artifact_id}")
        return False
    return True


def _is_artifact_blocker(blocker: str) -> bool:
    return blocker.startswith(
        (
            "duplicate_artifact_id:",
            "duplicate_artifact_path:",
            "artifact_root_not_supplied:",
            "artifact_invalid:",
            "artifact_hash_mismatch:",
            "missing_artifact_reference:",
            "unknown_artifact_reference:",
            "unverified_artifact_reference:",
            "artifact_registry_hash_mismatch:",
        )
    )


def _artifact_blocker_targets_id(blocker: str, artifact_id: str) -> bool:
    return blocker in {
        f"duplicate_artifact_id:{artifact_id}",
        f"artifact_root_not_supplied:{artifact_id}",
        f"artifact_hash_mismatch:{artifact_id}",
    } or blocker.startswith(
        f"artifact_invalid:{artifact_id}:"
    )


def _artifact_id_is_ambiguous(blockers: list[str], artifact_id: str | None) -> bool:
    return bool(
        artifact_id and f"duplicate_artifact_id:{artifact_id}" in blockers
    )


def _require_bound_artifact_hash(
    artifact_id: str | None,
    expected_sha256: str,
    *,
    field: str,
    artifacts: Mapping[str, dict[str, Any]],
    verified_paths: Mapping[str, Path],
    blockers: list[str],
) -> None:
    """Require a verified artifact whose declared bytes match a registry hash."""

    before = len(blockers)
    _require_artifact(
        artifact_id,
        field=field,
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    if len(blockers) != before or artifact_id is None:
        return
    if artifacts[artifact_id]["sha256"] != expected_sha256:
        blockers.append(f"artifact_registry_hash_mismatch:{field}:{artifact_id}")


def _candidate_report_base(payload: Any, blockers: list[str]) -> dict[str, Any]:
    candidate = payload.get("candidate", {}) if isinstance(payload, dict) else {}
    return {
        "artifact": "oph_string_vacuum_candidate_verification",
        "report_schema_version": 1,
        "receipt_target_registry_sha256": receipt_target_registry_sha256(),
        "receipt_subject": None,
        "candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "oph_equivalence_class_id": candidate.get("oph_equivalence_class_id"),
            "theory_family": candidate.get("theory_family"),
            "branch_definition_sha256": candidate.get("branch_definition_sha256"),
        },
        "candidate_status": "INVALID",
        "contract_integrity_receipt": False,
        "artifact_hash_receipt": False,
        "oph_target_registry_binding_receipt": False,
        "interval_contraction_algebra_receipt": False,
        "evaluator_enclosure_receipt": False,
        "full_system_closure_receipt": False,
        "flat_direction_classification_receipt": False,
        "physical_local_isolation_receipt": False,
        "selector_independence_receipt": False,
        "interval_algebra": {
            "interval_contraction_receipt": False,
            "rank_receipt": False,
            "existence_receipt": False,
            "local_isolation_receipt": False,
            "blockers": ["candidate_packet_invalid"],
        },
        "semantic_gate_receipts": {gate_id: False for gate_id in CANDIDATE_GATE_IDS},
        "receipts": _aggregate_receipts({gate_id: False for gate_id in CANDIDATE_GATE_IDS}),
        "blockers": sorted(set(blockers)),
        "nonclaims": [
            "certified string vacuum",
            "branch-global uniqueness",
            "catalogue-relative uniqueness",
            "globally correct string theory",
        ],
    }


def _candidate_receipt_subject(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload["candidate"]
    provenance = payload["provenance"]
    subject = {
        "subject_type": "string_vacuum_candidate",
        "candidate_id": candidate["candidate_id"],
        "oph_equivalence_class_id": candidate["oph_equivalence_class_id"],
        "branch_definition_sha256": candidate["branch_definition_sha256"],
        "source_freeze_sha256": provenance["source_freeze_sha256"],
        "target_registry_sha256": provenance["target_registry_sha256"],
        "constraint_registry_sha256": payload["constraint_registry"]["registry_sha256"],
        "receipt_target_registry_sha256": receipt_target_registry_sha256(),
    }
    subject["subject_scope_sha256"] = stable_json_hash(subject)
    return subject


def _aggregate_receipts(gates: Mapping[str, bool]) -> dict[str, bool]:
    operator_safety = gates.get("operator_safety_realization", False)
    superpotential_safety = bool(
        operator_safety and gates.get("superpotential_safety", False)
    )
    threshold = all(
        gates.get(name, False)
        for name in (
            "anomaly_bianchi_tadpole",
            "heavy_spectrum",
            "low_energy_continuation",
            "common_scheme_thresholds",
        )
    )
    completed_slice = all(
        gates.get(name, False)
        for name in ("anomaly_bianchi_tadpole", "vacuum_stability", "physical_quotient_descent")
    )
    moduli_locking = completed_slice and all(
        gates.get(name, False)
        for name in (
            "complete_target_match",
            "source_target_separation",
            "augmented_local_isolation",
        )
    )
    candidate_consistency = all(
        gates.get(name, False)
        for name in (
            "critical_edge_cft",
            "full_cohomology",
            "operator_safety_realization",
            "superpotential_safety",
            "anomaly_bianchi_tadpole",
            "vacuum_stability",
            "heavy_spectrum",
            "low_energy_continuation",
            "common_scheme_thresholds",
            "physical_quotient_descent",
        )
    )
    local_witness = candidate_consistency and all(
        gates.get(name, False)
        for name in (
            "complete_target_match",
            "source_target_separation",
            "augmented_local_isolation",
        )
    )
    canonical = {
        "STRING_CRITICAL_EDGE_CFT_RECEIPT": gates.get("critical_edge_cft", False),
        "STRING_FULL_MASSLESS_SECTOR_RECEIPT": gates.get("full_cohomology", False),
        "STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT": operator_safety,
        "STRING_SUPERPOTENTIAL_SAFETY_RECEIPT": superpotential_safety,
        "STRING_THRESHOLD_SPECTRUM_RECEIPT": threshold,
        "STRING_COMPLETED_PHYSICAL_SLICE_RECEIPT": completed_slice,
        "STRING_MODULI_LOCKING_RECEIPT": moduli_locking,
        "STRING_CANDIDATE_CONSISTENCY_RECEIPT": candidate_consistency,
        "LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT": local_witness,
    }
    aliases = {
        "CRITICAL_EDGE_CFT_RECEIPT": canonical["STRING_CRITICAL_EDGE_CFT_RECEIPT"],
        "BD_FULL_COHOMOLOGY_CERTIFICATE_RECEIPT": canonical[
            "STRING_FULL_MASSLESS_SECTOR_RECEIPT"
        ],
        "BD_Z4R_COMPACTIFICATION_REALIZATION_RECEIPT": canonical[
            "STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT"
        ],
        "BD_SUPERPOTENTIAL_SAFETY_RECEIPT": canonical[
            "STRING_SUPERPOTENTIAL_SAFETY_RECEIPT"
        ],
        "BD_THRESHOLD_SPECTRUM_CERTIFICATE_RECEIPT": canonical[
            "STRING_THRESHOLD_SPECTRUM_RECEIPT"
        ],
        "BD_COMPLETED_PHYSICAL_SLICE_RECEIPT": canonical[
            "STRING_COMPLETED_PHYSICAL_SLICE_RECEIPT"
        ],
        "BD_MODULI_LOCKING_CERTIFICATE_RECEIPT": canonical["STRING_MODULI_LOCKING_RECEIPT"],
    }
    return canonical | aliases


def verify_candidate_evidence(
    payload: dict[str, Any],
    *,
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """Verify one candidate packet without trusting producer verdicts."""

    blockers = _schema_errors(payload, CANDIDATE_SCHEMA_PATH)
    if blockers:
        report = _candidate_report_base(payload, blockers)
        report["report_sha256"] = stable_json_hash(report)
        return report

    artifacts, verified_paths, artifact_blockers = _verify_artifacts(payload["artifacts"], bundle_root)
    blockers.extend(artifact_blockers)

    candidate = payload["candidate"]
    provenance = payload["provenance"]
    quotient = payload["presentation_and_quotient"]
    constraint_registry = payload["constraint_registry"]
    target = payload["target_registry"]
    gates = payload["gates"]
    dimension = len(quotient["physical_coordinate_ids"])

    _require_bound_artifact_hash(
        candidate["branch_definition_artifact_id"],
        candidate["branch_definition_sha256"],
        field="branch_definition",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    for prefix in ("dependency_lock", "environment", "source_freeze"):
        _require_bound_artifact_hash(
            provenance[f"{prefix}_artifact_id"],
            provenance[f"{prefix}_sha256"],
            field=prefix,
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )

    if dimension > quotient["prequotient_dimension_real"]:
        blockers.append("physical_dimension_exceeds_prequotient_dimension")
    _require_bound_artifact_hash(
        quotient["coordinate_registry_artifact_id"],
        quotient["coordinate_registry_sha256"],
        field="coordinate_registry",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    _require_artifact(
        quotient["quotient_descent_proof_artifact_id"],
        field="quotient_descent_proof_artifact_id",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    for row in quotient["removed_redundancies"]:
        _require_artifact(
            row["evidence_artifact_id"],
            field=f"removed_redundancy:{row['direction_id']}",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )
    _require_artifact(
        quotient["duality_canonicalization_artifact_id"],
        field="duality_canonicalization_artifact_id",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )

    constraint_ids: set[str] = set()
    if not constraint_registry["complete"]:
        blockers.append("constraint_registry_incomplete")
    if not constraint_registry["frozen_before_solve"]:
        blockers.append("constraint_registry_not_frozen_before_solve")
    if constraint_registry["registry_sha256"] != stable_json_hash(constraint_registry["rows"]):
        blockers.append("constraint_registry_hash_mismatch")
    for row in constraint_registry["rows"]:
        constraint_id = row["constraint_id"]
        if constraint_id in constraint_ids:
            blockers.append(f"duplicate_constraint_id:{constraint_id}")
        constraint_ids.add(constraint_id)
        _require_artifact(
            row["definition_artifact_id"],
            field=f"constraint_definition:{constraint_id}",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )

    target_row_ids: set[str] = set()
    forbidden_selected_rows: set[str] = set()
    _require_artifact(
        target["precommit_artifact_id"],
        field="target_registry_precommitment",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    for row in target["rows"]:
        if row["row_id"] in target_row_ids:
            blockers.append(f"duplicate_target_row_id:{row['row_id']}")
        target_row_ids.add(row["row_id"])
        if row["required_for_candidate_pass"] and row["role"] in {"candidate_only", "diagnostic"}:
            blockers.append(f"self_targeting_or_diagnostic_required_row:{row['row_id']}")
        if row["role"] in {"candidate_only", "diagnostic"}:
            forbidden_selected_rows.add(row["row_id"])
        if row["covariance_artifact_id"] is not None:
            _require_artifact(
                row["covariance_artifact_id"],
                field=f"target_covariance:{row['row_id']}",
                artifacts=artifacts,
                verified_paths=verified_paths,
                blockers=blockers,
            )
    if not target["complete"]:
        blockers.append("target_registry_incomplete")
    if not target["common_scheme"]:
        blockers.append("target_registry_has_no_common_scheme")
    if not target["source_frozen_before_target_load"]:
        blockers.append("source_target_separation_not_established")
    if provenance["target_registry_sha256"] != stable_json_hash(target):
        blockers.append("target_registry_hash_mismatch")
    if provenance["source_dirty"]:
        blockers.append("source_worktree_dirty")

    for gate_id, gate in gates.items():
        if gate["status"] == "PASS" and not gate["evidence_artifact_ids"]:
            blockers.append(f"passing_gate_has_no_evidence:{gate_id}")
        if gate["status"] != "PASS" and not gate["blockers"]:
            blockers.append(f"nonpassing_gate_has_no_blocker:{gate_id}")
        for artifact_id in gate["evidence_artifact_ids"]:
            _require_artifact(
                artifact_id,
                field=f"gate:{gate_id}",
                artifacts=artifacts,
                verified_paths=verified_paths,
                blockers=blockers,
            )

    stability = payload["stability"]
    if gates["vacuum_stability"]["status"] == "PASS":
        _require_artifact(
            stability["evidence_artifact_id"],
            field="stability_evidence",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )
        if stability["background_class"] == "NOT_SUPPLIED" or stability["stability_criterion"] == "NOT_SUPPLIED":
            blockers.append("passing_stability_gate_has_no_stability_criterion")
        if stability["stationarity_residual_intervals"] is None:
            blockers.append("passing_stability_gate_has_no_stationarity_enclosure")
        if stability["physical_hessian_lower_bound"] is None:
            blockers.append("passing_stability_gate_has_no_hessian_bound")
        if stability["stability_threshold"] is None:
            blockers.append("passing_stability_gate_has_no_stability_threshold")
        _require_artifact(
            stability["control_parameters_artifact_id"],
            field="stability_control_parameters",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )
        residuals = stability["stationarity_residual_intervals"] or []
        if any(
            not (Fraction(row["lower"]) <= 0 <= Fraction(row["upper"]))
            for row in residuals
        ):
            blockers.append("stationarity_enclosure_excludes_zero")
        if (
            stability["physical_hessian_lower_bound"] is not None
            and stability["stability_threshold"] is not None
        ):
            lower_bound = Fraction(stability["physical_hessian_lower_bound"])
            threshold = Fraction(stability["stability_threshold"])
            if lower_bound <= threshold:
                blockers.append("physical_hessian_does_not_clear_stability_threshold")
            if stability["background_class"] in {"MINKOWSKI", "DE_SITTER"}:
                if stability["stability_criterion"] != "POSITIVE_PHYSICAL_HESSIAN" or threshold != 0:
                    blockers.append("minkowski_or_desitter_stability_rule_mismatch")
            elif stability["background_class"] == "ANTI_DE_SITTER":
                if stability["stability_criterion"] != "BREITENLOHNER_FREEDMAN":
                    blockers.append("ads_stability_rule_mismatch")

    augmented = payload["augmented_system"]
    expected_coordinate_hash = stable_json_hash(quotient["physical_coordinate_ids"])
    expected_row_hash = stable_json_hash(augmented["selected_square_row_ids"])
    expected_row_registry_hash = stable_json_hash(augmented["row_registry"])
    if augmented["status"] == "SUPPLIED":
        if augmented["coordinate_order_sha256"] != expected_coordinate_hash:
            blockers.append("augmented_coordinate_order_hash_mismatch")
        if augmented["row_order_sha256"] != expected_row_hash:
            blockers.append("augmented_row_order_hash_mismatch")
        if augmented["row_registry_sha256"] != expected_row_registry_hash:
            blockers.append("augmented_row_registry_hash_mismatch")
        augmented_rows: dict[str, dict[str, Any]] = {}
        for row in augmented["row_registry"]:
            row_id = row["row_id"]
            if row_id in augmented_rows:
                blockers.append(f"duplicate_augmented_row_id:{row_id}")
            augmented_rows[row_id] = row
            source_row_id = row["source_row_id"]
            if row["source_kind"] == "completion_constraint":
                if source_row_id not in constraint_ids:
                    blockers.append(f"unknown_augmented_constraint_row:{row_id}:{source_row_id}")
            elif source_row_id not in target_row_ids:
                blockers.append(f"unknown_augmented_target_row:{row_id}:{source_row_id}")
        if set(augmented["all_row_ids"]) != set(augmented_rows):
            blockers.append("augmented_all_row_ids_do_not_match_row_registry")
        for row_id in augmented["selected_square_row_ids"]:
            binding = augmented_rows.get(row_id)
            if binding is None:
                blockers.append(f"selected_augmented_row_unregistered:{row_id}")
            elif (
                binding["source_kind"] == "precommitted_target"
                and binding["source_row_id"] in forbidden_selected_rows
            ):
                blockers.append(f"inadmissible_target_rank_row:{row_id}")
        _require_artifact(
            augmented["enclosure_proof_artifact_id"],
            field="augmented_enclosure_proof",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )
        _require_artifact(
            augmented["full_system_closure"]["evidence_artifact_id"],
            field="augmented_full_system_closure",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )
    interval_report = verify_interval_contraction(augmented, physical_dimension=dimension)
    if gates["augmented_local_isolation"]["status"] == "PASS" and not interval_report[
        "interval_contraction_receipt"
    ]:
        blockers.append("passing_augmented_isolation_gate_failed_interval_audit")

    flat_rows_structurally_classified = True
    flat_scientific_blockers: list[str] = []
    visible_flat_direction = False
    unclassified_flat_direction = False
    flat_proof_levels = {
        "OPH_INVISIBLE_QUOTIENTED": {"GROUP_ORBIT"},
        "STABILIZED": {"MASS_BOUND"},
        "VISIBLE_FLAT": {"EXACT_CURVE", "CONSTANT_RANK"},
        "UNCLASSIFIED": {"INFINITESIMAL_ONLY", "EXACT_CURVE", "CONSTANT_RANK"},
    }
    seen_flat_direction_ids: set[str] = set()
    for row in payload["flat_directions"]:
        direction_id = row["direction_id"]
        if direction_id in seen_flat_direction_ids:
            flat_rows_structurally_classified = False
            blockers.append(f"duplicate_flat_direction_id:{direction_id}")
        seen_flat_direction_ids.add(direction_id)
        _require_artifact(
            row["evidence_artifact_id"],
            field=f"flat_direction:{direction_id}",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=blockers,
        )
        if len(row["basis_vector"]) != dimension:
            flat_rows_structurally_classified = False
            blockers.append(f"flat_direction_basis_dimension_mismatch:{direction_id}")
        elif not any(Fraction(value) != 0 for value in row["basis_vector"]):
            flat_rows_structurally_classified = False
            blockers.append(f"flat_direction_zero_basis_vector:{direction_id}")
        if row["proof_level"] not in flat_proof_levels[row["classification"]]:
            flat_rows_structurally_classified = False
            blockers.append(f"flat_direction_proof_level_mismatch:{direction_id}")
        if row["classification"] in {"VISIBLE_FLAT", "UNCLASSIFIED"}:
            flat_rows_structurally_classified = False
            flat_scientific_blockers.append(
                f"unresolved_physical_flat_direction:{direction_id}"
            )
            visible_flat_direction |= row["classification"] == "VISIBLE_FLAT"
            unclassified_flat_direction |= row["classification"] == "UNCLASSIFIED"

    integrity_blockers = blockers[:]

    # Gate semantics cannot be inferred from producer labels or matching hashes.
    semantic_gate_statuses: dict[str, str] = {}
    semantic_gate_receipts: dict[str, bool] = {}
    semantic_blockers: list[str] = []
    for gate_id in CANDIDATE_GATE_IDS:
        verifier = SEMANTIC_GATE_VERIFIERS.get(gate_id)
        if verifier is None:
            semantic_gate_statuses[gate_id] = "INCONCLUSIVE"
            semantic_gate_receipts[gate_id] = False
            if gates[gate_id]["status"] == "PASS":
                semantic_blockers.append(f"semantic_gate_verifier_not_implemented:{gate_id}")
            continue
        try:
            verified_status = verifier(payload, bundle_root or REPO_ROOT)
            if verified_status not in {"PASS", "FAIL", "INCONCLUSIVE"}:
                raise ValueError("semantic gate verifier returned an invalid status")
            semantic_gate_statuses[gate_id] = verified_status
            semantic_gate_receipts[gate_id] = verified_status == "PASS"
        except Exception as exc:  # pragma: no cover - future verifier firewall
            semantic_gate_statuses[gate_id] = "INCONCLUSIVE"
            semantic_gate_receipts[gate_id] = False
            semantic_blockers.append(f"semantic_gate_verifier_error:{gate_id}:{type(exc).__name__}")

    aggregate = _aggregate_receipts(semantic_gate_receipts)
    contract_integrity = not integrity_blockers
    artifact_hash_receipt = not any(
        _is_artifact_blocker(blocker) for blocker in integrity_blockers
    )
    if not contract_integrity:
        aggregate = {name: False for name in aggregate}
    oph_target_registry_binding = bool(
        contract_integrity and semantic_gate_receipts["source_target_separation"]
    )
    evaluator_enclosure = bool(
        contract_integrity and semantic_gate_receipts["augmented_local_isolation"]
    )
    interval_contraction_algebra = bool(
        contract_integrity
        and artifact_hash_receipt
        and interval_report["interval_contraction_receipt"]
    )
    full_system_closure = bool(
        evaluator_enclosure
        and interval_report.get("full_system_closure_method")
        in {"selected_rows_generate_full_system", "independent_exact_full_solution"}
    )
    flat_direction_classification = bool(
        contract_integrity
        and flat_rows_structurally_classified
        and semantic_gate_receipts["vacuum_stability"]
        and semantic_gate_receipts["physical_quotient_descent"]
    )
    physical_local_isolation = bool(
        evaluator_enclosure
        and full_system_closure
        and interval_contraction_algebra
    )
    moduli_locking_receipt = bool(
        aggregate["STRING_MODULI_LOCKING_RECEIPT"]
        and oph_target_registry_binding
        and physical_local_isolation
        and flat_direction_classification
    )
    aggregate["STRING_MODULI_LOCKING_RECEIPT"] = moduli_locking_receipt
    aggregate["BD_MODULI_LOCKING_CERTIFICATE_RECEIPT"] = moduli_locking_receipt
    semantic_candidate_pass = all(semantic_gate_receipts.values())
    if not contract_integrity:
        candidate_status = "INVALID"
    elif (
        any(status == "FAIL" for status in semantic_gate_statuses.values())
        or visible_flat_direction
    ):
        candidate_status = "FAIL"
    elif unclassified_flat_direction:
        candidate_status = "INCONCLUSIVE"
    elif (
        semantic_candidate_pass
        and moduli_locking_receipt
        and aggregate["STRING_CANDIDATE_CONSISTENCY_RECEIPT"]
        and not payload["blockers"]
    ):
        candidate_status = "PASS"
    else:
        candidate_status = "INCONCLUSIVE"
    aggregate["LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT"] = candidate_status == "PASS"
    report = {
        "artifact": "oph_string_vacuum_candidate_verification",
        "report_schema_version": 1,
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "oph_equivalence_class_id": candidate["oph_equivalence_class_id"],
            "theory_family": candidate["theory_family"],
            "branch_definition_sha256": candidate["branch_definition_sha256"],
        },
        "candidate_status": candidate_status,
        "receipt_target_registry_sha256": receipt_target_registry_sha256(),
        "receipt_subject": _candidate_receipt_subject(payload),
        "contract_integrity_receipt": contract_integrity,
        "artifact_hash_receipt": artifact_hash_receipt,
        "oph_target_registry_binding_receipt": oph_target_registry_binding,
        "interval_contraction_algebra_receipt": interval_contraction_algebra,
        "evaluator_enclosure_receipt": evaluator_enclosure,
        "full_system_closure_receipt": full_system_closure,
        "flat_direction_classification_receipt": flat_direction_classification,
        "physical_local_isolation_receipt": physical_local_isolation,
        "selector_independence_receipt": False,
        "producer_gate_hints": {gate_id: gates[gate_id]["status"] for gate_id in CANDIDATE_GATE_IDS},
        "semantic_gate_statuses": semantic_gate_statuses,
        "semantic_gate_receipts": semantic_gate_receipts,
        "interval_algebra": interval_report,
        "receipts": aggregate,
        "blockers": sorted(
            set(
                integrity_blockers
                + semantic_blockers
                + flat_scientific_blockers
                + payload["blockers"]
                + interval_report.get("blockers", [])
            )
        ),
        "nonclaims": [
            "branch-global uniqueness",
            "catalogue-relative uniqueness",
            "globally correct string theory",
        ],
    }
    report["report_sha256"] = stable_json_hash(report)
    return report


def _load_verified_candidate_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "artifact",
        "report_schema_version",
        "candidate",
        "candidate_status",
        "receipts",
        "report_sha256",
    }
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError("candidate_verification_report_shape_invalid")
    if payload["artifact"] != "oph_string_vacuum_candidate_verification":
        raise ValueError("candidate_verification_report_artifact_invalid")
    supplied_hash = payload["report_sha256"]
    without_hash = dict(payload)
    without_hash.pop("report_sha256")
    if supplied_hash != stable_json_hash(without_hash):
        raise ValueError("candidate_verification_report_internal_hash_mismatch")
    return payload


def _verify_catalogue_proof(
    proof_kind: str,
    payload: dict[str, Any],
    row: dict[str, Any] | None,
    *,
    bundle_root: Path | None,
    blockers: list[str],
) -> bool:
    """Run a code-owned semantic verifier for one catalogue proof obligation."""

    verifier = CATALOGUE_PROOF_VERIFIERS.get(proof_kind)
    if verifier is None:
        blockers.append(f"semantic_catalogue_verifier_not_implemented:{proof_kind}")
        return False
    try:
        result = verifier(payload, row, bundle_root or REPO_ROOT)
        if type(result) is not bool:
            raise ValueError("catalogue proof verifier must return an exact Boolean")
        return result
    except Exception as exc:  # pragma: no cover - future verifier firewall
        blockers.append(f"semantic_catalogue_verifier_error:{proof_kind}:{type(exc).__name__}")
        return False


def _catalogue_receipt_subject(payload: Mapping[str, Any]) -> dict[str, Any]:
    subject = {
        "subject_type": "string_vacuum_catalogue",
        "catalogue_id": payload["catalogue_id"],
        "declared_scope_sha256": stable_json_hash(payload["declared_scope"]),
        "equivalence_relation_sha256": payload["equivalence_partition"][
            "relation_definition_sha256"
        ],
        "receipt_target_registry_sha256": receipt_target_registry_sha256(),
    }
    subject["subject_scope_sha256"] = stable_json_hash(subject)
    return subject


def verify_catalogue_evidence(
    payload: dict[str, Any],
    *,
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """Verify catalogue coverage and derive only catalogue-scoped selection."""

    blockers = _schema_errors(payload, CATALOGUE_SCHEMA_PATH)
    report: dict[str, Any] = {
        "artifact": "oph_string_vacuum_catalogue_verification",
        "report_schema_version": 1,
        "receipt_target_registry_sha256": receipt_target_registry_sha256(),
        "receipt_subject": None,
        "catalogue_id": payload.get("catalogue_id") if isinstance(payload, dict) else None,
        "catalogue_status": "INVALID" if blockers else "INCONCLUSIVE",
        "selected_equivalence_class_id": None,
        "selection_scope": None,
        "candidate_rows": [],
        "branch_rows": [],
        "passing_equivalence_class_ids": [],
        "catalogue_enumeration_receipt": False,
        "equivalence_partition_receipt": False,
        "candidate_replay_receipt": False,
        "branch_global_coverage_receipt": False,
        "branch_verdict_replay_receipt": False,
        "COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT": False,
        "unrestricted_string_universe_coverage_receipt": False,
        "OPH_NATIVE_STRING_VACUUM_RECEIPT": False,
        "selector_independence_receipt": False,
        "catalogue_blockers": blockers[:],
        "unrestricted_blockers": [],
        "blockers": blockers[:],
    }
    if blockers:
        report["report_sha256"] = stable_json_hash(report)
        return report

    scope = payload["declared_scope"]
    equivalence = payload["equivalence_partition"]
    unrestricted_blockers: list[str] = []
    artifacts, verified_paths, artifact_blockers = _verify_artifacts(
        payload["artifacts"], bundle_root
    )
    reduction_artifact_id = (
        scope["reduction_theorem_artifact_id"]
        if scope["unrestricted_string_universe_covered"]
        else None
    )
    non_unrestricted_artifact_ids = {
        scope["enumeration_proof_artifact_id"],
        equivalence["partition_proof_artifact_id"],
        *(
            row["branch_domain_coverage_artifact_id"]
            for row in payload["candidate_reports"]
        ),
        *(row["branch_verdict_ledger_artifact_id"] for row in payload["candidate_reports"]),
    }
    reduction_is_unrestricted_only = bool(
        reduction_artifact_id
        and reduction_artifact_id not in non_unrestricted_artifact_ids
    )
    for blocker in artifact_blockers:
        if (
            reduction_is_unrestricted_only
            and reduction_artifact_id is not None
            and _artifact_blocker_targets_id(blocker, reduction_artifact_id)
        ):
            unrestricted_blockers.append(blocker)
        else:
            blockers.append(blocker)

    enumeration_artifact_ok = _require_artifact(
        scope["enumeration_proof_artifact_id"],
        field="enumeration_proof",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    equivalence_artifact_ok = _require_artifact(
        equivalence["partition_proof_artifact_id"],
        field="equivalence_partition_proof",
        artifacts=artifacts,
        verified_paths=verified_paths,
        blockers=blockers,
    )
    enumeration_semantics = _verify_catalogue_proof(
        "catalogue_enumeration",
        payload,
        None,
        bundle_root=bundle_root,
        blockers=blockers,
    )
    enumeration_receipt = bool(
        enumeration_artifact_ok
        and not _artifact_id_is_ambiguous(
            artifact_blockers, scope["enumeration_proof_artifact_id"]
        )
        and enumeration_semantics
    )
    if not enumeration_receipt:
        blockers.append("catalogue_enumeration_not_semantically_verified")
    equivalence_semantics = _verify_catalogue_proof(
        "equivalence_partition",
        payload,
        None,
        bundle_root=bundle_root,
        blockers=blockers,
    )
    equivalence_receipt = bool(
        equivalence_artifact_ok
        and not _artifact_id_is_ambiguous(
            artifact_blockers, equivalence["partition_proof_artifact_id"]
        )
        and equivalence_semantics
    )
    if not equivalence_receipt:
        blockers.append("equivalence_partition_not_semantically_verified")
    unrestricted_reduction_receipt = False
    if scope["unrestricted_string_universe_covered"]:
        reduction_artifact_ok = _require_artifact(
            scope["reduction_theorem_artifact_id"],
            field="unrestricted_reduction_theorem",
            artifacts=artifacts,
            verified_paths=verified_paths,
            blockers=unrestricted_blockers,
        )
        unrestricted_reduction_semantics = _verify_catalogue_proof(
            "unrestricted_reduction",
            payload,
            None,
            bundle_root=bundle_root,
            blockers=unrestricted_blockers,
        )
        unrestricted_reduction_receipt = bool(
            reduction_artifact_ok
            and not _artifact_id_is_ambiguous(
                artifact_blockers, scope["reduction_theorem_artifact_id"]
            )
            and unrestricted_reduction_semantics
        )
        if not unrestricted_reduction_receipt:
            unrestricted_blockers.append("unrestricted_reduction_not_semantically_verified")
    if not scope["catalogue_complete_within_scope"]:
        blockers.append("catalogue_not_complete_within_declared_scope")
    if payload["provenance"]["source_dirty"]:
        blockers.append("catalogue_source_worktree_dirty")
    if payload["unresolved_regions"]:
        blockers.append("catalogue_has_unresolved_regions")

    passing_classes: set[str] = set()
    inconclusive = False
    candidate_rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    if bundle_root is None:
        blockers.append("catalogue_bundle_root_not_supplied")
    for row in payload["candidate_reports"]:
        candidate_id = row["candidate_id"]
        if candidate_id in seen_candidates:
            blockers.append(f"duplicate_catalogue_candidate:{candidate_id}")
        seen_candidates.add(candidate_id)
        producer_branch_status = row["branch_domain_coverage_status"]
        verified_branch_status = "INCONCLUSIVE"
        branch_verdict_semantics = False
        if producer_branch_status in {"PASS", "FAIL"}:
            branch_coverage_artifact_ok = _require_artifact(
                row["branch_domain_coverage_artifact_id"],
                field=f"branch_domain_coverage:{candidate_id}",
                artifacts=artifacts,
                verified_paths=verified_paths,
                blockers=blockers,
            )
            branch_semantics = _verify_catalogue_proof(
                "branch_domain_coverage",
                payload,
                row,
                bundle_root=bundle_root,
                blockers=blockers,
            )
            if (
                branch_coverage_artifact_ok
                and not _artifact_id_is_ambiguous(
                    artifact_blockers, row["branch_domain_coverage_artifact_id"]
                )
                and branch_semantics
            ):
                verified_branch_status = producer_branch_status
            else:
                blockers.append(f"branch_domain_coverage_not_semantically_verified:{candidate_id}")
            branch_verdict_artifact_ok = _require_artifact(
                row["branch_verdict_ledger_artifact_id"],
                field=f"branch_verdict_ledger:{candidate_id}",
                artifacts=artifacts,
                verified_paths=verified_paths,
                blockers=blockers,
            )
            branch_verdict_semantics = bool(
                branch_verdict_artifact_ok
                and not _artifact_id_is_ambiguous(
                    artifact_blockers, row["branch_verdict_ledger_artifact_id"]
                )
                and _verify_catalogue_proof(
                "branch_verdict_replay",
                payload,
                row,
                bundle_root=bundle_root,
                blockers=blockers,
                )
            )
            if not branch_verdict_semantics:
                blockers.append(f"branch_verdict_not_semantically_verified:{candidate_id}")
        if verified_branch_status != "PASS":
            inconclusive = True
        loaded_status = "INVALID"
        loaded_branch_definition_sha256: str | None = None
        row_candidate_replay_receipt = False
        if bundle_root is not None:
            try:
                evidence_path = _resolve_regular_file(bundle_root, row["evidence_packet_path"])
                if _sha256_file(evidence_path) != row["evidence_packet_sha256"]:
                    raise ValueError("candidate_evidence_file_hash_mismatch")
                candidate_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if not isinstance(candidate_evidence, dict):
                    raise ValueError("candidate_evidence_not_object")
                recomputed_report = verify_candidate_evidence(
                    candidate_evidence,
                    bundle_root=bundle_root,
                )
                report_path = _resolve_regular_file(bundle_root, row["verification_report_path"])
                if _sha256_file(report_path) != row["verification_report_sha256"]:
                    raise ValueError("candidate_report_file_hash_mismatch")
                candidate_report = _load_verified_candidate_report(report_path)
                if candidate_report != recomputed_report:
                    raise ValueError("candidate_verification_report_not_reproducible")
                if candidate_report["candidate"]["candidate_id"] != candidate_id:
                    raise ValueError("candidate_report_id_mismatch")
                if (
                    candidate_report["candidate"]["oph_equivalence_class_id"]
                    != row["oph_equivalence_class_id"]
                ):
                    raise ValueError("candidate_report_equivalence_class_mismatch")
                loaded_status = candidate_report["candidate_status"]
                loaded_branch_definition_sha256 = candidate_report["candidate"][
                    "branch_definition_sha256"
                ]
                row_candidate_replay_receipt = True
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"candidate_report_invalid:{candidate_id}:{exc}")
        verified_branch_verdict = (
            loaded_status
            if (
                verified_branch_status == "PASS"
                and branch_verdict_semantics
                and loaded_status in {"PASS", "FAIL"}
            )
            else "INCONCLUSIVE"
        )
        if verified_branch_verdict == "PASS":
            passing_classes.add(row["oph_equivalence_class_id"])
        elif verified_branch_verdict == "INCONCLUSIVE":
            inconclusive = True
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "oph_equivalence_class_id": row["oph_equivalence_class_id"],
                "branch_definition_sha256": loaded_branch_definition_sha256,
                "candidate_replay_receipt": row_candidate_replay_receipt,
                "verified_candidate_status": loaded_status,
                "producer_branch_domain_coverage_status": producer_branch_status,
                "verified_branch_domain_coverage_status": verified_branch_status,
                "verified_branch_verdict_status": verified_branch_verdict,
                "branch_global_uniqueness_receipt": verified_branch_verdict == "PASS",
            }
        )

    candidate_replay_receipt = bool(
        candidate_rows
        and all(row["candidate_replay_receipt"] for row in candidate_rows)
    )
    branch_global_coverage_receipt = bool(
        candidate_replay_receipt
        and all(
            row["verified_branch_domain_coverage_status"] == "PASS"
            for row in candidate_rows
        )
    )
    branch_verdict_replay_receipt = bool(
        branch_global_coverage_receipt
        and all(
            row["verified_branch_verdict_status"] in {"PASS", "FAIL"}
            for row in candidate_rows
        )
    )
    branch_groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        branch_key = row["branch_definition_sha256"] or f"unverified:{row['candidate_id']}"
        branch_groups.setdefault(branch_key, []).append(row)
    branch_rows: list[dict[str, Any]] = []
    for branch_key, rows in sorted(branch_groups.items()):
        branch_candidate_replay = bool(
            rows and all(row["candidate_replay_receipt"] for row in rows)
        )
        branch_coverage = bool(
            branch_candidate_replay
            and all(
                row["verified_branch_domain_coverage_status"] == "PASS" for row in rows
            )
        )
        branch_terminal = bool(
            branch_coverage
            and all(
                row["verified_branch_verdict_status"] in {"PASS", "FAIL"} for row in rows
            )
        )
        branch_passing_classes = sorted(
            {
                row["oph_equivalence_class_id"]
                for row in rows
                if row["verified_branch_verdict_status"] == "PASS"
            }
        )
        branch_unique = branch_terminal and len(branch_passing_classes) == 1
        for row in rows:
            row["branch_global_uniqueness_receipt"] = branch_unique
        branch_rows.append(
            {
                "branch_definition_sha256": branch_key,
                "candidate_ids": sorted(row["candidate_id"] for row in rows),
                "candidate_replay_receipt": branch_candidate_replay,
                "passing_equivalence_class_ids": branch_passing_classes,
                "branch_domain_coverage_receipt": branch_coverage,
                "branch_verdict_replay_receipt": branch_terminal,
                "branch_global_uniqueness_receipt": branch_unique,
            }
        )

    complete = bool(
        enumeration_receipt
        and equivalence_receipt
        and candidate_replay_receipt
        and branch_global_coverage_receipt
        and branch_verdict_replay_receipt
        and not blockers
        and not inconclusive
    )
    unique = complete and len(passing_classes) == 1
    if unique:
        selected = next(iter(passing_classes))
        catalogue_status = "SELECTED_WITHIN_DECLARED_CATALOGUE"
        selection_scope = payload["declared_scope"]["scope_statement"]
    elif complete and not passing_classes:
        selected = None
        catalogue_status = "NO_PASSING_CLASS"
        selection_scope = payload["declared_scope"]["scope_statement"]
    else:
        selected = None
        catalogue_status = "INCONCLUSIVE"
        selection_scope = None

    report.update(
        {
            "catalogue_status": catalogue_status,
            "receipt_subject": _catalogue_receipt_subject(payload),
            "candidate_rows": candidate_rows,
            "branch_rows": branch_rows,
            "passing_equivalence_class_ids": sorted(passing_classes),
            "selected_equivalence_class_id": selected,
            "selection_scope": selection_scope,
            "catalogue_enumeration_receipt": enumeration_receipt,
            "equivalence_partition_receipt": equivalence_receipt,
            "candidate_replay_receipt": candidate_replay_receipt,
            "branch_global_coverage_receipt": branch_global_coverage_receipt,
            "branch_verdict_replay_receipt": branch_verdict_replay_receipt,
            "COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT": unique,
            "unrestricted_string_universe_coverage_receipt": bool(
                unrestricted_reduction_receipt and not unrestricted_blockers
            ),
            "OPH_NATIVE_STRING_VACUUM_RECEIPT": bool(
                unique and unrestricted_reduction_receipt and not unrestricted_blockers
            ),
            "catalogue_blockers": sorted(set(blockers)),
            "unrestricted_blockers": sorted(set(unrestricted_blockers)),
            "blockers": sorted(set(blockers + unrestricted_blockers)),
        }
    )
    report["report_sha256"] = stable_json_hash(report)
    return report
