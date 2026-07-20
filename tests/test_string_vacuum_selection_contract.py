from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import oph_fpe.string_vacuum.contract as contract_module
from oph_fpe.evidence.hashes import stable_json_hash
from oph_fpe.string_vacuum.cli import main as string_vacuum_cli
from oph_fpe.string_vacuum.contract import (
    CANDIDATE_SCHEMA_PATH,
    CATALOGUE_SCHEMA_PATH,
    CANDIDATE_GATE_IDS,
    verify_candidate_evidence,
    verify_catalogue_evidence,
)
from oph_fpe.string_vacuum.receipt_targets import (
    CANONICAL_CANDIDATE_RECEIPT_IDS,
    OBSERVABLE_TARGETS_SCHEMA_PATH,
    RECEIPT_TARGETS_SCHEMA_PATH,
    ReceiptTargetRegistryError,
    observable_target_registry,
    receipt_target_registry,
    validate_observable_target_registry,
    validate_receipt_target_registry,
)
from oph_fpe.string_vacuum.verified_numerics import verify_interval_contraction


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(root: Path, name: str, payload: dict) -> dict:
    path = root / name
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact_id": name.removesuffix(".json"),
        "path": name,
        "sha256": _sha256(path),
        "format": "json",
        "semantic_role": name.removesuffix(".json"),
    }


def _interval(lower: str, upper: str) -> dict[str, str]:
    return {"lower": lower, "upper": upper}


def _augmented_system(artifacts: dict[str, dict]) -> dict:
    coordinates = ["physical_x"]
    rows = ["target_x"]
    row_registry = [
        {
            "row_id": "target_x",
            "source_kind": "precommitted_target",
            "source_row_id": "target_x",
        }
    ]
    return {
        "status": "SUPPLIED",
        "coordinate_order_sha256": stable_json_hash(coordinates),
        "row_order_sha256": stable_json_hash(rows),
        "row_registry_sha256": stable_json_hash(row_registry),
        "all_row_ids": rows,
        "row_registry": row_registry,
        "selected_square_row_ids": rows,
        "box": [_interval("-1", "1")],
        "center": ["0"],
        "residual_interval_at_center": [_interval("0", "0")],
        "jacobian_interval": [[_interval("1", "1")]],
        "preconditioner": [["1"]],
        "enclosure_proof_artifact_id": artifacts["interval_proof"]["artifact_id"],
        "full_system_closure": {
            "method": "selected_rows_generate_full_system",
            "evidence_artifact_id": artifacts["closure_proof"]["artifact_id"],
        },
    }


def _candidate_packet(root: Path) -> dict:
    artifact_rows = [
        _write_artifact(root, "branch_definition.json", {"branch": "unit branch"}),
        _write_artifact(root, "dependency_lock.json", {"dependencies": []}),
        _write_artifact(root, "environment.json", {"environment": "unit"}),
        _write_artifact(root, "source_freeze.json", {"source": "unit source"}),
        _write_artifact(root, "target_precommit.json", {"registry": "unit_target_registry_v1"}),
        _write_artifact(root, "coordinate_registry.json", {"physical_coordinate_ids": ["physical_x"]}),
        _write_artifact(root, "duality_canonicalization.json", {"canonical_class": "unit_class"}),
        _write_artifact(root, "constraint_definition.json", {"constraint": "unit physical constraint"}),
        _write_artifact(root, "quotient_proof.json", {"proof": "quotient descent"}),
        _write_artifact(root, "gate_proof.json", {"proof": "upstream gate hints only"}),
        _write_artifact(root, "interval_proof.json", {"proof": "interval evaluator enclosure"}),
        _write_artifact(root, "closure_proof.json", {"proof": "selected rows generate full system"}),
    ]
    artifacts = {row["artifact_id"]: row for row in artifact_rows}
    gate_artifact_id = artifacts["gate_proof"]["artifact_id"]
    target_registry = {
        "registry_id": "unit_target_registry",
        "registry_version": 1,
        "authority": "code_owned_allowlisted_registry",
        "precommit_artifact_id": artifacts["target_precommit"]["artifact_id"],
        "complete": True,
        "common_scheme": True,
        "source_frozen_before_target_load": True,
        "rows": [
            {
                "row_id": "target_x",
                "role": "empirical_acceptance",
                "required_for_candidate_pass": True,
                "units": "dimensionless",
                "scheme": "unit_test_scheme",
                "scale": "unit_test_scale",
                "target_interval": _interval("0", "0"),
                "covariance_artifact_id": None,
            }
        ],
    }
    constraint_rows = [
        {
            "constraint_id": "unit_constraint",
            "kind": "other_physical",
            "definition_artifact_id": artifacts["constraint_definition"]["artifact_id"],
        }
    ]
    return {
        "$schema": "schemas/string_vacuum_candidate_evidence_v1.schema.json",
        "artifact": "oph_string_vacuum_candidate_evidence",
        "schema_version": 1,
        "candidate": {
            "candidate_id": "unit_candidate",
            "presentation_id": "unit_presentation",
            "oph_equivalence_class_id": "unit_class",
            "candidate_class_id": "unit_catalogue",
            "theory_family": "unit_test",
            "branch_definition_sha256": artifacts["branch_definition"]["sha256"],
            "branch_definition_artifact_id": artifacts["branch_definition"]["artifact_id"],
        },
        "provenance": {
            "run_id": "unit_run",
            "producer_commit": "a" * 40,
            "verifier_commit": "b" * 40,
            "source_dirty": False,
            "dependency_lock_sha256": artifacts["dependency_lock"]["sha256"],
            "dependency_lock_artifact_id": artifacts["dependency_lock"]["artifact_id"],
            "environment_sha256": artifacts["environment"]["sha256"],
            "environment_artifact_id": artifacts["environment"]["artifact_id"],
            "source_freeze_sha256": artifacts["source_freeze"]["sha256"],
            "source_freeze_artifact_id": artifacts["source_freeze"]["artifact_id"],
            "target_registry_sha256": stable_json_hash(target_registry),
            "arithmetic": {"backend": "exact-rational-test", "precision_bits": 256, "rounding": "outward"},
            "discovery_randomness": {
                "used": False,
                "seed_manifest_sha256": None,
                "certification_role": "discovery_only",
            },
            "replay_argv": ["oph-string-vacuum", "verify-candidate", "candidate.json"],
        },
        "artifacts": artifact_rows,
        "presentation_and_quotient": {
            "prequotient_dimension_real": 2,
            "physical_coordinate_ids": ["physical_x"],
            "coordinate_registry_sha256": artifacts["coordinate_registry"]["sha256"],
            "coordinate_registry_artifact_id": artifacts["coordinate_registry"]["artifact_id"],
            "removed_redundancies": [],
            "quotient_descent_proof_artifact_id": artifacts["quotient_proof"]["artifact_id"],
            "duality_canonicalization_artifact_id": artifacts["duality_canonicalization"]["artifact_id"],
        },
        "constraint_registry": {
            "complete": True,
            "frozen_before_solve": True,
            "registry_sha256": stable_json_hash(constraint_rows),
            "rows": constraint_rows,
        },
        "target_registry": target_registry,
        "gates": {
            gate_id: {"status": "PASS", "evidence_artifact_ids": [gate_artifact_id], "blockers": []}
            for gate_id in CANDIDATE_GATE_IDS
        },
        "stability": {
            "background_class": "MINKOWSKI",
            "stationarity_residual_intervals": [_interval("0", "0")],
            "physical_hessian_lower_bound": "1",
            "stability_threshold": "0",
            "stability_criterion": "POSITIVE_PHYSICAL_HESSIAN",
            "control_parameters_artifact_id": gate_artifact_id,
            "evidence_artifact_id": gate_artifact_id,
        },
        "augmented_system": _augmented_system(artifacts),
        "flat_directions": [],
        "blockers": [],
        "producer_classification": "EVIDENCE_COMPLETE",
    }


def _catalogue_packet(root: Path, packets: list[dict], *, prefix: str) -> dict:
    catalogue_artifacts = [
        _write_artifact(root, f"{prefix}_enumeration.json", {"proof": "complete"}),
        _write_artifact(root, f"{prefix}_partition.json", {"proof": "partition"}),
        _write_artifact(root, f"{prefix}_cover.json", {"proof": "covered"}),
        _write_artifact(root, f"{prefix}_verdict.json", {"proof": "terminal ledger"}),
    ]
    candidate_rows = []
    for index, packet in enumerate(packets):
        candidate_path = root / f"{prefix}_candidate_{index}.json"
        candidate_path.write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        candidate_report = verify_candidate_evidence(packet, bundle_root=root)
        report_path = root / f"{prefix}_candidate_{index}_verification.json"
        report_path.write_text(
            json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        candidate_rows.append(
            {
                "candidate_id": packet["candidate"]["candidate_id"],
                "oph_equivalence_class_id": packet["candidate"][
                    "oph_equivalence_class_id"
                ],
                "evidence_packet_path": candidate_path.name,
                "evidence_packet_sha256": _sha256(candidate_path),
                "verification_report_path": report_path.name,
                "verification_report_sha256": _sha256(report_path),
                "branch_domain_coverage_status": "PASS",
                "branch_domain_coverage_artifact_id": f"{prefix}_cover",
                "branch_verdict_ledger_artifact_id": f"{prefix}_verdict",
                "producer_status_hint": "PASS",
            }
        )
    return {
        "$schema": "schemas/string_vacuum_catalogue_evidence_v1.schema.json",
        "artifact": "oph_string_vacuum_catalogue_evidence",
        "schema_version": 1,
        "catalogue_id": prefix,
        "declared_scope": {
            "scope_statement": f"test catalogue {prefix}",
            "catalogue_complete_within_scope": True,
            "enumeration_proof_artifact_id": f"{prefix}_enumeration",
            "unrestricted_string_universe_covered": False,
            "reduction_theorem_artifact_id": None,
        },
        "provenance": {
            "run_id": f"{prefix}_run",
            "producer_commit": "a" * 40,
            "verifier_commit": "b" * 40,
            "source_dirty": False,
            "replay_argv": ["oph-string-vacuum", "verify-catalogue", "catalogue.json"],
        },
        "artifacts": catalogue_artifacts,
        "equivalence_partition": {
            "relation_definition_sha256": "sha256:" + "7" * 64,
            "partition_proof_artifact_id": f"{prefix}_partition",
            "duality_families_covered": [],
        },
        "candidate_reports": candidate_rows,
        "unresolved_regions": [],
        "producer_classification": "CATALOGUE_COMPLETE",
    }


def test_schemas_are_valid_draft_2020_12() -> None:
    for path in (
        CANDIDATE_SCHEMA_PATH,
        CATALOGUE_SCHEMA_PATH,
        RECEIPT_TARGETS_SCHEMA_PATH,
        OBSERVABLE_TARGETS_SCHEMA_PATH,
    ):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_machine_readable_receipt_and_observable_targets_are_fail_closed() -> None:
    receipt_registry = receipt_target_registry()
    observable_registry = observable_target_registry()

    receipt_ids = {row["receipt_id"] for row in receipt_registry["targets"]}
    assert CANONICAL_CANDIDATE_RECEIPT_IDS <= receipt_ids
    assert len(receipt_ids) == len(receipt_registry["targets"])
    assert observable_registry["status"] == "OPEN_INCOMPLETE"
    assert observable_registry["promotion_allowed"] is False
    assert observable_registry["rank_eligible_row_ids"] == [
        "oph.alpha2_mz",
        "oph.alphaY_mz",
        "oph.v_gev",
    ]
    assert observable_registry["rank_forbidden_row_ids"] == [
        "comparison.mH_pole_gev",
        "comparison.mt_pole_gev",
    ]


def test_receipt_target_registry_rejects_dependency_drift() -> None:
    registry = receipt_target_registry()
    tampered = copy.deepcopy(registry)
    tampered["targets"][0]["dependencies"] = ["MISSING_RECEIPT"]

    with pytest.raises(ReceiptTargetRegistryError, match="unknown dependencies"):
        validate_receipt_target_registry(tampered)

    tampered = copy.deepcopy(registry)
    tampered["targets"][-1]["dependencies"].pop()
    with pytest.raises(ReceiptTargetRegistryError, match="content hash drift"):
        validate_receipt_target_registry(tampered)

    tampered = copy.deepcopy(registry)
    tampered["receipt_aliases"].pop("BD_MODULI_LOCKING_CERTIFICATE_RECEIPT")
    with pytest.raises(ReceiptTargetRegistryError, match="alias registry drift"):
        validate_receipt_target_registry(tampered)


def test_observable_target_registry_rejects_value_and_role_drift() -> None:
    registry = observable_target_registry()
    tampered = copy.deepcopy(registry)
    tampered["rows"][0]["comparison_value"] = "0.04"
    with pytest.raises(ReceiptTargetRegistryError, match="content hash drift"):
        validate_observable_target_registry(tampered)

    tampered = copy.deepcopy(registry)
    tampered["compare_only_inputs_not_targets"].append("oph.alpha2_mz")
    with pytest.raises(ReceiptTargetRegistryError, match="overlap"):
        validate_observable_target_registry(tampered)


def test_candidate_report_locators_and_compatibility_aliases_do_not_drift(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    report = verify_candidate_evidence(packet, bundle_root=tmp_path)
    registry = receipt_target_registry()

    for row in registry["targets"]:
        if row["report_scope"] != "candidate":
            continue
        value: object = report
        for part in row["report_locator"].split("."):
            assert isinstance(value, dict), row["receipt_id"]
            assert part in value, row["receipt_id"]
            value = value[part]
    assert CANONICAL_CANDIDATE_RECEIPT_IDS <= set(report["receipts"])
    assert set(report["receipts"]) == (
        CANONICAL_CANDIDATE_RECEIPT_IDS | set(registry["receipt_aliases"])
    )
    for alias, canonical in registry["receipt_aliases"].items():
        assert report["receipts"][alias] == report["receipts"][canonical]


def test_branch_and_catalogue_report_locators_are_declared_by_contract() -> None:
    registry = receipt_target_registry()
    catalogue_report = verify_catalogue_evidence({})
    branch_row_fields = {
        "verified_candidate_status",
        "verified_branch_domain_coverage_status",
        "verified_branch_verdict_status",
        "branch_global_uniqueness_receipt",
    }
    branch_summary_fields = {
        "candidate_replay_receipt",
        "branch_domain_coverage_receipt",
        "branch_verdict_replay_receipt",
        "branch_global_uniqueness_receipt",
    }
    for row in registry["targets"]:
        locator = row["report_locator"]
        if row["report_scope"] == "branch":
            candidate_prefix = "candidate_rows[*]."
            branch_prefix = "branch_rows[*]."
            if locator.startswith(candidate_prefix):
                assert locator.removeprefix(candidate_prefix) in branch_row_fields
            elif locator.startswith(branch_prefix):
                assert locator.removeprefix(branch_prefix) in branch_summary_fields
            else:
                assert locator in catalogue_report
        elif row["report_scope"] == "catalogue":
            assert "." not in locator
            assert locator in catalogue_report


def test_interval_contraction_is_recomputed_exactly(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    report = verify_interval_contraction(packet["augmented_system"], physical_dimension=1)

    assert report["interval_contraction_receipt"] is True
    assert report["preconditioner_determinant"] == "1"
    assert report["contraction_bound_infinity_norm"] == "0"
    assert report["interval_image"] == [_interval("0", "0")]


def test_interval_contraction_rejects_noncontractive_jacobian(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    packet["augmented_system"]["jacobian_interval"] = [[_interval("2", "3")]]

    report = verify_interval_contraction(packet["augmented_system"], physical_dimension=1)

    assert report["interval_contraction_receipt"] is False
    assert "interval_contraction_bound_not_below_one" in report["blockers"]


def test_candidate_verifier_ignores_producer_pass_hints(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["contract_integrity_receipt"] is True
    assert report["artifact_hash_receipt"] is True
    assert report["interval_algebra"]["interval_contraction_receipt"] is True
    assert report["candidate_status"] == "INCONCLUSIVE"
    assert report["receipts"]["LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT"] is False
    assert all(value is False for value in report["semantic_gate_receipts"].values())
    assert "semantic_gate_verifier_not_implemented:critical_edge_cft" in report["blockers"]


def test_semantic_pass_cannot_override_failed_interval_isolation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    packet = _candidate_packet(tmp_path)
    packet["augmented_system"]["jacobian_interval"] = [[_interval("2", "3")]]

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["candidate_status"] == "INVALID"
    assert report["physical_local_isolation_receipt"] is False
    assert report["receipts"]["BD_MODULI_LOCKING_CERTIFICATE_RECEIPT"] is False
    assert report["receipts"]["LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT"] is False


def test_candidate_verifier_rejects_self_targeting_rank_row(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    packet["target_registry"]["rows"][0]["role"] = "candidate_only"

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["contract_integrity_receipt"] is False
    assert "self_targeting_or_diagnostic_required_row:target_x" in report["blockers"]
    assert "inadmissible_target_rank_row:target_x" in report["blockers"]


def test_candidate_verifier_rejects_unregistered_augmented_source_row(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    packet["augmented_system"]["row_registry"][0] = {
        "row_id": "target_x",
        "source_kind": "completion_constraint",
        "source_row_id": "invented_coordinate_equation",
    }
    packet["augmented_system"]["row_registry_sha256"] = stable_json_hash(
        packet["augmented_system"]["row_registry"]
    )

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["contract_integrity_receipt"] is False
    assert report["physical_local_isolation_receipt"] is False
    assert any("unknown_augmented_constraint_row" in row for row in report["blockers"])


def test_candidate_verifier_rejects_artifact_mutation(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    (tmp_path / "interval_proof.json").write_text('{"mutated":true}\n', encoding="utf-8")

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["artifact_hash_receipt"] is False
    assert report["interval_algebra"]["interval_contraction_receipt"] is True
    assert report["interval_contraction_algebra_receipt"] is False
    assert "artifact_hash_mismatch:interval_proof" in report["blockers"]


def test_candidate_verifier_rejects_unknown_target_covariance(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    packet["target_registry"]["rows"][0]["covariance_artifact_id"] = "missing_covariance"
    packet["provenance"]["target_registry_sha256"] = stable_json_hash(packet["target_registry"])

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["contract_integrity_receipt"] is False
    assert report["artifact_hash_receipt"] is False
    assert "unknown_artifact_reference:target_covariance:target_x:missing_covariance" in report[
        "blockers"
    ]


def test_superpotential_receipt_enforces_operator_safety_dependency() -> None:
    gates = {gate_id: False for gate_id in CANDIDATE_GATE_IDS}
    gates["superpotential_safety"] = True

    receipts = contract_module._aggregate_receipts(gates)

    assert receipts["STRING_OPERATOR_SAFETY_REALIZATION_RECEIPT"] is False
    assert receipts["STRING_SUPERPOTENTIAL_SAFETY_RECEIPT"] is False
    assert receipts["BD_SUPERPOTENTIAL_SAFETY_RECEIPT"] is False


def test_flat_direction_requires_classification_appropriate_proof(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    packet = _candidate_packet(tmp_path)
    packet["flat_directions"] = [
        {
            "direction_id": "putatively_stabilized",
            "classification": "STABILIZED",
            "proof_level": "INFINITESIMAL_ONLY",
            "physical_meaning": "test scalar",
            "basis_vector": ["1"],
            "evidence_artifact_id": "gate_proof",
        }
    ]

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["candidate_status"] == "INVALID"
    assert report["flat_direction_classification_receipt"] is False
    assert "flat_direction_proof_level_mismatch:putatively_stabilized" in report["blockers"]


@pytest.mark.parametrize(
    ("classification", "proof_level", "expected_status"),
    [
        ("VISIBLE_FLAT", "EXACT_CURVE", "FAIL"),
        ("UNCLASSIFIED", "INFINITESIMAL_ONLY", "INCONCLUSIVE"),
    ],
)
def test_valid_flat_direction_receipt_retracts_only_selection(
    tmp_path: Path,
    monkeypatch,
    classification: str,
    proof_level: str,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    packet = _candidate_packet(tmp_path)
    packet["flat_directions"] = [
        {
            "direction_id": "residual_scalar",
            "classification": classification,
            "proof_level": proof_level,
            "physical_meaning": "adjustable observer-visible scalar",
            "basis_vector": ["1"],
            "evidence_artifact_id": "gate_proof",
        }
    ]

    report = verify_candidate_evidence(packet, bundle_root=tmp_path)

    assert report["contract_integrity_receipt"] is True
    assert report["candidate_status"] == expected_status
    assert report["flat_direction_classification_receipt"] is False
    assert report["receipts"]["STRING_MODULI_LOCKING_RECEIPT"] is False
    assert report["receipts"]["LOCAL_OPH_STRING_VACUUM_WITNESS_RECEIPT"] is False


def test_catalogue_replays_candidate_verifier_and_stays_inconclusive(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_report = verify_candidate_evidence(packet, bundle_root=tmp_path)
    report_path = tmp_path / "candidate_verification.json"
    report_path.write_text(json.dumps(candidate_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    catalogue_artifacts = [
        _write_artifact(tmp_path, "enumeration_proof.json", {"proof": "finite unit catalogue"}),
        _write_artifact(tmp_path, "partition_proof.json", {"proof": "unit equivalence partition"}),
        _write_artifact(tmp_path, "branch_cover.json", {"proof": "unit branch cover"}),
        _write_artifact(tmp_path, "branch_verdict_ledger.json", {"regions": ["passing root"]}),
    ]
    catalogue = {
        "$schema": "schemas/string_vacuum_catalogue_evidence_v1.schema.json",
        "artifact": "oph_string_vacuum_catalogue_evidence",
        "schema_version": 1,
        "catalogue_id": "unit_catalogue",
        "declared_scope": {
            "scope_statement": "one unit-test candidate",
            "catalogue_complete_within_scope": True,
            "enumeration_proof_artifact_id": "enumeration_proof",
            "unrestricted_string_universe_covered": False,
            "reduction_theorem_artifact_id": None,
        },
        "provenance": {
            "run_id": "unit_catalogue_run",
            "producer_commit": "a" * 40,
            "verifier_commit": "b" * 40,
            "source_dirty": False,
            "replay_argv": ["oph-string-vacuum", "verify-catalogue", "catalogue.json"],
        },
        "artifacts": catalogue_artifacts,
        "equivalence_partition": {
            "relation_definition_sha256": "sha256:" + "7" * 64,
            "partition_proof_artifact_id": "partition_proof",
            "duality_families_covered": [],
        },
        "candidate_reports": [
            {
                "candidate_id": "unit_candidate",
                "oph_equivalence_class_id": "unit_class",
                "evidence_packet_path": "candidate.json",
                "evidence_packet_sha256": _sha256(candidate_path),
                "verification_report_path": "candidate_verification.json",
                "verification_report_sha256": _sha256(report_path),
                "branch_domain_coverage_status": "PASS",
                "branch_domain_coverage_artifact_id": "branch_cover",
                "branch_verdict_ledger_artifact_id": "branch_verdict_ledger",
                "producer_status_hint": "PASS",
            }
        ],
        "unresolved_regions": [],
        "producer_classification": "CATALOGUE_COMPLETE",
    }

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_status"] == "INCONCLUSIVE"
    assert report["selected_equivalence_class_id"] is None
    assert report["COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT"] is False
    assert report["OPH_NATIVE_STRING_VACUUM_RECEIPT"] is False


def test_catalogue_branch_fail_cannot_suppress_a_passing_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
        },
    )
    packet = _candidate_packet(tmp_path)
    candidate_path = tmp_path / "branch_fail_candidate.json"
    candidate_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_report = verify_candidate_evidence(packet, bundle_root=tmp_path)
    assert candidate_report["candidate_status"] == "PASS"
    report_path = tmp_path / "branch_fail_verification.json"
    report_path.write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalogue_artifacts = [
        _write_artifact(tmp_path, "branch_fail_enumeration.json", {"proof": "complete"}),
        _write_artifact(tmp_path, "branch_fail_partition.json", {"proof": "partition"}),
        _write_artifact(tmp_path, "branch_fail_cover.json", {"proof": "producer says failed"}),
        _write_artifact(tmp_path, "branch_fail_verdict.json", {"regions": ["unresolved"]}),
    ]
    catalogue = {
        "$schema": "schemas/string_vacuum_catalogue_evidence_v1.schema.json",
        "artifact": "oph_string_vacuum_catalogue_evidence",
        "schema_version": 1,
        "catalogue_id": "branch_fail_catalogue",
        "declared_scope": {
            "scope_statement": "one passing test candidate",
            "catalogue_complete_within_scope": True,
            "enumeration_proof_artifact_id": "branch_fail_enumeration",
            "unrestricted_string_universe_covered": False,
            "reduction_theorem_artifact_id": None,
        },
        "provenance": {
            "run_id": "branch_fail_run",
            "producer_commit": "a" * 40,
            "verifier_commit": "b" * 40,
            "source_dirty": False,
            "replay_argv": ["oph-string-vacuum", "verify-catalogue", "catalogue.json"],
        },
        "artifacts": catalogue_artifacts,
        "equivalence_partition": {
            "relation_definition_sha256": "sha256:" + "7" * 64,
            "partition_proof_artifact_id": "branch_fail_partition",
            "duality_families_covered": [],
        },
        "candidate_reports": [
            {
                "candidate_id": "unit_candidate",
                "oph_equivalence_class_id": "unit_class",
                "evidence_packet_path": candidate_path.name,
                "evidence_packet_sha256": _sha256(candidate_path),
                "verification_report_path": report_path.name,
                "verification_report_sha256": _sha256(report_path),
                "branch_domain_coverage_status": "FAIL",
                "branch_domain_coverage_artifact_id": "branch_fail_cover",
                "branch_verdict_ledger_artifact_id": "branch_fail_verdict",
                "producer_status_hint": "PASS",
            }
        ],
        "unresolved_regions": [],
        "producer_classification": "CATALOGUE_COMPLETE",
    }

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_status"] == "INCONCLUSIVE"
    assert report["selected_equivalence_class_id"] is None
    assert report["candidate_rows"][0]["verified_branch_domain_coverage_status"] == "FAIL"


def test_branch_global_uniqueness_rejects_two_passing_classes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
        },
    )
    first = _candidate_packet(tmp_path)
    second = copy.deepcopy(first)
    second["candidate"]["candidate_id"] = "unit_candidate_2"
    second["candidate"]["oph_equivalence_class_id"] = "unit_class_2"
    catalogue = _catalogue_packet(
        tmp_path, [first, second], prefix="two_passing_classes"
    )

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_status"] == "INCONCLUSIVE"
    assert report["COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT"] is False
    assert len(report["branch_rows"]) == 1
    assert report["branch_rows"][0]["passing_equivalence_class_ids"] == [
        "unit_class",
        "unit_class_2",
    ]
    assert report["branch_rows"][0]["branch_global_uniqueness_receipt"] is False


def test_branch_receipts_keep_branch_local_scope(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
        },
    )
    first = _candidate_packet(tmp_path)
    second = copy.deepcopy(first)
    second["candidate"]["candidate_id"] = "other_branch_candidate"
    second["candidate"]["oph_equivalence_class_id"] = "other_branch_class"
    second_branch = _write_artifact(
        tmp_path, "other_branch_definition.json", {"branch": "other branch"}
    )
    second["artifacts"].append(second_branch)
    second["candidate"]["branch_definition_artifact_id"] = second_branch["artifact_id"]
    second["candidate"]["branch_definition_sha256"] = second_branch["sha256"]
    catalogue = _catalogue_packet(
        tmp_path, [first, second], prefix="branch_local_scope"
    )
    catalogue["candidate_reports"][1]["branch_domain_coverage_status"] = "FAIL"

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    selected_branch = next(
        row for row in report["branch_rows"] if row["candidate_ids"] == ["unit_candidate"]
    )
    assert report["branch_global_coverage_receipt"] is False
    assert report["catalogue_status"] == "INCONCLUSIVE"
    assert selected_branch["candidate_replay_receipt"] is True
    assert selected_branch["branch_domain_coverage_receipt"] is True
    assert selected_branch["branch_verdict_replay_receipt"] is True
    assert selected_branch["branch_global_uniqueness_receipt"] is True


def test_catalogue_proof_receipt_requires_verified_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
        },
    )
    catalogue = _catalogue_packet(
        tmp_path, [_candidate_packet(tmp_path)], prefix="bad_enumeration_artifact"
    )
    (tmp_path / "bad_enumeration_artifact_enumeration.json").write_text(
        '{"mutated":true}\n', encoding="utf-8"
    )

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_enumeration_receipt"] is False
    assert report["COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT"] is False
    assert "artifact_hash_mismatch:bad_enumeration_artifact_enumeration" in report[
        "catalogue_blockers"
    ]


def test_catalogue_proof_receipt_rejects_duplicate_artifact_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
        },
    )
    catalogue = _catalogue_packet(
        tmp_path, [_candidate_packet(tmp_path)], prefix="duplicate_enumeration"
    )
    duplicate = _write_artifact(
        tmp_path, "duplicate_enumeration_copy.json", {"proof": "ambiguous copy"}
    )
    duplicate["artifact_id"] = "duplicate_enumeration_enumeration"
    catalogue["artifacts"].append(duplicate)

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_enumeration_receipt"] is False
    assert "duplicate_artifact_id:duplicate_enumeration_enumeration" in report[
        "catalogue_blockers"
    ]


def test_unrestricted_failure_does_not_retract_catalogue_selection(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
            "unrestricted_reduction": lambda payload, row, root: True,
        },
    )
    packet = _candidate_packet(tmp_path)
    candidate_path = tmp_path / "scoped_candidate.json"
    candidate_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_report = verify_candidate_evidence(packet, bundle_root=tmp_path)
    assert candidate_report["candidate_status"] == "PASS"
    report_path = tmp_path / "scoped_candidate_verification.json"
    report_path.write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalogue_artifacts = [
        _write_artifact(tmp_path, "scoped_enumeration.json", {"proof": "complete"}),
        _write_artifact(tmp_path, "scoped_partition.json", {"proof": "partition"}),
        _write_artifact(tmp_path, "scoped_cover.json", {"proof": "covered"}),
        _write_artifact(tmp_path, "scoped_verdict.json", {"regions": ["passing root"]}),
        _write_artifact(tmp_path, "unrestricted_reduction.json", {"proof": "producer hint only"}),
    ]
    catalogue = {
        "$schema": "schemas/string_vacuum_catalogue_evidence_v1.schema.json",
        "artifact": "oph_string_vacuum_catalogue_evidence",
        "schema_version": 1,
        "catalogue_id": "scoped_catalogue",
        "declared_scope": {
            "scope_statement": "one semantically covered unit-test candidate",
            "catalogue_complete_within_scope": True,
            "enumeration_proof_artifact_id": "scoped_enumeration",
            "unrestricted_string_universe_covered": True,
            "reduction_theorem_artifact_id": "unrestricted_reduction",
        },
        "provenance": {
            "run_id": "scoped_catalogue_run",
            "producer_commit": "a" * 40,
            "verifier_commit": "b" * 40,
            "source_dirty": False,
            "replay_argv": ["oph-string-vacuum", "verify-catalogue", "catalogue.json"],
        },
        "artifacts": catalogue_artifacts,
        "equivalence_partition": {
            "relation_definition_sha256": "sha256:" + "7" * 64,
            "partition_proof_artifact_id": "scoped_partition",
            "duality_families_covered": [],
        },
        "candidate_reports": [
            {
                "candidate_id": "unit_candidate",
                "oph_equivalence_class_id": "unit_class",
                "evidence_packet_path": candidate_path.name,
                "evidence_packet_sha256": _sha256(candidate_path),
                "verification_report_path": report_path.name,
                "verification_report_sha256": _sha256(report_path),
                "branch_domain_coverage_status": "PASS",
                "branch_domain_coverage_artifact_id": "scoped_cover",
                "branch_verdict_ledger_artifact_id": "scoped_verdict",
                "producer_status_hint": "PASS",
            }
        ],
        "unresolved_regions": [],
        "producer_classification": "CATALOGUE_COMPLETE",
    }

    (tmp_path / "unrestricted_reduction.json").write_text(
        '{"mutated":true}\n', encoding="utf-8"
    )

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_status"] == "SELECTED_WITHIN_DECLARED_CATALOGUE"
    assert report["COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT"] is True
    assert report["unrestricted_string_universe_coverage_receipt"] is False
    assert report["OPH_NATIVE_STRING_VACUUM_RECEIPT"] is False
    assert report["catalogue_blockers"] == []
    assert "artifact_hash_mismatch:unrestricted_reduction" in report[
        "unrestricted_blockers"
    ]
    assert "unrestricted_reduction_not_semantically_verified" in report[
        "unrestricted_blockers"
    ]


def test_duplicate_unrestricted_artifact_id_stays_unrestricted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        contract_module,
        "SEMANTIC_GATE_VERIFIERS",
        {gate_id: (lambda payload, root: "PASS") for gate_id in CANDIDATE_GATE_IDS},
    )
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {
            "catalogue_enumeration": lambda payload, row, root: True,
            "equivalence_partition": lambda payload, row, root: True,
            "branch_domain_coverage": lambda payload, row, root: True,
            "branch_verdict_replay": lambda payload, row, root: True,
            "unrestricted_reduction": lambda payload, row, root: True,
        },
    )
    catalogue = _catalogue_packet(
        tmp_path, [_candidate_packet(tmp_path)], prefix="duplicate_reduction"
    )
    first = _write_artifact(tmp_path, "reduction_copy_a.json", {"proof": "a"})
    second = _write_artifact(tmp_path, "reduction_copy_b.json", {"proof": "b"})
    first["artifact_id"] = "reduction_duplicate"
    second["artifact_id"] = "reduction_duplicate"
    catalogue["artifacts"].extend([first, second])
    catalogue["declared_scope"]["unrestricted_string_universe_covered"] = True
    catalogue["declared_scope"]["reduction_theorem_artifact_id"] = (
        "reduction_duplicate"
    )

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_status"] == "SELECTED_WITHIN_DECLARED_CATALOGUE"
    assert report["COMPARATIVE_UNIQUENESS_CERTIFICATE_RECEIPT"] is True
    assert report["unrestricted_string_universe_coverage_receipt"] is False
    assert report["OPH_NATIVE_STRING_VACUUM_RECEIPT"] is False
    assert report["catalogue_blockers"] == []
    assert "duplicate_artifact_id:reduction_duplicate" in report[
        "unrestricted_blockers"
    ]


def test_unrestricted_artifact_blocker_matching_uses_exact_ids() -> None:
    assert contract_module._artifact_blocker_targets_id(
        "artifact_hash_mismatch:red", "red"
    )
    assert not contract_module._artifact_blocker_targets_id(
        "artifact_hash_mismatch:redextra", "red"
    )
    assert contract_module._artifact_blocker_targets_id(
        "artifact_invalid:red:artifact_not_regular_file", "red"
    )
    assert not contract_module._artifact_blocker_targets_id(
        "artifact_invalid:redextra:artifact_not_regular_file", "red"
    )


def test_catalogue_semantic_verifier_rejects_truthy_non_boolean(monkeypatch) -> None:
    monkeypatch.setattr(
        contract_module,
        "CATALOGUE_PROOF_VERIFIERS",
        {"catalogue_enumeration": lambda payload, row, root: "FAIL"},
    )
    blockers: list[str] = []

    receipt = contract_module._verify_catalogue_proof(
        "catalogue_enumeration",
        {},
        None,
        bundle_root=None,
        blockers=blockers,
    )

    assert receipt is False
    assert any("semantic_catalogue_verifier_error" in row for row in blockers)


def test_cli_returns_nonzero_for_inconclusive_candidate(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    packet_path = tmp_path / "cli_candidate.json"
    report_path = tmp_path / "cli_report.json"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    exit_code = string_vacuum_cli(
        [
            "verify-candidate",
            str(packet_path),
            "--bundle-root",
            str(tmp_path),
            "--out",
            str(report_path),
        ]
    )

    assert exit_code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["candidate_status"] == "INCONCLUSIVE"


def test_cli_writes_invalid_report_for_malformed_json(tmp_path: Path) -> None:
    packet_path = tmp_path / "malformed.json"
    report_path = tmp_path / "malformed_report.json"
    packet_path.write_text("{", encoding="utf-8")

    exit_code = string_vacuum_cli(
        [
            "verify-candidate",
            str(packet_path),
            "--bundle-root",
            str(tmp_path),
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["candidate_status"] == "INVALID"
    assert "input_packet_unreadable:JSONDecodeError" in report["blockers"]


def test_cli_describes_receipt_and_observable_targets(tmp_path: Path) -> None:
    output_path = tmp_path / "string_vacuum_targets.json"

    exit_code = string_vacuum_cli(["describe-targets", "--out", str(output_path)])

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["artifact"] == "oph_string_vacuum_simulator_target_specification"
    assert len(report["receipt_targets"]["targets"]) == 27
    assert report["observable_targets"]["status"] == "OPEN_INCOMPLETE"


def test_report_internal_hash_detects_forged_candidate_status(tmp_path: Path) -> None:
    packet = _candidate_packet(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_report = verify_candidate_evidence(packet, bundle_root=tmp_path)
    forged = copy.deepcopy(candidate_report)
    forged["candidate_status"] = "PASS"
    forged.pop("report_sha256")
    forged["report_sha256"] = stable_json_hash(forged)
    report_path = tmp_path / "candidate_verification.json"
    report_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    catalogue_artifacts = [
        _write_artifact(tmp_path, "enumeration_proof.json", {"proof": "finite unit catalogue"}),
        _write_artifact(tmp_path, "partition_proof.json", {"proof": "unit equivalence partition"}),
        _write_artifact(tmp_path, "branch_cover.json", {"proof": "unit branch cover"}),
        _write_artifact(tmp_path, "branch_verdict_ledger.json", {"regions": ["passing root"]}),
    ]
    catalogue = {
        "$schema": "schemas/string_vacuum_catalogue_evidence_v1.schema.json",
        "artifact": "oph_string_vacuum_catalogue_evidence",
        "schema_version": 1,
        "catalogue_id": "unit_catalogue",
        "declared_scope": {
            "scope_statement": "one unit-test candidate",
            "catalogue_complete_within_scope": True,
            "enumeration_proof_artifact_id": "enumeration_proof",
            "unrestricted_string_universe_covered": False,
            "reduction_theorem_artifact_id": None,
        },
        "provenance": {
            "run_id": "unit_catalogue_run",
            "producer_commit": "a" * 40,
            "verifier_commit": "b" * 40,
            "source_dirty": False,
            "replay_argv": ["oph-string-vacuum", "verify-catalogue", "catalogue.json"],
        },
        "artifacts": catalogue_artifacts,
        "equivalence_partition": {
            "relation_definition_sha256": "sha256:" + "7" * 64,
            "partition_proof_artifact_id": "partition_proof",
            "duality_families_covered": [],
        },
        "candidate_reports": [
            {
                "candidate_id": "unit_candidate",
                "oph_equivalence_class_id": "unit_class",
                "evidence_packet_path": "candidate.json",
                "evidence_packet_sha256": _sha256(candidate_path),
                "verification_report_path": "candidate_verification.json",
                "verification_report_sha256": _sha256(report_path),
                "branch_domain_coverage_status": "PASS",
                "branch_domain_coverage_artifact_id": "branch_cover",
                "branch_verdict_ledger_artifact_id": "branch_verdict_ledger",
                "producer_status_hint": "PASS",
            }
        ],
        "unresolved_regions": [],
        "producer_classification": "CATALOGUE_COMPLETE",
    }

    report = verify_catalogue_evidence(catalogue, bundle_root=tmp_path)

    assert report["catalogue_status"] == "INCONCLUSIVE"
    assert any("candidate_verification_report_not_reproducible" in row for row in report["blockers"])
