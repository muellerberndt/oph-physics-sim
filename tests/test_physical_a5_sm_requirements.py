from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.gauge import physical_a5_sm_requirements as a5
from oph_fpe.evidence.production_envelope import (
    CANONICAL_JSON_POLICY,
    FREEZE_ANCHOR_SCHEMA,
    PRODUCTION_BUNDLE_MANIFEST_SCHEMA,
    PRODUCTION_ENVELOPE_SCHEMA,
    TARGET_FIREWALL_FALSE_FIELDS,
    canonical_json_sha256,
)
from oph_fpe.gauge.physical_a5_sm_requirements import (
    BASE_GLOBAL_PASS_STAGES,
    ClaimScope,
    FULL_INTERACTING_PASS_STAGES,
    Q2_E_RECEIPTS,
    Q2_H_RECEIPTS,
    REQUIRED_ROOT_ROLES,
    RootManifestError,
    ROOT_MANIFEST_SCHEMA,
    STAGE_IDS,
    TARGET_SELECTION_FIELDS,
    TerminalStatus,
    _compute_claim_tiers,
    _physical_stage_pass_authorized,
    _report_status_for_scope,
    verify_physical_a5_sm_requirements,
    verify_physical_a5_sm_root_manifest,
    verify_physical_a5_sm_root_report_file,
    verify_physical_a5_sm_stage_envelope,
    write_physical_a5_sm_root_report,
)


EXPECTED_STAGE_IDS = (
    "ROOT",
    "GEOMETRY_565",
    "CURRENT_566",
    "GLOBAL_FORM_567",
    "SPIN_EXCHANGE_314",
    "SOURCE_REGISTRY_AND_PRIMITIVE_RESIDUES",
    "SCALAR_CHANNEL",
    "FAMILY_ATTACHMENT_569",
    "Q1_LOCAL_ACTION",
    "Q2_H",
    "Q2_E",
    "POSITIVITY_OR_POSITIVE_TRANSFER",
    "REFINEMENT_COMPLETENESS",
    "PHYSICAL_IDENTIFICATION",
    "COMPLETE_COUPLED_DYNAMICS",
    "FAMILY_BREAKING_OR_DESCENT",
    "VERTEX_1PI",
    "Q4_OS",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _build_valid_root(
    root: Path,
    *,
    extra_evidence: dict | None = None,
) -> Path:
    artifact_rows: list[dict] = []
    by_role: dict[str, dict] = {}
    for role in REQUIRED_ROOT_ROLES:
        artifact_id = role.replace("_", ".")
        path = root / "artifacts" / f"{role}.json"
        _write_json(path, {"kind": role, "frozen": "before_outcomes"})
        row = {
            "artifact_id": artifact_id,
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "media_type": "application/json",
            "semantic_role": role,
            "provenance_kind": (
                "source_operation" if role == "dynamics_generator" else "source_primitive"
            ),
        }
        artifact_rows.append(row)
        by_role[role] = row

    derived_path = root / "artifacts" / "derived_response.json"
    _write_json(derived_path, {"values": [1, 2, 3]})
    artifact_rows.append(
        {
            "artifact_id": "derived.response",
            "path": derived_path.relative_to(root).as_posix(),
            "sha256": _sha256(derived_path),
            "media_type": "application/json",
            "semantic_role": "derived_response",
            "provenance_kind": "derived_array",
        }
    )

    if extra_evidence is not None:
        evidence_path = root / "artifacts" / "candidate_evidence.json"
        _write_json(evidence_path, extra_evidence)
        artifact_rows.append(
            {
                "artifact_id": "candidate.evidence",
                "path": evidence_path.relative_to(root).as_posix(),
                "sha256": _sha256(evidence_path),
                "media_type": "application/json",
                "semantic_role": "lane_evidence",
                "provenance_kind": "source_primitive",
            }
        )

    candidate = by_role["candidate_domain"]
    selection = by_role["selection_law"]
    manifest = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "packet_id": "root.packet.v1",
        "code_commit": "a" * 40,
        "dirty_tree_digest": by_role["dirty_tree_manifest"]["sha256"],
        "regulator_id": "finite.regulator.v1",
        "boundary_condition_id": "closed.boundary.v1",
        "superselection_sector_id": "vacuum.sector.v1",
        "random_seeds": [7, 11, 13],
        "reproducibility": {
            "mode": "seeded_stochastic",
            "seed_schedule": "explicit_manifest_order",
            "parallel_reduction": "deterministic",
            "blas_threads_per_worker": 1,
        },
        "numerical_policy": {
            "backend": "exact-test-backend",
            "backend_version": "1.0",
            "precision_bits": 128,
            "interval_method": "outward_rounded_rational_endpoints",
            "rank_certification_method": "exact_fraction_elimination",
        },
        "artifacts": artifact_rows,
        "source_ancestry_edges": [
            {
                "parent_artifact_id": by_role["source_operator_registry"]["artifact_id"],
                "child_artifact_id": "derived.response",
                "operation_artifact_id": by_role["dynamics_generator"]["artifact_id"],
            }
        ],
        "candidate_domain_artifact_id": candidate["artifact_id"],
        "candidate_domain_hash": candidate["sha256"],
        "selection_law_artifact_id": selection["artifact_id"],
        "selection_law_hash": selection["sha256"],
        "target_selection_dependencies": {
            field_name: False for field_name in TARGET_SELECTION_FIELDS
        },
    }
    path = root / "physical_a5_sm_root_manifest.json"
    _write_json(path, manifest)
    return path


def _mutate_manifest(path: Path, mutator) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    _write_json(path, payload)


def _build_common_stage_production_bundle(
    root: Path,
    *,
    source_root_hash: str,
    stage_id: str = "GEOMETRY_565",
    claim_scope: str = "structural",
) -> Path:
    files: dict[str, tuple[Path, str]] = {}

    def add_json(relative: str, payload: object, media_type: str = "application/json") -> None:
        path = root / relative
        _write_json(path, payload)
        files[relative] = (path, media_type)

    def add_text(relative: str, value: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        files[relative] = (path, "text/plain")

    schema = {
        "$id": "oph.test.a5-stage-evidence",
        "schema_version": "1.0.0",
        "type": "object",
    }
    subject = {"stage_id": stage_id, "frozen": True}
    output = {"diagnostic": "inventory-only"}
    anchor = {
        "schema": FREEZE_ANCHOR_SCHEMA,
        "source_root_hash": source_root_hash,
        "branch_id": "branch-a5-test",
        "freeze_id": "freeze-a5-test",
        "commitment_phase": "pre_outcome",
        "frozen_utc": "2026-07-20T00:00:00Z",
    }
    add_json("schema.json", schema, "application/schema+json")
    add_json("subject.json", subject)
    add_json("output.json", output)
    add_json("freeze_anchor.json", anchor)
    add_text("producer/source.txt", "producer-source")
    add_text("producer/executable.txt", "producer-executable")
    add_text("producer/environment.txt", "producer-environment")
    add_text("checker/source.txt", "checker-source")
    add_text("checker/executable.txt", "checker-executable")

    def ref(relative: str) -> dict[str, object]:
        path, media_type = files[relative]
        return {
            "path": relative,
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
            "media_type": media_type,
        }

    envelope = {
        "envelope_schema": PRODUCTION_ENVELOPE_SCHEMA,
        "profile": "COMMON_STAGE",
        "schema_id": schema["$id"],
        "schema_version": schema["schema_version"],
        "schema_sha256": _sha256(files["schema.json"][0]),
        "schema_ref": ref("schema.json"),
        "artifact_id": "geometry-envelope",
        "stage_id": stage_id,
        "receipt_type": "A5_STAGE_EVIDENCE",
        "claim_lane": stage_id,
        "claim_scope": claim_scope,
        "branch_id": anchor["branch_id"],
        "freeze_id": anchor["freeze_id"],
        "source_root_hash": source_root_hash,
        "subject_type": "A5_STAGE_SUBJECT",
        "subject_canonicalization": CANONICAL_JSON_POLICY,
        "subject_ref": ref("subject.json"),
        "subject_digest": canonical_json_sha256(subject),
        "output_ref": ref("output.json"),
        "output_digest": canonical_json_sha256(output),
        "producer": {
            "producer_id": "a5-stage-producer",
            "source_tree_ref": ref("producer/source.txt"),
            "executable_ref": ref("producer/executable.txt"),
            "environment_lock_ref": ref("producer/environment.txt"),
        },
        "checker": {
            "checker_id": "a5-stage-checker",
            "source_tree_ref": ref("checker/source.txt"),
            "executable_ref": ref("checker/executable.txt"),
            "checker_independence_class": "separate-implementation",
        },
        "freeze_anchor_ref": ref("freeze_anchor.json"),
        "parent_receipts": [],
        "shared_contract_hashes": {},
        "numeric_precision_and_rounding": {
            "arithmetic": "exact",
            "precision_bits": None,
            "rounding_mode": "exact",
            "numeric_backend": "test-rational",
        },
        "target_firewall": {
            **{key: False for key in TARGET_FIREWALL_FALSE_FIELDS},
            "comparison_boundary": "read_only_separate_process",
            "exposure_status": "prospective_blind",
        },
        "status": "PASS",
        "blockers": [],
        "generated_utc": "2026-07-20T00:00:01Z",
    }
    add_json("envelope.json", envelope)

    file_rows = [ref(relative) for relative in sorted(files)]
    manifest_without_hash = {
        "schema": PRODUCTION_BUNDLE_MANIFEST_SCHEMA,
        "bundle_id": "a5-common-stage-bundle",
        "files": file_rows,
        "envelope_paths": ["envelope.json"],
    }
    manifest = {
        **manifest_without_hash,
        "manifest_payload_sha256": canonical_json_sha256(manifest_without_hash),
    }
    manifest_path = root / "production_bundle_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_valid_root_replays_but_cannot_promote_downstream_lanes(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)

    root = verify_physical_a5_sm_root_manifest(manifest)
    report = verify_physical_a5_sm_requirements(manifest)

    assert root["inventory_passed"] is True
    assert root["status"] == TerminalStatus.OPEN.value
    assert root["passed"] is False
    assert root["receipts"]["ROOT_INVENTORY_REPLAY_RECEIPT"] is True
    assert root["receipts"]["ROOT_TYPED_ROLE_SEMANTICS_RECEIPT"] is False
    assert root["receipts"]["ROOT_CODE_BUILD_BINDING_RECEIPT"] is False
    assert root["receipts"]["ROOT_PREOUTCOME_COMMITMENT_RECEIPT"] is False
    assert root["receipts"]["ROOT_IMMUTABLE_PACKET_REPLAY_RECEIPT"] is False
    assert STAGE_IDS == EXPECTED_STAGE_IDS
    assert report["stage_order"] == list(EXPECTED_STAGE_IDS)
    assert report["status"] == TerminalStatus.OPEN.value
    assert report["stages"]["ROOT"]["status"] == TerminalStatus.OPEN.value
    assert report["stages"]["ROOT"]["passed"] is False
    assert report["stages"]["GEOMETRY_565"]["passed"] is False
    assert report["stages"]["FAMILY_ATTACHMENT_569"]["passed"] is False
    assert report["receipts"]["PHYSICAL_A5_SM_GLOBAL_PASS"] is False
    assert report["receipts"]["PHYSICAL_A5_SM_STRUCTURAL_GLOBAL_PASS"] is False
    assert report["receipts"]["PHYSICAL_A5_SM_FULL_INTERACTING_PASS"] is False
    assert report["receipts"]["FINITE_EUCLIDEAN_STRUCTURAL_PASS"] is False
    assert report["receipts"]["NUMERICAL_YUKAWA_MATRICES_PHYSICAL_PASS"] is False
    assert report["receipts"]["NONPERTURBATIVE_CONTINUUM_WIGHTMAN_PHYSICAL_PASS"] is False
    assert report["registered_downstream_physical_producer_count"] == 0
    assert report["registered_physical_producer_count"] == 0
    assert report["registered_inventory_verifiers"][0]["verifier_id"].endswith(
        "root_inventory_replayer_v1"
    )
    assert report["nonphysical_campaign_gates"]["EXACT_SMALL_ORACLE"]["passed"] is False
    assert report["nonphysical_campaign_gates"]["SCALE_CAMPAIGN_ALLOWED"]["passed"] is False
    assert set(
        report["nonphysical_campaign_gates"]["EXACT_SMALL_ORACLE"][
            "required_checks"
        ]
    ) >= {
        "FULL_STATE_SPACE_ENUMERATION",
        "EXACT_HAMILTONIAN_AND_GAUSS_KERNEL",
        "FULL_SCREEN_J_ALL_REPLAY",
        "INJECTED_NEGATIVE_CONTROL_SUITE",
    }
    assert set(
        report["nonphysical_campaign_gates"]["SCALE_CAMPAIGN_ALLOWED"][
            "required_checks"
        ]
    ) >= {
        "EXACT_SMALL_ORACLE_PASS",
        "SAME_VERIFIER_IDENTITY",
        "FROZEN_THRESHOLD_POLICY",
        "FROZEN_SCALE_GRID",
    }
    assert "PHYSICAL_IDENTIFICATION" in report["issue_closure"]["569"][
        "required_stages"
    ]
    assert "passed" not in report["issue_closure"]["590"]


def test_structural_and_q2_pass_cannot_promote_interacting_or_continuum_claims():
    stages = {stage_id: {"passed": False} for stage_id in STAGE_IDS}
    for stage_id in BASE_GLOBAL_PASS_STAGES:
        stages[stage_id]["passed"] = True
    stages["Q2_H"]["passed"] = True
    stages["Q4_OS"]["passed"] = True

    tiers = _compute_claim_tiers(stages)

    assert tiers["Q2_PHYSICAL_BRANCH_PASS"] is True
    assert tiers["PHYSICAL_A5_SM_STRUCTURAL_GLOBAL_PASS"] is True
    assert tiers["PHYSICAL_A5_SM_FULL_INTERACTING_PASS"] is False
    assert tiers["NONPERTURBATIVE_CONTINUUM_WIGHTMAN_PHYSICAL_PASS"] is False

    for stage_id in FULL_INTERACTING_PASS_STAGES:
        stages[stage_id]["passed"] = True
    tiers = _compute_claim_tiers(stages)
    assert tiers["PHYSICAL_A5_SM_FULL_INTERACTING_PASS"] is True
    assert tiers["NONPERTURBATIVE_CONTINUUM_WIGHTMAN_PHYSICAL_PASS"] is True


def test_top_level_report_status_includes_the_selected_claim_scope() -> None:
    stages = {
        stage_id: {"status": TerminalStatus.OPEN.value}
        for stage_id in STAGE_IDS
    }
    for stage_id in BASE_GLOBAL_PASS_STAGES:
        stages[stage_id]["status"] = TerminalStatus.PASS.value

    assert (
        _report_status_for_scope(
            stages,
            scope=ClaimScope.STRUCTURAL,
            q2_status=TerminalStatus.PASS,
        )
        is TerminalStatus.PASS
    )
    assert (
        _report_status_for_scope(
            stages,
            scope=ClaimScope.FULL_INTERACTING,
            q2_status=TerminalStatus.PASS,
        )
        is TerminalStatus.OPEN
    )
    for stage_id in FULL_INTERACTING_PASS_STAGES:
        stages[stage_id]["status"] = TerminalStatus.PASS.value
    assert (
        _report_status_for_scope(
            stages,
            scope=ClaimScope.FULL_INTERACTING,
            q2_status=TerminalStatus.PASS,
        )
        is TerminalStatus.PASS
    )
    assert (
        _report_status_for_scope(
            stages,
            scope=ClaimScope.CONTINUUM,
            q2_status=TerminalStatus.PASS,
        )
        is TerminalStatus.OPEN
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("all_dependencies_passed", False),
        ("any_dependency_group_passed", False),
        ("evidence_passed", False),
        ("production_identity_passed", False),
        ("production_identity_passed", 1),
        ("evidence_passed", "true"),
    ],
)
def test_stage_pass_requires_four_literal_true_gates(field: str, value: object) -> None:
    gates: dict[str, object] = {
        "all_dependencies_passed": True,
        "any_dependency_group_passed": True,
        "evidence_passed": True,
        "production_identity_passed": True,
    }
    gates[field] = value

    assert _physical_stage_pass_authorized(**gates) is False  # type: ignore[arg-type]


def test_bare_q2_e_does_not_promote_the_structural_physical_pass():
    stages = {stage_id: {"passed": False} for stage_id in STAGE_IDS}
    for stage_id in BASE_GLOBAL_PASS_STAGES:
        stages[stage_id]["passed"] = True
    stages["Q2_E"]["passed"] = True

    tiers = _compute_claim_tiers(stages)

    assert tiers["Q2_PHYSICAL_BRANCH_PASS"] is False
    assert tiers["PHYSICAL_A5_SM_STRUCTURAL_GLOBAL_PASS"] is False


def test_q2_e_plus_positivity_promotes_the_q2_branch_only():
    stages = {stage_id: {"passed": False} for stage_id in STAGE_IDS}
    stages["Q2_E"]["passed"] = True
    stages["POSITIVITY_OR_POSITIVE_TRANSFER"]["passed"] = True

    tiers = _compute_claim_tiers(stages)

    assert tiers["Q2_PHYSICAL_BRANCH_PASS"] is True
    assert tiers["PHYSICAL_A5_SM_STRUCTURAL_GLOBAL_PASS"] is False
    assert "Q2_E_MANDATORY_UPSTREAM_OBJECT_BINDING_RECEIPT" in Q2_E_RECEIPTS
    assert "Q2_H_MANDATORY_UPSTREAM_OBJECT_BINDING_RECEIPT" in Q2_H_RECEIPTS


def test_root_manifest_must_be_on_disk_not_a_caller_mapping(tmp_path: Path):
    report = verify_physical_a5_sm_root_manifest({"schema": ROOT_MANIFEST_SCHEMA})  # type: ignore[arg-type]

    assert report["passed"] is False
    assert any("on_disk_file" in blocker for blocker in report["blockers"])


def test_path_escape_is_rejected(tmp_path: Path):
    manifest = _build_valid_root(tmp_path / "bundle")
    outside = tmp_path / "outside.json"
    _write_json(outside, {"outside": True})

    def mutate(payload: dict) -> None:
        payload["artifacts"][0]["path"] = "../outside.json"
        payload["artifacts"][0]["sha256"] = _sha256(outside)

    _mutate_manifest(manifest, mutate)
    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["passed"] is False
    assert report["receipts"]["ROOT_ARTIFACT_CONTAINMENT_RECEIPT"] is False


def test_hash_drift_invalidates_root(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = tmp_path / payload["artifacts"][0]["path"]
    _write_json(target, {"mutated": True})

    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["passed"] is False
    assert report["receipts"]["ROOT_ARTIFACT_HASH_REPLAY_RECEIPT"] is False
    assert any("declared_sha256_mismatch" in blocker for blocker in report["blockers"])


def test_target_selector_flip_invalidates_root(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)

    def mutate(payload: dict) -> None:
        payload["target_selection_dependencies"][TARGET_SELECTION_FIELDS[2]] = True

    _mutate_manifest(manifest, mutate)
    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["passed"] is False
    assert report["receipts"]["ROOT_TARGET_SELECTION_DECLARATIONS_RECEIPT"] is False
    assert any("target_selection_dependency_not_false" in item for item in report["blockers"])


def test_source_ancestry_cycle_invalidates_root(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)

    def mutate(payload: dict) -> None:
        payload["source_ancestry_edges"].append(
            {
                "parent_artifact_id": "derived.response",
                "child_artifact_id": "derived.response",
                "operation_artifact_id": "dynamics.generator",
            }
        )

    _mutate_manifest(manifest, mutate)
    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["passed"] is False
    assert report["receipts"]["ROOT_SOURCE_ANCESTRY_DAG_RECEIPT"] is False
    assert "source_ancestry_cycle_detected" in report["source_ancestry"]["blockers"]


def test_derived_array_missing_ancestry_invalidates_root(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)

    def mutate(payload: dict) -> None:
        payload["source_ancestry_edges"] = []

    _mutate_manifest(manifest, mutate)
    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["passed"] is False
    assert "derived_array_missing_ancestry:derived.response" in report[
        "source_ancestry"
    ]["blockers"]


def test_forged_all_true_lane_json_is_unadmitted(tmp_path: Path):
    forged = {
        "passed": True,
        "GEOMETRY_565_PHYSICAL_RECEIPT": True,
        "CURRENT_566_PHYSICAL_RECEIPT": True,
        "FAMILY_ATTACHMENT_569_PHYSICAL_RECEIPT": True,
        "Q2_H_PHYSICAL_RECEIPT": True,
        "PHYSICAL_IDENTIFICATION_RECEIPT": True,
    }
    manifest = _build_valid_root(tmp_path, extra_evidence=forged)

    report = verify_physical_a5_sm_requirements(manifest)
    evidence = next(
        row
        for row in report["embedded_evidence_inventory"]
        if row["artifact_id"] == "candidate.evidence"
    )

    assert report["stages"]["ROOT"]["status"] == TerminalStatus.OPEN.value
    assert report["stages"]["ROOT"]["passed"] is False
    assert evidence["classification"] == "unadmitted"
    assert evidence["physical_receipt_admission"] is False
    assert "caller_authored_receipt_booleans" in evidence["reason"]
    assert report["receipts"]["PHYSICAL_A5_SM_GLOBAL_PASS"] is False


def test_open_family_pole_fixture_is_diagnostic_only(tmp_path: Path):
    open_fixture = {
        "schema": "PHYSICAL-FAMILY-POLE-RECEIPT-v1",
        "receipt_kind": "OPEN_CHECKLIST",
        "empirical_verified": False,
        "receipt_verdict": "OPEN",
        "FAMILY_ATTACHMENT_569_PHYSICAL_RECEIPT": True,
    }
    manifest = _build_valid_root(tmp_path, extra_evidence=open_fixture)

    report = verify_physical_a5_sm_requirements(manifest)
    evidence = next(
        row
        for row in report["embedded_evidence_inventory"]
        if row["artifact_id"] == "candidate.evidence"
    )

    assert evidence["classification"] == "diagnostic_only"
    assert "family_pole_contract" in evidence["reason"]
    assert report["stages"]["FAMILY_ATTACHMENT_569"]["passed"] is False
    assert report["issue_closure"]["569"]["passed"] is False


def test_demo_assumption_is_visualization_only_and_nonpromoting(tmp_path: Path):
    manifest = _build_valid_root(
        tmp_path,
        extra_evidence={
            "run_mode": "DEMO_ASSUMPTION",
            "visualization_only": True,
            "passed": True,
        },
    )

    report = verify_physical_a5_sm_requirements(manifest)
    evidence = next(
        row
        for row in report["embedded_evidence_inventory"]
        if row["artifact_id"] == "candidate.evidence"
    )

    assert evidence["classification"] == "diagnostic_only"
    assert "visualization_only" in evidence["reason"]
    assert report["passed"] is False
    assert report["nonphysical_campaign_gates"]["SCALE_CAMPAIGN_ALLOWED"][
        "passed"
    ] is False


@pytest.mark.parametrize(
    "label",
    ["FORCED_RECEIPT", "FROZEN_TARGET_VALUE", "SYNTHETIC_PLACEHOLDER"],
)
def test_force_and_target_freeze_labels_are_diagnostic_only(
    tmp_path: Path,
    label: str,
) -> None:
    manifest = _build_valid_root(
        tmp_path,
        extra_evidence={
            "receipt_type": label,
            "passed": True,
        },
    )

    report = verify_physical_a5_sm_requirements(manifest)
    evidence = next(
        row
        for row in report["embedded_evidence_inventory"]
        if row["artifact_id"] == "candidate.evidence"
    )

    assert evidence["classification"] == "diagnostic_only"
    assert evidence["physical_receipt_admission"] is False
    assert report["passed"] is False


def test_stored_root_report_is_replayed_exactly(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)
    report_path = write_physical_a5_sm_root_report(manifest)

    initial = verify_physical_a5_sm_root_report_file(report_path)
    assert initial["status"] == TerminalStatus.OPEN.value
    assert initial["passed"] is False
    assert initial["inventory_replay_passed"] is True

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["packet_id"] = "forged.packet"
    _write_json(report_path, payload)
    verification = verify_physical_a5_sm_root_report_file(report_path)

    assert verification["passed"] is False
    assert verification["exact_report_replay"] is False


def test_candidate_domain_cannot_be_used_as_ancestry_operation(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)

    def mutate(payload: dict) -> None:
        for artifact in payload["artifacts"]:
            if artifact["semantic_role"] == "candidate_domain":
                artifact["provenance_kind"] = "source_operation"
        payload["source_ancestry_edges"][0]["operation_artifact_id"] = "candidate.domain"

    _mutate_manifest(manifest, mutate)
    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["status"] == TerminalStatus.FAIL.value
    assert report["inventory_passed"] is False
    assert any(
        "required_role_provenance_mismatch:candidate_domain:source_primitive" in blocker
        for blocker in report["blockers"]
    )


def test_family_stage_exposes_two_noninterchangeable_open_routes(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)
    report = verify_physical_a5_sm_requirements(
        manifest, claim_scope=ClaimScope.FULL_INTERACTING
    )
    family = report["stages"]["FAMILY_BREAKING_OR_DESCENT"]

    assert family["status"] == TerminalStatus.OPEN.value
    assert set(family["routes"]) == {
        "exact_surviving_family_symmetry_compatible",
        "target_blind_family_breaking_or_descent",
    }
    exact = family["routes"]["exact_surviving_family_symmetry_compatible"]
    breaking = family["routes"]["target_blind_family_breaking_or_descent"]
    assert exact["status"] == TerminalStatus.OPEN.value
    assert breaking["status"] == TerminalStatus.OPEN.value
    assert "FAMILY_BREAKING_CANDIDATE_DOMAIN_PREOUTCOME_FREEZE_RECEIPT" not in exact[
        "receipts"
    ]
    assert not any(key.startswith("FAMILY_BREAKING_") for key in exact["receipts"])
    assert "FAMILY_BREAKING_CANDIDATE_DOMAIN_PREOUTCOME_FREEZE_RECEIPT" in breaking[
        "receipts"
    ]


def test_generic_stage_envelope_requires_same_root_and_never_promotes_science(
    tmp_path: Path,
):
    root_manifest = _build_valid_root(tmp_path / "root")
    root_report = verify_physical_a5_sm_root_manifest(root_manifest)
    expected_root_hash = root_report["manifest_sha256"]
    other_root_hash = "sha256:" + "b" * 64
    bundle = _build_common_stage_production_bundle(
        tmp_path / "production",
        source_root_hash=other_root_hash,
    )

    matching_inventory = verify_physical_a5_sm_stage_envelope(
        bundle,
        expected_root_hash=other_root_hash,
        expected_stage_id="GEOMETRY_565",
        expected_claim_scope=ClaimScope.STRUCTURAL,
    )
    mixed_root = verify_physical_a5_sm_stage_envelope(
        bundle,
        expected_root_hash=expected_root_hash,
        expected_stage_id="GEOMETRY_565",
        expected_claim_scope=ClaimScope.STRUCTURAL,
    )
    full_report = verify_physical_a5_sm_requirements(
        root_manifest,
        production_bundle_manifest_path=bundle,
    )

    assert matching_inventory["status"] == TerminalStatus.OPEN.value
    assert matching_inventory["inventory_identity_passed"] is True
    assert matching_inventory["scientific_admission_passed"] is False
    assert matching_inventory["promotion_allowed"] is False
    assert matching_inventory["passed"] is False
    assert mixed_root["status"] == TerminalStatus.FAIL.value
    assert mixed_root["inventory_identity_passed"] is False
    assert "stage_envelope_source_root_mismatch" in mixed_root["blockers"]
    geometry = full_report["stages"]["GEOMETRY_565"]
    assert geometry["status"] == TerminalStatus.FAIL.value
    assert geometry["passed"] is False
    assert geometry["production_envelope_admission"]["promotion_allowed"] is False


def test_stage_admission_rejects_mixed_scope_and_stage_lane_reports() -> None:
    root_hash = "sha256:" + "c" * 64
    base_row = {
        "profile": "COMMON_STAGE",
        "claim_lane": "GEOMETRY_565",
        "claim_scope": "structural",
        "stage_id": "GEOMETRY_565",
        "source_root_hash": root_hash,
        "branch_id": "branch-a",
        "inventory_replay_passed": True,
        "scientific_replay_passed": False,
        "promotion_allowed": False,
        "producer_status_trusted": False,
        "receipt_type": "A5_STAGE_EVIDENCE",
    }
    report = {
        "schema": a5.PRODUCTION_BUNDLE_REPORT_SCHEMA,
        "artifact_type": a5.PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE,
        "inventory_replay_passed": True,
        "scientific_replay_passed": False,
        "promotion_allowed": False,
        "envelopes": {
            "geometry": base_row,
            "mixed": {
                **base_row,
                "claim_lane": "CURRENT_566",
                "claim_scope": "full_interacting",
                "stage_id": "WRONG_STAGE",
            },
        },
    }

    admission = a5._stage_envelope_admission_from_report(
        report,
        expected_root_hash=root_hash,
        expected_stage_id="GEOMETRY_565",
        expected_claim_scope=ClaimScope.STRUCTURAL,
        expected_branch_id="branch-a",
    )

    assert admission["status"] == TerminalStatus.FAIL.value
    assert admission["inventory_identity_passed"] is False
    assert "production_bundle_mixed_or_wrong_claim_scope" in admission["blockers"]
    assert "production_bundle_stage_lane_identity_mismatch" in admission["blockers"]


@pytest.mark.parametrize(
    "receipt_type",
    ["DEMO_ASSUMPTION", "FORCED_RECEIPT", "FROZEN_TARGET_VALUE"],
)
def test_stage_demo_or_force_receipt_is_unresolved_not_physical(
    receipt_type: str,
) -> None:
    root_hash = "sha256:" + "d" * 64
    row = {
        "profile": "COMMON_STAGE",
        "claim_lane": "GEOMETRY_565",
        "claim_scope": "structural",
        "stage_id": "GEOMETRY_565",
        "source_root_hash": root_hash,
        "branch_id": "branch-a",
        "inventory_replay_passed": True,
        "scientific_replay_passed": False,
        "promotion_allowed": False,
        "producer_status_trusted": False,
        "receipt_type": receipt_type,
    }
    report = {
        "schema": a5.PRODUCTION_BUNDLE_REPORT_SCHEMA,
        "artifact_type": a5.PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE,
        "inventory_replay_passed": True,
        "scientific_replay_passed": False,
        "promotion_allowed": False,
        "envelopes": {"geometry": row},
    }

    admission = a5._stage_envelope_admission_from_report(
        report,
        expected_root_hash=root_hash,
        expected_stage_id="GEOMETRY_565",
        expected_claim_scope=ClaimScope.STRUCTURAL,
        expected_branch_id="branch-a",
    )

    assert admission["status"] == TerminalStatus.UNRESOLVED.value
    assert admission["inventory_identity_passed"] is False
    assert admission["promotion_allowed"] is False
    assert "demo_assumption_is_visualization_only" in admission["blockers"]


def test_structural_scope_marks_interacting_and_continuum_stages_not_applicable(
    tmp_path: Path,
):
    manifest = _build_valid_root(tmp_path)
    report = verify_physical_a5_sm_requirements(manifest)

    assert report["stages"]["VERTEX_1PI"]["status"] == TerminalStatus.NOT_APPLICABLE.value
    assert report["stages"]["Q4_OS"]["status"] == TerminalStatus.NOT_APPLICABLE.value
    assert report["stages"]["GEOMETRY_565"]["status"] == TerminalStatus.OPEN.value


def test_root_report_must_stay_beside_manifest(tmp_path: Path):
    manifest = _build_valid_root(tmp_path / "bundle")
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(RootManifestError, match="must_be_beside_manifest"):
        write_physical_a5_sm_root_report(manifest, outside / "report.json")


def test_root_report_output_symlink_is_rejected(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)
    target = tmp_path / "report.json"
    target.symlink_to(manifest.name)

    with pytest.raises(RootManifestError, match="output_is_symlink"):
        write_physical_a5_sm_root_report(manifest, target)


def test_root_report_is_append_only(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)
    target = write_physical_a5_sm_root_report(manifest)

    with pytest.raises(RootManifestError, match="output_already_exists"):
        write_physical_a5_sm_root_report(manifest, target)


def test_duplicate_json_key_in_manifest_is_rejected(tmp_path: Path):
    manifest = _build_valid_root(tmp_path)
    raw = manifest.read_text(encoding="utf-8")
    manifest.write_text(raw.replace('{"artifacts":', '{"schema":"duplicate","artifacts":', 1))

    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["status"] == TerminalStatus.FAIL.value
    assert any("duplicate_json_key:schema" in blocker for blocker in report["blockers"])


def test_overflow_to_nonfinite_json_number_is_rejected(tmp_path: Path) -> None:
    manifest = _build_valid_root(tmp_path)
    raw = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        raw.replace('"precision_bits": 128', '"precision_bits": 1e999', 1),
        encoding="utf-8",
    )

    report = verify_physical_a5_sm_root_manifest(manifest)

    assert report["status"] == TerminalStatus.FAIL.value
    assert any("nonfinite_json_number" in blocker for blocker in report["blockers"])
