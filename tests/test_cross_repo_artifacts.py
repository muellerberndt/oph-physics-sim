from __future__ import annotations

import json
from pathlib import Path
import subprocess

from oph_fpe.cosmology.particle_frontier import particle_frontier_report
from oph_fpe.bulk.paper_geometry_regressions import paper_geometry_regression_report
from oph_fpe.evidence.cross_repo_artifacts import (
    ArtifactSpec,
    import_cross_repo_artifacts,
    verify_cross_repo_artifact_manifest,
)
from oph_fpe.evidence.particle_input_policy import (
    ParticleInputRecord,
    particle_input_non_circularity_report,
)


def test_cross_repo_import_is_hash_pinned_and_never_promotes(tmp_path: Path):
    source = tmp_path / "research"
    destination = tmp_path / "run"
    _write(source / "paper/release_info.tex", r"\newcommand{\OPHPaperReleaseID}{r-test}")
    specs = _fixture_artifacts(source)
    _init_git(source)

    manifest = import_cross_repo_artifacts(source, destination, specs=specs)
    verified = verify_cross_repo_artifact_manifest(destination)
    frontier = particle_frontier_report(destination)

    assert manifest["paper_release_id"] == "r-test"
    assert manifest["source_repository"]["dirty"] is False
    assert verified["verified"] is True
    assert all(row["simulation_receipt_eligible"] is False for row in manifest["artifacts"])
    assert frontier["neutrino"]["rejection_receipt"] is True
    assert frontier["neutrino"]["oph_mass_prediction"] is None
    assert frontier["conditional_electroweak"]["display_allowed"] is True
    assert frontier["conditional_electroweak"]["simulation_input_allowed"] is False
    assert frontier["empirical_hadron_closure"]["empirical_closure_validation_receipt"] is True
    assert frontier["source_hadron_endpoint_receipt"] is False
    assert frontier["simulation_receipts_promoted_by_import"] is False


def test_checked_in_current_snapshot_is_verified_and_nonpromoting():
    root = Path("data/oph_cross_repo_current")

    verified = verify_cross_repo_artifact_manifest(root)
    frontier = particle_frontier_report()
    geometry = paper_geometry_regression_report()

    assert verified["verified"] is True
    assert frontier["neutrino"]["rejection_receipt"] is True
    assert frontier["conditional_electroweak"]["simulation_input_allowed"] is False
    assert frontier["source_hadron_endpoint_receipt"] is False
    assert geometry["all_golden_regressions_pass"] is True
    assert geometry["sim_native_geometry_receipt"] is False
    assert geometry["einstein_branch_entry_receipt"] is False


def test_manifest_verifier_rejects_tampering(tmp_path: Path):
    source = tmp_path / "research"
    destination = tmp_path / "run"
    specs = _fixture_artifacts(source)
    _init_git(source)
    manifest = import_cross_repo_artifacts(source, destination, specs=specs)
    row = next(item for item in manifest["artifacts"] if item["key"] == "conditional_ew_envelope")
    (destination / row["target_relpath"]).write_text("{}\n", encoding="utf-8")

    verified = verify_cross_repo_artifact_manifest(destination)

    assert verified["verified"] is False
    assert "artifact_hash_mismatch:conditional_ew_envelope" in verified["blockers"]


def test_source_dirty_state_is_recorded_but_does_not_change_import_role(tmp_path: Path):
    source = tmp_path / "research"
    destination = tmp_path / "run"
    specs = _fixture_artifacts(source)
    _init_git(source)
    _write(source / "untracked-note.txt", "working tree change")

    manifest = import_cross_repo_artifacts(source, destination, specs=specs)
    verified = verify_cross_repo_artifact_manifest(destination)

    assert manifest["source_repository"]["dirty"] is True
    assert manifest["source_worktree_clean"] is False
    assert manifest["run_receipts_promoted_by_import"] is False
    assert verified["verified"] is False
    assert "source_worktree_not_clean" in verified["blockers"]


def test_import_supports_schema_identity_without_fabricating_artifact_field(tmp_path: Path):
    source = tmp_path / "research"
    destination = tmp_path / "run"
    _json(source / "artifact.json", {"schema": "conditional-witness/v1", "passed": True})
    spec = ArtifactSpec(
        "schema_identity",
        "artifact.json",
        "conditional-witness/v1",
        "conditional_nonpromoting",
        identity_path=("schema",),
    )
    _init_git(source)

    manifest = import_cross_repo_artifacts(source, destination, specs=(spec,))

    assert manifest["artifacts"][0]["artifact"] == "conditional-witness/v1"
    assert manifest["artifacts"][0]["identity_path"] == ["schema"]
    assert manifest["artifacts"][0]["simulation_receipt_eligible"] is False


def test_manifest_verifier_fails_when_required_artifact_is_missing(tmp_path: Path):
    source = tmp_path / "research"
    source.mkdir()
    _write(source / "README.md", "fixture\n")
    _init_git(source)
    destination = tmp_path / "run"
    spec = ArtifactSpec(
        "required",
        "missing.json",
        "expected",
        "paper_status",
    )

    import_cross_repo_artifacts(source, destination, specs=(spec,))
    verified = verify_cross_repo_artifact_manifest(destination)

    assert verified["verified"] is False
    assert "required_artifacts_missing" in verified["blockers"]


def test_particle_input_policy_blocks_empirical_or_measured_inputs_in_generators():
    digest = "sha256:" + "a" * 64
    safe = particle_input_non_circularity_report(
        [ParticleInputRecord("ew-envelope", "conditional_compare_only_envelope", "comparison", digest)]
    )
    leaky = particle_input_non_circularity_report(
        [ParticleInputRecord("ew-envelope", "conditional_compare_only_envelope", "repair_kernel", digest)]
    )

    assert safe["reference_free_receipt"] is True
    assert leaky["reference_free_receipt"] is False
    assert "non_generative_input_entered_repair_kernel:ew-envelope" in leaky["blockers"]


def _fixture_artifacts(source: Path) -> tuple[ArtifactSpec, ...]:
    specs = (
        ArtifactSpec(
            "neutrino_lane_closure",
            "artifacts/neutrino_closure.json",
            "oph_neutrino_lane_closure_contract",
            "rejected_target_informed_candidate_status",
        ),
        ArtifactSpec(
            "neutrino_nufit61_score",
            "artifacts/neutrino_score.json",
            "oph_neutrino_nufit61_retrospective_profile_score",
            "retrospective_rejection_evidence",
        ),
        ArtifactSpec(
            "conditional_ew_envelope",
            "artifacts/ew.json",
            "oph_conditional_ew_predictions",
            "conditional_compare_only_envelope",
        ),
        ArtifactSpec(
            "empirical_hadron_spectral_measure",
            "artifacts/hadron.json",
            "oph_empirical_ward_projected_hadronic_spectral_measure",
            "empirical_external_data_closure",
        ),
    )
    _json(
        source / "artifacts/neutrino_closure.json",
        {
            "artifact": "oph_neutrino_lane_closure_contract",
            "bridge_prediction_promotion_allowed": False,
            "public_promotion_allowed": False,
        },
    )
    _json(
        source / "artifacts/neutrino_score.json",
        {
            "artifact": "oph_neutrino_nufit61_retrospective_profile_score",
            "decision": {
                "current_weighted_cycle_candidate_rejected_by_declared_gate": True,
                "threshold_delta_chi2_2d_3sigma": 11.829,
            },
            "scores": {"TBoff-NO": {"joint_fixed_candidate_delta_chi2_lower_bound": 18.435}},
        },
    )
    _json(
        source / "artifacts/ew.json",
        {
            "artifact": "oph_conditional_ew_predictions",
            "row_class": "conditional_on_P_and_repair_selection",
            "guards": {"public_promotion_allowed": False, "conditional_display_allowed": True},
            "inputs": {"P_calibration": 1.63},
            "conditional_envelope": {"mH_gev": [125.1, 125.2]},
        },
    )
    _json(
        source / "artifacts/hadron.json",
        {
            "artifact": "oph_empirical_ward_projected_hadronic_spectral_measure",
            "row_class": "oph_plus_empirical_hadron_closure",
            "profile_id": "fixture",
            "projection": {"ward_projected": True},
            "rho_had_or_measure": {
                "positivity_status": "verified_nonnegative_on_exported_grids_and_atoms"
            },
            "consistency": {"within_tolerance": True},
            "systematics": {"normalization_budget": {}},
            "transport_moments": {"timelike": {"value": 0.0267}},
            "guards": {
                "empirical_hadron_closure": True,
                "external_cross_section_data_used": True,
                "promotable_as_oph_source_theorem": False,
            },
        },
    )
    return specs


def _init_git(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)


def _json(path: Path, value: dict) -> None:
    _write(path, json.dumps(value, indent=2) + "\n")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
