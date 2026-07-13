from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.cosmology.hadron_source_backend import (
    CLAIM_TIERS,
    FORBIDDEN_SOURCE_INPUTS,
    REQUIRED_FILES,
    HadronSourceBackendInputs,
    hadron_source_backend_report,
    write_hadron_source_backend_bundle,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_valid_evidence_bundle(root: Path) -> None:
    write_hadron_source_backend_bundle(root)
    payloads = {
        "source_dag.json": {"source_inputs": ["P", "source QCD parameter map"], "forbidden_edges": []},
        "qcd_ensemble/quotient_schema.json": {"quotient_defined": True, "finite_regulator": True},
        "qcd_ensemble/gamma_groupoid.json": {
            "groupoid_action_verified": True,
            "inert_labels_quotiented": True,
        },
        "qcd_ensemble/base_measure.json": {
            "base_measure_normalized": True,
            "positive_measure": True,
            "total_weight": 1.0,
        },
        "qcd_ensemble/source_action.json": {
            "source_action_explicit": True,
            "action_bounded_below": True,
            "action_lower_bound": 0.0,
        },
        "qcd_ensemble/source_parameter_map.json": {
            "source_parameter_map_complete": True,
            "parameters": {
                "g3": 1.1,
                "theta_qcd": 0.0,
                "quark_masses": {"u": 0.002, "d": 0.005, "s": 0.095},
                "renormalization_scheme": "MSbar-2GeV",
            },
            "forbidden_target_inputs": [],
        },
        "vacuum/euclidean_slab.json": {"euclidean_slab_constructed": True, "time_slices": 32},
        "vacuum/transfer_operator.json": {
            "transfer_operator_constructed": True,
            "minimum_eigenvalue": 0.01,
        },
        "vacuum/reflection_positivity.json": {
            "positivity_tested": True,
            "minimum_eigenvalue": -1.0e-10,
            "tolerance": 1.0e-8,
            "sample_count": 32,
        },
        "vacuum/vacuum_promotion.json": {
            "hadronic_hilbert_quotient_constructed": True,
            "null_space_quotiented": True,
            "positive_transfer_operator": True,
        },
        "currents/ward_current_definition.json": {"ward_current_defined": True},
        "currents/current_normalization_ZV.json": {"z_v": 0.98, "ward_identity_verified": True},
        "currents/contact_terms.json": {"contact_terms_accounted": True},
        "correlators/vector_current_2pt_raw.json": {
            "two_point_correlator_computed": True,
            "sample_count": 128,
        },
        "correlators/vector_current_2pt_covariance.json": {"covariance_psd": True},
        "correlators/disconnected_diagrams.json": {"disconnected_diagrams_included": True},
        "correlators/autocorrelation_report.json": {"effective_sample_size": 64.0},
        "spectral/moments.json": {"moments": [1.0, 0.5, 0.3]},
        "spectral/hankel_positivity.json": {
            "positivity_tested": True,
            "minimum_eigenvalue": 0.01,
            "tolerance": 1.0e-8,
            "sample_count": 3,
        },
        "spectral/stieltjes_bounds.json": {
            "stieltjes_bounds_verified": True,
            "lower": 0.1,
            "upper": 0.2,
        },
        "spectral/J24Q.json": {"j24q": 0.15},
        "spectral/omegaQ.json": {"omega_q": 1.0},
        "spectral/spectral_interval.json": {"lower": 0.1, "upper": 0.2},
        "endpoint/Xi_same_scheme.json": {
            "same_scheme_remainder_computed": True,
            "scheme": "MSbar-2GeV",
            "lower": 0.01,
            "upper": 0.02,
        },
        "endpoint/Delta_had_interval.json": {"scheme": "MSbar-2GeV", "lower": 0.02, "upper": 0.03},
        "endpoint/ATh_interval.json": {"scheme": "MSbar-2GeV", "lower": 0.03, "upper": 0.04},
        "endpoint/pixel_contraction_interval.json": {
            "scheme": "MSbar-2GeV",
            "lower": 1.62,
            "upper": 1.64,
        },
        "higher_point/Q4_HLbL_receipt.json": {
            "spectral_export_receipt": True,
            "positive_measure_receipt": True,
            "sample_count": 32,
        },
        "higher_point/transition_B_to_K_receipt.json": {
            "spectral_export_receipt": True,
            "positive_measure_receipt": True,
            "sample_count": 32,
        },
        "higher_point/transition_Sigma_to_p_receipt.json": {
            "spectral_export_receipt": True,
            "positive_measure_receipt": True,
            "sample_count": 32,
        },
        "controls/no_target_leak_dag.json": {
            "audit_passed": True,
            "source_frozen_before_comparison": True,
            "forbidden_edges": [],
        },
        "controls/empirical_data_exclusion_manifest.json": {
            "forbidden_inputs_present": [],
            "source_frozen_before_comparison": True,
        },
        "controls/frozen_code_hashes.json": {
            "code_hashes": {"solver": "sha256:" + "a" * 64}
        },
        "controls/replay_receipts.json": {
            "independent_replay_passed": True,
            "deterministic_replay_passed": True,
            "replay_count": 2,
        },
        "controls/comparison_data_manifest.json": {
            "comparison_data": [],
            "attached_after_source_freeze": True,
        },
        "controls/systematics_ledger.json": {
            "components": {
                "statistical": 0.01,
                "discretization": 0.01,
                "finite_volume": 0.01,
                "renormalization": 0.01,
                "continuum": 0.01,
            },
            "total_uncertainty": 0.03,
            "all_systematics_bounded": True,
            "refinement_level_count": 3,
            "independent_seed_count": 3,
        },
    }
    for rel_path, payload in payloads.items():
        _write_json(root / rel_path, payload)
    (root / "currents/ward_residuals.csv").write_text(
        "momentum,residual,bound,status\n0.1,1e-9,1e-8,PASS\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "oph_qcd_source_promotion_evidence_v1",
        "artifact": "oph_qcd_source_promotion_evidence",
        "generator": "independent-hadron-evidence-fixture",
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "source_dirty": False,
        "generated_utc": "2026-07-12T00:00:00Z",
        "file_hashes": {},
    }
    for rel_path in REQUIRED_FILES:
        if rel_path == "manifest.json":
            continue
        manifest["file_hashes"][rel_path] = "sha256:" + hashlib.sha256(
            (root / rel_path).read_bytes()
        ).hexdigest()
    _write_json(root / "manifest.json", manifest)


def _refresh_manifest_hash(root: Path, rel_path: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["file_hashes"][rel_path] = "sha256:" + hashlib.sha256(
        (root / rel_path).read_bytes()
    ).hexdigest()
    _write_json(root / "manifest.json", manifest)


def test_default_hadron_source_backend_is_fail_closed():
    report = hadron_source_backend_report()

    assert report["mode"] == "oph_qcd_hadron_source_backend_v1"
    assert report["milestone"] == "HVP_ALPHA_SOURCE_PROTOTYPE"
    assert report["claim"] == "SOURCE_PROTOTYPE_NOT_PROMOTED"
    assert report["claim_tier"] == "H2"
    assert set(report["claim_tiers"]) == set(CLAIM_TIERS)
    assert report["promotion_allowed"] is False
    assert report["source_open"] is True
    assert report["forbidden_source_inputs"] == list(FORBIDDEN_SOURCE_INPUTS)
    assert report["readiness_gates"]["two_current_hadronic_backend_receipt"] is False
    assert report["readiness_gates"]["full_hadronic_precision_backend_receipt"] is False
    assert report["readiness_gates"]["fine_structure_endpoint_promotion_receipt"] is False
    assert "source_qcd_law_not_promoted" in report["blockers"]


def test_hadron_source_backend_bundle_writes_required_receipts(tmp_path: Path):
    report = write_hadron_source_backend_bundle(tmp_path)

    assert (tmp_path / "hadron_source_backend_report.json").exists()
    assert (tmp_path / "hadron_source_backend_report.md").exists()
    assert (tmp_path / "claim.md").read_text(encoding="utf-8") == "SOURCE_PROTOTYPE_NOT_PROMOTED\n"
    assert report["manifest"]["missing_files"] == []
    for rel_path in REQUIRED_FILES:
        assert (tmp_path / rel_path).exists(), rel_path
    assert "manifest.json" in report["manifest"]["file_hashes"]
    assert report["readiness_gates"]["forbidden_source_inputs_excluded"] is False


def test_source_interval_claim_cannot_self_attest_backend_receipts(tmp_path: Path):
    report = write_hadron_source_backend_bundle(
        tmp_path,
        HadronSourceBackendInputs(claim="SOURCE_INTERVAL_PROMOTED", tier="H7"),
    )

    assert report["promotion_allowed"] is False
    assert report["source_open"] is True
    assert report["readiness_gates"]["two_current_hadronic_backend_receipt"] is False
    assert report["readiness_gates"]["full_hadronic_precision_backend_receipt"] is False
    assert report["readiness_gates"]["fine_structure_endpoint_promotion_receipt"] is False
    assert "source_interval_promotion_evidence_invalid" in report["blockers"]


def test_source_interval_promotes_only_valid_hash_pinned_typed_evidence(tmp_path: Path):
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    _write_valid_evidence_bundle(evidence)

    report = write_hadron_source_backend_bundle(
        output,
        HadronSourceBackendInputs(
            claim="SOURCE_INTERVAL_PROMOTED",
            tier="H7",
            evidence_dir=evidence,
        ),
    )

    assert report["promotion_allowed"] is True
    assert report["source_open"] is False
    assert report["promotion_evidence"]["artifact_hashes_passed"] is True
    assert report["promotion_evidence"]["provenance_passed"] is True
    assert report["readiness_gates"]["two_current_hadronic_backend_receipt"] is True
    assert report["readiness_gates"]["full_hadronic_precision_backend_receipt"] is True
    assert report["readiness_gates"]["fine_structure_endpoint_promotion_receipt"] is False


def test_hadron_promotion_rejects_truthy_nonboolean_no_target_flag(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_valid_evidence_bundle(evidence)
    no_target = json.loads(
        (evidence / "controls/no_target_leak_dag.json").read_text(encoding="utf-8")
    )
    no_target["audit_passed"] = "true"
    _write_json(evidence / "controls/no_target_leak_dag.json", no_target)
    _refresh_manifest_hash(evidence, "controls/no_target_leak_dag.json")

    report = hadron_source_backend_report(
        HadronSourceBackendInputs(
            claim="SOURCE_INTERVAL_PROMOTED",
            tier="H7",
            evidence_dir=evidence,
        )
    )

    assert report["promotion_evidence"]["artifact_hashes_passed"] is True
    assert report["readiness_gates"]["qcd_no_target_leak_dag_receipt"] is False
    assert report["promotion_allowed"] is False


def test_hadron_promotion_rejects_missing_positivity_primitive(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_valid_evidence_bundle(evidence)
    positivity = json.loads(
        (evidence / "spectral/hankel_positivity.json").read_text(encoding="utf-8")
    )
    positivity.pop("minimum_eigenvalue")
    _write_json(evidence / "spectral/hankel_positivity.json", positivity)
    _refresh_manifest_hash(evidence, "spectral/hankel_positivity.json")

    report = hadron_source_backend_report(
        HadronSourceBackendInputs(
            claim="SOURCE_INTERVAL_PROMOTED",
            tier="H7",
            evidence_dir=evidence,
        )
    )

    assert report["promotion_evidence"]["artifact_hashes_passed"] is True
    assert report["readiness_gates"]["two_current_spectral_export_receipt"] is False
    assert report["promotion_allowed"] is False


def test_hadron_promotion_rejects_tampered_artifact_hash(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_valid_evidence_bundle(evidence)
    (evidence / "claim.md").write_text("tampered\n", encoding="utf-8")

    report = hadron_source_backend_report(
        HadronSourceBackendInputs(
            claim="SOURCE_INTERVAL_PROMOTED",
            tier="H7",
            evidence_dir=evidence,
        )
    )

    assert report["promotion_evidence"]["artifact_hashes_passed"] is False
    assert report["promotion_allowed"] is False


def test_hadron_writer_will_not_overwrite_its_evidence_bundle(tmp_path: Path):
    _write_valid_evidence_bundle(tmp_path)

    with pytest.raises(ValueError, match="evidence_dir must be separate"):
        write_hadron_source_backend_bundle(
            tmp_path,
            HadronSourceBackendInputs(
                claim="SOURCE_INTERVAL_PROMOTED",
                tier="H7",
                evidence_dir=tmp_path,
            ),
        )


def test_hadron_source_backend_lazy_export_and_measurement_pack(tmp_path: Path):
    from oph_fpe.cosmology import hadron_source_backend_report as exported_report
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    pack_dir = tmp_path / "pack"
    write_hadron_source_backend_bundle(run)
    pack = export_measurement_pack([run], pack_dir)

    assert exported_report()["mode"] == "oph_qcd_hadron_source_backend_v1"
    assert (pack_dir / "hadron_source_backend_report.json").exists()
    assert (pack_dir / "hadron_source_backend_report.md").exists()
    assert pack["claims"]["hadron_source_backend_written"] is True
    assert pack["claims"]["hadron_source_backend_claim"] == "SOURCE_PROTOTYPE_NOT_PROMOTED"
    assert pack["claims"]["hadron_source_backend_two_current_receipt"] is False
    assert pack["claims"]["hadron_source_backend_full_precision_receipt"] is False
