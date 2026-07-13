from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oph_fpe.bulk.particle_contract import particle_promotion_contract_report
from oph_fpe.bulk.proof_certificate import bulk_proof_certificate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_evidence(run: Path, *, candidate_kind: str = "neutral") -> dict:
    proto = {
        "localized_shared_bulk_support": True,
        "bulk_localization_margin": 0.2,
        "maximum_support_fraction": 0.25,
        "stable_topological_sector_charge": True,
        "maximum_sector_charge_variation": 1.0e-8,
        "contractible_path_transport": True,
        "maximum_contractible_holonomy_residual": 1.0e-8,
        "fusion_charge_conservation": True,
        "maximum_fusion_charge_residual": 1.0e-8,
        "scattering_reproducibility": True,
        "minimum_scattering_replay_agreement": 1.0,
        "speed_causality_controls": True,
        "maximum_causality_violation": 0.0,
        "observer_resampling_stability": True,
        "maximum_observer_resampling_drift": 1.0e-8,
        "refinement_stability": True,
        "maximum_refinement_drift": 1.0e-8,
        "worldline_count": 3,
        "independent_seed_count": 4,
        "refinement_level_count": 3,
    }
    classical = {
        "background": "source-free vacuum",
        "phase": "deconfined" if candidate_kind == "colored" else "ordinary vacuum",
        "quadratic_action_explicit": True,
        "quadratic_action_rank": 4,
        "physical_kinetic_coefficient": 0.5,
        "constraint_reduction": "brst_cohomology",
        "physical_projector_rank": 2,
        "reduced_hamiltonian_minimum": 0.1,
        "hessian_wave_operator_residual": 1.0e-8,
        "dispersion_speed": 1.0,
        "forbidden_mass_operator_norm": 0.0,
    }
    quantum = {
        "vacuum_quantization": "brst",
        "vacuum_energy_lower_bound": 0.0,
        "physical_hilbert_construction": "brst_cohomology",
        "negative_norm_state_count": 0,
        "pole_residue": 0.8,
        "spectral_negative_weight": 0.0,
        "positive_energy_shell_minimum": 0.1,
        "mass_shell_residual": 1.0e-8,
        "asymptotic_state_norm": 1.0,
        "decay_width_upper_bound": 0.0,
        "lsz_residue": 0.8,
        "deconfined_asymptotic_sector": candidate_kind == "colored",
    }
    refs = {}
    for name, payload in (
        ("proto", proto),
        ("classical_carrier", classical),
        ("quantum_particle", quantum),
    ):
        path = run / "particle_evidence" / f"{name}.json"
        _write_json(path, payload)
        refs[name] = {"path": str(path.relative_to(run)), "sha256": _sha256(path)}
    return {
        "schema": "oph_particle_promotion_evidence_v1",
        "candidate_id": "candidate-001",
        "candidate_kind": candidate_kind,
        "structural_speed": 1.0,
        "tolerance": 1.0e-6,
        "proto": proto,
        "classical_carrier": classical,
        "quantum_particle": quantum,
        "provenance": {
            "generator": "independent-particle-fixture",
            "source_commit": "0123456789abcdef0123456789abcdef01234567",
            "source_dirty": False,
            "generated_utc": "2026-07-12T00:00:00Z",
            "lane_artifacts": refs,
        },
    }


def test_complete_self_authored_particle_evidence_remains_nonpromoting(tmp_path: Path):
    evidence = _valid_evidence(tmp_path)
    _write_json(tmp_path / "particle_promotion_evidence.json", evidence)

    report = particle_promotion_contract_report(tmp_path)

    assert all(report["candidate_lane_contracts"].values())
    assert report["BULK_WORLDLINE_PRECURSOR_RECEIPT"] is False
    assert report["CLASSICAL_CARRIER_MODE_RECEIPT"] is False
    assert report["QUANTUM_PARTICLE_RECEIPT"] is False
    assert report["PRODUCTION_PARTICLE_MATTER_RECEIPT"] is False
    assert report["physical_claim"] is False
    assert "runtime_binding:runtime_particle_evidence_producer_not_implemented" in report["blockers"]


def test_particle_contract_rejects_truthy_nonbooleans_and_forged_top_level_flags(tmp_path: Path):
    evidence = _valid_evidence(tmp_path)
    evidence["particle_matter_receipt"] = True
    evidence["PRODUCTION_PARTICLE_MATTER_RECEIPT"] = "true"
    evidence["proto"]["fusion_charge_conservation"] = 1
    _write_json(tmp_path / "particle_promotion_evidence.json", evidence)

    report = particle_promotion_contract_report(tmp_path)

    assert report["PRODUCTION_PARTICLE_MATTER_RECEIPT"] is False
    assert report["lanes"]["P0_proto_worldline"]["checks"]["fusion_charge_conservation"] is False
    assert report["ignored_caller_promotion_fields"]["particle_matter_receipt"] is True


def test_particle_contract_rejects_missing_quantum_primitives_even_with_valid_sidecar_hash(tmp_path: Path):
    evidence = _valid_evidence(tmp_path)
    evidence["quantum_particle"].pop("pole_residue")
    sidecar = tmp_path / evidence["provenance"]["lane_artifacts"]["quantum_particle"]["path"]
    _write_json(sidecar, evidence["quantum_particle"])
    evidence["provenance"]["lane_artifacts"]["quantum_particle"]["sha256"] = _sha256(sidecar)
    _write_json(tmp_path / "particle_promotion_evidence.json", evidence)

    report = particle_promotion_contract_report(tmp_path)

    assert report["candidate_lane_contracts"]["P0_proto_worldline"] is True
    assert report["candidate_lane_contracts"]["classical_carrier_mode"] is True
    assert report["candidate_lane_contracts"]["quantum_particle"] is False
    assert report["BULK_WORLDLINE_PRECURSOR_RECEIPT"] is False
    assert report["CLASSICAL_CARRIER_MODE_RECEIPT"] is False
    assert report["QUANTUM_PARTICLE_RECEIPT"] is False
    assert report["PRODUCTION_PARTICLE_MATTER_RECEIPT"] is False


def test_colored_particle_requires_deconfined_asymptotic_sector(tmp_path: Path):
    evidence = _valid_evidence(tmp_path, candidate_kind="colored")
    evidence["quantum_particle"]["deconfined_asymptotic_sector"] = False
    sidecar = tmp_path / evidence["provenance"]["lane_artifacts"]["quantum_particle"]["path"]
    _write_json(sidecar, evidence["quantum_particle"])
    evidence["provenance"]["lane_artifacts"]["quantum_particle"]["sha256"] = _sha256(sidecar)
    _write_json(tmp_path / "particle_promotion_evidence.json", evidence)

    report = particle_promotion_contract_report(tmp_path)

    assert report["COLORED_DECONFINEMENT_RECEIPT"] is False
    assert report["PRODUCTION_PARTICLE_MATTER_RECEIPT"] is False
    assert "quantum:colored_deconfinement_missing" in report["blockers"]


def test_bulk_certificate_ignores_legacy_particle_booleans(tmp_path: Path):
    _write_json(tmp_path / "emergence_status_report.json", {"particle_matter_receipt": True})
    _write_json(tmp_path / "particle_likeness_report.json", {"particle_matter_receipt": True})
    _write_json(
        tmp_path / "controlled_defect_particle_assay_report.json",
        {"physical_particle_emergence": True},
    )

    report = bulk_proof_certificate(tmp_path)

    assert report["production_particle_matter_receipt"] is False
    assert report["proof_tiers"]["P1_production_particle_matter"]["passed"] is False
    assert report["particle_promotion_contract_summary"]["ignored_legacy_producer_fields"] == {
        "emergence_status_report.particle_matter_receipt": True,
        "particle_likeness_report.particle_matter_receipt": True,
        "controlled_defect_particle_assay_report.physical_particle_emergence": True,
    }


def test_bulk_certificate_refuses_unbound_complete_particle_contract(tmp_path: Path):
    evidence = _valid_evidence(tmp_path)
    _write_json(tmp_path / "particle_promotion_evidence.json", evidence)
    _write_json(tmp_path / "emergence_status_report.json", {"particle_matter_receipt": False})

    report = bulk_proof_certificate(tmp_path)

    assert report["bulk_worldline_precursor_receipt"] is False
    assert report["classical_carrier_mode_receipt"] is False
    assert report["quantum_particle_receipt"] is False
    assert report["production_particle_matter_receipt"] is False
