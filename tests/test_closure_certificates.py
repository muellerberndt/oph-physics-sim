"""Frozen tests for the issue-633 and issue-311 closure certificates."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from oph_fpe.local_domain.classical_realization import (
    harmonic_transport_stiffness,
    serialized_interface_census,
)
from oph_fpe.local_domain.clock_unit_verdict import (
    SI_ATTACHMENT_CHANNEL,
    interface_classification,
    positive_gap_binding,
    producer_channel_nonuse_experiment,
    source_gap_proof_binding,
    walk_interface,
)
from oph_fpe.local_domain.defect_sector_spectra import sector_laplacian

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"


def test_interface_walker_classifies_and_flags():
    census, hits = walk_interface(
        {
            "count": 3,
            "gap": 0.5,
            "digest": "sha256:" + "0" * 64,
            "note": "prose",
            "nested": {"holds": True, "empty": None},
        },
        "probe",
    )
    assert census == {
        "integer_leaf": 1,
        "float_leaf": 1,
        "sha256_string": 1,
        "string_leaf": 1,
        "boolean_leaf": 1,
        "null": 1,
    }
    assert hits == []
    _, injected = walk_interface({"si_frequency_row": 1}, "probe")
    assert injected


def test_channel_nonuse_experiment_rejects_two_identical_failed_runs(
    monkeypatch,
):
    def failed_producer():
        return {"verdict": "NOT_ATTAINED", "blockers": ["stale"]}

    monkeypatch.setattr(
        "oph_fpe.local_domain.defect_sector_spectra."
        "produce_defect_sector_receipt",
        failed_producer,
    )
    monkeypatch.setenv(SI_ATTACHMENT_CHANNEL, "preexisting")
    experiment = producer_channel_nonuse_experiment()
    assert experiment["payload_hashes"]["one"] == experiment[
        "payload_hashes"
    ]["two"]
    assert not experiment["both_producer_runs_attained"]
    assert not experiment["witness_holds"]
    assert os.environ[SI_ATTACHMENT_CHANNEL] == "preexisting"


def test_interface_audits_reject_unpinned_corrupt_array(
    tmp_path, monkeypatch
):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
        "stage4_receipt.json",
        "source_gap_receipt.json",
        "defect_sector_receipt.json",
        "matter_attachment_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)
    (tmp_path / "stage1_arrays.npz.gz").write_bytes(b"not a gzip file")
    monkeypatch.setattr(
        "oph_fpe.local_domain.clock_unit_verdict.DATA_DIR", tmp_path
    )
    monkeypatch.setattr(
        "oph_fpe.local_domain.classical_realization.DATA_DIR", tmp_path
    )
    clock = interface_classification()
    classical = serialized_interface_census()
    assert "stage1_arrays.npz.gz" in clock["manifest_pin_failures"]
    assert "stage1_arrays.npz.gz" in classical["manifest_pin_failures"]
    assert clock["array_artifact"] is None
    assert classical["array_artifact"] is None


def test_positive_gap_binding_rejects_wrong_source_or_domain():
    gap = {
        "schema": "oph.source-clock-gap.v1",
        "issue": 633,
        "physical_promotion_allowed": False,
        "verdict": "ATTAINED",
        "exact_gap": {"positive": True},
        "source_projection_sha256": "sha256:source",
        "domain_freeze_sha256": "sha256:domain",
    }
    bundle = {
        "source_projection_sha256": "sha256:source",
        "domain_freeze_sha256": "sha256:domain",
    }
    assert positive_gap_binding(gap, bundle)["retained"]
    assert not positive_gap_binding(
        gap, {**bundle, "domain_freeze_sha256": "sha256:other"}
    )["retained"]
    assert not positive_gap_binding(
        gap, {**bundle, "source_projection_sha256": "sha256:other"}
    )["retained"]


def test_source_gap_proof_projection_rejects_witness_tamper():
    raw = (DATA_DIR / "source_gap_receipt.json").read_bytes()
    gap = json.loads(raw.decode("utf-8"))
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert source_gap_proof_binding(gap, digest)[
        "proof_projection_complete"
    ]
    tampered = json.loads(json.dumps(gap))
    witness = tampered["kernel_certificate"]["component_certificates"][0][
        "negative_cycle_witness"
    ]
    witness["sign_product"] = 1
    witness["negative"] = False
    assert not source_gap_proof_binding(tampered, digest)[
        "proof_projection_complete"
    ]


def test_frozen_clock_unit_verdict_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "clock_unit_verdict.json").read_bytes()
    assert manifest["clock_unit_verdict_sha256"] == (
        "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    )
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-clock-unit-verdict.v1"
    assert receipt["issue"] == 633
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == (
        "PHYSICAL_UNITS_NOT_EVALUABLE_ON_DECLARED_SERIALIZED_INTERFACE"
    )
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    classification = receipt["interface_classification"]
    assert classification["no_unit_labeled_field"] is True
    assert classification["unit_vocabulary_hits"] == []
    experiment = receipt["producer_channel_nonuse_experiment"]
    assert experiment["witness_holds"] is True
    assert experiment["both_producer_runs_attained"] is True
    assert experiment["producer_rerun"] == (
        "produce_defect_sector_receipt"
    )
    assert receipt["interface_classification"]["array_artifact"]["clean"]
    assert receipt["interface_classification"]["manifest_pin_failures"] == []
    source_scan = receipt["bounded_source_scan"]
    assert source_scan["bounded_scan_clean"] is True
    assert source_scan["main_config_within_allowlist"]
    assert (
        "bulk/physical_h3_kms_source_capture.py"
        in receipt["bounded_source_scan"]["modules_scanned"]
    )
    assert receipt["positive_clause_retained"][
        "source_hamiltonian_with_positive_gap"
    ]
    upstream_pins = receipt["upstream_pins"]
    assert set(upstream_pins) == {
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
        "stage4_receipt.json",
        "source_gap_receipt.json",
        "defect_sector_receipt.json",
        "matter_attachment_receipt.json",
        "stage1_arrays.npz.gz",
    }
    for name, digest in upstream_pins.items():
        assert digest == (
            "sha256:" + hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        )
    proof = receipt["positive_clause_retained"]["proof_binding"]
    assert proof["source_gap_receipt_sha256"] == upstream_pins[
        "source_gap_receipt.json"
    ]
    assert proof["twisted_kernel_dimension"] == 0
    assert proof["rank_theorem_applied"] is True
    assert proof["negative_cycle_witnesses_verified"] is True
    assert proof["negative_cycle_witness_count"] == proof[
        "frustrated_component_count"
    ]
    assert proof["proof_projection_complete"] is True
    assert receipt["CLOCK_UNIT_BOUNDED_INTERFACE_AUDIT"] is True
    assert receipt["source_projection_sha256"].startswith("sha256:")
    assert receipt["domain_freeze_sha256"].startswith("sha256:")


def test_harmonic_transport_stiffness_realifies_spectrum():
    def _row(index, a, b):
        return {
            "overlap_id": f"seam-{index:04d}",
            "left_carrier_id": f"carrier-{a:05d}",
            "right_carrier_id": f"carrier-{b:05d}",
            "left_ports": [0],
            "right_ports": [0],
            "orientation_signs": [-1],
            "visible_to_observer_tokens": ["observer-0000"],
            "interface_algebra_sha256": "sha256:a",
        }

    from oph_fpe.local_domain.defect_sector_spectra import oriented_seams

    oriented = oriented_seams(
        [_row(0, 0, 1), _row(1, 1, 2), _row(2, 2, 0)]
    )
    for sector in (0, 1, 3):
        complex_matrix = sector_laplacian(oriented, sector).toarray()
        complex_spectrum = np.sort(np.linalg.eigvalsh(complex_matrix))
        real_matrix = harmonic_transport_stiffness(
            sector_laplacian(oriented, sector)
        ).toarray()
        real_spectrum = np.sort(np.linalg.eigvalsh(real_matrix))
        doubled = np.sort(np.concatenate([complex_spectrum, complex_spectrum]))
        assert np.allclose(real_spectrum, doubled, atol=1.0e-12)


def test_frozen_classical_realization_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (
        DATA_DIR / "classical_realization_receipt.json"
    ).read_bytes()
    assert manifest["classical_realization_receipt_sha256"] == (
        "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    )
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-classical-realization.v1"
    assert receipt["issue"] == 311
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == (
        "CLASSICAL_REALIZATION_MATCHES_DECLARED_FINITE_SPECTRAL_INTERFACE"
    )
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    readings = receipt["classical_completion"]["sector_readings"]
    gates = receipt["numerical_gates"]
    assert len(readings) == 6
    assert all(row["gap_match"] and row["kernel_match"] for row in readings)
    assert all(
        row["gap_match"]
        == (row["gap_abs_residual"] < gates["gap_match_abs_tolerance"])
        for row in readings
    )
    assert all(
        row["symmetric"]
        == (
            row["symmetry_max_residual"]
            < gates["symmetry_max_residual_tolerance"]
        )
        for row in readings
    )
    assert all(
        row["kernel_eigenvalue_abs_floor"]
        == gates["kernel_eigenvalue_abs_floor"]
        for row in readings
    )
    assert all(
        row["kernel_match"]
        == (
            row["classical_kernel_count"]
            == row["expected_doubled_kernel"]
            and (
                row["expected_doubled_kernel"] == 0
                or row["kernel_max_abs_eigenvalue"]
                < gates["kernel_eigenvalue_abs_floor"]
            )
            and row["first_nonkernel_abs_eigenvalue"]
            > gates["kernel_eigenvalue_abs_floor"]
        )
        for row in readings
    )
    assert readings[3]["classical_kernel_count"] == 2
    census = receipt["serialized_interface_census"]
    assert census["no_declared_discriminator_key_match"] is True
    assert census["array_artifact"] is not None
    upstream_pins = receipt["upstream_pins"]
    assert upstream_pins == census["artifact_sha256"]
    assert set(upstream_pins) == {
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
        "stage4_receipt.json",
        "source_gap_receipt.json",
        "defect_sector_receipt.json",
        "matter_attachment_receipt.json",
        "stage1_arrays.npz.gz",
    }
    for name, digest in upstream_pins.items():
        assert digest == (
            "sha256:" + hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        )
    assert set(census["total_leaf_census"]).issubset(
        {
            "null",
            "boolean_leaf",
            "integer_leaf",
            "float_leaf",
            "sha256_string",
            "string_leaf",
        }
    )
    ladder = receipt["classical_completion"]["ladder_point_readings"]
    assert len(ladder) == 6
    assert all(row["gap_match"] and row["kernel_match"] for row in ladder)
    assert all(
        row["gap_match"]
        == (row["gap_abs_residual"] < gates["gap_match_abs_tolerance"])
        for row in ladder
    )
    assert receipt["classical_completion"]["scalar_gap_match"] == (
        receipt["classical_completion"]["scalar_gap_abs_residual"]
        < gates["gap_match_abs_tolerance"]
    )
    identity = receipt["spectral_interface_identity"]
    assert identity["producer_schema"] == (
        "oph.local-domain-defect-sector-spectra.v1"
    )
    assert identity[
        "rer_exact_flux_12_42_vertex_identity_bridge"
    ] is False
    assert identity["separate_from_rer_exact_flux_certificate"] is True
    assert identity["main_domain"]["visible_node_count"] == 8662
    assert identity["ladder_domain"]["source_carrier_count"] == 2048
    assert identity["ladder_domain"]["visible_node_count"] == 1052
    result = receipt["finite_interface_result"]
    assert "refinement_scope" in result
    assert "does not establish a cofinal refinement map" in result[
        "refinement_scope"
    ]
    sector = json.loads(
        (DATA_DIR / "defect_sector_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == sector[
        "source_projection_sha256"
    ]
