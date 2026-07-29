"""Frozen tests for the issue-633 and issue-311 closure certificates."""

import hashlib
import json
from pathlib import Path

import numpy as np

from oph_fpe.local_domain.classical_realization import rotor_stiffness
from oph_fpe.local_domain.clock_unit_verdict import walk_interface
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
        "integer_count": 1,
        "dimensionless_real": 1,
        "hash": 1,
        "text": 1,
        "boolean_verdict": 1,
        "null": 1,
    }
    assert hits == []
    _, injected = walk_interface({"si_frequency_row": 1}, "probe")
    assert injected


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
    assert receipt["verdict"] == "PHYSICAL_UNITS_NOT_EVALUABLE"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    classification = receipt["interface_classification"]
    assert classification["interface_dimensionless"] is True
    assert classification["unit_vocabulary_hits"] == []
    assert receipt["two_completion_witness"]["witness_holds"] is True
    assert receipt["source_input_closure"]["closed"] is True
    assert receipt["positive_clause_retained"][
        "source_hamiltonian_with_positive_gap"
    ]


def test_rotor_stiffness_realifies_spectrum():
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
        real_matrix = rotor_stiffness(
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
        "CLASSICAL_QUANTUM_INDISTINGUISHABLE_ON_DECLARED_DOMAIN"
    )
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    readings = receipt["classical_completion"]["sector_readings"]
    assert len(readings) == 6
    assert all(row["gap_match"] and row["kernel_match"] for row in readings)
    assert readings[3]["classical_kernel_count"] == 2
    partition = receipt["interface_partition"]
    assert partition["no_quantum_discriminator"] is True
    sector = json.loads(
        (DATA_DIR / "defect_sector_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["capture_sha256"] == sector["capture_sha256"]
