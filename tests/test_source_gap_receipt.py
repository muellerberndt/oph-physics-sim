"""Frozen tests for the issue-633 source-gap instrument."""

import hashlib
import json
from pathlib import Path

import numpy as np

from oph_fpe.local_domain.source_gap_receipt import (
    exact_gap_floor,
    signed_laplacian,
)
from oph_fpe.local_domain.stage2_spin_layer import seam_complex

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"


def _row(index, a, b, signs=(-1,)):
    return {
        "overlap_id": f"seam-{index:04d}",
        "left_carrier_id": f"carrier-{a:05d}",
        "right_carrier_id": f"carrier-{b:05d}",
        "left_ports": [0],
        "right_ports": [0],
        "orientation_signs": list(signs),
        "visible_to_observer_tokens": ["observer-0000"],
        "interface_algebra_sha256": "sha256:a",
    }


def test_signed_laplacian_quadratic_form_matches_derivative_norm():
    complex_data = seam_complex(
        [_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)]
    )
    matrix = signed_laplacian(complex_data).toarray()
    assert np.allclose(matrix, matrix.T)
    sign_of = complex_data["edge_sign_of"]
    rng = np.random.default_rng(633)
    for _ in range(4):
        section = rng.integers(-5, 6, size=complex_data["node_count"])
        quadratic = float(section @ matrix @ section)
        index = {v: i for i, v in enumerate(complex_data["nodes"])}
        derivative_norm = 0.0
        for (a, b), sign in sign_of.items():
            derivative_norm += float(
                (sign * section[index[b]] - section[index[a]]) ** 2
            )
        assert quadratic == derivative_norm


def test_frustrated_triangle_has_positive_gap_and_square_does_not():
    triangle = seam_complex([_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)])
    eigenvalues = np.linalg.eigvalsh(signed_laplacian(triangle).toarray())
    assert eigenvalues.min() > 1.0e-9

    square = seam_complex(
        [_row(0, 0, 1), _row(1, 1, 2), _row(2, 2, 3), _row(3, 0, 3)]
    )
    eigenvalues = np.linalg.eigvalsh(signed_laplacian(square).toarray())
    assert abs(eigenvalues.min()) < 1.0e-12


def test_exact_gap_floor_positive():
    triangle = seam_complex([_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)])
    floor = exact_gap_floor(triangle)
    assert floor["floor_positive"]
    assert floor["max_degree"] == 2
    assert floor["floor_log2"] < 0.0


def test_frozen_gap_receipt_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "source_gap_receipt.json").read_bytes()
    assert manifest["source_gap_receipt_sha256"] == (
        "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    )
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.source-clock-gap.v1"
    assert receipt["issue"] == 633
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["exact_gap"]["positive"] is True
    assert receipt["measured_gap"]["smallest_eigenvalue"] > 0.0
    assert receipt["measured_gap"]["residual_within_gate"] is True
    assert receipt["si_binding"]["status"] == (
        "OPEN_INTERFACE_PENDING_FIBER_SELECTION"
    )
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["capture_sha256"] == stage1["capture_sha256"]
