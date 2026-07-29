"""Frozen tests for the issue-633 source-gap instrument."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from oph_fpe.local_domain.receipt_io import (
    load_manifest_pinned_receipt,
    manifest_pinned_artifact_sha256,
    stage2_matches_source_domain,
)
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


def test_stage2_loader_and_binding_fail_closed(tmp_path):
    shutil.copyfile(DATA_DIR / "manifest.json", tmp_path / "manifest.json")
    shutil.copyfile(
        DATA_DIR / "stage2_receipt.json",
        tmp_path / "stage2_receipt.json",
    )
    stage2 = load_manifest_pinned_receipt(
        tmp_path, "stage2_receipt.json", "stage2_receipt_sha256"
    )
    assert stage2 is not None
    assert manifest_pinned_artifact_sha256(
        tmp_path,
        "stage2_receipt.json",
        "stage2_receipt_sha256",
    ) == "sha256:" + hashlib.sha256(
        (tmp_path / "stage2_receipt.json").read_bytes()
    ).hexdigest()
    source = stage2["source_projection_sha256"]
    domain = stage2["seam_layer"]["domain_complex"][
        "complex_freeze_sha256"
    ]
    assert stage2_matches_source_domain(stage2, source, domain)
    assert not stage2_matches_source_domain(
        {**stage2, "verdict": "NOT_ATTAINED"}, source, domain
    )
    assert not stage2_matches_source_domain(
        {**stage2, "stage1_binding": []}, "sha256:wrong", domain
    )

    with (tmp_path / "stage2_receipt.json").open("ab") as stream:
        stream.write(b" ")
    assert (
        manifest_pinned_artifact_sha256(
            tmp_path,
            "stage2_receipt.json",
            "stage2_receipt_sha256",
        )
        is None
    )
    assert (
        load_manifest_pinned_receipt(
            tmp_path, "stage2_receipt.json", "stage2_receipt_sha256"
        )
        is None
    )

    (tmp_path / "manifest.json").write_text("[]\n")
    assert (
        load_manifest_pinned_receipt(
            tmp_path, "stage2_receipt.json", "stage2_receipt_sha256"
        )
        is None
    )

    nonobject = b"[]\n"
    (tmp_path / "stage2_receipt.json").write_bytes(nonobject)
    manifest = {
        "schema": "oph.local-domain-stage1.manifest.v1",
        "stage2_receipt_sha256": (
            "sha256:" + hashlib.sha256(nonobject).hexdigest()
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    assert (
        load_manifest_pinned_receipt(
            tmp_path, "stage2_receipt.json", "stage2_receipt_sha256"
        )
        is None
    )


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
    kernel = receipt["kernel_certificate"]
    assert kernel["twisted_kernel_dimension"] == 0
    assert kernel["rank_theorem"]["applied"] is True
    assert kernel["frustrated_component_witnesses_verified"] is True
    assert all(
        row["consistent"]
        or (
            row["negative_cycle_witness"]["negative"]
            and row["negative_cycle_witness"]["sign_product"] == -1
        )
        for row in kernel["component_certificates"]
    )
    assert receipt["measured_gap"]["smallest_eigenvalue"] > 0.0
    measured = receipt["measured_gap"]
    residual_gate = receipt["numerical_gates"][
        "measured_relative_residual_tolerance"
    ]
    assert measured["residual_within_gate"] == (
        measured["relative_residual"] < residual_gate
    )
    assert receipt["si_binding"]["status"] == "NOT_PART_OF_THIS_RECEIPT"
    assert receipt["stage2_binding"][
        "receipt_present_and_manifest_pinned"
    ] is True
    assert receipt["stage2_binding"]["receipt_attained"] is True
    assert receipt["stage2_binding"]["same_source_and_domain"] is True
    assert receipt["stage2_binding"]["receipt_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            (DATA_DIR / "stage2_receipt.json").read_bytes()
        ).hexdigest()
    )
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == stage1[
        "source_projection_sha256"
    ]
