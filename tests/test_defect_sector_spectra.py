"""Frozen tests for the issue-311 sector-spectra instrument."""

import hashlib
import json
from pathlib import Path

import numpy as np

from oph_fpe.local_domain.defect_sector_spectra import (
    flat_section_verdict,
    oriented_seams,
    sector_laplacian,
    sector_phase_exponent,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"


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


def test_sector_exponents_are_z6_orbit_of_convention():
    exponents = [sector_phase_exponent(k) for k in range(6)]
    assert exponents == [6, 8, 10, 0, 2, 4]
    assert sector_phase_exponent(7) == sector_phase_exponent(1)


def test_oriented_seams_keep_declared_direction_and_dedupe():
    rows = [_row(0, 5, 2), _row(1, 2, 5), _row(2, 1, 7)]
    oriented = oriented_seams(rows)
    assert oriented == [(5, 2), (1, 7)]


def test_flat_sectors_depend_on_the_orientation_field():
    chain_triangle = oriented_seams(
        [_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)]
    )
    for sector in range(6):
        expected_flat = sector_phase_exponent(sector) == 0
        verdict = flat_section_verdict(chain_triangle, sector)
        assert verdict["kernel_dimension"] == (1 if expected_flat else 0)

    cyclic_triangle = oriented_seams(
        [_row(0, 0, 1), _row(1, 1, 2), _row(2, 2, 0)]
    )
    for sector in range(6):
        expected_flat = (3 * sector_phase_exponent(sector)) % 12 == 0
        verdict = flat_section_verdict(cyclic_triangle, sector)
        assert verdict["kernel_dimension"] == (1 if expected_flat else 0)


def test_tree_always_flat_in_every_sector():
    path = oriented_seams([_row(0, 0, 1), _row(1, 2, 1)])
    for sector in range(6):
        assert flat_section_verdict(path, sector)["kernel_dimension"] == 1


def test_sector_laplacian_hermitian_and_kernel_agreement():
    cyclic_triangle = oriented_seams(
        [_row(0, 0, 1), _row(1, 1, 2), _row(2, 2, 0)]
    )
    for sector in range(6):
        matrix = sector_laplacian(cyclic_triangle, sector).toarray()
        assert np.allclose(matrix, matrix.conj().T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        verdict = flat_section_verdict(cyclic_triangle, sector)
        near_zero = int((np.abs(eigenvalues) < 1.0e-9).sum())
        assert near_zero == verdict["kernel_dimension"]


def test_frozen_sector_receipt_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "defect_sector_receipt.json").read_bytes()
    assert manifest["defect_sector_receipt_sha256"] == (
        "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    )
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-defect-sector-spectra.v1"
    assert receipt["issue"] == 311
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    assert receipt["stage2_binding"]["receipt_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            (DATA_DIR / "stage2_receipt.json").read_bytes()
        ).hexdigest()
    )
    rows = receipt["sector_table"]
    gates = receipt["numerical_gates"]
    gaps = [row["gap_above_kernel"] for row in rows]
    assert all(
        row["residual_within_gate"]
        == (
            row["relative_residual"]
            < gates["measured_relative_residual_tolerance"]
        )
        for row in rows
    )
    assert all(
        row["kernel_dimension"] == 0
        or row["kernel_eigenvalue_leak"]
        < gates["kernel_eigenvalue_abs_floor"]
        for row in rows
    )
    assert abs(gaps[1] - gaps[5]) < 1.0e-9
    assert abs(gaps[2] - gaps[4]) < 1.0e-9
    assert receipt["gap_separations"]["distinct_gap_count"] >= 3
    assert receipt["gap_separations"]["distinct_gap_count"] == len(
        {
            round(value, gates["spectral_distinct_round_decimals"])
            for value in gaps
        }
    )
    assert receipt["gap_separations"][
        "conjugate_sector_pairs_degenerate"
    ] == all(
        abs(gaps[k] - gaps[(6 - k) % 6])
        < gates["conjugate_degeneracy_abs_tolerance"]
        for k in range(6)
    )
    identity = receipt["spectral_interface_identity"]
    assert identity["schema"] == receipt["schema"]
    assert identity["issue"] == receipt["issue"]
    assert identity["local_domain_issue"] == 634
    assert identity["rer_exact_flux_12_42_vertex_identity_bridge"] is False
    assert identity["separate_from_rer_exact_flux_certificate"] is True
    assert identity["main_domain"]["source_carrier_count"] == 16384
    assert identity["main_domain"]["source_projection_sha256"] == receipt[
        "source_projection_sha256"
    ]
    assert identity["main_domain"]["domain_freeze_sha256"] == receipt[
        "domain_freeze_sha256"
    ]
    assert identity["main_domain"]["visible_node_count"] == 8662
    ladder = receipt["scale_ladder"]
    assert ladder["small_source_carrier_count"] == 2048
    assert ladder["small_visible_node_count"] == 1052
    assert ladder["small_visible_edge_count"] == 1663
    assert ladder["small_source_projection_sha256"].startswith("sha256:")
    assert ladder["small_domain_freeze_sha256"].startswith("sha256:")
    assert identity["ladder_domain"]["source_carrier_count"] == ladder[
        "small_source_carrier_count"
    ]
    assert identity["ladder_domain"]["source_projection_sha256"] == ladder[
        "small_source_projection_sha256"
    ]
    assert identity["ladder_domain"]["domain_freeze_sha256"] == ladder[
        "small_domain_freeze_sha256"
    ]
    assert identity["ladder_domain"]["visible_node_count"] == ladder[
        "small_visible_node_count"
    ]
    assert identity["ladder_domain"]["visible_edge_count"] == ladder[
        "small_visible_edge_count"
    ]
    gap_receipt = json.loads(
        (DATA_DIR / "source_gap_receipt.json").read_text(encoding="utf-8")
    )
    assert abs(
        gaps[0] - gap_receipt["measured_gap"]["smallest_eigenvalue"]
    ) < 1.0e-9
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == stage1[
        "source_projection_sha256"
    ]
