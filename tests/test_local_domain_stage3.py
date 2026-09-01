"""Frozen tests for the stage-3 typed domain."""

import hashlib
import json
from pathlib import Path

from oph_fpe.local_domain.stage2_spin_layer import seam_complex
from oph_fpe.local_domain.stage3_typed_domain import (
    adjoint_certificate,
    boundary_typing,
    gauge_covariance_certificate,
    kinetic_kernel_certificate,
    locality_certificate,
    probe_section,
    refinement_naturality_certificate,
    reject_nonlocal_operator,
    restriction_gluing_certificate,
    seam_adjoint,
    seam_derivative,
)

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


def _triangle_complex():
    return seam_complex([_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)])


def _square_complex():
    return seam_complex(
        [
            _row(0, 0, 1),
            _row(1, 1, 2),
            _row(2, 2, 3),
            _row(3, 0, 3),
        ]
    )


def test_adjoint_and_kinetic_identities_exact():
    certificate = adjoint_certificate(_triangle_complex())
    assert certificate["adjoint_identity_exact"]
    assert certificate["kinetic_identity_exact"]
    assert certificate["kinetic_nonnegative"]


def test_derivative_twist_and_adjoint_shapes():
    complex_data = _triangle_complex()
    section = probe_section(complex_data["nodes"], 1, salt=5)
    derivative = seam_derivative(complex_data, section)
    assert set(derivative) == set(complex_data["edges"])
    adjoint = seam_adjoint(complex_data, derivative)
    assert set(adjoint) == set(complex_data["nodes"])


def test_kinetic_kernel_matches_sign_structure():
    frustrated = kinetic_kernel_certificate(_triangle_complex())
    assert frustrated["sign_consistent_component_count"] == 0
    assert frustrated["twisted_kernel_dimension"] == 0
    assert frustrated["witnesses_annihilated"]
    assert frustrated["frustrated_component_witnesses_verified"]
    assert frustrated["rank_theorem"]["applied"]
    negative_cycle = frustrated["component_certificates"][0][
        "negative_cycle_witness"
    ]
    assert negative_cycle["negative"]
    assert negative_cycle["sign_product"] == -1
    assert negative_cycle["nodes"][0] == negative_cycle["nodes"][-1]
    assert len(negative_cycle["edge_signs"]) == len(
        negative_cycle["nodes"]
    ) - 1

    consistent = kinetic_kernel_certificate(_square_complex())
    assert consistent["sign_consistent_component_count"] == 1
    assert consistent["twisted_kernel_dimension"] == 1
    assert consistent["witnesses_annihilated"]
    assert consistent["rank_theorem"]["applied"]
    assert consistent["component_certificates"][0][
        "negative_cycle_witness"
    ] is None


def test_gauge_covariance_exact():
    certificate = gauge_covariance_certificate(_square_complex())
    assert certificate["derivative_covariant"]
    assert certificate["kinetic_norm_invariant"]


def test_locality_gate_accepts_and_rejects():
    complex_data = _square_complex()
    assert locality_certificate(complex_data)["local"]
    gate = reject_nonlocal_operator(complex_data, [(0, 2)])
    assert not gate["accepted"]
    assert gate["nonlocal_couplings"] == [(0, 2)]
    assert reject_nonlocal_operator(complex_data, [(0, 1)])["accepted"]


def test_restriction_gluing_and_naturality():
    complex_data = _square_complex()
    gluing = restriction_gluing_certificate(
        complex_data, [[0, 1, 2], [2, 3, 0]]
    )
    assert gluing["cover_complete"]
    assert gluing["overlap_sections_agree"]
    assert gluing["gluing_reconstructs_section"]
    naturality = refinement_naturality_certificate(complex_data)
    assert naturality["restriction_commutes_with_derivative"]


def test_boundary_typing_free_ports():
    complex_data = dict(_square_complex())
    complex_data["port_bound_of"] = {0: 12, 1: 12, 2: 3, 3: 1}
    event_table = {"count": 3, "depth": {0: 0, 1: 1, 2: 2}}
    boundary = boundary_typing(complex_data, event_table)
    assert boundary["spatial_interior_count"] == 2
    assert boundary["spatial_boundary_count"] == 2
    assert boundary["initial_slice_events"] == 1
    assert boundary["final_slice_events"] == 1
    assert boundary["dirichlet_restriction_exact"]


def test_frozen_stage3_receipt_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "stage3_receipt.json").read_bytes()
    assert manifest["stage3_receipt_sha256"] == "sha256:" + hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-stage3.v1"
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == stage1[
        "source_projection_sha256"
    ]
    kernel = receipt["kinetic_kernel_certificate"]
    assert kernel["matches_stage2"] is True
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
