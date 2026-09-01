"""Frozen tests for the stage-2 seam sign layer."""

import hashlib
import json
from pathlib import Path

from oph_fpe.local_domain.stage2_spin_layer import (
    atlas_frame_layer,
    holonomy_census,
    lift_ambiguity,
    seam_complex,
    seam_convention_gate,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"


def _row(index, a, b, signs=(-1,), ports=((0,), (0,)), algebra="sha256:a"):
    return {
        "overlap_id": f"seam-{index:04d}",
        "left_carrier_id": f"carrier-{a:05d}",
        "right_carrier_id": f"carrier-{b:05d}",
        "left_ports": list(ports[0]),
        "right_ports": list(ports[1]),
        "orientation_signs": list(signs),
        "visible_to_observer_tokens": ["observer-0000"],
        "interface_algebra_sha256": algebra,
    }


def test_convention_gate_uniform_and_violations():
    rows = [_row(0, 0, 1), _row(1, 1, 2)]
    assert seam_convention_gate(rows)["uniform_reversing_single_port"]

    flipped = [_row(0, 0, 1, signs=(1,)), _row(1, 1, 2)]
    gate = seam_convention_gate(flipped)
    assert not gate["uniform_reversing_single_port"]
    assert any("nonreversing_sign" in v for v in gate["violations"])

    multi = [_row(0, 0, 1, ports=((0, 1), (0, 1)))]
    assert not seam_convention_gate(multi)["uniform_reversing_single_port"]

    two_algebras = [_row(0, 0, 1), _row(1, 1, 2, algebra="sha256:b")]
    gate = seam_convention_gate(two_algebras)
    assert "multiple_interface_algebras" in gate["violations"]


def test_seam_complex_triangle_and_sign_consistency():
    triangle = [_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)]
    complex_data = seam_complex(triangle)
    assert complex_data["node_count"] == 3
    assert complex_data["edge_count"] == 3
    assert complex_data["triangle_count"] == 1
    assert complex_data["component_count"] == 1
    assert complex_data["sign_consistent_component_count"] == 0

    square = [_row(0, 0, 1), _row(1, 1, 2), _row(2, 2, 3), _row(3, 0, 3)]
    complex_data = seam_complex(square)
    assert complex_data["triangle_count"] == 0
    assert complex_data["sign_consistent_component_count"] == 1

    conflicted = [_row(0, 0, 1), _row(1, 0, 1, signs=(1,))]
    complex_data = seam_complex(conflicted)
    assert complex_data["parallel_seam_sign_conflicts"] == 1


def test_holonomy_census_from_edge_signs():
    mixed = [_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2, signs=(1,))]
    complex_data = seam_complex(mixed)
    census = holonomy_census(complex_data)
    assert census["triangle_count"] == 1
    assert census["plus_holonomy_count"] == 1
    assert census["minus_holonomy_count"] == 0
    assert not census["all_triangles_frustrated"]


def test_lift_ambiguity_rank_cross_checked():
    square = [_row(0, 0, 1), _row(1, 1, 2), _row(2, 2, 3), _row(3, 0, 3)]
    ambiguity = lift_ambiguity(seam_complex(square))
    assert ambiguity["lift_ambiguity_rank"] == 1
    assert ambiguity["rank_cross_checked"]

    filled = [_row(0, 0, 1), _row(1, 1, 2), _row(2, 0, 2)]
    ambiguity = lift_ambiguity(seam_complex(filled))
    assert ambiguity["triangle_boundary_rank"] == 1
    assert ambiguity["lift_ambiguity_rank"] == 0
    assert ambiguity["rank_cross_checked"]


def test_atlas_frame_layer_on_frozen_stage1():
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    layer = atlas_frame_layer(stage1)
    assert layer["transition_determinants_positive"]
    assert layer["sign_lift_solvable"]
    assert layer["nerve_lift_ambiguity_rank"] == 0
    assert layer["nerve_rank_cross_checked"]


def test_frozen_stage2_receipt_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "stage2_receipt.json").read_bytes()
    assert manifest["stage2_receipt_sha256"] == "sha256:" + hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-stage2.v1"
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["stage1_binding"]["same_source"] is True
    assert receipt["seam_layer"]["holonomy_census"]["all_triangles_frustrated"]
    assert receipt["seam_layer"]["lift_ambiguity"]["rank_cross_checked"]
    assert receipt["controls_fail_closed"] is True
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == stage1[
        "source_projection_sha256"
    ]
