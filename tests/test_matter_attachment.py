"""Frozen tests for the issue-569 finite matter attachment."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from oph_fpe.local_domain.matter_attachment import (
    GENERATION_TABLE,
    chirality_certificate,
    gap_inheritance_certificate,
    generation_certificate,
    z6_kernel_certificate,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"


def test_generation_table_exact_arithmetic():
    generation = generation_certificate()
    assert generation["weyl_state_count"] == 15
    assert generation["weak_doublets_per_family"] == 4
    assert generation["witten_parity_even"]
    assert generation["anomalies_vanish"]
    assert all(
        value == "0" for value in generation["anomaly_forms"].values()
    )


def test_z6_kernel_fixes_all_states_and_mutation_fails():
    z6 = z6_kernel_certificate()
    assert z6["all_states_fixed"]
    assert set(z6["row_phases"].values()) <= {"0", "1"}

    mutated = tuple(
        dict(row, hypercharge=Fraction(row["hypercharge"]) + Fraction(1, 6))
        if row["label"] == "e_c"
        else row
        for row in GENERATION_TABLE
    )
    assert not z6_kernel_certificate(mutated)["all_states_fixed"]
    assert not generation_certificate(mutated)["anomalies_vanish"]


def test_chirality_nondegenerate_and_vectorlike_fails():
    chirality = chirality_certificate()
    assert chirality["chirality_nondegenerate"]
    assert chirality["conjugate_overlap"] == []

    vectorlike = GENERATION_TABLE + tuple(
        {
            "label": row["label"] + "_bar",
            "color": -int(row["color"]) if abs(int(row["color"])) == 3
            else int(row["color"]),
            "weak": row["weak"],
            "hypercharge": -Fraction(row["hypercharge"]),
        }
        for row in GENERATION_TABLE
    )
    assert not chirality_certificate(vectorlike)["chirality_nondegenerate"]


def test_gap_inheritance_requires_same_source_and_domain():
    gap = {
        "schema": "oph.source-clock-gap.v1",
        "issue": 633,
        "physical_promotion_allowed": False,
        "verdict": "ATTAINED",
        "source_projection_sha256": "sha256:source",
        "domain_freeze_sha256": "sha256:domain",
        "exact_gap": {"positive": True},
        "measured_gap": {"smallest_eigenvalue": 0.1},
    }
    valid = gap_inheritance_certificate(
        gap,
        source_projection_sha256="sha256:source",
        domain_freeze_sha256="sha256:domain",
    )
    assert valid["inherited"]
    assert valid["status"] == (
        "conditional_algebraic_inheritance_under_declared_tensor_extension"
    )
    assert valid["matter_action_source_selected"] is False

    wrong_source = gap_inheritance_certificate(
        gap,
        source_projection_sha256="sha256:other",
        domain_freeze_sha256="sha256:domain",
    )
    assert not wrong_source["inherited"]
    wrong_domain = gap_inheritance_certificate(
        gap,
        source_projection_sha256="sha256:source",
        domain_freeze_sha256="sha256:other",
    )
    assert not wrong_domain["inherited"]


def test_frozen_matter_attachment_receipt_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "matter_attachment_receipt.json").read_bytes()
    assert manifest["matter_attachment_receipt_sha256"] == (
        "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    )
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-matter-attachment.v1"
    assert receipt["issue"] == 569
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    assert receipt["attachment"]["complex_rank"] == 45
    assert receipt["attachment"]["band_rank_measured"] == 3
    assert receipt["declared_matter_packet"]["status"] == (
        "declared_imported_matter_packet"
    )
    assert receipt["declared_matter_packet"]["source_selected"] is False
    assert receipt["upstream_pins"]["stage2_receipt_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            (DATA_DIR / "stage2_receipt.json").read_bytes()
        ).hexdigest()
    )
    assert receipt["upstream_pins"]["source_gap_receipt_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            (DATA_DIR / "source_gap_receipt.json").read_bytes()
        ).hexdigest()
    )
    assert receipt["matter_operator_certificate"]["probe_count"] == 1
    assert receipt["matter_operator_certificate"]["status"] == (
        "declared_tensor_extension"
    )
    assert receipt["matter_operator_certificate"]["source_selected"] is False
    assert receipt["gap_inheritance_certificate"]["status"] == (
        "conditional_algebraic_inheritance_under_declared_tensor_extension"
    )
    assert receipt["gap_inheritance_certificate"][
        "matter_action_source_selected"
    ] is False
    assert receipt["clause_verdicts"][
        "declared_matter_operator_probe_identities_exact"
    ]
    assert receipt["clause_verdicts"]["conditional_gap_inheritance_exact"]
    assert (
        "one deterministic rank-45"
        in receipt["matter_operator_certificate"]["identity_scope"]
    )
    assert receipt["bounded_declared_key_scan"]["hits"] == []
    gate_rows = receipt["spin_layer"]["issue_314_artifact"]["gate_rows"]
    assert gate_rows["laboratory_exchange_measurement"] is False
    assert gate_rows["continuum_spin_statistics_theorem"] is False
    spin_layer = receipt["spin_layer"]
    assert spin_layer["packet_status"] == (
        "separate_pinned_issue_314_packet"
    )
    assert spin_layer["spin_to_local_domain_bridge_certified"] is False
    assert spin_layer["same_source_domain_certified"] is False
    assert spin_layer["spin_support_identity"]["cell_counts"] == {
        "vertices": 12,
        "edges": 30,
        "faces": 20,
    }
    assert spin_layer["local_domain_identity"]["visible_node_count"] == 8662
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == stage1[
        "source_projection_sha256"
    ]
