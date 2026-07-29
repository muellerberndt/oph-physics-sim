"""Frozen tests for the issue-569 finite matter attachment."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from oph_fpe.local_domain.matter_attachment import (
    GENERATION_TABLE,
    chirality_certificate,
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
    gate_rows = receipt["spin_layer"]["issue_314_artifact"]["gate_rows"]
    assert gate_rows["laboratory_exchange_measurement"] is False
    assert gate_rows["continuum_spin_statistics_theorem"] is False
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["capture_sha256"] == stage1["capture_sha256"]
