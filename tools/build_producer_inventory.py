#!/usr/bin/env python3
"""Render the Phase-0 verifier/producer inventory.

The scope is the verifier-without-producer surface identified by the
2026-07-13 simulator audit, plus the K1 and H2 repair rows that share that
Phase-0 work item.  This is deliberately narrower than an inventory of every
diagnostic boolean emitted anywhere in OPH-FPE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "docs" / "PRODUCER_INVENTORY.md"

PRODUCER_STATUSES = {
    "PRODUCED",
    "PRODUCED_NONPROMOTING",
    "DECLARED_INPUT_NONPROMOTING",
    "NO_PRODUCER_NOT_TESTABLE",
    "INVENTORY_ONLY_NOT_TESTABLE",
    "RETIRED_NONPROMOTING",
}
PUBLIC_STATUSES = {"TESTABLE", "NOT_ATTAINED", "NOT_TESTABLE", "NONPROMOTING"}


@dataclass(frozen=True)
class InventoryRow:
    row_id: str
    receipt_or_gate: str
    verifier: str
    producer: str | None
    producer_status: str
    public_status: str
    evidence: str
    boundary: str


ROWS = (
    InventoryRow(
        row_id="WZH_V1_DECLARATIONS",
        receipt_or_gate="WZH0--WZH4 legacy W/Z/H gate inputs",
        verifier="oph_fpe/bosons/pipeline.py",
        producer=None,
        producer_status="DECLARED_INPUT_NONPROMOTING",
        public_status="NONPROMOTING",
        evidence="tests/test_wzh_source_closure_backend.py",
        boundary=(
            "Caller booleans remain visible only inside the v1 diagnostic. "
            "promotion_allowed is hard-false and the report names the missing "
            "runtime resolver and independent scientific checker."
        ),
    ),
    InventoryRow(
        row_id="E1_NULL_STRESS",
        receipt_or_gate="EINSTEIN_NULL_STRESS_CHARGE_RECEIPT (E1)",
        verifier="oph_fpe/bulk/einstein_bridge.py",
        producer=None,
        producer_status="NO_PRODUCER_NOT_TESTABLE",
        public_status="NOT_TESTABLE",
        evidence="tests/test_paper_side_realized_branch.py; tests/test_theorem_contract.py",
        boundary=(
            "The typed source tower exposes null-charge and stress readouts, "
            "but no primitive producer emits the theorem-tagged E1 sidecar."
        ),
    ),
    InventoryRow(
        row_id="E3_SMALL_BALL_AREA",
        receipt_or_gate="EINSTEIN_SMALL_BALL_AREA_BRIDGE_RECEIPT (E3)",
        verifier="oph_fpe/bulk/einstein_bridge.py",
        producer=None,
        producer_status="NO_PRODUCER_NOT_TESTABLE",
        public_status="NOT_TESTABLE",
        evidence="tests/test_paper_side_realized_branch.py; tests/test_theorem_contract.py",
        boundary=(
            "No primitive producer emits the theorem-tagged small-ball-area "
            "sidecar; manifest or legacy booleans cannot substitute for it."
        ),
    ),
    InventoryRow(
        row_id="E5_LAMBDA_CONSTANCY",
        receipt_or_gate="EINSTEIN_LAMBDA_CONSTANCY_CONSERVATION_RECEIPT (E5)",
        verifier="oph_fpe/bulk/einstein_bridge.py",
        producer=None,
        producer_status="NO_PRODUCER_NOT_TESTABLE",
        public_status="NOT_TESTABLE",
        evidence="tests/test_paper_side_realized_branch.py; tests/test_theorem_contract.py",
        boundary=(
            "No primitive producer emits the theorem-tagged lambda-closure "
            "sidecar; a declared capacity or configured Lambda is not evidence."
        ),
    ),
    InventoryRow(
        row_id="COMMON_SOURCE_TOWER",
        receipt_or_gate="COMMON_DOMAIN_SOURCE_TOWER_RECEIPT",
        verifier="oph_fpe/common_source_tower.py",
        producer="oph_fpe/bulk/einstein_tower_producer.py",
        producer_status="PRODUCED_NONPROMOTING",
        public_status="NOT_ATTAINED",
        evidence="tests/test_einstein_tower_producer.py",
        boundary=(
            "All thirteen typed role readouts are source-produced and replayed. "
            "The generator firewall and federation binding remain pinned false, "
            "so this does not manufacture E1, E3, E5, or gravity receipts."
        ),
    ),
    InventoryRow(
        row_id="EINSTEIN_MEASURED_INSTRUMENTS",
        receipt_or_gate="issues 573--577 measured branch instruments",
        verifier="oph_fpe/bulk/*_producer.py",
        producer=(
            "oph_fpe/bulk/modular_normalization_producer.py; "
            "oph_fpe/bulk/gns_tower_producer.py; "
            "oph_fpe/bulk/event_manifold_producer.py; "
            "oph_fpe/bulk/stress_coupling_producer.py; "
            "oph_fpe/bulk/einstein_branch_countermodels.py"
        ),
        producer_status="PRODUCED_NONPROMOTING",
        public_status="NOT_ATTAINED",
        evidence=(
            "tests/test_modular_normalization_producer.py; "
            "tests/test_gns_tower_producer.py; "
            "tests/test_event_manifold_producer.py; "
            "tests/test_stress_coupling_producer.py"
        ),
        boundary=(
            "Deterministic producers and negative controls exist. Their current "
            "scientific verdict is NOT_ATTAINED and physical promotion is false."
        ),
    ),
    InventoryRow(
        row_id="H3_WORLDLINE_STITCH",
        receipt_or_gate="H3 cross-shard worldline stitch primitives",
        verifier="oph_fpe/bulk/h3_worldline_stitch.py",
        producer="oph_fpe/bulk/h3_worldline_stitch_producer.py",
        producer_status="INVENTORY_ONLY_NOT_TESTABLE",
        public_status="NOT_TESTABLE",
        evidence="tests/test_h3_worldline_stitch_producer.py",
        boundary=(
            "The producer inventories every missing measured input and never "
            "synthesizes primitives. The cross-shard measurement lane is absent."
        ),
    ),
    InventoryRow(
        row_id="H2_MEASURED_OVERLAP",
        receipt_or_gate="measured_overlap_geometry_receipt",
        verifier="oph_fpe/bulk/neutral_bulk.py",
        producer="oph_fpe/observers/subjective.py",
        producer_status="PRODUCED_NONPROMOTING",
        public_status="NOT_TESTABLE",
        evidence="tests/test_subjective_observers.py; tests/test_neutral_bulk.py",
        boundary=(
            "Literal cross-observer intersections are produced, but the current "
            "supports descend from the S2 screen adjacency. Provenance therefore "
            "marks them strict_neutral_eligible=false."
        ),
    ),
    InventoryRow(
        row_id="H2_STRICT_NEUTRAL",
        receipt_or_gate="STRICT_NEUTRAL_BULK_RECEIPT (H2)",
        verifier="oph_fpe/bulk/neutral_bulk.py",
        producer=None,
        producer_status="NO_PRODUCER_NOT_TESTABLE",
        public_status="NOT_TESTABLE",
        evidence="tests/test_neutral_bulk.py",
        boundary=(
            "The claim-bearing extractor is non-hash, but a chart-blind support "
            "carrier and a paired perturb-resettle producer are still absent. "
            "The H2 receipt remains false."
        ),
    ),
    InventoryRow(
        row_id="H2_HASH_TOKEN_LEGACY",
        receipt_or_gate="legacy H2 boundary/transition hash-token histograms",
        verifier="oph_fpe/bulk/neutral_bulk.py",
        producer="oph_fpe/scale/bw_array.py",
        producer_status="RETIRED_NONPROMOTING",
        public_status="NONPROMOTING",
        evidence="tests/test_neutral_bulk.py; tests/test_bw_array.py",
        boundary=(
            "Retained only to replay old artifacts. Default claim-bearing H2 "
            "weights are zero and producer provenance forbids hash buckets from "
            "satisfying a required neutral channel."
        ),
    ),
    InventoryRow(
        row_id="POSITIVE_GEOMETRY_PLUGIN",
        receipt_or_gate="positive-geometry accelerator readiness",
        verifier="oph_fpe/dynamics/positive_geometry.py",
        producer=None,
        producer_status="DECLARED_INPUT_NONPROMOTING",
        public_status="NONPROMOTING",
        evidence="tests/test_positive_geometry_kernel.py",
        boundary=(
            "The external manifest is a declared plugin input. Even native "
            "geometry certification is an accelerator result, not an OPH "
            "scattering, neutral-bulk, or physical-prediction receipt."
        ),
    ),
    InventoryRow(
        row_id="K1_GAUGE_COVARIANT_MISMATCH",
        receipt_or_gate="K1 production mismatch and perturb/resettle replay",
        verifier="oph_fpe/gauge/covariant_overlap.py",
        producer="oph_fpe/scale/bw_array.py",
        producer_status="PRODUCED",
        public_status="TESTABLE",
        evidence=(
            "tests/test_covariant_overlap.py; "
            "tests/test_modular_probe_sector_replay.py; tests/test_bw_array.py"
        ),
        boundary=(
            "Production mismatch uses p_i != g_ij p_j and is invariant under "
            "independent node-frame changes. Enabled sector-link repair is "
            "replayed in both perturb/remeasure implementations."
        ),
    ),
)


def validate_inventory() -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in ROWS:
        if row.row_id in seen:
            errors.append(f"duplicate row id: {row.row_id}")
        seen.add(row.row_id)
        if row.producer_status not in PRODUCER_STATUSES:
            errors.append(
                f"{row.row_id}: unknown producer status {row.producer_status}"
            )
        if row.public_status not in PUBLIC_STATUSES:
            errors.append(
                f"{row.row_id}: unknown public status {row.public_status}"
            )
        for path in (item.strip() for item in row.verifier.split(";") if item.strip()):
            if "*" in path:
                if not any(REPOSITORY_ROOT.glob(path)):
                    errors.append(f"{row.row_id}: verifier glob matched nothing: {path}")
            elif not (REPOSITORY_ROOT / path).is_file():
                errors.append(f"{row.row_id}: missing verifier path {path}")
        if row.producer is not None:
            for path in (
                item.strip() for item in row.producer.split(";") if item.strip()
            ):
                if not (REPOSITORY_ROOT / path).is_file():
                    errors.append(f"{row.row_id}: missing producer path {path}")
        for path in (item.strip() for item in row.evidence.split(";") if item.strip()):
            if not (REPOSITORY_ROOT / path).is_file():
                errors.append(f"{row.row_id}: missing evidence path {path}")
        if row.producer is None and row.producer_status not in {
            "DECLARED_INPUT_NONPROMOTING",
            "NO_PRODUCER_NOT_TESTABLE",
        }:
            errors.append(
                f"{row.row_id}: missing producer lacks an explicit safe classification"
            )
        if (
            row.producer_status
            in {"NO_PRODUCER_NOT_TESTABLE", "INVENTORY_ONLY_NOT_TESTABLE"}
            and row.public_status != "NOT_TESTABLE"
        ):
            errors.append(
                f"{row.row_id}: producerless/inventory-only gate must be NOT_TESTABLE"
            )
    return errors


def render_inventory() -> str:
    lines = [
        "# Phase-0 verifier/producer inventory",
        "",
        "Generated by `python tools/build_producer_inventory.py`. Do not edit this file by hand.",
        "",
        "Scope: the verifier-without-producer findings in the 2026-07-13 simulator audit, "
        "including the K1 mismatch and H2 hash-token retirement rows. This is not a list of "
        "every diagnostic boolean emitted by OPH-FPE.",
        "",
        "A missing physical producer is reported as `NOT_TESTABLE`. Declared inputs and "
        "legacy artifacts are explicitly `NONPROMOTING`; neither may be interpreted as a "
        "pending positive receipt.",
        "",
        "| ID | Receipt or gate | Verifier | Producer | Producer status | Public status | Evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        producer = row.producer or "none"
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{row.row_id}`",
                    row.receipt_or_gate,
                    f"`{row.verifier}`",
                    f"`{producer}`",
                    f"`{row.producer_status}`",
                    f"`{row.public_status}`",
                    f"`{row.evidence}`",
                )
            )
            + " |"
        )
    lines.extend(["", "## Claim boundaries", ""])
    for row in ROWS:
        lines.append(f"- `{row.row_id}`: {row.boundary}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated inventory is missing or stale",
    )
    args = parser.parse_args(argv)

    errors = validate_inventory()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    rendered = render_inventory()
    if args.check:
        if not OUTPUT_PATH.is_file():
            print(f"missing generated inventory: {OUTPUT_PATH}", file=sys.stderr)
            return 1
        current = OUTPUT_PATH.read_text(encoding="utf-8")
        if current != rendered:
            print(
                "producer inventory is stale; run "
                "python tools/build_producer_inventory.py",
                file=sys.stderr,
            )
            return 1
        print(f"producer inventory is current: {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)}")
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
