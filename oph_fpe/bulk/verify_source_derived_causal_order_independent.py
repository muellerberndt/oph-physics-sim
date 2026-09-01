"""Independent verifier for the source-derived causal-order receipt.

The producer is intentionally not imported. The verifier recomputes the
generated edge set from the receipt's embedded semantic events using only
read and write resource identifiers, recomputes every hash and clause
verdict, and fails closed on any disagreement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data" / "causal_order" / (
    "source_derived_causal_order_receipt.json"
)

EXPECTED_SCHEMA = "oph.source-derived-causal-order.v1"


class IndependentVerificationError(RuntimeError):
    """Raised when the receipt fails an independent recomputation."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fail(message: str) -> None:
    raise IndependentVerificationError(message)


def _regenerate_edges(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    writer_of: dict[str, str] = {}
    for event in events:
        for resource in event["write_resource_ids"]:
            if resource in writer_of:
                _fail(f"two writers for resource {resource!r}")
            writer_of[str(resource)] = str(event["event_key"])
    shared: dict[tuple[str, str], list[str]] = {}
    for event in events:
        child = str(event["event_key"])
        for resource in event["read_resource_ids"]:
            parent = writer_of.get(str(resource))
            if parent is None or parent == child:
                continue
            shared.setdefault((parent, child), []).append(str(resource))
    edges = [
        {
            "parent_event_id": parent,
            "child_event_id": child,
            "shared_resource_ids": sorted(resources),
        }
        for (parent, child), resources in shared.items()
    ]
    edges.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    return edges


def verify_receipt(path: Path | str = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="ascii"))

    if receipt.get("schema") != EXPECTED_SCHEMA:
        _fail(f"unexpected schema {receipt.get('schema')!r}")
    if receipt.get("issue") != 763:
        _fail("unexpected issue number")

    body = {k: v for k, v in receipt.items() if k != "report_sha256"}
    if _sha256(body) != receipt.get("report_sha256"):
        _fail("report_sha256 does not match the receipt body")

    events = receipt["semantic_events"]
    if len(events) != receipt["event_count"]:
        _fail("event_count does not match the embedded events")

    regenerated = _regenerate_edges(events)
    if regenerated != receipt["generated_edges"]:
        _fail("generated edges do not match an independent regeneration")
    if _sha256(regenerated) != receipt["generated_edges_sha256"]:
        _fail("generated_edges_sha256 mismatch")
    if _sha256(receipt["declared_edges"]) != receipt["declared_edges_sha256"]:
        _fail("declared_edges_sha256 mismatch")
    if len(regenerated) != receipt["generated_edge_count"]:
        _fail("generated_edge_count mismatch")
    if len(receipt["declared_edges"]) != receipt["declared_edge_count"]:
        _fail("declared_edge_count mismatch")

    clause = receipt["byte_identity_clause"]
    byte_identical = _canonical_bytes(regenerated) == _canonical_bytes(
        receipt["declared_edges"]
    )
    if clause["byte_identical"] is not byte_identical:
        _fail("byte_identical verdict mismatch")
    expected_verdict = "ATTAINED" if byte_identical else "NOT_ATTAINED"
    if clause["verdict"] != expected_verdict:
        _fail("byte-identity clause verdict mismatch")

    generated_pairs = {
        (row["parent_event_id"], row["child_event_id"]) for row in regenerated
    }
    declared_pairs = {
        (row["parent_event_id"], row["child_event_id"])
        for row in receipt["declared_edges"]
    }
    declared_only = sorted(declared_pairs - generated_pairs)
    generated_only = sorted(generated_pairs - declared_pairs)
    if [list(pair) for pair in declared_only] != [
        list(pair) for pair in clause["declared_only_pairs"]
    ]:
        _fail("declared_only_pairs mismatch")
    if [list(pair) for pair in generated_only] != [
        list(pair) for pair in clause["generated_only_pairs"]
    ]:
        _fail("generated_only_pairs mismatch")
    if clause["declared_only_pair_count"] != len(declared_only):
        _fail("declared_only_pair_count mismatch")
    if clause["generated_only_pair_count"] != len(generated_only):
        _fail("generated_only_pair_count mismatch")

    indegree = {str(e["event_key"]): 0 for e in events}
    children: dict[str, list[str]] = {str(e["event_key"]): [] for e in events}
    for row in regenerated:
        indegree[row["child_event_id"]] += 1
        children[row["parent_event_id"]].append(row["child_event_id"])
    frontier = sorted(k for k, v in indegree.items() if v == 0)
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    acyclic = visited == len(events)
    if receipt["generated_acyclic"] is not acyclic:
        _fail("generated_acyclic verdict mismatch")

    sequence_of = {
        str(e["event_key"]): int(e["source_sequence_index"]) for e in events
    }
    sequence_compatible = all(
        sequence_of[row["parent_event_id"]] < sequence_of[row["child_event_id"]]
        for row in regenerated
    )
    if receipt["sequence_compatible"] is not sequence_compatible:
        _fail("sequence_compatible verdict mismatch")

    if receipt["physical_promotion_allowed"] is not False:
        _fail("physical_promotion_allowed must be false")
    expected_receipt = bool(
        acyclic and sequence_compatible and receipt["controls_fail_closed"]
    )
    if receipt["SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT"] is not expected_receipt:
        _fail("receipt flag does not match its clauses")

    return {
        "receipt": True,
        "byte_identical": byte_identical,
        "generated_edge_count": len(regenerated),
        "declared_edge_count": len(receipt["declared_edges"]),
        "generated_only_pair_count": len(generated_only),
        "declared_only_pair_count": len(declared_only),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Independently verify the source-derived causal-order receipt."
        )
    )
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    args = parser.parse_args()
    try:
        result = verify_receipt(args.receipt)
    except IndependentVerificationError as error:
        print(f"REFUSED: {error}")
        return 1
    print(
        "verified: "
        f"byte_identical={result['byte_identical']} "
        f"generated={result['generated_edge_count']} "
        f"declared={result['declared_edge_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
