"""Independent finite replay for a capacity-indexed public-record family.

This module deliberately does not import the producer in
``reverse-engineering-reality``.  It rebuilds each declared deterministic
channel, constructs its confusability graph, and proves the finite zero-error
capacity by the graph's clique-component decomposition.

The replay is bounded by the rungs carried in the input projection.  It checks
the advertised closed forms on those rungs.  It is not an all-rung proof and
does not promote a physical or cosmological ``N`` closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


SCHEMA_ID = "oph.capacity_indexed_source_family_projection.v1"
SCIENTIFIC_VERDICT = "BOUNDED_COMPLETION_CLASS_NONIDENTIFIABLE"
SOURCE_RULE_ID = "oph.public-record-capacity.branch-completion-family.v1"
BASE_PUBLIC_ATOMS = 24
MAX_INPUT_BYTES = 2_000_000
MAX_GRAPH_VERTICES = 4_096
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "cosmology"
    / "capacity_indexed_source_family_projection.schema.json"
)

BRANCH_RULES = {
    "reversible_identity": "output=(port,copy)",
    "copy_collapse_erasure": "output=port",
    "capped_two_class": "output=(port,min(copy,1))",
    "hidden_spectator": "output=(port,copy);spectator-hidden",
}

FORMULA_IDS = {
    "reversible_identity": "M0=24*k",
    "copy_collapse_erasure": "M0=24",
    "capped_two_class": "M0=24*min(k,2)",
    "hidden_spectator": "raw_D=24*k*s;M0=24*k",
}

TARGET_CLEANLINESS_KEYS = {
    "measured_cosmological_constant_read",
    "observed_horizon_radius_read",
    "electroweak_target_read",
    "desired_capacity_read",
    "external_fit_read",
}


class ProjectionError(ValueError):
    """The projection is malformed or its finite claims do not replay."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the projection's single canonical JSON representation."""

    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ProjectionError(f"non-finite JSON number: {token}")


def load_projection(
    path: str | Path,
    *,
    schema_path: str | Path = SCHEMA_PATH,
) -> dict[str, Any]:
    """Load one byte-canonical projection and validate its strict schema."""

    candidate_path = Path(path)
    raw = candidate_path.read_bytes()
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ProjectionError("projection is empty or exceeds the bounded input limit")
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionError(f"projection is not strict ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ProjectionError("projection root must be an object")
    if raw != canonical_json_bytes(payload):
        raise ProjectionError("projection bytes are not canonical JSON")

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise ProjectionError(f"schema violation at {location}: {first.message}")
    return payload


def _source_signature(shared_source: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(shared_source)).hexdigest()


def _records_and_outputs(
    branch_id: str,
    *,
    k: int,
    spectator_multiplicity: int,
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], tuple[int, ...]]]:
    records: list[tuple[int, ...]] = []
    outputs: dict[tuple[int, ...], tuple[int, ...]] = {}
    if branch_id == "hidden_spectator":
        for port in range(BASE_PUBLIC_ATOMS):
            for copy in range(k):
                for spectator in range(spectator_multiplicity):
                    record = (port, copy, spectator)
                    records.append(record)
                    outputs[record] = (port, copy)
    else:
        for port in range(BASE_PUBLIC_ATOMS):
            for copy in range(k):
                record = (port, copy)
                records.append(record)
                if branch_id == "reversible_identity":
                    outputs[record] = record
                elif branch_id == "copy_collapse_erasure":
                    outputs[record] = (port,)
                elif branch_id == "capped_two_class":
                    outputs[record] = (port, min(copy, 1))
                else:
                    raise ProjectionError(f"unknown branch id: {branch_id}")
    if len(records) > MAX_GRAPH_VERTICES:
        raise ProjectionError("generated graph exceeds the bounded vertex limit")
    return records, outputs


def _confusability_graph(
    records: Sequence[tuple[int, ...]],
    outputs: Mapping[tuple[int, ...], tuple[int, ...]],
) -> dict[tuple[int, ...], set[tuple[int, ...]]]:
    """Build the graph independently from equal deterministic outputs."""

    fibers: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for record in records:
        fibers.setdefault(outputs[record], []).append(record)
    graph = {record: set() for record in records}
    for fiber in fibers.values():
        for index, left in enumerate(fiber):
            for right in fiber[index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def _connected_components(
    graph: Mapping[tuple[int, ...], set[tuple[int, ...]]],
) -> list[set[tuple[int, ...]]]:
    remaining = set(graph)
    components: list[set[tuple[int, ...]]] = []
    while remaining:
        root = min(remaining)
        component: set[tuple[int, ...]] = set()
        frontier = [root]
        while frontier:
            vertex = frontier.pop()
            if vertex in component:
                continue
            component.add(vertex)
            frontier.extend(graph[vertex] - component)
        remaining -= component
        components.append(component)
    return components


def _exact_capacity_from_graph(
    graph: Mapping[tuple[int, ...], set[tuple[int, ...]]],
) -> tuple[int, list[tuple[int, ...]], list[int]]:
    """Prove alpha(G) for a disjoint union of cliques.

    A lower witness chooses one vertex per connected component.  Completeness
    of every component as a clique supplies the matching upper bound.
    """

    components = _connected_components(graph)
    witness: list[tuple[int, ...]] = []
    component_sizes: list[int] = []
    for component in components:
        component_sizes.append(len(component))
        for vertex in component:
            if graph[vertex] & component != component - {vertex}:
                raise ProjectionError(
                    "confusability graph is outside the certified clique-component class"
                )
        witness.append(min(component))
    for index, left in enumerate(witness):
        for right in witness[index + 1 :]:
            if right in graph[left]:
                raise ProjectionError("constructed lower witness is not independent")
    return len(components), witness, sorted(component_sizes)


def _expected_capacity(branch_id: str, k: int) -> int:
    if branch_id == "reversible_identity":
        return BASE_PUBLIC_ATOMS * k
    if branch_id == "copy_collapse_erasure":
        return BASE_PUBLIC_ATOMS
    if branch_id == "capped_two_class":
        return BASE_PUBLIC_ATOMS * min(k, 2)
    if branch_id == "hidden_spectator":
        return BASE_PUBLIC_ATOMS * k
    raise ProjectionError(f"unknown branch id: {branch_id}")


def _expected_row_keys(
    branch_id: str,
    sample_rungs: Sequence[int],
    spectator_multiplicities: Sequence[int],
) -> set[tuple[int, int]]:
    if branch_id == "hidden_spectator":
        return {
            (k, spectator)
            for k in sample_rungs
            for spectator in spectator_multiplicities
        }
    return {(k, 1) for k in sample_rungs}


def _validate_common_ancestry(payload: Mapping[str, Any]) -> None:
    shared_source = payload["shared_source"]
    if shared_source["source_rule_id"] != SOURCE_RULE_ID:
        raise ProjectionError("unexpected shared source rule")
    if shared_source["base_public_atoms"] != BASE_PUBLIC_ATOMS:
        raise ProjectionError("unexpected base public-atom count")
    if shared_source.get("full_a1_a3_packet_lift_required") is not True:
        raise ProjectionError("full A1-A3 packet-lift boundary was removed")
    if payload["shared_source_signature_sha256"] != _source_signature(shared_source):
        raise ProjectionError("shared source signature does not match its canonical material")

    common_pins = payload["upstream_pins"]
    if not all(SHA256_RE.fullmatch(value) for value in common_pins.values()):
        raise ProjectionError("upstream pins must be lowercase sha256 digests")
    for branch in payload["branches"]:
        if branch["shared_source_signature_sha256"] != payload[
            "shared_source_signature_sha256"
        ]:
            raise ProjectionError("branch does not carry the common source signature")
        if branch["upstream_pins"] != common_pins:
            raise ProjectionError("branch does not carry the common upstream pins")


def verify_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Replay every declared finite row and return a fail-closed report."""

    try:
        if payload.get("schema") != SCHEMA_ID:
            raise ProjectionError("unexpected projection schema")
        if payload.get("scientific_verdict") != SCIENTIFIC_VERDICT:
            raise ProjectionError("unexpected scientific verdict")
        if payload.get("physical_n_closure_promoted") is not False:
            raise ProjectionError("bounded replay cannot promote physical N closure")
        cleanliness = payload.get("target_cleanliness")
        if (
            not isinstance(cleanliness, Mapping)
            or set(cleanliness) != TARGET_CLEANLINESS_KEYS
            or any(
                value is not False for value in cleanliness.values()
            )
        ):
            raise ProjectionError("target-cleanliness flags must all be false")
        _validate_common_ancestry(payload)

        sample_rungs = payload["shared_source"]["sample_rungs"]
        spectators = payload["shared_source"]["spectator_multiplicities"]
        expected_branches = set(BRANCH_RULES)
        observed_branches = {branch["branch_id"] for branch in payload["branches"]}
        if observed_branches != expected_branches:
            raise ProjectionError("projection does not contain the complete branch grammar")
        if len(payload["branches"]) != len(expected_branches):
            raise ProjectionError("branch ids must be unique")

        branch_reports: list[dict[str, Any]] = []
        bounded_zero_sets: dict[str, list[dict[str, int]]] = {}
        for branch in sorted(payload["branches"], key=lambda row: row["branch_id"]):
            branch_id = branch["branch_id"]
            if branch["channel_rule"] != BRANCH_RULES[branch_id]:
                raise ProjectionError(f"{branch_id}: channel rule drift")
            if branch["capacity_formula"] != FORMULA_IDS[branch_id]:
                raise ProjectionError(f"{branch_id}: capacity formula drift")
            rows = branch["sample_rows"]
            observed_keys = {
                (row["k"], row["spectator_multiplicity"]) for row in rows
            }
            expected_keys = _expected_row_keys(branch_id, sample_rungs, spectators)
            if len(observed_keys) != len(rows) or observed_keys != expected_keys:
                raise ProjectionError(f"{branch_id}: incomplete or duplicate sample grid")

            zero_rows: list[dict[str, int]] = []
            replayed_rows: list[dict[str, Any]] = []
            for row in sorted(
                rows, key=lambda item: (item["k"], item["spectator_multiplicity"])
            ):
                k = row["k"]
                spectator = row["spectator_multiplicity"]
                records, outputs = _records_and_outputs(
                    branch_id,
                    k=k,
                    spectator_multiplicity=spectator,
                )
                graph = _confusability_graph(records, outputs)
                capacity, witness, component_sizes = _exact_capacity_from_graph(graph)
                expected_capacity = _expected_capacity(branch_id, k)
                raw_dimension = (
                    BASE_PUBLIC_ATOMS * k * spectator
                    if branch_id == "hidden_spectator"
                    else BASE_PUBLIC_ATOMS * k
                )
                public_dimension = BASE_PUBLIC_ATOMS * k
                slack_zero = raw_dimension == capacity
                expected_row = {
                    "k": k,
                    "spectator_multiplicity": spectator,
                    "raw_dimension": raw_dimension,
                    "public_dimension": public_dimension,
                    "claimed_capacity_M0": expected_capacity,
                    "claimed_slack_zero": slack_zero,
                }
                if row != expected_row:
                    raise ProjectionError(f"{branch_id}: claimed row does not replay")
                if capacity != expected_capacity or len(set(outputs.values())) != capacity:
                    raise ProjectionError(f"{branch_id}: formula or channel replay failed")
                if slack_zero:
                    zero_rows.append(
                        {"k": k, "spectator_multiplicity": spectator}
                    )
                replayed_rows.append(
                    {
                        **expected_row,
                        "graph_vertex_count": len(graph),
                        "graph_edge_count": sum(map(len, graph.values())) // 2,
                        "clique_component_count": capacity,
                        "component_size_multiset": component_sizes,
                        "independent_witness_size": len(witness),
                    }
                )
            if zero_rows != branch["claimed_bounded_zero_set"]:
                raise ProjectionError(f"{branch_id}: bounded zero-set claim drift")
            bounded_zero_sets[branch_id] = zero_rows
            branch_reports.append(
                {
                    "branch_id": branch_id,
                    "capacity_formula": FORMULA_IDS[branch_id],
                    "sample_rows": replayed_rows,
                    "bounded_zero_set": zero_rows,
                    "exact_graph_certificate": "disjoint-clique-components",
                }
            )

        if len({json.dumps(value, sort_keys=True) for value in bounded_zero_sets.values()}) < 2:
            raise ProjectionError("branch grammar did not realize distinct bounded zero sets")

        return {
            "schema": "oph.capacity_indexed_source_family_independent_replay.v1",
            "status": "PASS",
            "scientific_verdict_replayed": SCIENTIFIC_VERDICT,
            "shared_source_signature_sha256": payload[
                "shared_source_signature_sha256"
            ],
            "upstream_pins": payload["upstream_pins"],
            "target_clean": True,
            "complete_declared_branch_grammar": True,
            "distinct_bounded_zero_sets": True,
            "branch_reports": branch_reports,
            "scope": {
                "finite_sample_replay": True,
                "all_positive_integer_rungs_proved": False,
                "producer_implementation_independent": True,
                "physical_n_closure_promoted": False,
                "full_a1_a3_packet_lift_replayed": False,
            },
        }
    except (KeyError, TypeError, ProjectionError, ValueError) as error:
        return {
            "schema": "oph.capacity_indexed_source_family_independent_replay.v1",
            "status": "FAIL",
            "error": str(error),
            "scope": {
                "finite_sample_replay": False,
                "all_positive_integer_rungs_proved": False,
                "producer_implementation_independent": True,
                "physical_n_closure_promoted": False,
            },
        }


def verify_projection_file(
    path: str | Path,
    *,
    schema_path: str | Path = SCHEMA_PATH,
) -> dict[str, Any]:
    try:
        payload = load_projection(path, schema_path=schema_path)
    except (OSError, ProjectionError, ValueError) as error:
        return {
            "schema": "oph.capacity_indexed_source_family_independent_replay.v1",
            "status": "FAIL",
            "error": str(error),
            "scope": {
                "finite_sample_replay": False,
                "all_positive_integer_rungs_proved": False,
                "producer_implementation_independent": True,
                "physical_n_closure_promoted": False,
            },
        }
    return verify_projection(payload)


def compact_replay_receipt(
    report: Mapping[str, Any],
    *,
    projection_path: str | Path,
    schema_path: str | Path = SCHEMA_PATH,
) -> dict[str, Any]:
    """Bind a compact custody receipt to the exact independent replay."""

    if report.get("status") != "PASS":
        raise ProjectionError("cannot bind a failed replay")
    projection_bytes = Path(projection_path).read_bytes()
    verifier_bytes = Path(__file__).read_bytes()
    schema_bytes = Path(schema_path).read_bytes()
    full_report_bytes = canonical_json_bytes(report)
    return {
        "schema": "oph.capacity_indexed_source_family_independent_receipt.v1",
        "issue": 551,
        "status": "PASS",
        "scientific_verdict_replayed": report["scientific_verdict_replayed"],
        "projection_sha256": (
            "sha256:" + hashlib.sha256(projection_bytes).hexdigest()
        ),
        "schema_sha256": "sha256:" + hashlib.sha256(schema_bytes).hexdigest(),
        "independent_verifier_sha256": (
            "sha256:" + hashlib.sha256(verifier_bytes).hexdigest()
        ),
        "full_replay_report_sha256": (
            "sha256:" + hashlib.sha256(full_report_bytes).hexdigest()
        ),
        "shared_source_signature_sha256": report[
            "shared_source_signature_sha256"
        ],
        "upstream_pins": report["upstream_pins"],
        "branch_ids_replayed": [
            row["branch_id"] for row in report["branch_reports"]
        ],
        "scope": report["scope"],
        "target_clean": report["target_clean"],
    }


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", type=Path)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--receipt-output", type=Path)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    report = verify_projection_file(arguments.projection, schema_path=arguments.schema)
    if arguments.receipt_output is not None and report["status"] == "PASS":
        receipt = compact_replay_receipt(
            report,
            projection_path=arguments.projection,
            schema_path=arguments.schema,
        )
        arguments.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt_output.write_bytes(canonical_json_bytes(receipt))
    print(canonical_json_bytes(report).decode("ascii"), end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
