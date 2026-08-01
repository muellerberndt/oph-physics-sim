"""Independent replay for the refined equal-seam source-selection gate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/refinement/refined_equal_seam_source_gate_receipt.json"
COORDINATE_RESIDUAL_GATE = 5.0e-11


class VerificationError(RuntimeError):
    """Raised when a committed source-gate claim does not replay."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _independent_rotations() -> tuple[np.ndarray, ...]:
    base = build_geodesic_icosahedral_tower(0).levels[0]
    anchor = [int(value) for value in base.faces[0]]
    source = np.asarray(base.vertices[anchor], dtype=float).T
    inverse_source = np.linalg.inv(source)
    rotations: list[np.ndarray] = []
    for permutation in icosahedral_a5_port_permutations():
        target = np.asarray(
            base.vertices[[int(permutation[index]) for index in anchor]],
            dtype=float,
        ).T
        rotation = target @ inverse_source
        _fail(float(np.linalg.det(rotation)) > 0.0, "improper A5 rotation")
        _fail(
            float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) < 5.0e-12,
            "non-orthogonal A5 rotation",
        )
        rotations.append(rotation)
    _fail(len(rotations) == 60, "proper A5 action does not have order 60")
    return tuple(rotations)


def _independent_orbit_rows(max_level: int) -> list[dict[str, Any]]:
    tower = build_geodesic_icosahedral_tower(max_level)
    rotations = _independent_rotations()
    output: list[dict[str, Any]] = []
    for level, mesh in enumerate(tower.levels):
        tree = cKDTree(np.asarray(mesh.vertices, dtype=float))
        actions: list[np.ndarray] = []
        maximum_residual = 0.0
        for rotation in rotations:
            mapped = np.asarray(mesh.vertices, dtype=float) @ rotation.T
            distances, indices = tree.query(mapped, k=1)
            permutation = np.asarray(indices, dtype=np.int64)
            _fail(
                np.unique(permutation).size == mesh.vertex_count,
                "refined action is not a permutation",
            )
            maximum_residual = max(maximum_residual, float(np.max(distances)))
            actions.append(permutation)
        _fail(
            maximum_residual <= COORDINATE_RESIDUAL_GATE,
            "refined coordinate residual exceeds gate",
        )

        edges = {
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in mesh.edges
        }
        remaining = set(edges)
        orbit_sizes: list[int] = []
        while remaining:
            edge = next(iter(remaining))
            orbit = {
                tuple(
                    sorted(
                        (
                            int(permutation[edge[0]]),
                            int(permutation[edge[1]]),
                        )
                    )
                )
                for permutation in actions
            }
            _fail(orbit <= edges, "A5 edge incidence failed")
            remaining.difference_update(orbit)
            orbit_sizes.append(len(orbit))
        multiplicities = Counter(orbit_sizes)
        output.append(
            {
                "level": level,
                "edge_count": len(edges),
                "edge_orbit_count": len(orbit_sizes),
                "edge_orbit_size_multiplicities": {
                    str(size): int(count)
                    for size, count in sorted(multiplicities.items())
                },
                "maximum_coordinate_residual": maximum_residual,
                "geometry_hash": mesh.geometry_hash,
            }
        )
    return output


def verify_receipt(receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _fail(
        receipt.get("schema") == "oph.refined-equal-seam-source-selection-gate.v1",
        "schema mismatch",
    )
    stored_payload_hash = receipt.pop("payload_sha256", None)
    _fail(
        stored_payload_hash == _sha256_bytes(_canonical_bytes(receipt)),
        "payload hash mismatch",
    )
    receipt["payload_sha256"] = stored_payload_hash

    for pin in receipt.get("source_pins", []):
        path = ROOT / str(pin["path"])
        payload = path.read_bytes()
        _fail(len(payload) == int(pin["bytes"]), f"source size drift: {path}")
        _fail(
            _sha256_bytes(payload) == pin["sha256"],
            f"source hash drift: {path}",
        )

    parent = receipt["parent_bridge"]
    bounded_path = ROOT / parent["bounded_repair_receipt"]
    biposh_path = ROOT / parent["biposh_receipt"]
    bounded = json.loads(bounded_path.read_text(encoding="utf-8"))
    biposh = json.loads(biposh_path.read_text(encoding="utf-8"))
    _fail(
        bounded.get("certificate_payload_sha256")
        == parent["bounded_repair_certificate_payload_sha256"],
        "bounded parent certificate pin mismatch",
    )
    _fail(
        biposh.get("payload_sha256") == parent["biposh_payload_sha256"],
        "BipoSH parent payload pin mismatch",
    )

    committed_rows = receipt["edge_orbit_rows"]
    max_level = max(int(row["level"]) for row in committed_rows)
    rebuilt_rows = _independent_orbit_rows(max_level)
    _fail(len(rebuilt_rows) == len(committed_rows), "level count mismatch")
    checked_edges = 0
    for committed, rebuilt in zip(committed_rows, rebuilt_rows, strict=True):
        for key in (
            "level",
            "edge_count",
            "edge_orbit_count",
            "edge_orbit_size_multiplicities",
            "geometry_hash",
        ):
            _fail(committed[key] == rebuilt[key], f"orbit row drift at {key}")
        _fail(
            abs(
                float(committed["maximum_coordinate_residual"])
                - float(rebuilt["maximum_coordinate_residual"])
            )
            <= 5.0e-14,
            "coordinate residual drift",
        )
        _fail(
            float(committed["coordinate_residual_gate"]) == COORDINATE_RESIDUAL_GATE,
            "coordinate residual gate drift",
        )
        _fail(
            committed["registered_mesh_permutation_residual_gate_passed"] is True,
            "registered-mesh residual gate was not attained",
        )
        _fail(
            int(committed["symmetry_invariant_normalized_weight_simplex_dimension"])
            == int(committed["edge_orbit_count"]) - 1,
            "invariant-weight dimension mismatch",
        )
        _fail(
            bool(committed["a5_symmetry_alone_forces_one_weight_on_all_edges"])
            == (int(committed["edge_orbit_count"]) == 1),
            "A5 transitivity verdict mismatch",
        )
        checked_edges += int(committed["edge_count"])

    finding = receipt["classification_finding"]
    _fail(finding["base_edge_alphabet_is_one_a5_orbit"] is True, "base verdict drift")
    _fail(
        finding["refined_edge_alphabets_have_multiple_a5_orbits"] is True,
        "refined orbit verdict drift",
    )
    _fail(
        finding["a5_forces_relative_weights_between_distinct_edge_orbits"] is False,
        "cross-orbit symmetry verdict drift",
    )
    decision = receipt["selection_decision"]
    _fail(
        decision["registered_mesh_a5_edge_orbits_classified_with_residual_gate"]
        is True,
        "registered-mesh orbit classification gate was not attained",
    )
    _fail(
        decision["all_level_complete_atomic_counting_law_source_emitted"] is False,
        "source-emitter gate was promoted without a receipt",
    )
    _fail(decision["promotion_allowed"] is False, "promotion gate must remain closed")
    clause = receipt["minimal_constructive_clause"]
    _fail(
        clause["fourth_axiom_logically_required"] is False,
        "the finite audit cannot require a fourth axiom",
    )
    _fail(
        clause["canonical_basis_amendment_required_before_unconditional_use"] is True,
        "canonical clause-amendment boundary was lost",
    )
    _fail(
        clause["additional_branch_or_source_premise_until_derived"] is True
        and clause["derived_from_canonical_a1_a3_by_this_packet"] is False,
        "unit-counting premise boundary was lost",
    )
    return {
        "status": "PASS",
        "checked_levels": len(committed_rows),
        "checked_edges": checked_edges,
        "observed_orbit_counts": [
            int(row["edge_orbit_count"]) for row in committed_rows
        ],
    }


def main() -> int:
    result = verify_receipt()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
