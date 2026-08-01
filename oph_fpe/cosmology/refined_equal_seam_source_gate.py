"""Audit the source-selection gate for equal seams on the A5 refinement tower.

This target-clean packet asks a narrow question needed by issue 659.  Does the
proper icosahedral action make every registered edge at a refinement level the
same atomic event?  It reconstructs the full 60-element action in binary64,
matches it to registered mesh permutations under an explicit residual gate,
and classifies the induced edge orbits.  No sky, particle, or laboratory datum
is read.

The result separates two mechanisms which must not be conflated:

* A5 presentation symmetry equates weights inside an edge orbit.
* A source-emitted complete atomic counting law can equate weights across
  distinct edge orbits.

The latter is a constructive open interface.  It may be supplied by precise
A1-R/A2-R/A3 clauses without introducing a fourth axiom, but it is not a
consequence of the canonical A1--A3 structures as presently registered.
"""

from __future__ import annotations

import argparse
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
BOUNDED_REPAIR_RECEIPT = (
    ROOT / "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
)
BIPOSH_RECEIPT = ROOT / "data/refinement/a5_biposh_dual_operator_receipt.json"
MAX_LEVEL = 5
COORDINATE_RESIDUAL_GATE = 5.0e-11

EXPECTED_BOUNDED_STATUS = (
    "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_"
    "FROZEN_ADVERSARIAL_SUITE"
)
EXPECTED_BOUNDED_CERTIFICATE_SHA256 = (
    "sha256:9e87c5e4abfb3baed80058ffc832a6dbd3412f386eb383d68fee4ebee10c00d5"
)
EXPECTED_BIPOSH_STATUS = (
    "FINITE_DUAL_OPERATOR_FINGERPRINT_ATTAINED__CONTINUUM_RESIDUAL_AND_"
    "PHYSICAL_COVARIANCE_OPEN"
)


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


def _file_pin(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _load_parent_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    bounded = json.loads(BOUNDED_REPAIR_RECEIPT.read_text(encoding="utf-8"))
    biposh = json.loads(BIPOSH_RECEIPT.read_text(encoding="utf-8"))
    if bounded.get("schema") != "oph.bounded_atomic_self_readback_closure.v1":
        raise ValueError("unexpected bounded-repair parent schema")
    if bounded.get("status") != EXPECTED_BOUNDED_STATUS:
        raise ValueError("bounded-repair parent status drift")
    if bounded.get("certificate_payload_sha256") != EXPECTED_BOUNDED_CERTIFICATE_SHA256:
        raise ValueError("bounded-repair certificate payload drift")
    if biposh.get("schema") != "oph.a5-biposh-dual-operator-refinement.v1":
        raise ValueError("unexpected BipoSH parent schema")
    if biposh.get("status") != EXPECTED_BIPOSH_STATUS:
        raise ValueError("BipoSH parent status drift")
    bridge = biposh.get("bounded_repair_generator_bridge", {})
    if (
        bridge.get("parent_certificate_payload_sha256")
        != EXPECTED_BOUNDED_CERTIFICATE_SHA256
    ):
        raise ValueError("BipoSH-to-bounded-repair parent pin drift")
    if (
        bridge.get(
            "base_carrier_operator_matches_bounded_reconstructed_one_atom_mean_generator_up_to_scale"
        )
        is not True
    ):
        raise ValueError("base equal-seam generator bridge is not attained")
    return bounded, biposh


def _rotation_matrices() -> tuple[np.ndarray, ...]:
    """Recover binary64 matrices from the public 60-permutation A5 action."""

    base = build_geodesic_icosahedral_tower(0).levels[0]
    anchor = tuple(int(value) for value in base.faces[0])
    source = np.asarray(base.vertices[list(anchor)], dtype=float).T
    inverse_source = np.linalg.inv(source)
    matrices: list[np.ndarray] = []
    for permutation in icosahedral_a5_port_permutations():
        target_indices = [int(permutation[index]) for index in anchor]
        target = np.asarray(base.vertices[target_indices], dtype=float).T
        rotation = target @ inverse_source
        if float(np.linalg.det(rotation)) <= 0.0:
            raise AssertionError("public A5 permutation induced an improper rotation")
        if float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) > 5.0e-12:
            raise AssertionError("public A5 permutation induced a non-orthogonal map")
        matrices.append(rotation)
    if len(matrices) != 60:
        raise AssertionError("the public proper-rotation action must contain 60 rows")
    return tuple(matrices)


def _vertex_permutation(
    vertices: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    mapped = np.asarray(vertices, dtype=float) @ np.asarray(rotation, dtype=float).T
    distances, indices = cKDTree(np.asarray(vertices, dtype=float)).query(mapped, k=1)
    permutation = np.asarray(indices, dtype=np.int64)
    if np.unique(permutation).size != vertices.shape[0]:
        raise AssertionError("A5 rotation did not induce a vertex permutation")
    residual = float(np.max(distances)) if distances.size else 0.0
    return permutation, residual


def classify_edge_orbits(max_level: int = MAX_LEVEL) -> list[dict[str, Any]]:
    """Classify registered-mesh edge orbits under the residual-gated action."""

    if max_level < 0:
        raise ValueError("max_level must be nonnegative")
    tower = build_geodesic_icosahedral_tower(max_level)
    rotations = _rotation_matrices()
    rows: list[dict[str, Any]] = []
    for level, mesh in enumerate(tower.levels):
        edges = {
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in mesh.edges
        }
        vertex_actions: list[np.ndarray] = []
        maximum_coordinate_residual = 0.0
        for rotation in rotations:
            permutation, residual = _vertex_permutation(mesh.vertices, rotation)
            maximum_coordinate_residual = max(maximum_coordinate_residual, residual)
            vertex_actions.append(permutation)
        if maximum_coordinate_residual > COORDINATE_RESIDUAL_GATE:
            raise AssertionError("refined A5 action misses registered vertices")

        unseen = set(edges)
        orbit_sizes: list[int] = []
        incidence_preserved = True
        while unseen:
            representative = min(unseen)
            orbit: set[tuple[int, int]] = set()
            for permutation in vertex_actions:
                left = int(permutation[representative[0]])
                right = int(permutation[representative[1]])
                mapped_edge = (min(left, right), max(left, right))
                if mapped_edge not in edges:
                    incidence_preserved = False
                    break
                orbit.add(mapped_edge)
            if not incidence_preserved:
                break
            unseen.difference_update(orbit)
            orbit_sizes.append(len(orbit))
        if not incidence_preserved or unseen:
            raise AssertionError("A5 action does not partition the registered edges")

        size_multiplicities = Counter(orbit_sizes)
        orbit_count = len(orbit_sizes)
        edge_count = len(edges)
        expected_edge_count = 30 * (4**level)
        if edge_count != expected_edge_count:
            raise AssertionError("geodesic tower edge count drift")
        if sum(orbit_sizes) != edge_count:
            raise AssertionError("edge orbits do not exhaust the registered alphabet")
        rows.append(
            {
                "level": level,
                "frequency": 2**level,
                "vertex_count": int(mesh.vertex_count),
                "edge_count": edge_count,
                "proper_rotation_count": len(vertex_actions),
                "edge_orbit_count": orbit_count,
                "edge_orbit_size_multiplicities": {
                    str(size): int(count)
                    for size, count in sorted(size_multiplicities.items())
                },
                "symmetry_invariant_normalized_weight_simplex_dimension": (
                    orbit_count - 1
                ),
                "a5_symmetry_alone_forces_one_weight_on_all_edges": (orbit_count == 1),
                "edge_incidence_preserved": incidence_preserved,
                "maximum_coordinate_residual": maximum_coordinate_residual,
                "coordinate_residual_gate": COORDINATE_RESIDUAL_GATE,
                "registered_mesh_permutation_residual_gate_passed": (
                    maximum_coordinate_residual <= COORDINATE_RESIDUAL_GATE
                ),
                "geometry_hash": mesh.geometry_hash,
            }
        )
    return rows


def build_refined_equal_seam_source_gate(
    max_level: int = MAX_LEVEL,
) -> dict[str, Any]:
    bounded, biposh = _load_parent_receipts()
    rows = classify_edge_orbits(max_level)
    expected_counts = [1] + [
        2 * (4 ** (level - 1)) for level in range(1, max_level + 1)
    ]
    observed_counts = [int(row["edge_orbit_count"]) for row in rows]
    if observed_counts != expected_counts:
        raise AssertionError("refined edge-orbit inventory drift")

    source_files = [
        ROOT / "oph_fpe/core/icosahedral.py",
        BOUNDED_REPAIR_RECEIPT,
        BIPOSH_RECEIPT,
        Path(__file__).resolve(),
        ROOT / "oph_fpe/cosmology/verify_refined_equal_seam_source_gate_independent.py",
        ROOT / "tests/test_refined_equal_seam_source_gate.py",
    ]
    receipt: dict[str, Any] = {
        "schema": "oph.refined-equal-seam-source-selection-gate.v1",
        "issue": 659,
        "status": (
            "BASE_EQUAL_SEAM_GENERATOR_EXACT__REGISTERED_MESH_A5_EDGE_ORBITS_"
            "CLASSIFIED_WITH_RESIDUAL_GATE__SOURCE_COUNTING_EMITTER_OPEN"
        ),
        "source_scope": {
            "geometry": "registered nested geodesic icosahedral vertex tower",
            "levels": list(range(max_level + 1)),
            "event_candidate": "one completed atomic reconciliation attempt per registered unoriented seam",
            "proper_rotation_group_order": 60,
            "orbit_classification_arithmetic": (
                "binary64 rotations, nearest registered-vertex permutations under "
                "the declared residual gate, then integer incidence and orbit census"
            ),
            "external_comparison_data_used": False,
            "sky_data_used": False,
            "particle_data_used": False,
            "target_values_used": False,
        },
        "parent_bridge": {
            "bounded_repair_receipt": BOUNDED_REPAIR_RECEIPT.relative_to(
                ROOT
            ).as_posix(),
            "bounded_repair_schema": bounded["schema"],
            "bounded_repair_status": bounded["status"],
            "bounded_repair_certificate_payload_sha256": bounded[
                "certificate_payload_sha256"
            ],
            "base_one_atom_conditional_mean": "E[X_next | X=x] = (I - L_icosahedron/60) x",
            "base_equal_seam_generator_exact_in_bounded_realization": True,
            "biposh_receipt": BIPOSH_RECEIPT.relative_to(ROOT).as_posix(),
            "biposh_payload_sha256": biposh["payload_sha256"],
            "biposh_status": biposh["status"],
        },
        "edge_orbit_rows": rows,
        "classification_finding": {
            "base_edge_alphabet_is_one_a5_orbit": True,
            "refined_edge_alphabets_have_multiple_a5_orbits": all(
                int(row["edge_orbit_count"]) > 1 for row in rows[1:]
            ),
            "a5_forces_equal_weights_within_each_edge_orbit": True,
            "a5_forces_relative_weights_between_distinct_edge_orbits": False,
            "a5_symmetry_is_not_the_source_counting_emitter": True,
            "canonical_a1_a3_registered_structures_supply_cross_orbit_weights": False,
            "framework_wide_no_go": False,
        },
        "minimal_constructive_clause": {
            "a1_atomic_identity": (
                "At every registered finite level, the complete primitive event "
                "alphabet contains exactly one event of the same type and unit "
                "counting measure for each registered unoriented seam, with no "
                "duplicates, omitted seams, hidden event species, or orbit-dependent "
                "multiplicity. The registered refinement map preserves that event "
                "identity and its unit measure."
            ),
            "a2_completed_reconciliation": (
                "Each primitive event completes the local scalar reconciliation on "
                "its two endpoints, changes no other scalar entry, makes the endpoint "
                "readings agree, preserves their sum, and is natural under every "
                "admitted presentation and refinement map."
            ),
            "a3_counting_reference": (
                "The exact reference on the complete primitive alphabet is normalized "
                "unit counting across all edge orbits. Its information objective is "
                "nonnegative and has that reference as its unique feasible zero; no "
                "additional scheduling constraint distinguishes an orbit."
            ),
            "refinement_compatibility": (
                "The event identification, completed reconciliation, reference, and "
                "coarse-graining maps form the declared refinement commuting diagram. "
                "A continuum claim additionally requires a proved limit theorem."
            ),
            "may_be_integrated_as_a1_a2_a3_clause_refinement": True,
            "additional_branch_or_source_premise_until_derived": True,
            "derived_from_canonical_a1_a3_by_this_packet": False,
            "fourth_axiom_logically_required": False,
            "canonical_basis_amendment_required_before_unconditional_use": True,
        },
        "conditional_theorem_bridge": {
            "repository": "FloatingPragma/observer-patch-holography",
            "path": "Lean/ObserverPatchHolography/EqualSeamSelection.lean",
            "theorems": [
                "completed_eq_pairAverage",
                "exactCounting_selected_eq_inverse_card",
                "exactCountingRepair_eq_id_sub_graphLaplacian",
                "selected_thirty_seam_repair",
            ],
            "finite_conclusion": "K = I - L/(2*edge_count)",
            "uses_new_lean_axiom": False,
            "cross_repository_source_hash_pinned_here": False,
        },
        "selection_decision": {
            "base_equal_seam_operator_selected_in_bounded_realization": True,
            "registered_mesh_a5_edge_orbits_classified_with_residual_gate": True,
            "all_level_complete_atomic_counting_law_source_emitted": False,
            "refinement_commuting_diagram_discharged": False,
            "continuum_equal_seam_operator_selected": False,
            "physical_repair_law_selected": False,
            "physical_covariance_selected": False,
            "promotion_allowed": False,
        },
        "reopen_or_advance_condition": (
            "Advance the equal-seam producer only after a target-clean source emitter "
            "constructs the complete all-level atomic event alphabet, common unit "
            "counting reference across every listed A5 edge orbit, completed A2 "
            "reconciliation grammar, and refinement compatibility. A5 symmetry by "
            "itself is not that emitter."
        ),
        "claim_boundary": (
            "Target-clean residual-gated classification of the proper-rotation edge "
            "orbits on the registered finite tower, joined to the exact bounded base "
            "generator. The orbit census uses binary64 rotations and nearest registered-"
            "vertex permutations with a maximum residual gate of 5e-11, followed by "
            "integer incidence checks. The packet identifies a constructive source "
            "contract for equal seams. The unit-counting law is an additional branch or "
            "source premise unless it is derived from the canonical structures; placing "
            "it inside an A1/A2/A3 clause refinement does not itself derive it. The "
            "packet does not claim that canonical A1--A3 discharge the contract, that "
            "the refined equal-seam operator is physical, or that a continuum covariance "
            "or observable has been selected. The multiple refined A5 orbits are not a "
            "framework-wide no-go: a target-clean source emitter can still supply one "
            "common primitive counting law."
        ),
        "source_pins": [_file_pin(path) for path in source_files],
    }
    receipt["payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return receipt


def write_refined_equal_seam_source_gate(
    receipt_path: Path = DEFAULT_RECEIPT,
    max_level: int = MAX_LEVEL,
) -> dict[str, Any]:
    receipt = build_refined_equal_seam_source_gate(max_level=max_level)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(_canonical_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--max-level", type=int, default=MAX_LEVEL)
    args = parser.parse_args()
    receipt = write_refined_equal_seam_source_gate(
        receipt_path=args.receipt,
        max_level=args.max_level,
    )
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
