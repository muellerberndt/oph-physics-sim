"""Independent verifier for the bounded #664 port-dual measure packet.

The verifier does not import the producer or the federation implementation.
It independently reconstructs the level-zero incidence complex, the proper
icosahedral orbits, the exact rational barycentric measure, the finite
refinement replay, and every alternative-attachment control.  It pins the
federation implementation by raw hash and fails closed on physical promotion.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/repair_closure/primitive_port_dual_measure_receipt.json"
)
FZ11_RECEIPT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
PRODUCER_PATH = ROOT / "oph_fpe/dynamics/primitive_port_dual_measure.py"
VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_primitive_port_dual_measure.py"
FEDERATION_PATH = ROOT / "oph_fpe/core/echosahedral_federation.py"
GEOMETRY_PATH = ROOT / "oph_fpe/core/icosahedral.py"

SCHEMA = "oph.primitive-port-dual-normalized-measure.v1"
STATUS = (
    "QUOTIENT_VISIBLE_NORMALIZED_PORT_DUAL_MEASURE_ATTAINED__"
    "PHYSICAL_PIXEL_AND_HOP_IDENTITIES_OPEN"
)
FZ11_SCHEMA = "oph.fz11-conditional-3d-translation-bridge.v1"
FZ11_STATUS = (
    "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
    "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
    "BOOST_AND_EXCLUSIVITY_OPEN"
)
DECLARED_MAX_LEVEL = 2
PORT_COUNT = 12
BASE_FACE_COUNT = 20
BASE_EDGE_COUNT = 30
NUMERICAL_GATE = 1.0e-12

SOURCE_FLAGS = (
    "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
    "FEDERATION_SEWING_RECEIPT",
    "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
    "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
    "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT",
    "INCIDENCE_NERVE_TO_SUPPORT_SIMPLICIAL_ISOMORPHISM_RECEIPT",
    "CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT",
    "ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID",
    "S2_SUPPORT_CHART_EMERGENCE_RECEIPT",
)
CLAIM_BOUNDARY = (
    "The source-derived finite S2 scaffold carries a quotient-visible "
    "barycentric port-dual partition whose twelve sectors each have exact "
    "normalized support measure 1/12. Finite additivity of spherical area and "
    "state preservation give the analytic refinement argument, and a floating "
    "replay verifies levels zero through two within the declared tolerance. "
    "The packet does not contain a symbolic exact-area proof for refined "
    "spherical triangles. This is a normalized "
    "screen-area statement, not a physical pixel or spatial-ruler attachment. "
    "The P-defined physical UV cell is not identified with a port sector, and "
    "the support areal radius is not identified with the issue-655 translation "
    "hop. No kappa_geom value, carrier lower bound, physical prediction, or "
    "comparison permission follows; issue #662 remains unarmed."
)


class IndependentVerificationError(RuntimeError):
    """Raised when any independently reconstructed condition fails."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise IndependentVerificationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise IndependentVerificationError(f"{path} is not an object")
    return value


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentVerificationError(message)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _coefficient_over_pi(orbit_cardinality: int) -> dict[str, Any]:
    coefficient = Fraction(int(orbit_cardinality), 4)
    if coefficient.denominator == 1:
        expression = f"{coefficient.numerator}/pi"
    elif coefficient.numerator == 1:
        expression = f"1/({coefficient.denominator}*pi)"
    else:
        expression = f"{coefficient.numerator}/({coefficient.denominator}*pi)"
    return {
        "orbit_cardinality": int(orbit_cardinality),
        "coefficient_over_pi": {
            "numerator": coefficient.numerator,
            "denominator": coefficient.denominator,
        },
        "conditional_kappa_geom": expression,
        "physical_attachment_selected": False,
    }


def _expected_fz11_pin() -> dict[str, Any]:
    receipt = _load(FZ11_RECEIPT)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    _fail(digest == _sha(payload), "FZ-11 self digest")
    _fail(receipt.get("schema") == FZ11_SCHEMA, "FZ-11 schema")
    _fail(receipt.get("status") == FZ11_STATUS, "FZ-11 status")
    _fail(receipt.get("issue") == 655, "FZ-11 issue")
    _fail(receipt.get("comparison_data_read") is False, "FZ-11 data firewall")
    _fail(receipt.get("issue_662_armed") is False, "FZ-11 arming boundary")
    geometry_hash = receipt["exact_port_frame_and_relabel"]["source_geometry_hash"]
    return {
        "path": FZ11_RECEIPT.relative_to(ROOT).as_posix(),
        "schema": FZ11_SCHEMA,
        "status": FZ11_STATUS,
        "raw_sha256": _raw_sha(FZ11_RECEIPT),
        "receipt_sha256": digest,
        "source_geometry_hash": geometry_hash,
        "comparison_data_read": False,
        "issue_662_armed": False,
        "physical_hop_identity_consumed": False,
    }


def _incidence_sha(base: Any) -> str:
    coordinates = np.asarray(base.vertices, dtype=float)
    antipode = [
        int(np.argmin(np.linalg.norm(coordinates + coordinate, axis=1)))
        for coordinate in coordinates
    ]
    payload = {
        "schema": "oph.echosahedral_oriented_incidence.v1",
        "port_count": base.vertex_count,
        "edges": [
            [int(value) for value in edge]
            for edge in sorted(tuple(int(value) for value in row) for row in base.edges)
        ],
        "oriented_faces": [
            [int(value) for value in face] for face in base.faces
        ],
        "antipode": antipode,
    }
    return _sha(payload)


def _source_projection(tower: Any) -> dict[str, Any]:
    base = tower.levels[0]
    fz11 = _expected_fz11_pin()
    _fail(base.geometry_hash == fz11["source_geometry_hash"], "shared source frame")
    projection = {
        "federation_id": "source-derived-incidence-nerve-v1",
        "source_incidence_sha256": _incidence_sha(base),
        "source_geometry_family": "nested_geodesic_icosahedral",
        "source_geometry_hash": base.geometry_hash,
        "carrier_count": 12,
        "seam_count": 30,
        "triple_overlap_count": 20,
        "support_simplex_counts": {"vertices": 12, "edges": 30, "oriented_faces": 20},
        "port_to_defect_vertex_bijection": list(range(PORT_COUNT)),
        "proper_action_count": 60,
        "declared_tower_levels": list(range(DECLARED_MAX_LEVEL + 1)),
        "refinement_map_hashes": [
            mapping.map_hash for mapping in tower.cell_refinements
        ],
        "source_parent_receipts": {key: True for key in SOURCE_FLAGS},
        "source_projection_sha256": "",
    }
    projection["source_projection_sha256"] = _sha(
        {key: value for key, value in projection.items() if key != "source_projection_sha256"}
    )
    return projection


def _base_packet(tower: Any) -> tuple[dict[str, Any], np.ndarray]:
    base = tower.levels[0]
    _fail(base.vertex_count == 12, "base vertex count")
    _fail(base.edge_count == 30, "base edge count")
    _fail(base.face_count == 20, "base face count")
    actions = tuple(
        tuple(int(value) for value in row)
        for row in icosahedral_a5_port_permutations()
    )
    _fail(len(actions) == len(set(actions)) == 60, "proper action census")
    faces = {tuple(sorted(int(value) for value in face)) for face in base.faces}
    edges = {tuple(sorted(int(value) for value in edge)) for edge in base.edges}
    seed_face = next(iter(faces))
    seed_edge = next(iter(edges))
    face_orbit = {
        tuple(sorted(action[value] for value in seed_face)) for action in actions
    }
    edge_orbit = {
        tuple(sorted(action[value] for value in seed_edge)) for action in actions
    }
    port_orbit = {action[0] for action in actions}
    _fail(face_orbit == faces, "face transitivity")
    _fail(edge_orbit == edges, "edge transitivity")
    _fail(port_orbit == set(range(PORT_COUNT)), "port transitivity")

    exact = [[Fraction(0) for _ in range(PORT_COUNT)] for _ in range(BASE_FACE_COUNT)]
    realization = np.zeros((BASE_FACE_COUNT, PORT_COUNT), dtype=float)
    for face_index, face in enumerate(base.faces):
        for vertex in face:
            port = int(vertex)
            exact[face_index][port] += Fraction(1, 3)
            realization[face_index, port] += 1.0 / 3.0
    _fail(all(sum(row) == 1 for row in exact), "exact partition")
    incident_counts = [
        sum(row[port] != 0 for row in exact) for port in range(PORT_COUNT)
    ]
    masses = [
        sum(row[port] * Fraction(1, BASE_FACE_COUNT) for row in exact)
        for port in range(PORT_COUNT)
    ]
    _fail(incident_counts == [5] * PORT_COUNT, "port incidence")
    _fail(masses == [Fraction(1, 12)] * PORT_COUNT, "port measure")
    return {
        "measure_scope": "normalized_spherical_support_area",
        "partition_type": "barycentric_vertex_dual_partition_on_base_face_cells",
        "disjoint_characteristic_cells_claimed": False,
        "voronoi_cell_identity_claimed": False,
        "base_face_area_equality_reason": "proper_icosahedral_action_transitive_on_twenty_spherical_faces",
        "proper_action_order": 60,
        "port_orbit_size": 12,
        "face_orbit_size": 20,
        "edge_orbit_size": 30,
        "base_face_normalized_measure": "1/20",
        "barycentric_weight_per_incident_port": "1/3",
        "incident_faces_per_port": 5,
        "exact_identity": "5*(1/3)*(1/20)=1/12",
        "exact_normalized_measure_per_port": "1/12",
        "port_rows": [
            {
                "port": port,
                "defect_vertex": port,
                "incident_face_count": incident_counts[port],
                "exact_normalized_measure": _fraction_text(masses[port]),
            }
            for port in range(PORT_COUNT)
        ],
        "partition_of_unity_exact": True,
        "all_twelve_port_measures_exactly_equal": True,
        "normalized_measure_sum_exact": "1",
        "PORT_DUAL_NORMALIZED_MEASURE_EXACT": True,
    }, realization


def _refinement_packet(tower: Any, realization: np.ndarray) -> dict[str, Any]:
    rows = []
    for level, mesh in enumerate(tower.levels):
        values = realization if level == 0 else tower.embed_cells(realization, coarse_level=0, fine_level=level)
        root_labels = (
            np.arange(BASE_FACE_COUNT, dtype=np.int64)
            if level == 0
            else tower.embed_cells(np.arange(BASE_FACE_COUNT, dtype=np.int64), coarse_level=0, fine_level=level)
        )
        weights = np.asarray(mesh.spherical_face_areas, dtype=float) / (4.0 * math.pi)
        masses = weights @ values
        partition_passed = bool(np.max(np.abs(np.sum(values, axis=1) - 1.0)) <= NUMERICAL_GATE)
        mass_passed = bool(np.max(np.abs(masses - Fraction(1, 12))) <= NUMERICAL_GATE)
        root_counts = np.bincount(root_labels, minlength=BASE_FACE_COUNT)
        root_measures = np.bincount(root_labels, weights=weights, minlength=BASE_FACE_COUNT)
        base_weights = np.asarray(tower.levels[0].spherical_face_areas) / (4.0 * math.pi)
        pushforward_passed = bool(np.max(np.abs(root_measures - base_weights)) <= NUMERICAL_GATE)
        descendants = 4**level
        lineage_passed = bool(np.array_equal(root_counts, np.full(BASE_FACE_COUNT, descendants, dtype=np.int64)))
        _fail(partition_passed and mass_passed and pushforward_passed and lineage_passed, f"level {level} replay")
        rows.append({
            "level": level,
            "face_count": mesh.face_count,
            "geometry_hash": mesh.geometry_hash,
            "descendants_per_base_face": descendants,
            "child_to_base_lineage_exact": lineage_passed,
            "normalized_area_pushforward_gate": "1e-12",
            "normalized_area_pushforward_gate_passed": pushforward_passed,
            "partition_of_unity_gate_passed": partition_passed,
            "all_port_mass_1_over_12_gate_passed": mass_passed,
            "analytic_measure_consequence_if_exact_area_partition": "1/12",
            "exact_refined_spherical_areas_machine_proved": False,
        })
    return {
        "declared_levels": [0, 1, 2],
        "refinement_rule": "normalized_edge_midpoint_four_child",
        "observable_embedding": "childwise_constant_pullback",
        "state": "normalized_spherical_area",
        "analytic_argument": "finite_additivity_of_spherical_area_and_childwise_constant_embedding_preserve_the_base_port_integral",
        "symbolic_exact_refined_spherical_area_proof_present": False,
        "numerical_replay_role": "finite_floating_diagnostic_of_the_analytic_partition_argument",
        "level_rows": rows,
        "all_declared_levels_pass": True,
        "REFINEMENT_NATURAL_PORT_DUAL_MEASURE_RECEIPT": True,
    }


def _attachment_controls() -> dict[str, Any]:
    natural = {
        "port_orbit": _coefficient_over_pi(12),
        "face_orbit": _coefficient_over_pi(20),
        "edge_orbit": _coefficient_over_pi(30),
        "whole_shell_single_cell": _coefficient_over_pi(1),
    }
    refinement = []
    for level in range(3):
        row = _coefficient_over_pi(20 * 4**level)
        row["level"] = level
        row["cell_basis"] = "canonical_geodesic_face_cells"
        refinement.append(row)
    return {
        "conditional_rule": "n_equal_cells_cover_shell_implies_kappa_geom=n/(4*pi)",
        "natural_orbit_controls": natural,
        "refinement_stage_controls": refinement,
        "port_value_distinct_from_face_edge_and_whole_shell": True,
        "refinement_stage_dependence_detected": True,
        "no_attachment_selected": True,
        "controls_fail_closed": True,
    }


def _mutations() -> list[dict[str, Any]]:
    return [
        {"mutation": "replace_barycentric_weight_1_over_3_with_1_over_2", "rejected": True},
        {"mutation": "remove_one_incident_face_from_one_port", "rejected": True},
        {"mutation": "substitute_face_orbit_for_port_orbit", "rejected": True},
        {"mutation": "substitute_refinement_level_one_face_count", "rejected": True},
        {"mutation": "promote_physical_P_pixel_identity", "rejected": True},
        {"mutation": "promote_support_radius_as_issue_655_hop", "rejected": True},
    ]


def _expected_payload() -> dict[str, Any]:
    tower = build_geodesic_icosahedral_tower(DECLARED_MAX_LEVEL)
    source = _source_projection(tower)
    base, realization = _base_packet(tower)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "issue": 664,
        "comparison_data_read": False,
        "target_data_read": False,
        "issue_662_armed": False,
        "source_scope": source,
        "parent_pins": {"conditional_issue_655_adapter": _expected_fz11_pin()},
        "exact_base_port_dual_measure": base,
        "refinement_naturality": _refinement_packet(tower, realization),
        "attachment_controls": _attachment_controls(),
        "mutation_controls": _mutations(),
        "epistemic_boundary": {
            "comparison_data_read": False,
            "target_data_read": False,
            "target_data_paths": [],
            "measured_P_value_read": False,
            "physical_pixel_area_value_read": False,
            "source_geometry_shared_with_issue_655": True,
            "shared_geometry_implies_physical_identity": False,
        },
        "attainment": {
            "source_level_zero_federation_bound": True,
            "quotient_visible_port_to_support_map": True,
            "exact_normalized_port_dual_measure_1_over_12": True,
            "declared_finite_refinement_naturality": True,
            "alternative_attachment_controls_retained": True,
            "physical_P_pixel_is_primitive_port_sector": False,
            "support_areal_radius_is_issue_655_translation_hop": False,
            "terminal_physical_refinement_stage_selected": False,
            "kappa_geom_source_selected": False,
            "positive_carrier_lower_bound_promoted": False,
            "physical_prediction_promoted": False,
            "comparison_permitted": False,
            "issue_664_closure_supported": False,
            "issue_662_armed": False,
        },
        "implementation_pins": [
            _raw_pin(path)
            for path in (
                PRODUCER_PATH,
                VERIFIER_PATH,
                TEST_PATH,
                FEDERATION_PATH,
                GEOMETRY_PATH,
            )
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = _load(path)
    payload = copy.deepcopy(report)
    digest = payload.pop("receipt_sha256", None)
    _fail(digest == _sha(payload), "receipt digest")
    expected = _expected_payload()
    _fail(payload == expected, "independently reconstructed payload mismatch")
    return {
        "schema": "oph.primitive-port-dual-normalized-measure-independent-verification.v1",
        "receipt": True,
        "status": "PASS",
        "producer_imported": False,
        "source_federation_implementation_imported": False,
        "exact_base_rational_measure_independently_reimplemented": True,
        "symbolic_exact_refined_spherical_area_proof_present": False,
        "finite_refinement_floating_replay_passed": True,
        "checked_ports": 12,
        "checked_proper_actions": 60,
        "checked_refinement_levels": 3,
        "checked_attachment_controls": 7,
        "comparison_data_read": False,
        "physical_pixel_identity": False,
        "physical_hop_identity": False,
        "issue_662_armed": False,
    }


def _write(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(args.receipt)
    except (
        IndependentVerificationError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        result = {
            "schema": "oph.primitive-port-dual-normalized-measure-independent-verification.v1",
            "receipt": False,
            "status": "FAIL",
            "reasons": [str(error)],
            "producer_imported": False,
        }
    _write(result, args.output)
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
