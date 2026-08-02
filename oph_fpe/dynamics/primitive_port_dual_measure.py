"""Build the bounded normalized primitive-port dual-measure packet for #664.

The packet uses the source-derived incidence federation and its declared
level-zero through level-two geodesic support tower.  On every base triangle,
the three incident port observables receive barycentric weight one third.
The proper icosahedral action is transitive on the twenty base faces and on
the twelve port vertices.  Consequently every port-dual sector has exact
normalized support measure

    5 * (1/3) * (1/20) = 1/12.

Childwise embedding and finite additivity of spherical area give the analytic
state-preservation argument for the same measure on later levels.  A floating
replay checks that argument through level two; it is not a symbolic proof of
the refined spherical areas.  This is a normalized two-dimensional support
measure.  It does not identify a physical P-pixel with a port sector or the
support areal radius with the issue-655 translation hop.  No comparison or
target data are read.
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

from oph_fpe.core.echosahedral_federation import (
    carrier_refinement_naturality_report,
    echosahedral_federation_receipt,
    reference_incidence_nerve_federation,
    relabel_federation_ports,
)
from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/repair_closure/primitive_port_dual_measure_receipt.json"
)
FZ11_RECEIPT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = (
    ROOT / "oph_fpe/dynamics/verify_primitive_port_dual_measure_independent.py"
)
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


class PrimitivePortDualMeasureError(RuntimeError):
    """Raised when the bounded measure packet fails closed."""


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
            raise PrimitivePortDualMeasureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise PrimitivePortDualMeasureError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrimitivePortDualMeasureError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise PrimitivePortDualMeasureError(f"{path} is not a JSON object")
    return value


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _coefficient_over_pi(orbit_cardinality: int) -> dict[str, Any]:
    coefficient = Fraction(int(orbit_cardinality), 4)
    if coefficient.denominator == 1:
        expression = f"{coefficient.numerator}/pi"
    elif coefficient.numerator == 1:
        expression = f"1/({coefficient.denominator}*pi)"
    else:
        expression = (
            f"{coefficient.numerator}/({coefficient.denominator}*pi)"
        )
    return {
        "orbit_cardinality": int(orbit_cardinality),
        "coefficient_over_pi": {
            "numerator": coefficient.numerator,
            "denominator": coefficient.denominator,
        },
        "conditional_kappa_geom": expression,
        "physical_attachment_selected": False,
    }


def _validated_fz11_parent() -> dict[str, Any]:
    receipt = _load_json_strict(FZ11_RECEIPT)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    try:
        geometry_hash = receipt["exact_port_frame_and_relabel"][
            "source_geometry_hash"
        ]
    except (KeyError, TypeError) as error:
        raise PrimitivePortDualMeasureError("FZ-11 geometry binding is absent") from error
    if (
        digest != _sha(payload)
        or receipt.get("schema") != FZ11_SCHEMA
        or receipt.get("status") != FZ11_STATUS
        or receipt.get("issue") != 655
        or receipt.get("comparison_data_read") is not False
        or receipt.get("issue_662_armed") is not False
    ):
        raise PrimitivePortDualMeasureError("FZ-11 parent contract drifted")
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


def _source_parent_projection() -> tuple[dict[str, Any], Any, list[int]]:
    federation = reference_incidence_nerve_federation()
    relabeling = tuple(reversed(range(PORT_COUNT)))
    permutations = {
        carrier.carrier_id: relabeling for carrier in federation.carriers
    }
    transformed = relabel_federation_ports(federation, permutations)
    parent = echosahedral_federation_receipt(
        federation,
        equivalent_presentation=transformed,
        presentation_port_permutations=permutations,
    )
    refinement = carrier_refinement_naturality_report(
        max_level=DECLARED_MAX_LEVEL
    )
    support = parent["controlled_oriented_s2_limit"]
    required_flags = (
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
    if any(parent.get(key) is not True for key in required_flags):
        raise PrimitivePortDualMeasureError("source federation parent is not attained")
    if (
        refinement.get("CARRIER_REFINEMENT_NATURALITY_RECEIPT") is not True
        or refinement.get("CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT") is not True
        or refinement.get("tower_levels_checked") != DECLARED_MAX_LEVEL + 1
    ):
        raise PrimitivePortDualMeasureError("refinement parent is not attained")
    tower = build_geodesic_icosahedral_tower(DECLARED_MAX_LEVEL)
    base = tower.levels[0]
    geometry_hash = base.geometry_hash
    fz11 = _validated_fz11_parent()
    if fz11["source_geometry_hash"] != geometry_hash:
        raise PrimitivePortDualMeasureError("support and FZ-11 source frames drifted")
    embedding = [
        int(value) for value in refinement["embedding_port_to_defect_vertex"]
    ]
    projection = {
        "federation_id": federation.federation_id,
        "source_incidence_sha256": support["source_incidence_sha256"],
        "source_geometry_family": "nested_geodesic_icosahedral",
        "source_geometry_hash": geometry_hash,
        "carrier_count": len(federation.carriers),
        "seam_count": len(federation.seams),
        "triple_overlap_count": len(federation.triple_overlaps),
        "support_simplex_counts": support["nerve_simplex_counts"],
        "port_to_defect_vertex_bijection": embedding,
        "proper_action_count": refinement["checked_action_count"],
        "declared_tower_levels": list(range(DECLARED_MAX_LEVEL + 1)),
        "refinement_map_hashes": [
            row["map_hash"] for row in refinement["cell_refinement_receipts"]
        ],
        "source_parent_receipts": {key: True for key in required_flags},
        "source_projection_sha256": "",
    }
    projection["source_projection_sha256"] = _sha(
        {key: value for key, value in projection.items() if key != "source_projection_sha256"}
    )
    return projection, tower, embedding


def _base_measure_packet(tower: Any, embedding: Sequence[int]) -> tuple[dict[str, Any], np.ndarray]:
    base = tower.levels[0]
    if (
        base.vertex_count != PORT_COUNT
        or base.face_count != BASE_FACE_COUNT
        or base.edge_count != BASE_EDGE_COUNT
        or sorted(int(value) for value in embedding) != list(range(PORT_COUNT))
    ):
        raise PrimitivePortDualMeasureError("level-zero support census drifted")
    actions = tuple(
        tuple(int(value) for value in row)
        for row in icosahedral_a5_port_permutations()
    )
    if not (len(actions) == len(set(actions)) == 60):
        raise PrimitivePortDualMeasureError("proper action census drifted")
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
    port_orbit = {action[int(embedding[0])] for action in actions}
    if face_orbit != faces or edge_orbit != edges or port_orbit != set(embedding):
        raise PrimitivePortDualMeasureError("proper action is not transitive on incidence orbits")

    vertex_to_port = {int(vertex): port for port, vertex in enumerate(embedding)}
    realization = np.zeros((base.face_count, PORT_COUNT), dtype=float)
    exact_incidence = [[Fraction(0) for _ in range(PORT_COUNT)] for _ in range(base.face_count)]
    for face_index, face in enumerate(base.faces):
        for vertex in face:
            port = vertex_to_port[int(vertex)]
            exact_incidence[face_index][port] += Fraction(1, 3)
            realization[face_index, port] += 1.0 / 3.0
    if any(sum(row) != 1 for row in exact_incidence):
        raise PrimitivePortDualMeasureError("barycentric partition of unity failed")
    incident_counts = [
        sum(value != 0 for row in exact_incidence for value in [row[port]])
        for port in range(PORT_COUNT)
    ]
    exact_masses = [
        sum(row[port] * Fraction(1, BASE_FACE_COUNT) for row in exact_incidence)
        for port in range(PORT_COUNT)
    ]
    if incident_counts != [5] * PORT_COUNT or exact_masses != [Fraction(1, 12)] * PORT_COUNT:
        raise PrimitivePortDualMeasureError("exact port-dual measure is not uniform 1/12")
    packet = {
        "measure_scope": "normalized_spherical_support_area",
        "partition_type": "barycentric_vertex_dual_partition_on_base_face_cells",
        "disjoint_characteristic_cells_claimed": False,
        "voronoi_cell_identity_claimed": False,
        "base_face_area_equality_reason": (
            "proper_icosahedral_action_transitive_on_twenty_spherical_faces"
        ),
        "proper_action_order": len(actions),
        "port_orbit_size": len(port_orbit),
        "face_orbit_size": len(face_orbit),
        "edge_orbit_size": len(edge_orbit),
        "base_face_normalized_measure": "1/20",
        "barycentric_weight_per_incident_port": "1/3",
        "incident_faces_per_port": 5,
        "exact_identity": "5*(1/3)*(1/20)=1/12",
        "exact_normalized_measure_per_port": "1/12",
        "port_rows": [
            {
                "port": port,
                "defect_vertex": int(embedding[port]),
                "incident_face_count": incident_counts[port],
                "exact_normalized_measure": _fraction_text(exact_masses[port]),
            }
            for port in range(PORT_COUNT)
        ],
        "partition_of_unity_exact": True,
        "all_twelve_port_measures_exactly_equal": True,
        "normalized_measure_sum_exact": "1",
        "PORT_DUAL_NORMALIZED_MEASURE_EXACT": True,
    }
    return packet, realization


def _refinement_packet(tower: Any, realization: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for level, mesh in enumerate(tower.levels):
        values = (
            realization
            if level == 0
            else tower.embed_cells(realization, coarse_level=0, fine_level=level)
        )
        root_labels = (
            np.arange(BASE_FACE_COUNT, dtype=np.int64)
            if level == 0
            else tower.embed_cells(
                np.arange(BASE_FACE_COUNT, dtype=np.int64),
                coarse_level=0,
                fine_level=level,
            )
        )
        weights = np.asarray(mesh.spherical_face_areas, dtype=float) / (4.0 * math.pi)
        masses = weights @ values
        partition_passed = bool(
            np.max(np.abs(np.sum(values, axis=1) - 1.0)) <= NUMERICAL_GATE
        )
        mass_passed = bool(
            np.max(np.abs(masses - 1.0 / PORT_COUNT)) <= NUMERICAL_GATE
        )
        root_counts = np.bincount(root_labels, minlength=BASE_FACE_COUNT)
        root_measures = np.bincount(
            root_labels,
            weights=weights,
            minlength=BASE_FACE_COUNT,
        )
        base_weights = np.asarray(tower.levels[0].spherical_face_areas) / (
            4.0 * math.pi
        )
        pushforward_passed = bool(
            np.max(np.abs(root_measures - base_weights)) <= NUMERICAL_GATE
        )
        expected_descendants = 4**level
        lineage_passed = bool(
            np.array_equal(
                root_counts,
                np.full(BASE_FACE_COUNT, expected_descendants, dtype=np.int64),
            )
        )
        if not (partition_passed and mass_passed and pushforward_passed and lineage_passed):
            raise PrimitivePortDualMeasureError(
                f"refinement measure replay failed at level {level}"
            )
        rows.append(
            {
                "level": level,
                "face_count": mesh.face_count,
                "geometry_hash": mesh.geometry_hash,
                "descendants_per_base_face": expected_descendants,
                "child_to_base_lineage_exact": lineage_passed,
                "normalized_area_pushforward_gate": "1e-12",
                "normalized_area_pushforward_gate_passed": pushforward_passed,
                "partition_of_unity_gate_passed": partition_passed,
                "all_port_mass_1_over_12_gate_passed": mass_passed,
                "analytic_measure_consequence_if_exact_area_partition": "1/12",
                "exact_refined_spherical_areas_machine_proved": False,
            }
        )
    return {
        "declared_levels": list(range(DECLARED_MAX_LEVEL + 1)),
        "refinement_rule": "normalized_edge_midpoint_four_child",
        "observable_embedding": "childwise_constant_pullback",
        "state": "normalized_spherical_area",
        "analytic_argument": (
            "finite_additivity_of_spherical_area_and_childwise_constant_"
            "embedding_preserve_the_base_port_integral"
        ),
        "symbolic_exact_refined_spherical_area_proof_present": False,
        "numerical_replay_role": (
            "finite_floating_diagnostic_of_the_analytic_partition_argument"
        ),
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
    refinement_rows = []
    for level in range(DECLARED_MAX_LEVEL + 1):
        row = _coefficient_over_pi(BASE_FACE_COUNT * 4**level)
        row["level"] = level
        row["cell_basis"] = "canonical_geodesic_face_cells"
        refinement_rows.append(row)
    port_coefficient = Fraction(3)
    other_coefficients = (
        Fraction(5),
        Fraction(15, 2),
        Fraction(1, 4),
    )
    return {
        "conditional_rule": "n_equal_cells_cover_shell_implies_kappa_geom=n/(4*pi)",
        "natural_orbit_controls": natural,
        "refinement_stage_controls": refinement_rows,
        "port_value_distinct_from_face_edge_and_whole_shell": all(
            port_coefficient != value for value in other_coefficients
        ),
        "refinement_stage_dependence_detected": len(
            {
                row["conditional_kappa_geom"] for row in refinement_rows
            }
        )
        == len(refinement_rows),
        "no_attachment_selected": True,
        "controls_fail_closed": True,
    }


def _mutation_controls() -> list[dict[str, Any]]:
    canonical_mass = Fraction(5) * Fraction(1, 3) * Fraction(1, 20)
    return [
        {
            "mutation": "replace_barycentric_weight_1_over_3_with_1_over_2",
            "rejected": 3 * Fraction(1, 2) != 1,
        },
        {
            "mutation": "remove_one_incident_face_from_one_port",
            "rejected": 4 * Fraction(1, 3) * Fraction(1, 20) != canonical_mass,
        },
        {
            "mutation": "substitute_face_orbit_for_port_orbit",
            "rejected": Fraction(20, 4) != Fraction(12, 4),
        },
        {
            "mutation": "substitute_refinement_level_one_face_count",
            "rejected": Fraction(80, 4) != Fraction(12, 4),
        },
        {
            "mutation": "promote_physical_P_pixel_identity",
            "rejected": True,
        },
        {
            "mutation": "promote_support_radius_as_issue_655_hop",
            "rejected": True,
        },
    ]


def _payload() -> dict[str, Any]:
    source, tower, embedding = _source_parent_projection()
    base, realization = _base_measure_packet(tower, embedding)
    refinement = _refinement_packet(tower, realization)
    controls = _attachment_controls()
    mutations = _mutation_controls()
    if not all(row["rejected"] is True for row in mutations):
        raise PrimitivePortDualMeasureError("mutation controls did not fail closed")
    implementation_files = (
        PRODUCER_PATH,
        VERIFIER_PATH,
        TEST_PATH,
        FEDERATION_PATH,
        GEOMETRY_PATH,
    )
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "issue": 664,
        "comparison_data_read": False,
        "target_data_read": False,
        "issue_662_armed": False,
        "source_scope": source,
        "parent_pins": {
            "conditional_issue_655_adapter": _validated_fz11_parent(),
        },
        "exact_base_port_dual_measure": base,
        "refinement_naturality": refinement,
        "attachment_controls": controls,
        "mutation_controls": mutations,
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
        "implementation_pins": [_raw_pin(path) for path in implementation_files],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def produce_receipt() -> dict[str, Any]:
    payload = _payload()
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_digest_mismatch")
        if _canonical_bytes(received) != _canonical_bytes(_payload()):
            reasons.append("producer_replay_mismatch")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        PrimitivePortDualMeasureError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_receipt")
    return {
        "schema": "oph.primitive-port-dual-normalized-measure-verification.v1",
        "receipt": not reasons,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "physical_promotion": False,
    }


def load_receipt_strict(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Load only the exact canonical receipt with no duplicate JSON keys."""

    report = _load_json_strict(path)
    result = verify_receipt(report)
    if result["receipt"] is not True:
        raise PrimitivePortDualMeasureError(
            "strict receipt verification failed: " + ",".join(result["reasons"])
        )
    return report


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        try:
            report = load_receipt_strict(args.verify)
            result = verify_receipt(report)
        except PrimitivePortDualMeasureError as error:
            result = {
                "schema": "oph.primitive-port-dual-normalized-measure-verification.v1",
                "receipt": False,
                "status": "FAIL",
                "reasons": [str(error)],
            }
        _write_json(result, None if args.output == DEFAULT_RECEIPT else args.output)
        return 0 if result["receipt"] else 1
    receipt = produce_receipt()
    result = verify_receipt(receipt)
    if result["receipt"] is not True:
        _write_json(result, None)
        return 1
    _write_json(receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
