"""Supplied support/regulator geometry for the legacy large-array engines.

This module does not construct the microscopic carrier federation.  The v2
physical instrument keeps exact-count echosahedral carriers and any spherical
support chart as separately typed objects; identifying a chart cell with a
carrier requires a realization theorem that this adapter does not provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    geodesic_icosahedral_patch_arrays,
    icosahedral_a5_equivariance_report,
    icosahedral_count_bracket,
    icosahedral_defect_port_report,
    resolve_icosahedral_level,
    supported_icosahedral_count,
)


ICOSAHEDRAL_ARRAY_FAMILIES = frozenset(
    {
        "subdivided_icosahedral_screen",
        "nested_geodesic_icosahedral",
        "geodesic_icosahedral_refinement",
        "nested_icosahedral_cap_net",
    }
)


@dataclass(frozen=True)
class ArrayScreenGeometry:
    points: np.ndarray
    edge_left: np.ndarray
    edge_right: np.ndarray
    report: dict[str, Any]

    @property
    def patch_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_left.size)


def array_screen_geometry_from_config(
    graph_config: Mapping[str, Any] | None,
    *,
    knn_builder: Any,
) -> ArrayScreenGeometry:
    """Instantiate a strict icosahedral support chart or a legacy control.

    Existing array engines still place one local state row on each returned
    chart point.  Reports expose that compatibility shortcut and never promote
    it as a valid carrier-to-support realization.
    """

    graph = dict(graph_config or {})
    family = str(graph.get("family", "fibonacci_sphere"))
    requested = int(graph.get("patch_count", graph.get("nodes", 65_536)))
    if family not in ICOSAHEDRAL_ARRAY_FAMILIES:
        neighbors = int(graph.get("neighbors", graph.get("degree", 8)))
        points = fibonacci_sphere_points(requested)
        left, right = knn_builder(points, neighbors)
        report = {
            "schema": "oph.array_screen_geometry.v1",
            "mesh_family": family,
            "geometry_family": "legacy_fibonacci_knn_control",
            "nominal_patch_count": requested,
            "actual_patch_count": int(points.shape[0]),
            "edge_count": int(left.size),
            "patch_basis": "points",
            "TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT": False,
            "NESTED_ICOSAHEDRAL_LINEAGE_RECEIPT": False,
            "A5_EQUIVARIANT_REFINEMENT_RECEIPT": False,
            "PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE": False,
            "CARRIER_TO_SUPPORT_REALIZATION_RECEIPT": False,
            "CARRIER_SUPPORT_CONFLATION_PRESENT_IN_LEGACY_ENGINE": True,
            "claim_boundary": (
                "Legacy Fibonacci/KNN global cellulation retained only as a control. "
                "The legacy engine may attach local state rows to these points, but "
                "that compatibility shortcut is not a carrier federation or an "
                "earned support chart."
            ),
        }
        return ArrayScreenGeometry(points, left, right, report)

    patch_basis = str(graph.get("patch_basis", graph.get("node_basis", "cells")))
    if patch_basis not in {"cells", "vertices"}:
        raise ValueError("icosahedral array patch_basis must be 'cells' or 'vertices'")
    nominal_raw = graph.get("nominal_patch_count")
    nominal_count = int(nominal_raw) if nominal_raw is not None else None
    if "refinement_level" in graph:
        level = int(graph["refinement_level"])
        actual_count = supported_icosahedral_count(level, patch_basis)
        if "patch_count" in graph and int(graph["patch_count"]) != actual_count:
            raise ValueError(
                f"refinement_level={level} on {patch_basis} has {actual_count} "
                f"patches, not patch_count={int(graph['patch_count'])}; use "
                "nominal_patch_count for the campaign rung label"
            )
    else:
        policy = str(graph.get("patch_count_policy", "exact"))
        level, bracket = resolve_icosahedral_level(
            requested,
            patch_basis=patch_basis,
            policy=policy,
        )
        actual_count = supported_icosahedral_count(level, patch_basis)
        if not bracket["exact_supported_count"] and nominal_count is None:
            nominal_count = requested
    if nominal_count is None and "patch_count" not in graph:
        nominal_count = actual_count
    points, left, right = geodesic_icosahedral_patch_arrays(
        level,
        patch_basis=patch_basis,
    )
    tower = build_geodesic_icosahedral_tower(level)
    tower_receipt = tower.receipt()
    a5_receipt = icosahedral_a5_equivariance_report(level)
    defect_ports = icosahedral_defect_port_report(level)
    levels: list[dict[str, Any]] = []
    expectations: list[dict[str, Any]] = []
    for item in tower.levels:
        levels.append(
            {
                "level_id": f"ico-L{item.level}",
                "refinement_level": item.level,
                "patch_basis": patch_basis,
                "patch_count": (
                    item.face_count if patch_basis == "cells" else item.vertex_count
                ),
                "vertex_count": item.vertex_count,
                "cell_count": item.face_count,
                "geometry_hash": item.geometry_hash,
                **(
                    {}
                    if item.level == 0
                    else {
                        "parent_level_id": f"ico-L{item.level - 1}",
                        "lineage_hash": tower.cell_refinements[
                            item.level - 1
                        ].map_hash,
                    }
                ),
            }
        )
    for mapping in tower.cell_refinements:
        expectations.append(
            {
                "fine_level_id": f"ico-L{mapping.fine_level}",
                "coarse_level_id": f"ico-L{mapping.coarse_level}",
                "operator_hash": mapping.map_hash,
                "scope": "commutative_cell_geometry_scaffold",
                "unital": True,
                "positive": True,
                "state_preserving": mapping.state_preservation_residual <= 5.0e-14,
                "cap_isotony_compatible": True,
                "noncommutative_prime_cap_expectation": False,
            }
        )
    report = {
        "schema": "oph.array_screen_geometry.v1",
        "mesh_family": "nested_geodesic_icosahedral",
        "geometry_family": "nested_geodesic_icosahedral",
        "patch_basis": patch_basis,
        "refinement_level": level,
        "nominal_patch_count": nominal_count,
        "actual_patch_count": int(points.shape[0]),
        "edge_count": int(left.size),
        "nominal_count_mapping": (
            icosahedral_count_bracket(nominal_count, patch_basis)
            if nominal_count is not None
            else None
        ),
        "levels": levels,
        "conditional_expectations": expectations,
        "nested_lineage_receipt": tower_receipt[
            "GEODESIC_ICOSAHEDRAL_TOWER_RECEIPT"
        ],
        "conditional_expectations_receipt": bool(
            expectations and all(row["state_preserving"] for row in expectations)
        ),
        "TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT": tower_receipt[
            "GEODESIC_ICOSAHEDRAL_TOWER_RECEIPT"
        ],
        "NESTED_ICOSAHEDRAL_LINEAGE_RECEIPT": tower_receipt[
            "GEODESIC_ICOSAHEDRAL_TOWER_RECEIPT"
        ],
        "A5_EQUIVARIANT_REFINEMENT_RECEIPT": a5_receipt[
            "A5_EQUIVARIANT_REFINEMENT_RECEIPT"
        ],
        "REFERENCE_ICOSAHEDRAL_A5_GEOMETRY_RECEIPT": a5_receipt[
            "REFERENCE_ICOSAHEDRAL_A5_GEOMETRY_RECEIPT"
        ],
        "TWELVE_PERSISTENT_COMBINATORIAL_DEFECT_PORTS_RECEIPT": defect_ports[
            "TWELVE_PERSISTENT_COMBINATORIAL_DEFECT_PORTS_RECEIPT"
        ],
        "PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE": False,
        "CARRIER_TO_SUPPORT_REALIZATION_RECEIPT": False,
        "CARRIER_SUPPORT_CONFLATION_PRESENT_IN_LEGACY_ENGINE": True,
        "tower": tower_receipt,
        "a5_action": a5_receipt,
        "global_screen_sieve_defect_ports": defect_ports,
        "full_multiresolution_blockers": [
            "finite_detail_algebras_not_instantiated",
            "faithful_prime_cap_states_not_instantiated",
            "local_presentation_circuits_not_instantiated",
            "noncommutative_state_preserving_expectations_not_instantiated",
            "mixed_gns_reference_tower_not_instantiated",
        ],
        "claim_boundary": (
            "Reference global nested icosahedral support chart plus exact "
            "integer-permutation A5 equivariance and the commutative cell-area "
            "expectation scaffold. It is separate from the exact-count microscopic "
            "carrier federation: no global cell is identified with one carrier by "
            "this report. The carrier-to-support realization and full "
            "noncommutative multiresolution/BW certificates remain false."
        ),
    }
    return ArrayScreenGeometry(points, left, right, report)
