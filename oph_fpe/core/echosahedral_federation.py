"""Two-level echosahedral carrier/federation conformance contracts.

The microscopic carrier and the inter-carrier federation are deliberately
different objects.  Each carrier has a hidden regular-icosahedron
presentation with twelve ports; seams connect typed, connected subsets of
those local ports.  This module validates that finite implementation surface.

It does *not* identify one carrier with an observer, screen cap, S2 point, H3
point, event, or BW/KMS source.  Those remain downstream, independently gated
outputs.  Existing screen-port reports can be checked through a fail-closed
reference bridge, and a compact JSON instrument-bundle API is provided, but
neither path promotes the current engine to a physical federation source.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import numpy as np

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)
from oph_fpe.core.screen_ports import (
    echosahedral_patch_architecture_report,
    echosahedral_port_names,
)


CollarKind = Literal[
    "single_port",
    "antipodal_pair",
    "edge_bundle",
    "face_collar",
    "connected_bundle",
]
BoundaryCondition = Literal[
    "open_external",
    "fixed_external",
    "measured_external",
]

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PORT_COUNT = 12
_REFERENCE_CARRIER_TEMPLATE_ID = "__shared_regular_icosahedron_template__"
_REPORT_DETAIL_LIMIT = 64
_CARDINALITY_SEMANTICS = (
    "exact_declared_finite_source_carrier_cardinality_separate_from_support_regulator"
)
_INTERFACE_HASH_CHECK_SCOPE = (
    "content_addressed_schema_identity_only_no_algebra_map_or_higher_overlap_proof"
)

HIDDEN_PRESENTATION_FIELDS = frozenset(
    {
        "port_coordinates",
        "hidden_port_coordinates",
        "port_names",
        "local_port_names",
        "local_a5_frame",
        "a5_actions",
        "worker_id",
        "worker_assignment",
        "shard_id",
        "shard_assignment",
        "queue_order",
        "queue_position",
        "memory_layout",
        "repair_iteration",
        "repair_depth",
        "retry_count",
        "candidate_kappa",
        "candidate_clock_scale",
        "target_h3_label",
        "local_outward_normal",
    }
)

CARRIER_FORBIDDEN_PROMOTION_FIELDS = HIDDEN_PRESENTATION_FIELDS | frozenset(
    {
        "s2_point",
        "s2_cap",
        "h3_point",
        "event_position",
        "event_id_from_carrier_id",
        "bw_source_from_carrier_coordinates",
        "cap_normal_from_local_carrier",
    }
)

QUOTIENT_VISIBLE_CARRIER_FIELDS = frozenset(
    {
        "carrier_id",
        "port_response",
        "seam_interface_packet",
        "semantic_record",
        "checkpoint_continuation",
        "repair_normal_form",
        "interface_algebra_sha256",
    }
)


@dataclass(frozen=True)
class EchosahedralCarrier:
    """One finite twelve-port hidden carrier presentation."""

    carrier_id: str
    port_names: tuple[str, ...]
    port_coordinates: tuple[tuple[float, float, float], ...]
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, int, int], ...]
    antipode: tuple[int, ...]
    a5_actions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class InterfaceAlgebraBinding:
    """Hashes binding the same overlap-visible algebra at both endpoints."""

    interface_algebra_id: str
    interface_algebra_sha256: str
    left_interface_algebra_sha256: str
    right_interface_algebra_sha256: str


@dataclass(frozen=True)
class SeamBundle:
    """A typed connected collar with an explicit orientation-reversing bijection."""

    seam_id: str
    left_carrier_id: str
    right_carrier_id: str
    left_ports: tuple[int, ...]
    right_ports: tuple[int, ...]
    left_to_right_ports: tuple[int, ...]
    right_to_left_ports: tuple[int, ...]
    left_to_right_orientation: tuple[int, ...]
    right_to_left_orientation: tuple[int, ...]
    collar_kind: CollarKind
    interface_algebra: InterfaceAlgebraBinding


@dataclass(frozen=True)
class ExternalBoundaryBundle:
    """An explicit declaration for ports not sewn to another carrier."""

    boundary_id: str
    carrier_id: str
    ports: tuple[int, ...]
    boundary_condition: BoundaryCondition
    boundary_algebra_sha256: str


@dataclass(frozen=True)
class ObserverSupport:
    """One observer token supported on a connected carrier subfederation."""

    observer_token: str
    carrier_ids: frozenset[str]
    visible_seam_ids: frozenset[str]
    record_algebra_sha256: str
    checkpoint_cut_sha256: str


@dataclass(frozen=True)
class EchosahedralFederation:
    """Carrier set, typed seams, external boundary, and observer supports."""

    federation_id: str
    carriers: tuple[EchosahedralCarrier, ...]
    seams: tuple[SeamBundle, ...]
    external_boundaries: tuple[ExternalBoundaryBundle, ...]
    observer_supports: tuple[ObserverSupport, ...] = ()


def interface_algebra_sha256(schema: Mapping[str, Any] | Sequence[Any] | str) -> str:
    """Hash a JSON-serializable interface-algebra schema."""

    encoded = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _reference_echosahedral_carrier_template() -> EchosahedralCarrier:
    """Build the immutable 12-port template once for all reference carriers."""

    base = build_geodesic_icosahedral_tower(0).levels[0]
    coordinates = np.asarray(base.vertices, dtype=float)
    antipode = tuple(
        int(np.argmin(np.linalg.norm(coordinates + coordinate, axis=1)))
        for coordinate in coordinates
    )
    return EchosahedralCarrier(
        carrier_id=_REFERENCE_CARRIER_TEMPLATE_ID,
        port_names=tuple(echosahedral_port_names(12)),
        port_coordinates=tuple(
            tuple(float(value) for value in row) for row in coordinates
        ),
        edges=tuple(tuple(int(value) for value in row) for row in base.edges),
        faces=tuple(tuple(int(value) for value in row) for row in base.faces),
        antipode=antipode,
        a5_actions=icosahedral_a5_port_permutations(),
    )


def reference_echosahedral_carrier(carrier_id: str) -> EchosahedralCarrier:
    """Instantiate one ID over the shared immutable 12-port template.

    Every tuple-valued presentation field is shared with the single cached
    reference template.  Only ``carrier_id`` is replaced, so a federation of
    canonical carriers does not rebuild or duplicate the 12/30/20/A5 data.
    """

    if not isinstance(carrier_id, str) or not carrier_id:
        raise ValueError("carrier_id must be a nonempty string")
    return replace(_reference_echosahedral_carrier_template(), carrier_id=carrier_id)


def echosahedral_carrier_conformance_report(
    carrier: EchosahedralCarrier,
    *,
    tolerance: float = 5.0e-11,
) -> dict[str, Any]:
    """Return an isolated copy of an ID-independent cached local audit.

    Carrier identity is not part of local icosahedral conformance.  Normalizing
    the ID before cache lookup makes 256k canonical carriers share one exact
    local audit instead of populating a cache with ID-distinguished copies.
    """

    normalized = replace(carrier, carrier_id=_REFERENCE_CARRIER_TEMPLATE_ID)
    report = copy.deepcopy(
        _echosahedral_carrier_conformance_cached(normalized, tolerance)
    )
    identifier_valid = bool(
        isinstance(carrier.carrier_id, str) and carrier.carrier_id
    )
    report["carrier_id"] = carrier.carrier_id
    report["carrier_identifier_valid"] = identifier_valid
    if not identifier_valid:
        report["blockers"] = sorted(
            set(report["blockers"]) | {"carrier_id_must_be_nonempty_string"}
        )
        report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"] = False
        report["ECHOSAHEDRAL_CARRIER_CONFORMANCE_RECEIPT"] = False
    return report


@lru_cache(maxsize=512)
def _echosahedral_carrier_conformance_cached(
    carrier: EchosahedralCarrier,
    tolerance: float = 5.0e-11,
) -> dict[str, Any]:
    """Recompute complete local 12/30/20, antipode, and A5 conformance."""

    blockers: list[str] = []
    coordinates = np.asarray(carrier.port_coordinates, dtype=float)
    edges = tuple(tuple(int(value) for value in edge) for edge in carrier.edges)
    faces = tuple(tuple(int(value) for value in face) for face in carrier.faces)
    antipode = tuple(int(value) for value in carrier.antipode)
    actions = tuple(
        tuple(int(value) for value in action) for action in carrier.a5_actions
    )
    if (
        len(carrier.port_names) != _PORT_COUNT
        or len(set(carrier.port_names)) != _PORT_COUNT
    ):
        blockers.append("port_names_are_not_twelve_unique_hidden_labels")
    if coordinates.shape != (_PORT_COUNT, 3) or not np.all(np.isfinite(coordinates)):
        blockers.append("hidden_coordinates_are_not_finite_12_by_3")
        coordinates = np.zeros((_PORT_COUNT, 3), dtype=float)
    coordinate_norm_residual = float(
        np.max(np.abs(np.linalg.norm(coordinates, axis=1) - 1.0))
    )
    if coordinate_norm_residual > tolerance:
        blockers.append("port_coordinates_are_not_on_unit_reference_sphere")

    edge_set: set[tuple[int, int]] = set()
    edge_shape_valid = len(edges) == 30
    for edge in edges:
        if (
            len(edge) != 2
            or edge[0] == edge[1]
            or min(edge) < 0
            or max(edge) >= _PORT_COUNT
        ):
            edge_shape_valid = False
            continue
        edge_set.add(tuple(sorted(edge)))
    edge_shape_valid = edge_shape_valid and len(edge_set) == 30
    if not edge_shape_valid:
        blockers.append("edge_incidence_is_not_30_distinct_simple_edges")
    degrees = np.zeros(_PORT_COUNT, dtype=int)
    for left, right in edge_set:
        degrees[left] += 1
        degrees[right] += 1
    degree_five = bool(np.all(degrees == 5))
    if not degree_five:
        blockers.append("not_every_port_has_local_degree_five")

    face_shape_valid = len(faces) == 20
    oriented_face_set: set[tuple[int, int, int]] = set()
    unoriented_face_set: set[tuple[int, int, int]] = set()
    outward_residual = math.inf
    outward_values: list[float] = []
    for face in faces:
        if (
            len(face) != 3
            or len(set(face)) != 3
            or min(face) < 0
            or max(face) >= _PORT_COUNT
        ):
            face_shape_valid = False
            continue
        cyclic = _cyclic_face_key(face)
        unoriented = tuple(sorted(face))
        oriented_face_set.add(cyclic)
        unoriented_face_set.add(unoriented)
        a, b, c = face
        outward_values.append(
            float(
                np.dot(
                    np.cross(
                        coordinates[b] - coordinates[a], coordinates[c] - coordinates[a]
                    ),
                    coordinates[a] + coordinates[b] + coordinates[c],
                )
            )
        )
    face_shape_valid = bool(
        face_shape_valid
        and len(oriented_face_set) == 20
        and len(unoriented_face_set) == 20
    )
    if outward_values:
        outward_residual = float(min(outward_values))
    outward_oriented = bool(face_shape_valid and outward_residual > tolerance)
    if not face_shape_valid:
        blockers.append("face_incidence_is_not_20_distinct_triangles")
    if not outward_oriented:
        blockers.append("faces_are_not_consistently_outward_oriented")

    face_edges_valid = True
    edge_face_counts: dict[tuple[int, int], int] = {edge: 0 for edge in edge_set}
    for face in faces:
        if len(face) != 3:
            face_edges_valid = False
            continue
        for pair in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted(pair))
            if key not in edge_set:
                face_edges_valid = False
            else:
                edge_face_counts[key] += 1
    closed_surface = bool(
        face_edges_valid
        and len(edge_face_counts) == 30
        and all(count == 2 for count in edge_face_counts.values())
        and _PORT_COUNT - len(edge_set) + len(unoriented_face_set) == 2
    )
    if not closed_surface:
        blockers.append("edge_face_incidence_is_not_a_closed_euler_two_surface")

    antipode_valid = bool(
        len(antipode) == _PORT_COUNT
        and sorted(antipode) == list(range(_PORT_COUNT))
        and all(antipode[antipode[index]] == index for index in range(_PORT_COUNT))
        and all(antipode[index] != index for index in range(_PORT_COUNT))
    )
    antipodal_residual = math.inf
    if antipode_valid:
        antipodal_residual = float(
            np.max(
                np.linalg.norm(
                    coordinates + coordinates[np.asarray(antipode, dtype=int)], axis=1
                )
            )
        )
        antipode_valid = antipodal_residual <= tolerance
    if not antipode_valid:
        blockers.append("antipode_is_not_a_fixed_point_free_geometric_involution")

    a5 = _a5_local_action_audit(
        actions,
        edge_set=edge_set,
        oriented_face_set=oriented_face_set,
        antipode=antipode,
        coordinates=coordinates,
        tolerance=tolerance,
    )
    if not a5["receipt"]:
        blockers.extend(a5["blockers"])

    reference_matches = _reference_icosahedron_isomorphism_receipt(
        coordinates,
        edge_set=edge_set,
        oriented_face_set=oriented_face_set,
        antipode=antipode,
        tolerance=tolerance,
    )
    if not reference_matches["receipt"]:
        blockers.append("carrier_is_not_an_exact_relabeling_of_reference_icosahedron")

    adjacency = np.zeros((_PORT_COUNT, _PORT_COUNT), dtype=float)
    for left, right in edge_set:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    spectrum = np.linalg.eigvalsh(adjacency)
    multiplicities = _eigenvalue_multiplicities(spectrum, tolerance=1.0e-8)
    irrep_multiplicities = sorted(multiplicity for _, multiplicity in multiplicities)
    response_decomposition = irrep_multiplicities == [1, 3, 3, 5]
    if not response_decomposition:
        blockers.append("adjacency_response_does_not_have_1_3_3prime_5_sectors")

    reference = echosahedral_patch_architecture_report(12)
    passed = not blockers
    structural_payload = {
        "schema": "oph.echosahedral_carrier.structural_class.v1",
        "port_count": 12,
        "edge_count": 30,
        "face_count": 20,
        "degree_profile": [5] * 12,
        "antipodal_pair_count": 6,
        "a5_order": 60,
        "a5_order_profile": {"1": 1, "2": 15, "3": 20, "5": 24},
        "response_multiplicities": [1, 3, 3, 5],
    }
    return {
        "schema": "oph.echosahedral_carrier.conformance.v1",
        "carrier_id": carrier.carrier_id,
        "port_count": len(carrier.port_names),
        "edge_count": len(edge_set),
        "face_count": len(oriented_face_set),
        "vertex_degree_profile": {
            str(value): int(np.sum(degrees == value)) for value in sorted(set(degrees))
        },
        "maximum_unit_norm_residual": coordinate_norm_residual,
        "minimum_outward_face_orientation_witness": outward_residual,
        "maximum_antipodal_residual": antipodal_residual,
        "antipode_fixed_point_count": sum(
            index == partner for index, partner in enumerate(antipode)
        )
        if len(antipode) == _PORT_COUNT
        else None,
        "closed_euler_two_surface": closed_surface,
        "a5_action": a5,
        "reference_isomorphism": reference_matches,
        "adjacency_spectrum": [float(value) for value in spectrum],
        "adjacency_eigenvalue_multiplicities": [
            {"eigenvalue": value, "multiplicity": multiplicity}
            for value, multiplicity in multiplicities
        ],
        "declared_response_sector_dimensions": [1, 3, 3, 5],
        "hidden_presentation_coordinates": True,
        "hidden_coordinates_eligible_for_promoted_geometry": False,
        "shared_reference_template_hash": reference.get("template_hash"),
        "structural_class_sha256": interface_algebra_sha256(structural_payload),
        "blockers": sorted(set(blockers)),
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE": passed,
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE_RECEIPT": passed,
        "LOCAL_RESPONSE_1_3_3PRIME_5_DECOMPOSITION_RECEIPT": response_decomposition,
        "claim_boundary": (
            "This certifies a hidden finite carrier presentation only. The local XYZ "
            "coordinates and port frame are ineligible as S2, H3, event, cap-normal, "
            "clock, or BW source coordinates."
        ),
    }


def federation_sewing_report(
    federation: EchosahedralFederation,
) -> dict[str, Any]:
    """Validate typed seams, explicit dangling boundaries, and supports."""

    blockers: list[str] = []
    carrier_by_id: dict[str, EchosahedralCarrier] = {}
    carrier_reports: dict[str, dict[str, Any]] = {}
    for carrier in federation.carriers:
        if carrier.carrier_id in carrier_by_id:
            blockers.append(f"duplicate_carrier_id:{carrier.carrier_id}")
        carrier_by_id[carrier.carrier_id] = carrier
        carrier_reports[carrier.carrier_id] = echosahedral_carrier_conformance_report(
            carrier
        )
    if not federation.carriers:
        blockers.append("federation_has_no_carriers")
    carrier_conformance = bool(
        federation.carriers
        and len(carrier_by_id) == len(federation.carriers)
        and all(
            report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"]
            for report in carrier_reports.values()
        )
    )

    seam_ids: set[str] = set()
    seam_rows: list[dict[str, Any]] = []
    occupied_ports: set[tuple[str, int]] = set()
    carrier_graph = nx.Graph()
    carrier_graph.add_nodes_from(carrier_by_id)
    seam_by_id: dict[str, SeamBundle] = {}
    for seam in federation.seams:
        row = _seam_bundle_report(seam, carrier_by_id)
        seam_rows.append(row)
        if not seam.seam_id or seam.seam_id in seam_ids:
            blockers.append(f"duplicate_or_empty_seam_id:{seam.seam_id}")
        seam_ids.add(seam.seam_id)
        seam_by_id[seam.seam_id] = seam
        if not row["SEAM_BUNDLE_RECEIPT"]:
            blockers.extend(f"seam:{seam.seam_id}:{item}" for item in row["blockers"])
        if (
            seam.left_carrier_id in carrier_by_id
            and seam.right_carrier_id in carrier_by_id
        ):
            carrier_graph.add_edge(
                seam.left_carrier_id, seam.right_carrier_id, seam_id=seam.seam_id
            )
        for carrier_id, ports in (
            (seam.left_carrier_id, seam.left_ports),
            (seam.right_carrier_id, seam.right_ports),
        ):
            for port in ports:
                endpoint = (carrier_id, int(port))
                if endpoint in occupied_ports:
                    blockers.append(
                        f"local_port_used_by_more_than_one_seam:{carrier_id}:P{port}"
                    )
                occupied_ports.add(endpoint)

    boundary_ids: set[str] = set()
    boundary_rows: list[dict[str, Any]] = []
    declared_external_ports: set[tuple[str, int]] = set()
    for boundary in federation.external_boundaries:
        row = _external_boundary_report(boundary, carrier_by_id)
        boundary_rows.append(row)
        if not boundary.boundary_id or boundary.boundary_id in boundary_ids:
            blockers.append(
                f"duplicate_or_empty_external_boundary_id:{boundary.boundary_id}"
            )
        boundary_ids.add(boundary.boundary_id)
        if not row["EXPLICIT_EXTERNAL_BOUNDARY_RECEIPT"]:
            blockers.extend(
                f"boundary:{boundary.boundary_id}:{item}" for item in row["blockers"]
            )
        for port in boundary.ports:
            endpoint = (boundary.carrier_id, int(port))
            if endpoint in declared_external_ports:
                blockers.append(
                    f"local_port_declared_external_more_than_once:{boundary.carrier_id}:P{port}"
                )
            if endpoint in occupied_ports:
                blockers.append(
                    f"sewn_port_also_declared_external:{boundary.carrier_id}:P{port}"
                )
            declared_external_ports.add(endpoint)

    all_ports = {
        (carrier_id, port)
        for carrier_id in carrier_by_id
        for port in range(_PORT_COUNT)
    }
    dangling_ports = sorted(all_ports - occupied_ports - declared_external_ports)
    unknown_declared_ports = sorted(
        (occupied_ports | declared_external_ports) - all_ports
    )
    if dangling_ports:
        blockers.append("dangling_ports_lack_explicit_external_boundary_declaration")
    if unknown_declared_ports:
        blockers.append("seam_or_boundary_references_unknown_local_port")

    federation_connected = bool(
        carrier_graph.number_of_nodes() == 1
        or (carrier_graph.number_of_nodes() > 1 and nx.is_connected(carrier_graph))
    )
    if not federation_connected:
        blockers.append("intercarrier_seam_graph_is_disconnected")

    observer_rows = [
        observer_support_report(support, carrier_by_id, seam_by_id)
        for support in federation.observer_supports
    ]
    observer_tokens = [
        support.observer_token for support in federation.observer_supports
    ]
    if len(set(observer_tokens)) != len(observer_tokens):
        blockers.append("duplicate_observer_token")
    for row in observer_rows:
        if not row["CONNECTED_OBSERVER_SUPPORT_RECEIPT"]:
            blockers.extend(
                f"observer:{row['observer_token']}:{item}" for item in row["blockers"]
            )

    passed = bool(
        carrier_conformance
        and not blockers
        and all(row["SEAM_BUNDLE_RECEIPT"] for row in seam_rows)
        and all(row["EXPLICIT_EXTERNAL_BOUNDARY_RECEIPT"] for row in boundary_rows)
        and all(row["CONNECTED_OBSERVER_SUPPORT_RECEIPT"] for row in observer_rows)
    )
    return {
        "schema": "oph.echosahedral_federation.sewing.v1",
        "federation_id": federation.federation_id,
        "cardinality_semantics": "finite_carrier_regulator_count_only",
        "carrier_count": len(federation.carriers),
        "carrier_count_is_support_chart_cell_count": False,
        "carrier_count_is_screen_entropy_capacity_N_star": False,
        "carrier_count_is_primitive_observer_count": False,
        "seam_count": len(federation.seams),
        "external_boundary_bundle_count": len(federation.external_boundaries),
        "observer_support_count": len(federation.observer_supports),
        "sewn_local_port_count": len(occupied_ports),
        "declared_external_local_port_count": len(declared_external_ports),
        "undeclared_dangling_ports": [
            {"carrier_id": carrier_id, "port": port}
            for carrier_id, port in dangling_ports
        ],
        "carrier_conformance": carrier_reports,
        "seams": seam_rows,
        "external_boundaries": boundary_rows,
        "observer_supports": observer_rows,
        "federation_carrier_graph_connected": federation_connected,
        "blockers": sorted(set(blockers)),
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE": carrier_conformance,
        "FEDERATION_SEWING_RECEIPT": passed,
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
        "claim_boundary": (
            "Carrier cardinality is only a finite regulator parameter. It is not an "
            "S2 chart-cell count, N_star, primitive-observer count, H3-point count, "
            "or event count. Sewing certifies typed local interfaces only."
        ),
    }


def observer_support_report(
    support: ObserverSupport,
    carrier_by_id: Mapping[str, EchosahedralCarrier],
    seam_by_id: Mapping[str, SeamBundle],
) -> dict[str, Any]:
    """Check that an observer token has connected visible carrier support."""

    blockers: list[str] = []
    if not support.observer_token:
        blockers.append("empty_observer_token")
    if not support.carrier_ids:
        blockers.append("observer_support_has_no_carriers")
    unknown_carriers = sorted(set(support.carrier_ids) - set(carrier_by_id))
    if unknown_carriers:
        blockers.append("observer_support_references_unknown_carrier")
    unknown_seams = sorted(set(support.visible_seam_ids) - set(seam_by_id))
    if unknown_seams:
        blockers.append("observer_support_references_unknown_visible_seam")
    if not _is_sha256(support.record_algebra_sha256):
        blockers.append("invalid_record_algebra_sha256")
    if not _is_sha256(support.checkpoint_cut_sha256):
        blockers.append("invalid_checkpoint_cut_sha256")

    graph = nx.Graph()
    graph.add_nodes_from(support.carrier_ids)
    for seam_id in support.visible_seam_ids:
        seam = seam_by_id.get(seam_id)
        if seam is None:
            continue
        endpoints = {seam.left_carrier_id, seam.right_carrier_id}
        if not endpoints <= set(support.carrier_ids):
            blockers.append("visible_seam_leaves_observer_carrier_support")
            continue
        graph.add_edge(seam.left_carrier_id, seam.right_carrier_id)
    connected = bool(
        graph.number_of_nodes() == 1
        or (graph.number_of_nodes() > 1 and nx.is_connected(graph))
    )
    if not connected:
        blockers.append("observer_carrier_support_is_disconnected")
    passed = not blockers
    return {
        "schema": "oph.echosahedral_federation.observer_support.v1",
        "observer_token": support.observer_token,
        "carrier_ids": sorted(support.carrier_ids),
        "visible_seam_ids": sorted(support.visible_seam_ids),
        "carrier_count": len(support.carrier_ids),
        "one_carrier_support_allowed": True,
        "connected": connected,
        "blockers": sorted(set(blockers)),
        "CONNECTED_OBSERVER_SUPPORT_RECEIPT": passed,
        "OBSERVER_EQUALS_ONE_CARRIER_ASSUMPTION": False,
    }


def _seam_bundle_report(
    seam: SeamBundle,
    carrier_by_id: Mapping[str, EchosahedralCarrier],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not seam.seam_id:
        blockers.append("empty_seam_id")
    if seam.left_carrier_id == seam.right_carrier_id:
        blockers.append("seam_endpoints_must_be_distinct_carriers")
    left = carrier_by_id.get(seam.left_carrier_id)
    right = carrier_by_id.get(seam.right_carrier_id)
    if left is None:
        blockers.append("unknown_left_carrier")
    if right is None:
        blockers.append("unknown_right_carrier")
    left_ports = tuple(int(port) for port in seam.left_ports)
    right_ports = tuple(int(port) for port in seam.right_ports)
    forward = tuple(int(port) for port in seam.left_to_right_ports)
    backward = tuple(int(port) for port in seam.right_to_left_ports)
    bundle_sizes_valid = bool(
        left_ports
        and len(left_ports) == len(set(left_ports))
        and len(right_ports) == len(set(right_ports)) == len(left_ports)
        and len(forward) == len(left_ports)
        and len(backward) == len(right_ports)
    )
    if not bundle_sizes_valid:
        blockers.append("port_bundles_and_gluing_maps_do_not_have_equal_nonzero_size")
    port_ranges_valid = bool(
        all(
            0 <= port < _PORT_COUNT
            for port in left_ports + right_ports + forward + backward
        )
    )
    if not port_ranges_valid:
        blockers.append("seam_port_out_of_range")
    bijection_valid = bool(
        bundle_sizes_valid
        and set(forward) == set(right_ports)
        and set(backward) == set(left_ports)
    )
    inverse_composition = False
    if bijection_valid:
        forward_map = dict(zip(left_ports, forward, strict=True))
        backward_map = dict(zip(right_ports, backward, strict=True))
        inverse_composition = bool(
            all(backward_map[forward_map[port]] == port for port in left_ports)
            and all(forward_map[backward_map[port]] == port for port in right_ports)
        )
    if not bijection_valid:
        blockers.append("gluing_map_is_not_a_bijection_between_endpoint_bundles")
    if not inverse_composition:
        blockers.append("forward_reverse_gluing_composition_is_not_identity")

    orientation_valid = bool(
        len(seam.left_to_right_orientation) == len(left_ports)
        and len(seam.right_to_left_orientation) == len(right_ports)
        and all(sign == -1 for sign in seam.left_to_right_orientation)
        and all(sign == -1 for sign in seam.right_to_left_orientation)
    )
    orientation_inverse_composition = False
    if orientation_valid and bijection_valid:
        forward_sign = dict(
            zip(left_ports, seam.left_to_right_orientation, strict=True)
        )
        backward_sign = dict(
            zip(right_ports, seam.right_to_left_orientation, strict=True)
        )
        forward_map = dict(zip(left_ports, forward, strict=True))
        backward_map = dict(zip(right_ports, backward, strict=True))
        orientation_inverse_composition = bool(
            all(
                forward_sign[port] * backward_sign[forward_map[port]] == 1
                and backward_map[forward_map[port]] == port
                for port in left_ports
            )
        )
    if not orientation_valid:
        blockers.append("seam_orientation_maps_are_not_explicit_reversals")
    if not orientation_inverse_composition:
        blockers.append("forward_reverse_orientation_composition_is_not_identity")

    left_connected = bool(
        left is not None
        and (
            _port_subset_connected(left, left_ports)
            or (
                seam.collar_kind == "antipodal_pair"
                and _collar_kind_matches(left, left_ports, seam.collar_kind)
            )
        )
    )
    right_connected = bool(
        right is not None
        and (
            _port_subset_connected(right, right_ports)
            or (
                seam.collar_kind == "antipodal_pair"
                and _collar_kind_matches(right, right_ports, seam.collar_kind)
            )
        )
    )
    if not left_connected:
        blockers.append("left_port_bundle_is_not_connected_in_local_incidence")
    if not right_connected:
        blockers.append("right_port_bundle_is_not_connected_in_local_incidence")
    collar_type_valid = bool(
        left is not None
        and right is not None
        and _collar_kind_matches(left, left_ports, seam.collar_kind)
        and _collar_kind_matches(right, right_ports, seam.collar_kind)
    )
    if not collar_type_valid:
        blockers.append("declared_collar_kind_does_not_match_local_port_incidence")

    binding = seam.interface_algebra
    hashes_valid = all(
        _is_sha256(value)
        for value in (
            binding.interface_algebra_sha256,
            binding.left_interface_algebra_sha256,
            binding.right_interface_algebra_sha256,
        )
    )
    algebra_preserved = bool(
        hashes_valid
        and binding.interface_algebra_sha256
        == binding.left_interface_algebra_sha256
        == binding.right_interface_algebra_sha256
    )
    if not binding.interface_algebra_id:
        blockers.append("empty_interface_algebra_id")
    if not hashes_valid:
        blockers.append("invalid_interface_algebra_sha256")
    if not algebra_preserved:
        blockers.append("endpoint_interface_algebra_hashes_do_not_agree")

    passed = not blockers
    return {
        "schema": "oph.echosahedral_federation.seam_bundle.v1",
        "seam_id": seam.seam_id,
        "left_carrier_id": seam.left_carrier_id,
        "right_carrier_id": seam.right_carrier_id,
        "left_ports": list(left_ports),
        "right_ports": list(right_ports),
        "bundle_size": len(left_ports),
        "collar_kind": seam.collar_kind,
        "left_bundle_connected": left_connected,
        "right_bundle_connected": right_connected,
        "gluing_map_bijective": bijection_valid,
        "forward_reverse_composition_identity": inverse_composition,
        "orientation_reversing": orientation_valid,
        "forward_reverse_orientation_composition_identity": (
            orientation_inverse_composition
        ),
        "bundle_connectivity_mode": (
            "declared_antipode_involution"
            if seam.collar_kind == "antipodal_pair"
            else "induced_local_edge_incidence"
        ),
        "interface_algebra_id": binding.interface_algebra_id,
        "interface_algebra_sha256": binding.interface_algebra_sha256,
        "endpoint_interface_algebra_hashes_agree": algebra_preserved,
        "blockers": sorted(set(blockers)),
        "SEAM_BUNDLE_RECEIPT": passed,
    }


def _external_boundary_report(
    boundary: ExternalBoundaryBundle,
    carrier_by_id: Mapping[str, EchosahedralCarrier],
) -> dict[str, Any]:
    blockers: list[str] = []
    carrier = carrier_by_id.get(boundary.carrier_id)
    ports = tuple(int(port) for port in boundary.ports)
    if not boundary.boundary_id:
        blockers.append("empty_boundary_id")
    if carrier is None:
        blockers.append("unknown_boundary_carrier")
    if not ports or len(ports) != len(set(ports)):
        blockers.append("external_boundary_ports_must_be_nonempty_and_unique")
    if any(port < 0 or port >= _PORT_COUNT for port in ports):
        blockers.append("external_boundary_port_out_of_range")
    connected = bool(carrier is not None and _port_subset_connected(carrier, ports))
    if not connected:
        blockers.append("external_boundary_port_bundle_is_not_connected")
    if boundary.boundary_condition not in {
        "open_external",
        "fixed_external",
        "measured_external",
    }:
        blockers.append("unknown_external_boundary_condition")
    if not _is_sha256(boundary.boundary_algebra_sha256):
        blockers.append("invalid_boundary_algebra_sha256")
    return {
        "schema": "oph.echosahedral_federation.external_boundary.v1",
        "boundary_id": boundary.boundary_id,
        "carrier_id": boundary.carrier_id,
        "ports": list(ports),
        "port_count": len(ports),
        "connected": connected,
        "boundary_condition": boundary.boundary_condition,
        "boundary_algebra_sha256": boundary.boundary_algebra_sha256,
        "blockers": sorted(set(blockers)),
        "EXPLICIT_EXTERNAL_BOUNDARY_RECEIPT": not blockers,
    }


def presentation_firewall_report(
    federation: EchosahedralFederation,
    promoted_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject hidden carrier presentation data from promoted payloads."""

    payload = {} if promoted_payload is None else promoted_payload
    encountered_keys = _recursive_mapping_keys(payload)
    forbidden_keys = sorted(encountered_keys & CARRIER_FORBIDDEN_PROMOTION_FIELDS)
    coordinate_leaks = _find_hidden_coordinate_values(payload, federation.carriers)
    port_label_leaks = _find_hidden_port_label_values(payload, federation.carriers)
    passed = not forbidden_keys and not coordinate_leaks and not port_label_leaks
    return {
        "schema": "oph.echosahedral_federation.presentation_firewall.v1",
        "hidden_presentation_fields": sorted(HIDDEN_PRESENTATION_FIELDS),
        "quotient_visible_carrier_fields": sorted(QUOTIENT_VISIBLE_CARRIER_FIELDS),
        "forbidden_promoted_fields": sorted(CARRIER_FORBIDDEN_PROMOTION_FIELDS),
        "encountered_payload_keys": sorted(encountered_keys),
        "forbidden_keys_present": forbidden_keys,
        "hidden_coordinate_value_paths": coordinate_leaks,
        "hidden_port_label_value_paths": port_label_leaks,
        "CARRIER_PRESENTATION_FIREWALL_RECEIPT": passed,
        "HIDDEN_COORDINATES_EXCLUDED_FROM_PROMOTED_PAYLOAD_RECEIPT": bool(
            not coordinate_leaks
            and not (
                {"port_coordinates", "hidden_port_coordinates"} & set(forbidden_keys)
            )
        ),
        "LOCAL_PORT_NAMES_EXCLUDED_FROM_PROMOTED_PAYLOAD_RECEIPT": bool(
            not port_label_leaks
            and not ({"port_names", "local_port_names"} & set(forbidden_keys))
        ),
        "claim_boundary": (
            "The schema firewall excludes known presentation fields and exact hidden "
            "coordinate/name values. It does not by itself prove quotient invariance "
            "of a future dynamical emergence evaluator."
        ),
    }


def relabel_federation_ports(
    federation: EchosahedralFederation,
    port_permutations: Mapping[str, Sequence[int]],
) -> EchosahedralFederation:
    """Co-transform every carrier, seam, and external-boundary port label."""

    carrier_ids = {carrier.carrier_id for carrier in federation.carriers}
    if set(port_permutations) != carrier_ids:
        raise ValueError("one explicit port permutation is required per carrier")
    normalized = {
        carrier_id: _validated_port_permutation(permutation)
        for carrier_id, permutation in port_permutations.items()
    }
    carriers = tuple(
        _relabel_carrier(carrier, normalized[carrier.carrier_id])
        for carrier in federation.carriers
    )
    seams = []
    for seam in federation.seams:
        left_permutation = normalized[seam.left_carrier_id]
        right_permutation = normalized[seam.right_carrier_id]
        seams.append(
            replace(
                seam,
                left_ports=tuple(left_permutation[port] for port in seam.left_ports),
                right_ports=tuple(right_permutation[port] for port in seam.right_ports),
                left_to_right_ports=tuple(
                    right_permutation[port] for port in seam.left_to_right_ports
                ),
                right_to_left_ports=tuple(
                    left_permutation[port] for port in seam.right_to_left_ports
                ),
            )
        )
    boundaries = tuple(
        replace(
            boundary,
            ports=tuple(
                normalized[boundary.carrier_id][port] for port in boundary.ports
            ),
        )
        for boundary in federation.external_boundaries
    )
    return replace(
        federation,
        carriers=carriers,
        seams=tuple(seams),
        external_boundaries=boundaries,
    )


def carrier_quotient_invariance_report(
    source: EchosahedralFederation,
    transformed: EchosahedralFederation,
    port_permutations: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Verify one exact presentation relabeling and its quotient-visible export."""

    blockers: list[str] = []
    try:
        expected = relabel_federation_ports(source, port_permutations)
    except (KeyError, TypeError, ValueError) as exc:
        expected = None
        blockers.append(f"invalid_presentation_transform:{type(exc).__name__}")
    exact_cotransformation = bool(expected is not None and expected == transformed)
    if not exact_cotransformation:
        blockers.append("transformed_federation_is_not_exact_co_transformation")
    source_sewing = federation_sewing_report(source)
    transformed_sewing = federation_sewing_report(transformed)
    if not source_sewing["FEDERATION_SEWING_RECEIPT"]:
        blockers.append("source_federation_sewing_invalid")
    if not transformed_sewing["FEDERATION_SEWING_RECEIPT"]:
        blockers.append("transformed_federation_sewing_invalid")
    source_payload = _quotient_visible_contract_payload(source)
    transformed_payload = _quotient_visible_contract_payload(transformed)
    source_hash = interface_algebra_sha256(source_payload)
    transformed_hash = interface_algebra_sha256(transformed_payload)
    quotient_exports_equal = source_hash == transformed_hash
    if not quotient_exports_equal:
        blockers.append("quotient_visible_contract_export_changed_under_relabeling")
    source_firewall = presentation_firewall_report(source)
    transformed_firewall = presentation_firewall_report(transformed)
    passed = bool(
        not blockers
        and exact_cotransformation
        and quotient_exports_equal
        and source_firewall["CARRIER_PRESENTATION_FIREWALL_RECEIPT"]
        and transformed_firewall["CARRIER_PRESENTATION_FIREWALL_RECEIPT"]
    )
    return {
        "schema": "oph.echosahedral_federation.quotient_invariance.v1",
        "source_federation_id": source.federation_id,
        "transformed_federation_id": transformed.federation_id,
        "transformation_class": "explicit_local_port_relabeling_with_seams_and_boundaries_cotransformed",
        "carrier_permutation_count": len(port_permutations),
        "exact_cotransformation_verified": exact_cotransformation,
        "source_quotient_visible_contract_sha256": source_hash,
        "transformed_quotient_visible_contract_sha256": transformed_hash,
        "quotient_visible_contract_exports_equal": quotient_exports_equal,
        "blockers": sorted(set(blockers)),
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": passed,
        "FULL_DYNAMICAL_QUOTIENT_INVARIANCE_RECEIPT": False,
        "claim_boundary": (
            "This receipt proves invariance of the finite carrier/sewing contract under "
            "the supplied exact relabeling. Engine dynamics, repair schedules, and "
            "downstream geometry evaluators are not yet integrated or certified."
        ),
    }


def carrier_refinement_naturality_report() -> dict[str, Any]:
    """Fail closed until a genuine carrier refinement/coarse-graining map exists."""

    return {
        "schema": "oph.echosahedral_federation.refinement_naturality.v1",
        "local_twelve_port_type_declared": True,
        "carrier_embedding_map_supplied": False,
        "carrier_coarse_graining_map_supplied": False,
        "coarse_after_embedding_identity_verified": False,
        "seam_law_naturality_verified": False,
        "quotient_map_commutation_verified": False,
        "carrier_to_support_realization_verified": False,
        "blockers": [
            "carrier_refinement_embedding_not_implemented",
            "carrier_coarse_graining_not_implemented",
            "seam_refinement_naturality_not_implemented",
            "carrier_quotient_refinement_commutation_not_implemented",
            "carrier_to_support_chart_realization_not_implemented",
        ],
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT": False,
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
    }


def echosahedral_federation_receipt(
    federation: EchosahedralFederation,
    *,
    promoted_payload: Mapping[str, Any] | None = None,
    equivalent_presentation: EchosahedralFederation | None = None,
    presentation_port_permutations: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Emit the four non-promoting carrier-level parent receipts."""

    sewing = federation_sewing_report(federation)
    firewall = presentation_firewall_report(federation, promoted_payload)
    if (
        equivalent_presentation is not None
        and presentation_port_permutations is not None
    ):
        quotient = carrier_quotient_invariance_report(
            federation,
            equivalent_presentation,
            presentation_port_permutations,
        )
    else:
        quotient = {
            "schema": "oph.echosahedral_federation.quotient_invariance.v1",
            "blockers": ["independent_equivalent_presentation_witness_missing"],
            "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": False,
            "FULL_DYNAMICAL_QUOTIENT_INVARIANCE_RECEIPT": False,
        }
    refinement = carrier_refinement_naturality_report()
    carrier_receipt = bool(sewing["ECHOSAHEDRAL_CARRIER_CONFORMANCE"])
    sewing_receipt = bool(sewing["FEDERATION_SEWING_RECEIPT"])
    quotient_receipt = bool(
        quotient["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"]
        and firewall["CARRIER_PRESENTATION_FIREWALL_RECEIPT"]
    )
    refinement_receipt = bool(refinement["CARRIER_REFINEMENT_NATURALITY_RECEIPT"])
    source_instrument_valid = bool(
        carrier_receipt and sewing_receipt and quotient_receipt and refinement_receipt
    )
    return {
        "schema": "oph.echosahedral_federation.parent_receipts.v1",
        "instrument_scope": "finite_two_level_echosahedral_carrier_federation_contract",
        "federation_id": federation.federation_id,
        "cardinality_semantics": "carrier_count_is_regulator_cardinality_not_support_chart_discretization",
        "carrier_count": len(federation.carriers),
        "support_chart_cell_count": None,
        "carrier_to_support_chart_realization": "unproved",
        "sewing": sewing,
        "presentation_firewall": firewall,
        "quotient_invariance": quotient,
        "refinement_naturality": refinement,
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE": carrier_receipt,
        "FEDERATION_SEWING_RECEIPT": sewing_receipt,
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": quotient_receipt,
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT": refinement_receipt,
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
        "ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID": source_instrument_valid,
        "S2_SUPPORT_CHART_EMERGENCE_RECEIPT": False,
        "H3_FRAME_EMERGENCE_RECEIPT": False,
        "EVENT_MANIFOLD_RECEIPT": False,
        "BW_KMS_CLOCK_RECEIPT": False,
        "PHYSICAL_H3_KMS_EMERGENCE_RECEIPT": False,
        "claim_boundary": (
            "These parent receipts stop at the finite carrier contract. Carrier count "
            "is not an S2 chart-cell count, N_star, observer count, H3-point count, or "
            "event count. S2, H3, events, BW/KMS, the 2pi normalization, and a "
            "carrier-to-support realization must be earned by downstream receipts."
        ),
    }


def reference_federation_instrument_bundle(
    federation: EchosahedralFederation,
    *,
    promoted_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a compact shared-template bundle without duplicating XYZ data.

    This exporter intentionally accepts only carriers in the canonical shared
    reference presentation.  Presentation-relabeling witnesses use the typed
    API above; a compact bundle must first be canonicalized explicitly.
    """

    if any(
        carrier != reference_echosahedral_carrier(carrier.carrier_id)
        for carrier in federation.carriers
    ):
        raise ValueError(
            "reference instrument bundles require canonical shared-template carriers"
        )
    return {
        "schema": "oph.echosahedral_federation.instrument_bundle.v1",
        "federation_id": federation.federation_id,
        "local_carrier_template": "regular_icosahedron_12_30_20_antipode_a5_v1",
        "carrier_ids": [carrier.carrier_id for carrier in federation.carriers],
        "seams": [
            {
                "seam_id": seam.seam_id,
                "left_carrier_id": seam.left_carrier_id,
                "right_carrier_id": seam.right_carrier_id,
                "left_ports": list(seam.left_ports),
                "right_ports": list(seam.right_ports),
                "left_to_right_ports": list(seam.left_to_right_ports),
                "right_to_left_ports": list(seam.right_to_left_ports),
                "left_to_right_orientation": list(seam.left_to_right_orientation),
                "right_to_left_orientation": list(seam.right_to_left_orientation),
                "collar_kind": seam.collar_kind,
                "interface_algebra": {
                    "interface_algebra_id": (
                        seam.interface_algebra.interface_algebra_id
                    ),
                    "interface_algebra_sha256": (
                        seam.interface_algebra.interface_algebra_sha256
                    ),
                    "left_interface_algebra_sha256": (
                        seam.interface_algebra.left_interface_algebra_sha256
                    ),
                    "right_interface_algebra_sha256": (
                        seam.interface_algebra.right_interface_algebra_sha256
                    ),
                },
            }
            for seam in federation.seams
        ],
        "external_boundaries": [
            {
                "boundary_id": boundary.boundary_id,
                "carrier_id": boundary.carrier_id,
                "ports": list(boundary.ports),
                "boundary_condition": boundary.boundary_condition,
                "boundary_algebra_sha256": boundary.boundary_algebra_sha256,
            }
            for boundary in federation.external_boundaries
        ],
        "observer_supports": [
            {
                "observer_token": support.observer_token,
                "carrier_ids": sorted(support.carrier_ids),
                "visible_seam_ids": sorted(support.visible_seam_ids),
                "record_algebra_sha256": support.record_algebra_sha256,
                "checkpoint_cut_sha256": support.checkpoint_cut_sha256,
            }
            for support in federation.observer_supports
        ],
        "promoted_payload": {} if promoted_payload is None else promoted_payload,
    }


def verify_reference_federation_instrument_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse and verify the compact JSON-facing carrier instrument bundle."""

    try:
        if bundle.get("schema") != "oph.echosahedral_federation.instrument_bundle.v1":
            raise ValueError("instrument bundle schema mismatch")
        if (
            bundle.get("local_carrier_template")
            != "regular_icosahedron_12_30_20_antipode_a5_v1"
        ):
            raise ValueError("unknown local carrier template")
        carrier_ids = tuple(str(value) for value in bundle["carrier_ids"])
        if not carrier_ids or len(set(carrier_ids)) != len(carrier_ids):
            raise ValueError("carrier_ids must be nonempty and unique")
        carriers = tuple(reference_echosahedral_carrier(value) for value in carrier_ids)
        seams = tuple(_seam_from_bundle_row(row) for row in bundle.get("seams", ()))
        boundaries = tuple(
            _boundary_from_bundle_row(row)
            for row in bundle.get("external_boundaries", ())
        )
        supports = tuple(
            _observer_from_bundle_row(row)
            for row in bundle.get("observer_supports", ())
        )
        federation = EchosahedralFederation(
            federation_id=str(bundle["federation_id"]),
            carriers=carriers,
            seams=seams,
            external_boundaries=boundaries,
            observer_supports=supports,
        )
        report = echosahedral_federation_receipt(
            federation,
            promoted_payload=bundle.get("promoted_payload", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema": "oph.echosahedral_federation.instrument_bundle_verification.v1",
            "parse_error": f"{type(exc).__name__}:{exc}",
            "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT": False,
            "ECHOSAHEDRAL_CARRIER_CONFORMANCE": False,
            "FEDERATION_SEWING_RECEIPT": False,
            "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": False,
            "CARRIER_REFINEMENT_NATURALITY_RECEIPT": False,
            "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
            "ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID": False,
        }
    report = dict(report)
    report["bundle_schema"] = bundle["schema"]
    report["shared_template_encoded_once"] = True
    report["per_carrier_coordinate_tables_embedded"] = False
    report["INSTRUMENT_BUNDLE_SCHEMA_RECEIPT"] = True
    return report


def screen_port_map_carrier_bridge_report(
    screen_port_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit current ScreenPortMap output without promoting it to full sewing."""

    local = bool(
        screen_port_report.get("ECHOSAHEDRAL_CARRIER_CONFORMANCE_RECEIPT") is True
        and screen_port_report.get("ports_per_patch") == 12
    )
    singleton_reference = bool(
        screen_port_report.get("REFERENCE_SINGLETON_FEDERATION_SEWING_RECEIPT") is True
    )
    federation_section = screen_port_report.get("federation_sewing", {})
    general_bundle_schema = bool(
        isinstance(federation_section, Mapping)
        and federation_section.get("general_bundle_schema_implemented") is True
    )
    interface_hashes = bool(
        isinstance(federation_section, Mapping)
        and federation_section.get("interface_algebra_hashes_bound") is True
    )
    blockers = [
        item
        for condition, item in (
            (not local, "local_twelve_port_carrier_conformance_missing"),
            (not singleton_reference, "singleton_reference_sewing_missing"),
            (not general_bundle_schema, "typed_general_seam_bundle_schema_missing"),
            (not interface_hashes, "interface_algebra_hash_binding_missing"),
            (True, "explicit_external_boundary_bundle_ledger_missing"),
            (True, "connected_observer_support_ledger_missing"),
            (True, "carrier_to_support_chart_realization_missing"),
            (True, "carrier_refinement_naturality_missing"),
        )
        if condition
    ]
    return {
        "schema": "oph.echosahedral_federation.screen_port_map_bridge.v1",
        "bridge_scope": "existing_screen_port_map_reference_audit_only",
        "local_carrier_conformance_imported": local,
        "singleton_reference_sewing_imported": singleton_reference,
        "typed_general_bundle_schema_present": general_bundle_schema,
        "interface_algebra_hashes_bound": interface_hashes,
        "blockers": blockers,
        "REFERENCE_SCREEN_PORT_MAP_CARRIER_BRIDGE_RECEIPT": bool(
            local and singleton_reference
        ),
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE": local,
        "FEDERATION_SEWING_RECEIPT": False,
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": False,
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT": False,
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
        "ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID": False,
        "S2_SUPPORT_CHART_EMERGENCE_RECEIPT": False,
        "H3_FRAME_EMERGENCE_RECEIPT": False,
        "EVENT_MANIFOLD_RECEIPT": False,
        "BW_KMS_CLOCK_RECEIPT": False,
        "claim_boundary": (
            "The current engine report may certify the shared local carrier template "
            "and singleton routing reference. It does not yet instantiate typed collar "
            "bundles, interface hashes, explicit external boundaries, observer supports, "
            "carrier-to-support realization, quotient dynamics, or refinement."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a compact bundle with ``python -m ...echosahedral_federation``."""

    parser = argparse.ArgumentParser(
        description="Verify an OPH echosahedral federation instrument bundle"
    )
    parser.add_argument("bundle", type=Path, help="JSON instrument bundle path")
    parser.add_argument(
        "--require",
        choices=("schema", "carrier", "sewing", "quotient", "refinement", "source"),
        default="sewing",
        help="receipt tier controlling the process exit status",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report = {
            "schema": "oph.echosahedral_federation.instrument_bundle_verification.v1",
            "parse_error": f"{type(exc).__name__}:{exc}",
            "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT": False,
        }
    else:
        report = verify_reference_federation_instrument_bundle(payload)
    print(json.dumps(report, sort_keys=True, indent=2))
    receipt_by_tier = {
        "schema": "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
        "carrier": "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
        "sewing": "FEDERATION_SEWING_RECEIPT",
        "quotient": "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
        "refinement": "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
        "source": "ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID",
    }
    return 0 if report.get(receipt_by_tier[args.require]) is True else 1


def _a5_local_action_audit(
    actions: Sequence[Sequence[int]],
    *,
    edge_set: set[tuple[int, int]],
    oriented_face_set: set[tuple[int, int, int]],
    antipode: Sequence[int],
    coordinates: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    permutations = tuple(tuple(int(value) for value in row) for row in actions)
    shape_valid = bool(
        len(permutations) == 60
        and all(
            len(row) == _PORT_COUNT and sorted(row) == list(range(_PORT_COUNT))
            for row in permutations
        )
        and len(set(permutations)) == 60
    )
    if not shape_valid:
        blockers.append("a5_action_is_not_60_unique_twelve_port_permutations")
    identity = tuple(range(_PORT_COUNT))
    group_closed = False
    inverses_present = False
    order_profile: dict[int, int] = {}
    if shape_valid:
        action_set = set(permutations)
        group_closed = all(
            _compose_permutations(left, right) in action_set
            for left in permutations
            for right in permutations
        )
        inverses_present = all(
            _inverse_permutation(row) in action_set for row in permutations
        )
        for row in permutations:
            try:
                order = _permutation_order(row)
            except ValueError:
                blockers.append("registered_permutation_order_exceeds_a5_bound")
                continue
            order_profile[order] = order_profile.get(order, 0) + 1
    expected_order_profile = {1: 1, 2: 15, 3: 20, 5: 24}
    if not group_closed:
        blockers.append("a5_permutation_family_not_closed")
    if not inverses_present:
        blockers.append("a5_permutation_inverses_missing")
    if identity not in set(permutations) or order_profile != expected_order_profile:
        blockers.append("a5_element_order_profile_invalid")

    edge_preserved = bool(shape_valid)
    face_preserved = bool(shape_valid)
    antipode_commutes = bool(
        shape_valid
        and len(antipode) == _PORT_COUNT
        and sorted(int(value) for value in antipode) == list(range(_PORT_COUNT))
    )
    maximum_gram_residual = math.inf
    maximum_rotation_residual = math.inf
    minimum_rotation_determinant = -math.inf
    if shape_valid:
        gram = coordinates @ coordinates.T
        gram_residuals: list[float] = []
        rotation_residuals: list[float] = []
        determinants: list[float] = []
        for row in permutations:
            edge_preserved = edge_preserved and all(
                tuple(sorted((row[left], row[right]))) in edge_set
                for left, right in edge_set
            )
            face_preserved = face_preserved and all(
                _cyclic_face_key((row[a], row[b], row[c])) in oriented_face_set
                for a, b, c in oriented_face_set
            )
            if antipode_commutes:
                antipode_commutes = all(
                    row[int(antipode[index])] == int(antipode[row[index]])
                    for index in range(_PORT_COUNT)
                )
            indexed = np.asarray(row, dtype=int)
            gram_residuals.append(
                float(np.max(np.abs(gram - gram[np.ix_(indexed, indexed)])))
            )
            rotation, residual = _proper_rotation_fit(coordinates, coordinates[indexed])
            determinants.append(float(np.linalg.det(rotation)))
            rotation_residuals.append(residual)
        maximum_gram_residual = max(gram_residuals, default=math.inf)
        maximum_rotation_residual = max(rotation_residuals, default=math.inf)
        minimum_rotation_determinant = min(determinants, default=-math.inf)
    if not edge_preserved:
        blockers.append("a5_action_does_not_preserve_all_edges")
    if not face_preserved:
        blockers.append("a5_action_does_not_preserve_all_oriented_faces")
    if not antipode_commutes:
        blockers.append("a5_action_does_not_commute_with_antipode")
    if maximum_gram_residual > tolerance:
        blockers.append("a5_action_does_not_preserve_coordinate_gram_matrix")
    if maximum_rotation_residual > tolerance or minimum_rotation_determinant <= 0.0:
        blockers.append("a5_action_is_not_realized_by_proper_coordinate_rotations")
    return {
        "registered_action_count": len(permutations),
        "unique_action_count": len(set(permutations)),
        "group_closed": group_closed,
        "inverses_present": inverses_present,
        "element_order_profile": {
            str(order): count for order, count in sorted(order_profile.items())
        },
        "all_actions_preserve_edges": edge_preserved,
        "all_actions_preserve_oriented_faces": face_preserved,
        "all_actions_commute_with_antipode": antipode_commutes,
        "maximum_coordinate_gram_residual": maximum_gram_residual,
        "maximum_proper_rotation_fit_residual": maximum_rotation_residual,
        "minimum_rotation_determinant": minimum_rotation_determinant,
        "blockers": sorted(set(blockers)),
        "receipt": not blockers,
        "A5_ORDER_60_LOCAL_ACTION_RECEIPT": not blockers,
    }


def _reference_icosahedron_isomorphism_receipt(
    coordinates: np.ndarray,
    *,
    edge_set: set[tuple[int, int]],
    oriented_face_set: set[tuple[int, int, int]],
    antipode: Sequence[int],
    tolerance: float,
) -> dict[str, Any]:
    reference = reference_echosahedral_carrier("reference")
    reference_coordinates = np.asarray(reference.port_coordinates, dtype=float)
    reference_edges = {tuple(sorted(edge)) for edge in reference.edges}
    reference_faces = {_cyclic_face_key(face) for face in reference.faces}
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(range(_PORT_COUNT))
    candidate_graph.add_edges_from(edge_set)
    reference_graph = nx.Graph()
    reference_graph.add_nodes_from(range(_PORT_COUNT))
    reference_graph.add_edges_from(reference_edges)
    valid_maps = 0
    minimum_coordinate_residual = math.inf
    if len(antipode) == _PORT_COUNT and sorted(
        int(value) for value in antipode
    ) == list(range(_PORT_COUNT)):
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            candidate_graph, reference_graph
        )
        for mapping in matcher.isomorphisms_iter():
            if not all(
                _cyclic_face_key((mapping[face[0]], mapping[face[1]], mapping[face[2]]))
                in reference_faces
                for face in oriented_face_set
            ):
                continue
            if not all(
                mapping[int(antipode[index])] == reference.antipode[mapping[index]]
                for index in range(_PORT_COUNT)
            ):
                continue
            target = np.zeros_like(reference_coordinates)
            for candidate_port, reference_port in mapping.items():
                target[candidate_port] = reference_coordinates[reference_port]
            rotation, residual = _proper_rotation_fit(coordinates, target)
            if np.linalg.det(rotation) > 0.0 and residual <= tolerance:
                valid_maps += 1
                minimum_coordinate_residual = min(minimum_coordinate_residual, residual)
    receipt = valid_maps > 0
    return {
        "proper_oriented_reference_isomorphism_count": valid_maps,
        "minimum_coordinate_rotation_residual": minimum_coordinate_residual,
        "receipt": receipt,
        "EXACT_REFERENCE_ICOSAHEDRON_ISOMORPHISM_RECEIPT": receipt,
    }


def _proper_rotation_fit(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    covariance = np.asarray(source, dtype=float).T @ np.asarray(target, dtype=float)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    residual = float(
        np.max(
            np.linalg.norm(
                np.asarray(source, dtype=float) @ rotation
                - np.asarray(target, dtype=float),
                axis=1,
            )
        )
    )
    return rotation, residual


def _eigenvalue_multiplicities(
    eigenvalues: np.ndarray, *, tolerance: float
) -> list[tuple[float, int]]:
    result: list[tuple[float, int]] = []
    for value in np.sort(np.asarray(eigenvalues, dtype=float)):
        if result and abs(value - result[-1][0]) <= tolerance:
            mean, count = result[-1]
            result[-1] = ((mean * count + float(value)) / (count + 1), count + 1)
        else:
            result.append((float(value), 1))
    return result


def _cyclic_face_key(face: Sequence[int]) -> tuple[int, int, int]:
    a, b, c = (int(value) for value in face)
    return min((a, b, c), (b, c, a), (c, a, b))


def _compose_permutations(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(left[int(right[index])]) for index in range(len(left)))


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[int(target)] = source
    return tuple(inverse)


def _permutation_order(permutation: Sequence[int]) -> int:
    identity = tuple(range(len(permutation)))
    current = identity
    for order in range(1, 61):
        current = _compose_permutations(permutation, current)
        if current == identity:
            return order
    raise ValueError("local permutation order exceeds A5 bound")


def _port_subset_connected(carrier: EchosahedralCarrier, ports: Sequence[int]) -> bool:
    port_set = {int(port) for port in ports}
    if not port_set or not port_set <= set(range(_PORT_COUNT)):
        return False
    graph = nx.Graph()
    graph.add_nodes_from(port_set)
    graph.add_edges_from(
        (left, right)
        for left, right in carrier.edges
        if left in port_set and right in port_set
    )
    return graph.number_of_nodes() == 1 or nx.is_connected(graph)


def _collar_kind_matches(
    carrier: EchosahedralCarrier,
    ports: Sequence[int],
    collar_kind: str,
) -> bool:
    port_tuple = tuple(int(port) for port in ports)
    port_set = set(port_tuple)
    if not port_tuple or not port_set <= set(range(_PORT_COUNT)):
        return False
    edge_set = {tuple(sorted(edge)) for edge in carrier.edges}
    face_set = {tuple(sorted(face)) for face in carrier.faces}
    if collar_kind == "single_port":
        return len(port_tuple) == 1
    if collar_kind == "antipodal_pair":
        return bool(
            len(port_tuple) == 2
            and len(carrier.antipode) == _PORT_COUNT
            and carrier.antipode[port_tuple[0]] == port_tuple[1]
            and carrier.antipode[port_tuple[1]] == port_tuple[0]
        )
    if collar_kind == "edge_bundle":
        return len(port_tuple) == 2 and tuple(sorted(port_tuple)) in edge_set
    if collar_kind == "face_collar":
        return len(port_tuple) == 3 and tuple(sorted(port_tuple)) in face_set
    if collar_kind == "connected_bundle":
        return bool(port_set and _port_subset_connected(carrier, port_tuple))
    return False


def _validated_port_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in permutation)
    if len(normalized) != _PORT_COUNT or sorted(normalized) != list(range(_PORT_COUNT)):
        raise ValueError("port permutation must be a bijection of 0..11")
    return normalized


def _relabel_carrier(
    carrier: EchosahedralCarrier,
    permutation: Sequence[int],
) -> EchosahedralCarrier:
    mapping = _validated_port_permutation(permutation)
    inverse = _inverse_permutation(mapping)
    names: list[str] = [""] * _PORT_COUNT
    coordinates: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * _PORT_COUNT
    antipode = [0] * _PORT_COUNT
    for old, new in enumerate(mapping):
        names[new] = carrier.port_names[old]
        coordinates[new] = carrier.port_coordinates[old]
        antipode[new] = mapping[carrier.antipode[old]]
    actions = []
    for action in carrier.a5_actions:
        conjugated = tuple(
            mapping[action[inverse[new_port]]] for new_port in range(_PORT_COUNT)
        )
        actions.append(conjugated)
    return replace(
        carrier,
        port_names=tuple(names),
        port_coordinates=tuple(coordinates),
        edges=tuple((mapping[left], mapping[right]) for left, right in carrier.edges),
        faces=tuple(
            (mapping[first], mapping[second], mapping[third])
            for first, second, third in carrier.faces
        ),
        antipode=tuple(antipode),
        a5_actions=tuple(actions),
    )


def _quotient_visible_contract_payload(
    federation: EchosahedralFederation,
) -> dict[str, Any]:
    carrier_rows = []
    for carrier in federation.carriers:
        report = echosahedral_carrier_conformance_report(carrier)
        carrier_rows.append(
            {
                "carrier_id": carrier.carrier_id,
                "structural_class_sha256": report["structural_class_sha256"],
                "conforming": report["ECHOSAHEDRAL_CARRIER_CONFORMANCE"],
            }
        )
    seam_rows = [
        {
            "seam_id": seam.seam_id,
            "endpoint_carrier_ids": sorted(
                (seam.left_carrier_id, seam.right_carrier_id)
            ),
            "collar_kind": seam.collar_kind,
            "bundle_size": len(seam.left_ports),
            "orientation_reversing": bool(
                all(sign == -1 for sign in seam.left_to_right_orientation)
                and all(sign == -1 for sign in seam.right_to_left_orientation)
            ),
            "interface_algebra_id": seam.interface_algebra.interface_algebra_id,
            "interface_algebra_sha256": seam.interface_algebra.interface_algebra_sha256,
        }
        for seam in federation.seams
    ]
    boundary_rows = [
        {
            "boundary_id": boundary.boundary_id,
            "carrier_id": boundary.carrier_id,
            "port_count": len(boundary.ports),
            "boundary_condition": boundary.boundary_condition,
            "boundary_algebra_sha256": boundary.boundary_algebra_sha256,
        }
        for boundary in federation.external_boundaries
    ]
    observer_rows = [
        {
            "observer_token": support.observer_token,
            "carrier_ids": sorted(support.carrier_ids),
            "visible_seam_ids": sorted(support.visible_seam_ids),
            "record_algebra_sha256": support.record_algebra_sha256,
            "checkpoint_cut_sha256": support.checkpoint_cut_sha256,
        }
        for support in federation.observer_supports
    ]
    return {
        "schema": "oph.echosahedral_federation.quotient_visible_contract.v1",
        "federation_id": federation.federation_id,
        "cardinality_semantics": "finite_carrier_regulator_count_only",
        "carriers": sorted(carrier_rows, key=lambda row: row["carrier_id"]),
        "seams": sorted(seam_rows, key=lambda row: row["seam_id"]),
        "external_boundaries": sorted(
            boundary_rows, key=lambda row: row["boundary_id"]
        ),
        "observer_supports": sorted(
            observer_rows, key=lambda row: row["observer_token"]
        ),
    }


def _recursive_mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_recursive_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(_recursive_mapping_keys(child))
    return keys


def _walk_payload_values(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    rows = [(path, value)]
    if isinstance(value, Mapping):
        for key, child in value.items():
            rows.extend(_walk_payload_values(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            rows.extend(_walk_payload_values(child, f"{path}[{index}]"))
    return rows


def _find_hidden_coordinate_values(
    payload: Mapping[str, Any], carriers: Sequence[EchosahedralCarrier]
) -> list[str]:
    reference_signatures = {
        _coordinate_set_signature(np.asarray(carrier.port_coordinates, dtype=float))
        for carrier in carriers
    }
    leaks: list[str] = []
    for path, value in _walk_payload_values(payload):
        if isinstance(value, (str, bytes, Mapping)):
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            array.shape == (_PORT_COUNT, 3)
            and _coordinate_set_signature(array) in reference_signatures
        ):
            leaks.append(path)
    return sorted(set(leaks))


def _coordinate_set_signature(coordinates: np.ndarray) -> tuple[tuple[float, ...], ...]:
    rounded = np.round(np.asarray(coordinates, dtype=float), decimals=12)
    return tuple(sorted(tuple(float(value) for value in row) for row in rounded))


def _find_hidden_port_label_values(
    payload: Mapping[str, Any], carriers: Sequence[EchosahedralCarrier]
) -> list[str]:
    label_sets = {frozenset(carrier.port_names) for carrier in carriers}
    leaks: list[str] = []
    for path, value in _walk_payload_values(payload):
        if not isinstance(value, (list, tuple)):
            continue
        if value and all(isinstance(item, str) for item in value):
            if frozenset(value) in label_sets:
                leaks.append(path)
    return sorted(set(leaks))


def _seam_from_bundle_row(row: Mapping[str, Any]) -> SeamBundle:
    binding = row["interface_algebra"]
    if not isinstance(binding, Mapping):
        raise TypeError("interface_algebra must be an object")
    return SeamBundle(
        seam_id=str(row["seam_id"]),
        left_carrier_id=str(row["left_carrier_id"]),
        right_carrier_id=str(row["right_carrier_id"]),
        left_ports=tuple(int(value) for value in row["left_ports"]),
        right_ports=tuple(int(value) for value in row["right_ports"]),
        left_to_right_ports=tuple(int(value) for value in row["left_to_right_ports"]),
        right_to_left_ports=tuple(int(value) for value in row["right_to_left_ports"]),
        left_to_right_orientation=tuple(
            int(value) for value in row["left_to_right_orientation"]
        ),
        right_to_left_orientation=tuple(
            int(value) for value in row["right_to_left_orientation"]
        ),
        collar_kind=str(row["collar_kind"]),
        interface_algebra=InterfaceAlgebraBinding(
            interface_algebra_id=str(binding["interface_algebra_id"]),
            interface_algebra_sha256=str(binding["interface_algebra_sha256"]),
            left_interface_algebra_sha256=str(binding["left_interface_algebra_sha256"]),
            right_interface_algebra_sha256=str(
                binding["right_interface_algebra_sha256"]
            ),
        ),
    )


def _boundary_from_bundle_row(row: Mapping[str, Any]) -> ExternalBoundaryBundle:
    return ExternalBoundaryBundle(
        boundary_id=str(row["boundary_id"]),
        carrier_id=str(row["carrier_id"]),
        ports=tuple(int(value) for value in row["ports"]),
        boundary_condition=str(row["boundary_condition"]),
        boundary_algebra_sha256=str(row["boundary_algebra_sha256"]),
    )


def _observer_from_bundle_row(row: Mapping[str, Any]) -> ObserverSupport:
    return ObserverSupport(
        observer_token=str(row["observer_token"]),
        carrier_ids=frozenset(str(value) for value in row["carrier_ids"]),
        visible_seam_ids=frozenset(str(value) for value in row["visible_seam_ids"]),
        record_algebra_sha256=str(row["record_algebra_sha256"]),
        checkpoint_cut_sha256=str(row["checkpoint_cut_sha256"]),
    )


def _is_sha256(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value)))


if __name__ == "__main__":
    raise SystemExit(main())
