from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import networkx as nx
import numpy as np

from oph_fpe.claims import (
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
    SHAPE_DODECA_CELL_RECEIPT,
    SHAPE_LOOP_MODE_RECEIPT,
)
from oph_fpe.microphysics.shape_constants import PHI


@dataclass(frozen=True)
class DodecaCell:
    graph: nx.Graph
    faces: list[list[int]]
    vertex_coordinates: dict[int, np.ndarray]


def dodecahedral_cell() -> DodecaCell:
    graph = nx.dodecahedral_graph()
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise RuntimeError("networkx dodecahedral graph unexpectedly non-planar")

    seen: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for u, v in embedding.edges():
        edge = (int(u), int(v))
        if edge in seen:
            continue
        face = [int(node) for node in embedding.traverse_face(u, v)]
        faces.append(face)
        for left, right in zip(face, face[1:] + face[:1]):
            seen.add((int(left), int(right)))

    return DodecaCell(
        graph=graph,
        faces=faces,
        vertex_coordinates=_dodecahedron_coordinates_for_graph(graph),
    )


def dodeca_cell_report() -> dict[str, Any]:
    cell = dodecahedral_cell()
    degrees = [int(degree) for _, degree in cell.graph.degree()]
    face_lengths = [len(face) for face in cell.faces]
    checks = {
        "vertex_count_20": cell.graph.number_of_nodes() == 20,
        "edge_count_30": cell.graph.number_of_edges() == 30,
        "face_count_12": len(cell.faces) == 12,
        "all_vertices_degree_3": all(degree == 3 for degree in degrees),
        "all_faces_pentagons": all(length == 5 for length in face_lengths),
        "euler_characteristic_2": (
            cell.graph.number_of_nodes() - cell.graph.number_of_edges() + len(cell.faces) == 2
        ),
        "golden_ratio_closure": abs(math.cos(math.pi / 5.0) - PHI / 2.0) < 1.0e-12,
        "lowest_pentagon_mode_kL": abs(5.0 * (2.0 * math.pi / 5.0) - 2.0 * math.pi) < 1.0e-12,
    }
    return {
        "receipt": SHAPE_DODECA_CELL_RECEIPT,
        "receipt_name": SHAPE_DODECA_CELL_RECEIPT,
        "checks": checks,
        "passed": all(checks.values()),
        "node_count": cell.graph.number_of_nodes(),
        "edge_count": cell.graph.number_of_edges(),
        "face_count": len(cell.faces),
        "face_lengths": face_lengths,
        "phi": PHI,
        "cos_pi_over_5": math.cos(math.pi / 5.0),
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "physical_cmb_prediction": False,
    }


def pentagon_loop_modes(cell: DodecaCell, max_mode: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for face_id, face in enumerate(cell.faces):
        loop_length = len(face)
        for mode in range(1, int(max_mode) + 1):
            k_l = 2.0 * math.pi * mode / loop_length
            rows.append(
                {
                    "face_id": int(face_id),
                    "loop_length": int(loop_length),
                    "mode": int(mode),
                    "kL": float(k_l),
                    "closure_phase": float(loop_length * k_l),
                    "closure_residual": float(abs(np.exp(1j * loop_length * k_l) - 1.0)),
                }
            )
    return rows


def loop_mode_report(cell: DodecaCell | None = None, max_mode: int = 5) -> dict[str, Any]:
    cell = cell or dodecahedral_cell()
    rows = pentagon_loop_modes(cell, max_mode=max_mode)
    fundamental = [row for row in rows if row["mode"] == 1]
    return {
        "receipt": SHAPE_LOOP_MODE_RECEIPT,
        "receipt_name": SHAPE_LOOP_MODE_RECEIPT,
        "passed": bool(fundamental and all(row["closure_residual"] < 1.0e-12 for row in fundamental)),
        "loop_mode_count": len(rows),
        "fundamental_loop_count": len(fundamental),
        "rows": rows,
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "physical_cmb_prediction": False,
    }


def dodecahedron_face_normals(cell: DodecaCell | None = None) -> np.ndarray:
    cell = cell or dodecahedral_cell()
    rows: list[np.ndarray] = []
    for face in cell.faces:
        centroid = np.mean([cell.vertex_coordinates[int(node)] for node in face], axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1.0e-12:
            rows.append(np.array([0.0, 0.0, 1.0], dtype=float))
        else:
            rows.append(centroid / norm)
    return np.asarray(rows, dtype=float)


def _dodecahedron_coordinates_for_graph(graph: nx.Graph) -> dict[int, np.ndarray]:
    coords = _canonical_dodecahedron_coordinates()
    coord_graph = nx.Graph()
    coord_graph.add_nodes_from(range(coords.shape[0]))
    distances = []
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            distances.append(float(np.linalg.norm(coords[i] - coords[j])))
    edge_length = min(value for value in distances if value > 1.0e-9)
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            if abs(float(np.linalg.norm(coords[i] - coords[j])) - edge_length) < 1.0e-9:
                coord_graph.add_edge(i, j)
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, coord_graph)
    if not matcher.is_isomorphic():
        raise RuntimeError("failed to map networkx dodecahedral graph to canonical coordinates")
    mapping = next(matcher.isomorphisms_iter())
    return {int(node): coords[int(coord_node)].astype(float) for node, coord_node in mapping.items()}


def _canonical_dodecahedron_coordinates() -> np.ndarray:
    inv_phi = 1.0 / PHI
    rows: list[tuple[float, float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rows.append((sx, sy, sz))
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            rows.append((0.0, sy * inv_phi, sz * PHI))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            rows.append((sx * inv_phi, sy * PHI, 0.0))
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            rows.append((sx * PHI, 0.0, sz * inv_phi))
    return np.asarray(rows, dtype=float)
