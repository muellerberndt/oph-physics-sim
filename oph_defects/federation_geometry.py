"""Faces of the exact federation: twenty per carrier and one per screen vertex.

The exact federation (``oph_exact.federation``) glues twelve-port carriers
on the cells of the geodesic tower. Two closed oriented surfaces carry loop
sums of a seam register:

* the twenty template faces inside every carrier (intra tier), bounded by
  intra seams only;
* the dual faces of the tiling, one around every vertex of the tower mesh
  (macro tier), bounded by inter-carrier seams only, in the counterclockwise
  order of the outward-oriented mesh.

Every seam borders exactly two faces of its tier, once in each direction,
so a transfer across a seam changes the two face classes by ``+delta`` and
``-delta``: births, moves, fusions and annihilations keep each tier's total
class at zero mod 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oph_exact import carrier as C
from oph_exact import federation as F
from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower

BASE_FACES = np.array(C.base_faces(), dtype=np.int64) if hasattr(C, 'base_faces') else None


@dataclass(frozen=True)
class FederationFaces:
    level: int
    carriers: int
    vertices: int           # screen vertices = macro faces
    seams: int
    intra_seams: int
    face_count: int         # 20 * carriers + vertices
    seam_plus: np.ndarray   # (seams,) int64 face traversing the seam a->b
    seam_minus: np.ndarray  # (seams,) int64 face traversing it b->a
    seam_carrier: np.ndarray  # (seams,) int64 owning carrier (intra) or left carrier (inter)
    face_xyz: np.ndarray    # (face_count, 3) float32 unit-sphere positions for display
    mesh_vertex_xyz: np.ndarray
    carrier_xyz: np.ndarray


def _template_faces() -> np.ndarray:
    from oph_fpe.dynamics.self_readback_repair_closure import ORIENTED_BASE_FACES
    return np.array(ORIENTED_BASE_FACES, dtype=np.int64)


def build(level: int) -> tuple[F.ExactFederation, FederationFaces]:
    fed = F.build_federation(level, 'port_pair')
    mesh = build_geodesic_icosahedral_tower(level).levels[level]
    N, V = fed.carriers, mesh.vertex_count
    tri = np.asarray(mesh.faces, dtype=np.int64)
    assert tri.shape[0] == N
    # ---- intra tier: template faces and template seams
    tf = _template_faces()                       # (20, 3) oriented
    ts = np.asarray(C.seams(), dtype=np.int64)   # (30, 2) sorted pairs, template order
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(ts)}
    plus_t = np.full(30, -1, np.int64)
    minus_t = np.full(30, -1, np.int64)
    for f, (p, q, r) in enumerate(tf):
        for u, v in ((p, q), (q, r), (r, p)):
            k = key[(min(u, v), max(u, v))]
            if u < v:
                assert plus_t[k] < 0
                plus_t[k] = f
            else:
                assert minus_t[k] < 0
                minus_t[k] = f
    assert (plus_t >= 0).all() and (minus_t >= 0).all()
    nI = fed.intra_count
    assert nI == 30 * N
    c = np.arange(N, dtype=np.int64)
    seam_plus_intra = (c[:, None] * 20 + plus_t[None, :]).ravel()
    seam_minus_intra = (c[:, None] * 20 + minus_t[None, :]).ravel()
    carrier_intra = np.repeat(c, 30)
    # ---- macro tier: dual faces around mesh vertices, bounded by inter seams
    ip = fed.inter_pairs                         # rows: left, lp, right, rp
    left, right = ip[:, 0], ip[:, 2]
    tl, tr = tri[left], tri[right]
    # the two shared vertices of the adjacent triangles
    shared = np.zeros((len(left), 2), np.int64)
    for k in range(3):
        m = (tl[:, k][:, None] == tr).any(axis=1)
        # fill the first free slot
        slot0 = shared[:, 0] == 0
        pass
    # robust shared-vertex extraction
    common = []
    for k in range(3):
        common.append(np.where((tl[:, k][:, None] == tr).any(axis=1), tl[:, k], -1))
    common = np.stack(common, axis=1)
    common_sorted = -np.sort(-common, axis=1)[:, :2]   # two largest = the two shared (others -1)
    assert (common_sorted >= 0).all(), 'adjacent carriers share one mesh edge'
    v_a, v_b = common_sorted[:, 0], common_sorted[:, 1]
    # orientation: around vertex v, counterclockwise order goes left -> right iff
    # in triangle `left` the vertex after v (cyclically) is NOT w, i.e. w is two steps after v.
    def successor_is_right(v, w):
        pos = np.argmax(tl == v[:, None], axis=1)
        after2 = tl[np.arange(len(left)), (pos + 2) % 3]
        return after2 == w
    plus_is_a = successor_is_right(v_a, v_b)
    plus_is_b = successor_is_right(v_b, v_a)
    assert (plus_is_a ^ plus_is_b).all(), 'inconsistent orientation around the shared edge'
    seam_plus_inter = 20 * N + np.where(plus_is_a, v_a, v_b)
    seam_minus_inter = 20 * N + np.where(plus_is_a, v_b, v_a)
    seam_plus = np.concatenate([seam_plus_intra, seam_plus_inter])
    seam_minus = np.concatenate([seam_minus_intra, seam_minus_inter])
    seam_carrier = np.concatenate([carrier_intra, left.astype(np.int64)])
    assert seam_plus.size == fed.seams
    # ---- display positions
    vxyz = np.asarray(mesh.vertices, dtype=np.float64)
    cxyz = vxyz[tri].mean(axis=1)
    cxyz /= np.linalg.norm(cxyz, axis=1, keepdims=True)
    # intra faces: carrier centre pushed toward the template face direction in the
    # carrier's local frame (a small cluster of twenty around the centre)
    tv = np.asarray([v['coordinates_display'] for v in []] or _template_vertices())
    tcent = tv[tf].mean(axis=1)
    tcent /= np.linalg.norm(tcent, axis=1, keepdims=True)
    spacing = np.linalg.norm(vxyz[tri[:, 0]] - vxyz[tri[:, 1]], axis=1).mean()
    # local frame per carrier: east/north on the tangent plane
    ref = np.where(np.abs(cxyz[:, 2:3]) > 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    east = np.cross(ref, cxyz); east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(cxyz, east)
    off = 0.22 * spacing
    intra_xyz = (cxyz[:, None, :] + off * (tcent[None, :, 0:1] * east[:, None, :] + tcent[None, :, 1:2] * north[:, None, :]
                                            + 0.0 * tcent[None, :, 2:3] * cxyz[:, None, :]))
    intra_xyz = intra_xyz.reshape(-1, 3)
    intra_xyz /= np.linalg.norm(intra_xyz, axis=1, keepdims=True)
    face_xyz = np.concatenate([intra_xyz, vxyz / np.linalg.norm(vxyz, axis=1, keepdims=True)]).astype(np.float32)
    faces = FederationFaces(level, N, V, fed.seams, nI, 20 * N + V, seam_plus, seam_minus, seam_carrier,
                            face_xyz, vxyz.astype(np.float32), cxyz.astype(np.float32))
    return fed, faces


def _template_vertices() -> np.ndarray:
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi),
                  (0, -1, -phi), (0, 1, -phi), (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)], float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def face_classes(faces: FederationFaces, register: np.ndarray, modulus: int = 6) -> np.ndarray:
    """Loop sums of the seam register on every face of both tiers."""
    cls = np.zeros(faces.face_count, np.int64)
    np.add.at(cls, faces.seam_plus, register.astype(np.int64))
    np.add.at(cls, faces.seam_minus, -register.astype(np.int64))
    return (cls % modulus).astype(np.int8)
