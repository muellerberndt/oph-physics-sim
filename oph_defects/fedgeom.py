"""Compact geometry of the exact federation for universe-scale runs.

The federation is the one of ``oph_exact.federation.build_federation(level,
'port_pair')``: ``N = 20 * 4^level`` twelve-port carriers on the cells of the
geodesic icosahedral tower, three ports of every carrier glued to a
neighbour. This module calls the same two defining functions that the pinned
builder calls (``geodesic_icosahedral_patch_arrays`` and
``assign_echosahedral_ports``) and keeps only the arrays that define the
seams. It skips the adjacency matrix, Laplacian and port graph of the pinned
builder, which are derived objects and cost tens of gigabytes at level 10.
``verify_against_pinned`` asserts seam-for-seam equality with the pinned
builder; the tests run it at small levels.

Index conventions (identical to ``oph_exact.federation``):

* port ``p`` of carrier ``c`` is global port ``12 c + p``;
* seam ``s < 30 N`` is template seam ``s % 30`` of carrier ``s // 30`` with
  ends ``(12 c + TA[k], 12 c + TB[k])``, ``TA < TB`` in the order of
  ``oph_exact.carrier.seams()``;
* seam ``30 N + g`` is glue seam ``g`` with ends ``(glue_a[g], glue_b[g])``
  in dual-edge order (left end first);
* face ``20 c + f`` is template face ``f`` of carrier ``c`` (internal tier);
* face ``20 N + v`` is the screen plaquette around vertex ``v`` of the tower
  mesh (screen tier), bounded by the glue seams of the carriers around ``v``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

import numpy as np

from oph_exact import carrier as C
from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower, geodesic_icosahedral_patch_arrays
from oph_fpe.core.screen_ports import assign_echosahedral_ports

PORTS = 12
PHI = (1.0 + 5.0 ** 0.5) / 2.0
# the twelve port vectors u_p, squared radius phi + 2, in port order
PORT_VECTORS = np.array([(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0), (0, -1, PHI), (0, 1, PHI),
                         (0, -1, -PHI), (0, 1, -PHI), (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)], float)
PORT_UNIT = PORT_VECTORS / np.linalg.norm(PORT_VECTORS, axis=1, keepdims=True)
TEMPLATE_SEAMS = np.asarray(C.seams(), dtype=np.int64)            # (30, 2), a < b
TEMPLATE_FACES = np.asarray(C.oriented_faces(), dtype=np.int64)   # (20, 3), oriented


def _template_face_sides() -> tuple[np.ndarray, np.ndarray]:
    """For each template seam (a, b), a < b: the face that traverses it a->b and the one that traverses b->a."""
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(TEMPLATE_SEAMS)}
    plus = np.full(30, -1, np.int64)
    minus = np.full(30, -1, np.int64)
    for f, (p, q, r) in enumerate(TEMPLATE_FACES):
        for u, v in ((p, q), (q, r), (r, p)):
            k = key[(min(u, v), max(u, v))]
            if u < v:
                assert plus[k] < 0
                plus[k] = f
            else:
                assert minus[k] < 0
                minus[k] = f
    assert (plus >= 0).all() and (minus >= 0).all()
    return plus, minus


TEMPLATE_PLUS, TEMPLATE_MINUS = _template_face_sides()


@dataclass
class Geometry:
    level: int
    carriers: int
    vertices: int                 # mesh vertices = screen plaquettes
    glue_a: np.ndarray            # int32 [G] global port of the left end
    glue_b: np.ndarray            # int32 [G] global port of the right end
    glue_fp: np.ndarray           # int32 [G] global face id (20 N + v) that traverses the seam a->b
    glue_fm: np.ndarray           # int32 [G] global face id that traverses it b->a
    tri: np.ndarray               # int32 [N, 3] mesh vertices of each carrier's cell
    mesh_xyz: np.ndarray          # float64 [V, 3] unit vectors
    carrier_xyz: np.ndarray       # float64 [N, 3] unit vectors (cell centres as the pinned builder uses them)
    source_sha256: str = ''
    gluing: str = 'port_pair'
    _nbr: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ counts
    @property
    def ports(self) -> int: return PORTS * self.carriers
    @property
    def intra_seams(self) -> int: return 30 * self.carriers
    @property
    def glue_seams(self) -> int: return int(self.glue_a.size)
    @property
    def seams(self) -> int: return self.intra_seams + self.glue_seams
    @property
    def intra_faces(self) -> int: return 20 * self.carriers
    @property
    def faces(self) -> int: return self.intra_faces + self.vertices

    # ------------------------------------------------------------------ seams and faces
    def seam_ends(self, seams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(seams, dtype=np.int64)
        intra = s < self.intra_seams
        c, k = np.divmod(np.where(intra, s, 0), 30)
        g = np.where(intra, 0, s - self.intra_seams)
        a = np.where(intra, 12 * c + TEMPLATE_SEAMS[k, 0], self.glue_a[g])
        b = np.where(intra, 12 * c + TEMPLATE_SEAMS[k, 1], self.glue_b[g])
        return a, b

    def seam_faces(self, seams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(seams, dtype=np.int64)
        intra = s < self.intra_seams
        c, k = np.divmod(np.where(intra, s, 0), 30)
        g = np.where(intra, 0, s - self.intra_seams)
        fp = np.where(intra, 20 * c + TEMPLATE_PLUS[k], self.glue_fp[g])
        fm = np.where(intra, 20 * c + TEMPLATE_MINUS[k], self.glue_fm[g])
        return fp, fm

    def all_seam_ends(self) -> tuple[np.ndarray, np.ndarray]:
        return self.seam_ends(np.arange(self.seams, dtype=np.int64))

    def face_classes(self, register: np.ndarray, modulus: int = 6) -> np.ndarray:
        """Loop sums of a seam register on every face of both tiers."""
        fp, fm = self.seam_faces(np.arange(self.seams, dtype=np.int64))
        cls = np.zeros(self.faces, np.int64)
        np.add.at(cls, fp, register.astype(np.int64))
        np.add.at(cls, fm, -register.astype(np.int64))
        return (cls % modulus).astype(np.int8)

    def face_curl(self, offsets: np.ndarray) -> np.ndarray:
        """Signed integer loop sums of seam offsets on every face (the holonomy of the offsets around it)."""
        fp, fm = self.seam_faces(np.arange(self.seams, dtype=np.int64))
        curl = np.zeros(self.faces, np.int64)
        np.add.at(curl, fp, offsets.astype(np.int64))
        np.add.at(curl, fm, -offsets.astype(np.int64))
        return curl

    # ------------------------------------------------------------------ frames and positions
    def frames(self, carriers: np.ndarray) -> np.ndarray:
        """Local frames [n, 3, 3] with columns (tangent_x, tangent_y, normal), the routing convention."""
        normals = self.carrier_xyz[np.asarray(carriers, dtype=np.int64)].astype(float)
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        reference = np.tile(np.array([0.0, 0.0, 1.0]), (normals.shape[0], 1))
        reference[np.abs(normals[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
        tx = np.cross(reference, normals)
        tx /= np.maximum(np.linalg.norm(tx, axis=1, keepdims=True), 1e-15)
        ty = np.cross(normals, tx)
        return np.stack((tx, ty, normals), axis=2)

    def cell_spacing(self) -> float:
        """Mean angular distance between adjacent cell centres."""
        nb = self.neighbours()
        k = min(self.carriers, 4096)
        idx = np.linspace(0, self.carriers - 1, k).astype(np.int64)
        d = np.linalg.norm(self.carrier_xyz[idx][:, None, :] - self.carrier_xyz[nb[idx]], axis=2)
        return float(d.mean())

    def face_xyz(self, faces: np.ndarray, spread: float = 0.22) -> np.ndarray:
        """Display positions (unit vectors) of faces. Internal faces sit in a small cluster around their carrier."""
        f = np.asarray(faces, dtype=np.int64)
        out = np.empty((f.size, 3), np.float32)
        intra = f < self.intra_faces
        if intra.any():
            c, t = np.divmod(f[intra], 20)
            fr = self.frames(c)
            tc = PORT_UNIT[TEMPLATE_FACES].mean(axis=1)
            tc /= np.linalg.norm(tc, axis=1, keepdims=True)
            off = spread * self.cell_spacing()
            p = self.carrier_xyz[c] + off * (tc[t][:, 0:1] * fr[:, :, 0] + tc[t][:, 1:2] * fr[:, :, 1])
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            out[intra] = p
        if (~intra).any():
            out[~intra] = self.mesh_xyz[f[~intra] - self.intra_faces]
        return out

    # ------------------------------------------------------------------ carrier graph
    def neighbours(self) -> np.ndarray:
        """int32 [N, 3]: the three carriers glued to each carrier, in glue-seam order."""
        if self._nbr is None:
            left = (self.glue_a // 12).astype(np.int64)
            right = (self.glue_b // 12).astype(np.int64)
            ends = np.concatenate([left, right])
            other = np.concatenate([right, left])
            order = np.argsort(ends, kind='stable')
            nb = other[order].reshape(self.carriers, 3)
            self._nbr = nb.astype(np.int32)
        return self._nbr

    def carrier_glue_seams(self) -> np.ndarray:
        """int32 [N, 3]: the glue-seam indices (0..G-1) of each carrier, same order as ``neighbours``."""
        left = (self.glue_a // 12).astype(np.int64)
        right = (self.glue_b // 12).astype(np.int64)
        ends = np.concatenate([left, right])
        which = np.concatenate([np.arange(self.glue_seams), np.arange(self.glue_seams)])
        order = np.argsort(ends, kind='stable')
        return which[order].reshape(self.carriers, 3).astype(np.int32)

    def ball(self, centre: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
        """Carriers within ``radius`` glue steps of ``centre`` and their distances, in breadth-first order."""
        nb = self.neighbours()
        dist = {int(centre): 0}
        order = [int(centre)]
        frontier = [int(centre)]
        for r in range(1, radius + 1):
            nxt = []
            for c in frontier:
                for w in nb[c].tolist():
                    if w not in dist:
                        dist[w] = r
                        nxt.append(w)
            order.extend(nxt)
            frontier = nxt
            if not frontier:
                break
        cells = np.array(order, dtype=np.int64)
        return cells, np.array([dist[int(c)] for c in cells], dtype=np.int32)

    def carrier_incident_seams(self, carrier: int) -> np.ndarray:
        """The 33 seams incident to a carrier: its 30 template seams, then its 3 glue seams."""
        own = 30 * carrier + np.arange(30, dtype=np.int64)
        glue = self.intra_seams + self.carrier_glue_seams()[carrier].astype(np.int64)
        return np.concatenate([own, glue])


def _source_hash() -> str:
    import oph_fpe.core.icosahedral as ico
    import oph_fpe.core.screen_ports as sp
    h = hashlib.sha256()
    for mod in (ico, sp, C):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()


def _cache_dir() -> Path:
    d = Path(os.environ.get('OPH_UNIVERSE_CACHE', Path.home() / '.cache' / 'oph_universe'))
    d.mkdir(parents=True, exist_ok=True)
    return d


# control gluing: the belt port across the template edge opposite the k-th vertex of template face (0, 11, 5)
CONTROL_FACE = (0, 11, 5)


def _control_ports() -> tuple[int, int, int]:
    t = CONTROL_FACE
    out = []
    for k in range(3):
        e = {t[(k + 1) % 3], t[(k + 2) % 3]}
        third = [int(v) for f in TEMPLATE_FACES.tolist() if e <= set(f) and set(f) != set(t) for v in f if v not in e]
        assert len(third) == 1
        out.append(third[0])
    return tuple(out)


def build(level: int, cache: bool = True, gluing: str = 'port_pair') -> Geometry:
    """Build the federation geometry (cached on disk per source hash).

    ``port_pair`` is the declared gluing of the model, built with the pinned defining functions.
    ``face_axis`` is a control with no global axis: every carrier puts the three-fold axis of template face
    (0, 11, 5) along its cell normal, and the mesh edge opposite the k-th vertex of the cell is glued through
    the belt port across the template edge opposite the k-th vertex of that face. It keeps the rotation
    group of the icosahedral mesh. It is not the model's gluing.
    """
    if gluing not in ('port_pair', 'face_axis'):
        raise ValueError(f'unknown gluing {gluing!r}')
    src = _source_hash()
    path = _cache_dir() / (f'fedgeom_L{level}_{src[:16]}.npz' if gluing == 'port_pair' else f'fedgeom_L{level}_{gluing}_{src[:16]}.npz')
    if cache and path.exists():
        z = np.load(path)
        return Geometry(level, int(z['carriers']), int(z['vertices']), z['glue_a'], z['glue_b'], z['glue_fp'], z['glue_fm'],
                        z['tri'], z['mesh_xyz'], z['carrier_xyz'], src, gluing)
    points, left, right = geodesic_icosahedral_patch_arrays(level, patch_basis='cells')
    N = int(points.shape[0])
    if N != 20 * 4 ** level:
        raise AssertionError('cell count differs from 20 * 4^L')
    port_map = assign_echosahedral_ports(left, right, N, points=points)
    if port_map.routing_mode != 'icosahedral_directional_assignment':
        raise AssertionError('production geometric routing was not selected')
    if port_map.overflow_count != 0:
        raise AssertionError('production routing overflowed a twelve-slot carrier')
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    lp = port_map.left_port.astype(np.int64)
    rp = port_map.right_port.astype(np.int64)
    if gluing == 'face_axis':
        tri0 = np.asarray(build_geodesic_icosahedral_tower(level).levels[level].faces, dtype=np.int64)
        ports = np.array(_control_ports(), dtype=np.int64)
        def edge_port(cells, others):
            # the vertex of the cell that the neighbour does not share is opposite the common edge
            shared = (tri0[cells][:, :, None] == tri0[others][:, None, :]).any(axis=2)
            assert (shared.sum(axis=1) == 2).all()
            return ports[np.argmin(shared, axis=1)]
        lp, rp = edge_port(left, right), edge_port(right, left)
    slots = np.concatenate([left * PORTS + lp, right * PORTS + rp])
    if np.unique(slots).size != slots.size:
        raise AssertionError('a carrier port is glued more than once')
    glue_a = left * PORTS + lp
    glue_b = right * PORTS + rp
    mesh = build_geodesic_icosahedral_tower(level).levels[level]
    tri = np.asarray(mesh.faces, dtype=np.int64)
    V = int(mesh.vertex_count)
    if tri.shape[0] != N:
        raise AssertionError('mesh faces and cells differ')
    # screen plaquettes: the two mesh vertices shared by adjacent cells
    tl, tr = tri[left], tri[right]
    common = np.stack([np.where((tl[:, k][:, None] == tr).any(axis=1), tl[:, k], -1) for k in range(3)], axis=1)
    two = -np.sort(-common, axis=1)[:, :2]
    if not (two >= 0).all():
        raise AssertionError('adjacent carriers do not share one mesh edge')
    v_a, v_b = two[:, 0], two[:, 1]
    rows = np.arange(left.size)

    def successor_is_right(v, w):
        pos = np.argmax(tl == v[:, None], axis=1)
        return tl[rows, (pos + 2) % 3] == w
    plus_is_a = successor_is_right(v_a, v_b)
    plus_is_b = successor_is_right(v_b, v_a)
    if not (plus_is_a ^ plus_is_b).all():
        raise AssertionError('inconsistent orientation around a shared edge')
    glue_fp = 20 * N + np.where(plus_is_a, v_a, v_b)
    glue_fm = 20 * N + np.where(plus_is_a, v_b, v_a)
    vxyz = np.asarray(mesh.vertices, dtype=np.float64)
    vxyz = vxyz / np.linalg.norm(vxyz, axis=1, keepdims=True)
    geo = Geometry(level, N, V, glue_a.astype(np.int32), glue_b.astype(np.int32), glue_fp.astype(np.int32),
                   glue_fm.astype(np.int32), tri.astype(np.int32), vxyz, np.asarray(points, dtype=np.float64), src, gluing)
    if cache:
        tmp = path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(tmp, carriers=N, vertices=V, glue_a=geo.glue_a, glue_b=geo.glue_b, glue_fp=geo.glue_fp, glue_fm=geo.glue_fm,
                 tri=geo.tri, mesh_xyz=geo.mesh_xyz, carrier_xyz=geo.carrier_xyz)
        os.replace(tmp, path)
    return geo


def verify_against_pinned(geo: Geometry) -> dict:
    """Assert seam-for-seam equality with ``oph_exact.federation.build_federation`` (use at small levels)."""
    from oph_exact import federation as F
    fed = F.build_federation(geo.level, 'port_pair')
    a, b = geo.all_seam_ends()
    assert fed.carriers == geo.carriers and fed.seams == geo.seams and fed.intra_count == geo.intra_seams
    assert (fed.seam_a == a).all() and (fed.seam_b == b).all(), 'seam ends differ from the pinned builder'
    assert tuple(map(tuple, C.oriented_faces())) == tuple(map(tuple, TEMPLATE_FACES))
    return {'level': geo.level, 'seams': int(geo.seams), 'equal_to_pinned_builder': True,
            'port_pairs_sha256': fed.gluing_receipt.get('port_pairs_sha256')}
