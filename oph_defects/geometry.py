"""The icosahedral carrier refined ``level`` times: an oriented closed surface.

Level 0 is the certified twelve-port carrier (12 observers, 30 seams, 20
faces) with the oriented base faces of the exact carrier. Each refinement
splits every face into four by its edge midpoints, which is the refinement
step the Z6 flux certificate checks once (42, 120, 80). At level ``L`` the
surface has ``10*4^L + 2`` observers, ``30*4^L`` seams and ``20*4^L`` faces.

Everything is vectorized and deterministic. A face is addressed as
``f = b*n*n + local`` with ``n = 2**level``; the base face ``b`` and the
local grid cell are recovered by :func:`decode_faces`, so a viewer can place
any face from its index alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHI = (1 + 5 ** 0.5) / 2

# Oriented base faces of the exact carrier (oph_exact.carrier / ORIENTED_BASE_FACES).
BASE_FACES = np.array([
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], dtype=np.int64)

BASE_VERTICES = np.array([
    (-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0),
    (0, -1, PHI), (0, 1, PHI), (0, -1, -PHI), (0, 1, -PHI),
    (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)], dtype=np.float64)
BASE_VERTICES /= np.linalg.norm(BASE_VERTICES, axis=1, keepdims=True)


def base_edges() -> np.ndarray:
    """The thirty base seams as sorted pairs, in lexicographic order."""
    pairs = set()
    for a, b, c in BASE_FACES:
        for u, v in ((a, b), (b, c), (c, a)):
            pairs.add((min(u, v), max(u, v)))
    return np.array(sorted(pairs), dtype=np.int64)


@dataclass(frozen=True)
class Surface:
    level: int
    n: int
    vertex_count: int
    edge_count: int
    face_count: int
    face_edges: np.ndarray    # (F, 3) int32 edge ids in traversal order
    face_signs: np.ndarray    # (F, 3) int8: +1 if traversed low->high vertex
    edge_plus: np.ndarray     # (E,) int32 face that traverses the edge low->high
    edge_minus: np.ndarray    # (E,) int32 face that traverses the edge high->low
    face_vertices: np.ndarray  # (F, 3) int32 global vertex ids in traversal order

    @property
    def euler_characteristic(self) -> int:
        return self.vertex_count - self.edge_count + self.face_count


def decode_faces(face: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (base face, i, j, up) for face indices."""
    n = 1 << level
    face = np.asarray(face, dtype=np.int64)
    b, local = np.divmod(face, n * n)
    p, q = np.divmod(local, n)
    up = p + q <= n - 1
    i = np.where(up, p, n - 1 - p)
    j = np.where(up, q, n - 1 - q)
    return b, i, j, up


def encode_faces(b: np.ndarray, i: np.ndarray, j: np.ndarray, up: np.ndarray, level: int) -> np.ndarray:
    n = 1 << level
    p = np.where(up, i, n - 1 - i)
    q = np.where(up, j, n - 1 - j)
    return (b * n * n + p * n + q).astype(np.int64)


def vertex_ids(b: np.ndarray, i: np.ndarray, j: np.ndarray, level: int) -> np.ndarray:
    """Global observer id of grid point (i, j) of base face b; shared points agree."""
    n = 1 << level
    faces = BASE_FACES
    edges = base_edges()
    edge_index = {(int(u), int(v)): k for k, (u, v) in enumerate(edges)}
    A, B, C = faces[b, 0], faces[b, 1], faces[b, 2]
    out = np.full(b.shape, -1, dtype=np.int64)
    # corners
    out = np.where((i == 0) & (j == 0), A, out)
    out = np.where((i == n) & (j == 0), B, out)
    out = np.where((i == 0) & (j == n), C, out)
    per_edge = n - 1

    def on_edge(mask, u, v, t_from_u):
        nonlocal out
        if not mask.any():
            return
        uu, vv, tt = u[mask], v[mask], t_from_u[mask]
        lo = np.minimum(uu, vv)
        hi = np.maximum(uu, vv)
        t = np.where(uu < vv, tt, n - tt)
        k = np.array([edge_index[(int(x), int(y))] for x, y in zip(lo, hi)], dtype=np.int64)
        out[mask] = 12 + k * per_edge + (t - 1)

    on_edge((j == 0) & (i > 0) & (i < n), A, B, i)
    on_edge((i + j == n) & (i > 0) & (i < n), B, C, j)
    on_edge((i == 0) & (j > 0) & (j < n), A, C, j)
    interior = (i > 0) & (j > 0) & (i + j < n)
    if interior.any():
        ii, jj = i[interior], j[interior]
        # row-major enumeration of interior points: rows i = 1..n-2, j = 1..n-1-i
        per_face = (n - 1) * (n - 2) // 2
        before = (ii - 1) * (n - 2) - (ii - 1) * (ii - 2) // 2
        out[interior] = 12 + 30 * per_edge + b[interior] * per_face + before + (jj - 1)
    assert (out >= 0).all()
    return out


def build(level: int) -> Surface:
    n = 1 << level
    F = 20 * n * n
    f = np.arange(F, dtype=np.int64)
    b, i, j, up = decode_faces(f, level)
    # corners in traversal order; up: (i,j),(i+1,j),(i,j+1); down: (i+1,j),(i+1,j+1),(i,j+1)
    ci = np.stack([np.where(up, i, i + 1), np.where(up, i + 1, i + 1), np.where(up, i, i)], axis=1)
    cj = np.stack([np.where(up, j, j), np.where(up, j, j + 1), np.where(up, j + 1, j + 1)], axis=1)
    bb = np.repeat(b[:, None], 3, axis=1)
    fv = vertex_ids(bb.ravel(), ci.ravel(), cj.ravel(), level).reshape(F, 3)
    V = int(fv.max()) + 1
    assert V == 10 * n * n + 2, (V, 10 * n * n + 2)
    a = fv
    c = np.roll(fv, -1, axis=1)
    lo = np.minimum(a, c)
    hi = np.maximum(a, c)
    key = (lo * V + hi).ravel()
    uniq, inverse = np.unique(key, return_inverse=True)
    E = uniq.shape[0]
    assert E == 30 * n * n, (E, 30 * n * n)
    face_edges = inverse.reshape(F, 3).astype(np.int32)
    face_signs = np.where(a < c, 1, -1).astype(np.int8)
    edge_plus = np.full(E, -1, dtype=np.int32)
    edge_minus = np.full(E, -1, dtype=np.int32)
    ff = np.repeat(np.arange(F, dtype=np.int32), 3)
    plus = face_signs.ravel() > 0
    edge_plus[face_edges.ravel()[plus]] = ff[plus]
    edge_minus[face_edges.ravel()[~plus]] = ff[~plus]
    assert (edge_plus >= 0).all() and (edge_minus >= 0).all(), 'surface is not consistently oriented'
    return Surface(level, n, V, E, F, face_edges, face_signs, edge_plus, edge_minus, fv.astype(np.int32))


def face_positions(face: np.ndarray, level: int) -> np.ndarray:
    """Unit-sphere centroid of each face (used by tests and exports)."""
    n = 1 << level
    b, i, j, up = decode_faces(face, level)
    A, B, C = (BASE_VERTICES[BASE_FACES[b, k]] for k in range(3))
    ci = np.where(up, i + 1 / 3, i + 2 / 3)[:, None]
    cj = np.where(up, j + 1 / 3, j + 2 / 3)[:, None]
    p = A * (1 - ci / n - cj / n) + B * (ci / n) + C * (cj / n)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def face_curvature(surface: Surface, links: np.ndarray, modulus: int = 6) -> np.ndarray:
    """Loop sum of the link register around every face, modulo ``modulus``."""
    s = surface.face_signs.astype(np.int64) * links[surface.face_edges].astype(np.int64)
    return (s.sum(axis=1) % modulus).astype(np.int8)
