"""Signal cone and incidence transport on the federation complex.

    python -m oph_defects.light cone --levels 3,5,7 --sweeps 40 --seeds 4
    python -m oph_defects.light complex --levels 1,2            # K, K B = 0 and the rank
    python -m oph_defects.light spectrum --levels 3,4,5 --gluing port_pair
    python -m oph_defects.light leapfrog --levels 5,7 --nu 0.25 --steps 160

The complex. 0-cells are the ports, 1-cells the seams (oriented from their first end a to their second end b),
2-cells the 20 faces of every carrier and one plaquette per screen vertex. B is the seam-port incidence,
(B x)_s = x_a - x_b. K is the face-seam incidence: +1 where the oriented boundary of the face runs along the seam
from a to b, -1 where it runs from b to a. The boundary of a plaquette consists of the glue seams of the carriers
around its vertex, joined inside each carrier by a shortest path of internal seams from the port where the cycle
enters to the port where it leaves (breadth-first in template order; the choice is part of the complex).
K B = 0, and rank K = M - P + 1.

cone      One port is marked. The schedule of law C is executed as a relation only: an attempt on a seam with an
          influenced end influences both ends. Reported: the equal-area radius of the influenced carriers against
          sweeps and against causal layers (the native depth of the attempt that first reached each port), the
          front speed, and the dependence of the arrival time on direction.
spectrum  Lowest nonzero eigenvalues of K K^T with unit weights, scaled by 4^level like D_l, next to the port
          Laplacian B^T B.
leapfrog  f <- f - nu K e, e <- e + nu K^T f from one excited plaquette: the support (strict cone) and the radius
          that holds half of the energy, per step.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from . import fedgeom as FG
from .closed_loop import Mesh

SQRT3 = 3.0 ** 0.5

C_SOURCE = r'''
#include <stdint.h>
/* One chunk of the schedule as a causal relation. first_t / first_layer are -1 until a port is influenced. */
void oph_influence(const int32_t *a, const int32_t *b, const int32_t *seq, int64_t n, int64_t t0,
                   int32_t *depth, int64_t *first_t, int32_t *first_layer, int64_t *count)
{
    for (int64_t t = 0; t < n; t++) {
        const int32_t s = seq[t], i = a[s], j = b[s];
        const int32_t d = 1 + (depth[i] > depth[j] ? depth[i] : depth[j]);
        depth[i] = d; depth[j] = d;
        const int fi = first_t[i] >= 0, fj = first_t[j] >= 0;
        if (fi != fj) {
            const int32_t k = fi ? j : i;
            first_t[k] = t0 + t; first_layer[k] = d; (*count)++;
        }
    }
}

/* Event-driven unrecorded transport: one event on seam e is an exact rotation by one angle of the plane spanned by
   the seam variable E[e] and the unit combination u_e of node variables (faces for the transverse sector, ports for
   the scalar sector). The total norm is conserved under every schedule. */
void oph_rotate(const int64_t *indptr, const int32_t *idx, const double *coef, const int32_t *seq, int64_t n,
                double c, double s, double *E, double *X)
{
    for (int64_t t = 0; t < n; t++) {
        const int32_t e = seq[t];
        double beta = 0.0;
        for (int64_t j = indptr[e]; j < indptr[e + 1]; j++) beta += coef[j] * X[idx[j]];
        const double Ee = E[e];
        const double d = (-s * Ee + c * beta) - beta;
        E[e] = c * Ee + s * beta;
        for (int64_t j = indptr[e]; j < indptr[e + 1]; j++) X[idx[j]] += d * coef[j];
    }
}
'''
_LIB = {}


def _lib():
    if 'lib' not in _LIB:
        directory = Path(os.environ.get('OPH_EXACT_KERNEL_DIR', Path.home() / '.cache' / 'oph_exact_kernels'))
        directory.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(C_SOURCE.encode()).hexdigest()[:16]
        path = directory / f'oph_influence_{tag}.so'
        if not path.exists():
            cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
            src = directory / f'oph_influence_{tag}.c'
            src.write_text(C_SOURCE)
            tmp = directory / f'oph_influence_{tag}.{os.getpid()}.so'
            subprocess.run([cc, '-O2', '-shared', '-fPIC', '-o', str(tmp), str(src)], check=True, capture_output=True)
            os.replace(tmp, path)
        lib = ctypes.CDLL(str(path))
        vp, i64 = ctypes.c_void_p, ctypes.c_int64
        lib.oph_influence.argtypes = [vp, vp, vp, i64, i64, vp, vp, vp, vp]
        lib.oph_influence.restype = None
        lib.oph_rotate.argtypes = [vp, vp, vp, vp, i64, ctypes.c_double, ctypes.c_double, vp, vp]
        lib.oph_rotate.restype = None
        _LIB['lib'] = lib
    return _LIB['lib']


def equal_area_radius(geo: FG.Geometry, centre: int) -> np.ndarray:
    """Radius, in mesh edges, of the flat disc that holds as many carriers as lie closer to ``centre`` on the
    sphere: a distance in units of the mesh edge that does not depend on how the mesh is drawn on the sphere."""
    X = geo.carrier_xyz / np.linalg.norm(geo.carrier_xyz, axis=1, keepdims=True)
    ang = np.arccos(np.clip(X @ X[centre], -1, 1))
    rank = np.empty(geo.carriers); rank[np.argsort(ang, kind='stable')] = np.arange(geo.carriers)
    return np.sqrt((rank + 0.5) * (SQRT3 / 4) / np.pi)


# ---------------------------------------------------------------------------- L1
def cone(level: int, sweeps: int, seed: int, source_carrier: int | None = None) -> dict:
    geo = FG.build(level)
    a, b = geo.all_seam_ends()
    a = np.ascontiguousarray(a, np.int32); b = np.ascontiguousarray(b, np.int32)
    P, M, N = geo.ports, geo.seams, geo.carriers
    src = 4 ** level - 1 if source_carrier is None else source_carrier
    depth = np.zeros(P, np.int32)
    first_t = np.full(P, -1, np.int64); first_layer = np.full(P, -1, np.int32)
    first_t[12 * src] = 0; first_layer[12 * src] = 0
    count = np.zeros(1, np.int64)
    rng = np.random.default_rng(7000 + 100 * level + seed)
    lib = _lib()
    r = equal_area_radius(geo, src)
    front_sweep = np.zeros(sweeps + 1); depth_mean = np.zeros(sweeps + 1)
    for s in range(sweeps):
        for lo in range(0, M, 1 << 24):
            n = min(M - lo, 1 << 24)
            seq = rng.integers(0, M, size=n, dtype=np.int32)
            lib.oph_influence(a.ctypes.data, b.ctypes.data, seq.ctypes.data, n, s * M + lo, depth.ctypes.data,
                              first_t.ctypes.data, first_layer.ctypes.data, count.ctypes.data)
        got = (first_t.reshape(N, 12) >= 0).any(axis=1)
        front_sweep[s + 1] = np.sqrt(got.sum() * (SQRT3 / 4) / np.pi)
        depth_mean[s + 1] = depth.mean()
    reached = first_t >= 0
    # the influenced set grows as a disc: its equal-area radius over the second half of the run gives the speed
    # (a fit of arrival times over the carriers reached so far would keep only the early arrivals)
    half = sweeps // 2
    speed = (front_sweep[sweeps] - front_sweep[half]) / (sweeps - half)
    layers = (depth_mean[sweeps] - depth_mean[half]) / (sweeps - half)
    # the same in causal layers: carriers first reached by an attempt of native depth <= L
    l_c = np.where(reached, first_layer, np.iinfo(np.int32).max).reshape(N, 12).min(axis=1)
    lmax = int(depth_mean[half] + 0.6 * (depth_mean[sweeps] - depth_mean[half]))
    cnt = np.bincount(np.minimum(l_c[l_c < np.iinfo(np.int32).max], lmax + 1), minlength=lmax + 2).cumsum()
    rl = np.sqrt(cnt * (SQRT3 / 4) / np.pi)
    lo = lmax // 2
    speed_layer = (rl[lmax] - rl[lo]) / (lmax - lo)
    return {'level': level, 'seed': seed, 'sweeps': sweeps, 'source_carrier': int(src), 'ports_reached': int(reached.sum()),
            'front_radius_edges_by_sweep': front_sweep.tolist(), 'mean_native_depth_by_sweep': depth_mean.tolist(),
            'speed_edges_per_sweep': float(speed), 'speed_edges_per_layer': float(speed_layer),
            'layers_per_sweep': float(layers), 'front_radius_edges_by_layer': rl[:lmax + 1][::max(1, lmax // 40)].tolist()}


def flat_face_coordinates(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact lattice coordinates, in mesh edges, of the carriers of base face 0 (cells 0 .. 4^level - 1): the
    centroid of each cell in the flat equilateral triangle of side 2^level that the face is intrinsically.
    Returns (cells, xy) with the centre of the face at the origin."""
    from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
    tower = build_geodesic_icosahedral_tower(level)
    key = lambda x: tuple(np.round(x, 9))
    m0 = tower.levels[0]
    f0 = m0.faces[0]
    flat = {key(m0.vertices[int(f0[0])]): np.array([0.0, 0.0]), key(m0.vertices[int(f0[1])]): np.array([1.0, 0.0]),
            key(m0.vertices[int(f0[2])]): np.array([0.5, SQRT3 / 2])}
    for l in range(level):
        mesh = tower.levels[l]
        for c in range(4 ** l):
            P = [mesh.vertices[int(v)] for v in mesh.faces[c]]
            Xf = [flat[key(q)] for q in P]
            for i, j in ((0, 1), (1, 2), (2, 0)):
                mid = P[i] + P[j]; mid = mid / np.linalg.norm(mid)
                flat.setdefault(key(mid), (Xf[i] + Xf[j]) / 2)
    mesh = tower.levels[level]
    cells = np.arange(4 ** level)
    xy = np.array([np.mean([flat[key(mesh.vertices[int(v)])] for v in mesh.faces[c]], axis=0) for c in cells])
    xy = (xy - np.array([0.5, SQRT3 / 6])) * 2 ** level
    return cells, xy


def front_shape(level: int, realisations: int, gluing: str = 'port_pair', seed: int = 0) -> dict:
    """Direction dependence of the influence front in exact lattice coordinates. Every seam fires at unit rate, so
    first-passage times with independent exponential seam weights have the law of the first influence times."""
    from scipy.sparse.csgraph import dijkstra
    geo = FG.build(level, gluing=gluing)
    a, b = geo.all_seam_ends()
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    P, M, N = geo.ports, geo.seams, geo.carriers
    cells, xy = flat_face_coordinates(level)
    src = 4 ** level - 1
    xy = xy - xy[src]
    r = np.hypot(xy[:, 0], xy[:, 1]); phi = np.arctan2(xy[:, 1], xy[:, 0])
    inr = 2 ** level / (2 * SQRT3)
    band = (r > 0.45 * inr) & (r < 0.9 * inr)
    rng = np.random.default_rng(9100 + 10 * level + seed)
    acc = np.zeros(cells.size)
    for _ in range(realisations):
        w = rng.exponential(1.0, M)
        A = sp.coo_matrix((w, (a, b)), shape=(P, P)).tocsr()
        t = dijkstra(A, directed=False, indices=(12 * src + np.arange(12)), min_only=True)
        acc += t.reshape(N, 12).min(axis=1)[cells]
    T = acc / realisations
    slope, icpt = np.polyfit(r[band], T[band], 1)
    rel = T[band] / (slope * r[band] + icpt) - 1.0
    harm = {}
    for m in (1, 2, 3, 4, 6, 12):
        cs, sn = float((rel * np.cos(m * phi[band])).mean() * 2), float((rel * np.sin(m * phi[band])).mean() * 2)
        harm[m] = {'amplitude': float(np.hypot(cs, sn)), 'phase_degrees': float(np.degrees(np.arctan2(sn, cs)) / m)}
    # unweighted seam distance in the same coordinates
    hops = dijkstra(sp.coo_matrix((np.ones(M), (a, b)), shape=(P, P)).tocsr(), directed=False, indices=(12 * src + np.arange(12)),
                    unweighted=True, min_only=True).reshape(N, 12).min(axis=1)[cells]
    hs, hi = np.polyfit(r[band], hops[band], 1)
    hrel = hops[band] / (hs * r[band] + hi) - 1.0
    hh = {m: float(np.hypot((hrel * np.cos(m * phi[band])).mean(), (hrel * np.sin(m * phi[band])).mean()) * 2) for m in (2, 3, 6)}
    return {'level': level, 'gluing': gluing, 'realisations': realisations, 'carriers_in_band': int(band.sum()),
            'band_edges': [float(0.45 * inr), float(0.9 * inr)], 'front_speed_edges_per_sweep': float(1 / slope),
            'arrival_time_harmonics': harm, 'scatter_of_mean_arrival': float(rel.std()),
            'seams_per_edge': float(hs), 'seam_distance_harmonics': hh}


# ---------------------------------------------------------------------------- L2
def incidence(geo: FG.Geometry):
    """B (seams x ports) and K (faces x seams) of the federation complex."""
    N, P, M, V = geo.carriers, geo.ports, geo.seams, geo.vertices
    a, b = geo.all_seam_ends()
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    s = np.arange(M)
    B = sp.csr_matrix((np.concatenate([np.ones(M), -np.ones(M)]), (np.concatenate([s, s]), np.concatenate([a, b]))), shape=(M, P))
    rows, cols, vals = [], [], []
    for c in range(N):                                   # carrier faces
        for k in range(30):
            rows += [20 * c + int(FG.TEMPLATE_PLUS[k]), 20 * c + int(FG.TEMPLATE_MINUS[k])]
            cols += [30 * c + k, 30 * c + k]; vals += [1.0, -1.0]
    mesh = Mesh(geo)
    SI = geo.intra_seams
    ga, gb = geo.glue_a.astype(np.int64), geo.glue_b.astype(np.int64)
    vp, vm = geo.glue_fp.astype(np.int64) - 20 * N, geo.glue_fm.astype(np.int64) - 20 * N
    enter = {}                                           # (plaquette, carrier) -> port where the cycle enters / leaves
    leave = {}
    for g in range(ga.size):
        # plaquette vp[g] runs along the seam from a to b: it leaves the carrier of a and enters the carrier of b
        rows.append(20 * N + int(vp[g])); cols.append(SI + g); vals.append(1.0)
        leave[(int(vp[g]), int(ga[g] // 12))] = int(ga[g] % 12); enter[(int(vp[g]), int(gb[g] // 12))] = int(gb[g] % 12)
        rows.append(20 * N + int(vm[g])); cols.append(SI + g); vals.append(-1.0)
        leave[(int(vm[g]), int(gb[g] // 12))] = int(gb[g] % 12); enter[(int(vm[g]), int(ga[g] // 12))] = int(ga[g] % 12)
    path_cache = {}
    for (v, c), p_in in enter.items():
        p_out = leave[(v, c)]
        if (p_in, p_out) not in path_cache:
            path_cache[(p_in, p_out)] = mesh.template_path(p_in, p_out)
        for k, sign in path_cache[(p_in, p_out)]:
            rows.append(20 * N + v); cols.append(30 * c + k); vals.append(float(sign))
    K = sp.csr_matrix((vals, (rows, cols)), shape=(20 * N + V, M))
    return B, K


def check_complex(level: int) -> dict:
    geo = FG.build(level)
    B, K = incidence(geo)
    kb = abs(K @ B).max()
    out = {'level': level, 'ports': geo.ports, 'seams': geo.seams, 'faces': int(K.shape[0]), 'max_abs_KB': float(kb),
           'expected_rank': geo.seams - geo.ports + 1, 'plaquette_boundary_lengths': np.unique(np.asarray(abs(K[20 * geo.carriers:]).sum(axis=1)).ravel(), return_counts=True)[0].tolist()}
    if K.shape[0] <= 7000:
        sv = np.linalg.svd(K.toarray(), compute_uv=False)
        out['rank'] = int((sv > 1e-8 * sv[0]).sum())
    return out


# ---------------------------------------------------------------------------- L3
def spectra(level: int, gluing: str = 'port_pair', k: int = 25) -> dict:
    geo = FG.build(level, gluing=gluing)
    B, K = incidence(geo)
    N = geo.carriers
    F = K.shape[0]
    # the kernel of K^T holds one constant per carrier (its 20 faces close up) and one global vector; the carrier
    # constants are lifted out of the way, the global zero is dropped from the list
    C = sp.csr_matrix((np.full(20 * N, 20 ** -0.5), (np.arange(20 * N), np.repeat(np.arange(N), 20))), shape=(F, N))
    A = (K @ K.T + 50.0 * (C @ C.T)).tocsc()
    vals = np.sort(spl.eigsh(A, k=k + 1, sigma=-1e-5 / 4 ** level, which='LM', return_eigenvectors=False))
    L0 = (B.T @ B).tocsc()
    pv = np.sort(spl.eigsh(L0, k=k + 1, sigma=-1e-4 / 4 ** level, which='LM', return_eigenvectors=False))
    face = (4 ** level * vals[1:]).tolist(); port = (4 ** level * pv[1:]).tolist()
    return {'level': level, 'gluing': gluing, 'zero_modes_dropped': [float(vals[0]), float(pv[0])], 'face_scaled': face, 'port_scaled': port,
            'triplet_face': face[:3], 'triplet_port': port[:3], 'ratio_of_triplet_means': float(np.mean(face[:3]) / np.mean(port[:3])),
            'speed_ratio_light_over_matter': float(np.sqrt(np.mean(face[:3]) / np.mean(port[:3])))}


# ---------------------------------------------------------------------------- L4
def leapfrog(level: int, nu: float, steps: int, gluing: str = 'port_pair', pulse: float = 0.0) -> dict:
    """f <- f - nu K e, e <- e + nu K^T f. Start: one excited plaquette (pulse = 0) or a smooth ring-free bump of
    width ``pulse`` edges on the plaquettes, relaxed onto the slow modes by damping the carrier faces first."""
    geo = FG.build(level, gluing=gluing)
    _, K = incidence(geo)
    KT = K.T.tocsr()
    N, V = geo.carriers, geo.vertices
    v0 = int(geo.tri[4 ** level - 1].max())
    X = geo.mesh_xyz
    ang = np.arccos(np.clip(X @ X[v0], -1, 1))
    rank = np.empty(V); rank[np.argsort(ang, kind='stable')] = np.arange(V)
    rv = np.sqrt((rank + 0.5) * (SQRT3 / 2) / np.pi)                    # equal-area radius of every plaquette, edges
    f = np.zeros(K.shape[0])
    if pulse > 0:
        f[20 * N:] = np.exp(-rv ** 2 / (2 * pulse ** 2))
        A = (K @ KT).tocsr()
        keep = f[20 * N:].copy()
        for _ in range(400):                                            # carrier faces follow the plaquettes: minimise |K^T f|^2 over them
            g = A @ f
            f[:20 * N] -= g[:20 * N] / 6.0
            f[20 * N:] = keep
    else:
        f[20 * N + v0] = 1.0
    e = np.zeros(geo.seams)
    order = np.argsort(rv)
    support, half, mean_r, energy = [], [], [], []
    for n in range(steps):
        e += nu * (KT @ f)
        f -= nu * (K @ e)
        w = f[20 * N:] ** 2
        nz = np.flatnonzero(np.abs(f[20 * N:]) > 1e-14 * np.abs(f).max())
        support.append(float(rv[nz].max()) if nz.size else 0.0)
        tot = w.sum()
        cum = np.cumsum(w[order])
        half.append(float(rv[order][np.searchsorted(cum, 0.5 * tot)]))
        mean_r.append(float((w * rv).sum() / tot))
        energy.append(float((f ** 2).sum() + (e ** 2).sum()))
    st = np.arange(1, steps + 1)
    lim = 0.7 * rv.max()
    sup, mr = np.array(support), np.array(mean_r)
    ok = (sup < lim) & (st > 4)
    okm = (mr < 0.5 * lim) & (st > steps // 3)
    return {'level': level, 'gluing': gluing, 'nu': nu, 'steps': steps, 'pulse_width_edges': pulse, 'support_radius_edges': support[::max(1, steps // 80)],
            'half_energy_radius_edges': half[::max(1, steps // 80)], 'mean_radius_edges': mean_r[::max(1, steps // 80)],
            'energy_first_last': [energy[0], energy[-1]],
            'cone_speed_edges_per_step': float(np.polyfit(st[ok], sup[ok], 1)[0]) if ok.sum() > 3 else None,
            'mean_radius_speed_edges_per_step': float(np.polyfit(st[okm], mr[okm], 1)[0]) if okm.sum() > 3 else None}


# ---------------------------------------------------------------------------- event-driven transport under three schedules
_ROT = {}


def rotation_setup(level: int, sector: str, gluing: str = 'port_pair', modes: int = 90) -> dict:
    """Per-seam unit vectors u_e (columns of K for the transverse sector, of B^T for the scalar sector), the
    operator U U^T that the regular clock gives (frequency phi sqrt(eigenvalue) per sweep) and its low modes."""
    key = (level, sector, gluing)
    if key in _ROT:
        return _ROT[key]
    geo = FG.build(level, gluing=gluing)
    B, K = incidence(geo)
    U = (K if sector == 'transverse' else B.T).tocsc().astype(np.float64)
    norms = np.sqrt(np.asarray(U.multiply(U).sum(axis=0)).ravel())
    U = (U @ sp.diags(1.0 / norms)).tocsc()
    U.sort_indices()
    A = (U @ U.T).tocsc()
    N = geo.carriers
    if sector == 'transverse':
        C = sp.csr_matrix((np.full(20 * N, 20 ** -0.5), (np.arange(20 * N), np.repeat(np.arange(N), 20))), shape=(A.shape[0], N))
        A = (A + 50.0 * (C @ C.T)).tocsc()
    vals, vecs = spl.eigsh(A, k=modes + 1, sigma=-1e-5 / 4 ** level, which='LM')
    order = np.argsort(vals)
    radius = float(np.sqrt(geo.vertices * SQRT3 / 2 / (4 * np.pi)))
    _ROT[key] = {'geo': geo, 'indptr': U.indptr.astype(np.int64), 'idx': U.indices.astype(np.int32), 'coef': U.data.astype(np.float64),
                 'nodes': U.shape[0], 'seams': U.shape[1], 'vals': vals[order][1:], 'vecs': vecs[:, order][:, 1:], 'radius': radius,
                 'faces_per_seam': np.unique(np.diff(U.indptr), return_counts=True)}
    return _ROT[key]


def rotation_run(spec: dict) -> dict:
    """Amplitude <v, X(t)> of one eigenmode v under event-driven rotations by the angle phi."""
    st = rotation_setup(spec['level'], spec['sector'], spec.get('gluing', 'port_pair'))
    M, phi, sched = st['seams'], spec['phi'], spec['schedule']
    j = spec['mode']
    v = np.ascontiguousarray(st['vecs'][:, j]); v = v / np.linalg.norm(v)
    lam = float(st['vals'][j])
    omega0 = phi * np.sqrt(lam)
    sweeps = spec['sweeps']
    rng = np.random.default_rng(spec['seed'])
    fixed = np.random.default_rng(424242).permutation(M).astype(np.int32)
    # a structured fixed order: the thirty internal seam classes in turn (the same template seam in every carrier),
    # then the glue seams
    SI = st['geo'].intra_seams
    classes = np.concatenate([np.argsort(np.arange(SI) % 30, kind='stable'), np.arange(SI, M)]).astype(np.int32)
    X = v.copy(); E = np.zeros(M)
    lib = _lib()
    c, sn = float(np.cos(phi)), float(np.sin(phi))
    amp = np.empty(sweeps + 1); amp[0] = 1.0
    for t in range(sweeps):
        if sched == 'replayed':
            seq = fixed
        elif sched == 'classes':
            seq = classes
        elif sched == 'permuted':
            seq = rng.permutation(M).astype(np.int32)
        else:
            seq = rng.integers(0, M, size=M, dtype=np.int32)
        lib.oph_rotate(st['indptr'].ctypes.data, st['idx'].ctypes.data, st['coef'].ctypes.data, seq.ctypes.data, M, c, sn, E.ctypes.data, X.ctypes.data)
        amp[t + 1] = float(v @ X)
    norm = float(X @ X + E @ E)
    return {**{k: spec[k] for k in ('level', 'sector', 'schedule', 'phi', 'mode', 'seed')}, 'eigenvalue': lam, 'omega_regular': omega0,
            'norm_end': norm, 'amplitude': amp}


def fit_damped(t: np.ndarray, a: np.ndarray, omega0: float) -> tuple[float, float]:
    """Least-squares exp(-gamma t) cos(omega t)."""
    from scipy.optimize import least_squares
    best = None
    for g0 in (0.0, 0.1 * omega0, omega0, 5 * omega0):
        r = least_squares(lambda q: np.exp(-abs(q[0]) * t) * np.cos(q[1] * t) - a, x0=[g0, omega0])
        if best is None or r.cost < best.cost:
            best = r
    return float(abs(best.x[0])), float(abs(best.x[1]))


def transverse(level: int, sectors, phis, ells, realisations: int, workers: int, periods: float = 4.0, gluing: str = 'port_pair') -> list:
    from multiprocessing import Pool
    specs, meta = [], {}
    for sector in sectors:
        st = rotation_setup(level, sector, gluing)
        for ell in ells:
            j = ell * ell - 1                                   # first member of the multiplet l (the constant mode is dropped)
            lam = float(st['vals'][j])
            for phi in phis:
                omega0 = phi * np.sqrt(lam)
                sweeps = int(min(6000, max(200, periods * 2 * np.pi / omega0)))
                for sched in ('classes', 'replayed', 'permuted', 'poisson'):
                    for seed in range(1 if sched in ('replayed', 'classes') else realisations):
                        specs.append(dict(level=level, sector=sector, schedule=sched, phi=phi, mode=j, seed=1000 + seed, sweeps=sweeps, ell=ell, gluing=gluing))
    with Pool(workers) as pool:
        got = pool.map(rotation_run, specs, chunksize=1)
    rows = []
    keys = sorted({(g['sector'], g['mode'], g['phi'], g['schedule']) for g in got}, key=lambda q: (q[0], q[1], q[2], q[3]))
    for sector, j, phi, sched in keys:
        st = rotation_setup(level, sector, gluing)
        sub = [g for g in got if (g['sector'], g['mode'], g['phi'], g['schedule']) == (sector, j, phi, sched)]
        a = np.mean([g['amplitude'] for g in sub], axis=0)
        t = np.arange(a.size, dtype=float)
        omega0 = sub[0]['omega_regular']
        gamma, omega = fit_damped(t, a, omega0)
        ell = int(round(np.sqrt(j + 1)))
        k = np.sqrt(ell * (ell + 1)) / st['radius']
        rows.append({'level': level, 'gluing': gluing, 'sector': sector, 'ell': ell, 'k_per_edge': float(k), 'phi': phi, 'schedule': sched, 'realisations': len(sub),
                     'eigenvalue': sub[0]['eigenvalue'], 'omega_regular_per_sweep': omega0, 'omega_fit_per_sweep': omega, 'gamma_fit_per_sweep': gamma,
                     'speed_edges_per_sweep_per_phi': float(omega / k / phi), 'quality': float(omega / (2 * gamma)) if gamma > 0 else None,
                     'range_edges': float(omega / k / gamma) if gamma > 0 else None, 'norm_end': float(np.mean([g['norm_end'] for g in sub])),
                     'amplitude_sampled': a[::max(1, a.size // 60)].tolist()})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['cone', 'front', 'complex', 'spectrum', 'leapfrog', 'transverse'])
    p.add_argument('--levels', default='3,5')
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--gluing', default='port_pair')
    p.add_argument('--nu', type=float, default=0.25)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--pulse', type=float, default=0.0, help='leapfrog: width in edges of a smooth initial bump (0: one plaquette)')
    p.add_argument('--json', type=Path, default=None)
    p.add_argument('--phis', default='0.1,0.2,0.4')
    p.add_argument('--ells', default='2,4,8')
    p.add_argument('--sectors', default='transverse,scalar')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(argv)
    levels = [int(v) for v in a.levels.split(',')]
    out = []
    for level in levels:
        t0 = time.time()
        if a.what == 'transverse':
            rows = transverse(level, a.sectors.split(','), [float(v) for v in a.phis.split(',')], [int(v) for v in a.ells.split(',')], a.seeds, a.workers, gluing=a.gluing)
            out += rows
            print(f"level {level}: event-driven rotations; speed in edges per sweep per unit phi, damping per sweep  ({time.time() - t0:.0f}s)")
            print(f"{'sector':>11}{'l':>4}{'k':>7}{'phi':>6}{'schedule':>10}{'omega/reg':>10}{'speed':>8}{'gamma':>11}{'gamma/phi^2':>12}{'gamma/phi^4':>12}{'Q':>8}{'range':>9}")
            for r in rows:
                g = r['gamma_fit_per_sweep']
                print(f"{r['sector']:>11}{r['ell']:>4}{r['k_per_edge']:7.3f}{r['phi']:6.2f}{r['schedule']:>10}{r['omega_fit_per_sweep'] / r['omega_regular_per_sweep']:10.4f}"
                      f"{r['speed_edges_per_sweep_per_phi']:8.4f}{g:11.3e}{g / r['phi'] ** 2:12.4f}{g / r['phi'] ** 4:12.4f}"
                      f"{(r['quality'] if r['quality'] is not None else float('inf')):8.1f}{(r['range_edges'] if r['range_edges'] is not None else float('inf')):9.1f}", flush=True)
            continue
        if a.what == 'cone':
            runs = [cone(level, a.sweeps, seed) for seed in range(a.seeds)]
            sp_s = np.array([r['speed_edges_per_sweep'] for r in runs]); sp_l = np.array([r['speed_edges_per_layer'] for r in runs])
            print(f"level {level}: front {sp_s.mean():.4f} +- {sp_s.std() / np.sqrt(len(runs)):.4f} edges per sweep, {sp_l.mean():.5f} +- {sp_l.std() / np.sqrt(len(runs)):.5f} edges per causal layer, "
                  f"{np.mean([r['layers_per_sweep'] for r in runs]):.2f} native layers per sweep  ({time.time() - t0:.0f}s)", flush=True)
            out += runs
        elif a.what == 'front':
            r = front_shape(level, a.seeds, a.gluing); out.append(r)
            h = r['arrival_time_harmonics']
            print(f"level {level} {a.gluing}: front {r['front_speed_edges_per_sweep']:.4f} edges per sweep in lattice coordinates ({r['realisations']} realisations, {r['carriers_in_band']} carriers); "
                  'relative harmonics of the arrival time ' + ', '.join(f"m={m}: {h[m]['amplitude']:.4f}" for m in h) + f"; scatter {r['scatter_of_mean_arrival']:.4f}; "
                  f"{r['seams_per_edge']:.3f} seams per edge, seam-distance harmonics {r['seam_distance_harmonics']}  ({time.time() - t0:.0f}s)", flush=True)
        elif a.what == 'complex':
            r = check_complex(level); out.append(r); print(r, flush=True)
        elif a.what == 'spectrum':
            r = spectra(level, a.gluing); out.append(r)
            print(f"level {level} {a.gluing}: face triplet {['%.5f' % v for v in r['triplet_face']]} port triplet {['%.5f' % v for v in r['triplet_port']]} "
                  f"ratio {r['ratio_of_triplet_means']:.4f} speed ratio {r['speed_ratio_light_over_matter']:.4f}; face 4..12: {['%.4f' % v for v in r['face_scaled'][3:12]]}; "
                  f"port 4..12: {['%.4f' % v for v in r['port_scaled'][3:12]]}  ({time.time() - t0:.0f}s)", flush=True)
        else:
            r = leapfrog(level, a.nu, a.steps, a.gluing, a.pulse); out.append(r)
            print(f"level {level} nu {a.nu} pulse {a.pulse}: strict cone {r['cone_speed_edges_per_step']} edges per step, energy-weighted mean radius {r['mean_radius_speed_edges_per_step']} edges per step "
                  f"({'n/a' if r['mean_radius_speed_edges_per_step'] is None else round(r['mean_radius_speed_edges_per_step'] / a.nu, 4)} per unit nu), energy {r['energy_first_last']}, "
                  f"mean radius {['%.1f' % v for v in r['mean_radius_edges'][::10]]}  ({time.time() - t0:.0f}s)", flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
