"""Exact band structure of the port operator on the infinite flat lattice for symmetric wirings.

    python -m oph_defects.bloch --json runs/light/bloch.json

Two symmetric wirings of the twelve-port carrier, both with the three-fold axis of template face (0, 11, 5) along
the screen normal:

  honeycomb3    carriers on the cells of the triangular mesh (the federation), each glued to its three edge
                neighbours through the three belt ports next to that face. This is the flat-lattice form of the
                control gluing ``face_axis``. Unit cell: one up and one down carrier, 24 ports.
  triangular6   carriers on the mesh vertices, each glued to its six neighbours through its six belt ports, a
                port to the antipodal port of the neighbour it points to. Unit cell: one carrier, 12 ports.

L(q) is the Bloch form of the port Laplacian B^T B, q in radians per mesh edge. Under a continuous flow the
scalar sector has frequencies nu sqrt(lambda). Reported: the speed of the massless band, every level at q = 0 with
its curvature (a massive band is lambda_0 + c_b^2 q^2, and it shares the cone of the massless band only if
c_b = c), the flat bands, and for the honeycomb wiring the crossing points of the two lowest bands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import fedgeom as FG

A1 = np.array([1.0, 0.0]); A2 = np.array([0.5, 3 ** 0.5 / 2])
TOP = (0, 11, 5)


def template():
    T = FG.TEMPLATE_SEAMS
    A = np.zeros((12, 12))
    for a, b in T:
        A[a, b] = A[b, a] = 1
    return A


def belt_ports():
    """The six belt ports in the frame of the top face, ordered by azimuth, with their azimuths and antipodes."""
    U = FG.PORT_UNIT
    n = U[list(TOP)].sum(axis=0); n /= np.linalg.norm(n)
    ref = U[TOP[0]] - (U[TOP[0]] @ n) * n; ref /= np.linalg.norm(ref)
    e2 = np.cross(n, ref)
    polar = np.degrees(np.arccos(U @ n))
    belt = [p for p in range(12) if 60 < polar[p] < 120]
    az = np.degrees(np.arctan2(U[belt] @ e2, U[belt] @ ref)) % 360
    order = np.argsort(az)
    belt = [belt[i] for i in order]; az = az[order]
    anti = {p: int(np.argmin(U @ U[p])) for p in range(12)}
    return belt, az, anti, polar


def bloch_honeycomb(q: np.ndarray, ports=None) -> np.ndarray:
    P = list(FG._control_ports()) if ports is None else list(ports)
    A = template()
    L = np.zeros((24, 24), complex)
    for blk in (0, 12):
        L[blk:blk + 12, blk:blk + 12] = np.diag(A.sum(axis=1)) - A
    for k, delta in enumerate((np.zeros(2), A1, A2)):
        u, d = P[k], 12 + P[k]
        L[u, u] += 1; L[d, d] += 1
        ph = np.exp(-1j * (q @ delta))
        L[u, d] -= ph; L[d, u] -= np.conj(ph)
    return L


def bloch_triangular(q: np.ndarray) -> np.ndarray:
    belt, az, anti, _ = belt_ports()
    A = template()
    L = (np.diag(A.sum(axis=1)) - A).astype(complex)
    for j, p in enumerate(belt):                       # port p points to the neighbour in direction az[j]
        e = np.array([np.cos(np.radians(az[j] - az[0])), np.sin(np.radians(az[j] - az[0]))])
        pb = anti[p]
        L[p, p] += 0.5; L[pb, pb] += 0.5               # every bond is listed from both of its ends
        L[p, pb] -= 0.5 * np.exp(1j * (q @ e)); L[pb, p] -= 0.5 * np.exp(-1j * (q @ e))
    return L


def analyse(name: str) -> dict:
    H = bloch_honeycomb if name == 'honeycomb3' else bloch_triangular
    n = 24 if name == 'honeycomb3' else 12
    ev0 = np.linalg.eigvalsh(H(np.zeros(2)))
    h = 1e-3
    dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.radians([0, 30, 60, 90])]
    curv = np.array([[(np.linalg.eigvalsh(H(h * d)) - ev0) / h ** 2 for d in dirs]])[0]     # per direction, per band
    # flat bands: eigenvalues that do not move anywhere on a coarse grid of the zone
    grid = [np.array([x, y]) for x in np.linspace(-np.pi, np.pi, 13) for y in np.linspace(-np.pi, np.pi, 13)]
    allv = np.array([np.linalg.eigvalsh(H(g)) for g in grid])
    width = allv.max(axis=0) - allv.min(axis=0)
    cells = 2 if name == 'honeycomb3' else 1
    density = 12 * cells / (3 ** 0.5 / 2)              # ports per unit area (mesh edge squared)
    c2 = float(curv[:, 0].mean())                      # massless band: lambda = c^2 q^2
    rows = []
    for b in range(n):
        rows.append({'band': b, 'level_at_zero': float(ev0[b]), 'band_minimum': float(allv[:, b].min()), 'band_maximum': float(allv[:, b].max()),
                     'flat': bool(width[b] < 1e-9), 'curvature_by_direction': curv[:, b].tolist(),
                     'curvature_over_massless': float(curv[:, b].mean() / c2) if c2 else None,
                     'anisotropy_of_curvature': float((curv[:, b].max() - curv[:, b].min()) / max(1e-12, abs(curv[:, b].mean())))})
    out = {'wiring': name, 'ports_per_unit_cell': n, 'ports_per_edge_squared': density, 'massless_c_squared': c2,
           'conductivity': c2 * density, 'bands': rows}
    if name == 'honeycomb3':
        K = np.array([2 * np.pi / 3, 2 * np.pi / 3 ** 0.5])
        evK = np.linalg.eigvalsh(H(K))
        d = 1e-3
        split = [float(np.linalg.eigvalsh(H(K + d * e))[1] - np.linalg.eigvalsh(H(K + d * e))[0]) / (2 * d) for e in dirs]
        out['crossing_point'] = {'q': K.tolist(), 'two_lowest_levels': evK[:2].tolist(), 'gap': float(evK[1] - evK[0]),
                                 'linear_slope_of_lambda_by_direction': split, 'top_of_lower_band': float(allv[:, 0].max()),
                                 'bottom_of_upper_band': float(allv[:, 1].min())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    res = []
    for name in ('honeycomb3', 'triangular6'):
        r = analyse(name); res.append(r)
        print(f"\n{name}: massless band lambda = {r['massless_c_squared']:.6f} q^2 (conductivity {r['conductivity']:.4f}); the flow wave speed is nu times {np.sqrt(r['massless_c_squared']):.4f} edges")
        print(f"{'band':>5}{'level at q=0':>14}{'range':>22}{'curvature / massless':>22}{'anisotropy':>12}")
        for b in r['bands']:
            rng = f"[{b['band_minimum']:.4f}, {b['band_maximum']:.4f}]"
            tag = 'flat' if b['flat'] else f"{b['curvature_over_massless']:+.4f}"
            print(f"{b['band']:>5}{b['level_at_zero']:14.6f}{rng:>22}{tag:>22}{b['anisotropy_of_curvature']:12.1e}")
        if 'crossing_point' in r:
            cp = r['crossing_point']
            print(f"  lowest two bands at K: {cp['two_lowest_levels'][0]:.6f}, {cp['two_lowest_levels'][1]:.6f} (gap {cp['gap']:.2e}); linear slope of lambda {['%.5f' % v for v in cp['linear_slope_of_lambda_by_direction']]} per unit q")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res))


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------- the declared tangent-frame routing on the flat lattice
def pinned_ports(gamma_degrees: float):
    """Ports chosen by the declared router for an up and a down carrier when the mesh is turned by gamma against
    the tangent frame: each neighbour direction goes to the port whose template direction, read in the frame
    (tangent_x, tangent_y, normal), has the largest inner product with it."""
    U = FG.PORT_UNIT
    out = []
    for base in ((30, 150, 270), (210, 330, 90)):
        ports = []
        for ang in base:
            t = np.radians(ang + gamma_degrees)
            score = U[:, 0] * np.cos(t) + U[:, 1] * np.sin(t)
            score[ports] = -9                                   # a port is glued once
            ports.append(int(np.argmax(score)))
        out.append(ports)
    return out


def bloch_pinned(q: np.ndarray, ports) -> np.ndarray:
    pu, pd = ports
    A = template()
    L = np.zeros((24, 24), complex)
    for blk in (0, 12):
        L[blk:blk + 12, blk:blk + 12] = np.diag(A.sum(axis=1)) - A
    for k, delta in enumerate((np.zeros(2), A1, A2)):
        u, d = pu[k], 12 + pd[k]
        L[u, u] += 1; L[d, d] += 1
        ph = np.exp(-1j * (q @ delta))
        L[u, d] -= ph; L[d, u] -= np.conj(ph)
    return L


def pinned_scan(angles=(0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90)) -> list:
    from scipy.optimize import minimize
    rows = []
    for g in angles:
        ports = pinned_ports(g)
        H = lambda q: bloch_pinned(np.asarray(q, float), ports)
        gap = lambda q: float(np.diff(np.linalg.eigvalsh(H(q))[:2])[0])
        best = (9.0, None)
        for x in np.linspace(-np.pi, np.pi, 41):
            for y in np.linspace(-np.pi, np.pi, 41):
                v = gap((x * 4 / 3, y * 4 / 3))
                if v < best[0]:
                    best = (v, (x * 4 / 3, y * 4 / 3))
        r = minimize(gap, best[1], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 4000})
        q0 = r.x
        ev = np.linalg.eigvalsh(H(q0))
        h = 1e-3
        c = [(np.linalg.eigvalsh(H(h * np.array([np.cos(t), np.sin(t)])))[0]) / h ** 2 for t in np.radians([0, 30, 60, 90, 120, 150])]
        dist = [int(np.round(np.degrees(np.arccos(np.clip(FG.PORT_UNIT[a] @ FG.PORT_UNIT[b], -1, 1))))) for a, b in ((ports[0][0], ports[0][1]), (ports[0][1], ports[0][2]), (ports[0][0], ports[0][2]))]
        rows.append({'frame_angle_degrees': g, 'ports_up': ports[0], 'ports_down': ports[1], 'minimum_gap_of_lowest_pair': float(r.fun), 'at_q': q0.tolist(),
                     'level_there': float(ev[0]), 'massless_curvature_min_max': [float(min(c)), float(max(c))],
                     'angles_between_glued_ports_up': dist})
    return rows


# ---------------------------------------------------------------------------- every wiring of three glue ports per carrier
def port_rotations():
    """The 60 rotations of the carrier as permutations of its twelve ports."""
    U = FG.PORT_UNIT
    A = template()
    def frame(a, b):
        e1 = U[a]; e2 = U[b] - (U[b] @ e1) * e1; e2 /= np.linalg.norm(e2)
        return np.stack([e1, e2, np.cross(e1, e2)], axis=1)
    F0 = frame(0, int(np.flatnonzero(A[0])[0]))
    perms = []
    for a in range(12):
        for b in np.flatnonzero(A[a]):
            Rm = frame(a, int(b)) @ F0.T
            img = (U @ Rm.T) @ U.T                                  # image of port p is the port closest to R u_p
            perms.append(tuple(int(v) for v in np.argmax(img, axis=1)))
    perms = sorted(set(perms))
    assert len(perms) == 60
    return perms


def triple_classes():
    """Ordered triples of distinct ports up to rotation of the carrier: representative, mutual seam distances."""
    perms = port_rotations()
    A = template()
    dist = np.where(A > 0, 1, 2).astype(int); np.fill_diagonal(dist, 0)
    for p in range(12):
        dist[p, int(np.argmin(FG.PORT_UNIT @ FG.PORT_UNIT[p]))] = 3
    seen, classes = set(), []
    for t in ((a, b, c) for a in range(12) for b in range(12) for c in range(12) if len({a, b, c}) == 3):
        if t in seen:
            continue
        orbit = {tuple(g[v] for v in t) for g in perms}
        seen |= orbit
        classes.append({'representative': min(orbit), 'distances': (int(dist[t[0], t[1]]), int(dist[t[1], t[2]]), int(dist[t[0], t[2]])), 'size': len(orbit)})
    return classes


def wiring_atlas() -> list:
    """For every pair of triple classes (up carrier, down carrier) and every cyclic alignment: is the massless band
    isotropic, do the two lowest bands touch, and if not what is the gap and the mass-shell slope."""
    from scipy.optimize import minimize
    cls = triple_classes()
    grid = [np.array([x, y]) * 4 / 3 for x in np.linspace(-np.pi, np.pi, 31) for y in np.linspace(-np.pi, np.pi, 31)]
    dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.radians([0, 30, 60, 90, 120, 150])]
    rows, seen = [], set()
    for i, cu in enumerate(cls):
        for j, cd in enumerate(cls):
            for shift in range(3):
                tu = list(cu['representative']); td = list(cd['representative']); td = td[shift:] + td[:shift]
                H = lambda q: bloch_pinned(np.asarray(q, float), (tu, td))
                h = 1e-3
                curv = [float(np.linalg.eigvalsh(H(h * d))[0]) / h ** 2 for d in dirs]
                gap = lambda q: float(np.diff(np.linalg.eigvalsh(H(q))[:2])[0])
                q0 = min(grid, key=gap)
                r = minimize(gap, q0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 3000})
                q0 = r.x; ev = np.linalg.eigvalsh(H(q0)); D = (ev[1] - ev[0]) / 2
                halves = []
                for d in dirs:
                    e = np.linalg.eigvalsh(H(q0 + 0.02 * d)); halves.append(np.sqrt(max(((e[1] - e[0]) / 2) ** 2 - D ** 2, 0)) / 0.02)
                key = (tuple(sorted(cu['distances'])), tuple(sorted(cd['distances'])), round(float(np.mean(curv)), 6), round(D, 6), round(float(np.mean(halves)), 5))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({'up': tu, 'down': td, 'up_distances': cu['distances'], 'down_distances': cd['distances'],
                             'massless_curvature': float(np.mean(curv)), 'massless_anisotropy': float((max(curv) - min(curv)) / np.mean(curv)),
                             'half_gap': float(D), 'crossing_level': float((ev[0] + ev[1]) / 2), 'cone_slope': float(np.mean(halves)),
                             'cone_anisotropy': float((max(halves) - min(halves)) / max(1e-12, np.mean(halves)))})
    return rows


# ---------------------------------------------------------------------------- G1: one quantum of the hard-core flow G(Delta)
def g1_bands(delta: float) -> dict:
    """One quantum under G(Delta) = sum_seams [-(1/2)(b_a^+ b_b + h.c.) + (Delta/2)(n_a (1 - n_b) + n_b (1 - n_a))] on the
    symmetric wiring: H(q) = (1/2) [L(q) - (1 - Delta) Deg]. Delta = 1 is the generator of the native transfer.
    Energies per unit flow time, q per mesh edge. For every band: level at q = 0, curvature, band mass 1 / (2 curvature);
    for the two lowest bands the crossing at K with its speed."""
    belt = list(FG._control_ports())
    deg = np.array([5.0] * 24)
    for k in range(3):
        deg[belt[k]] = 6; deg[12 + belt[k]] = 6
    H = lambda q: 0.5 * (bloch_pinned(np.asarray(q, float), (belt, belt)) - (1 - delta) * np.diag(deg))
    ev0 = np.linalg.eigvalsh(H(np.zeros(2)))
    h = 1e-3
    dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.radians([0, 30, 60, 90])]
    curv = np.array([(np.linalg.eigvalsh(H(h * d)) - ev0) / h ** 2 for d in dirs])
    grid = [np.array([x, y]) for x in np.linspace(-np.pi, np.pi, 13) for y in np.linspace(-np.pi, np.pi, 13)]
    allv = np.array([np.linalg.eigvalsh(H(g)) for g in grid])
    width = allv.max(axis=0) - allv.min(axis=0)
    K = np.array([2 * np.pi / 3, 2 * np.pi / 3 ** 0.5])
    evK = np.linalg.eigvalsh(H(K))
    d = 1e-3
    slope = float(np.mean([(np.linalg.eigvalsh(H(K + d * e))[1] - np.linalg.eigvalsh(H(K + d * e))[0]) / (2 * d) for e in dirs]))
    bands = []
    for b in range(24):
        c = float(curv[:, b].mean())
        bands.append({'band': b, 'level_at_zero': float(ev0[b]), 'above_ground': float(ev0[b] - ev0[0]), 'flat': bool(width[b] < 1e-9),
                      'curvature': c, 'band_mass': (1 / (2 * c)) if abs(c) > 1e-9 else None,
                      'anisotropy': float((curv[:, b].max() - curv[:, b].min()) / max(1e-12, abs(c)))})
    v = slope
    for bnd in bands:
        if bnd['band_mass'] and not bnd['flat'] and bnd['above_ground'] > 1e-9:
            bnd['gap_over_band_mass_v2'] = bnd['above_ground'] / (abs(bnd['band_mass']) * v * v)
    return {'delta': delta, 'ground_level': float(ev0[0]), 'ground_band_mass': bands[0]['band_mass'], 'crossing_level_above_ground': float(evK[0] - ev0[0]),
            'crossing_gap': float(evK[1] - evK[0]), 'cone_speed': v, 'ground_mass_times_cone_speed_squared': bands[0]['band_mass'] * v * v, 'bands': bands}


# ---------------------------------------------------------------------------- the seam-face (transverse) sector on the flat lattice
def incidence_bloch(q: np.ndarray):
    """B(q) (63 seams x 24 ports) and K(q) (41 faces x 63 seams) of the symmetric wiring. Seams: 30 internal of the up
    carrier, 30 of the down carrier, 3 glue seams (up port to down port). Faces: 20 + 20 carrier faces and one
    plaquette around a mesh vertex, whose boundary runs through the six carriers around the vertex; inside a carrier
    it goes from one glued belt port to the next through the top-face port between them (the three-fold symmetric
    choice; it is part of the complex)."""
    belt = list(FG._control_ports()); top = list(TOP)
    T = FG.TEMPLATE_SEAMS
    key = {(int(a), int(b)): (k, 1) for k, (a, b) in enumerate(T)}
    key.update({(int(b), int(a)): (k, -1) for k, (a, b) in enumerate(T)})
    B = np.zeros((63, 24), complex); Km = np.zeros((41, 63), complex)
    for blk, off in ((0, 0), (1, 30)):
        for k, (a, b) in enumerate(T):
            B[off + k, 12 * blk + a] += 1; B[off + k, 12 * blk + b] -= 1
        for k in range(30):
            Km[20 * blk + int(FG.TEMPLATE_PLUS[k]), off + k] += 1; Km[20 * blk + int(FG.TEMPLATE_MINUS[k]), off + k] -= 1
    delta = (np.zeros(2), A1, A2)
    for k in range(3):                                   # glue seam k: up port of cell 0 (first end) to down port of cell -delta_k
        B[60 + k, belt[k]] += 1; B[60 + k, 12 + belt[k]] -= np.exp(-1j * (q @ delta[k]))
    # the plaquette around the vertex at the origin, counter-clockwise: (carrier type, cell R, bond in, bond out)
    def path(i, j):                                      # belt port i to belt port j through the top port between them
        t = top[3 - i - j]
        return [key[(belt[i], t)], key[(t, belt[j])]]
    ring = [('U', np.zeros(2), 2, 1), ('D', -A1, 1, 0), ('U', -A1, 0, 2), ('D', -A1 - A2, 2, 1), ('U', -A2, 1, 0), ('D', -A2, 0, 2)]
    for kind, R, kin, kout in ring:
        ph = np.exp(1j * (q @ R)); off = 0 if kind == 'U' else 30
        for s, sign in path(kin, kout):
            Km[40, off + s] += sign * ph
        # the glue seam taken when leaving this carrier: bond kout. Its home is the up carrier of that bond.
        if kind == 'U':                                  # up -> down: along the seam orientation, home cell R
            Km[40, 60 + kout] += ph
        else:                                            # down(R) -> up(R + delta_kout): against the orientation, home cell R + delta
            Km[40, 60 + kout] -= np.exp(1j * (q @ (R + delta[kout])))
    return B, Km


def transverse_bands() -> dict:
    q0 = np.zeros(2)
    B0, K0 = incidence_bloch(np.array([0.37, -0.21]))
    check = float(np.abs(K0 @ B0).max())
    M = lambda q: (lambda K: K @ K.conj().T)(incidence_bloch(np.asarray(q, float))[1])
    ev0 = np.linalg.eigvalsh(M(q0))
    h = 1e-3
    dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.radians([0, 30, 60, 90])]
    curv = np.array([(np.linalg.eigvalsh(M(h * d)) - ev0) / h ** 2 for d in dirs])
    grid = [np.array([x, y]) for x in np.linspace(-np.pi, np.pi, 13) for y in np.linspace(-np.pi, np.pi, 13)]
    allv = np.array([np.linalg.eigvalsh(M(g)) for g in grid])
    width = allv.max(axis=0) - allv.min(axis=0)
    bands = [{'band': b, 'level_at_zero': float(ev0[b]), 'minimum': float(allv[:, b].min()), 'maximum': float(allv[:, b].max()), 'flat': bool(width[b] < 1e-9),
              'curvature': float(curv[:, b].mean()), 'anisotropy': float((curv[:, b].max() - curv[:, b].min()) / max(1e-12, abs(curv[:, b].mean())))} for b in range(41)]
    return {'KB_max': check, 'zero_modes_at_generic_q': int((np.linalg.eigvalsh(M(np.array([0.37, -0.21]))) < 1e-10).sum()), 'bands': bands}
